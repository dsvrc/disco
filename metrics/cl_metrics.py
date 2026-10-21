"""
Continual-learning metrics for the DiscoForcing TMR benchmark.

Why not the stock M2DMetrics for forgetting: it computes one feature vector per test sequence (~40 per
AIST++ session, 3-8 per FineDance session) and re-normalises by each call's GT std + 1e-10. Measured on
ground truth alone (scoring session i's GT against another session's GT = complete forgetting, versus the
split-half noise of session i's own GT), FID_k / FID_g give signal-to-noise around 0 on these genre shifts,
and FID_g overflows (up to 1e16) whenever a geometric feature is constant in the reference. A linear genre
probe on the same handcrafted features separates GT but confuses quality with style (smoothing sigma = 2
frames: 0.86 -> 0.22), so it is kept only as the baseline of the style-encoder validation.

Protocol, for session i evaluated after session t (m_{i,t}):
  * StyleRetention : PRIMARY forgetting metric. Mean probability mass on session i's genres that the frozen
               learned style encoder (metrics/style_encoder.py, with the reject class "other") assigns to
               generated 5 s windows for session-i music. Validated on aistpp_genre5: robust to smoothing /
               jitter / tempo in range, 0.77 10-genre accuracy on unseen dancers, right session 0.85-0.95 vs
               <= 0.08 for any wrong one, and non-dance (untrained generator, static, shuffled, everyday motion)
               <= 0.017 on any session. F uses the drop m_{i,i} - m_{i,t}. StyleAcc is its hard (arg-max) version.
  * DanceRate: 1 - P(other), the share of the output recognised as dance at all
  * StyleFD  : Frechet distance of encoder embeddings to session i's reference windows; drift() is the
               label-free version between two generated sets (the model after session i vs after t)
  * FID_k / FID_g : quality metrics on the repo's kinetic (66) / geometric (32) features averaged over the
               same windows (whole-sequence values match the repo extractors exactly, see self_test), with ONE
               frozen benchmark-wide normalisation (pooled reference windows; constant features dropped)
  * reference: VAE-reconstructed GT when fid_target='vae' (rebuild with --ref_feature_dir)
  * samples  : pool several generated samples per test music
  * summary  : the paper's F_avg / A_avg via forgetting_summary(), plus each session's GT noise floor

CLI (precompute references, probe and noise floor for a benchmark from GT; re-run with --ref_feature_dir
pointing at VAE reconstructions once the frozen VAE exists):
  python -m metrics.cl_metrics build --benchmark aistpp_genre5
  python -m metrics.cl_metrics self_test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from metrics.m2d_features.kinetic import extract_kinetic_features
from metrics.m2d_features.manual_new import extract_manual_features
from metrics.tools.utils import calculate_activation_statistics_np, calculate_frechet_distance_np
from utils.motion_process import recover_joint_positions_272

FPS, WIN, STRIDE = 30, 150, 30
JOINTS = 22


# ----------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------
def kinetic_per_frame(pos: np.ndarray, frame_time: float = 1.0 / FPS, sliding_window: int = 2) -> np.ndarray:
    """Vectorised per-frame terms of metrics.m2d_features.kinetic (frames 1..T-1), shape (T-1, J*3).

    Per joint: horizontal KE, vertical KE (y-up), energy expenditure. The acceleration term keeps the
    upstream operator-precedence quirk (v1 = p[k] - p[k-1] / dt) so FID_k stays comparable with the
    AIST++ / Bailando / LODGE literature.
    """
    T, sw, dt = len(pos), sliding_window, frame_time
    i = np.arange(1, T)

    d = np.zeros_like(pos, dtype=np.float64)
    d[1:] = pos[1:] - pos[:-1]
    cs = np.concatenate([np.zeros((1,) + pos.shape[1:]), np.cumsum(d, 0)])
    lo, hi = np.maximum(i - sw, 1), np.minimum(i + sw, T - 1)
    v = (cs[hi + 1] - cs[lo]) / ((hi - lo + 1)[:, None, None] * dt)
    ke_h = v[..., 0] ** 2 + v[..., 2] ** 2
    ke_v = v[..., 1] ** 2

    term = np.zeros_like(pos, dtype=np.float64)
    k = np.arange(1, T - 1)
    term[k] = ((pos[k + 1] - pos[k]) / dt - (pos[k] - pos[k - 1] / dt)) / dt
    ct = np.concatenate([np.zeros((1,) + pos.shape[1:]), np.cumsum(term, 0)])
    lo2, hi2 = np.maximum(i - sw, 1), np.minimum(i + sw, T - 2)
    ee = np.linalg.norm((ct[hi2 + 1] - ct[lo2]) / (hi2 - lo2 + 1)[:, None, None], axis=-1)

    return np.stack([ke_h, ke_v, ee], -1).reshape(T - 1, -1)


def joints_from_272(feat: np.ndarray) -> np.ndarray:
    j = recover_joint_positions_272(feat, JOINTS).astype(np.float64)
    j[:, 1:, :] -= j[:, 0:1, :]  # root-relative, as in M2DMetrics.update
    return j


CACHE_ROOT = Path("raw_data/cl_sessions/_feature_cache")


def cached_window_features(feature_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """window_features of a stored 272-dim sequence, cached per feature directory (deterministic)."""
    path = CACHE_ROOT / Path(feature_dir).name / f"{name}_w{WIN}_s{STRIDE}.npz"
    if path.exists():
        z = np.load(path)
        return z["kin"], z["geo"]
    kin, geo = window_features(joints_from_272(np.load(Path(feature_dir) / f"{name}.npy")))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, kin=kin, geo=geo)
    return kin, geo


def _cache_one(args):
    cached_window_features(*args)


def fill_cache(feature_dir: Path, names: list[str], workers: int) -> None:
    from multiprocessing import Pool

    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_cache_one, [(feature_dir, n) for n in names], chunksize=4)):
            if (i + 1) % 100 == 0 or i + 1 == len(names):
                print(f"  cached {i + 1}/{len(names)} ({Path(feature_dir).name})", flush=True)


def window_features(joints: np.ndarray, win: int = WIN, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """(n_windows, 66) kinetic and (n_windows, 32) geometric window features of one root-relative sequence."""
    per = [kinetic_per_frame(joints), extract_manual_features(joints, reduce=False).astype(np.float64)]
    T = len(joints)
    # a window over frames [s, s+win) uses per-frame rows s .. s+win-2 (frames s+1 .. s+win-1)
    spans = [(s, s + win - 1) for s in range(0, T - win + 1, stride)] or [(0, T - 1)]
    out = []
    for p in per:
        c = np.concatenate([np.zeros((1, p.shape[1])), np.cumsum(p, 0)])
        out.append(np.stack([(c[b] - c[a]) / (b - a) for a, b in spans]))
    return out[0], out[1]


# ----------------------------------------------------------------------------------------------
# Distances, probe, summaries
# ----------------------------------------------------------------------------------------------
def frozen_fid(ref: np.ndarray, gen: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    r, g = (ref - mean) / std, (gen - mean) / std
    m1, c1 = calculate_activation_statistics_np(r)
    m2, c2 = calculate_activation_statistics_np(g)
    return float(calculate_frechet_distance_np(m1, c1, m2, c2))


class GenreProbe:
    """Multinomial logistic regression over fine labels on standardised [kinetic, geometric] windows;
    `session_of_class` maps each fine label (e.g. AIST++ genre) to the session that contains it."""

    def __init__(self, scaler_mean, scaler_scale, coef, intercept, session_of_class):
        self.mu, self.sd, self.W, self.b = scaler_mean, scaler_scale, coef, intercept
        self.session_of_class = np.asarray(session_of_class)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, session_of_class) -> "GenreProbe":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(X), y)
        return cls(sc.mean_, sc.scale_, clf.coef_, clf.intercept_, session_of_class)

    def predict_session(self, X: np.ndarray) -> np.ndarray:
        return self.session_of_class[np.argmax(((X - self.mu) / self.sd) @ self.W.T + self.b, axis=1)]

    def save(self, path: Path) -> None:
        np.savez(path, mu=self.mu, sd=self.sd, W=self.W, b=self.b, session_of_class=self.session_of_class)

    @classmethod
    def load(cls, path: Path) -> "GenreProbe":
        z = np.load(path)
        return cls(z["mu"], z["sd"], z["W"], z["b"], z["session_of_class"])


def forgetting_summary(M: np.ndarray, higher_is_better: bool = False) -> dict:
    """M[i, t] = metric of session i after training session t (NaN for t < i).

    Paper definitions: F_i = mean_{t>i} (m_{i,t} - m_{i,i}) over old sessions i = 0..K-2,
    F_avg = mean_i F_i, A_avg = mean of m_{i,i} over new sessions i = 1..K-1. For higher-is-better
    metrics the sign of the difference is flipped so that F > 0 always means forgetting.
    Also returns the relative version, mean_{i,t>i} (m_{i,t} - m_{i,i}) / |m_{i,i}|.
    """
    K = M.shape[0]
    sgn = -1.0 if higher_is_better else 1.0
    F_i = [float(np.mean([sgn * (M[i, t] - M[i, i]) for t in range(i + 1, K)])) for i in range(K - 1)]
    F_rel = [float(np.mean([sgn * (M[i, t] - M[i, i]) / abs(M[i, i]) for t in range(i + 1, K)])) for i in range(K - 1)]
    return {
        "F_i": F_i,
        "F_avg": float(np.mean(F_i)),
        "F_rel_avg": float(np.mean(F_rel)),
        "A_avg": float(np.mean([M[i, i] for i in range(1, K)])),
        "final_old_mean": float(np.mean([M[i, K - 1] for i in range(K - 1)])),
    }


def root_relative(joints: np.ndarray) -> np.ndarray:
    """Global joints -> M2DMetrics convention (joints 1.. relative to the root; the root keeps its position)."""
    j = joints.astype(np.float64).copy()
    j[:, 1:, :] -= j[:, 0:1, :]
    return j


def style_fd(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between two sets of (already L2-normalised) style embeddings."""
    return frozen_fid(a, b, np.zeros(a.shape[1]), np.ones(a.shape[1]))


class CLEvaluator:
    """Precomputed references + the frozen style encoder; scores generated motion for one session."""

    def __init__(self, benchmark_dir: Path):
        from metrics.style_encoder import StyleEncoder, benchmark_session_of_class, ckpt_path

        e = Path(benchmark_dir) / "eval"
        self.refs = np.load(e / "refs.npz")
        self.noise = json.loads((e / "noise.json").read_text())
        man = json.loads((Path(benchmark_dir) / "manifest.json").read_text())
        self.style = StyleEncoder(ckpt_path(man["dataset"], -1))
        self.soc = benchmark_session_of_class(self.style.classes, man)

    def session_metrics(self, session: int, generated_joints: list[np.ndarray]) -> dict:
        """generated_joints: GLOBAL joint positions (T, 22, 3), e.g. features2joints output, for all
        generated samples of all test music of `session`."""
        enc = self.style.encode(generated_joints)
        k, g = map(np.concatenate, zip(*(window_features(root_relative(j)) for j in generated_joints)))
        R, keep = self.refs, self.refs["geo_keep"]
        return {
            **self.style.session_scores(enc["prob"], session, self.soc),  # StyleRetention (primary), StyleAcc
            "StyleFD": style_fd(R[f"s{session}_style"], enc["emb"]),
            "FID_k": frozen_fid(R[f"s{session}_kin"], k, R["kin_mean"], R["kin_std"]),
            "FID_g": frozen_fid(R[f"s{session}_geo"][:, keep], g[:, keep], R["geo_mean"], R["geo_std"]),
            "n_windows": int(len(k)),
        }

    def drift(self, generated_a: list[np.ndarray], generated_b: list[np.ndarray]) -> float:
        """Label-free behavioural drift: StyleFD between two generated sets for the same music
        (e.g. the model right after session i vs after session t)."""
        return style_fd(self.style.encode(generated_a)["emb"], self.style.encode(generated_b)["emb"])


# ----------------------------------------------------------------------------------------------
# Build references / probe / noise floor from ground truth
# ----------------------------------------------------------------------------------------------
def build(benchmark: str, ref_feature_dir: str | None, probe_max_windows: int, seed: int) -> None:
    root = Path("raw_data/cl_sessions") / benchmark
    man = json.loads((root / "manifest.json").read_text())
    gt_dir = Path(man["feature_dir"])
    ref_dir = Path(ref_feature_dir) if ref_feature_dir else gt_dir
    K = len(man["sessions"])
    rng = np.random.default_rng(seed)
    read = lambda p: [l.strip() for l in p.open() if l.strip()]

    # fine probe labels: AIST++ genre code from the sequence name ("gLH_sBM_..."), FineDance family = session
    fine_label = (lambda n, s: n[1:3]) if man["dataset"] == "aistpp" else (lambda n, s: str(s))

    test, owners, probe_X, probe_lab = {}, {}, [], []
    for s in range(K):
        names = read(root / f"s{s}" / "test.txt")
        per_seq = [cached_window_features(ref_dir, n) for n in names]
        test[s] = (np.concatenate([p[0] for p in per_seq]), np.concatenate([p[1] for p in per_seq]))
        owners[s] = np.concatenate([[i] * len(p[0]) for i, p in enumerate(per_seq)])
        for n in read(root / f"s{s}" / "train.txt"):  # probe data: GT train windows (evaluator only)
            k, g = cached_window_features(gt_dir, n)
            probe_X.append(np.concatenate([k, g], 1))
            probe_lab += [fine_label(n, s)] * len(k)

    classes = sorted(set(probe_lab))
    session_of_class = [next(s for s in range(K) for n in read(root / f"s{s}" / "train.txt")
                             if fine_label(n, s) == c) for c in classes]
    X, y = np.concatenate(probe_X), np.array([classes.index(c) for c in probe_lab])
    keep_idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:probe_max_windows] for c in range(len(classes))])
    probe = GenreProbe.fit(X[keep_idx], y[keep_idx], session_of_class)

    # one frozen benchmark-wide normalisation from the pooled reference windows of all sessions
    pool_k = np.concatenate([test[s][0] for s in range(K)])
    pool_g = np.concatenate([test[s][1] for s in range(K)])
    geo_keep = pool_g.std(0) > 1e-6
    refs = {"kin_mean": pool_k.mean(0), "kin_std": pool_k.std(0) + 1e-10, "geo_keep": geo_keep,
            "geo_mean": pool_g[:, geo_keep].mean(0), "geo_std": pool_g[:, geo_keep].std(0)}
    fidk = lambda a, b: frozen_fid(a, b, refs["kin_mean"], refs["kin_std"])
    fidg = lambda a, b: frozen_fid(a[:, geo_keep], b[:, geo_keep], refs["geo_mean"], refs["geo_std"])
    acc = lambda kin, geo, s: float((probe.predict_session(np.concatenate([kin, geo], 1)) == s).mean())

    noise = {}
    for s in range(K):
        kin, geo = test[s]
        refs[f"s{s}_kin"], refs[f"s{s}_geo"] = kin, geo
        fk, fg, ga = [], [], []
        for _ in range(20):  # split-half by sequence, so windows of one sequence stay together
            half = rng.permutation(owners[s].max() + 1)[: (owners[s].max() + 1) // 2]
            a = np.isin(owners[s], half)
            fk.append(fidk(kin[a], kin[~a]))
            fg.append(fidg(geo[a], geo[~a]))
            ga.append(acc(kin[~a], geo[~a], s))
        others = [o for o in range(K) if o != s]
        noise[f"s{s}"] = {
            "n_test_seqs": int(owners[s].max() + 1), "n_windows": int(len(kin)),
            "GenreAcc_gt": acc(kin, geo, s), "GenreAcc_halfset_std": float(np.std(ga)),
            # complete forgetting = another session's GT scored as if generated for session s
            "GenreAcc_complete_forgetting": max(acc(*test[o], s) for o in others),
            "FID_k_noise_mean": float(np.mean(fk)), "FID_k_noise_std": float(np.std(fk)),
            "FID_k_complete_forgetting": min(fidk(kin, test[o][0]) for o in others),
            "FID_g_noise_mean": float(np.mean(fg)), "FID_g_noise_std": float(np.std(fg)),
            "FID_g_complete_forgetting": min(fidg(geo, test[o][1]) for o in others),
        }
    # learned style encoder: the primary forgetting metrics (see metrics/style_encoder.py)
    from metrics.style_encoder import StyleEncoder, benchmark_session_of_class, ckpt_path

    style = StyleEncoder(ckpt_path(man["dataset"], -1))
    soc = benchmark_session_of_class(style.classes, man)
    scores = lambda prob, s: style.session_scores(prob, s, soc)
    enc = {s: style.encode([recover_joint_positions_272(np.load(ref_dir / f"{n}.npy"), JOINTS)
                            for n in read(root / f"s{s}" / "test.txt")]) for s in range(K)}
    for s in range(K):
        e, others = enc[s], [o for o in range(K) if o != s]
        refs[f"s{s}_style"] = e["emb"]
        seqs, fd_n, ret_n = np.unique(e["owner"]), [], []
        for _ in range(20):
            a = np.isin(e["owner"], rng.permutation(seqs)[: len(seqs) // 2])
            fd_n.append(style_fd(e["emb"][a], e["emb"][~a]))
            ret_n.append(scores(e["prob"][~a], s)["StyleRetention"])
        gt = scores(e["prob"], s)
        noise[f"s{s}"].update({
            "StyleRetention_gt": gt["StyleRetention"], "StyleAcc_gt": gt["StyleAcc"],
            "StyleRetention_halfset_std": float(np.std(ret_n)),
            "StyleRetention_complete_forgetting": max(scores(enc[o]["prob"], s)["StyleRetention"]
                                                      for o in others),
            "StyleFD_noise_mean": float(np.mean(fd_n)), "StyleFD_noise_std": float(np.std(fd_n)),
            "StyleFD_complete_forgetting": min(style_fd(e["emb"], enc[o]["emb"]) for o in others),
        })
        n = noise[f"s{s}"]
        print(f"[{benchmark}] S{s}: {n['n_test_seqs']} seqs / {n['n_windows']} windows | StyleRetention GT "
              f"{n['StyleRetention_gt']:.2f} (half-set std {n['StyleRetention_halfset_std']:.3f}) -> complete forgetting "
              f"{n['StyleRetention_complete_forgetting']:.2f} | StyleFD noise {n['StyleFD_noise_mean']:.3f}+-"
              f"{n['StyleFD_noise_std']:.3f} vs {n['StyleFD_complete_forgetting']:.3f}")

    out = root / "eval"
    out.mkdir(exist_ok=True)
    np.savez(out / "refs.npz", **refs)
    probe.save(out / "probe.npz")
    (out / "noise.json").write_text(json.dumps(
        {"window_frames": WIN, "stride_frames": STRIDE, "reference_dir": str(ref_dir),
         "probe_classes": classes, "session_of_class": session_of_class, **noise}, indent=2))
    print(f"[{benchmark}] wrote {out}")


def self_test() -> None:
    """Vectorised kinetic features must reproduce the repo extractor on whole sequences."""
    files = sorted(Path("raw_data/aistpp/motions_272_30fps").glob("*.npy"))[:3]
    for f in files:
        j = joints_from_272(np.load(f))
        ref = extract_kinetic_features(j.astype(np.float32))
        ours = kinetic_per_frame(j).mean(0)
        rel = np.abs(ours - ref).max() / (np.abs(ref).max() + 1e-12)
        g_ref = extract_manual_features(j)
        g_ours = extract_manual_features(j, reduce=False).mean(0)
        print(f"{f.stem}: kinetic max rel err {rel:.2e} | geometric max abs err {np.abs(g_ours - g_ref).max():.2e}")


def cache_benchmark(benchmark: str, workers: int) -> None:
    root = Path("raw_data/cl_sessions") / benchmark
    man = json.loads((root / "manifest.json").read_text())
    names = sorted({l.strip() for split in ("train", "val", "test")
                    for l in (root / f"joint_{split}.txt").open() if l.strip()})
    fill_cache(Path(man["feature_dir"]), names, workers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "cache", "self_test"])
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--ref_feature_dir", default=None, help="272-dim reference features (default: GT)")
    ap.add_argument("--probe_max_windows", type=int, default=4000, help="probe training windows per class")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.benchmark, a.ref_feature_dir, a.probe_max_windows, a.seed)
    elif a.cmd == "cache":
        cache_benchmark(a.benchmark, a.workers)
    else:
        self_test()
