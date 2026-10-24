#!/usr/bin/env python3
"""
E1 -- spectral evidence for the temporal feature operator C_k (forward passes only, no gradients).

For every session boundary k (theta_{k-1} = last.ckpt of session k-1; past pool = training windows of sessions < k):
  * Theorem 1 (Ky Fan): Psi(P_hat) equals the discarded spectral mass and no random rank-M_hat projector beats it
  * rank: BIC rank M_hat (N_eff = N D, and the conservative N_eff = N), spectral gap, effective rank r_eff
  * temporal vs channel axis: variance kept by equal-size temporal and channel representations
  * Theorem 2: ||C(N') - C||_2 and the subspace distance against sqrt(1/N' - 1/N) (slope 1 = the sqrt(1/N) rate),
    and the data-reuse check of Sec. app:setup: C from held-out test windows vs C from training windows
  * Proposition fw: OMP modal error eps_mod(t) against 2 diam_Z / sqrt(t + 2), with the fitted empirical rate
  * Lemma modal: eps_B <= sqrt(lambda_1) eps_mod + tail, for the TMR buffer and random buffers (tightness)
  * what temporal modes are: Toeplitz index of C_k, DCT alignment and frequency of the leading modes, Szego
    comparison of the spectrum with the circulant (power-spectrum) eigenvalues, and what modal energies encode
    (tempo, genre) on held-out music pieces
  * stability of the modal subspace across checkpoints (theta_{k-1} vs theta_k and theta_0) and linear CKA of phi
  * buffer composition and the Fig. 3c grid coverage for TMR, its Tab. 4 ablations and the classical baselines

Usage (repo root):
  python scripts/evidence/e1_spectral.py --config configs/cl/ldf_aistpp_genre5_s0.yaml --benchmark aistpp_genre5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev_common as E
import tmr_select as S


# ----------------------------------------------------------------------------------------------
# Theory checks
# ----------------------------------------------------------------------------------------------
def spectrum_stats(ops, N, D) -> dict:
    lam = ops["lam"]
    M, bic = S.bic_rank(lam, N * D)
    M_N, _ = S.bic_rank(lam, N)
    tr = lam.sum()
    gap = lam[M - 1] - lam[M] if M < len(lam) else float("nan")
    return {
        "M_hat": M, "M_hat_conservative_Neff_N": M_N, "rank": bic["rank"], "M_max": bic["M_max"],
        "eigenvalues": lam, "trace": tr, "r_eff": tr / lam[0], "explained_at_M_hat": lam[:M].sum() / tr,
        "gap_at_M_hat": gap, "relative_gap_at_M_hat": gap / lam[M - 1], "lambda_1": lam[0],
        "tail_Psi": lam[M:].sum(), "bic_criterion": bic["criterion"],
    }


def ky_fan(ops, M, n_random, rng) -> dict:
    """Theorem 1: Psi(P) = tr((I - P) C_k) is minimised by the top-M eigenspace. Controls: uniformly random rank-M
    subspaces, and random rank-M subspaces of the leading 2M eigenvectors (a much harder control)."""
    C, U, lam = ops["C"], ops["U"], ops["lam"]
    T = len(C)
    psi = lambda V: float(np.trace(C) - np.trace(V.T @ C @ V))
    psi_hat = psi(U[:, :M])
    rand = np.array([psi(np.linalg.qr(rng.standard_normal((T, M)))[0]) for _ in range(n_random)])
    W = U[:, :min(T, 2 * M)]
    near = np.array([psi(W @ np.linalg.qr(rng.standard_normal((W.shape[1], M)))[0]) for _ in range(n_random)])
    return {"Psi_hat": psi_hat, "discarded_mass": float(lam[M:].sum()),
            "identity_abs_err": abs(psi_hat - float(lam[M:].sum())), "trace": float(np.trace(C)),
            "Psi_random_mean": rand.mean(), "Psi_random_min": rand.min(),
            "frac_random_beating_hat": float((rand < psi_hat - 1e-12).mean()),
            "Psi_top2M_random_mean": near.mean(), "Psi_top2M_random_min": near.min(),
            "frac_top2M_random_beating_hat": float((near < psi_hat - 1e-12).mean()),
            "random_samples": rand, "top2M_samples": near}


def axis_comparison(ops, T, D) -> dict:
    """Variance kept by the top-M temporal modes vs the top-(M D / T) channel modes (equal representation size)."""
    ft = np.cumsum(ops["lam"]) / ops["lam"].sum()
    fc = np.cumsum(ops["lam_c"]) / ops["lam_c"].sum()
    rows = []
    for M in range(1, T + 1):
        Mc = int(round(M * D / T))
        if Mc > len(fc):
            break
        rows.append({"M_temporal": M, "M_channel": Mc, "kept_temporal": ft[M - 1], "kept_channel": fc[Mc - 1]})
    return {"equal_size_curve": rows}


def consistency(phi, ops, M, reps, rng, phi_test=None) -> dict:
    """Theorem 2 rate by subsampling the pool without replacement (variance ~ 1/N' - 1/N), and the data-reuse check:
    the operator of held-out windows of the same sessions vs what resampling noise alone predicts at that size."""
    N = len(phi)
    C, P = ops["C"], ops["U"][:, :M] @ ops["U"][:, :M].T
    nC = np.linalg.norm(C, 2)
    sizes = sorted({int(x) for x in np.geomspace(max(8, N // 64), max(9, N // 2), 7)})
    rows = []
    for n in sizes:
        op, sub = [], []
        for _ in range(reps):
            Cn, _ = S.temporal_operator(phi[rng.choice(N, n, replace=False)])
            _, Un = S.eig_desc(Cn)
            op.append(np.linalg.norm(Cn - C, 2) / nC)
            sub.append(np.linalg.norm(Un[:, :M] @ Un[:, :M].T - P))
        rows.append({"N_sub": n, "x": math.sqrt(1 / n - 1 / N), "op_err_mean": np.mean(op), "op_err_std": np.std(op),
                     "subspace_err_mean": np.mean(sub), "subspace_err_std": np.std(sub)})
    x = np.log([r["x"] for r in rows])
    fits = {key: np.polyfit(x, np.log([r[key] for r in rows]), 1) for key in ("op_err_mean", "subspace_err_mean")}
    out = {"curve": rows, "slope_operator": float(fits["op_err_mean"][0]),
           "slope_subspace": float(fits["subspace_err_mean"][0])}
    if phi_test is not None and len(phi_test) >= 8:
        Ct, _ = S.temporal_operator(phi_test)
        _, Ut = S.eig_desc(Ct)
        n = len(phi_test)
        xt = math.log(math.sqrt(1 / n + 1 / N))  # two independent samples: both operators carry noise
        pred = lambda key: float(math.exp(np.polyval(fits[key], xt)))
        out["heldout"] = {"N_test": n, "op_err": float(np.linalg.norm(Ct - C, 2) / nC),
                          "op_err_expected_from_sampling": pred("op_err_mean"),
                          "subspace_err": float(np.linalg.norm(Ut[:, :M] @ Ut[:, :M].T - P)),
                          "subspace_err_expected_from_sampling": pred("subspace_err_mean")}
    return out


def fw_rate(Z, B) -> dict:
    """Proposition fw on the pooled budget: eps_mod after t OMP iterations vs 2 diam_Z / sqrt(t + 2), with the
    Frank-Wolfe-gap termination; the literal positive-correlation filter of Algorithm 1 is run alongside to show
    where it stops."""
    sq = (Z.astype(np.float64) ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * (Z @ Z.T).astype(np.float64)
    diam = float(np.sqrt(max(0.0, d2.max())))
    _, _, trace = S.omp_simplex(Z, B, trace=True)
    _, _, literal = S.omp_simplex(Z, B, trace=True, stop="positive")
    t = np.arange(1, len(trace) + 1)
    lo = max(2, len(t) // 10)
    tr = np.maximum(np.array(trace), 1e-300)
    rate = float(-np.polyfit(np.log(t[lo - 1:]), np.log(tr[lo - 1:]), 1)[0]) if len(t) > lo + 2 else float("nan")
    bound = 2 * diam / np.sqrt(t + 2)
    return {"diam_Z": diam, "eps_mod": trace, "bound": bound, "iterations": len(trace), "budget": B,
            "empirical_rate_exponent": rate,
            "max_ratio_to_bound": float(np.max(tr / bound)) if trace else float("nan"),
            "eps_mod_final": trace[-1] if trace else float("nan"), "eps_mod_at_1": trace[0] if trace else float("nan"),
            "stopped_early": len(trace) < B, "literal_algorithm1_atoms": len(literal),
            "literal_algorithm1_eps_mod": literal[-1] if literal else float("nan")}


def lemma_context(phi, ops, M) -> dict:
    """Buffer-independent pieces of Lemma modal: the projector and tau_k = max_i ||(I - P_hat) phi~_i||_F."""
    U = ops["U"][:, :M]
    tau2 = 0.0
    for s in range(0, len(phi), 64):
        x = phi[s:s + 64] - ops["mean"]
        r = x - np.einsum("tm,nmd->ntd", U, np.einsum("tm,ntd->nmd", U, x, optimize=True), optimize=True)
        tau2 = max(tau2, float((r ** 2).sum((1, 2)).max()))
    N, T, D = phi.shape
    return {"U": U, "lam": ops["lam"], "mean": ops["mean"], "tau": math.sqrt(tau2), "Psi": float(ops["lam"][M:].sum()),
            "N": N, "D": D}


def lemma_check(ctx, phi, Z, idx, w) -> dict:
    """Lemma modal: eps_B <= sqrt(lambda_1) eps_mod + min{tau_k, sqrt(||w||_inf N D Psi)}, with both halves of the
    proof (in-subspace and out-of-subspace parts of Xi) reported separately."""
    U, lam = ctx["U"], ctx["lam"]
    Xi = ctx["mean"] - np.tensordot(np.asarray(w, np.float64), phi[np.asarray(idx, int)].astype(np.float64), 1)
    eps_B = float(np.linalg.norm(Xi))
    eps_mod = S.mean_error(Z, idx, w)
    in_sub = float(np.linalg.norm(U.T @ Xi))
    out_sub = float(np.linalg.norm(Xi - U @ (U.T @ Xi)))
    tail = min(ctx["tau"], math.sqrt(float(np.max(w)) * ctx["N"] * ctx["D"] * ctx["Psi"]))
    rhs = math.sqrt(lam[0]) * eps_mod + tail
    return {"eps_B": eps_B, "eps_mod": eps_mod, "in_subspace": in_sub, "in_subspace_bound": math.sqrt(lam[0]) * eps_mod,
            "out_subspace": out_sub, "tail_bound": tail, "tau_k": ctx["tau"], "rhs": rhs, "ratio": eps_B / rhs,
            "holds": eps_B <= rhs * (1 + 1e-9)}


# ----------------------------------------------------------------------------------------------
# What the temporal modes are
# ----------------------------------------------------------------------------------------------
def mode_interpretation(ops, M, fps=E.TOKEN_FPS, k_show=12) -> dict:
    """If the audio-conditioned trajectories are (wide-sense) stationary over a window, C_k is Toeplitz, its
    eigenvectors approach the DCT basis and its spectrum approaches the circulant eigenvalues, i.e. samples of the
    power spectrum (Szego). Then the modal projector is a data-adaptive low-/band-pass filter and whitening equalises
    the retained frequency bands."""
    C, U, lam = ops["C"], ops["U"], ops["lam"]
    T = len(C)
    r = np.array([np.diagonal(C, d).mean() for d in range(T)])
    toep = r[np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])]
    toep_index = float(np.linalg.norm(C - toep) / np.linalg.norm(C))
    n = np.arange(T)
    dct = np.array([np.cos(np.pi * (n + 0.5) * k / T) for k in range(T)])
    dct /= np.linalg.norm(dct, axis=1, keepdims=True)
    f = np.fft.rfftfreq(8 * T, d=1 / fps)
    modes = []
    for m in range(min(k_show, T)):
        u = U[:, m]
        al = (dct @ u) ** 2
        spec = np.abs(np.fft.rfft(u, n=8 * T)) ** 2
        modes.append({"m": m + 1, "lambda_share": lam[m] / lam.sum(), "dct_best_k": int(al.argmax()),
                      "dct_alignment": float(al.max()), "dct_freq_hz": al.argmax() * fps / (2 * T),
                      "peak_freq_hz": float(f[spec.argmax()]), "in_M_hat": m < M})
    # Strang circulant approximation of the Toeplitz projection: eigenvalues = DFT of its first row
    c = np.array([r[min(j, T - j)] for j in range(T)])
    psi = np.sort(np.clip(np.real(np.fft.fft(c)), 0, None))[::-1]
    szego = float(np.abs(psi / psi.sum() - lam / lam.sum()).sum() / 2)
    top = modes[:max(1, min(M, k_show))]
    # mode 1 ~ constant over the window: windows differ mostly in their time-averaged audio content (song / genre),
    # and the remaining modes carry the within-window dynamics that whitening puts on an equal footing
    dc = modes[0]["dct_best_k"] == 0 and modes[0]["dct_alignment"] > 0.9
    return {"toeplitz_index": toep_index, "autocov": r, "modes": modes, "szego_tv_distance": szego,
            "circulant_eigs_sorted": psi, "mean_dct_alignment_top_M": float(np.mean([x["dct_alignment"] for x in top])),
            "max_freq_in_M_hat_hz": float(max(x["dct_freq_hz"] for x in top)),
            "frequency_resolution_hz": fps / (2 * T), "mode1_is_window_average": bool(dc),
            "mode1_share": float(lam[0] / lam.sum()),
            "whitening_gain_dynamic_vs_mode1": float(np.sqrt(lam[0] / lam[min(M, T) - 1]))}


def tempo_of(names, music_dir: Path | None, cache: Path) -> tuple[dict, str]:
    """BPM per sequence: librosa beat tracking on one wav per AIST++ music piece (folded into [70, 140)); falls back
    to the AIST++ design tempos (80 + 10 id; House 110 + 5 id) when wavs are unavailable."""
    meta = {n: E.parse_meta(n) for n in set(names)}
    if not meta or not all(meta.values()):
        return {}, "unavailable"
    table = {n: (110 + 5 * m["music_id"]) if m["music"].startswith("HO") else 80 + 10 * m["music_id"]
             for n, m in meta.items()}
    if music_dir is None or not Path(music_dir).exists():
        return table, "aist_design_table"
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    piece_wav = {}
    for n, m in sorted(meta.items()):
        if m["music"] not in cached and (Path(music_dir) / f"{n}.wav").exists():
            piece_wav.setdefault(m["music"], Path(music_dir) / f"{n}.wav")
    if piece_wav:
        import librosa

        for piece, wav in sorted(piece_wav.items()):
            y, sr = librosa.load(wav, sr=22050)
            bpm = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
            while bpm >= 140:
                bpm /= 2
            while 0 < bpm < 70:
                bpm *= 2
            cached[piece] = bpm
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached, indent=1))
    est = {n: cached.get(m["music"], table[n]) for n, m in meta.items()}
    agree = float(np.mean([abs(est[n] - table[n]) <= 3 for n in est]))
    E.log(f"tempo: librosa estimate within 3 BPM of the AIST++ design tempo for {agree:.0%} of sequences")
    return est, f"librosa (within 3 BPM of the AIST++ design tempo for {agree:.0%})"


def what_modes_encode(phi, ops, M, pool, tempo: dict, seed: int) -> dict:
    """Cross-validated decoding of tempo (ridge R^2) and genre (logistic accuracy) from equal-size summaries of a
    window: whitened temporal-modal energies, channel-modal energies, PCA of the time-pooled features. Folds are
    grouped by music piece, so the test songs are never seen in training."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    phi_c = phi - ops["mean"]
    energy = lambda x: np.log((x ** 2).mean(2) + 1e-12)
    feats = {"temporal_modes": energy(np.einsum("tm,ntd->nmd", ops["U"][:, :M] / np.sqrt(ops["lam"][:M]), phi_c,
                                                optimize=True))}
    if "U_c" in ops:
        feats["channel_modes"] = energy(np.einsum("dm,ntd->nmt", ops["U_c"][:, :M] / np.sqrt(ops["lam_c"][:M]), phi_c,
                                                  optimize=True))
    feats["time_pooled_pca"] = PCA(min(M, phi.shape[2], len(phi)), random_state=seed).fit_transform(phi.mean(1))
    piece = [m.get("music") or n for m, n in zip(pool.meta, pool.names)]
    groups = np.unique(piece, return_inverse=True)[1]
    if len(np.unique(groups)) < 3:
        return {"n_features": M, "skipped": "fewer than 3 music pieces"}
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    y_tempo = np.array([tempo.get(n, np.nan) for n in pool.names], float)
    genres = np.array([m.get("genre") or "" for m in pool.meta])
    out = {"n_features": M}
    # scores on the pooled out-of-fold predictions: a held-out fold may hold a single tempo, where a per-fold R^2
    # is undefined
    for name, X in feats.items():
        r = {}
        ok = np.isfinite(y_tempo)
        if ok.sum() > 20 and len(np.unique(y_tempo[ok])) > 2:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
            pred = cross_val_predict(model, X[ok], y_tempo[ok], cv=cv, groups=groups[ok])
            r["tempo_r2"] = float(r2_score(y_tempo[ok], pred))
        if len(set(genres)) > 1:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            r["genre_acc"] = float(accuracy_score(genres, cross_val_predict(model, X, genres, cv=cv, groups=groups)))
        out[name] = r
    return out


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X.reshape(len(X), -1).astype(np.float64)
    Y = Y.reshape(len(Y), -1).astype(np.float64)
    X -= X.mean(0)
    Y -= Y.mean(0)
    Kx, Ky = X @ X.T, Y @ Y.T
    return float((Kx * Ky).sum() / np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()))


# ----------------------------------------------------------------------------------------------
# Buffers
# ----------------------------------------------------------------------------------------------
def buffers_report(phi, ops, Z, M, pool, groups, budget, n_random, seed) -> dict:
    raw = phi.reshape(len(phi), -1)
    X2 = E.pca2(raw)
    bufs, info = S.select_buffers(phi, groups, budget, seed=seed, ops=ops)
    rng = np.random.default_rng(seed + 1)

    def metrics(idx, w):
        return {"eps_B": S.mean_error(raw, idx, w), "eps_mod": S.mean_error(Z, idx, w),
                "grid_coverage": S.grid_coverage(X2, idx, groups), "n_nonzero": int((np.asarray(w) > 1e-12).sum()),
                "max_weight": float(np.max(w)), **S.composition(pool.meta, groups, idx)}

    rules = {name: {**metrics(b["idx"], b["w"]), "seconds": b["seconds"]} for name, b in bufs.items()}
    rand_bufs = [S.random_quota(groups, info["quotas"], rng) for _ in range(n_random)]
    rand = [metrics(i, S.uniform(i)) for i in rand_bufs]
    rules["random_mean"] = {k: float(np.mean([r[k] for r in rand])) for k in rand[0]}
    rules["random_std"] = {k: float(np.std([r[k] for r in rand])) for k in rand[0]}
    ctx = lemma_context(phi, ops, M)
    lem = [lemma_check(ctx, phi, Z, i, S.uniform(i)) for i in rand_bufs[:50]]
    lemma = {"tmr": lemma_check(ctx, phi, Z, bufs["tmr"]["idx"], bufs["tmr"]["w"]),
             "random_ratio_mean": float(np.mean([l["ratio"] for l in lem])),
             "random_ratio_max": float(np.max([l["ratio"] for l in lem])),
             "random_all_hold": bool(all(l["holds"] for l in lem))}
    return {"info": info, "rules": rules, "lemma": lemma,
            "buffers": {k: {"idx": v["idx"], "w": v["w"]} for k, v in bufs.items()}}


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def figures(res: dict, out: Path, k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp, kf, cs, fw, mi = res["spectrum"], res["ky_fan"], res["consistency"], res["fw"], res["modes"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
    lam = np.array(sp["eigenvalues"])
    ax[0].semilogy(np.arange(1, len(lam) + 1), lam / lam.sum(), "o-", ms=3, label="temporal $C_k$")
    lc = np.array(res["lam_c_share"])
    ax[0].semilogy(np.arange(1, len(lc) + 1), lc, "-", lw=1, alpha=.6, label="channel $C^{chan}$")
    ax[0].axvline(sp["M_hat"], color="k", ls="--", lw=.8, label=f"BIC $\\hat M$={sp['M_hat']}")
    ax[0].set(xlabel="mode m", ylabel="$\\lambda_m$ / tr", title="spectrum")
    ax[0].legend(fontsize=7)
    ax[1].hist(kf["random_samples"], bins=30, alpha=.7, label="random rank-$\\hat M$")
    ax[1].hist(kf["top2M_samples"], bins=30, alpha=.7, label="random in top-2$\\hat M$")
    ax[1].axvline(kf["Psi_hat"], color="k", lw=2, label="eigenspace (Thm 1)")
    ax[1].set(xlabel="$\\Psi(P)$", title="Theorem 1")
    ax[1].legend(fontsize=7)
    x = [r["x"] for r in cs["curve"]]
    ax[2].loglog(x, [r["op_err_mean"] for r in cs["curve"]], "o-", label=f"operator (slope {cs['slope_operator']:.2f})")
    ax[2].loglog(x, [r["subspace_err_mean"] for r in cs["curve"]], "s-",
                 label=f"subspace (slope {cs['slope_subspace']:.2f})")
    ax[2].set(xlabel="$\\sqrt{1/N' - 1/N}$", ylabel="error", title="Theorem 2")
    ax[2].legend(fontsize=7)
    if fw["eps_mod"]:
        t = np.arange(1, len(fw["eps_mod"]) + 1)
        ax[3].loglog(t, fw["eps_mod"], label=f"OMP (fit $t^{{-{fw['empirical_rate_exponent']:.2f}}}$)")
        ax[3].loglog(t, fw["bound"], "--", label="$2\\,\\mathrm{diam}_Z/\\sqrt{t+2}$")
    ax[3].set(xlabel="iteration t", ylabel="$\\varepsilon^{modal}_B$", title="Proposition fw")
    ax[3].legend(fontsize=7)
    fig.suptitle(f"boundary k={k}")
    E.savefig(fig, out / f"e1_theory_k{k}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    U = np.array(res["leading_modes"])
    tt = np.arange(U.shape[0]) / E.TOKEN_FPS
    for m in range(min(6, U.shape[1])):
        ax[0].plot(tt, U[:, m] + 0.35 * m, lw=1.2, label=f"$u_{m + 1}$ ({mi['modes'][m]['peak_freq_hz']:.2f} Hz)")
    ax[0].set(xlabel="time in window (s)", yticks=[], title="leading temporal modes")
    ax[0].legend(fontsize=6, ncol=2)
    psi = np.array(mi["circulant_eigs_sorted"])
    ax[1].semilogy(lam / lam.sum(), "o", ms=3, label="eigenvalues of $C_k$")
    ax[1].semilogy(np.where(psi > 0, psi / psi.sum(), np.nan), "-", label="circulant (power spectrum)")
    ax[1].set(xlabel="rank", title=f"Szego: TV={mi['szego_tv_distance']:.3f}, Toeplitz index={mi['toeplitz_index']:.3f}")
    ax[1].legend(fontsize=7)
    E.savefig(fig, out / f"e1_modes_k{k}")


# ----------------------------------------------------------------------------------------------
def main() -> int:
    ap = E.common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--boundaries", type=int, nargs="*", default=None, help="session boundaries k (default: all)")
    ap.add_argument("--n_random", type=int, default=200)
    ap.add_argument("--reps", type=int, default=10, help="repeats per subsample size (Theorem 2)")
    ap.add_argument("--music_dir", default=None, help="wav dir for tempo (default: manifest music_dir)")
    a = E.finish_args(ap.parse_args())
    E.set_precision(not a.bf16_attention)
    cfg = E.load_cfg(a.config)
    runs = E.runs_of(a, cfg)
    man = E.manifest(cfg, a.benchmark)
    K = len(man["sessions"])
    ks = a.boundaries or [k for k in range(1, K) if k - 1 in runs]
    E.log(f"runs: { {k: str(v) for k, v in runs.items()} } -> boundaries {ks}")
    model = E.build_model(cfg, a.device)
    D = model.model.dim
    music_dir = a.music_dir or man.get("music_dir")

    for k in ks:
        rng = np.random.default_rng(a.seed + k)
        past = list(range(k))
        pool = E.build_pool(cfg, a.benchmark, past, "train", a.T, max_per_session=a.max_windows, seed=a.seed)
        ck = runs[k - 1] / "last.ckpt"
        phi = E.phi_cached(a, model, ck, pool, "train")
        N, T, _ = phi.shape
        groups = pool.session
        ops = S.operators(phi)
        sp = spectrum_stats(ops, N, D)
        M = sp["M_hat"]
        E.log(f"k={k}: N={N} windows, M_hat={M} (conservative {sp['M_hat_conservative_Neff_N']}), "
              f"r_eff={sp['r_eff']:.2f}, kept {sp['explained_at_M_hat']:.3f}")
        kf = ky_fan(ops, M, a.n_random, rng)
        try:
            test_pool = E.build_pool(cfg, a.benchmark, past, "test", a.T, max_per_session=a.max_windows, seed=a.seed)
            phi_test = E.phi_cached(a, model, ck, test_pool, "test")
        except SystemExit:
            phi_test = None
        cons = consistency(phi, ops, M, a.reps, rng, phi_test)
        Z, _ = S.modal_dictionary(phi, "temporal", M=M, ops=ops)
        fw = fw_rate(Z, sum(S.quotas_for(groups, a.budget).values()))
        E.log(f"k={k}: FW fit t^-{fw['empirical_rate_exponent']:.2f}, max eps/bound {fw['max_ratio_to_bound']:.3f}; "
              f"Thm 2 slopes {cons['slope_operator']:.2f} / {cons['slope_subspace']:.2f}")
        modes = mode_interpretation(ops, M)
        tempo, tempo_src = tempo_of(pool.names, Path(music_dir) if music_dir else None,
                                    a.out_dir / "cache" / "tempo_bpm.json")
        decode = dict(what_modes_encode(phi, ops, M, pool, tempo, a.seed), tempo_source=tempo_src)
        bufs = buffers_report(phi, ops, Z, M, pool, groups, a.budget, a.n_random, a.seed + k)

        stab = {}
        for label, run_k in (("theta_k", k), ("theta_0", 0)):
            if run_k in runs and run_k != k - 1:
                phi_b = E.phi_cached(a, model, runs[run_k] / "last.ckpt", pool, "train")
                ob = S.operators(phi_b, channel=False)
                Mb = S.bic_rank(ob["lam"], N * D)[0]
                Ua, Ub = ops["U"][:, :M], ob["U"][:, :M]
                stab[label] = {"checkpoint": str(runs[run_k] / "last.ckpt"),
                               "subspace_overlap": float(np.linalg.norm(Ua.T @ Ub) ** 2 / M), "M_hat_there": Mb,
                               "cka_phi": cka(phi, phi_b),
                               "relative_operator_change": float(np.linalg.norm(ob["C"] - ops["C"], 2)
                                                                 / np.linalg.norm(ops["C"], 2))}

        res = {"boundary": k, "checkpoint": str(ck), "N": N, "T": T, "D": D, "pool_mode": a.pool_mode,
               "windows_per_session": {int(s): int((groups == s).sum()) for s in past},
               "spectrum": sp, "lam_c_share": ops["lam_c"][: 4 * T] / ops["lam_c"].sum(),
               "channel_M_hat": S.bic_rank(ops["lam_c"], N * T)[0],
               "ky_fan": kf, "axis": axis_comparison(ops, T, D), "consistency": cons, "fw": fw,
               "modes": modes, "leading_modes": ops["U"][:, :12], "decode": decode, "buffers": bufs,
               "stability": stab}
        E.dump_json(a.out_dir / "e1" / f"boundary_k{k}.json", res)
        figures(res, a.out_dir / "figures", k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
