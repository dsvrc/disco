#!/usr/bin/env python3
"""
Fast checks of the TMR analysis on DiscoForcing (AIST++ aistpp_genre5), on the checkpoints and buffers of the reported
screen (default v2_40000_full_s0v2). One GPU, about 30-60 min with the defaults. Writes <out>/THEORY.md (tables to
read the results from) and <out>/theory.json. Stages (all by default; --stages to pick):

  select       DiscoForcing version of Tab. rank: for each session j = 0..3, at the model right after it (S0: the shared
               S0 model; S1-S3: the TMR+KD chain) and on the session's replay candidates, with the joint trajectory TMR
               uses: M_hat under the n_eff that was used, N*D, N, N*D_eff and 90 % energy; where OMP stopped before the
               fixed-budget fill (Sec. app:algorithm), |B|/B, diam_Z, ||y||, eps_mod, the Prop. fw bound
               2 diam_Z / sqrt(B + 2) and the eps_mod(t) curve, the effective size 1/sum w^2, the overlap with the
               origin-invariant variant (Remark centering) and with the rank N_eff = N, Lemma modal
               (||P Xi|| <= sqrt(lam_1) eps_mod, ||(I-P) Xi|| <= tau), and whether re-running the buffer builder gives
               exactly the saved buffer (selection is deterministic: "selection adds no noise")
  buffers      at the shared S0 model, pool = S0 candidates: every saved S0 buffer of the screen (ER, OCS, GCR, MIR,
               CSReL, TMR, ...), TMR's ablations on the same trajectory (no whitening, random subspace, raw-feature OMP,
               channel PCA, TMR clips with uniform weights, random clips with fitted weights, herding, k-center) and
               random ER draws: eps_mod (TMR's objective), eps_B (raw trajectory mismatch), grid coverage, composition
  grads        per-clip gradients of the diffusion loss at the S0 model, averaged over R frozen noise draws and stored as
               CountSketches: for every buffer the gradient gap delta of Prop. 1 (unbiased, from two independent noise
               halves), kappa = delta / eps_B (Def. kappa), the first-order excess forgetting <g, g - g_B> against its
               Cauchy-Schwarz bound ||g|| delta, and the share of delta that a linear map of the trajectory explains
               (Thm. anatomy, J fitted by least squares on the pool)
  sensitivity  per temporal mode m, ||J_m v|| = || d gbar / d(trajectory along u_m v^T) || by central differences of the
               pool-mean gradient under a perturbation of the music embedding (the trajectory's music block): does the
               sensitivity scale as 1/sqrt(lambda_m) (the standardized class that motivates whitening) or stay flat?
  shortrun     from the S0 model, train Session 1 for --steps steps with each buffer (plus no replay and full replay of
               the whole S0 pool, theta^full of Prop. 1), ground-truth and/or distillation replay as in train_ldf.py, the
               same S1 batches and noise for every buffer; the rise of the S0 loss (pool and test) and the S1 test loss
  coverage     DiscoForcing version of Tab. coverage: grid coverage of each earlier session by every run's buffer
               (fixed ruler: the S0 model's trajectories of that session), and its Spearman correlation across runs with
               the session's SR after S4 and with its drop
  The DiscoForcing version of Tab. theory (Spearman over buffers of eps_mod, delta, eps_B and coverage with short-run
  forgetting, eps_mod with delta, the median linear share) is written at the end from whatever stages ran.

  python scripts/supp/theory_fast.py --tag v2_40000_full_s0v2 --out results/aistpp_genre5/supp/theory
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "evidence"))
import build_replay_buffer as BRB  # noqa: E402
import tmr_select as S  # noqa: E402
from datasets.aistpp import collate_fn  # noqa: E402
from utils.cl_replay import ClipDataset, clip_loader, clip_losses, clip_music_features, fixed_rng, load_model, model_batch  # noqa: E402
from utils.initialize import load_config  # noqa: E402

AIST_RE = re.compile(r"g(\w\w)_s(\w\w)_c\w+_d(\d\d)_m(\w\w)(\d)_ch(\d\d)")
NAME = {"vanilla": "Fine-tuning", "random": "ER", "random_kd": "ER+KD", "random_star": "STAR", "ocs": "OCS",
        "gcr": "GCR", "gcr_kd": "GCR+KD", "mir": "MIR", "mir_kd": "MIR+KD", "csrel": "CSReL", "csrel_kd": "CSReL+KD",
        "tmr_joint_neff": "TMR", "tmr_joint_neff_kd": "TMR+KD"}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def f3(x, nd=3):
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return "n/a"
    if x != 0 and (abs(x) < 10 ** -nd or abs(x) >= 1e5):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def base_method(v: str) -> str:
    return re.sub(r"_seed\d+$", "", v)


def spear(x, y):
    nan = lambda v: np.nan if v is None else float(v)
    x, y = np.array([nan(v) for v in x], float), np.array([nan(v) for v in y], float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"rho": None, "p": None, "n": int(ok.sum())}
    r, p = spearmanr(x[ok], y[ok])
    t, pt = kendalltau(x[ok], y[ok])
    return {"rho": float(r), "p": float(p), "tau": float(t), "p_tau": float(pt), "n": int(ok.sum())}


# ----------------------------------------------------------------------------------------------
# Runs, pools, trajectories
# ----------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, a):
        self.a = a
        self.B = a.benchmark
        self.res = Path("results") / self.B / "screen" / a.tag
        self.outp = Path("outputs") / "screen" / a.tag
        self.bufr = Path("buffers") / self.B / "screen" / a.tag
        self.out = Path(a.out)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfgs = {}
        self.man = json.loads((Path("raw_data/cl_sessions") / self.B / "manifest.json").read_text())
        self.K = len(self.man["sessions"])
        s0 = os.environ.get("S0_RUN")
        if not s0:
            p = self.res / "s0_model" / "after_s0_sess0.json"
            s0 = str(Path(json.loads(p.read_text())["ckpt"]).parent) if p.exists() else None
        self.s0_run = Path(s0) if s0 else None

    def cfg(self, j):
        if j not in self.cfgs:
            self.cfgs[j] = load_config(f"configs/cl/ldf_{self.B}_s{j}.yaml", {})
        return self.cfgs[j]

    def run_dir(self, variant, k):
        if k == 0:
            return self.s0_run
        ds = sorted(glob.glob(str(self.outp / f"*_cl_ldf_{self.B}_s{k}_{variant}")))
        return Path(ds[-1]) if ds else None

    def model(self, variant, j):
        d = self.run_dir(variant, j)
        if d is None or not (d / "last.ckpt").exists():
            return None, None
        return load_model(self.cfg(j), str(d / "last.ckpt"), self.device), d / "last.ckpt"

    def buffers(self, j):
        """{variant: buffer json} of session j in the screen."""
        out = {}
        for p in sorted(self.bufr.glob(f"*/s{j}.json")):
            out[p.parent.name] = json.loads(p.read_text())
        return out


def train_list(cfg):
    return str(cfg.data.train_meta_paths[0])


def split_list(cfg, split):
    p = train_list(cfg).replace("_train.txt", f"_{split}.txt")
    return p if Path(p).exists() else None


class Pool:
    def __init__(self, cfg, list_path, clips=None):
        self.cfg = cfg
        self.ds = ClipDataset(cfg.config, [list_path], clips=clips)
        self.clips = self.ds.clips
        self.N = len(self.clips)
        self.key = {(c["name"], int(c["start"]), int(c["length"])): i for i, c in enumerate(self.clips)}

    def indices(self, clips):
        idx = [self.key.get((c["name"], int(c["start"]), int(c["length"]))) for c in clips]
        return None if any(i is None for i in idx) else np.asarray(idx, int)


def joint_trajectory(model, pool, device):
    """The trajectory tmr_joint_neff selects on (scripts/build_replay_buffer.py): music embedding mean-pooled per latent
    frame next to the clip's motion latents, each block scaled to unit mean squared deviation."""
    music, mask, tokens = clip_music_features(model, clip_loader(pool.ds, batch_size=16), device)
    cond = BRB.condition_groups(music, mask, 0.02)
    blocks = [music, tokens]
    T = max(b.shape[1] for b in blocks)
    blocks, mask = [BRB._pad_to(b, T) for b in blocks], BRB._pad_to(mask.float(), T).bool()
    phi = BRB.balanced_blocks(blocks, mask)
    m = mask.float()
    g = blocks[0].float()
    bar = (g * m[..., None]).sum(0) / m.sum(0).clamp(min=1)[:, None]
    dev = (((g - bar) ** 2).sum(-1) * m).sum() / m.sum().clamp(min=1)
    return {"phi": phi, "mask": mask, "blocks": blocks, "cond": cond, "music_scale": float(dev.sqrt()),
            "D_music": int(music.shape[-1])}


def tmr_core(tj):
    phi, mask = tj["phi"], tj["mask"]
    N, T, D = phi.shape
    C = BRB.temporal_operator_masked(phi, mask).numpy()
    lam, U = S.eig_desc(C)
    tok = phi[mask.bool()].double()
    Cc = torch.cov(tok.T)
    d_eff = float(torch.trace(Cc) ** 2 / (Cc ** 2).sum())
    n_eff, pairs = BRB.effective_samples(Cc, N, tj["cond"], tj["blocks"][0].shape[-1])
    ranks = {"n_eff": S.bic_rank(lam, n_eff)[0], "N*D": S.bic_rank(lam, N * D)[0], "N": S.bic_rank(lam, N)[0],
             "N*D_eff": S.bic_rank(lam, N * d_eff)[0],
             "energy90": int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.9) + 1)}
    M = ranks["n_eff"]
    phin = phi.numpy()
    Z, _ = S.modal_dictionary(phin, "temporal", ops={"lam": lam, "U": U}, M=M)
    return {"lam": lam, "U": U, "C": C, "M": M, "Z": Z, "phin": phin, "ranks": ranks, "n_eff": float(n_eff),
            "same_music_pairs": float(pairs), "d_eff": d_eff, "N": N, "T": T, "D": D,
            "variance_kept": float(lam[:M].sum() / lam.sum())}


def pursue(A, budget):
    """TMR's OMP as run by tmr_joint_neff: simplex OMP with the Frank-Wolfe-gap test; if it stops early, the same
    objective at a fixed budget (stop='none'; the first picks are unchanged)."""
    idx, w, hist = S.omp_simplex(A, budget, stop="fw", trace=True)
    n_fw = len(idx)
    if n_fw < budget:
        idx, w, hist = S.omp_simplex(A, budget, stop="none", trace=True)
    return np.asarray(idx, int), np.asarray(w, float), hist, n_fw


def omp_variant(Z, budget, stop="fw"):
    """Origin-invariant variant of Remark centering: first pick argmin ||y - z_i||, Frank-Wolfe-gap filter."""
    y = Z.mean(0, dtype=np.float64)
    first = int(np.argmin(np.linalg.norm(Z.astype(np.float64) - y, axis=1)))
    sel, w = [first], np.ones(1)
    z0 = Z[first].astype(np.float64)
    G, b = np.array([[z0 @ z0]]), np.array([z0 @ y])
    u = y - z0
    used = np.zeros(len(Z), bool)
    used[first] = True
    hist = [float(np.linalg.norm(u))]
    while len(sel) < budget:
        corr = (Z @ u.astype(np.float32)).astype(np.float64)
        thr = float((y - u) @ u) if stop == "fw" else -np.inf
        ok = (corr > thr + (1e-9 * abs(thr) if np.isfinite(thr) else 0.0) + 1e-12 * float(u @ u)) & ~used
        if not ok.any():
            break
        i = int(np.argmax(np.where(ok, corr, -np.inf)))
        zi = Z[i].astype(np.float64)
        gi = Z[sel].astype(np.float64) @ zi
        G = np.block([[G, gi[:, None]], [gi[None, :], np.array([[zi @ zi]])]])
        b = np.append(b, zi @ y)
        sel.append(i)
        used[i] = True
        w = S.simplex_lsq(G, b, 0.0, np.append(w, 0.0))
        u = y - Z[sel].astype(np.float64).T @ w
        hist.append(float(np.linalg.norm(u)))
    return np.asarray(sel, int), w, hist


def gram_diam(Z, device):
    X = torch.from_numpy(np.ascontiguousarray(Z)).to(device, torch.float64)
    sq = (X * X).sum(1)
    D2 = (sq[:, None] + sq[None, :] - 2 * X @ X.T).clamp(min=0)
    return float(D2.max().sqrt())


def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    G = Xc @ Xc.T
    ev, V = np.linalg.eigh(G)
    return V[:, -2:][:, ::-1] * np.sqrt(np.maximum(ev[-2:][::-1], 0))


def eps_of(A, idx, w, y=None):
    y = A.mean(0, dtype=np.float64) if y is None else y
    return float(np.linalg.norm(y - np.asarray(w, np.float64) @ A[np.asarray(idx, int)].astype(np.float64)))


def simplex(w):
    w = np.asarray(w, float)
    return w / w.sum()


# ----------------------------------------------------------------------------------------------
# Stage select (Tab. rank for DiscoForcing)
# ----------------------------------------------------------------------------------------------
def stage_select(c: Ctx) -> dict:
    rows = []
    for vi_, variant in enumerate(c.a.select_variants):
        for j in range(c.K - 1):
            if j == 0 and vi_ > 0:
                continue  # S0 is shared by every chain
            model, ck = (c.model("s0", 0) if j == 0 else c.model(variant, j))
            if model is None:
                log(f"[select] {variant} S{j}: no model, skipped")
                continue
            cfg = c.cfg(j)
            pool = Pool(cfg, train_list(cfg))
            N = pool.N
            budget = max(1, round(c.a.ratio * N))
            t0 = time.time()
            tj = joint_trajectory(model, pool, c.device)
            t_feat = time.time() - t0
            t1 = time.time()
            core = tmr_core(tj)
            Z, M, lam = core["Z"], core["M"], core["lam"]
            idx, w, hist, n_fw = pursue(Z, budget)
            t_sel = time.time() - t1
            y = Z.mean(0, dtype=np.float64)
            diam = gram_diam(Z, c.device)
            eps_mod = eps_of(Z, idx, w, y)
            bound = 2 * diam / math.sqrt(len(idx) + 2)
            ts = np.arange(1, len(hist) + 1)
            lo = min(2, len(hist) - 1)
            slope = float(np.polyfit(np.log(ts[lo:n_fw]), np.log(np.maximum(hist[lo:n_fw], 1e-30)), 1)[0]) \
                if n_fw - lo >= 3 else None
            curve = [{"t": int(t), "eps_mod": float(e), "bound": float(2 * diam / math.sqrt(t + 2))}
                     for t, e in zip(ts, hist) if t in {1, 2, 3, 5, 10, 20, 50, 100, 200, 500} or t == len(hist)]
            # origin-invariant variant, and the rank with N_eff = N
            vi, vw, vh = omp_variant(Z, budget)
            if len(vi) < budget:
                vi, vw, vh = omp_variant(Z, budget, stop="none")
            overlap_var = len(set(idx.tolist()) & set(vi.tolist())) / max(len(idx), 1)
            rankN = core["ranks"]["N"]
            alt = {}
            if rankN != M:
                ZN, _ = S.modal_dictionary(core["phin"], "temporal", ops={"lam": lam, "U": core["U"]}, M=rankN)
                ni, nw, _, n_fwN = pursue(ZN, budget)
                alt = {"M": rankN, "overlap": len(set(idx.tolist()) & set(ni.tolist())) / max(len(idx), 1),
                       "eps_mod_in_tmr_coords": eps_of(Z, ni, nw, y), "omp_stop": n_fwN}
            # Lemma modal on the padded balanced trajectories
            phin = core["phin"]
            Xr = phin.reshape(N, -1).astype(np.float64)
            mean = Xr.mean(0)
            Xi = (mean - simplex(w) @ Xr[idx]).reshape(core["T"], core["D"])
            Um = core["U"][:, :M]
            PXi = Um @ (Um.T @ Xi)
            tau = 0.0
            for s0 in range(0, N, 128):  # tau = max_i ||(I - P) phi~_i||_F, in chunks of clips
                tilde = phin[s0:s0 + 128].astype(np.float64) - mean.reshape(core["T"], core["D"])
                resid = tilde - np.einsum("tm,nmd->ntd", Um, np.einsum("tm,ntd->nmd", Um, tilde))
                tau = max(tau, float(np.sqrt((resid ** 2).sum((1, 2))).max()))
            lemma = {"eps_B": float(np.linalg.norm(Xi)), "P_Xi": float(np.linalg.norm(PXi)),
                     "sqrt_lam1_eps_mod": float(math.sqrt(lam[0]) * eps_mod), "out_Xi": float(np.linalg.norm(Xi - PXi)),
                     "tau": tau}
            # does the builder reproduce the saved buffer exactly?
            repro = {}
            saved = c.buffers(j).get(variant) or c.buffers(j).get(base_method(variant))
            dance_id = {n: i for i, n in enumerate(sorted({cl["name"] for cl in pool.clips}))}
            bidx, bw, _ = BRB.select_tmr_dance(tj["blocks"], tj["mask"], [dance_id[cl["name"]] for cl in pool.clips],
                                               budget=budget, rng=np.random.default_rng(j), device=c.device, seed=0,
                                               per_dance=0, rank="bic_neff", energy=0.9, weights="modal", stop="fw",
                                               label="tmr_joint_neff", moments=1, cond_groups=tj["cond"], fill="pursuit")
            repro["same_as_internal"] = sorted(map(int, bidx)) == sorted(idx.tolist())
            if saved is not None:
                sidx = pool.indices(saved["clips"])
                if sidx is None:
                    repro["saved"] = "clips not among the candidates"
                else:
                    sw = np.array([float(cl.get("weight", 1.0)) for cl in saved["clips"]])
                    mine = dict(zip(map(int, bidx), map(float, bw)))
                    repro["same_clips_as_saved"] = sorted(sidx.tolist()) == sorted(mine)
                    if repro["same_clips_as_saved"]:
                        repro["max_weight_diff"] = float(max(abs(mine[int(i)] - x) for i, x in zip(sidx, sw)))
                    repro["overlap_with_saved"] = len(set(sidx.tolist()) & set(mine)) / max(len(sidx), 1)
                    repro["saved_ckpt"] = saved.get("ckpt")
            row = {"variant": variant, "session": j, "ckpt": str(ck), "N": N, "budget": budget, "T": core["T"],
                   "D": core["D"], "ranks": core["ranks"], "M": M, "n_eff": core["n_eff"],
                   "same_music_pairs": core["same_music_pairs"], "d_eff": core["d_eff"],
                   "variance_kept": core["variance_kept"], "lam_top": lam[: min(len(lam), 12)].tolist(),
                   "clips": int(len(idx)), "omp_stopped_at": int(n_fw), "stopped_early": bool(n_fw < budget),
                   "diam_Z": diam, "norm_y": float(np.linalg.norm(y)), "eps_mod": eps_mod, "eps_mod_rel": eps_mod / max(np.linalg.norm(y), 1e-30),
                   "eps_mod_at_stop": float(hist[n_fw - 1]) if n_fw else None, "fw_bound": bound,
                   "fw_bound_at_stop": 2 * diam / math.sqrt(n_fw + 2) if n_fw else None,
                   "rate_slope_loglog": slope, "curve": curve, "effective_size": float(1.0 / (simplex(w) ** 2).sum()),
                   "overlap_variant": overlap_var, "variant_eps_mod": eps_of(Z, vi, vw, y),
                   "rank_N_alternative": alt, "lemma_modal": lemma, "reproduce": repro,
                   "seconds_features": t_feat, "seconds_selection": t_sel}
            rows.append(row)
            log(f"[select] {variant} S{j}: N={N} B={budget} M={M} ranks {core['ranks']} OMP stop {n_fw}/{budget} "
                f"eps_mod {eps_mod:.4g} (bound {bound:.4g}) overlap variant {overlap_var:.2f} repro {repro}")
            del model
            torch.cuda.empty_cache()
    return {"rows": rows}


# ----------------------------------------------------------------------------------------------
# Stage buffers (all S0 buffers at the S0 model)
# ----------------------------------------------------------------------------------------------
def s0_setup(c: Ctx, st: dict):
    if "pool0" in st:
        return
    model, ck = c.model("s0", 0)
    if model is None:
        raise RuntimeError("no S0 model (set S0_RUN)")
    cfg0 = c.cfg(0)
    pool = Pool(cfg0, train_list(cfg0))
    tj = joint_trajectory(model, pool, c.device)
    core = tmr_core(tj)
    st.update(model0=model, ck0=ck, pool0=pool, tj0=tj, core0=core, budget0=max(1, round(c.a.ratio * pool.N)))


def stage_buffers(c: Ctx, st: dict) -> dict:
    s0_setup(c, st)
    pool, core, B = st["pool0"], st["core0"], st["budget0"]
    N, Z, phin, M = pool.N, core["Z"], core["phin"], core["M"]
    y = Z.mean(0, dtype=np.float64)
    Xr = phin.reshape(N, -1)
    rng = np.random.default_rng(c.a.seed)
    bufs = []  # {name, kind, members, idx, w (simplex)}
    seen = {}

    def add(name, kind, idx, w, members=None):
        idx = np.asarray(idx, int)
        w = simplex(w if w is not None else np.ones(len(idx)))
        key = (tuple(sorted(idx.tolist())), tuple(np.round(w[np.argsort(idx)], 6)))
        if key in seen:
            seen[key]["members"] += members or [name]
            return
        if any(x["name"] == name for x in bufs):  # same display name, different buffer: keep both apart
            name = f"{name} [{(members or [name])[0]}]"
        b = {"name": name, "kind": kind, "members": list(members or [name]), "idx": idx, "w": w}
        seen[key] = b
        bufs.append(b)

    for v, b in sorted(c.buffers(0).items()):
        idx = pool.indices(b["clips"])
        if idx is None:
            log(f"[buffers] {v}: its clips are not among the S0 candidates, skipped")
            continue
        add(NAME.get(base_method(v), v), "saved", idx, [float(cl.get("weight", 1.0)) for cl in b["clips"]], [v])
    t_idx, t_w, _, _ = pursue(Z, B)
    add("TMR (recomputed)", "tmr", t_idx, t_w)
    ops = {"lam": core["lam"], "U": core["U"], "C": core["C"]}
    gen = {}
    Zn, _ = S.modal_dictionary(phin, "temporal", whiten=False, ops=ops, M=M)
    gen["No whitening"] = pursue(Zn, B)[:2]
    Zr, _ = S.modal_dictionary(phin, "temporal", basis="random", rng=np.random.default_rng(c.a.seed), ops=ops, M=M)
    gen["Random subspace"] = pursue(Zr, B)[:2]
    del Zr
    gen["Raw-feature OMP"] = pursue(Xr, B)[:2]
    mean = phin.mean(0, dtype=np.float64)
    Cc = S.channel_operator(phin, mean)
    lam_c, U_c = S.eig_desc(Cc)
    Zc, ci = S.modal_dictionary(phin, "channel", ops={"lam_c": lam_c, "U_c": U_c, "C_c": Cc})
    gen["Channel PCA"] = pursue(Zc, B)[:2]
    del Zc
    gen["TMR clips, uniform weights"] = (t_idx, np.ones(len(t_idx)))
    groups, quotas = np.zeros(N, int), {0: B}
    gen["Herding"] = (S.herding(Xr, groups, quotas), None)
    gen["k-center"] = (S.kcenter(Xr, groups, quotas), None)
    for name, (idx, w) in gen.items():
        add(name, "ablation", idx, w)
    for r in range(c.a.n_random):
        idx = np.sort(np.random.default_rng(1000 + r).choice(N, B, replace=False))
        if r == 0:
            Gs = Z[idx].astype(np.float64)
            add("Random clips, fitted weights", "ablation", idx, S.simplex_lsq(Gs @ Gs.T, Gs @ y))
        add(f"ER draw {r}", "random", idx, None)
    X2 = pca2(Xr.astype(np.float64))
    meta = [AIST_RE.match(cl["name"]) for cl in pool.clips]
    for b in bufs:
        idx, w = b["idx"], b["w"]
        b["eps_mod"] = eps_of(Z, idx, w, y)
        b["eps_mod_unwhitened"] = eps_of(Zn, idx, w)
        b["eps_B"] = eps_of(Xr, idx, w)
        b["coverage"] = S.grid_coverage(X2, idx, None, bins=c.a.bins)
        b["effective_size"] = float(1.0 / (w ** 2).sum())
        b["size"] = int(len(idx))
        names = [pool.clips[i]["name"] for i in idx]
        b["dances"] = len(set(names))
        b["music_pieces"] = len({m.group(4) + m.group(5) for m in (meta[i] for i in idx) if m})
        b["dancers"] = len({m.group(3) for m in (meta[i] for i in idx) if m})
    st["bufs"] = bufs
    st["Zn0"] = Zn
    return {"N": N, "budget": B, "M": M, "ranks": core["ranks"], "M_channel": ci["M"], "pool_dances": len({cl["name"] for cl in pool.clips}),
            "buffers": [{k: v for k, v in b.items() if k not in ("idx", "w")} | {"idx": b["idx"].tolist(), "w": b["w"].tolist()}
                        for b in bufs]}


# ----------------------------------------------------------------------------------------------
# Stage grads (delta of Prop. 1, kappa, linear share)
# ----------------------------------------------------------------------------------------------
class CountSketch:
    def __init__(self, numel, m, seed, device):
        g = torch.Generator().manual_seed(seed)
        self.m = m
        self.bucket = torch.randint(0, m, (numel,), generator=g).to(device)
        self.sign = (torch.randint(0, 2, (numel,), generator=g, dtype=torch.int8) * 2 - 1).to(device)

    def __call__(self, v):
        return torch.zeros(self.m, device=v.device, dtype=torch.float64).index_add_(0, self.bucket, (v * self.sign).double())


def repeat_batch(mb, h):
    out = {}
    for k, v in mb.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
            out[k] = v.repeat_interleave(h, dim=0)
        elif isinstance(v, list) and len(v) == 1:
            out[k] = v * h
        else:
            out[k] = v
    return out


def flat_grad(loss, params):
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


def stage_grads(c: Ctx, st: dict) -> dict:
    s0_setup(c, st)
    if "bufs" not in st:
        stage_buffers(c, st)
    model, pool, core = st["model0"], st["pool0"], st["core0"]
    params = [p for p in model.parameters() if p.requires_grad]
    numel = sum(p.numel() for p in params)
    cs = CountSketch(numel, c.a.sketch, 12345, c.device)
    N, h = pool.N, max(1, c.a.R // 2)
    GA, GB = np.zeros((N, c.a.sketch), np.float32), np.zeros((N, c.a.sketch), np.float32)
    LA = np.zeros(N)
    model.eval()
    loader = clip_loader(pool.ds, batch_size=1)
    t0 = time.time()
    for i, batch in enumerate(loader):
        mb = repeat_batch(model_batch(batch, c.device), h)
        for half, G in ((0, GA), (1, GB)):
            with fixed_rng(7919 * (i + 1) + half):
                loss = model(mb, per_sample=True)["per_sample"].float().mean()
            G[i] = cs(flat_grad(loss, params)).float().cpu().numpy()
            if half == 0:
                LA[i] = float(loss)
        if i % 100 == 0:
            log(f"[grads] {i + 1}/{N} clips ({time.time() - t0:.0f} s)")
    model.zero_grad(set_to_none=True)
    gA, gB = GA.astype(np.float64).mean(0), GB.astype(np.float64).mean(0)
    gnorm = math.sqrt(max(float(gA @ gB), 0.0))
    # linear model of the gradient in the trajectory: PCA scores of the (whitened) atoms, least squares per half
    Z = core["Z"].astype(np.float64)
    Zc = Z - Z.mean(0)
    ev, V = np.linalg.eigh(Zc @ Zc.T)
    r = int(min(c.a.lin_rank, max(2, N // 6)))
    Sc = V[:, ::-1][:, :r] * np.sqrt(np.maximum(ev[::-1][:r], 0))
    GcA, GcB = GA.astype(np.float64) - gA, GB.astype(np.float64) - gB
    betaA = np.linalg.lstsq(Sc, GcA, rcond=None)[0]
    betaB = np.linalg.lstsq(Sc, GcB, rcond=None)[0]
    # how much of the per-clip gradient variation is linear in the trajectory (5-fold CV), and the noise ceiling
    folds = np.random.default_rng(0).permutation(N) % 5
    Gm = (GcA + GcB) / 2
    sse = sst = 0.0
    for f in range(5):
        tr, te = folds != f, folds == f
        bm = np.linalg.lstsq(Sc[tr], Gm[tr], rcond=None)[0]
        pred = Sc[te] @ bm
        sse += float(((Gm[te] - pred) ** 2).sum())
        sst += float(((Gm[te] - Gm[tr].mean(0)) ** 2).sum())
    signal = float((GcA * GcB).sum())
    total = float((Gm ** 2).sum())
    out_b = []
    for b in st["bufs"]:
        idx, w = b["idx"], b["w"]
        dA, dB = gA - w @ GA[idx].astype(np.float64), gB - w @ GB[idx].astype(np.float64)
        d2 = float(dA @ dB)
        delta = math.sqrt(max(d2, 0.0))
        xi = -(w @ Sc[idx])
        jA, jB = xi @ betaA, xi @ betaB
        lin = math.sqrt(max(float(jA @ jB), 0.0))
        first = 0.5 * float(gA @ dB + gB @ dA)
        b.update(delta=delta, delta_plain=float(np.linalg.norm((dA + dB) / 2)), delta_mc=float(np.linalg.norm(dA - dB) / 2),
                 kappa=delta / max(b["eps_B"], 1e-30), linear_part=lin, linear_share=lin / max(delta, 1e-30),
                 first_order_excess=first, cs_bound=gnorm * delta, cs_tightness=first / max(gnorm * delta, 1e-30))
        out_b.append({k: b[k] for k in ("name", "kind", "members", "delta", "delta_plain", "delta_mc", "kappa",
                                        "linear_part", "linear_share", "first_order_excess", "cs_bound", "cs_tightness",
                                        "eps_mod", "eps_B")})
    st["grads_done"] = True
    return {"R": c.a.R, "sketch": c.a.sketch, "params": numel, "norm_gbar": gnorm, "seconds": time.time() - t0,
            "lin_rank": r, "linear_cv_r2": 1 - sse / max(sst, 1e-30), "signal_fraction_of_clip_gradients": signal / max(total, 1e-30),
            "mean_clip_loss": float(LA.mean()), "buffers": out_b}


# ----------------------------------------------------------------------------------------------
# Stage sensitivity (||J_m v|| per temporal mode)
# ----------------------------------------------------------------------------------------------
def stage_sensitivity(c: Ctx, st: dict) -> dict:
    s0_setup(c, st)
    model, pool, tj, core = st["model0"], st["pool0"], st["tj0"], st["core0"]
    T, M, lam, U = core["T"], core["M"], core["lam"], core["U"]
    full = [i for i, cl in enumerate(pool.clips) if int(cl["length"]) == T]
    full = sorted(np.random.default_rng(0).permutation(full)[: c.a.sens_clips].tolist())
    if len(full) < 4:
        return {"skipped": f"only {len(full)} full-length clips"}
    loader = DataLoader(Subset(pool.ds, full), batch_size=8, shuffle=False, collate_fn=collate_fn)
    batches = [model_batch(b, c.device) for b in loader]
    params = [p for p in model.parameters() if p.requires_grad]
    state = {"delta": None}
    emb = model.model.music_embedding

    def hook(mod, inp, out):
        d = state["delta"]
        if d is None:
            return None
        if out.shape[0] != d.shape[0]:
            raise RuntimeError(f"music context rows {out.shape[0]} != batch x frames {d.shape[0]}")
        return out + d.to(out.dtype)

    handle = emb.register_forward_hook(hook)
    model.eval()

    def mean_grad(direction):  # direction: (T, D_music) raw-unit perturbation of the trajectory, or None
        acc, n = None, 0
        for bi, mb in enumerate(batches):
            b = mb["token"].shape[0]
            state["delta"] = None if direction is None else direction[None].expand(b, -1, -1).reshape(b * T, 1, -1)
            for r in range(c.a.sens_R):
                with fixed_rng(4241 + 31 * bi + r):
                    loss = model(mb, per_sample=True)["per_sample"].float().mean()
                g = flat_grad(loss, params).double()
                acc = g if acc is None else acc + g
                n += 1
        state["delta"] = None
        return acc / n

    try:
        g0 = mean_grad(None)
        g0b = mean_grad(None)
        floor = float((g0 - g0b).norm())
        modes = sorted({int(x) for x in np.unique(np.round(np.geomspace(1, M, min(c.a.sens_modes, M))))}
                       | {m for m in (M + 1, min(T, 2 * M + 1), T) if m <= T and lam[m - 1] > 1e-12 * lam[0]})
        vgen = np.random.default_rng(1)
        vs = []
        for _ in range(c.a.sens_v):
            v = vgen.standard_normal(tj["D_music"])
            vs.append(v / np.linalg.norm(v))
        rows = []
        scale = tj["music_scale"]
        for m in modes:
            u = torch.from_numpy(U[:, m - 1].copy()).to(c.device, torch.float32)
            eps = c.a.sens_eps * math.sqrt(max(lam[m - 1], 1e-30))  # in balanced units: a fraction of the mode's sd
            resp = []
            for v in vs:
                d = (eps * scale) * u[:, None] * torch.from_numpy(v).to(c.device, torch.float32)[None, :]
                gp, gm = mean_grad(d), mean_grad(-d)
                diff = float((gp - gm).norm())
                resp.append({"J_m_v": diff / (2 * eps), "diff": diff, "above_floor": diff / max(floor, 1e-30)})
            r_m = float(np.mean([x["J_m_v"] for x in resp]))
            rows.append({"mode": m, "retained": m <= M, "lambda": float(lam[m - 1]), "eps": eps, "J_m_v": r_m,
                         "standardized": math.sqrt(lam[m - 1]) * r_m,
                         "diff_over_floor_min": float(min(x["above_floor"] for x in resp)), "per_v": resp})
            log(f"[sensitivity] mode {m} (lambda {lam[m - 1]:.3g}): ||J_m v|| {r_m:.4g}, "
                f"standardized {math.sqrt(lam[m - 1]) * r_m:.4g}, signal/floor {rows[-1]['diff_over_floor_min']:.1f}")
    finally:
        handle.remove()
    kept = [r for r in rows if r["retained"] and r["diff_over_floor_min"] > 3]
    slope = float(np.polyfit(np.log([r["lambda"] for r in kept]), np.log([r["J_m_v"] for r in kept]), 1)[0]) \
        if len(kept) >= 3 else None
    model.zero_grad(set_to_none=True)
    return {"clips": len(full), "R": c.a.sens_R, "M": M, "eps_fraction_of_sd": c.a.sens_eps, "music_scale": scale,
            "norm_gbar_probe": float(g0.norm()), "nondeterminism_floor": floor, "rows": rows,
            "slope_log_J_vs_log_lambda_retained": slope,
            "reading": "slope -0.5: ||J_m|| ~ 1/sqrt(lambda_m), the standardized class (whitening is the right units); "
                       "slope 0: flat sensitivity (raw units)"}


# ----------------------------------------------------------------------------------------------
# Stage shortrun (short-run forgetting of every buffer)
# ----------------------------------------------------------------------------------------------
def stage_shortrun(c: Ctx, st: dict) -> dict:
    s0_setup(c, st)
    if "bufs" not in st:
        stage_buffers(c, st)
    model, pool, B = st["model0"], st["pool0"], st["budget0"]
    dev = c.device
    cfg0, cfg1 = c.cfg(0), c.cfg(1)
    Tm = int(cfg1.data.train_max_tokens)
    bs = int(cfg1.data.train_bs)
    rcfg = cfg1.replay
    bs_rep = int(rcfg.get("batch_size") or bs)
    lam_rep = float(rcfg.get("weight", 1.0))
    gt_w, kd_w = float(rcfg.distill.gt_weight), float(rcfg.distill.kd_weight)
    op = cfg1.optimizer.params
    steps, evals = c.a.steps, sorted({int(x) for x in c.a.eval_at.split(",") if int(x) <= c.a.steps} | {c.a.steps})

    # the same S1 windows (random 76-token crops, epochs over the sequences) for every run
    s1_all = ClipDataset(cfg1.config, [train_list(cfg1)])
    lengths = {n: int(d["token_length"]) for n, d in s1_all.by_name.items()}
    rng = np.random.default_rng(c.a.seed)
    names, wins = sorted(lengths), []
    while len(wins) < steps * bs:
        for n in rng.permutation(names):
            L = lengths[n]
            s = int(rng.integers(0, L - Tm + 1)) if L > Tm else 0
            wins.append({"name": str(n), "start": s, "length": min(L, Tm)})
    cur_ds = ClipDataset(cfg1.config, [train_list(cfg1)], clips=wins[: steps * bs])

    def eval_set(cfg, split, cap, seed):
        lp = train_list(cfg) if split == "train" else split_list(cfg, split)
        if lp is None:
            return None
        ds = ClipDataset(cfg.config, [lp], drop_missing=True)
        idx = np.sort(np.random.default_rng(seed).permutation(len(ds))[:cap])
        return clip_loader(Subset(ds, idx.tolist()), batch_size=16)

    sets = {"S0_pool": clip_loader(Subset(pool.ds, np.sort(np.random.default_rng(1).permutation(pool.N)[: c.a.eval_pool]).tolist()),
                                   batch_size=16),
            "S0_test": eval_set(cfg0, "test", c.a.eval_test, 2), "S1_test": eval_set(cfg1, "test", c.a.eval_test, 3)}
    sets = {k: v for k, v in sets.items() if v is not None}
    seeds = tuple(range(4))
    theta0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base = {k: float(clip_losses(model, v, dev, seeds).mean()) for k, v in sets.items()}
    log(f"[shortrun] S0 model losses {base}; {steps} steps, eval at {evals}")
    teacher = None
    if "kd" in c.a.replay:
        teacher, _ = c.model("s0", 0)
        teacher.requires_grad_(False)

    def rep_loader(idx, w, seed):
        clips = [dict(pool.clips[i], weight=float(x * len(idx))) for i, x in zip(idx, w)]
        ds = ClipDataset(cfg0.config, [train_list(cfg0)], clips=clips)
        r = np.random.default_rng(seed)
        need = steps * min(bs_rep, len(clips))
        order = []
        while len(order) < need:
            order += r.permutation(len(clips)).tolist()
        return DataLoader(Subset(ds, order[:need]), batch_size=min(bs_rep, len(clips)), shuffle=False,
                          collate_fn=collate_fn, num_workers=c.a.workers, drop_last=True)

    def run(name, idx, w, mode, seed):
        model.load_state_dict(theta0)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=float(op.lr), betas=tuple(op.betas), eps=float(op.eps),
                                weight_decay=float(op.weight_decay))
        cur = DataLoader(cur_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=c.a.workers, drop_last=True)
        rep = iter(rep_loader(idx, w, 77 + seed)) if idx is not None else None
        out, t0 = {"name": name, "mode": mode, "seed": seed, "evals": {}}, time.time()
        for step, cb in enumerate(cur):
            cmb = model_batch(cb, dev)
            with fixed_rng(100003 * (seed + 1) + step):
                total = model(cmb)["total"]
            if rep is not None:
                rb = model_batch(next(rep), dev)
                wt = torch.as_tensor(rb["weight"], dtype=torch.float32, device=dev)
                sd = 200003 * (seed + 1) + step
                if mode == "kd":
                    teacher.train(model.training)
                    with torch.no_grad(), fixed_rng(sd):
                        ref = [p.detach() for p in teacher(rb, return_prediction=True)["prediction"]]
                    with fixed_rng(sd):
                        o = model(rb, per_sample=True, extra_targets=ref)
                    term = gt_w * (wt * o["per_sample"]).mean() + kd_w * (wt * o["per_sample_extra"]).mean()
                else:
                    with fixed_rng(sd):
                        term = (wt * model(rb, per_sample=True)["per_sample"]).mean()
                total = total + lam_rep * term
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            if step + 1 in evals:
                out["evals"][step + 1] = {k: float(clip_losses(model, v, dev, seeds).mean()) - base[k] for k, v in sets.items()}
                model.train()
        out["seconds"] = time.time() - t0
        return out

    jobs = [("No replay", None, None)]
    jobs.append(("Full replay", np.arange(pool.N), np.full(pool.N, 1.0 / pool.N)))
    rand_names = {f"ER draw {r}" for r in range(c.a.train_random)}
    for b in st["bufs"]:
        if b["kind"] in ("saved", "ablation") or b["name"] in rand_names:
            if b["name"] == "TMR (recomputed)":
                continue
            jobs.append((b["name"], b["idx"], b["w"]))
    if not any(b["kind"] == "saved" for b in st["bufs"]):
        tb = next(b for b in st["bufs"] if b["name"] == "TMR (recomputed)")
        jobs.append((tb["name"], tb["idx"], tb["w"]))
    results = []
    try:
        for mode in c.a.replay:
            for name, idx, w in jobs:
                if mode == "kd" and name == "No replay":
                    continue  # identical to the ground-truth run
                try:
                    r = run(name, idx, w, mode, 0)
                except Exception:  # noqa: BLE001
                    log(f"[shortrun] {mode} {name} FAILED:\n{traceback.format_exc()}")
                    continue
                results.append(r)
                last = r["evals"][steps]
                log(f"[shortrun] {mode} {name}: dL S0 pool {last.get('S0_pool', float('nan')):+.5f}, S0 test "
                    f"{last.get('S0_test', float('nan')):+.5f}, S1 test {last.get('S1_test', float('nan')):+.5f} "
                    f"({r['seconds']:.0f} s)")
            for name in [n for n in ("TMR", "ER draw 0") if any(j[0] == n for j in jobs)]:
                _, idx, w = next(j for j in jobs if j[0] == name)
                for sd in range(1, c.a.repeat + 1):
                    r = run(name, idx, w, mode, sd)
                    r["repeat"] = True
                    results.append(r)
    finally:
        model.load_state_dict(theta0)  # later stages use the S0 model
        model.eval()
    del teacher
    return {"steps": steps, "evals": evals, "base_losses": base, "batch": bs, "replay_batch": bs_rep, "lr": float(op.lr),
            "eval_clips": {k: len(v.dataset) for k, v in sets.items()}, "runs": results}


# ----------------------------------------------------------------------------------------------
# Stage coverage (Tab. coverage for DiscoForcing)
# ----------------------------------------------------------------------------------------------
def stage_coverage(c: Ctx, st: dict) -> dict:
    s0_setup(c, st)
    model = st["model0"]

    def sr(v, t, i):
        p = (c.res / "s0_model" / f"after_s0_sess{i}.json") if t == 0 else (c.res / v / f"after_s{t}_sess{i}.json")
        try:
            return json.loads(p.read_text()).get("StyleRetention")
        except Exception:  # noqa: BLE001
            return None

    rows, per_session = [], {}
    for j in range(c.K - 1):
        cfg = c.cfg(j)
        pool = st["pool0"] if j == 0 else Pool(cfg, train_list(cfg))
        tj = st["tj0"] if j == 0 else joint_trajectory(model, pool, c.device)
        X2 = pca2(tj["phi"].numpy().reshape(pool.N, -1).astype(np.float64))
        for v, b in sorted(c.buffers(j).items()):
            idx = pool.indices(b["clips"])
            if idx is None:
                continue
            cov = S.grid_coverage(X2, idx, None, bins=c.a.bins)
            after = sr(v, c.K - 1, j)
            right = sr(v, j, j)
            rows.append({"variant": v, "method": base_method(v), "session": j, "coverage": cov, "SR_after_last": after,
                         "SR_right_after": right, "drop": (right - after) if (right is not None and after is not None) else None})
        sub = [r for r in rows if r["session"] == j]
        per_session[j] = {"vs_SR_after_last": spear([r["coverage"] for r in sub], [r["SR_after_last"] for r in sub]),
                          "vs_drop": spear([r["coverage"] for r in sub], [r["drop"] for r in sub])}
        log(f"[coverage] S{j}: {len(sub)} buffers, rho(coverage, SR after S{c.K - 1}) {per_session[j]['vs_SR_after_last']}")
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], {}).setdefault(r["session"], []).append(r["coverage"])
    table = {m: {f"S{j}": float(np.mean(v)) for j, v in d.items()} for m, d in by_method.items()}
    return {"bins": c.a.bins, "rows": rows, "per_session": per_session, "table": table,
            "ruler": "joint trajectories of the shared S0 model for every session (one fixed representation)"}


# ----------------------------------------------------------------------------------------------
# Correlations (Tab. theory for DiscoForcing) and THEORY.md
# ----------------------------------------------------------------------------------------------
def correlations(c: Ctx, R: dict) -> dict:
    out = {}
    bl = {b["name"]: b for b in (R.get("buffers") or {}).get("buffers", [])}
    gl = {b["name"]: b for b in (R.get("grads") or {}).get("buffers", [])}
    for b in bl.values():
        b.update({k: v for k, v in gl.get(b["name"], {}).items() if k not in b})
    sr_runs = (R.get("shortrun") or {}).get("runs", [])
    if gl:
        names = [n for n in gl if n in bl]
        out["eps_mod_vs_delta"] = spear([bl[n]["eps_mod"] for n in names], [gl[n]["delta"] for n in names])
        out["eps_B_vs_delta"] = spear([bl[n]["eps_B"] for n in names], [gl[n]["delta"] for n in names])
        out["eps_mod_unwhitened_vs_delta"] = spear([bl[n]["eps_mod_unwhitened"] for n in names], [gl[n]["delta"] for n in names])
        out["coverage_vs_delta"] = spear([bl[n]["coverage"] for n in names], [gl[n]["delta"] for n in names])
        out["median_linear_share"] = float(np.median([gl[n]["linear_share"] for n in names]))
        out["kappa_cv"] = float(np.std([gl[n]["kappa"] for n in names]) / max(np.mean([gl[n]["kappa"] for n in names]), 1e-30))
    if sr_runs:
        steps = R["shortrun"]["steps"]
        for mode in sorted({r["mode"] for r in sr_runs}):
            runs = [r for r in sr_runs if r["mode"] == mode and not r.get("repeat") and r["name"] in bl]
            full = next((r for r in sr_runs if r["mode"] == mode and r["name"] == "Full replay"), None)
            for key in ("S0_pool", "S0_test"):
                y = [r["evals"][steps].get(key) for r in runs]
                o = {"n": len(runs)}
                for q in ("eps_mod", "eps_B", "eps_mod_unwhitened", "coverage", "delta"):
                    o[q] = spear([bl[r["name"]].get(q) for r in runs], y)
                out[f"shortrun_{mode}_{key}"] = o
            rep = [r for r in sr_runs if r["mode"] == mode and (r.get("repeat") or (r["name"] in ("TMR", "ER draw 0") and r["seed"] == 0))]
            spread = {}
            for n in {r["name"] for r in rep}:
                v = [r["evals"][steps]["S0_pool"] for r in rep if r["name"] == n]
                if len(v) > 1:
                    spread[n] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n": len(v)}
            allv = [r["evals"][steps]["S0_pool"] for r in runs]
            out[f"shortrun_{mode}_noise"] = {"repeat_spread": spread, "between_buffer_sd": float(np.std(allv)) if allv else None,
                                             "full_replay": full["evals"][steps] if full else None}
            # long-run link: S0's SR after S1 (40k steps) of the screen runs that replayed these buffers
            pts = []
            for r in runs:
                mem = [m for m in bl[r["name"]].get("members", []) if not base_method(m).endswith("star")
                       and (base_method(m).endswith("_kd") == (mode == "kd"))]
                vals = []
                for m in mem:
                    p = c.res / m / "after_s1_sess0.json"
                    if p.exists():
                        vals.append(json.loads(p.read_text()).get("StyleRetention"))
                if vals:
                    pts.append((r["evals"][steps]["S0_pool"], float(np.mean(vals)), r["name"]))
            out[f"shortrun_{mode}_vs_longrun_SR"] = dict(spear([p[0] for p in pts], [p[1] for p in pts]),
                                                         points=[{"buffer": p[2], "dL": p[0], "SR_S0_after_S1": p[1]} for p in pts])
    return out


def write_md(c: Ctx, R: dict) -> Path:
    L = ["# Fast theory checks on DiscoForcing", "", f"Screen `{c.a.tag}`, S0 model `{c.s0_run}`. theory.json has every number.", ""]
    sel = (R.get("select") or {}).get("rows", [])
    if sel:
        L += ["## Tab. rank for DiscoForcing (selection at the model right after each session)", "",
              "| Chain | S | N | B | M (n_eff, used) | M (N*D) | M (N) | M (N*D_eff) | 90% energy | OMP stop / B | diam_Z | ‖y‖ | eps_mod | 2 diam/sqrt(B+2) | 1/sum w^2 | overlap variant | rebuild = saved |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sel:
            rk, rp = r["ranks"], r["reproduce"]
            L.append(f"| {r['variant']} | {r['session']} | {r['N']} | {r['budget']} | {rk['n_eff']} | {rk['N*D']} | {rk['N']} | "
                     f"{rk['N*D_eff']} | {rk['energy90']} | {r['omp_stopped_at']}/{r['budget']} | {f3(r['diam_Z'])} | "
                     f"{f3(r['norm_y'])} | {f3(r['eps_mod'], 4)} | {f3(r['fw_bound'])} | {f3(r['effective_size'], 1)} | "
                     f"{f3(100 * r['overlap_variant'], 0)}% | {rp.get('same_clips_as_saved', rp.get('saved', 'n/a'))} |")
        L += ["", "n_eff, same-music pairs, variance kept, Prop. fw rate and Lemma modal:", ""]
        for r in sel:
            lm = r["lemma_modal"]
            alt = r["rank_N_alternative"]
            L.append(f"- {r['variant']} S{r['session']}: n_eff {f3(r['n_eff'], 0)} (N*D = {r['N'] * r['D']}), "
                     f"{f3(r['same_music_pairs'], 0)} same-music pairs, D_eff {f3(r['d_eff'], 1)}, variance kept "
                     f"{f3(100 * r['variance_kept'], 1)}%; eps_mod(t) log-log slope {f3(r['rate_slope_loglog'], 2)} "
                     f"(bound: -0.5); at the stop eps_mod {f3(r['eps_mod_at_stop'], 4)} vs bound {f3(r['fw_bound_at_stop'])}; "
                     f"Lemma: ||P Xi|| {f3(lm['P_Xi'], 4)} <= sqrt(lam1) eps_mod {f3(lm['sqrt_lam1_eps_mod'], 4)}, "
                     f"||(I-P) Xi|| {f3(lm['out_Xi'], 4)} <= tau {f3(lm['tau'], 3)}; eps_B {f3(lm['eps_B'], 4)}"
                     + (f"; with N_eff = N: M = {alt['M']}, overlap {f3(100 * alt['overlap'], 0)}%, eps_mod {f3(alt['eps_mod_in_tmr_coords'], 4)}"
                        if alt else "; N_eff = N gives the same rank")
                     + f"; selection {f3(r['seconds_selection'], 1)} s after {f3(r['seconds_features'], 1)} s of features")
        L.append("")
    bl = {b["name"]: b for b in (R.get("buffers") or {}).get("buffers", [])}
    gl = {b["name"]: b for b in (R.get("grads") or {}).get("buffers", [])}
    srr = (R.get("shortrun") or {}).get("runs", [])
    if bl:
        bb = R["buffers"]
        L += [f"## Buffers at the shared S0 model (pool: {bb['N']} S0 clips of {bb['pool_dances']} dances, B = {bb['budget']}, M = {bb['M']})", ""]
        steps = (R.get("shortrun") or {}).get("steps")
        modes = sorted({r["mode"] for r in srr})
        hdr = "| Buffer | eps_mod | eps_B | coverage | delta | kappa | linear share | <g,g-g_B>/(‖g‖ delta) |" + \
              "".join(f" dL S0 pool {m} | dL S0 test {m} | dL S1 test {m} |" for m in modes)
        L += [hdr, "|" + "---|" * (8 + 3 * len(modes))]
        for n, b in bl.items():
            g = gl.get(n, {})
            cells = [n + (f" ({', '.join(b['members'])})" if b["kind"] == "saved" else ""), f3(b["eps_mod"], 4), f3(b["eps_B"], 4),
                     f3(b["coverage"]), f3(g.get("delta"), 4), f3(g.get("kappa"), 4), f3(g.get("linear_share"), 2),
                     f3(g.get("cs_tightness"), 2)]
            for m in modes:
                r = next((x for x in srr if x["name"] == n and x["mode"] == m and not x.get("repeat")), None)
                ev = r["evals"][steps] if r else {}
                cells += [f3(ev.get("S0_pool"), 5), f3(ev.get("S0_test"), 5), f3(ev.get("S1_test"), 5)]
            L.append("| " + " | ".join(cells) + " |")
        for m in modes:
            for n in ("No replay", "Full replay"):
                r = next((x for x in srr if x["name"] == n and x["mode"] == m), None)
                if r:
                    ev = r["evals"][steps]
                    L.append(f"- {m}, {n}: dL S0 pool {f3(ev.get('S0_pool'), 5)}, S0 test {f3(ev.get('S0_test'), 5)}, "
                             f"S1 test {f3(ev.get('S1_test'), 5)}")
        L.append("")
    if R.get("grads"):
        g = R["grads"]
        L += [f"Gradients: {g['params']} parameters sketched to {g['sketch']}, {g['R']} noise draws per clip (two halves); "
              f"||gbar|| {f3(g['norm_gbar'], 4)}; linear model on {g['lin_rank']} PCA directions of the atoms: CV R^2 "
              f"{f3(g['linear_cv_r2'], 3)} with a noise ceiling of {f3(g['signal_fraction_of_clip_gradients'], 3)} "
              f"(share of the clip-gradient variance that is not diffusion noise).", ""]
    co = R.get("correlations") or {}
    if co:
        L += ["## Tab. theory for DiscoForcing (Spearman over buffers at the S0 model)", "", "| Pair | rho | p | n |", "|---|---|---|---|"]

        def row(label, d):
            if d and d.get("rho") is not None:
                L.append(f"| {label} | {f3(d['rho'], 2)} | {f3(d['p'], 3)} | {d['n']} |")
        for mode in ("gt", "kd"):
            o = co.get(f"shortrun_{mode}_S0_pool")
            if o:
                row(f"eps_mod and short-run forgetting ({mode}, S0 pool loss)", o["eps_mod"])
                row(f"delta and short-run forgetting ({mode})", o["delta"])
                row(f"eps_B and short-run forgetting ({mode})", o["eps_B"])
                row(f"unwhitened eps_mod and short-run forgetting ({mode})", o["eps_mod_unwhitened"])
                row(f"coverage and short-run forgetting ({mode})", o["coverage"])
            o = co.get(f"shortrun_{mode}_S0_test")
            if o:
                row(f"eps_mod and short-run forgetting ({mode}, S0 test loss)", o["eps_mod"])
            o = co.get(f"shortrun_{mode}_vs_longrun_SR")
            if o and o.get("rho") is not None:
                row(f"short-run forgetting and S0 SR after S1, 40k steps ({mode}; expect < 0)", o)
        row("eps_mod and delta", co.get("eps_mod_vs_delta"))
        row("eps_B and delta", co.get("eps_B_vs_delta"))
        row("unwhitened eps_mod and delta", co.get("eps_mod_unwhitened_vs_delta"))
        if "median_linear_share" in co:
            L.append(f"| Share of delta explained by the linear term (median) | {f3(co['median_linear_share'], 2)} | | |")
        L.append("")
        for mode in ("gt", "kd"):
            nz = co.get(f"shortrun_{mode}_noise")
            if nz:
                L.append(f"- {mode}: run-to-run spread of dL (same buffer, other data/noise seeds) {nz['repeat_spread']}; "
                         f"between-buffer sd {f3(nz['between_buffer_sd'], 5)}")
        L.append("")
    se = R.get("sensitivity") or {}
    if se.get("rows"):
        L += ["## Sensitivity per temporal mode (is ||J_m|| ~ 1/sqrt(lambda_m)?)", "",
              "| mode | retained | lambda_m | ‖J_m v‖ | sqrt(lambda_m) ‖J_m v‖ | signal / floor |", "|---|---|---|---|---|---|"]
        for r in se["rows"]:
            L.append(f"| {r['mode']} | {r['retained']} | {f3(r['lambda'], 4)} | {f3(r['J_m_v'], 4)} | {f3(r['standardized'], 4)} | "
                     f"{f3(r['diff_over_floor_min'], 1)} |")
        L += ["", f"Slope of log ||J_m v|| on log lambda_m over the retained modes: {f3(se.get('slope_log_J_vs_log_lambda_retained'), 2)} "
              f"({se['reading']}). {se['clips']} full-length clips, {se['R']} draws each.", ""]
    cv = R.get("coverage") or {}
    if cv.get("table"):
        ss = sorted({k for d in cv["table"].values() for k in d})
        L += [f"## Tab. coverage for DiscoForcing ({cv['bins']}x{cv['bins']} grid, {cv['ruler']})", "",
              "| Method | " + " | ".join(ss) + " |", "|---|" + "---|" * len(ss)]
        for m, d in sorted(cv["table"].items()):
            L.append(f"| {NAME.get(m, m)} | " + " | ".join(f3(d.get(s)) for s in ss) + " |")
        L += ["", "Spearman across runs: " + "; ".join(
            f"S{j}: with SR after S4 {f3(v['vs_SR_after_last'].get('rho'), 2)} (n {v['vs_SR_after_last']['n']}), with its drop "
            f"{f3(v['vs_drop'].get('rho'), 2)}" for j, v in cv["per_session"].items()), ""]
    if R.get("errors"):
        L += ["## Errors", ""] + [f"- {k}: {v.strip().splitlines()[-1]}" for k, v in R["errors"].items()] + [""]
    p = c.out / "THEORY.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--tag", default="v2_40000_full_s0v2")
    ap.add_argument("--out", default=None, help="default results/<benchmark>/supp/theory")
    ap.add_argument("--stages", nargs="*", default=["select", "buffers", "grads", "sensitivity", "shortrun", "coverage"])
    ap.add_argument("--select_variants", nargs="*", default=["tmr_joint_neff_kd", "tmr_joint_neff"],
                    help="chains whose S1-S3 models re-run the selection (S0 is shared)")
    ap.add_argument("--ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_random", type=int, default=16, help="random ER buffers for the buffer / gradient stages")
    ap.add_argument("--bins", type=int, default=20, help="coverage grid cells per axis")
    ap.add_argument("--R", type=int, default=8, help="noise draws per clip for the gradients (two halves)")
    ap.add_argument("--sketch", type=int, default=16384, help="CountSketch dimension of the gradients")
    ap.add_argument("--lin_rank", type=int, default=48, help="PCA directions of the atoms for the linear map J")
    ap.add_argument("--sens_modes", type=int, default=8)
    ap.add_argument("--sens_v", type=int, default=2)
    ap.add_argument("--sens_clips", type=int, default=32)
    ap.add_argument("--sens_R", type=int, default=2)
    ap.add_argument("--sens_eps", type=float, default=0.5, help="perturbation in units of the mode's sd")
    ap.add_argument("--steps", type=int, default=300, help="short-run training steps of Session 1")
    ap.add_argument("--eval_at", default="50,150,300")
    ap.add_argument("--eval_pool", type=int, default=384, help="S0 pool clips scored for the short-run loss")
    ap.add_argument("--eval_test", type=int, default=192)
    ap.add_argument("--replay", nargs="*", default=["gt", "kd"], help="short-run replay: gt and / or kd")
    ap.add_argument("--train_random", type=int, default=6, help="ER draws trained in the short-run stage")
    ap.add_argument("--repeat", type=int, default=2, help="extra seeds for TMR and ER draw 0 (run-to-run spread)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_fp32", action="store_true", help="keep the bf16 attention kernels and TF32")
    a = ap.parse_args()
    a.out = a.out or f"results/{a.benchmark}/supp/theory"
    c = Ctx(a)
    c.out.mkdir(parents=True, exist_ok=True)
    if not a.no_fp32:
        try:
            import ev_common

            ev_common.set_precision(True)
        except Exception as e:  # noqa: BLE001
            log(f"fp32 attention not set ({e!r}); continuing with the default kernels")
    torch.manual_seed(a.seed)
    log(f"device {c.device} | screen {c.res} | S0 model {c.s0_run} | stages {a.stages}")
    jp = c.out / "theory.json"
    R = json.loads(jp.read_text()) if jp.exists() else {}
    R["errors"] = {}
    R["args"] = vars(a)
    st: dict = {}
    fns = {"select": lambda: stage_select(c), "buffers": lambda: stage_buffers(c, st), "grads": lambda: stage_grads(c, st),
           "sensitivity": lambda: stage_sensitivity(c, st), "shortrun": lambda: stage_shortrun(c, st),
           "coverage": lambda: stage_coverage(c, st)}
    for name in a.stages:
        t0 = time.time()
        log(f"stage {name}")
        try:
            R[name] = fns[name]()
        except Exception:  # noqa: BLE001
            R["errors"][name] = traceback.format_exc()
            log(f"stage {name} FAILED:\n{R['errors'][name]}")
        log(f"stage {name}: {time.time() - t0:.0f} s")
        if name in ("grads",) and "buffers" in R and "bufs" in st:  # buffer rows gain delta etc.
            R["buffers"]["buffers"] = [{k: v for k, v in b.items() if k not in ("idx", "w")}
                                       | {"idx": b["idx"].tolist(), "w": b["w"].tolist()} for b in st["bufs"]]
        try:
            R["correlations"] = correlations(c, R)
        except Exception:  # noqa: BLE001
            R["errors"]["correlations"] = traceback.format_exc()
        jp.write_text(json.dumps(jsonable(R), indent=1))
        write_md(c, R)
    log(f"wrote {jp} and {c.out / 'THEORY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
