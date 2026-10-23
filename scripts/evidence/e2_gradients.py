#!/usr/bin/env python3
"""
E2 -- gradient-level evidence at theta_{k-1}: Proposition 1's buffer error delta, the modulus kappa, and where
gradient interference lives in the generator.

For every session boundary k (theta_{k-1} = last.ckpt of session k-1):
  * per-sample gradients of the diffusion loss for every past training window, stored as CountSketches: bank A (used
    by the gradient-based selection rules) and an independent bank B split into two halves (used to evaluate); exact
    per-session mean gradients g_j (j < k) and g_k of the new session, plus g_k's per-coordinate second moment
  * delta(B, w) = ||g_bar_{<k} - sum w_i g_i|| (the operative quantity of Prop. 1) of the expected (noise-averaged)
    gradients, estimated without Monte-Carlo bias from the two halves (see Delta), for TMR, its Tab. 4 ablations,
    herding, k-center, ER, loss-based selection and the gradient-matching oracle (OMP on gradient sketches, i.e.
    GCR / GradMatch); with eps_B, eps_mod and kappa = delta / eps_B (Def. kappa)
  * kappa calibration (the open question of Sec. scope: why the temporal axis?): over random and metadata-tilted
    buffers, the rank correlation between delta and the selection error measured in each candidate space -- raw phi,
    whitened temporal modes (TMR), unwhitened, channel modes, random subspace, time-pooled -- i.e. in which space a
    small selection error certifies a small gradient error, and how stable kappa is there
  * gradient anatomy: per functional block (audio encoder, cross-attention K/V and Q/O, self-attention, FFN, ...)
    the cosine between old-session and new-session gradients and the first-order interference -<g_j, g_k> of Eq. 4
    (plain and Adam-preconditioned), plus the per-layer split

Usage (repo root):
  python scripts/evidence/e2_gradients.py --config configs/cl/ldf_aistpp_genre5_s0.yaml --benchmark aistpp_genre5
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev_common as E
import tmr_select as S


def block_energy(P: E.Params, g: torch.Tensor) -> np.ndarray:
    sizes = [int(e - s) for s, e in zip(P.offsets[:-1], P.offsets[1:])]
    sq = torch.stack([x.double().pow(2).sum() for x in g.split(sizes)]).cpu().numpy()
    return np.array([sq[[b == blk for b in P.blocks]].sum() for blk in E.BLOCKS])


def anatomy(P: E.Params, g_sess: dict, g_k: torch.Tensor, v_k: torch.Tensor) -> dict:
    """Eq. 4 per block: -<g_j, g_k> > 0 means the new session's descent direction raises old session j's loss."""
    pre = g_k / (v_k.sqrt() + 1e-12)
    nk = P.block_dots(g_k, g_k)
    out = {}
    for j, gj in g_sess.items():
        d, nj = P.block_dots(gj, g_k), P.block_dots(gj, gj)
        da = P.block_dots(gj, pre)
        tot = sum(d.values())
        out[j] = {
            "cos": tot / math.sqrt(sum(nj.values()) * sum(nk.values()) + 1e-300),
            "interference_gd": -tot, "interference_adam": -sum(da.values()),
            "blocks": {b: {"cos": d[b] / math.sqrt(nj[b] * nk[b] + 1e-300), "interference_gd": -d[b],
                           "interference_adam": -da[b], "share_gd": d[b] / tot if tot else float("nan"),
                           "norm_old": math.sqrt(nj[b]), "norm_new": math.sqrt(nk[b])} for b in d},
            "layers_interference_gd": {l: -x for l, x in P.block_dots(gj, g_k, by="layer").items()},
        }
    return out


def pairwise_cos(P: E.Params, g: dict) -> dict:
    keys = sorted(g)
    n = {k: float(torch.dot(g[k].double(), g[k].double())) for k in keys}
    return {f"{a}-{b}": float(torch.dot(g[a].double(), g[b].double())) / math.sqrt(n[a] * n[b] + 1e-300)
            for i, a in enumerate(keys) for b in keys[i + 1:]}


class Delta:
    """Squared gradient error delta^2(B, w) = ||g_bar - sum_i w_i g_i||^2 of the *expected* (noise-averaged)
    per-sample gradients. Each window's gradient is estimated twice, from independent halves of its noise draws; with
    d_h the buffer error in half h, E<d_1, d_2> = ||d||^2 exactly (the Monte-Carlo noise of the halves is independent
    and the sketch is unbiased for inner products), whereas ||(d_1 + d_2) / 2||^2 is inflated by the noise. The
    unbiased estimate can be negative when a buffer's true error is below the noise; comparisons therefore use
    delta^2 (ratios, ranks), and delta = sqrt(max(delta^2, 0)) only for display."""

    def __init__(self, S1: np.ndarray, S2: np.ndarray):
        self.S1, self.S2 = S1, S2
        self.t1, self.t2 = S1.mean(0, dtype=np.float64), S2.mean(0, dtype=np.float64)

    def __call__(self, idx, w) -> tuple[float, float]:
        """(unbiased delta^2, naive delta^2)."""
        idx, w = np.asarray(idx, int), np.asarray(w, np.float64)
        d1 = self.t1 - w @ self.S1[idx].astype(np.float64)
        d2 = self.t2 - w @ self.S2[idx].astype(np.float64)
        return float(d1 @ d2), float(((d1 + d2) / 2) @ ((d1 + d2) / 2))


def calibration(reps: dict, delta_of: Delta, groups, quotas, cells, n_random, seed) -> dict:
    """Rank correlation, over random and Dirichlet-tilted buffers, between delta and the selection error in each
    representation; kappa_r = delta / eps_r and its coefficient of variation."""
    rng = np.random.default_rng(seed)
    bufs = [S.random_quota(groups, quotas, rng) for _ in range(n_random)]
    bufs += [S.tilted_quota(groups, quotas, cells, rng) for _ in range(n_random)]
    both = np.array([delta_of(b, S.uniform(b)) for b in bufs])
    d2, naive2 = both[:, 0], both[:, 1]
    out = {"n_buffers": len(bufs), "delta2_mean": d2.mean(), "delta2_std": d2.std(),
           "noise_share_of_naive_delta2": float(1 - d2.mean() / naive2.mean()),
           "share_buffers_below_noise": float((d2 <= 0).mean()), "spaces": {}}
    delta = np.sqrt(np.maximum(d2, 0))
    for name, X in reps.items():
        tgt = X.mean(0, dtype=np.float64)
        eps = np.array([S.mean_error(X, b, S.uniform(b), tgt) for b in bufs])
        ok = d2 > 0
        kap = delta[ok] / np.maximum(eps[ok], 1e-300)
        out["spaces"][name] = {"spearman": float(spearmanr(eps, d2)[0]),
                               "pearson_sq": float(pearsonr(eps ** 2, d2)[0]),
                               "kappa_cv": float(kap.std() / kap.mean()) if ok.sum() > 2 else None,
                               "eps": eps, "delta": delta}
    return out


def figures(res: dict, out: Path, k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    cal = res["calibration"]["spaces"]
    names = list(cal)
    ax[0].barh(names, [cal[n]["spearman"] for n in names], color=["C3" if "TMR" in n else "C0" for n in names])
    ax[0].set(xlabel="Spearman($\\varepsilon_r$, $\\delta$) over buffers", title="which selection space tracks $\\delta$")
    raw = cal.get("raw phi", next(iter(cal.values())))
    ax[1].scatter(raw["eps"], raw["delta"], s=6, alpha=.4, label="random / tilted")
    for name, r in res["rules"].items():
        if "eps_B" in r:
            ax[1].scatter(r["eps_B"], r["delta"], s=40, marker="*", label=name)
    ax[1].set(xlabel="$\\varepsilon_B$ (raw trajectories)", ylabel="$\\delta$ (gradient error)", title="Prop. 1 / $\\kappa$")
    ax[1].legend(fontsize=5, ncol=2)
    rules = [n for n in res["rules"] if "delta_rel_random" in res["rules"][n]]
    ax[2].barh(rules, [res["rules"][n]["delta_rel_random"] for n in rules])
    ax[2].axvline(1, color="k", lw=.8)
    ax[2].set(xlabel="$\\delta$ / mean $\\delta$ of random buffers", title="gradient error by selection rule")
    fig.suptitle(f"boundary k={k}")
    E.savefig(fig, out / f"e2_kappa_k{k}")

    an = res["anatomy"]
    olds = sorted(an, key=int)
    blocks = [b for b in E.BLOCKS if b in an[olds[0]]["blocks"]]
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.6 + 0.45 * len(olds) + 2))
    M = np.array([[an[j]["blocks"][b]["cos"] for b in blocks] for j in olds])
    im = ax[0].imshow(M, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax[0].set(xticks=range(len(blocks)), yticks=range(len(olds)), yticklabels=[f"S{j} vs S{k}" for j in olds],
              title="cos($g_j$, $g_k$) per block")
    ax[0].set_xticklabels(blocks, rotation=40, ha="right", fontsize=7)
    plt.colorbar(im, ax=ax[0])
    share = np.array([[an[j]["blocks"][b]["interference_gd"] for b in blocks] for j in olds])
    for r, j in enumerate(olds):
        ax[1].barh([f"S{j}"] * len(blocks), share[r], left=np.concatenate([[0], np.cumsum(share[r])[:-1]]),
                   color=[f"C{i}" for i in range(len(blocks))])
    ax[1].legend([plt.Rectangle((0, 0), 1, 1, color=f"C{i}") for i in range(len(blocks))], blocks, fontsize=6)
    ax[1].set(xlabel="$-\\langle g_j^b, g_k^b\\rangle$ (first-order forgetting, Eq. 4)", title="interference by block")
    E.savefig(fig, out / f"e2_anatomy_k{k}")


def main() -> int:
    ap = E.common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--boundaries", type=int, nargs="*", default=None)
    ap.add_argument("--sketch_dim", type=int, default=32768)
    ap.add_argument("--R_eval", type=int, default=32,
                    help="noise draws per window of the evaluation bank B (even: two independent halves)")
    ap.add_argument("--n_random", type=int, default=150, help="random and tilted buffers, each, for calibration")
    ap.add_argument("--new_windows", type=int, default=None, help="cap on new-session windows for g_k")
    a = E.finish_args(ap.parse_args())
    E.set_precision(not a.bf16_attention)
    cfg = E.load_cfg(a.config)
    runs = E.runs_of(a, cfg)
    K = len(E.manifest(cfg, a.benchmark)["sessions"])
    ks = a.boundaries or [k for k in range(1, K) if k - 1 in runs]
    model = E.build_model(cfg, a.device)
    C = int(cfg.model.params.input_dim)
    train_bs = int(cfg.data.train_bs)

    for k in ks:
        past = list(range(k))
        pool = E.build_pool(cfg, a.benchmark, past, "train", a.T, max_per_session=a.max_windows, seed=a.seed)
        new = E.build_pool(cfg, a.benchmark, [k], "train", a.T, max_per_session=a.new_windows or a.max_windows,
                           seed=a.seed)
        ck = runs[k - 1] / "last.ckpt"
        phi = E.phi_cached(a, model, ck, pool, "train")
        E.load_weights(model, ck)
        P = E.model_params(model)
        N, T, _ = phi.shape
        groups = pool.session
        chunk = model.chunk_size
        bankA = E.noise_bank(N, T, C, a.R, chunk, seed=10_000 + k)
        bankB = E.noise_bank(N, T, C, a.R_eval, chunk, seed=20_000 + k)
        bankN = E.noise_bank(len(new), T, C, a.R_eval, chunk, seed=30_000 + k)
        skA = E.CountSketch(P.numel, a.sketch_dim, seed=101, device=a.device)
        skB = E.CountSketch(P.numel, a.sketch_dim, seed=202, device=a.device)
        SA = np.zeros((N, a.sketch_dim), np.float32)
        SB = [np.zeros((N, a.sketch_dim), np.float32) for _ in range(2)]
        lossA = np.zeros(N)
        energy = np.zeros((N, len(E.BLOCKS)))
        sums = {j: torch.zeros(P.numel, device=a.device) for j in past}
        E.log(f"k={k}: {N} past windows, {len(new)} new windows, {P.numel / 1e6:.1f}M parameters")

        def cb_a(pos, i, g, l):
            SA[pos] = skA(g).float().cpu().numpy()
            lossA[pos] = l

        def cb_b(half):
            def cb(pos, i, g, l):
                SB[half][pos] = skB(g).float().cpu().numpy()
                sums[int(groups[i])].add_(g, alpha=0.5)
                if half == 0:
                    energy[pos] = block_energy(P, g)
            return cb

        E.per_sample_grads(model, P, pool, bankA, a.device, range(N), cb_a)
        E.log(f"k={k}: pass A done")
        h = a.R_eval // 2
        for half, draws in enumerate((slice(0, h), slice(h, 2 * h))):
            E.per_sample_grads(model, P, pool, bankB, a.device, range(N), cb_b(half), draws=draws)
        E.log(f"k={k}: pass B done")
        delta_of = Delta(SB[0], SB[1])
        # per-window gradient signal-to-noise: the signal that distinguishes window i is its expected gradient's
        # deviation from the pool mean, unbiasedly <dev_1, dev_2>; s_1 - s_2 = n_1 - n_2 gives the noise of one half
        dev1, dev2 = SB[0] - delta_of.t1, SB[1] - delta_of.t2
        sig = (dev1.astype(np.float64) * dev2).sum(1)
        noise_draw = ((SB[0].astype(np.float64) - SB[1]) ** 2).sum(1) / 2 * h
        snr = {"median_signal_to_noise_one_draw": float(np.median(sig / noise_draw)),
               "share_windows_signal_below_one_draw_noise": float((sig < noise_draw).mean()),
               "draws_for_snr_1_median": float(np.median(noise_draw / np.maximum(sig, 1e-300))),
               "R_select": a.R, "R_eval": a.R_eval}
        gk_sum = torch.zeros(P.numel, device=a.device)
        gk_sq = torch.zeros(P.numel, device=a.device)

        def cb_n(pos, i, g, l):
            gk_sum.add_(g)
            gk_sq.add_(g * g)

        E.per_sample_grads(model, P, new, bankN, a.device, range(len(new)), cb_n)
        g_k = gk_sum / len(new)
        v_k = g_k * g_k + (gk_sq / len(new) - g_k * g_k).clamp_min(0) / train_bs  # minibatch second moment
        del gk_sum, gk_sq
        n_j = {j: int((groups == j).sum()) for j in past}
        g_sess = {j: sums[j] / n_j[j] for j in past}
        g_past = sum(g_sess[j] * (n_j[j] / N) for j in past)
        an = anatomy(P, g_sess, g_k, v_k)
        an_past = {"past_vs_new": anatomy(P, {"past": g_past}, g_k, v_k)["past"]}
        pair = pairwise_cos(P, {**{str(j): g_sess[j] for j in past}, str(k): g_k})

        # selection rules (phi-based as in E1, plus gradient-sketch oracle and loss-based)
        ops = S.operators(phi)
        M = S.bic_rank(ops["lam"], N * phi.shape[2])[0]
        Z = S.modal_dictionary(phi, "temporal", M=M, ops=ops)[0]
        bufs, info = S.select_buffers(phi, groups, a.budget, seed=a.seed + k, ops=ops, grad_atoms=SA, loss=lossA)
        raw = phi.reshape(N, -1)
        rules = {}
        for name, b in bufs.items():
            (d2, n2), eB = delta_of(b["idx"], b["w"]), S.mean_error(raw, b["idx"], b["w"])
            d = math.sqrt(max(d2, 0.0))
            rules[name] = {"delta2": d2, "delta": d, "delta_naive": math.sqrt(n2), "eps_B": eB,
                           "eps_mod": S.mean_error(Z, b["idx"], b["w"]), "kappa": d / eB, "seconds": b["seconds"],
                           "n": len(b["idx"])}
        rng = np.random.default_rng(a.seed + 7 * k)
        rand = [S.random_quota(groups, info["quotas"], rng) for _ in range(a.n_random)]
        d2_rand = np.array([delta_of(i, S.uniform(i))[0] for i in rand])
        e_rand = np.array([S.mean_error(raw, i, S.uniform(i)) for i in rand])
        m2 = d2_rand.mean()
        for name, r in rules.items():
            # ratios of unbiased squared errors, reported on the delta scale
            r["delta_rel_random"] = math.sqrt(max(r["delta2"], 0.0) / m2) if m2 > 0 else float("nan")
            r["delta_percentile_among_random"] = float((d2_rand < r["delta2"]).mean())
        if "grad_omp" in rules:
            gap = m2 - rules["grad_omp"]["delta2"]
            for r in rules.values():
                r["oracle_gap_closed"] = (m2 - r["delta2"]) / gap if gap > 0 else float("nan")
        rules["random_mean"] = {"delta2": m2, "delta2_std": d2_rand.std(), "delta": math.sqrt(max(m2, 0.0)),
                                "eps_B": e_rand.mean(),
                                "kappa": math.sqrt(max(m2, 0.0)) / e_rand.mean(), "n_buffers": len(rand)}

        reps = {"raw phi": raw, "temporal whitened (TMR)": Z,
                "temporal unwhitened": S.modal_dictionary(phi, "temporal", whiten=False, M=M, ops=ops)[0],
                "channel whitened": S.modal_dictionary(phi, "channel", ops=ops)[0],
                "random temporal subspace": S.modal_dictionary(phi, "temporal", basis="random", M=M, ops=ops,
                                                               rng=np.random.default_rng(a.seed))[0],
                "time-pooled phi": phi.mean(1), "gradient sketch A (noise ceiling)": SA}
        cells = [m.get("music") or n for m, n in zip(pool.meta, pool.names)]
        cal = calibration(reps, delta_of, groups, info["quotas"], cells, a.n_random, a.seed + 11 * k)
        E.log(f"k={k}: Spearman(eps_r, delta): " + ", ".join(f"{n} {v['spearman']:.2f}" for n, v in cal["spaces"].items()))
        E.log(f"k={k}: delta / random: " + ", ".join(f"{n} {r['delta_rel_random']:.2f}" for n, r in rules.items()
                                                    if "delta_rel_random" in r))

        res = {"boundary": k, "checkpoint": str(ck), "N": N, "n_new": len(new), "M_hat": M, "sketch_dim": a.sketch_dim,
               "R_select": a.R, "R_eval": a.R_eval, "quotas": info["quotas"], "rules": rules, "gradient_snr": snr,
               "calibration": cal, "anatomy": an, "anatomy_pooled_past": an_past, "pairwise_session_cos": pair,
               "grad_norms": {"g_past": float(g_past.norm()), "g_k": float(g_k.norm()),
                              **{f"g_{j}": float(g_sess[j].norm()) for j in past}},
               "per_sample_block_energy_share": dict(zip(E.BLOCKS, (energy / energy.sum(1, keepdims=True)).mean(0))),
               "block_numel": P.block_numel(),
               "per_sample_loss_mean": float(lossA.mean()),
               "buffers": {n: {"idx": b["idx"], "w": b["w"]} for n, b in bufs.items()},
               "pool_names": pool.names, "pool_starts": pool.start}
        E.dump_json(a.out_dir / "e2" / f"boundary_k{k}.json", res)
        figures(res, a.out_dir / "figures", k)
        del sums, g_sess, g_past, g_k, v_k, skA, skB
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
