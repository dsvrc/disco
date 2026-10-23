#!/usr/bin/env python3
"""
E4 -- forgetting along the training trajectory: when it happens, what is forgotten (motion prior vs audio grounding,
which diffusion noise levels) and where (which parameter blocks), from the stored checkpoints of every session.

Checkpoint path: theta_0 (S0 last.ckpt) -> S1 step checkpoints -> theta_1 -> S2 step checkpoints -> ... (every
--every-th intermediate checkpoint). At each checkpoint, on the held-out test windows of every session (frozen noise):
  * L_j^cond (the training objective), L_j^null (music zeroed, the CFG null condition) and the audio-grounding gap
    G_j = L_j^null - L_j^cond, i.e. how much the model's denoising of session j uses the music (for diffusion models
    the conditional-minus-unconditional denoising loss is the conditional-information term; Kong et al., ICLR 2023)
  * per-token losses binned by the diffusion-forcing noise level of the token
Exact decomposition of forgetting:  Delta L^cond_j = Delta L^null_j - Delta G_j  (prior drift minus lost grounding).
Path-integral attribution (complete up to quadrature error, which is reported): for every segment theta_a -> theta_b
  L_j(theta_b) - L_j(theta_a) = int_0^1 <grad L_j(theta_a + s (theta_b - theta_a)), theta_b - theta_a> ds,
evaluated with Simpson's rule and split over parameter blocks and layers: how much each block's movement during
session k raised (forgetting) or lowered (learning / backward transfer) every session's loss.

Usage (repo root):
  python scripts/evidence/e4_trajectory.py --config configs/cl/ldf_aistpp_genre5_s0.yaml --benchmark aistpp_genre5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev_common as E


def checkpoint_losses(a, model, ck: Path, pool, bank, sess_idx: dict) -> dict:
    """Per-session conditional / null losses and per-noise-bin sums at one checkpoint (cached per checkpoint file)."""
    key = E.file_tag(Path(ck), a.T, a.R_eval, a.max_windows, a.benchmark, len(pool), pool.names[0], pool.names[-1])
    path = a.out_dir / "cache" / f"e4_losses_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    meta = E.load_weights(model, ck)
    out = {"checkpoint": str(ck), "global_step": meta["global_step"], "sessions": {}}
    for j, idx in sess_idx.items():
        r = {}
        for cond in ("cond", "null"):
            lo = E.pool_losses(model, pool, bank, a.device, idx=idx, bs=a.bs, null=cond == "null")
            r[cond] = float(lo["per_window"].mean())
            r[f"{cond}_bins"] = (lo["bin_sum"] / np.maximum(lo["bin_count"], 1)).tolist()
            r["bin_count"] = lo["bin_count"].tolist()
        r["G"] = r["null"] - r["cond"]
        out["sessions"][str(j)] = r
    E.dump_json(path, out)
    return out


def attribute_session(a, model, P: E.Params, nodes: list, pool, bank, attr_idx: dict, targets: list) -> dict:
    """Path-integral attribution of every target session's loss change along nodes (checkpoint path), by adaptive
    composite Simpson per segment: start with 2 sub-intervals and double (up to --max_sub) until the integrated
    change matches the measured one (completeness) within --attr_tol. Only per-block dot products of each node's
    gradient with the segment direction are kept; endpoint gradients are reused by the next segment and parked on
    --accum_device (default cpu) meanwhile."""
    kinds = ("cond", "null") if a.attr_null else ("cond",)
    park = a.accum_device or "cpu"

    def grad(theta, j, c):
        P.assign(theta)
        return E.mean_grad(model, P, pool, bank, a.device, attr_idx[j], bs=a.bs, null=c == "null")

    def flat_of(ck):
        E.load_weights(model, ck)
        return P.flat().clone()

    def simpson(node_dots: dict, n: int) -> tuple[dict, dict]:
        wq = lambda q: (1 if q in (0, n) else 4 if q % 2 else 2) / (3 * n)
        blocks, layers = {}, {}
        for q in range(n + 1):
            bq, yq = node_dots[q * (8 // n)]  # node keys are in units of 1/8 of the segment
            for b, v in bq.items():
                blocks[b] = blocks.get(b, 0.0) + wq(q) * v
            for b, v in yq.items():
                layers[b] = layers.get(b, 0.0) + wq(q) * v
        return blocks, layers

    res = {f"{j}/{c}": {"blocks": {}, "layers": {}, "simpson": 0.0, "actual": 0.0, "segments": []}
           for j in targets for c in kinds}
    moved = {}
    th_a = flat_of(nodes[0][1])
    ends = {}
    for j in targets:
        for c in kinds:
            g, l = grad(th_a, j, c)
            ends[(j, c)] = (g.to(park), l)
    for (la, _), (lb, ckb) in zip(nodes[:-1], nodes[1:]):
        th_b = flat_of(ckb)
        d = th_b - th_a
        for blk, v in P.block_dots(d, d).items():
            moved[blk] = moved.get(blk, 0.0) + v ** 0.5
        new_ends = {}
        for (j, c), (ga, la_loss) in ends.items():
            gb, lb_loss = grad(th_b, j, c)
            node = {0: (P.block_dots(ga.to(a.device), d), P.block_dots(ga.to(a.device), d, by="layer")),
                    8: (P.block_dots(gb, d), P.block_dots(gb, d, by="layer"))}
            new_ends[(j, c)] = (gb.to(park), lb_loss)
            del gb
            actual, n = lb_loss - la_loss, 1
            while True:
                n *= 2
                for q in range(1, n, 2):  # new odd nodes at q / n of the segment
                    g, _ = grad(th_a + (q / n) * d, j, c)
                    node[q * (8 // n)] = (P.block_dots(g, d), P.block_dots(g, d, by="layer"))
                    del g
                blocks, layers = simpson(node, n)
                integ = sum(blocks.values())
                if abs(integ - actual) <= a.attr_tol * max(abs(actual), 1e-3 * abs(la_loss)) or n >= a.max_sub:
                    break
            r = res[f"{j}/{c}"]
            for b, v in blocks.items():
                r["blocks"][b] = r["blocks"].get(b, 0.0) + v
            for b, v in layers.items():
                r["layers"][b] = r["layers"].get(b, 0.0) + v
            r["simpson"] += integ
            r["actual"] += actual
            r["segments"].append({"from": la, "to": lb, "actual": actual, "simpson": integ, "sub_intervals": n,
                                  "blocks": blocks})
        ends, th_a = new_ends, th_b
        E.log(f"  attribution segment {la} -> {lb} done")
    for r in res.values():
        r["quadrature_rel_err"] = abs(r["simpson"] - r["actual"]) / max(abs(r["actual"]), 1e-30)
    return {"targets": targets, "nodes": [l for l, _ in nodes], "result": res, "block_distance_moved": moved}


def figures(res: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chain, K = res["chain"], res["K"]
    x = np.arange(len(chain))
    bounds = [i for i, c in enumerate(chain) if c["label"].endswith(":end")]
    fig, ax = plt.subplots(1, 3, figsize=(17, 3.6))
    for j in range(K):
        ys = [c["losses"].get(str(j), {}).get("cond", np.nan) for c in chain]
        ax[0].plot(x, ys, label=f"S{j}")
        gs = [c["losses"].get(str(j), {}).get("G", np.nan) for c in chain]
        ax[1].plot(x, gs, label=f"S{j}")
    for axx in ax[:2]:
        for b in bounds:
            axx.axvline(b, color="k", lw=.5, alpha=.4)
        axx.set_xticks(bounds)
        axx.set_xticklabels([chain[b]["label"].split(":")[0] for b in bounds], fontsize=7)
        axx.legend(fontsize=6)
    ax[0].set(ylabel="$L_j^{cond}$ (test windows)", title="loss of every session along training")
    ax[1].set(ylabel="$G_j = L_j^{null} - L_j^{cond}$", title="audio grounding along training")
    dec = res["decomposition"]
    keys = [k for k in dec if k.startswith("S")]
    if keys:
        dl = [dec[k]["dL_cond"] for k in keys]
        dn = [dec[k]["dL_null"] for k in keys]
        dg = [-dec[k]["dG"] for k in keys]
        w = 0.27
        xx = np.arange(len(keys))
        ax[2].bar(xx - w, dl, w, label="$\\Delta L^{cond}$ (forgetting)")
        ax[2].bar(xx, dn, w, label="$\\Delta L^{null}$ (prior drift)")
        ax[2].bar(xx + w, dg, w, label="$-\\Delta G$ (lost grounding)")
        ax[2].set_xticks(xx)
        ax[2].set_xticklabels(keys, rotation=40, ha="right", fontsize=6)
        ax[2].axhline(0, color="k", lw=.6)
        ax[2].legend(fontsize=6)
        ax[2].set(title="what is forgotten (old session, after S_t)")
    E.savefig(fig, out / "e4_trajectory")

    if res.get("noise_profile"):
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        centers = [(lo + hi) / 2 for lo, hi in zip(E.LOSS_BINS[:-1], E.LOSS_BINS[1:])]
        for key, prof in res["noise_profile"].items():
            ax.plot(centers, prof["dcond"], "o-", label=key)
        ax.axhline(0, color="k", lw=.6)
        ax.set(xlabel="noise level of token", ylabel="$\\Delta$ per-token loss", title="forgetting by noise level")
        ax.legend(fontsize=6)
        E.savefig(fig, out / "e4_noise_profile")

    att = res.get("attribution", {})
    if att:
        sessions = sorted(att, key=int)
        fig, axes = plt.subplots(1, len(sessions), figsize=(4.5 * len(sessions), 3.4), squeeze=False)
        for ax, k in zip(axes[0], sessions):
            rr = att[k]["result"]
            keys = [key for key in rr if key.endswith("/cond")]
            blocks = [b for b in E.BLOCKS if any(b in rr[key]["blocks"] for key in keys)]
            for i, key in enumerate(keys):
                vals = [rr[key]["blocks"].get(b, 0.0) for b in blocks]
                pos = np.cumsum([0] + [max(v, 0) for v in vals])[:-1]
                neg = np.cumsum([0] + [min(v, 0) for v in vals])[:-1]
                for bi, v in enumerate(vals):
                    ax.bar(i, v, bottom=pos[bi] if v >= 0 else neg[bi], color=f"C{bi}")
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels([f"S{key.split('/')[0]}" for key in keys])
            ax.axhline(0, color="k", lw=.6)
            ax.set(title=f"training S{k}: $\\Delta L_j$ by block", ylabel="contribution to $\\Delta L_j^{cond}$")
        axes[0][-1].legend([plt.Rectangle((0, 0), 1, 1, color=f"C{i}") for i in range(len(E.BLOCKS))], E.BLOCKS,
                           fontsize=6, loc="best")
        E.savefig(fig, out / "e4_attribution")


def main() -> int:
    ap = E.common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--R_eval", type=int, default=4, help="noise draws per test window for the loss curves")
    ap.add_argument("--every", type=int, default=1, help="use every n-th intermediate checkpoint for loss curves")
    ap.add_argument("--sessions", type=int, nargs="*", default=None, help="sessions whose training is attributed")
    ap.add_argument("--attr_every", type=int, default=2, help="every n-th intermediate checkpoint as Simpson segment end")
    ap.add_argument("--attr_windows", type=int, default=64, help="test windows per session for the attribution")
    ap.add_argument("--no_attr", action="store_true")
    ap.add_argument("--no_attr_null", dest="attr_null", action="store_false", help="attribute L^cond only")
    ap.add_argument("--attr_tol", type=float, default=0.05, help="completeness tolerance per segment (relative)")
    ap.add_argument("--max_sub", type=int, default=8, choices=[2, 4, 8], help="max Simpson sub-intervals per segment")
    ap.add_argument("--accum_device", default=None)
    a = E.finish_args(ap.parse_args())
    E.set_precision(not a.bf16_attention)
    cfg = E.load_cfg(a.config)
    runs = E.runs_of(a, cfg)
    if 0 not in runs:
        raise SystemExit("session 0 run dir not found")
    K = len(E.manifest(cfg, a.benchmark)["sessions"])
    trained = [k for k in range(1, K) if k in runs and k - 1 in runs]
    model = E.build_model(cfg, a.device)
    C = int(cfg.model.params.input_dim)

    pool = E.build_pool(cfg, a.benchmark, range(K), "test", a.T, max_per_session=a.max_windows, seed=a.seed)
    bank = E.noise_bank(len(pool), a.T, C, a.R_eval, model.chunk_size, seed=60_000)
    sess_idx = {j: np.where(pool.session == j)[0] for j in range(K) if (pool.session == j).any()}

    chain = [("S0:end", runs[0] / "last.ckpt", 0)]
    for k in trained:
        chain += [(l, p, k) for l, p in E.session_path(runs, k, a.every)[1:]]
    E.log(f"{len(chain)} checkpoints on the path; sessions trained: {[0] + trained}")
    rows = []
    for label, ck, k in chain:
        lo = checkpoint_losses(a, model, ck, pool, bank, sess_idx)
        rows.append({"label": label, "session_training": k, "checkpoint": str(ck), "global_step": lo["global_step"],
                     "losses": lo["sessions"]})
        E.log(f"{label}: " + " ".join(f"S{j}={v['cond']:.4f}/G={v['G']:.4f}" for j, v in lo["sessions"].items()))

    ends = {int(r["label"].split(":")[0][1:]): r for r in rows if r["label"].endswith(":end")}
    decomposition, noise_profile = {}, {}
    for j in sorted(ends):
        for t in sorted(ends):
            if t <= j:
                continue
            a0, a1 = ends[j]["losses"][str(j)], ends[t]["losses"][str(j)]
            key = f"S{j}@after_S{t}"
            decomposition[key] = {"dL_cond": a1["cond"] - a0["cond"], "dL_null": a1["null"] - a0["null"],
                                  "dG": a1["G"] - a0["G"], "G_retention": a1["G"] / a0["G"] if a0["G"] else None,
                                  "L_cond_at_learn": a0["cond"], "G_at_learn": a0["G"]}
            noise_profile[key] = {"dcond": (np.array(a1["cond_bins"]) - np.array(a0["cond_bins"])).tolist(),
                                  "dnull": (np.array(a1["null_bins"]) - np.array(a0["null_bins"])).tolist(),
                                  "cond_at_learn": a0["cond_bins"]}
    # forward transfer: sessions not yet trained, scored before / after each session
    zero_shot = {f"S{j}@after_S{t}": ends[t]["losses"][str(j)] for t in ends for j in sess_idx if j > t}

    res = {"benchmark": a.benchmark, "K": K, "trained_sessions": [0] + trained, "T": a.T, "R_eval": a.R_eval,
           "n_test_windows": {str(j): int(len(i)) for j, i in sess_idx.items()}, "chain": rows,
           "decomposition": decomposition, "noise_profile": noise_profile, "zero_shot": zero_shot,
           "loss_bins": list(E.LOSS_BINS)}

    if not a.no_attr:
        rng = np.random.default_rng(a.seed)
        attr_idx = {j: np.sort(rng.choice(i, min(len(i), a.attr_windows), replace=False)) for j, i in sess_idx.items()}
        abank = E.noise_bank(len(pool), a.T, C, a.R, model.chunk_size, seed=70_000)
        P = E.model_params(model)
        res["attribution"], res["block_numel"] = {}, P.block_numel()
        for k in (a.sessions or trained):
            nodes = E.session_path(runs, k, a.attr_every)
            targets = [j for j in range(k + 1) if j in attr_idx]
            E.log(f"attribution of session {k}: {len(nodes) - 1} segments, targets {targets}")
            res["attribution"][str(k)] = attribute_session(a, model, P, nodes, pool, abank, attr_idx, targets)
            E.dump_json(a.out_dir / "e4" / "trajectory.json", res)
    E.dump_json(a.out_dir / "e4" / "trajectory.json", res)
    figures(res, a.out_dir / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
