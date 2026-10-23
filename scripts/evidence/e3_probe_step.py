#!/usr/bin/env python3
"""
E3 -- direct test of the first-order forgetting analysis (Eq. 4, Eq. replay_forget, Proposition 1, Corollary 1) in
the regime it is stated for: one gradient step from theta_{k-1}.

At theta_{k-1}, with frozen-noise empirical losses L_j on the past training windows (the deterministic,
differentiable objects of the analysis), compute exact full gradients g_k (new session), g_j (each past session),
g_bar = g_bar_{<k} and, for every buffer selected by e2_gradients.py, g_B = sum_i w_i g_i. Then, for step sizes eta
given as relative step lengths of ||theta|| (default 1e-4 .. 3e-2):
  * no replay  measured L_j(theta - eta g_k) - L_j(theta)   vs  first order -eta <g_j, g_k>                (Eq. 4)
  * replay     measured F(theta^B) - F(theta^full)          vs  first order -eta lam <g_bar, g_B - g_bar>
               and Prop. 1's bound eta lam ||g_bar|| delta(B, w)                                      (Prop. 1)
  * remainder  |measured - first order| against eta (log-log slope 2 = the descent lemma's O(eta^2) term)
  * absolute   F(theta^B) against the first-order terms of Corollary 1
and the exact delta(B, w) of every buffer for this frozen-noise objective (ranked against E2's noise-averaged delta).

Needs e2_gradients.py output for the same boundaries and the same --T / --max_windows (buffer indices refer to it).

Usage (repo root):
  python scripts/evidence/e3_probe_step.py --config configs/cl/ldf_aistpp_genre5_s0.yaml --benchmark aistpp_genre5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev_common as E
import tmr_select as S


def session_losses(model, pool, bank, device, bs, sessions) -> dict:
    per = E.pool_losses(model, pool, bank, device, bs=bs)["per_window"]
    return {j: float(per[pool.session == j].mean()) for j in sessions}


def figures(res: dict, out: Path, k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = np.array(res["rel_steps"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    for j, r in res["no_replay"].items():
        m, p = np.array(r["measured"]), np.array(r["first_order"])
        ax[0].loglog(rel, np.abs(m), "o-", label=f"S{j} measured")
        ax[0].loglog(rel, np.abs(p), "--", color=ax[0].lines[-1].get_color(), label=f"S{j} $-\\eta\\langle g_j,g_k\\rangle$")
    ax[0].set(xlabel="relative step $\\eta\\|g\\|/\\|\\theta\\|$", ylabel="|$\\Delta L_j$|", title="Eq. 4 (no replay)")
    ax[0].legend(fontsize=6)
    i0 = int(np.argmin(np.abs(rel - res["reference_rel_step"])))
    names = list(res["replay"])
    meas = [res["replay"][n]["excess_measured"][i0] for n in names]
    bnd = [res["replay"][n]["prop1_bound"][i0] for n in names]
    ax[1].scatter(bnd, meas, s=25)
    for n, x, y in zip(names, bnd, meas):
        ax[1].annotate(n, (x, y), fontsize=6)
    lim = [min(min(bnd), min(meas), 0), max(max(bnd), max(meas)) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=.8)
    ax[1].set(xlabel="Prop. 1 bound $\\eta\\lambda\\|\\bar g\\|\\delta$", ylabel="measured excess forgetting",
              title=f"Prop. 1 at relative step {rel[i0]:g}")
    for n in names:
        r = res["replay"][n]
        ax[2].loglog(rel, np.maximum(np.abs(np.array(r["excess_measured"]) - np.array(r["excess_first_order"])), 1e-30),
                     lw=.8, alpha=.7)
    ax[2].loglog(rel, res["remainder_reference"], "k--", label="slope 2")
    ax[2].set(xlabel="relative step", ylabel="|measured - first order|", title="remainder $O(\\eta^2)$")
    ax[2].legend(fontsize=7)
    fig.suptitle(f"boundary k={k}")
    E.savefig(fig, out / f"e3_probe_k{k}")


def main() -> int:
    ap = E.common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--boundaries", type=int, nargs="*", default=None)
    ap.add_argument("--rel_steps", type=float, nargs="*", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--reference_rel_step", type=float, default=1e-3, help="step used for the Prop. 1 scatter")
    ap.add_argument("--lam", type=float, default=1.0, help="replay weight lambda_rep")
    ap.add_argument("--extra_random", type=int, default=4, help="additional random buffers besides E2's")
    ap.add_argument("--new_windows", type=int, default=None)
    ap.add_argument("--accum_device", default=None, help="where gradient accumulators live (default: --device)")
    a = E.finish_args(ap.parse_args())
    E.set_precision(not a.bf16_attention)
    acc = a.accum_device or a.device
    cfg = E.load_cfg(a.config)
    runs = E.runs_of(a, cfg)
    K = len(E.manifest(cfg, a.benchmark)["sessions"])
    ks = a.boundaries or [k for k in range(1, K) if k - 1 in runs and (a.out_dir / "e2" / f"boundary_k{k}.json").exists()]
    model = E.build_model(cfg, a.device)
    C = int(cfg.model.params.input_dim)

    for k in ks:
        e2 = json.loads((a.out_dir / "e2" / f"boundary_k{k}.json").read_text())
        past = list(range(k))
        pool = E.build_pool(cfg, a.benchmark, past, "train", a.T, max_per_session=a.max_windows, seed=a.seed)
        if pool.names != e2["pool_names"] or list(pool.start) != e2["pool_starts"]:
            raise SystemExit(f"k={k}: the past pool differs from e2's; rerun with the same --T / --max_windows / --seed")
        new = E.build_pool(cfg, a.benchmark, [k], "train", a.T, max_per_session=a.new_windows or a.max_windows,
                           seed=a.seed)
        ck = runs[k - 1] / "last.ckpt"
        E.load_weights(model, ck)
        P = E.model_params(model)
        theta0 = P.flat().clone()
        N, T = len(pool), a.T
        groups = pool.session
        bank = E.noise_bank(N, T, C, a.R, model.chunk_size, seed=40_000 + k)
        bankN = E.noise_bank(len(new), T, C, a.R, model.chunk_size, seed=50_000 + k)

        buffers = {n: (np.array(b["idx"], int), np.array(b["w"], float)) for n, b in e2["buffers"].items()}
        rng = np.random.default_rng(a.seed + 13 * k)
        for r in range(a.extra_random):
            i = S.random_quota(groups, {int(g): q for g, q in e2["quotas"].items()}, rng)
            buffers[f"random_{r + 2}"] = (i, S.uniform(i))
        member = {}
        for n, (idx, w) in buffers.items():
            for i, wi in zip(idx, w):
                member.setdefault(int(i), []).append((n, float(wi)))

        sess_sum = {j: torch.zeros(P.numel, device=acc) for j in past}
        buf_sum = {n: torch.zeros(P.numel, device=acc) for n in buffers}

        def cb(pos, i, g, l):
            g = g.to(acc)
            sess_sum[int(groups[i])].add_(g)
            for n, wi in member.get(int(i), []):
                buf_sum[n].add_(g, alpha=wi)

        E.log(f"k={k}: exact per-sample gradients over {N} past windows, {len(buffers)} buffers")
        E.per_sample_grads(model, P, pool, bank, a.device, range(N), cb)
        g_k, _ = E.mean_grad(model, P, new, bankN, a.device, range(len(new)), bs=max(1, a.bs // 2))
        g_k = g_k.to(acc)
        n_j = {j: int((groups == j).sum()) for j in past}
        g_j = {j: sess_sum[j] / n_j[j] for j in past}
        g_bar = sum(g_j[j] * (n_j[j] / N) for j in past)
        dot = lambda x, y: float(torch.dot(x.double(), y.double()))
        nb = math.sqrt(dot(g_bar, g_bar))
        delta = {n: math.sqrt(dot(g_bar - buf_sum[n], g_bar - buf_sum[n])) for n in buffers}

        L0 = session_losses(model, pool, bank, a.device, a.bs, past)
        pi = {j: n_j[j] / N for j in past}
        F = lambda L: sum(pi[j] * (L[j] - L0[j]) for j in past)
        base = float(theta0.norm()) / float((g_k + a.lam * g_bar).norm())

        def at(direction, eta):
            P.assign((theta0 - eta * direction.to(theta0.device)).to(theta0.dtype))
            return session_losses(model, pool, bank, a.device, a.bs, past)

        # numerical floor: a tiny step orthogonal to g_bar has no first-order effect on F and a negligible
        # second-order one, so what F reports there is fp32 rounding
        floor = 0.0
        for _ in range(2):
            xi = torch.randn_like(theta0)
            xi -= dot(xi.to(acc), g_bar) / nb ** 2 * g_bar.to(theta0.device)
            floor = max(floor, abs(F(at(xi, 1e-6 * float(theta0.norm()) / float(xi.norm())))))

        rel = list(a.rel_steps)
        no_rep = {j: {"measured": [], "first_order": []} for j in past}
        full_F, rep = [], {n: {"excess_measured": [], "excess_first_order": [], "prop1_bound": [], "F": [],
                               "F_first_order": [], "corollary_first_order_bound": []} for n in buffers}
        for rs in rel:
            eta = rs * base
            L = at(g_k, eta)
            for j in past:
                no_rep[j]["measured"].append(L[j] - L0[j])
                no_rep[j]["first_order"].append(-eta * dot(g_j[j], g_k))
            Ff = F(at(g_k + a.lam * g_bar, eta))
            full_F.append(Ff)
            for n in buffers:
                Fb = F(at(g_k + a.lam * buf_sum[n], eta))
                r = rep[n]
                r["F"].append(Fb)
                r["excess_measured"].append(Fb - Ff)
                r["excess_first_order"].append(-eta * a.lam * dot(g_bar, buf_sum[n] - g_bar))
                r["prop1_bound"].append(eta * a.lam * nb * delta[n])
                r["F_first_order"].append(-eta * dot(g_bar, g_k + a.lam * buf_sum[n]))
                r["corollary_first_order_bound"].append(eta * (-a.lam * nb ** 2 + nb * float(g_k.norm()) + a.lam * nb * delta[n]))
            E.log(f"k={k}: relative step {rs:g} done")
        P.assign(theta0)

        i0 = int(np.argmin(np.abs(np.array(rel) - a.reference_rel_step)))
        names = list(buffers)
        rel_err = lambda m, p: abs(m - p) / max(abs(p), 1e-30)
        summary = {
            "prop1_holds_all": {f"{rs:g}": bool(all(rep[n]["excess_measured"][i] <= rep[n]["prop1_bound"][i] + floor
                                                    for n in names)) for i, rs in enumerate(rel)},
            "spearman_excess_vs_delta": float(spearmanr([delta[n] for n in names],
                                                        [rep[n]["excess_measured"][i0] for n in names])[0]),
            # medians: a session or buffer with a tiny first-order term inflates a mean relative error
            "first_order_rel_error_no_replay": {f"{rs:g}": float(np.median([rel_err(no_rep[j]["measured"][i], no_rep[j]["first_order"][i])
                                                                            for j in past])) for i, rs in enumerate(rel)},
            "first_order_rel_error_replay": {f"{rs:g}": float(np.median([rel_err(rep[n]["excess_measured"][i], rep[n]["excess_first_order"][i])
                                                                         for n in names])) for i, rs in enumerate(rel)},
            "sign_agreement_no_replay": {f"{rs:g}": float(np.mean([np.sign(no_rep[j]["measured"][i]) == np.sign(no_rep[j]["first_order"][i])
                                                                   for j in past])) for i, rs in enumerate(rel)},
        }
        rem = [np.abs(np.array(rep[n]["excess_measured"]) - np.array(rep[n]["excess_first_order"])) for n in names]
        rem_med = np.median(np.stack(rem), 0)
        ok = rem_med > 10 * floor
        summary["remainder_slope"] = float(np.polyfit(np.log(np.array(rel)[ok]), np.log(rem_med[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")
        ref = rem_med[ok][0] * (np.array(rel) / np.array(rel)[ok][0]) ** 2 if ok.any() else np.full(len(rel), np.nan)
        # E2's delta is of the noise-averaged gradients (sketched), this one is exact for the frozen-noise objective
        e2_delta = {n: e2["rules"][n]["delta"] for n in names if n in e2["rules"]}
        if len(e2_delta) >= 3:
            summary["e2_vs_e3_delta_spearman"] = float(spearmanr([e2_delta[n] for n in e2_delta],
                                                                 [delta[n] for n in e2_delta])[0])

        res = {"boundary": k, "checkpoint": str(ck), "N": N, "n_new": len(new), "lam": a.lam, "rel_steps": rel,
               "base_eta": base, "reference_rel_step": rel[i0], "numerical_floor": floor,
               "norms": {"theta": float(theta0.norm()), "g_k": float(g_k.norm()), "g_bar": nb,
                         **{f"g_{j}": float(g_j[j].norm()) for j in past}},
               "cos_g_bar_g_k": dot(g_bar, g_k) / (nb * float(g_k.norm())),
               "delta_exact": delta, "L0": L0, "no_replay": no_rep, "full_replay_F": full_F, "replay": rep,
               "remainder_reference": ref, "summary": summary}
        E.dump_json(a.out_dir / "e3" / f"boundary_k{k}.json", res)
        figures(res, a.out_dir / "figures", k)
        E.log(f"k={k}: {summary}")
        del sess_sum, buf_sum, g_j, g_bar, g_k
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
