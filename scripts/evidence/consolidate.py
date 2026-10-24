#!/usr/bin/env python3
"""
Consolidate the evidence (e1-e5 outputs + the eval_cl_session.py results) into one place for the paper:
  <out_dir>/EVIDENCE.md            every claim with its numbers, a verdict, the figure and where it goes in the paper
  <out_dir>/evidence_tables.tex    booktabs tables, ready for \\input
  <out_dir>/evidence_summary.json  the headline numbers, machine-readable
Cross-analyses that need two sources are done here:
  * loss forgetting (E4) vs generated-dance forgetting (StyleRetention / FID_k from eval_cl_session.py)
  * forecasting forgetting: first-order interference at theta_{k-1} (E2) and the a-priori genre distance of the
    session design (manifest) vs the forgetting that training session k then caused
Every part is optional: whatever has been run is reported, the rest is listed as missing.

Usage (repo root):
  python scripts/evidence/consolidate.py --benchmark aistpp_genre5 --results_dir results/aistpp_genre5/vanilla
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev_common as E

RESULT_METRICS = {"StyleRetention": True, "DanceRate": True, "StyleAcc": True, "StyleFD": False, "FID_k": False,
                  "FID_g": False}
BLOCK_NAMES = {"audio_encoder": "audio encoder (music\\_embedding)", "cross_attn_kv": "cross-attn K/V (reads audio)",
               "cross_attn_qo": "cross-attn Q/O", "self_attn": "self-attention", "ffn": "FFN",
               "modulation": "adaLN modulation", "time_embed": "time embedding", "io": "in/out projections"}


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def load_dir(d: Path, pattern: str) -> dict[int, dict]:
    out = {}
    for p in sorted(d.glob(pattern)):
        m = re.search(r"_k(\d+)\.json$", p.name)
        if m:
            out[int(m.group(1))] = json.loads(p.read_text())
    return dict(sorted(out.items()))


def num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def nmean(xs) -> float | None:
    xs = [num(x) for x in xs]
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def f(x, nd: int = 3) -> str:
    if isinstance(x, (list, tuple)):
        return "/".join(f(v, nd) for v in x)
    x = num(x)
    if x is None:
        return "–"
    if x != 0 and (abs(x) < 10 ** (-nd) or abs(x) >= 1e5):
        return f"{x:.{max(1, nd - 1)}e}"
    return f"{x:.{nd}f}"


def pct(x, nd: int = 0) -> str:
    x = num(x)
    return "–" if x is None else f"{100 * x:.{nd}f}%"


def corr(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"n": int(ok.sum()), "spearman": None, "pearson": None}
    return {"n": int(ok.sum()), "spearman": float(spearmanr(x[ok], y[ok])[0]), "pearson": float(pearsonr(x[ok], y[ok])[0])}


def verdict(ok) -> str:
    return "**SUPPORTED**" if ok is True else "**NOT SUPPORTED**" if ok is False else "**INCONCLUSIVE**"


def results_matrix(rdir: Path | None, K: int) -> dict[str, np.ndarray]:
    out = {}
    if rdir is None or not rdir.exists():
        return out
    for p in rdir.glob("after_s*_sess*.json"):
        m = re.fullmatch(r"after_s(\d+)_sess(\d+)\.json", p.name)
        if not m:
            continue
        t, i = int(m.group(1)), int(m.group(2))
        e = json.loads(p.read_text())
        for name in RESULT_METRICS:
            if name in e and t < K and i < K:
                out.setdefault(name, np.full((K, K), np.nan))[i, t] = e[name]
    return out


class Report:
    def __init__(self):
        self.md, self.tex, self.summary, self.status = [], [], {}, []

    def h(self, s, level=2):
        self.md += ["", "#" * level + " " + s, ""]

    def p(self, *lines):
        self.md += list(lines)

    def table(self, header: list[str], rows: list[list]):
        self.md.append("| " + " | ".join(header) + " |")
        self.md.append("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            self.md.append("| " + " | ".join(str(c) for c in r) + " |")
        self.md.append("")

    def latex(self, label: str, caption: str, header: list[str], rows: list[list], align: str | None = None):
        align = align or "l" + "c" * (len(header) - 1)
        # escape _ and % outside math, leaving already-escaped ones alone
        esc = lambda s: str(s) if str(s).startswith("$") else re.sub(r"(?<!\\)([_%])", r"\\\1", str(s))
        self.tex += [f"% ---- {label} ----", "\\begin{table}[t]", "\\centering", "\\small",
                     f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\resizebox{\\linewidth}{!}{%",
                     f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(esc(h) for h in header) + "\\\\",
                     "\\midrule"]
        self.tex += [" & ".join(esc(c) for c in r) + "\\\\" for r in rows]
        self.tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]


# ----------------------------------------------------------------------------------------------
# A. Theory
# ----------------------------------------------------------------------------------------------
def theory_e1(R: Report, e1: dict):
    if not e1:
        R.status.append(("E1 spectral checks (Thm 1, Thm 2, Prop. fw, Lemma modal)", "missing"))
        return
    R.h("A1. Theorem 1 (Ky Fan optimality of the modal projector) and the BIC rank")
    rows, tex, ok_all = [], [], True
    for k, r in e1.items():
        sp, kf = r["spectrum"], r["ky_fan"]
        ok = kf["frac_random_beating_hat"] == 0 and kf["frac_top2M_random_beating_hat"] == 0 \
            and kf["identity_abs_err"] <= 1e-6 * max(kf["trace"], 1e-30)
        ok_all &= ok
        ratio = kf["Psi_random_min"] / kf["Psi_hat"] if kf["Psi_hat"] > 0 else None
        ratio2 = kf["Psi_top2M_random_min"] / kf["Psi_hat"] if kf["Psi_hat"] > 0 else None
        rows.append([k, r["N"], f"{sp['M_hat']} ({sp['M_hat_conservative_Neff_N']})", f(sp["r_eff"], 2),
                     pct(sp["explained_at_M_hat"], 1), f(sp["relative_gap_at_M_hat"]), f(kf["identity_abs_err"], 2),
                     f(ratio, 2), f(ratio2, 2), "yes" if ok else "NO"])
    R.p(f"{verdict(ok_all)}: at every boundary $\\Psi(\\hat P_k)$ equals the discarded spectral mass and no random "
        "rank-$\\hat M_k$ projector (uniform, or drawn inside the leading $2\\hat M_k$ modes) reaches it.")
    R.table(["k", "N windows", "M̂ (N_eff=N)", "r_eff", "variance kept", "rel. gap at M̂", "abs(Ψ̂ − tail)",
             "best random Ψ / Ψ̂", "best top-2M̂ random Ψ / Ψ̂", "Thm 1 holds"], rows)
    R.p("*Paper:* supp. Sec. app:thm1 right after the proof, and the BIC paragraph of Sec. 3.3 (rank actually selected).",
        "*Figure:* `figures/e1_theory_k*.pdf` panels (a) spectrum with $\\hat M_k$, (b) Ψ histogram.")
    R.summary["thm1_verified_all_boundaries"] = ok_all
    R.summary["M_hat_by_boundary"] = {k: r["spectrum"]["M_hat"] for k, r in e1.items()}

    R.h("A2. Theorem 2 (finite-sample rate) and the data-reuse caveat of Sec. app:setup")
    rows, slopes = [], []
    for k, r in e1.items():
        cs = r["consistency"]
        ho = cs.get("heldout", {})
        rat = ho["op_err"] / ho["op_err_expected_from_sampling"] if ho.get("op_err_expected_from_sampling") else None
        rat_s = ho["subspace_err"] / ho["subspace_err_expected_from_sampling"] if ho.get("subspace_err_expected_from_sampling") else None
        slopes.append(cs["slope_operator"])
        rows.append([k, f(cs["slope_operator"], 2), f(cs["slope_subspace"], 2), ho.get("N_test", "–"),
                     f(ho.get("op_err")), f(rat, 2), f(rat_s, 2)])
    ok = bool(np.all(np.abs(np.array(slopes) - 1) <= 0.25))
    R.p(f"{verdict(ok)} (rate): slope of the operator error against $\\sqrt{{1/N'-1/N}}$ is {f(slopes, 2)} "
        "(Theorem 2 predicts 1, i.e. $O(N^{-1/2})$). A held-out/expected ratio near 1 means the operator built on "
        "the training windows -- on which $\\theta_{k-1}$ was trained -- is no closer to them than a fresh sample is "
        "(the data-reuse case the theorem does not cover).")
    R.table(["k", "slope ‖C(N')−C‖", "slope subspace", "N held-out", "held-out op. error",
             "held-out / resampling (op.)", "held-out / resampling (subspace)"], rows)
    R.p("*Paper:* supp. Sec. app:thm2 (after the theorem) and Sec. app:setup's data-reuse remark. *Figure:* `e1_theory_k*` panel (c).")
    R.summary["thm2_slopes"] = slopes

    R.h("A3. Proposition fw (Frank-Wolfe rate of the OMP selector) and Lemma modal")
    rows, ok_fw, ok_lm, lit = [], True, True, []
    for k, r in e1.items():
        fw, lm = r["fw"], r["buffers"]["lemma"]
        ok_fw &= (num(fw["max_ratio_to_bound"]) or 0) <= 1 + 1e-9
        ok_lm &= bool(lm["tmr"]["holds"] and lm["random_all_hold"])
        lit.append(fw.get("literal_algorithm1_atoms"))
        rows.append([k, f"{fw['iterations']}/{fw['budget']}", f(fw["max_ratio_to_bound"]),
                     f(fw["empirical_rate_exponent"], 2), fw.get("literal_algorithm1_atoms", "–"),
                     f(lm["tmr"]["ratio"]), f(lm["random_ratio_mean"]), f(lm["random_ratio_max"]),
                     f(lm["tmr"]["in_subspace"] / max(lm["tmr"]["in_subspace_bound"], 1e-30))])
    R.p(f"Prop. fw bound {verdict(ok_fw)}; Lemma modal {verdict(ok_lm)}. The empirical OMP exponent is compared with "
        "the guaranteed $1/2$; the Lemma columns give $\\varepsilon_{\\mathcal B}$ / right-hand side (1 = tight) and the "
        "tightness of its in-subspace half, $\\|\\hat U^\\top\\Xi\\|_F/(\\sqrt{\\lambda_1}\\varepsilon^{modal})$.",
        "",
        f"**Algorithm 1 termination (fix the paper):** the literal augment filter $\\langle z_i,u\\rangle>0$ selected "
        f"only {f(lit, 0)} atoms per boundary against budgets {[r['fw']['budget'] for r in e1.values()]}. On the "
        "simplex an atom lowers the error iff $\\langle z_i-Z_{\\mathcal S}w_{\\mathcal S},u\\rangle>0$ (the Frank-Wolfe "
        "gap; KKT gives $\\langle z_j,u\\rangle=\\langle Z_{\\mathcal S}w_{\\mathcal S},u\\rangle$ on the support), and "
        "with uncentred whitened atoms sharing a large common component every raw correlation is negative after the "
        "first pick. Both tests choose the same atom whenever they continue, so Prop. fw (i) is unchanged; Algorithm 1's "
        "$\\mathcal Q$ should read $\\{i\\notin\\mathcal S:\\langle z_i-Z_{\\mathcal S}w_{\\mathcal S},u\\rangle>0,\\dots\\}$. "
        "All buffers here use the corrected test.")
    R.table(["k", "OMP atoms / budget", "max ε_mod / bound", "fitted exponent p (ε ~ t^-p)", "literal Alg. 1 atoms",
             "Lemma ratio TMR", "Lemma ratio random (mean)", "(max)", "in-subspace tightness"], rows)
    R.p("*Paper:* Algorithm 1 (termination test), supp. Sec. app:fw (after Remark fw_scope) and after Lemma modal. "
        "*Figure:* `e1_theory_k*` panel (d).")
    R.summary["literal_algorithm1_atoms"] = lit
    R.summary.update(fw_bound_holds=ok_fw, lemma_holds=ok_lm)
    R.latex("tab:ev_theory", "Empirical checks of the analysis on DiscoForcing (AIST++ genre sessions), per session "
            "boundary $k$: BIC rank, effective rank, retained variance, Theorem~1 margin over the best random projector, "
            "Theorem~2 rate (slope 1 = $O(N^{-1/2})$), worst ratio of OMP's modal error to the Frank--Wolfe bound and "
            "fitted decay exponent, and Lemma~1 tightness for the TMR buffer.",
            ["$k$", "$\\hat M_k$", "$r_{\\mathrm{eff}}$", "var. kept", "Thm.1 margin", "Thm.2 slope",
             "FW max ratio", "FW exponent", "Lemma ratio"],
            [[k, r["spectrum"]["M_hat"], f(r["spectrum"]["r_eff"], 2), pct(r["spectrum"]["explained_at_M_hat"], 1),
              f(r["ky_fan"]["Psi_random_min"] / r["ky_fan"]["Psi_hat"] if r["ky_fan"]["Psi_hat"] > 0 else None, 2) + "$\\times$",
              f(r["consistency"]["slope_operator"], 2), f(r["fw"]["max_ratio_to_bound"], 2),
              f(r["fw"]["empirical_rate_exponent"], 2), f(r["buffers"]["lemma"]["tmr"]["ratio"], 2)] for k, r in e1.items()])
    R.status.append(("E1 spectral checks", f"boundaries {list(e1)}"))


def theory_e2(R: Report, e2: dict):
    if not e2:
        R.status.append(("E2 gradient checks (Prop. 1 delta, kappa, calibration, anatomy)", "missing"))
        return
    R.h("A4. Proposition 1: the buffer gradient error δ of each selection rule (exact training-free comparison)")
    names = [n for n in next(iter(e2.values()))["rules"] if n != "random_mean"]
    sdiv = lambda x, y: x / y if num(x) is not None and num(y) else None
    rows, tex_rows = [], []
    for n in names:
        per = [e["rules"][n] for e in e2.values() if n in e["rules"]]
        rel = [r.get("delta_rel_random") for r in per]
        kap = [sdiv(e["rules"][n]["kappa"], e["rules"]["random_mean"]["kappa"]) for e in e2.values() if n in e["rules"]]
        gap = [r.get("oracle_gap_closed") for r in per]
        sec = [r.get("seconds") for r in per]
        pctl = [r.get("delta_percentile_among_random") for r in per]
        rows.append([n, f(rel, 2), f(nmean(rel), 2), f(pctl, 2), f(gap, 2), f(kap, 2), f(nmean(sec), 1)])
        tex_rows.append([n, f(nmean(rel), 2), f(nmean(gap), 2), f(nmean(kap), 2), f(nmean(sec), 1)])
    tmr_rel = nmean([e["rules"]["tmr"].get("delta_rel_random") for e in e2.values()])
    noise = [e["calibration"].get("noise_share_of_naive_delta2") for e in e2.values()]
    ok = None if tmr_rel is None else bool(tmr_rel < 1)
    R.p(f"{verdict(ok)} (TMR lowers δ below random replay): mean δ(TMR)/δ(ER) = {f(tmr_rel, 3)}. δ is the error of "
        "the noise-averaged per-sample gradients, estimated without Monte-Carlo bias from two independent halves of "
        "the diffusion-noise draws, and with draws and a sketch independent of those the gradient-based rules select "
        "on, so the gradient oracle (`grad_omp`, GCR/GradMatch in parameter space) is not favoured by reusing its own "
        "noise. 'gap closed' = (δ_ER − δ_rule)/(δ_ER − δ_oracle); 'percentile' = share of random buffers with lower δ.",
        f"Monte-Carlo noise was {'/'.join(pct(x) for x in noise)} of the naive δ² per boundary (removed by the "
        "estimator); above ~80% the per-sample gradient signal is weak and the comparisons should be read as rankings "
        "only (raise --R_eval).")
    R.summary["mc_noise_share"] = noise
    snr = [e.get("gradient_snr", {}) for e in e2.values()]
    if any(snr):
        R.p("", f"**Diffusion noise limits gradient-space selection.** Per training window, the signal that "
            "distinguishes it -- the deviation of its expected gradient from the pool mean -- against the gradient noise "
            f"of a single diffusion draw: median SNR {f([s.get('median_signal_to_noise_one_draw') for s in snr], 3)}; "
            f"{pct(nmean([s.get('share_windows_signal_below_one_draw_noise') for s in snr]))} of windows are "
            f"below one draw's noise, and a median window needs {f([s.get('draws_for_snr_1_median') for s in snr], 1)} "
            "draws to reach SNR 1. Gradient-matching coresets (GCR, the oracle above) select on such estimates, while "
            "TMR's trajectories φ_i do not depend on the diffusion noise at all: a cost *and* variance argument for "
            "selecting in audio-trajectory space (supp. Sec. scope, point (i)).")
        R.summary["gradient_snr"] = snr
    R.table(["rule", "δ / δ_ER per k", "mean", "percentile among ER", "oracle gap closed", "κ / κ_ER", "select s"], rows)
    R.p("*Paper:* Sec. 4.4 (new paragraph 'Does TMR approximate the right gradient?') or supp. after Prop. 1 -- this is "
        "the measurable quantity the paper says Prop. 1 is stated in. *Figure:* `e2_kappa_k*` panels (b, c).")
    R.latex("tab:ev_delta", "Training-free check of Proposition~1 on DiscoForcing: gradient error $\\delta(\\mathcal B,w)$ "
            "of each selection rule relative to random replay (ER), share of the ER-to-oracle gap closed (oracle = OMP "
            "on per-sample parameter gradients, i.e. gradient matching), modulus $\\kappa$ relative to ER, and "
            "selection time. Averaged over session boundaries.",
            ["rule", "$\\delta/\\delta_{\\mathrm{ER}}$", "gap closed", "$\\kappa/\\kappa_{\\mathrm{ER}}$", "time (s)"], tex_rows)
    R.summary["delta_rel_random"] = {n: nmean([e["rules"][n].get("delta_rel_random") for e in e2.values()
                                               if n in e["rules"]]) for n in names}

    R.h("A5. Why the temporal axis? Which selection space certifies a small gradient error (κ calibration)")
    spaces = list(next(iter(e2.values()))["calibration"]["spaces"])
    rows = []
    for s in spaces:
        sp = [e["calibration"]["spaces"][s]["spearman"] for e in e2.values() if s in e["calibration"]["spaces"]]
        cv = [e["calibration"]["spaces"][s]["kappa_cv"] for e in e2.values() if s in e["calibration"]["spaces"]]
        rows.append([s, f(sp, 2), f(nmean(sp), 3), f(nmean(cv), 3)])
    order = sorted(rows[:-1] if "noise ceiling" in rows[-1][0] else rows, key=lambda r: -float(r[2]) if r[2] != "–" else 0)
    tmr_rank = [r[0] for r in order].index("temporal whitened (TMR)") + 1 if any(r[0] == "temporal whitened (TMR)" for r in order) else None
    R.p(f"Over {next(iter(e2.values()))['calibration']['n_buffers']} random and metadata-tilted buffers per boundary, "
        "Spearman ρ between the buffer's selection error in each space and its gradient error δ (Prop. 1's quantity). "
        "The space with the highest ρ and the lowest κ coefficient of variation is the one in which minimising the "
        "selection error minimises the forgetting bound -- the paper's open question 'the analysis is symmetric in the "
        f"temporal and channel axes' answered empirically. TMR's space ranks **{tmr_rank}** of {len(order)}. The "
        "gradient-sketch row (independent noise) is the ceiling set by Monte-Carlo noise in δ.")
    R.table(["selection space", "ρ(ε, δ) per k", "mean ρ", "κ coefficient of variation"], rows)
    R.latex("tab:ev_calibration", "Which selection space tracks Proposition~1's gradient error: Spearman correlation "
            "between a buffer's selection error in each space and its gradient error $\\delta$, over random and "
            "metadata-tilted buffers, and the coefficient of variation of the modulus $\\kappa=\\delta/\\varepsilon$ "
            "(lower = a more stable exchange rate). Mean over session boundaries.",
            ["selection space", "Spearman $\\rho$", "$\\mathrm{CV}(\\kappa)$"], [[r[0], r[2], r[3]] for r in rows])
    R.p("*Paper:* Sec. 4.3 after the ablation (the ablation's 'temporal axis is operative' finding gets a mechanism), "
        "and supp. Sec. scope point (ii). *Figure:* `e2_kappa_k*` panel (a).")
    R.summary["calibration_mean_spearman"] = {r[0]: num(r[2]) for r in rows}
    R.summary["tmr_space_rank"] = tmr_rank
    R.status.append(("E2 gradient checks", f"boundaries {list(e2)}"))


def theory_e3(R: Report, e3: dict):
    if not e3:
        R.status.append(("E3 probe-step test of Eq. 4 / Prop. 1 / Cor. 1", "missing"))
        return
    R.h("A6. Direct test of Eq. 4, Proposition 1 and Corollary 1 in their regime (one exact gradient step)")
    rows, ok_all, sp_all = [], True, []
    for k, r in e3.items():
        s = r["summary"]
        x0 = f"{r['rel_steps'][0]:g}"
        # Prop. 1 is 'first-order term + C eta^2': check the first-order term where eta^2 is negligible (smallest
        # step) and that the remainder grows like eta^2
        holds0 = bool(s["prop1_holds_all"][x0])
        slope_ok = num(s["remainder_slope"]) is not None and abs(s["remainder_slope"] - 2) <= 0.5
        ok_all &= holds0 and slope_ok
        sp_all.append(s["spearman_excess_vs_delta"])
        rows.append([k, "yes" if holds0 else "NO", ", ".join(f"{x}:{'y' if v else 'n'}" for x, v in s["prop1_holds_all"].items()),
                     pct(s.get("first_order_rel_error_replay", {}).get(x0), 1), f(s["spearman_excess_vs_delta"], 2),
                     pct(s["first_order_rel_error_no_replay"][x0], 1), pct(s["sign_agreement_no_replay"][x0]),
                     f(s["remainder_slope"], 2), f(s.get("e2_vs_e3_delta_spearman"), 2), f(r["numerical_floor"], 2)])
    R.p(f"{verdict(ok_all)}: at the smallest step every buffer's measured excess forgetting F(θ^B) − F(θ^full) is "
        "below Prop. 1's first-order bound ηλ‖ḡ‖δ, and the gap to the first-order prediction grows with the slope "
        "shown (2 = the descent lemma's Cη² term, which is why larger steps can exceed the first-order bound alone). "
        f"Across buffers the measured excess forgetting rank-correlates with δ at ρ = {f(sp_all, 2)}. With no replay, "
        "−η⟨g_j, g_k⟩ (Eq. 4) predicts each old session's loss change with the median relative error and sign "
        "agreement shown at the smallest step.")
    R.table(["k", "Prop. 1 at smallest step", "holds per step (first-order term only)",
             "replay: rel. error vs first order", "ρ(excess, δ)", "Eq. 4 rel. error", "Eq. 4 sign agreement",
             "remainder slope", "ρ(E2 δ, frozen-noise δ)", "fp32 floor"], rows)
    R.p("*Paper:* supp. Sec. app:thm1, right after Prop. 1's proof ('Numerical check'), and one sentence in Sec. 3.4. "
        "*Figure:* `e3_probe_k*.pdf`.")
    R.summary["prop1_bound_holds_small_steps"] = ok_all
    R.summary["prop1_spearman_excess_delta"] = sp_all
    R.status.append(("E3 probe-step test", f"boundaries {list(e3)}"))


# ----------------------------------------------------------------------------------------------
# B. Mechanisms
# ----------------------------------------------------------------------------------------------
def mechanisms(R: Report, e1: dict, e2: dict, e4: dict | None):
    if e4:
        R.h("B1. What is forgotten: motion prior or audio grounding?")
        dec = e4["decomposition"]
        rows, shares = [], []
        for key, d in dec.items():
            share = -d["dG"] / d["dL_cond"] if d["dL_cond"] > 0 else None
            if share is not None:
                shares.append(share)
            # retention of G is only meaningful if the model used the music when it learned the session
            grounded = d["G_at_learn"] > 0.01 * abs(d["L_cond_at_learn"])
            rows.append([key, f(d["dL_cond"], 4), f(d["dL_null"], 4), f(-d["dG"], 4), pct(share),
                         f(d["G_retention"], 2) if grounded else "n/a (G≈0 at learn time)"])
        med = float(np.median(shares)) if shares else None
        R.p("Exact identity on held-out windows: ΔL^cond = ΔL^null − ΔG, with G = L^null − L^cond the loss the model "
            "saves by listening to the music (its conditional-information term). 'grounding share' = −ΔG/ΔL^cond: the "
            f"part of forgetting that is lost audio grounding rather than motion-prior drift. Median share: **{pct(med)}**.",
            "If the share is large, forgetting in this audio-conditioned generator is mostly a failure to *use old "
            "music*, which is exactly what replay selected on audio-encoder trajectories targets (assumption A1).")
        R.table(["old session @ after", "ΔL^cond (forgetting)", "ΔL^null (prior drift)", "−ΔG (lost grounding)",
                 "grounding share", "G retention G(θ_t)/G(θ_j)"], rows)
        R.latex("tab:ev_decomp", "What vanilla fine-tuning forgets on DiscoForcing: the increase of an old session's "
                "denoising loss splits exactly into drift of the unconditional motion prior ($\\Delta L^{\\mathrm{null}}$) "
                "and lost audio grounding ($-\\Delta G$, $G=L^{\\mathrm{null}}-L^{\\mathrm{cond}}$).",
                ["old session / after", "$\\Delta L^{\\mathrm{cond}}$", "$\\Delta L^{\\mathrm{null}}$", "$-\\Delta G$",
                 "grounding share", "$G$ retained"],
                [[r[0].replace("@", " / ").replace("_", " ")] + r[1:5] + [r[5] if not r[5].startswith("n/a") else "n/a"]
                 for r in rows])
        R.p("*Paper:* Sec. 1 (diagnosis paragraph next to Fig. 1) and Sec. 4.4 -- this is the strongest new finding "
            "candidate. *Figure:* `e4_trajectory.pdf` panels (b, c).")
        R.summary["median_grounding_share"] = med

        prof = e4.get("noise_profile", {})
        if prof:
            R.h("B2. At which diffusion noise levels is it forgotten?")
            bins = e4["loss_bins"]
            labels = [f"[{lo:.1f},{min(hi, 1):.1f}]" for lo, hi in zip(bins[:-1], bins[1:])]
            rows = [[key] + [f(v, 4) for v in p["dcond"]] for key, p in prof.items()]
            arg = [labels[int(np.nanargmax(p["dcond"]))] for p in prof.values() if np.isfinite(np.nanmax(p["dcond"]))]
            R.p("Per-token loss increase of old sessions, binned by the token's diffusion-forcing noise level "
                f"(0 = clean history, 1 = pure noise). Peak forgetting bins: {', '.join(arg)}. High-noise bins hold the "
                "coarse, music-dependent structure (genre), low-noise bins the fine detail.")
            R.table(["old session @ after"] + labels, rows)
            R.p("*Paper:* Sec. 4.4 or supp. (diffusion-specific insight). *Figure:* `e4_noise_profile.pdf`.")

        att = e4.get("attribution", {})
        if att:
            R.h("B3. Where it is forgotten: path-integral attribution to parameter blocks")
            numel = e4.get("block_numel", {})
            tot_p = sum(numel.values()) or 1
            agg, agg_null, moved, qerr = {}, {}, {}, []
            for k, A in att.items():
                for key, r in A["result"].items():
                    j, c = key.split("/")
                    if int(j) >= int(k):
                        continue  # learning of the new session is reported separately
                    tot = r["simpson"]  # shares of the integrated change sum to 1; completeness error reported
                    if r["actual"] <= 0 or tot <= 0:
                        continue
                    for b, v in r["blocks"].items():
                        (agg if c == "cond" else agg_null).setdefault(b, []).append(v / tot)
                    qerr.append(r["quadrature_rel_err"])
                for b, v in A["block_distance_moved"].items():
                    moved.setdefault(b, []).append(v)
            learn = {}
            for k, A in att.items():
                r = A["result"].get(f"{k}/cond")
                if r and r["actual"] < 0 and r["simpson"] < 0:
                    for b, v in r["blocks"].items():
                        learn.setdefault(b, []).append(v / r["simpson"])
            mv_tot = {b: np.mean(v) for b, v in moved.items()}
            mv_sum = sum(mv_tot.values()) or 1
            rows = []
            for b in E.BLOCKS:
                if b not in agg and b not in numel:
                    continue
                rows.append([BLOCK_NAMES.get(b, b), pct(numel.get(b, 0) / tot_p, 1), pct(mv_tot.get(b, 0) / mv_sum, 1),
                             pct(np.mean(agg[b]) if b in agg else None, 1),
                             pct(np.mean(agg_null[b]) if b in agg_null else None, 1),
                             pct(np.mean(learn[b]) if b in learn else None, 1)])
            R.p("Share of old-session forgetting (ΔL^cond, and ΔL^null when attributed) caused by each block's movement "
                "during the new session, from the exact path integral of the loss gradient along the checkpoint path "
                f"(Simpson; median quadrature error {pct(np.median(qerr) if qerr else None, 1)} of ΔL). Compare with the "
                "block's share of parameters and of the distance travelled; 'learning' is the same split for the new "
                "session's loss decrease. Forgetting concentrated in audio-reading blocks (audio encoder, cross-attn K/V) "
                "beyond their parameter share says the conditioning pathway is where old music stops being understood.")
            R.table(["block", "share of params", "share of distance moved", "share of forgetting ΔL^cond",
                     "share of ΔL^null", "share of learning (new session)"], rows)
            R.latex("tab:ev_attr", "Where forgetting happens in the DiscoForcing generator: share of the old sessions' "
                    "loss increase attributed to each block (exact path integral along the checkpoint path), against "
                    "the block's share of parameters and of parameter movement, and its share of the new session's "
                    "loss decrease.",
                    ["block", "params", "moved", "forgetting", "learning"],
                    [[r[0], r[1], r[2], r[3], r[5]] for r in rows])
            R.p("*Paper:* Sec. 4.4 ('Where does forgetting live?'); motivates A1 empirically. *Figure:* `e4_attribution.pdf`.")
            R.summary["forgetting_share_by_block"] = {b: float(np.mean(v)) for b, v in agg.items()}

        R.h("B4. When it is forgotten: dynamics inside a session")
        chain = e4["chain"]
        rows = []
        for k in e4["trained_sessions"][1:]:
            seg = [c for c in chain if c["session_training"] == k]
            start = next((c for c in chain if c["label"] == f"S{k - 1}:end"), None)
            if not seg or start is None:
                continue
            for j in range(k):
                l0 = start["losses"][str(j)]["cond"]
                dl = [c["losses"][str(j)]["cond"] - l0 for c in seg]
                if dl[-1] <= 0.02 * abs(l0):  # fractions of a ~0 net change are meaningless
                    rows.append([f"S{j} during S{k}", f(dl[-1], 4), "no net forgetting" if dl[-1] <= 0
                                 else "negligible net forgetting", "–", f(max(dl) / max(abs(l0), 1e-30), 2) + " of L"])
                    continue
                frac = [d / dl[-1] for d in dl]
                first = next((c["label"].split(":")[1] for c, x in zip(seg, frac) if x >= 0.5), "–")
                rows.append([f"S{j} during S{k}", f(dl[-1], 4), pct(frac[0]), first, f(max(frac), 2)])
        R.p("Fraction of the session's final forgetting already present at its first stored checkpoint, the first "
            "checkpoint by which half of it has happened, and the peak relative to the end (>1 = transient overshoot, "
            "the 'stability gap'). Front-loaded forgetting argues for replay from the first step of a session.")
        R.table(["old session", "final ΔL^cond", "at first checkpoint", "half reached at step", "peak / final"], rows)
        R.p("*Paper:* supp. (long-horizon section) or Sec. 4.4. *Figure:* `e4_trajectory.pdf` panel (a).")

    if e2:
        R.h("B5. Gradient anatomy at the session boundary (Eq. 4 per block)")
        rows = []
        for k, e in e2.items():
            for j, a in e["anatomy"].items():
                blocks = a["blocks"]
                worst = min(blocks, key=lambda b: blocks[b]["cos"])
                tot = a["interference_gd"]
                kv = sum(blocks[b]["interference_gd"] for b in ("audio_encoder", "cross_attn_kv") if b in blocks)
                rows.append([f"S{j} vs S{k}", f(a["cos"]), f(tot, 3), f(a["interference_adam"], 3),
                             f"{worst} ({f(blocks[worst]['cos'])})", pct(kv / tot if tot else None)])
        R.p("cos(g_j, g_k) of an old and the new session's mean gradients at θ_{k−1}; −⟨g_j, g_k⟩ > 0 means the new "
            "session's descent direction raises the old loss to first order (Eq. 4), also with an Adam-like "
            "preconditioner; the most conflicting block, and the share of the interference carried by the audio "
            "pathway (audio encoder + cross-attn K/V).")
        R.table(["pair", "cos", "−⟨g_j,g_k⟩", "Adam-preconditioned", "most conflicting block (cos)", "audio-pathway share"], rows)
        R.p("*Figure:* `e2_anatomy_k*.pdf`.")

    if e1:
        R.h("B6. What the temporal modes are (interpretation of C_k)")
        rows = []
        for k, r in e1.items():
            mi, dc = r["modes"], r["decode"]
            st = r.get("stability", {})
            rows.append([k, f(mi["toeplitz_index"]), f(mi["mean_dct_alignment_top_M"]), f(mi["szego_tv_distance"]),
                         ("window-average, " if mi.get("mode1_is_window_average") else "") + pct(mi.get("mode1_share"), 1),
                         f(mi.get("whitening_gain_dynamic_vs_mode1"), 1),
                         f(mi["max_freq_in_M_hat_hz"], 2), f(dc.get("temporal_modes", {}).get("tempo_r2"), 2),
                         f(dc.get("channel_modes", {}).get("tempo_r2"), 2), f(dc.get("time_pooled_pca", {}).get("tempo_r2"), 2),
                         f(dc.get("temporal_modes", {}).get("genre_acc"), 2), f(dc.get("channel_modes", {}).get("genre_acc"), 2),
                         f(st.get("theta_k", {}).get("subspace_overlap"), 3), f(st.get("theta_k", {}).get("cka_phi"), 3)])
        R.p("A small Toeplitz index, DCT alignment near 1 and a small Szego distance mean the audio-conditioned "
            "trajectories are stationary over a window, so the temporal modes are (near-)Fourier modes and the "
            "eigenvalues are the power spectrum: TMR's projector is a data-adaptive low-pass band of the encoder "
            "dynamics and whitening equalises the retained frequency bands. Decoding columns: cross-validated on unseen "
            "music pieces, from equal-size summaries. Stability: overlap of the modal subspace at θ_{k−1} and θ_k on the "
            "same data (1 = unchanged) and CKA of φ -- a selection made at θ_{k−1} stays valid while session k trains. "
            "When mode 1 is the window average (constant over time) and carries most of the variance, windows differ "
            "mainly in static audio content (song, genre) and the later modes hold the within-window dynamics; "
            "whitening multiplies the weakest retained mode by the 'whitening gain' relative to mode 1, which is the "
            "concrete form of 'balancing dominant and weak temporal modes'.")
        R.table(["k", "Toeplitz idx", "DCT align (top M̂)", "Szego TV", "mode 1 (share)", "whitening gain",
                 "max mode freq (Hz)", "tempo R² temporal",
                 "tempo R² channel", "tempo R² pooled", "genre acc temporal", "genre acc channel",
                 "subspace overlap θ_{k−1}→θ_k", "CKA φ"], rows)
        src = next(iter(e1.values()))["decode"].get("tempo_source")
        R.p(f"Tempo source: {src}. *Paper:* Sec. 3.3 after Eq. operator ('what are temporal modes'), supp. otherwise. "
            "*Figure:* `e1_modes_k*.pdf`.")

        R.h("B7. Buffer composition (Fig. 3c analogue on DiscoForcing)")
        k_last = max(e1)
        rr = e1[k_last]["buffers"]["rules"]
        keys = [x for x in ("grid_coverage", "music_coverage", "music_entropy", "dancer_coverage") if x in rr["random_mean"]]
        rows = [[n] + [f(rr[n].get(x)) for x in keys] + [f(rr[n].get("eps_B") / rr["random_mean"]["eps_B"], 2),
                                                          rr[n].get("n_nonzero", "–")]
                for n in rr if n not in ("random_std",)]
        R.table(["rule"] + keys + ["ε_B / ER", "non-zero weights"], rows)
        R.p(f"(boundary k={k_last}). *Paper:* Sec. 4.4 coverage paragraph.")


# ----------------------------------------------------------------------------------------------
# C. Behaviour
# ----------------------------------------------------------------------------------------------
def behaviour(R: Report, e5: dict | None):
    if not e5:
        R.status.append(("E5 behaviour (style flow, recency, rhythm)", "missing"))
        return
    R.h("C1. Recency attractor and the collapse of conditional style information")
    rows = []
    for t, r in sorted(e5["per_t"].items(), key=lambda x: int(x[0])):
        if r.get("recency_bias_index") is None:
            continue  # only one music session seen: nothing old to forget
        rows.append([t, f(r.get("old_correct_mass")), f(r.get("recency_bias_index")), f(r.get("gt_recency_bias_index")),
                     f(r.get("old_intermediate_mass")), f(r.get("old_other_mass")), f(r.get("style_information_bits")),
                     f(r.get("gt_style_information_bits")), f(r.get("max_information_bits"))])
    R.summary["recency"] = {t: {k: r.get(k) for k in ("old_correct_mass", "recency_bias_index", "gt_recency_bias_index",
                                                       "style_information_bits", "gt_style_information_bits",
                                                       "max_information_bits")}
                            for t, r in e5["per_t"].items() if r.get("recency_bias_index") is not None}
    R.p("For old-session music: share of generated windows danced in the right genre, in the newest session's genres "
        "(recency bias index, RBI; GT = the encoder's confusion on real dance), in intermediate sessions' genres, and "
        "not as dance at all. I(music session; danced style) in bits over all sessions seen so far: it falls to 0 when "
        "what is danced no longer depends on which session's music plays.")
    R.table(["after S_t", "old: right genre", "old: newest genre (RBI)", "RBI on real dance", "old: intermediate",
             "old: not dance", "I(music; style) bits", "real dance bits", "max bits"], rows)
    R.latex("tab:ev_behaviour", "Generated dances of the vanilla model after each session, for old-session music: "
            "share in the right genre, recency bias (share in the newest session's genres; real-dance confusion in "
            "parentheses) and the information the danced style carries about which session's music plays.",
            ["after", "right genre", "newest genre", "I(music;style) [bits]"],
            [[f"S{t}", f(r.get("old_correct_mass"), 2), f"{f(r.get('recency_bias_index'), 2)} ({f(r.get('gt_recency_bias_index'), 2)})",
              f"{f(r.get('style_information_bits'), 2)} / {f(r.get('max_information_bits'), 2)}"]
             for t, r in sorted(e5["per_t"].items(), key=lambda x: int(x[0])) if r.get("recency_bias_index") is not None])
    R.p("*Paper:* Sec. 1 diagnosis (forgetting = the generator dances its latest style to every song) and Sec. 4.4. "
        "*Figure:* `e5_style_flow.pdf`.")

    s = e5.get("summary", {})
    d = num(s.get("rescoring_max_abs_diff_vs_reported"))
    if d is not None:
        R.p(f"Rescoring check: own-genre mass recomputed here differs from the StyleRetention reported by "
            f"eval_cl_session.py by at most {f(d)}" + (" -- OK." if d <= 0.02 else
                                                         " -- **too large: check that the same style encoder is used.**"))
    if s.get("BAS_old_vs_learned") is not None or s.get("style_retention_old_vs_learned") is not None:
        R.h("C2. Rhythm survives, style does not")
        R.p(f"Old sessions vs sessions just learned: StyleRetention {f(s.get('style_retention_old_vs_learned'))}; BAS "
            f"{f(s.get('BAS_old_vs_learned'))}. Relative to real dance, old sessions keep {pct(s.get('BAS_old_rel_gt'))} "
            f"of the beat alignment but only {pct(s.get('style_old_rel_gt'))} of the style. If the first is near 100% "
            "and the second far below, forgetting is hierarchical: music-to-motion *timing* (shared by every genre, "
            "aligned gradients) is kept and genre *vocabulary* (session-specific, conflicting gradients) is lost.")
        R.p("*Paper:* Sec. 4.4. *Figure:* `e5_rhythm_vs_style.pdf`.")
        R.summary["rhythm_vs_style"] = s
    R.status.append(("E5 behaviour", f"after S{sorted(e5['per_t'], key=int)}"))


# ----------------------------------------------------------------------------------------------
# D. Cross-analyses
# ----------------------------------------------------------------------------------------------
def session_distance(man: dict, j: int, k: int) -> float | None:
    d = man.get("genre_distance", {})
    gj, gk = man["sessions"][j]["groups"], man["sessions"][k]["groups"]
    vals = [d.get(f"{min(a, b)}-{max(a, b)}") for a in gj for b in gk if a != b]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def cross(R: Report, e2: dict, e4: dict | None, res: dict, man: dict | None, K: int):
    R.h("D1. Does the loss the theory bounds track the forgetting the paper reports?")
    if not e4 or "StyleRetention" not in res:
        R.p("Needs E4 and the eval_cl_session.py results (`--results_dir`).")
    else:
        dec, SR = e4["decomposition"], res["StyleRetention"]
        FK = res.get("FID_k")
        xs, ys, zs, rows = [], [], [], []
        for key, d in dec.items():
            m = re.fullmatch(r"S(\d+)@after_S(\d+)", key)
            j, t = int(m.group(1)), int(m.group(2))
            ds = SR[j, j] - SR[j, t]
            dfk = FK[j, t] - FK[j, j] if FK is not None else np.nan
            xs.append(d["dL_cond"])
            ys.append(ds)
            zs.append(dfk)
            rows.append([key, f(d["dL_cond"], 4), f(-d["dG"], 4), f(ds), f(dfk, 2)])
        c1, c2 = corr(xs, ys), corr(xs, zs)
        cg = corr([-dec[k]["dG"] for k in dec], ys)
        R.p(f"Across all (old session, later checkpoint) pairs: Spearman(ΔL^cond, StyleRetention drop) = "
            f"{f(c1['spearman'], 2)} (n={c1['n']}), Spearman(ΔL^cond, ΔFID_k) = {f(c2['spearman'], 2)}, and "
            f"Spearman(lost grounding −ΔG, StyleRetention drop) = {f(cg['spearman'], 2)}. A strong positive value "
            "closes the gap between the analysis (which bounds the training loss) and the reported metrics.")
        R.table(["pair", "ΔL^cond", "−ΔG", "StyleRetention drop", "ΔFID_k"], rows)
        R.summary["loss_vs_style_spearman"] = c1["spearman"]
        R.summary["grounding_vs_style_spearman"] = cg["spearman"]
        R.p("*Paper:* Sec. 3.2 (after Eq. 4: 'the loss increase it bounds is what FGD/FID register') -- one sentence + supp. table.")

    R.h("D2. Forecasting forgetting before training a session")
    if not e2 or not e4:
        R.p("Needs E2 and E4.")
        return
    ends = {int(c["label"].split(":")[0][1:]): c for c in e4["chain"] if c["label"].endswith(":end")}
    SR = res.get("StyleRetention")
    rows, P = [], {"interference": [], "adam": [], "neg_cos": [], "design_distance": [], "dL": [], "dSR": []}
    for k, e in e2.items():
        if k not in ends or k - 1 not in ends:
            continue
        for j, a in e["anatomy"].items():
            j = int(j)
            dl = ends[k]["losses"][str(j)]["cond"] - ends[k - 1]["losses"][str(j)]["cond"]
            dsr = SR[j, k - 1] - SR[j, k] if SR is not None else np.nan
            dd = session_distance(man, j, k) if man else None
            P["interference"].append(a["interference_gd"] / e["grad_norms"]["g_k"] ** 2)
            P["adam"].append(a["interference_adam"])
            P["neg_cos"].append(-a["cos"])
            P["design_distance"].append(dd if dd is not None else np.nan)
            P["dL"].append(dl)
            P["dSR"].append(dsr)
            rows.append([f"S{j} during S{k}", f(-a["cos"]), f(P["interference"][-1], 3), f(dd), f(dl, 4), f(dsr)])
    out = {}
    for pred in ("interference", "adam", "neg_cos", "design_distance"):
        out[pred] = {"vs_loss": corr(P[pred], P["dL"]), "vs_style": corr(P[pred], P["dSR"])}
    R.p("Predictors available at θ_{k−1} (before session k trains): first-order interference −⟨g_j,g_k⟩/‖g_k‖² (Eq. 4), "
        "its Adam-preconditioned version, −cos(g_j,g_k), and the a-priori genre distance used to design the session "
        "order. Target: forgetting of S_j caused by training S_k (loss on held-out windows; StyleRetention drop).")
    R.table(["pair", "−cos", "−⟨g_j,g_k⟩/‖g_k‖²", "design distance", "ΔL^cond over S_k", "StyleRetention drop over S_k"],
            rows)
    R.table(["predictor", "Spearman vs ΔL", "Spearman vs style drop", "n"],
            [[p, f(v["vs_loss"]["spearman"], 2), f(v["vs_style"]["spearman"], 2), v["vs_loss"]["n"]] for p, v in out.items()])
    R.summary["forecast"] = {p: {"vs_loss": v["vs_loss"]["spearman"], "vs_style": v["vs_style"]["spearman"]}
                             for p, v in out.items()}
    R.p("With only 5 sessions there are at most 10 pairs: report as a trend, not a significance test. *Paper:* Sec. 3.2 "
        "(Eq. 4 is predictive in practice) or supp.")


def headline(R: Report) -> list[str]:
    """One line per finding, from the numbers collected by the sections."""
    s, out = R.summary, ["", "## Headline findings", "",
                         "Auto-generated from the sections below; each line points to the section with the evidence.", ""]
    if "thm1_verified_all_boundaries" in s:
        out.append(f"- **Thm 1** {'holds' if s['thm1_verified_all_boundaries'] else 'FAILS'} at every boundary; BIC "
                   f"rank by boundary {s['M_hat_by_boundary']} (A1).")
    if "thm2_slopes" in s:
        out.append(f"- **Thm 2** operator-error slope {f(s['thm2_slopes'], 2)} against √(1/N'−1/N) (theory: 1) (A2).")
    if "fw_bound_holds" in s:
        out.append(f"- **Prop. fw** bound {'holds' if s['fw_bound_holds'] else 'is VIOLATED'}; **Lemma modal** "
                   f"{'holds' if s.get('lemma_holds') else 'is VIOLATED'} (A3).")
    lit = [x for x in s.get("literal_algorithm1_atoms", []) if x is not None]
    if lit:
        out.append(f"- **Algorithm 1** with its literal positive-correlation filter selects {lit} atoms; the "
                   "Frank-Wolfe-gap test fills the budget with the same atom choices -- fix the termination test (A3).")
    if "prop1_bound_holds_small_steps" in s:
        out.append(f"- **Prop. 1 / Eq. 4 (one exact step)**: {'consistent' if s['prop1_bound_holds_small_steps'] else 'NOT consistent'}"
                   f"; ρ(measured excess forgetting, δ) = {f(s.get('prop1_spearman_excess_delta'), 2)} (A6).")
    d = s.get("delta_rel_random", {})
    if d:
        out.append(f"- **Gradient error δ** relative to ER: TMR {f(d.get('tmr'), 2)}, gradient oracle {f(d.get('grad_omp'), 2)}, "
                   f"herding {f(d.get('herding'), 2)}, channel PCA {f(d.get('channel_pca'), 2)} (A4).")
    if s.get("gradient_snr"):
        out.append(f"- **Per-window gradient SNR (one diffusion draw)**: "
                   f"{f([x.get('median_signal_to_noise_one_draw') for x in s['gradient_snr']], 3)} -- gradient-space "
                   "selection is noise-limited for diffusion; φ is noise-free (A4).")
    if s.get("tmr_space_rank"):
        out.append(f"- **Selection space vs δ**: TMR's whitened temporal space ranks {s['tmr_space_rank']} of the "
                   "candidate spaces at certifying a small gradient error (A5).")
    if s.get("median_grounding_share") is not None:
        out.append(f"- **What is forgotten**: median {pct(s['median_grounding_share'])} of old-session loss forgetting is "
                   "lost audio grounding (−ΔG), the rest prior drift (B1).")
    fb = s.get("forgetting_share_by_block")
    if fb:
        top = max(fb, key=fb.get)
        audio = sum(fb.get(b, 0.0) for b in ("audio_encoder", "cross_attn_kv"))
        out.append(f"- **Where**: largest share of forgetting in {BLOCK_NAMES.get(top, top)} ({pct(fb[top])}); audio "
                   f"pathway (encoder + cross-attn K/V) {pct(audio)} (B3).")
    rc = s.get("recency")
    if rc:
        t = max(rc, key=int)
        r = rc[t]
        out.append(f"- **Recency attractor** after S{t}: old music danced in the newest genre {f(r['recency_bias_index'], 2)} "
                   f"(real dance {f(r['gt_recency_bias_index'], 2)}); I(music; style) {f(r['style_information_bits'], 2)} "
                   f"bits (real dance {f(r['gt_style_information_bits'], 2)}, max {f(r['max_information_bits'], 2)}) (C1).")
    rv = s.get("rhythm_vs_style", {})
    if rv.get("BAS_old_rel_gt") is not None:
        out.append(f"- **Rhythm vs style**: old sessions keep {pct(rv['BAS_old_rel_gt'])} of real-dance beat alignment "
                   f"and {pct(rv.get('style_old_rel_gt'))} of its style (C2).")
    if s.get("loss_vs_style_spearman") is not None:
        out.append(f"- **Loss ↔ reported forgetting**: Spearman(ΔL^cond, StyleRetention drop) = "
                   f"{f(s['loss_vs_style_spearman'], 2)} (D1).")
    fc = s.get("forecast")
    if fc:
        best = max(fc, key=lambda p: num(fc[p]["vs_loss"]) if num(fc[p]["vs_loss"]) is not None else -9)
        out.append(f"- **Forecast**: best predictor of a session's forgetting before training it is `{best}` "
                   f"(Spearman {f(fc[best]['vs_loss'], 2)} vs loss forgetting) (D2).")
    return out + [""]


# ----------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--config", default="configs/cl/ldf_aistpp_genre5_s0.yaml", help="for the benchmark manifest")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--results_dir", type=Path, default=None, help="eval_cl_session.py --out JSONs of the method")
    a = ap.parse_args()
    out = a.out_dir or Path("results/evidence") / a.benchmark
    try:
        man = E.manifest(E.load_cfg(a.config), a.benchmark)
    except Exception as ex:  # the report still works without the manifest
        E.log(f"manifest unavailable ({ex}); design-distance forecast skipped")
        man = None
    e1, e2, e3 = load_dir(out / "e1", "boundary_k*.json"), load_dir(out / "e2", "boundary_k*.json"), \
        load_dir(out / "e3", "boundary_k*.json")
    e4, e5 = load(out / "e4" / "trajectory.json"), load(out / "e5" / "behavior.json")
    K = len(man["sessions"]) if man else (e4 or {}).get("K", 5)
    res = results_matrix(a.results_dir, K)

    R = Report()
    R.p(f"# TMR evidence on DiscoForcing ({a.benchmark})", "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `scripts/evidence/consolidate.py` from `{out}`"
        + (f" and `{a.results_dir}`" if a.results_dir else "") + ".",
        "Each block states the claim, a verdict computed from the numbers (SUPPORTED / NOT SUPPORTED / INCONCLUSIVE), "
        "the numbers, where it belongs in the paper and which figure to use. LaTeX tables: `evidence_tables.tex`.",
        "All checkpoints are the vanilla fine-tuning run; selection-rule comparisons are training-free (made at the "
        "same θ_{k−1}), so they do not need a TMR run.")
    R.h("A. Checks of the analysis (Sec. 3.4 and supp. Secs. app:thm1-app:thm2)", 1)
    theory_e1(R, e1)
    theory_e2(R, e2)
    theory_e3(R, e3)
    R.h("B. Mechanisms of forgetting in an audio-conditioned generator", 1)
    mechanisms(R, e1, e2, e4)
    if not e4:
        R.status.append(("E4 trajectory (decomposition, attribution, dynamics)", "missing"))
    else:
        R.status.append(("E4 trajectory", f"sessions {e4['trained_sessions']}"))
    R.h("C. What forgetting looks like in the generated dances", 1)
    behaviour(R, e5)
    R.h("D. Cross-analyses", 1)
    cross(R, e2, e4, res, man, K)

    status = ["", "## Status", "", "| part | state |", "|---|---|"] + [f"| {n} | {s} |" for n, s in R.status] + [""]
    figs = sorted(p.name for p in (out / "figures").glob("*.pdf")) if (out / "figures").exists() else []
    tail = ["", "## Figures", ""] + [f"- `figures/{p}`" for p in figs]
    md = R.md[:6] + status + headline(R) + R.md[6:] + tail
    (out / "EVIDENCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "evidence_tables.tex").write_text(
        "% generated by scripts/evidence/consolidate.py; needs \\usepackage{booktabs}\n" + "\n".join(R.tex) + "\n",
        encoding="utf-8")
    E.dump_json(out / "evidence_summary.json", R.summary)
    E.log(f"wrote {out / 'EVIDENCE.md'} and {out / 'evidence_tables.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
