#!/usr/bin/env python3
"""
DiscoForcing tables of the TMR paper from a screen (scripts/cl/test_kd_screens.sh runs it after testing):

  * main paper, Tab. 3 (tab:disco): one row per method; StyleRetention of the earlier genres after Sessions 1..4 and
    their mean over these checkpoints, and the SR of each session's genres right after it is learned (New genres)
  * supplement (tab:disco_full): SR of every genre seen so far after every session, one column per method

Every number is the mean over a method's training seeds (<variant> and <variant>_seed<k>); the seed count of each method
is listed in the .md file. A genre's SR is the style encoder's mass on its session's genres over the windows generated for
that genre's test music (scripts/cl/screen_matrix_table.py --by genre: same scores, same cache). After S0 every method
is the shared S0 model, scored once (results/.../s0_model). A session's value in the summary is the mean of its genres.
Bold: the best continual-learning method of a column / row (fine-tuning is the reference, never bold).

  python scripts/cl/disco_paper_tables.py --results results/aistpp_genre5/screen/v2_40000_full_s0v2
  -> <results>/paper_disco_tables.md (both tables, markdown) and paper_disco_tables.tex (LaTeX bodies to paste)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen_matrix_table import GENRE, method, rows_of  # noqa: E402  (same per-genre split and cache)

NAME = {"vanilla": "Fine-tuning", "random": "ER", "random_kd": "ER+KD", "random_star": "STAR", "ocs": "OCS",
        "gcr": "GCR", "gcr_kd": "GCR+KD", "mir": "MIR", "mir_kd": "MIR+KD", "csrel": "CSReL", "csrel_kd": "CSReL+KD",
        "tmr_joint_neff": "TMR", "tmr_joint_neff_kd": "TMR+KD"}
TEX = {"tmr_joint_neff": r"\TMRh", "tmr_joint_neff_kd": r"\TMRh+KD"}
# main table: row groups (fine-tuning | ground-truth replay | distillation replay); supplement: methods next to their +KD
GROUPS = [["vanilla"], ["random", "random_star", "ocs", "gcr", "mir", "csrel", "tmr_joint_neff"],
          ["random_kd", "gcr_kd", "mir_kd", "csrel_kd", "tmr_joint_neff_kd"]]
FULL = ["vanilla", "random", "random_kd", "random_star", "ocs", "gcr", "gcr_kd", "mir", "mir_kd", "csrel", "csrel_kd",
        "tmr_joint_neff", "tmr_joint_neff_kd"]
# the paper's Tab. 3 as laid out now: fine-tuning and the selection methods, every buffer replayed by distillation, each
# +KD row named after its selection method (the caption says the buffers are replayed by distillation)
PAPER = ["vanilla", "mir_kd", "csrel_kd", "gcr_kd", "tmr_joint_neff_kd"]


def collect(results: Path, benchmark: str):
    """{(method, t, session, genre): [SR per seed]}, the session groups, seed counts, files without saved dances."""
    man = json.loads((Path("raw_data/cl_sessions") / benchmark / "manifest.json").read_text())
    sessions = [list(s["groups"]) for s in man["sessions"]]
    from metrics.style_encoder import StyleEncoder, benchmark_session_of_class, ckpt_path

    enc = StyleEncoder(ckpt_path(man["dataset"], -1))
    soc = np.asarray(benchmark_session_of_class(enc.classes, man))
    data, seeds, unsplit = {}, {}, []
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        for p in sorted(d.glob("after_s*_sess*.json")):
            m = re.fullmatch(r"after_s(\d+)_sess(\d+)\.json", p.name)
            if not m:
                continue
            t, i = int(m[1]), int(m[2])
            if i > t or (t == 0) != (d.name == "s0_model"):  # S0 comes from the shared S0 model only
                continue
            rows = rows_of(p, i, "genre", enc, soc)
            if rows is None:
                unsplit.append(f"{d.name} after S{t} on S{i}")
                continue
            key = "s0_model" if t == 0 else method(d.name)
            if t == 4 and i == 0:
                seeds[key] = seeds.get(key, 0) + 1
            for (_, g), scores in rows.items():
                data.setdefault((key, t, i, g), []).append(scores["StyleRetention"])
    return data, sessions, seeds, unsplit


def sr(data, m, t, i, g):
    v = data.get(("s0_model" if t == 0 else m, t, i, g))
    return float(np.mean(v)) if v else None


def summary(data, sessions, methods):
    """{method: {S1..S4, Mean, New}} (None where a run is missing)."""
    K = len(sessions)
    out = {}
    for m in methods:
        row, every = {}, []
        for t in range(1, K):
            vals = [sr(data, m, t, i, g) for i in range(t) for g in sessions[i]]
            row[f"S{t}"] = None if None in vals else float(np.mean(vals))
            every += vals
        row["Mean"] = None if None in every or not every else float(np.mean(every))
        new = [sr(data, m, t, t, g) for t in range(1, K) for g in sessions[t]]
        row["New"] = None if None in new else float(np.mean(new))
        out[m] = row
    return out


def fmt(v, best=None, tex=False, m=None):
    """A value; bold when it is the column's / row's best (fine-tuning, the reference, is never bold)."""
    if v is None:
        return "--"
    s = f"{v:.3f}"
    if best is not None and m != "vanilla" and round(v, 3) == best:
        return r"\textbf{" + s + "}" if tex else f"**{s}**"
    return s


def best_of(values: dict, methods) -> float | None:
    v = [round(values[m], 3) for m in methods if m != "vanilla" and values.get(m) is not None]
    return max(v) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--benchmark", default="aistpp_genre5")
    a = ap.parse_args()
    data, sessions, seeds, unsplit = collect(a.results, a.benchmark)
    present = {m for (m, *_rest) in data if m != "s0_model"}
    groups = [[m for m in grp if m in present] for grp in GROUPS]
    groups = [grp for grp in groups if grp]
    main_methods = [m for grp in groups for m in grp]
    full = [m for m in FULL if m in present] + sorted(present - set(FULL))
    S = summary(data, sessions, main_methods)
    cols = [f"S{t}" for t in range(1, len(sessions))] + ["Mean", "New"]
    best = {c: best_of({m: S[m][c] for m in main_methods}, main_methods) for c in cols}

    md = [f"# DiscoForcing paper tables: {a.results.name}", "",
          "Means over training seeds; seeds per method (runs scored after S4): "
          + ", ".join(f"{NAME.get(m, m)} {seeds.get(m, 0)}" for m in main_methods), "",
          "## Tab. 3 (main paper): SR of earlier genres after each session, their mean, and of new genres", "",
          "| Method | " + " | ".join(c if c != "New" else "New genres" for c in cols) + " |",
          "|---|" + "---|" * len(cols)]
    tex = ["% Tab. 3 (main paper), body of \\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l ccccc c}",
           r"\toprule", r" & \multicolumn{5}{c}{Earlier genres $\uparrow$} & New $\uparrow$\\", r"\cmidrule(lr){2-6}",
           r"Method & S1 & S2 & S3 & S4 & Mean & genres\\"]
    for n, grp in enumerate(groups):
        tex.append(r"\midrule")
        for m in grp:
            md.append(f"| {NAME.get(m, m)} | " + " | ".join(fmt(S[m][c], best[c], m=m) for c in cols) + " |")
            shade = r"\rowcolor{gray!15} " if m.startswith("tmr") else ""
            tex.append(shade + TEX.get(m, NAME.get(m, m)) + " & "
                       + " & ".join(fmt(S[m][c], best[c], tex=True, m=m) for c in cols) + r"\\")
    tex += [r"\bottomrule", ""]

    # the same table in the paper's current layout: fine-tuning + the +KD methods, named after their selection method
    paper = [m for m in PAPER if m in present]
    pbest = {c: best_of({m: S[m][c] for m in paper}, paper) for c in cols}
    pname = lambda m: NAME.get(m[:-3] if m.endswith("_kd") else m, m)  # noqa: E731
    missing = [NAME[m] for m in PAPER if m not in present]
    md += ["", "## Tab. 3 in the paper's layout: every buffer replayed by distillation, rows named by selection method"
           + (f" (not run yet: {', '.join(missing)})" if missing else ""), "",
           "| Method | " + " | ".join(c if c != "New" else "New genres" for c in cols) + " |", "|---|" + "---|" * len(cols)]
    tex += ["% Tab. 3 in the paper's layout (every buffer replayed by distillation), same tabular* as above",
            r"\toprule", r" & \multicolumn{5}{c}{Earlier genres $\uparrow$} & New $\uparrow$\\", r"\cmidrule(lr){2-6}",
            r"Method & S1 & S2 & S3 & S4 & Mean & genres\\"]
    for n, m in enumerate(paper):
        if n <= 1:
            tex.append(r"\midrule")
        md.append(f"| {pname(m)} | " + " | ".join(fmt(S[m][c], pbest[c], m=m) for c in cols) + " |")
        shade = r"\rowcolor{gray!15} \TMRh" if m.startswith("tmr") else pname(m)
        tex.append(shade + " & " + " & ".join(fmt(S[m][c], pbest[c], tex=True, m=m) for c in cols) + r"\\")
    tex += [r"\bottomrule", ""]

    # supplement: per genre
    md += ["", "## Supplementary per-genre table: SR after every session (bold: best continual-learning method)", "",
           "| After | Genre | " + " | ".join(NAME.get(m, m) for m in full) + " |", "|---|---|" + "---|" * len(full)]
    tex += ["% supplementary per-genre table, body of \\resizebox{\\textwidth}{!}{\\begin{tabular}{cl" + "c" * len(full) + "}",
            r"\toprule", "After & Genre & " + " & ".join(TEX.get(m, NAME.get(m, m)) for m in full) + r"\\"]
    for t in range(len(sessions)):
        rows = [(i, g) for i in range(t + 1) for g in sessions[i]]
        tex.append(r"\midrule")
        for n, (i, g) in enumerate(rows):
            label = f"{GENRE.get(g, g)} (S{i})"
            vals = {m: sr(data, m, t, i, g) for m in full}
            b = best_of(vals, full) if t > 0 else None
            md.append(f"| {'**After S%d**' % t if n == 0 else ''} | {label} | "
                      + " | ".join(fmt(vals[m], b, m=m) for m in full) + " |")
            lead = (r"\multirow{%d}{*}{S%d}" % (len(rows), t)) if n == 0 else ""
            if t == 0:
                tex.append(f"{lead} & {label} & " + r"\multicolumn{%d}{c}{%s}" % (len(full), fmt(vals[full[0]])) + r"\\")
            else:
                tex.append(f"{lead} & {label} & " + " & ".join(fmt(vals[m], b, tex=True, m=m) for m in full) + r"\\")
    tex += [r"\bottomrule", ""]
    if unsplit:
        md += ["", "Not in the tables (scores without saved dances; rerun scripts/cl/test_kd_screens.sh or "
               "scripts/cl/screen_test_all.sh): " + "; ".join(unsplit)]

    (a.results / "paper_disco_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (a.results / "paper_disco_tables.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[paper] wrote {a.results / 'paper_disco_tables.md'} and paper_disco_tables.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
