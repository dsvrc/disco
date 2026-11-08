#!/usr/bin/env python3
"""
Table of a whole screen (scripts/cl/screen_test_all.sh runs it after testing): one row group per trained session t, in
it one sub-row per dance seen up to t, one column per baseline (its seeds <variant>_seed<k> averaged, mean +- standard
deviation). After S0 is the S0 model all baselines share. Bold is the best replay method of the row; vanilla (no
replay) is the reference and never bold.

--by picks the sub-rows: genre (LO, WA, ... the default together with session) | session (the session's genres together)
| dancer (d13, d14, ...). session reads the scores as they are. genre and dancer split the session's test windows by the
music they were generated for and apply the same metric to each part (still the probability mass on the SESSION's
genres, so a session's value is the window-weighted mean of its genres'); they need the dances saved next to the scores,
and give StyleRetention, DanceRate and StyleAcc only (the FID-style metrics are unreliable on so few windows).

  python scripts/cl/screen_matrix_table.py --results results/aistpp_genre5/screen/v2_40000_full_s0v2
Writes <results>/matrix_by_<by>.md (every metric) and prints StyleRetention.
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

# metric -> (higher is better, decimals)
METRICS = {"StyleRetention": (True, 3), "DanceRate": (True, 3), "StyleAcc": (True, 3), "StyleFD": (False, 3),
           "FID_k": (False, 1), "FID_g": (False, 1)}
GENRE = {"BR": "Break", "PO": "Pop", "LO": "Lock", "WA": "Waack", "MH": "Middle Hip-hop", "LH": "LA Hip-hop",
         "HO": "House", "KR": "Krump", "JS": "Street Jazz", "JB": "Ballet Jazz"}
KEY = {"genre": lambda n: n[1:3], "dancer": lambda n: n.split("_")[3]}  # gLO_sBM_cAll_d13_mLO2_ch02 -> LO, d13
COLUMNS = ["vanilla", "random", "random_star", "ocs", "gcr", "mir", "csrel", "tmr", "tmr_joint_neff"]


def method(variant: str) -> str:
    return re.sub(r"_seed\d+$", "", variant)


def column_order(m: str) -> tuple:  # each baseline next to its distillation version
    base = m[:-3] if m.endswith("_kd") else m
    return (COLUMNS.index(base) if base in COLUMNS else len(COLUMNS), base, m != base)


def encoded(score_path: Path, enc) -> tuple[np.ndarray, np.ndarray]:
    """Style probabilities per window of the saved dances (windows x classes, and the dance of each window), cached
    next to them so that rerunning the table costs nothing."""
    dances, cache = score_path.with_suffix(".joints.npz"), score_path.with_suffix(".style.npz")
    if cache.exists() and cache.stat().st_mtime >= dances.stat().st_mtime:
        z = np.load(cache)
        return z["prob"], z["owner"]
    z = np.load(dances)
    e = enc.encode([z[str(q)] for q in range(len(z.files))])
    np.savez_compressed(cache, prob=e["prob"].astype(np.float32), owner=e["owner"])
    return e["prob"], e["owner"]


def rows_of(path: Path, session: int, by: str, enc, soc) -> dict | None:
    """{(session, group): {metric: value}} of one score file; None when its dances were not saved."""
    score = json.loads(path.read_text())
    if by == "session":
        return {(session, ""): {m: score[m] for m in METRICS if m in score}}
    if "names" not in score or not path.with_suffix(".joints.npz").exists():
        return None
    prob, owner = encoded(path, enc)
    names = score["names"]  # dance q belongs to names[q % len(names)] (scripts/eval_cl_session.py)
    groups = np.array([KEY[by](names[q % len(names)]) for q in owner])
    return {(session, g): enc.session_scores(prob[groups == g], session, soc) for g in sorted(set(groups.tolist()))}


def table(data: dict, methods: list[str], metric: str, by: str, sessions: list) -> list[str]:
    higher, digits = METRICS[metric]
    head = {"genre": "Dance genre", "session": "Dance", "dancer": "Dancer"}[by]
    lines = [f"| Model after | {head} | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for t in sorted({t for _, t, _ in data}):
        for n, row in enumerate(sorted({r for _, tt, r in data if tt == t})):
            vals = {m: data[m, t, row].get(metric) for m in methods if (m, t, row) in data}
            ranked = [np.mean(v) for m, v in vals.items() if v and m != "vanilla"]
            best = (max if higher else min)(ranked) if ranked and t > 0 else None  # after S0 all columns are one model
            cells = []
            for m in methods:
                v = vals.get(m)
                if not v:
                    cells.append("-")
                    continue
                s = f"{np.mean(v):.{digits}f}" + (f" ± {np.std(v, ddof=1):.{digits}f}" if len(v) > 1 else "")
                cells.append(f"**{s}**" if best is not None and m != "vanilla" and np.mean(v) == best else s)
            i, g = row
            names = " + ".join(GENRE.get(x, x) for x in sessions[i])
            label = f"S{i}: {names}" if by == "session" else f"{GENRE.get(g, g)} (S{i})"
            lines.append(f"| {'**After S%d**' % t if n == 0 else ''} | {label} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--by", nargs="+", choices=["genre", "session", "dancer"], default=["genre", "session"])
    a = ap.parse_args()

    man = json.loads((Path("raw_data/cl_sessions") / a.benchmark / "manifest.json").read_text())
    sessions = [s["groups"] for s in man["sessions"]]
    enc, soc = None, None
    if set(a.by) - {"session"}:
        from metrics.style_encoder import StyleEncoder, benchmark_session_of_class, ckpt_path

        enc = StyleEncoder(ckpt_path(man["dataset"], -1))
        soc = np.asarray(benchmark_session_of_class(enc.classes, man))

    files = {}  # (variant, trained up to t, session tested) -> score file
    for d in a.results.iterdir():
        for p in d.glob("after_s*_sess*.json") if d.is_dir() else []:
            m = re.fullmatch(r"after_s(\d+)_sess(\d+)\.json", p.name)
            if m and int(m[2]) <= int(m[1]):
                files[d.name, int(m[1]), int(m[2])] = p
    methods = sorted({method(v) for v, _, _ in files if v != "s0_model"}, key=column_order)
    if not methods:
        raise SystemExit(f"no scored baselines in {a.results}")

    for by in a.by:
        data, unsplit = {}, []  # (method, t, (session, group)) -> {metric: [one value per seed]}
        for (v, t, i), p in sorted(files.items()):
            rows = rows_of(p, i, by, enc, soc)
            if rows is None:
                unsplit.append(f"{v} after S{t} on S{i}")
                continue
            for row, scores in rows.items():
                for m in methods if v == "s0_model" else [method(v)]:  # the S0 model is every baseline's start
                    for name, value in scores.items():
                        data.setdefault((m, t, row), {}).setdefault(name, []).append(value)
        out = [f"# {a.results.name}: every model on every dance seen so far, by {by}", "",
               "Columns are baselines (mean ± sd over seeds; no ± = one seed; - = not run). Bold = best replay method of "
               "the row (vanilla is the reference, never bold). After S0 is the S0 model all baselines share.", ""]
        for metric, (higher, _) in METRICS.items():
            if any(metric in v for v in data.values()):
                out += [f"## {metric} ({'higher' if higher else 'lower'} is better)", ""]
                out += table(data, methods, metric, by, sessions) + [""]
        path = a.results / f"matrix_by_{by}.md"
        path.write_text("\n".join(out), encoding="utf-8")
        print(f"\nStyleRetention by {by} (higher is better; every metric: {path})\n")
        print("\n".join(table(data, methods, "StyleRetention", by, sessions)))
        if unsplit:
            print(f"\n[table] no saved dances, so not in this table: {'; '.join(unsplit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
