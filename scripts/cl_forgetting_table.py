#!/usr/bin/env python3
"""
Collect the per-session evaluations of one continual-learning run into the matrix m_{i,t} (metric of session i
after training session t) and summarise forgetting with metrics.cl_metrics.forgetting_summary().

Expects the JSONs written by scripts/eval_cl_session.py --out, named
  <results_dir>/after_s{t}_sess{i}.json      (model after session t, scored on session i's test music)
Missing entries are shown as '-'; the summary uses the sessions trained so far (T = last t with a full column).

Usage:
  python scripts/cl_forgetting_table.py --results_dir results/aistpp_genre5/vanilla
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics.cl_metrics import forgetting_summary

# metric -> higher is better
METRICS = {"StyleRetention": True, "DanceRate": True, "StyleAcc": True, "StyleFD": False, "FID_k": False, "FID_g": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, required=True)
    a = ap.parse_args()

    entries = {}
    for p in a.results_dir.glob("after_s*_sess*.json"):
        m = re.fullmatch(r"after_s(\d+)_sess(\d+)\.json", p.name)
        if m:
            entries[int(m.group(1)), int(m.group(2))] = json.loads(p.read_text())
    if not entries:
        raise SystemExit(f"no after_s*_sess*.json in {a.results_dir}")
    K = max(max(t, i) for t, i in entries) + 1
    # sessions trained so far: every seen session (i <= t) is scored after t
    T = max((t + 1 for t in range(K) if all((t, i) in entries for i in range(t + 1))), default=0)

    for name, higher in METRICS.items():
        M = np.full((K, K), np.nan)
        for (t, i), e in entries.items():
            if name in e:
                M[i, t] = e[name]
        if np.isnan(M).all():
            continue
        print(f"\n{name} ({'higher' if higher else 'lower'} is better)   rows: session i, columns: after session t")
        print("        " + "".join(f"{'after S' + str(t):>11}" for t in range(K)))
        for i in range(K):
            print(f"  S{i}    " + "".join(f"{M[i, t]:>11.4f}" if not np.isnan(M[i, t]) else f"{'-':>11}" for t in range(K)))
        if T >= 2:
            s = forgetting_summary(M[:T, :T], higher_is_better=higher)
            print(f"  over S0..S{T - 1}: F_avg={s['F_avg']:.4f}  F_rel_avg={s['F_rel_avg']:.3f}  "
                  f"A_avg={s['A_avg']:.4f}  F_i={[round(f, 4) for f in s['F_i']]}  (F > 0 = forgetting)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
