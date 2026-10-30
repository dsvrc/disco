#!/usr/bin/env python3
"""
Summary table of a session k-1 -> k screen continuation (scripts/cl/screen_sk.sh, any k >= 2): per variant the scores
of sessions 0..k after training session k, their harmonic mean and mean (every session counts equally), how much of
each old session i is kept (after k / right after learning i: the variant's own after_s<i>_sess<i>, for S0 the shared
S0 model's), and the composition of the session k-1 buffer. When a variant was also trained with other seeds
(<variant>_seed<k>), a second table gives each method's mean and spread over its seeds.

  python scripts/cl/screen_summary_sk.py --session 3 --results results/aistpp_genre5/screen/v2_40000_full_s0v2 \
      --buffers buffers/aistpp_genre5/screen/v2_40000_full_s0v2
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

from screen_summary import composition, load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, required=True, help="the session k the screen trained")
    ap.add_argument("--results", required=True)
    ap.add_argument("--buffers", default=None)
    a = ap.parse_args()
    k, res = a.session, Path(a.results)
    ref0 = load(res / "s0_model" / "after_s0_sess0.json")
    rows, unfinished = [], []
    for d in sorted(p for p in res.iterdir() if p.is_dir() and p.name != "s0_model"):
        after = [load(d / f"after_s{k}_sess{i}.json") for i in range(k + 1)]
        if None in after:
            if any(after) or (d / f"after_s{k - 1}_sess{k - 1}.json").exists():
                unfinished.append(d.name)
            continue
        r = [s["StyleRetention"] for s in after]
        learned = [ref0] + [load(d / f"after_s{i}_sess{i}.json") for i in range(1, k)]
        kept = [r[i] / s["StyleRetention"] if s and s["StyleRetention"] else None for i, s in enumerate(learned)]
        buf = load(Path(a.buffers) / d.name / f"s{k - 1}.json") if a.buffers else None
        rows.append((d.name, dict(ret=r, mean=statistics.mean(r), hmean=statistics.harmonic_mean(r) if min(r) > 0 else 0.0,
                                  kept=kept, dance=[s.get("DanceRate", float("nan")) for s in after],
                                  comp=composition(buf))))

    olds = range(k)
    print(f"## S{k - 1} -> S{k} screen: {res}\n")
    print(f"Ranked by the harmonic mean of S0..S{k} retention after S{k} (high only when all are). 'S<i> kept' =\n"
          f"S<i> retention after S{k} / right after learning S<i> (S0: the S0 model's).\n")
    print("| variant | " + " | ".join(f"S{i} retention" for i in olds) + f" | S{k} retention (new) | harmonic mean | mean | "
          + " | ".join(f"S{i} kept" for i in olds) + f" | DanceRate S0..S{k} | S{k - 1} buffer |")
    print("|---" * (2 * k + 6) + "|")
    for name, r in sorted(rows, key=lambda x: -x[1]["hmean"]):
        kept = " | ".join(f"{x:.0%}" if x is not None else "-" for x in r["kept"])
        print(f"| {name} | " + " | ".join(f"{x:.3f}" for x in r["ret"]) + f" | **{r['hmean']:.3f}** | {r['mean']:.3f} | "
              f"{kept} | {' / '.join(f'{x:.2f}' for x in r['dance'])} | {r['comp']} |")
    for name in unfinished:
        print(f"| {name} | not finished" + " |" * (2 * k + 5))

    groups = {}
    for name, r in rows:
        groups.setdefault(re.sub(r"_seed\d+$", "", name), []).append(r)
    if any(len(v) > 1 for v in groups.values()):
        def ms(v):
            return f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"

        print("\nPer method over training seeds: mean ± standard deviation over runs, ranked by the mean harmonic mean.\n")
        print("| method | runs | " + " | ".join(f"S{i} retention" for i in range(k + 1))
              + " | harmonic mean | harmonic mean range |")
        print("|---" * (k + 5) + "|")
        for g, v in sorted(groups.items(), key=lambda kv: -statistics.mean(r["hmean"] for r in kv[1])):
            h = [r["hmean"] for r in v]
            print(f"| {g} | {len(v)} | " + " | ".join(ms([r["ret"][i] for r in v]) for i in range(k + 1))
                  + f" | **{ms(h)}** | {min(h):.3f} – {max(h):.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
