#!/usr/bin/env python3
"""
Summary table of an S1 -> S2 screen continuation (scripts/cl/screen_s2.sh): per variant the S0, S1 (old) and S2 (new)
scores after training S2, their harmonic mean and mean (every session counts equally), how much of each old session
is kept (S0: after S2 / the S0 model's; S1: after S2 / after S1, the same variant's own S1 score), and the S1 buffer's
composition. When a variant was also trained with other seeds (<variant>_seed<k>), a second table gives each method's
mean and spread over its seeds.

  python scripts/cl/screen_summary_s2.py --results results/aistpp_genre5/screen/v2_40000_full_s0v2 \
      --buffers buffers/aistpp_genre5/screen/v2_40000_full_s0v2
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

from screen_summary import composition, load


def row(s0, s1, s2, s0_before, s1_before, buf):
    r = [s0["StyleRetention"], s1["StyleRetention"], s2["StyleRetention"]]
    return dict(s0=r[0], s1=r[1], s2=r[2], mean=statistics.mean(r),
                hmean=statistics.harmonic_mean(r) if min(r) > 0 else 0.0,
                kept0=r[0] / s0_before if s0_before else None, kept1=r[1] / s1_before if s1_before else None,
                dance=[s.get("DanceRate", float("nan")) for s in (s0, s1, s2)], comp=composition(buf))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--buffers", default=None)
    a = ap.parse_args()
    res = Path(a.results)
    ref0 = load(res / "s0_model" / "after_s0_sess0.json")
    s0_before = ref0["StyleRetention"] if ref0 else None
    rows, unfinished = [], []
    for d in sorted(p for p in res.iterdir() if p.is_dir() and p.name != "s0_model"):
        after = [load(d / f"after_s2_sess{i}.json") for i in range(3)]
        if None in after:
            if any(after) or (d / "after_s1_sess1.json").exists():
                unfinished.append(d.name)
            continue
        s1_after_s1 = load(d / "after_s1_sess1.json")
        buf = load(Path(a.buffers) / d.name / "s1.json") if a.buffers else None
        rows.append((d.name, row(*after, s0_before, s1_after_s1["StyleRetention"] if s1_after_s1 else None, buf)))

    print(f"## S1 -> S2 screen: {res}\n")
    print("Ranked by the harmonic mean of S0, S1 and S2 retention after S2 (high only when all three are). 'S0 kept' =\n"
          "S0 retention after S2 / the S0 model's; 'S1 kept' = S1 retention after S2 / the same variant's after S1.\n")
    print("| variant | S0 retention | S1 retention | S2 retention (new) | harmonic mean | mean | S0 kept | S1 kept | "
          "DanceRate S0 / S1 / S2 | S1 buffer |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in sorted(rows, key=lambda x: -x[1]["hmean"]):
        k0 = f"{r['kept0']:.0%}" if r["kept0"] is not None else "-"
        k1 = f"{r['kept1']:.0%}" if r["kept1"] is not None else "-"
        print(f"| {name} | {r['s0']:.3f} | {r['s1']:.3f} | {r['s2']:.3f} | **{r['hmean']:.3f}** | {r['mean']:.3f} | "
              f"{k0} | {k1} | {' / '.join(f'{x:.2f}' for x in r['dance'])} | {r['comp']} |")
    for name in unfinished:
        print(f"| {name} | not finished | | | | | | | | |")

    groups = {}
    for name, r in rows:
        groups.setdefault(re.sub(r"_seed\d+$", "", name), []).append(r)
    if any(len(v) > 1 for v in groups.values()):
        def ms(v):
            return f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"

        print("\nPer method over training seeds: mean ± standard deviation over runs, ranked by the mean harmonic mean.\n")
        print("| method | runs | S0 retention | S1 retention | S2 retention (new) | harmonic mean | harmonic mean range |")
        print("|---|---|---|---|---|---|---|")
        for k, v in sorted(groups.items(), key=lambda kv: -statistics.mean(r["hmean"] for r in kv[1])):
            h = [r["hmean"] for r in v]
            print(f"| {k} | {len(v)} | {ms([r['s0'] for r in v])} | {ms([r['s1'] for r in v])} | "
                  f"{ms([r['s2'] for r in v])} | **{ms(h)}** | {min(h):.3f} – {max(h):.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
