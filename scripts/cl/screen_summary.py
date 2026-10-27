#!/usr/bin/env python3
"""
Summary table of an S0 -> S1 screen (scripts/cl/screen_s1.sh): per variant the S0 (old) and S1 (new) scores after
training S1, their mean (old and new count equally), the S0 model's scores on the same eval setting as reference, the
last fixed-noise probe of the replay diagnostics (non-buffered S0 clips: does replay generalise beyond the buffer?),
and the buffer's composition. With --baselines, the main runs' S1 scores of every method are added as rows (only valid
when the screen used the main runs' setup; screen_s1.sh passes it only then). When a variant was also trained with
other seeds (<variant>_seed<k>: same buffer), a second table gives each method's mean and spread over its seeds.

  python scripts/cl/screen_summary.py --results results/aistpp_genre5/screen/v1_40000_full \
      --logs logs/aistpp_genre5/screen/v1_40000_full --buffers buffers/aistpp_genre5/screen/v1_40000_full \
      --baselines results/aistpp_genre5 --baseline_buffers buffers/aistpp_genre5
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def last_probe(log: Path):
    """(start, end) loss of the non-buffered S0 probe set from the [replay-diag] blocks of a training log."""
    if not log.exists():
        return None
    vals = re.findall(r"S0 unbuffered ([\d.]+) \(d([+-][\d.]+)\)", log.read_text(errors="ignore"))
    if not vals:
        return None
    end, delta = float(vals[-1][0]), float(vals[-1][1])
    return end - delta, end


def composition(buf):
    if not buf:
        return "-"
    dancers = {c["name"].split("_")[3] for c in buf["clips"] if len(c["name"].split("_")) > 4}
    return f"{len(buf['clips'])} clips / {len({c['name'] for c in buf['clips']})} dances / {len(dancers)} dancers"


def row(old, new, probe, buf):
    o, n = old["StyleRetention"], new["StyleRetention"]
    return dict(old=o, new=n, mean=(o + n) / 2, hmean=2 * o * n / (o + n) if o + n > 0 else 0.0,
                old_dance=old.get("DanceRate", float("nan")), new_dance=new.get("DanceRate", float("nan")),
                old_fd=old["StyleFD"], new_fd=new["StyleFD"], old_fidk=old.get("FID_k", float("nan")),
                new_fidk=new.get("FID_k", float("nan")), probe=probe, comp=composition(buf))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--logs", default=None)
    ap.add_argument("--buffers", default=None)
    ap.add_argument("--baselines", default=None, help="results/<benchmark>: add every method's main-run S1 scores")
    ap.add_argument("--baseline_buffers", default=None, help="buffers/<benchmark>: the main runs' buffers")
    a = ap.parse_args()
    res = Path(a.results)
    ref0, ref1 = load(res / "s0_model" / "after_s0_sess0.json"), load(res / "s0_model" / "after_s0_sess1.json")
    rows, unfinished = [], []
    for d in sorted(p for p in res.iterdir() if p.is_dir() and p.name != "s0_model"):
        old, new = load(d / "after_s1_sess0.json"), load(d / "after_s1_sess1.json")
        if old is None or new is None:
            unfinished.append(d.name)
            continue
        probe = last_probe(Path(a.logs) / f"{d.name}.train.log") if a.logs else None
        buf = load(Path(a.buffers) / d.name / "s0.json") if a.buffers else None
        rows.append((d.name, row(old, new, probe, buf)))
    if a.baselines:
        for d in sorted(p for p in Path(a.baselines).iterdir() if p.is_dir() and p.name != "screen"):
            old, new = load(d / "after_s1_sess0.json"), load(d / "after_s1_sess1.json")
            if old is None or new is None:
                continue
            buf = load(Path(a.baseline_buffers) / d.name / "s0.json") if a.baseline_buffers else None
            rows.append((f"{d.name} (main run)", row(old, new, None, buf)))

    print(f"## S0 -> S1 screen: {res}\n")
    if ref0 and ref1:
        print(f"S0 model before S1 (same eval): S0 retention {ref0['StyleRetention']:.3f} (ceiling "
              f"{ref0['gt_reference']['StyleRetention_gt']:.3f}), S1 retention {ref1['StyleRetention']:.3f}; "
              f"complete-forgetting level of S0 {ref0['gt_reference']['StyleRetention_complete_forgetting']:.3f}\n")
    before = ref0["StyleRetention"] if ref0 else None
    print("Ranked by the harmonic mean of old and new retention (high only when both are; the plain mean rewards a\n"
          "method that learns S1 and forgets S0 completely). 'S0 kept' = S0 retention after S1 / before S1.\n")
    print("| variant | S0 retention (old) | S1 retention (new) | harmonic mean | mean | S0 kept | DanceRate old / new | "
          "StyleFD old / new | FID_k old / new | probe S0 non-buffered (start -> end) | S0 buffer |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in sorted(rows, key=lambda x: -x[1]["hmean"]):
        probe = f"{r['probe'][0]:.3f} -> {r['probe'][1]:.3f}" if r["probe"] else "-"
        kept = f"{r['old'] / before:.0%}" if before else "-"
        print(f"| {name} | {r['old']:.3f} | {r['new']:.3f} | **{r['hmean']:.3f}** | {r['mean']:.3f} | {kept} | "
              f"{r['old_dance']:.2f} / {r['new_dance']:.2f} | {r['old_fd']:.3f} / {r['new_fd']:.3f} | "
              f"{r['old_fidk']:.0f} / {r['new_fidk']:.0f} | {probe} | {r['comp']} |")
    for name in unfinished:
        print(f"| {name} | not finished | | | | | | | | | |")

    groups = {}
    for name, r in rows:
        if not name.endswith("(main run)"):
            groups.setdefault(re.sub(r"_seed\d+$", "", name), []).append(r)
    if any(len(v) > 1 for v in groups.values()):
        def ms(v):
            return f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"

        print("\nPer method over training seeds (<variant> and <variant>_seed<k> replay the same buffer): mean ± "
              "standard deviation over runs, ranked by the mean harmonic mean.\n")
        print("| method | runs | S0 retention (old) | S1 retention (new) | harmonic mean | harmonic mean range |")
        print("|---|---|---|---|---|---|")
        for k, v in sorted(groups.items(), key=lambda kv: -statistics.mean(r["hmean"] for r in kv[1])):
            h = [r["hmean"] for r in v]
            print(f"| {k} | {len(v)} | {ms([r['old'] for r in v])} | {ms([r['new'] for r in v])} | **{ms(h)}** | "
                  f"{min(h):.3f} – {max(h):.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
