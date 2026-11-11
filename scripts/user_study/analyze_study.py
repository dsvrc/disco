#!/usr/bin/env python3
"""
Score the StyleRetention human study: how often each metric picks the clip humans picked.

  python scripts/user_study/analyze_study.py --pack study_pack.zip --responses answers/ --out study_report.md

--responses: a folder of the JSON files the study page downloads (one per participant). numpy only.

Pre-registered analysis (fix these before collecting answers):
  exclusion   catch accuracy < 0.75 (Can't tell counts as wrong) or < 80 % of the assigned main trials answered
  primary     agreement of StyleRetention (genre level, P(music's genre)) with human choices on main trials,
              Can't tell excluded; 95 % CI by bootstrap over trials; H1: > 0.5
  secondary   (H2) the same agreement for every baseline (handcrafted probe = Robust Dancer-style classifier,
              kinetic / geometric feature distance = per-clip FID_k / FID_g, embedding similarity = StyleFD, and
              1 - P(other)), with the paired difference to StyleRetention; agreement by |delta| stratum (should rise
              with the encoder's confidence); human-human agreement (leave-one-rater-out majority), the ceiling no
              metric can beat; Krippendorff's alpha; repeat consistency; dance task: AUC of 1 - P(other)
  exploratory per genre, per rater experience, and the human "style match" rate by sessions since learning
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = [("ours_genre", "StyleRetention, genre level (ours)"),
           ("ours_session", "StyleRetention, session level (ours, as reported)"),
           ("emb_cos", "Style-embedding similarity to the genre centroid (StyleFD analogue)"),
           ("probe_genre", "Handcrafted kinetic+geometric classifier, genre (Robust Dancer-style)"),
           ("probe_session", "Handcrafted classifier, session"),
           ("kin", "Kinetic-feature distance (FID_k analogue)"),
           ("geo", "Geometric-feature distance (FID_g analogue)"),
           ("dance_rate", "1 - P(other) (is it dance at all)")]
B = 2000


def read_study(pack: Path) -> dict:
    if pack.suffix == ".zip":
        with zipfile.ZipFile(pack) as z:
            return json.loads(z.read("study_pack/study.json"))
    return json.loads((pack / "study.json").read_text())


def score(clip: dict, m: str) -> float:
    s = clip["scores"]
    return 1.0 - s["p_other"] if m == "dance_rate" else s[m]


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    r = ranks(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def kripp_alpha(units: list[list[int]], n_cat: int) -> float:
    """Krippendorff's alpha, nominal, any number of raters per unit."""
    o = np.zeros((n_cat, n_cat))
    for vals in units:
        if len(vals) < 2:
            continue
        c = np.bincount(vals, minlength=n_cat).astype(float)
        o += (np.outer(c, c) - np.diag(c)) / (len(vals) - 1)
    nc, n = o.sum(1), o.sum()
    de = n * n - (nc ** 2).sum()
    return float(1 - (n - 1) * (n - np.trace(o)) / de) if n > 1 and de > 0 else float("nan")


def boot(by_trial: dict, fn, rng) -> tuple[float, float, float]:
    """fn(list of per-trial records) -> value; percentile CI over trials resampled with replacement."""
    keys = list(by_trial)
    est = fn([r for k in keys for r in by_trial[k]])
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        vals.append(fn([r for p in pick for r in by_trial[keys[p]]]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return est, float(np.percentile(vals, 2.5)) if len(vals) else np.nan, float(np.percentile(vals, 97.5)) if len(vals) else np.nan


def agree_rate(recs: list[dict], m: str) -> float:
    a = [r["agree"][m] for r in recs if r["agree"][m] is not None]
    return float(np.mean(a)) if a else float("nan")


def fmt(v, d=3):
    return "-" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, default=Path("study_pack.zip"))
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("study_report.md"))
    ap.add_argument("--min_catch", type=float, default=0.75)
    ap.add_argument("--min_complete", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    study = read_study(a.pack)
    clips, trials = study["clips"], {t["id"]: t for t in study["trials"]}
    dtrials = {t["id"]: t for t in study["dance_trials"]}
    files = sorted(a.responses.glob("*.json"))
    if not files:
        raise SystemExit(f"no answer files in {a.responses}")

    # ---------------- participants and exclusions ---------------------------------------------------------------
    people, seen = [], set()
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("participant") in seen:
            print(f"[analyze] {f.name}: participant {d.get('participant')} already read, skipped")
            continue
        seen.add(d.get("participant"))
        resp = d["responses"]
        catch = [r for r in resp if r["kind"] == "catch"]
        c_acc = np.mean([r["chosen"] == trials[r["trial"]]["a"] for r in catch]) if catch else float("nan")
        n_main = sum(r["kind"] == "main" for r in resp)
        d["_catch"], d["_n_main"] = float(c_acc), n_main
        d["_excluded"] = []
        if not np.isfinite(c_acc) or c_acc < a.min_catch:
            d["_excluded"].append(f"catch accuracy {fmt(c_acc, 2)}")
        if d.get("finished") is None:
            d["_excluded"].append("did not finish")
        people.append(d)
    if any(d.get("finished") for d in people):
        full = max(d["_n_main"] for d in people)
        for d in people:
            if d["_n_main"] < a.min_complete * full and "did not finish" not in d["_excluded"]:
                d["_excluded"].append(f"answered {d['_n_main']}/{full} main trials")
    kept = [d for d in people if not d["_excluded"]]

    # ---------------- main trials: one record per (participant, trial) ---------------------------------------
    recs, rep_pairs = [], []
    for d in kept:
        first = {}
        for r in d["responses"]:
            if r["task"] != "style" or r["kind"] not in ("main", "repeat"):
                continue
            t = trials[r["trial"]]
            canon = "tie" if r["chosen"] is None else ("a" if r["chosen"] == t["a"] else "b")
            if r["kind"] == "repeat":
                if r["trial"] in first:
                    rep_pairs.append((first[r["trial"]], canon))
                continue
            first[r["trial"]] = canon
            agree = {}
            for m, _ in METRICS:
                sa, sb = score(clips[t["a"]], m), score(clips[t["b"]], m)
                agree[m] = None if canon == "tie" or sa == sb else float((sa > sb) == (canon == "a"))
            recs.append({"p": d["participant"], "trial": t["id"], "canon": canon, "agree": agree, "type": t["type"],
                         "bin": t["bin"], "genre": t["genre"], "exp": d["profile"].get("experience"),
                         "rt": r.get("rt_ms")})
    if not recs:
        raise SystemExit("no main-trial answers left after exclusions")
    by_trial = defaultdict(list)
    for r in recs:
        by_trial[r["trial"]].append(r)

    # human-human: each answer against the majority of the OTHER raters of the trial (ties and draws dropped)
    for r in recs:
        others = [o["canon"] for o in by_trial[r["trial"]] if o is not r and o["canon"] != "tie"]
        na, nb = others.count("a"), others.count("b")
        r["agree"]["human"] = None if r["canon"] == "tie" or na == nb else float((na > nb) == (r["canon"] == "a"))

    L = []
    w = L.append
    w(f"# Human study: does StyleRetention agree with people?\n")
    n_main = sum(t["type"] in ("forget", "cross") for t in trials.values())
    w(f"Pack: {study['benchmark']}, created {study['created']}, {n_main} main trials in the pack. "
      f"Answers: {len(people)} participants, {len(kept)} kept.\n")
    if len(kept) < len(people):
        w("Excluded: " + "; ".join(f"{d['participant']} ({', '.join(d['_excluded'])})" for d in people if d["_excluded"]) + "\n")
    exp_n = defaultdict(int)
    for d in kept:
        exp_n[d["profile"].get("experience")] += 1
    ratings = defaultdict(int)
    for r in recs:
        ratings[r["trial"]] += 1
    tie_rate = np.mean([r["canon"] == "tie" for r in recs])
    w(f"Raters by experience: {dict(exp_n)}. Main answers: {len(recs)} on {len(by_trial)} trials "
      f"(median {int(np.median(list(ratings.values())))} raters per trial); Can't tell: {tie_rate:.1%}. "
      f"Catch accuracy of kept raters: {np.mean([d['_catch'] for d in kept]):.2f}.\n")

    # ---------------- H1 / H2: agreement per metric ----------------------------------------------------------
    w("## Agreement with human choices (main trials, Can't tell excluded)\n")
    w("| Metric | Agreement | 95% CI | Δ vs ours (genre) | 95% CI of Δ | Trial-majority agreement |")
    w("|---|---|---|---|---|---|")
    for m, label in [("human", "Human-human (leave-one-rater-out majority): ceiling")] + METRICS:
        est, lo, hi = boot(by_trial, lambda rs, m=m: agree_rate(rs, m), rng)
        if m in ("human", "ours_genre"):
            dl = dh = dv = None
        else:
            dv, dl, dh = boot(by_trial, lambda rs, m=m: agree_rate(rs, "ours_genre") - agree_rate(rs, m), rng)
        maj = []
        for k, rs in by_trial.items():
            ca = [r["canon"] for r in rs if r["canon"] != "tie"]
            if m == "human" or not ca or ca.count("a") == ca.count("b"):
                continue
            t = trials[k]
            sa, sb = score(clips[t["a"]], m), score(clips[t["b"]], m)
            if sa != sb:
                maj.append((sa > sb) == (ca.count("a") > ca.count("b")))
        w(f"| {label} | {fmt(est)} | {fmt(lo)}–{fmt(hi)} | {fmt(dv)} | "
          f"{'-' if dl is None else fmt(dl) + '–' + fmt(dh)} | {fmt(np.mean(maj)) if maj else '-'} |")
    w("\nΔ > 0 means StyleRetention (genre) agrees with people more often than that metric.\n")

    # ---------------- agreement by stratum ---------------------------------------------------------------------
    bins = study["bins"] + [1.0]
    w("## By how far apart the encoder puts the two clips (|ΔP(genre)|)\n")
    w("| Stratum | Trials | Answers | Can't tell | Ours (genre) | Probe (genre) | Kinetic | Human ceiling |")
    w("|---|---|---|---|---|---|---|---|")
    for b in range(len(bins) - 1):
        sub = {k: v for k, v in by_trial.items() if trials[k]["bin"] == b}
        if not sub:
            continue
        rs = [r for v in sub.values() for r in v]
        w(f"| [{bins[b]:.1f}, {bins[b + 1]:.1f}) | {len(sub)} | {len(rs)} | {np.mean([r['canon'] == 'tie' for r in rs]):.0%} | "
          + " | ".join(fmt(agree_rate(rs, m)) for m in ("ours_genre", "probe_genre", "kin", "human"))
          + " |")
    w("\nIf the metric is right, agreement rises with |ΔP| and Can't tell falls.\n")
    w("| Pair type | Trials | Ours (genre) | 95% CI | Human ceiling |")
    w("|---|---|---|---|---|")
    for ty in ("forget", "cross"):
        sub = {k: v for k, v in by_trial.items() if trials[k]["type"] == ty}
        if sub:
            est, lo, hi = boot(sub, lambda x: agree_rate(x, "ours_genre"), rng)
            w(f"| {ty} | {len(sub)} | {fmt(est)} | {fmt(lo)}–{fmt(hi)} | {fmt(agree_rate([r for v in sub.values() for r in v], 'human'))} |")
    w("\n*forget*: a model right after learning the session vs the same lineage after a later session.\n")

    # ---------------- reliability -----------------------------------------------------------------------------
    cat = {"a": 0, "b": 1, "tie": 2}
    alpha = kripp_alpha([[cat[r["canon"]] for r in v] for v in by_trial.values()], 3)
    alpha2 = kripp_alpha([[cat[r["canon"]] for r in v if r["canon"] != "tie"] for v in by_trial.values()], 2)
    rep = np.mean([x == y for x, y in rep_pairs]) if rep_pairs else float("nan")
    w("## Reliability\n")
    w(f"- Krippendorff's alpha (nominal; A / B / Can't tell): {fmt(alpha)}; A / B only: {fmt(alpha2)}")
    w(f"- Same answer when a trial is shown again with sides swapped: {fmt(rep)} ({len(rep_pairs)} repeats)\n")

    # ---------------- dance task --------------------------------------------------------------------------------
    dans = [(r, clips[dtrials[r["trial"]]["clip"]]) for d in kept for r in d["responses"] if r["task"] == "dance"]
    if dans:
        w("## Is it dance at all? (checks DanceRate and the reject class)\n")
        yes = np.array([1 - c["scores"]["p_other"] for r, c in dans if r["choice"] == "yes"])
        no = np.array([1 - c["scores"]["p_other"] for r, c in dans if r["choice"] == "no"])
        w(f"AUC of 1 − P(other) for human 'dancing' vs 'not dancing': {fmt(auc(yes, no))} "
          f"({len(yes)} yes, {len(no)} no, {sum(r['choice'] == 'unsure' for r, _ in dans)} not sure).\n")
        w("| Clips | Answers | Dancing | Not dancing | Not sure |")
        w("|---|---|---|---|---|")
        dbins = study["dance_bins"] + [1.0]
        groups = [(f"generated, P(other) in [{dbins[b]:.1f}, {dbins[b + 1]:.1f})", lambda r, c, b=b: dtrials[r["trial"]]["bin"] == b)
                  for b in range(len(dbins) - 1)] + [("real dance", lambda r, c: c["kind"] == "gt")]
        for name, sel in groups:
            ch = [r["choice"] for r, c in dans if sel(r, c)]
            if ch:
                w(f"| {name} | {len(ch)} | {ch.count('yes') / len(ch):.0%} | {ch.count('no') / len(ch):.0%} | {ch.count('unsure') / len(ch):.0%} |")
        w("")

    # ---------------- exploratory --------------------------------------------------------------------------------
    w("## Exploratory\n")
    w("| Genre | Trials | Ours (genre) | Probe (genre) | Kinetic | Human ceiling |")
    w("|---|---|---|---|---|---|")
    for g in sorted({r["genre"] for r in recs}):
        rs = [r for r in recs if r["genre"] == g]
        w(f"| {study['genres'].get(g, g)} | {len({r['trial'] for r in rs})} | "
          + " | ".join(fmt(agree_rate(rs, m)) for m in ("ours_genre", "probe_genre", "kin", "human")) + " |")
    w("")
    w("| Rater experience | Raters | Answers | Ours (genre) | Human ceiling |")
    w("|---|---|---|---|---|")
    for e in ("none", "casual", "trained"):
        rs = [r for r in recs if r["exp"] == e]
        if rs:
            w(f"| {e} | {exp_n[e]} | {len(rs)} | {fmt(agree_rate(rs, 'ours_genre'))} | {fmt(agree_rate(rs, 'human'))} |")
    w("")
    # human style-match by sessions since the music's session was learned: share of comparisons a clip wins
    wins, comps, ours = defaultdict(int), defaultdict(int), defaultdict(list)
    for r in recs:
        if r["canon"] == "tie":
            continue
        t = trials[r["trial"]]
        for side in ("a", "b"):
            c = clips[t[side]]
            stage = c["after"] - c["session"]
            comps[stage] += 1
            wins[stage] += int(r["canon"] == side)
    for c in clips.values():
        if c["kind"] == "gen":
            ours[c["after"] - c["session"]].append(c["scores"]["ours_genre"])
    w("| Sessions since the genre was learned | Comparisons | Human win rate | Mean StyleRetention (genre) of these clips |")
    w("|---|---|---|---|")
    st = sorted(comps)
    for s in st:
        w(f"| {s} | {comps[s]} | {wins[s] / comps[s]:.2f} | {fmt(np.mean(ours[s]))} |")
    if len(st) >= 3:
        w(f"\nSpearman across stages, human win rate vs mean StyleRetention: "
          f"{fmt(spearman([wins[s] / comps[s] for s in st], [np.mean(ours[s]) for s in st]), 2)}")
    w("")

    a.out.write_text("\n".join(L), encoding="utf-8")
    with open(a.out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh)
        cw.writerow(["participant", "trial", "type", "bin", "genre", "experience", "answer", "rt_ms"]
                    + [f"agree_{m}" for m, _ in METRICS] + ["agree_human"])
        for r in recs:
            cw.writerow([r["p"], r["trial"], r["type"], r["bin"], r["genre"], r["exp"], r["canon"], r["rt"]]
                        + [r["agree"][m] for m, _ in METRICS] + [r["agree"]["human"]])
    try:  # Windows consoles may not have the Δ / – characters of the report
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    print("\n".join(L))
    print(f"\n[analyze] wrote {a.out} and {a.out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
