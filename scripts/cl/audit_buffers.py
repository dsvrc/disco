#!/usr/bin/env python3
"""
Audit the replay buffers of a continual-learning benchmark: is every buffer really 10 % of the right session's training
data, made of valid clips, with usable weights? Every check prints PASS / WARN / FAIL with the numbers behind it.

Checks per buffer (buffers/<benchmark>/<method>/s{j}.json):
  file        absolute path, method / session recorded in the file
  session     every clip's sequence is in session j's TRAIN list (no other session, no val / test leakage)
  genres      genre mix of the buffer vs the session's genres
  budget      #clips == budget == round(ratio * #candidates), with #candidates recomputed here from the token / feature
              / music files under the loader's length limits (independent of the builder)
  tokens      share of the session's training tokens kept (about the ratio when clips are full length)
  valid       0 <= start, start + length <= sequence length, length > 0, no duplicate clips
  weights     no zero / negative weights (a zero-weight clip takes a replay slot but adds nothing), effective sample
              size (sum w)^2 / sum w^2
  coverage    sequences / music pieces / dancers of the session that appear in the buffer
  motion      motion energy of the buffered clips (mean squared frame-to-frame change of their VAE tokens) against the
              session's clips, and the share of replay weight on its least / most moving quarter (25 % / 25 % for a
              representative buffer): over-weighted still clips teach freezing, over-weighted fast ones raise FID_k

  python scripts/cl/audit_buffers.py                      # every method under buffers/aistpp_genre5
  python scripts/cl/audit_buffers.py --methods tmr gcr    # some methods
  python scripts/cl/audit_buffers.py --sessions 0         # every method's session-0 buffer
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from utils.cl_replay import tile_clips
from utils.initialize import load_config


def read_list(p: Path) -> list[str]:
    return [l.strip() for l in p.open() if l.strip()]


def candidates(cfg, list_path: Path, clip_len: int) -> tuple[dict, list]:
    """Token length of every loadable training sequence (AistPPDataset's filters) and the tiled candidate clips."""
    d = list_path.parent
    tok_dir, mus_dir = d / cfg.data.token_path, d / cfg.data.music_path
    feat_dir = d / cfg.data.feature_path if cfg.data.get("feature_path") else None
    lo, hi = int(cfg.data.min_length), int(cfg.data.max_length)
    lengths = {}
    for n in read_list(list_path):
        tp, mp = tok_dir / f"{n}.npy", mus_dir / f"{n}.npy"
        if not (tp.exists() and mp.exists()):
            continue
        if feat_dir is not None:
            fp = feat_dir / f"{n}.npy"
            if not fp.exists() or not lo <= np.load(fp, mmap_mode="r").shape[0] <= hi:
                continue
        if not lo <= np.load(mp, mmap_mode="r").shape[0] <= hi:
            continue
        lengths[n] = int(np.load(tp, mmap_mode="r").shape[0])
    return lengths, [(n, s, l) for n in sorted(lengths) for s, l in tile_clips(lengths[n], clip_len)]


def motion_energy(tok_dir: Path, cands: list) -> dict:
    """Mean squared frame-to-frame change of the VAE tokens of every candidate clip, keyed by (name, start): the
    motion energy of the diffusion target, ~0 for a held pose."""
    seqs, out = {}, {}
    for n, s, l in cands:
        if n not in seqs:
            seqs[n] = np.load(tok_dir / f"{n}.npy").astype(np.float64)
        x = seqs[n][s:s + l]
        out[(n, s)] = float((np.diff(x.reshape(len(x), -1), axis=0) ** 2).sum(1).mean()) if len(x) > 1 else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--buffers", default=None, help="default buffers/<benchmark>")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--sessions", type=int, nargs="*", default=None, help="only these sessions' buffers (default all)")
    ap.add_argument("--ratio", type=float, default=0.10)
    ap.add_argument("--override", action="append", default=[], help="config overrides key=value (repeatable)")
    a = ap.parse_args()
    overrides = dict(o.split("=", 1) for o in a.override)
    root = Path(a.buffers or f"buffers/{a.benchmark}")
    if not root.exists():
        print(f"FAIL no buffer folder {root.resolve()} (buffers are written by scripts/build_replay_buffer.py, "
              f"called from scripts/cl/run_method.sh, under the repo root)")
        return 1
    methods = a.methods or sorted(p.name for p in root.iterdir() if p.is_dir())
    raw = load_config(f"configs/cl/ldf_{a.benchmark}_s0.yaml", overrides).get("dirs.raw_data")
    bench = Path(str(raw)) / "cl_sessions" / a.benchmark
    manifest = json.loads((bench / "manifest.json").read_text())
    n_fail = n_warn = 0
    cache, energy = {}, {}
    rows = []

    def report(level, what, msg):
        nonlocal n_fail, n_warn
        n_fail += level == "FAIL"
        n_warn += level == "WARN"
        print(f"  {level:4s} {what:9s} {msg}")

    for m in methods:
        files = sorted((root / m).glob("s*.json"), key=lambda p: int(p.stem[1:]))
        files = [f for f in files if a.sessions is None or int(f.stem[1:]) in a.sessions]
        print(f"\n=== {m}: {len(files)} buffer file(s) in {(root / m).resolve()}")
        if not files:
            n_fail += 1
            print("  FAIL file      no s*.json: this method never built a buffer")
            continue
        for f in files:
            j = int(f.stem[1:])
            b = json.loads(f.read_text())
            clips = b["clips"]
            print(f"--- {m} s{j}: {f.resolve()}  (method in file: {b.get('method')}, session in file: {b.get('session')}, "
                  f"built from {b.get('ckpt')})")
            # a runner method stores its selection method (tmr_dance_sel -> tmr_dance, random_distill -> random)
            if b.get("session") != j or not str(m).startswith(str(b.get("method"))):
                report("FAIL", "file", f"file says method {b.get('method')} session {b.get('session')}")
            cfg = load_config(f"configs/cl/ldf_{a.benchmark}_s{j}.yaml", overrides)
            clip_len = int(cfg.data.train_max_tokens)
            list_path = Path(str(cfg.data.train_meta_paths[0]))
            if j not in cache:
                cache[j] = candidates(cfg, list_path, clip_len)
            lengths, cands = cache[j]
            train_names = set(read_list(bench / f"s{j}" / "train.txt"))
            other = {}
            for k in range(len(manifest["sessions"])):
                if k != j:
                    for split in ("train", "val", "test"):
                        for n in read_list(bench / f"s{k}" / f"{split}.txt"):
                            other[n] = f"s{k}/{split}"
            for split in ("val", "test"):
                for n in read_list(bench / f"s{j}" / f"{split}.txt"):
                    other[n] = f"s{j}/{split}"

            # session
            names = [c["name"] for c in clips]
            wrong = collections.Counter(other.get(n, "unknown") for n in names if n not in train_names)
            report("FAIL" if wrong else "PASS", "session",
                   f"{len(names) - sum(wrong.values())}/{len(names)} clips from s{j} train" + (f"; others: {dict(wrong)}" if wrong else ""))
            # genres
            genres = collections.Counter(n[1:3] for n in names)
            expected = set(manifest["sessions"][j]["groups"])
            report("PASS" if set(genres) <= expected else "FAIL", "genres",
                   f"buffer {dict(genres)} vs session groups {sorted(expected)}")
            # budget
            n_c, budget = len(cands), max(1, round(a.ratio * len(cands)))
            level = "PASS" if len(clips) == budget == b.get("budget") and b.get("n_candidates") == n_c else (
                "WARN" if len(clips) >= 0.9 * budget else "FAIL")
            report(level, "budget", f"{len(clips)} clips, file budget {b.get('budget')} of {b.get('n_candidates')} candidates; "
                                    f"recomputed: {budget} = {a.ratio:.0%} of {n_c} candidates ({len(lengths)} sequences)")
            # tokens
            cand_tokens = sum(l for _, _, l in cands)
            buf_tokens = sum(c["length"] for c in clips)
            share = buf_tokens / max(cand_tokens, 1)
            report("PASS" if 0.6 * a.ratio <= share <= 1.5 * a.ratio else "WARN", "tokens",
                   f"{buf_tokens}/{cand_tokens} tokens = {share:.1%} of the session's training tokens "
                   f"({buf_tokens / 7.5 / 60:.1f} of {cand_tokens / 7.5 / 60:.1f} min)")
            # valid
            bad = [c for c in clips if c["name"] not in lengths or c["start"] < 0 or c["length"] <= 0
                   or c["start"] + c["length"] > lengths.get(c["name"], -1)]
            dup = len(clips) - len({(c["name"], c["start"]) for c in clips})
            report("FAIL" if bad or dup else "PASS", "valid",
                   f"{len(bad)} out-of-range clips, {dup} duplicates; clip lengths {min(c['length'] for c in clips)}.."
                   f"{max(c['length'] for c in clips)} tokens" + (f"; e.g. {bad[:2]}" if bad else ""))
            # weights
            w = np.array([c.get("weight", 1.0) for c in clips], float)
            zero = int((w <= 1e-8).sum())
            ess = w.sum() ** 2 / max((w ** 2).sum(), 1e-12)
            level = "FAIL" if zero else ("WARN" if ess < 0.5 * len(w) else "PASS")
            report(level, "weights", f"{zero} zero/negative, mean {w.mean():.3f} min {w.min():.3f} max {w.max():.3f}, "
                                     f"effective sample size {ess:.1f} of {len(w)}")
            # coverage
            seqs = set(names)
            part = lambda n, i: n.split("_")[i] if len(n.split("_")) > 4 else "?"  # gLO_sBM_cAll_d13_mLO2_ch02
            music, dancers = {part(n, 4) for n in names}, {part(n, 3) for n in names}
            pool_music, pool_dancers = {part(n, 4) for n in lengths}, {part(n, 3) for n in lengths}
            report("INFO", "coverage", f"{len(seqs)}/{len(lengths)} sequences, {len(music)}/{len(pool_music)} music pieces, "
                                       f"{len(dancers)}/{len(pool_dancers)} dancers")
            # motion
            if j not in energy:
                energy[j] = motion_energy(list_path.parent / cfg.data.token_path, cands)
            pool = np.array(list(energy[j].values()))
            q1, q3 = np.quantile(pool, [0.25, 0.75])
            e = np.array([energy[j].get((c["name"], c["start"]), np.nan) for c in clips])
            wn = np.where(np.isnan(e), 0.0, w) / max(float(w[~np.isnan(e)].sum()), 1e-12)
            report("INFO", "motion", f"token motion energy: session clips {pool.mean():.4f} (median {np.median(pool):.4f}); "
                                     f"buffer {np.nanmean(e):.4f}, replay-weighted {np.nansum(wn * e):.4f}; replay weight "
                                     f"on the session's least / most moving quarter {wn[e <= q1].sum():.0%} / "
                                     f"{wn[e >= q3].sum():.0%} (25% / 25% when representative)"
                                     + (f"; {int(np.isnan(e).sum())} clips not on the candidate grid" if np.isnan(e).any() else ""))
            rows.append((m, j, len(clips), budget, share, zero, ess))

    print("\n=== summary (method, session, clips, recomputed budget, token share, zero weights, effective size)")
    for r in rows:
        print(f"  {r[0]:12s} s{r[1]}  {r[2]:4d} / {r[3]:4d}  {r[4]:6.1%}  zero {r[5]:3d}  ESS {r[6]:6.1f}")
    print(f"\n{n_fail} FAIL, {n_warn} WARN")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
