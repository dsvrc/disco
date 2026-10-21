#!/usr/bin/env python3
"""
Parallel driver for utils/librosa_preprocess.py: causal windowed librosa music features (T, window, 35)
for a list of sequences, one process per file.

Example (all sequences of a CL benchmark):
  python scripts/librosa_preprocess_parallel.py --wav_dir raw_data/aistpp/wavs \
      --dest raw_data/aistpp/baseline_feats_window_64 --window_size 64 \
      --lists raw_data/cl_sessions/aistpp_genre5/joint_train.txt raw_data/cl_sessions/aistpp_genre5/joint_val.txt \
              raw_data/cl_sessions/aistpp_genre5/joint_test.txt
"""

from __future__ import annotations

import os

# one thread per worker process: otherwise every worker's numpy / numba spawns a full thread pool and
# dozens of workers thrash the CPU (observed: extraction stalls at ~2/3)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())


def _one(wav: str, dest: str, window: int) -> str:
    from utils.librosa_preprocess import extract_audio_features

    feats, path = extract_audio_features(wav, skip_completed=True, dest_dir=dest, slice_length=window)
    if feats is not None:
        np.save(path, feats.astype(np.float32))
    return Path(wav).stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--window_size", type=int, default=64)
    ap.add_argument("--lists", type=Path, nargs="*", default=[], help="name lists (default: every wav)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    names = sorted({l.strip() for p in a.lists for l in p.open() if l.strip()}) if a.lists else \
        sorted(p.stem for p in a.wav_dir.glob("*.wav"))
    todo = [str(a.wav_dir / f"{n}.wav") for n in names if not (a.dest / f"{n}.npy").exists()]
    print(f"{len(names)} sequences, {len(todo)} to extract -> {a.dest}", flush=True)
    a.dest.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(partial(_one, dest=str(a.dest), window=a.window_size), todo)):
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
