#!/usr/bin/env python3
"""
Training data for the frozen motion VAE of a continual-learning benchmark: session 0 + BABEL-272.

A VAE trained on session 0 alone erases the style of every unseen genre (style retention of its
reconstructions on aistpp_genre5 S1-S4: 0.09-0.27 vs 0.79-0.87 for the real dance; most windows are
classified as LA Hip-hop / Lock), so later sessions could never be learned. BABEL-272 (general AMASS motion,
already in the 272-dim representation, no AIST++) gives the tokenizer broad coverage without touching
sessions 1..K.

Writes (lists live next to the features, as the repo's loaders require):
  raw_data/babel_272/cl_vae_babel_train.txt         random subset of train_stream (feature_path train_stream)
  raw_data/babel_272/cl_vae_babel_val.txt           all of val_stream            (feature_path val_stream)
  raw_data/aistpp/cl_<benchmark>_s0_train_x<R>.txt  session-0 train list repeated R times (sampling weight)
  raw_data/cl_sessions/<benchmark>/vae_{Mean,Std}_272.npy   normalisation over S0 train + the BABEL subset

Usage:
  python scripts/make_cl_vae_data.py --benchmark aistpp_genre5 --babel_train 6000 --s0_repeat 10

--alltrain instead writes the data of the frozen tokenizer actually used: the TRAIN split of all sessions
(raw_data/aistpp/cl_<benchmark>_joint_{train,val,test}.txt and alltrain_{Mean,Std}_272.npy). The S0 + BABEL VAE
above also failed on unseen genres (S1-S4 reconstruction style retention 0.15-0.42 after 2.5k steps).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_cl_sessions import grouped_std_272, read_list, write_list


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--babel_root", type=Path, default=Path("raw_data/babel_272"))
    ap.add_argument("--babel_train", type=int, default=6000, help="random train_stream sequences to use")
    ap.add_argument("--s0_repeat", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alltrain", action="store_true", help="tokenizer data = train split of all sessions")
    a = ap.parse_args()

    if a.alltrain:
        root = Path("raw_data/cl_sessions") / a.benchmark
        for split in ("train", "val", "test"):
            write_list(Path("raw_data/aistpp") / f"cl_{a.benchmark}_joint_{split}.txt", read_list(root / f"joint_{split}.txt"))
        names = read_list(root / "joint_train.txt")
        X = np.concatenate([np.load(Path("raw_data/aistpp/motions_272_30fps") / f"{n}.npy") for n in names], 0)
        mean, std = grouped_std_272(X.astype(np.float64))
        np.save(root / "alltrain_Mean_272.npy", mean)
        np.save(root / "alltrain_Std_272.npy", std)
        print(f"{len(names)} train sequences of all sessions, {len(X)} frames -> {root}/alltrain_*_272.npy")
        return 0

    train_names = sorted(p.stem for p in (a.babel_root / "train_stream").glob("*.npy"))
    val_names = sorted(p.stem for p in (a.babel_root / "val_stream").glob("*.npy"))
    sub = sorted(random.Random(a.seed).sample(train_names, min(a.babel_train, len(train_names))))
    write_list(a.babel_root / "cl_vae_babel_train.txt", sub)
    write_list(a.babel_root / "cl_vae_babel_val.txt", val_names)

    s0 = read_list(Path("raw_data/cl_sessions") / a.benchmark / "s0" / "train.txt")
    write_list(Path("raw_data/aistpp") / f"cl_{a.benchmark}_s0_train_x{a.s0_repeat}.txt", s0 * a.s0_repeat)

    frames = [np.load(Path("raw_data/aistpp/motions_272_30fps") / f"{n}.npy").astype(np.float32) for n in s0]
    frames += [np.load(a.babel_root / "train_stream" / f"{n}.npy").astype(np.float32) for n in sub]
    frames += [np.load(a.babel_root / "val_stream" / f"{n}.npy").astype(np.float32) for n in val_names]
    mean, std = grouped_std_272(np.concatenate(frames, 0).astype(np.float64))
    out = Path("raw_data/cl_sessions") / a.benchmark
    np.save(out / "vae_Mean_272.npy", mean)
    np.save(out / "vae_Std_272.npy", std)
    print(f"BABEL: {len(sub)} train_stream + {len(val_names)} val_stream sequences; S0: {len(s0)} x {a.s0_repeat}; "
          f"normalisation over {sum(len(f) for f in frames)} frames -> {out}/vae_*_272.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
