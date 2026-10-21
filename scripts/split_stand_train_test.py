#!/usr/bin/env python3
"""
Randomly split stand segment npy files into train/test lists (9:1 by default).

Writes basenames (optionally without extension) one per line.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("raw_data/babel_272/all_stream_stand_30fps"),
        help="Directory containing stand segment .npy files",
    )
    parser.add_argument(
        "--train_out",
        type=Path,
        default=Path("raw_data/babel_272/train_stand.txt"),
        help="Output path for train list",
    )
    parser.add_argument(
        "--test_out",
        type=Path,
        default=Path("raw_data/babel_272/test_stand.txt"),
        help="Output path for test list",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio (default 0.9 => 9:1 split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split (default 42)",
    )
    parser.add_argument(
        "--keep_ext",
        action="store_true",
        help="If set, write names with extension; otherwise write stem only.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    train_out: Path = args.train_out
    test_out: Path = args.test_out
    train_ratio: float = args.train_ratio
    seed: int = args.seed
    keep_ext: bool = args.keep_ext

    if not input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {input_dir}")
    if not (0.0 < train_ratio < 1.0):
        raise SystemExit(f"train_ratio must be in (0,1): {train_ratio}")

    npy_files = sorted([p for p in input_dir.glob("*.npy") if p.is_file()])
    names = [p.name if keep_ext else p.stem for p in npy_files]
    if not names:
        raise SystemExit(f"No .npy files found in: {input_dir}")

    rng = random.Random(seed)
    rng.shuffle(names)

    n_total = len(names)
    n_train = int(round(n_total * train_ratio))
    n_train = max(1, min(n_total - 1, n_train))  # ensure both splits non-empty

    train = sorted(names[:n_train])
    test = sorted(names[n_train:])

    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    train_out.write_text("\n".join(train) + "\n", encoding="utf-8")
    test_out.write_text("\n".join(test) + "\n", encoding="utf-8")

    print(f"Total: {n_total} | Train: {len(train)} | Test: {len(test)} | seed={seed}")
    print(f"Train file: {train_out}")
    print(f"Test file:  {test_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


