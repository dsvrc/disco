"""
Generate near-silence wav files whose durations match motion sequences.

For each motion file (e.g. val_seq_123.npy) with shape (T, D) at 30 fps,
we generate a wav with duration = T / fps seconds.

Output naming:
  val_seq_123.npy -> val_seq_123.wav

Example:
  python3 scripts/generate_near_silence_wavs.py \
    --motion_dir raw_data/babel_272/all_stream_30fps \
    --out_dir raw_data/babel_272/near_silence_wavs_15360 \
    --fps 30 --sr 15360 --noise_dbfs -75 --seed 123
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.io import wavfile


def dbfs_to_amp(dbfs: float) -> float:
    """Convert dBFS to linear amplitude where 0 dBFS == 1.0 peak scale."""
    return float(10.0 ** (dbfs / 20.0))


def make_near_silence(
    n_samples: int,
    *,
    noise_dbfs: float = -75.0,
    dc_offset_dbfs: Optional[float] = None,
    fade_ms: float = 10.0,
    sr: int = 15360,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create near-silence audio (float32 in [-1, 1]).

    - noise_dbfs controls RMS of white noise (Gaussian).
    - optional dc_offset_dbfs adds a tiny DC component (disabled by default).
    - fade in/out to avoid clicks.
    """
    if n_samples <= 0:
        return np.zeros((0,), dtype=np.float32)

    if np.isneginf(noise_dbfs):
        y = np.zeros((n_samples,), dtype=np.float32)
    else:
        noise_rms = dbfs_to_amp(noise_dbfs)
        y = rng.normal(loc=0.0, scale=noise_rms, size=(n_samples,)).astype(np.float32)

    if dc_offset_dbfs is not None and not np.isneginf(dc_offset_dbfs):
        y = y + np.float32(dbfs_to_amp(dc_offset_dbfs))

    # fade
    fade_len = int(round((fade_ms / 1000.0) * sr))
    fade_len = max(0, min(fade_len, n_samples // 2))
    if fade_len > 0:
        fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        y[:fade_len] *= fade
        y[-fade_len:] *= fade[::-1]

    # hard clip just in case
    np.clip(y, -1.0, 1.0, out=y)
    return y


def float_to_int16(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).round().astype(np.int16)


def motion_len_and_fps(motion_path: str, *, fps: float) -> Tuple[int, float]:
    arr = np.load(motion_path, mmap_mode="r")
    if arr.ndim < 1:
        raise ValueError(f"Unexpected motion array shape: {arr.shape} ({motion_path})")
    t = int(arr.shape[0])
    duration = float(t / float(fps))
    return t, duration


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate near-silence wavs matching motion lengths.")
    parser.add_argument("--motion_dir", type=str, required=True, help="Directory containing motion .npy files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for wav files")
    parser.add_argument(
        "--id_list",
        type=str,
        default=None,
        help="Optional text file containing one motion id per line (e.g. train_seq_123). If set, only generate for these ids.",
    )

    parser.add_argument("--fps", type=float, default=30.0, help="Motion fps (default: 30)")
    parser.add_argument("--sr", type=int, default=15360, help="Wav sample rate (default: 15360)")
    parser.add_argument("--noise_dbfs", type=float, default=-75.0, help="Noise RMS in dBFS (default: -75)")
    parser.add_argument("--dc_offset_dbfs", type=float, default=None, help="Optional tiny DC offset in dBFS")
    parser.add_argument("--fade_ms", type=float, default=10.0, help="Fade in/out milliseconds (default: 10)")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = no limit)")
    parser.add_argument("--pattern", type=str, default="*.npy", help="Glob pattern used only when --id_list is not set (default: *.npy)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing wavs")
    parser.add_argument("--strict", action="store_true", help="If set, fail when any id in --id_list is missing")

    args = parser.parse_args()

    motion_dir = Path(args.motion_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    requested_ids = 0
    unique_ids = 0
    missing: list[str] = []
    if args.id_list is not None:
        ids: list[str] = []
        with open(args.id_list, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                ids.append(s)
        requested_ids = len(ids)
        unique_ids = len(set(ids))
        if args.limit and args.limit > 0:
            ids = ids[: args.limit]

        motion_paths = []
        seen = set()
        for mid in ids:
            if mid in seen:
                continue
            seen.add(mid)
            mp = motion_dir / f"{mid}.npy"
            if not mp.exists():
                missing.append(mid)
                continue
            motion_paths.append(mp)
        if not motion_paths:
            msg = f"No motion files found from id_list={args.id_list} under {motion_dir}"
            if missing:
                msg += f" (missing_count={len(missing)})"
            raise SystemExit(msg)
    else:
        motion_paths = sorted(motion_dir.glob(args.pattern))
        if args.limit and args.limit > 0:
            motion_paths = motion_paths[: args.limit]
        if not motion_paths:
            raise SystemExit(f"No motion files matched {motion_dir / args.pattern}")

    wrote = 0
    skipped = 0
    for mp in motion_paths:
        stem = mp.stem  # val_seq_123
        wav_path = out_dir / f"{stem}.wav"
        if wav_path.exists() and not args.overwrite:
            skipped += 1
            continue

        t, duration = motion_len_and_fps(str(mp), fps=args.fps)
        n_samples = int(round(duration * args.sr))

        y = make_near_silence(
            n_samples,
            noise_dbfs=args.noise_dbfs,
            dc_offset_dbfs=args.dc_offset_dbfs,
            fade_ms=args.fade_ms,
            sr=args.sr,
            rng=rng,
        )
        wavfile.write(str(wav_path), args.sr, float_to_int16(y))
        wrote += 1

        if wrote <= 3:
            print(f"Wrote {wav_path}  T={t}  duration={duration:.3f}s  samples={n_samples}")

    if args.id_list is not None and missing:
        print(f"Missing motion npy for {len(missing)} ids (first 10): {missing[:10]}")
        if args.strict:
            raise SystemExit(3)

    if args.id_list is not None:
        found = len(motion_paths)
        print(
            f"Summary: requested_lines={requested_ids} unique_ids={unique_ids} found_npy={found} "
            f"missing_npy={len(missing)} wrote={wrote} skipped_existing={skipped} out_dir={out_dir}"
        )
    else:
        print(f"Done. wrote={wrote} skipped={skipped} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python3 scripts/generate_near_silence_wavs.py \
#   --motion_dir raw_data/babel_272/all_stream_stand_30fps \
#   --out_dir raw_data/babel_272/near_silence_wavs_15360_stand \
#   --fps 30 --sr 15360 --noise_dbfs -75 --seed 123