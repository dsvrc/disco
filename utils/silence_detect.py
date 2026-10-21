"""
Silence / near-silence detection utilities.

Why this exists:
- You have "zero" clips generated from near-silent wavs and "test" clips from normal music.
- This module provides a robust condition to classify a wav as silent using RMS (dBFS)
  and the ratio of non-silent frames.

Usage (CLI):
  python -m utils.silence_detect --wav /path/to/audio.wav
  python -m utils.silence_detect --wav /path/to/audio.wav --db_threshold -45 --min_ratio 0.08
  python -m utils.silence_detect --npy /path/to/features.npy

Usage (Python):
  from utils.silence_detect import is_silent_wav
  silent, info = is_silent_wav("a.wav", sr=30*512, hop_length=512)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


@dataclass(frozen=True)
class SilenceInfo:
    # Common
    method: str
    silent: bool

    # Audio metrics (dBFS)
    n_samples: int = 0
    sr: int = 0
    duration_sec: float = 0.0
    frame_length: int = 0
    hop_length: int = 0
    db_threshold: float = 0.0
    min_ratio: float = 0.0
    rms_db_mean: float = float("nan")
    rms_db_p50: float = float("nan")
    rms_db_p95: float = float("nan")
    rms_db_max: float = float("nan")
    non_silent_ratio: float = float("nan")

    # Feature-based metrics
    feats_shape: str = ""
    env_p50: float = float("nan")
    env_nonzero_ratio: float = float("nan")
    mfcc0_mean: float = float("nan")
    mfcc0_p50: float = float("nan")
    mfcc0_p95: float = float("nan")
    peaks_sum: float = float("nan")
    beats_sum: float = float("nan")


def _rms_dbfs(
    y: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    RMS in dBFS (0 dBFS ~= full scale for float audio in [-1, 1]).
    """
    if y.size < frame_length:
        rms = np.sqrt(np.mean(np.square(y), dtype=np.float64)) if y.size else 0.0
        return np.array([20.0 * np.log10(max(float(rms), eps))], dtype=np.float64)

    # Efficient RMS via moving average of squared signal, then hop sampling.
    # center=False, valid windowing.
    y2 = np.square(y, dtype=np.float64)
    window = np.ones(frame_length, dtype=np.float64) / float(frame_length)
    mean_power = np.convolve(y2, window, mode="valid")  # len = n - frame_length + 1
    rms = np.sqrt(mean_power)
    rms = rms[::hop_length] if hop_length > 1 else rms
    return 20.0 * np.log10(np.maximum(rms, eps))


def _wav_to_float32(y: np.ndarray) -> np.ndarray:
    """
    Convert common PCM wav dtypes to float32 in [-1, 1].
    """
    if y.dtype == np.float32 or y.dtype == np.float64:
        y_f = y.astype(np.float32, copy=False)
        return np.clip(y_f, -1.0, 1.0)

    if y.dtype == np.int16:
        return (y.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if y.dtype == np.int32:
        return (y.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    if y.dtype == np.uint8:
        # uint8 wav is unsigned PCM with 128 as "zero"
        return ((y.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)

    # Fallback: best-effort normalization
    y_f = y.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(y_f))) if y_f.size else 1.0
    if max_abs > 0:
        y_f = y_f / max_abs
    return np.clip(y_f, -1.0, 1.0)


def _load_wav_mono_resampled(wav_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    sr, y = wavfile.read(wav_path)
    y = _wav_to_float32(y)
    if y.ndim == 2:
        y = np.mean(y, axis=1, dtype=np.float32)
    if target_sr and sr != target_sr:
        # Polyphase resampling (good quality + fast)
        y = resample_poly(y, up=target_sr, down=sr).astype(np.float32, copy=False)
        sr = int(target_sr)
    return y, int(sr)


def is_silent_wav(
    wav_path: str,
    *,
    sr: int = 30 * 512,
    frame_length: int = 2048,
    hop_length: int = 512,
    db_threshold: float = -45.0,
    min_ratio: float = 0.08,
) -> Tuple[bool, SilenceInfo]:
    """
    Decide if a wav is silent / near-silent using RMS (dBFS).

    Rule:
      silent := (ratio of frames with rms_db > db_threshold) < min_ratio

    Recommended starting thresholds:
      db_threshold=-45 dBFS, min_ratio=0.08

    Returns:
      (silent, info)
    """
    try:
        y, loaded_sr = _load_wav_mono_resampled(wav_path, sr)
    except Exception:
        info = SilenceInfo(method="wav_rms_dbfs", silent=True)
        return True, info

    n_samples = int(y.shape[0])
    duration_sec = float(n_samples / float(loaded_sr)) if loaded_sr else 0.0
    if n_samples == 0 or not np.isfinite(y).all():
        info = SilenceInfo(
            method="wav_rms_dbfs",
            silent=True,
            n_samples=n_samples,
            sr=int(loaded_sr) if loaded_sr else 0,
            duration_sec=duration_sec,
            frame_length=frame_length,
            hop_length=hop_length,
            db_threshold=db_threshold,
            min_ratio=min_ratio,
        )
        return True, info

    rms_db = _rms_dbfs(y, frame_length=frame_length, hop_length=hop_length)
    non_silent_ratio = float(np.mean(rms_db > db_threshold)) if rms_db.size else 0.0
    silent = non_silent_ratio < min_ratio

    info = SilenceInfo(
        method="wav_rms_dbfs",
        silent=bool(silent),
        n_samples=n_samples,
        sr=int(loaded_sr),
        duration_sec=duration_sec,
        frame_length=frame_length,
        hop_length=hop_length,
        db_threshold=db_threshold,
        min_ratio=min_ratio,
        rms_db_mean=float(np.mean(rms_db)) if rms_db.size else float("nan"),
        rms_db_p50=float(np.percentile(rms_db, 50)) if rms_db.size else float("nan"),
        rms_db_p95=float(np.percentile(rms_db, 95)) if rms_db.size else float("nan"),
        rms_db_max=float(np.max(rms_db)) if rms_db.size else float("nan"),
        non_silent_ratio=non_silent_ratio,
    )
    return bool(silent), info


def is_silent_feats(
    npy_path: str,
    *,
    env_index: int = 0,
    peaks_index: int = 33,
    beats_index: int = 34,
    mfcc0_index: int = 1,
    mfcc0_mean_threshold: float = -500.0,
    env_p50_threshold: float = 0.10,
    env_nonzero_ratio_threshold: float = 0.60,
) -> Tuple[bool, SilenceInfo]:
    """
    Heuristic for already-extracted baseline features (shape: [n_frames, slice_len, 35]).

    Rule (heuristic, feature-only):
      - MFCC[0] (a.k.a. c0, strongly correlated with log-energy) becomes a very large negative
        number for near-silence (because log-mel energies collapse).
      - Envelope median/nonzero-ratio also drops for silence, but can be noisy depending on padding.

      silent := (mean(mfcc0) < mfcc0_mean_threshold)
                OR (median(envelope) < env_p50_threshold AND nonzero_ratio(envelope) < env_nonzero_ratio_threshold)

    Defaults are calibrated on your two examples:
      - tmp/zero_feats_32/null.npy (near-silent): mfcc0_mean ~ -1076
      - tmp/test_feats_32/1.npy (normal music):  mfcc0_mean ~ -147
    """
    try:
        feats = np.load(npy_path)
    except Exception:
        info = SilenceInfo(method="feats_envelope_heuristic", silent=True)
        return True, info

    needed = max(env_index, mfcc0_index, peaks_index, beats_index)
    if feats.ndim != 3 or feats.shape[-1] <= needed:
        info = SilenceInfo(method="feats_envelope_heuristic", silent=True, feats_shape=str(getattr(feats, "shape", "")))
        return True, info

    env = feats[:, :, env_index].reshape(-1)
    mfcc0 = feats[:, :, mfcc0_index].reshape(-1)
    peaks = feats[:, :, peaks_index].reshape(-1)
    beats = feats[:, :, beats_index].reshape(-1)

    env_p50 = float(np.percentile(env, 50)) if env.size else 0.0
    env_nonzero_ratio = float(np.mean(env != 0.0)) if env.size else 0.0
    mfcc0_mean = float(np.mean(mfcc0)) if mfcc0.size else 0.0
    mfcc0_p50 = float(np.percentile(mfcc0, 50)) if mfcc0.size else 0.0
    mfcc0_p95 = float(np.percentile(mfcc0, 95)) if mfcc0.size else 0.0

    silent_by_mfcc = mfcc0_mean < mfcc0_mean_threshold
    silent_by_env = (env_p50 < env_p50_threshold) and (env_nonzero_ratio < env_nonzero_ratio_threshold)
    silent = bool(silent_by_mfcc or silent_by_env)

    info = SilenceInfo(
        method="feats_envelope_heuristic",
        silent=bool(silent),
        feats_shape=str(feats.shape),
        env_p50=env_p50,
        env_nonzero_ratio=env_nonzero_ratio,
        mfcc0_mean=mfcc0_mean,
        mfcc0_p50=mfcc0_p50,
        mfcc0_p95=mfcc0_p95,
        peaks_sum=float(np.sum(peaks)) if peaks.size else 0.0,
        beats_sum=float(np.sum(beats)) if beats.size else 0.0,
        db_threshold=mfcc0_mean_threshold,
        min_ratio=env_nonzero_ratio_threshold,
    )
    return bool(silent), info


def framewise_silence_mask(
    feats: np.ndarray,
    *,
    env_index: int = 0,
    mfcc0_index: int = 1,
    peaks_index: int = 33,
    beats_index: int = 34,
    mfcc0_mean_threshold: float = -500.0,
    env_p50_threshold: float = 0.10,
    env_nonzero_ratio_threshold: float = 0.60,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Per-frame silence decision for baseline features.

    Args:
      feats: np.ndarray of shape (n_frames, 32, 35) typically.

    Returns:
      silent_mask: bool array of shape (n_frames,)
      metrics: dict of arrays (each shape (n_frames,))
        - mfcc0_mean: mean of MFCC0 over 32 steps
        - env_p50: median envelope over 32 steps
        - env_nonzero_ratio: ratio(env != 0) over 32 steps
        - peaks_sum: sum of peak onehots over 32 steps
        - beats_sum: sum of beat onehots over 32 steps
        - silent_by_mfcc: bool
        - silent_by_env: bool
    """
    if feats.ndim != 3:
        raise ValueError(f"Expected feats.ndim==3, got shape={getattr(feats, 'shape', None)}")
    if feats.shape[-1] <= max(env_index, mfcc0_index, peaks_index, beats_index):
        raise ValueError(f"Expected feats last dim >= 35, got shape={feats.shape}")

    env = feats[:, :, env_index]
    mfcc0 = feats[:, :, mfcc0_index]
    peaks = feats[:, :, peaks_index]
    beats = feats[:, :, beats_index]

    mfcc0_mean = mfcc0.mean(axis=1)
    env_p50 = np.quantile(env, 0.5, axis=1)
    env_nonzero_ratio = (env != 0.0).mean(axis=1)
    peaks_sum = peaks.sum(axis=1)
    beats_sum = beats.sum(axis=1)

    silent_by_mfcc = mfcc0_mean < mfcc0_mean_threshold
    silent_by_env = (env_p50 < env_p50_threshold) & (env_nonzero_ratio < env_nonzero_ratio_threshold)
    silent_mask = silent_by_mfcc | silent_by_env

    metrics: Dict[str, np.ndarray] = {
        "mfcc0_mean": mfcc0_mean,
        "env_p50": env_p50,
        "env_nonzero_ratio": env_nonzero_ratio,
        "peaks_sum": peaks_sum,
        "beats_sum": beats_sum,
        "silent_by_mfcc": silent_by_mfcc,
        "silent_by_env": silent_by_env,
    }
    return silent_mask.astype(bool), metrics


def framewise_silence_mask_from_npy(
    npy_path: str,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    feats = np.load(npy_path)
    silent_mask, metrics = framewise_silence_mask(feats, **kwargs)
    return silent_mask, metrics, str(feats.shape)


def _to_jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    # Make sure JSON serialization is stable
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detect silent / near-silent audio.")
    parser.add_argument("--wav", type=str, default=None, help="Path to wav file")
    parser.add_argument("--npy", type=str, default=None, help="Path to baseline features .npy (n_frames, slice_len, 35)")
    parser.add_argument("--framewise", action="store_true", help="If set with --npy, output per-frame silence mask + metrics")
    parser.add_argument("--out_csv", type=str, default=None, help="CSV path for --framewise output")

    # wav params
    parser.add_argument("--sr", type=int, default=30 * 512, help="Resample rate used for analysis (default: 30*512)")
    parser.add_argument("--frame_length", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--db_threshold", type=float, default=-45.0, help="Non-silent threshold in dBFS")
    parser.add_argument("--min_ratio", type=float, default=0.08, help="Minimum non-silent frame ratio to be considered non-silent")

    # feats params
    parser.add_argument("--mfcc0_mean_threshold", type=float, default=-500.0)
    parser.add_argument("--env_p50_threshold", type=float, default=0.10)
    parser.add_argument("--env_nonzero_ratio_threshold", type=float, default=0.60)

    args = parser.parse_args(argv)

    if args.wav is None and args.npy is None:
        parser.print_help()
        return 2

    if args.wav is not None:
        silent, info = is_silent_wav(
            args.wav,
            sr=args.sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            db_threshold=args.db_threshold,
            min_ratio=args.min_ratio,
        )
        print(json.dumps(_to_jsonable(asdict(info)), ensure_ascii=False, indent=2))
        return 0 if not silent else 1

    if args.framewise:
        silent_mask, metrics, feats_shape = framewise_silence_mask_from_npy(
            args.npy,
            mfcc0_mean_threshold=args.mfcc0_mean_threshold,
            env_p50_threshold=args.env_p50_threshold,
            env_nonzero_ratio_threshold=args.env_nonzero_ratio_threshold,
        )
        silent_ratio = float(np.mean(silent_mask)) if silent_mask.size else 1.0
        summary = {
            "method": "feats_framewise",
            "feats_shape": feats_shape,
            "n_frames": int(silent_mask.shape[0]),
            "silent_ratio": silent_ratio,
            "mfcc0_mean_threshold": float(args.mfcc0_mean_threshold),
            "env_p50_threshold": float(args.env_p50_threshold),
            "env_nonzero_ratio_threshold": float(args.env_nonzero_ratio_threshold),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        out_csv = args.out_csv
        if out_csv is None:
            out_csv = args.npy.replace(".npy", "_frame_silence.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "frame",
                    "silent",
                    "silent_by_mfcc",
                    "silent_by_env",
                    "mfcc0_mean",
                    "env_p50",
                    "env_nonzero_ratio",
                    "peaks_sum",
                    "beats_sum",
                ]
            )
            for i in range(int(silent_mask.shape[0])):
                w.writerow(
                    [
                        i,
                        int(bool(silent_mask[i])),
                        int(bool(metrics["silent_by_mfcc"][i])),
                        int(bool(metrics["silent_by_env"][i])),
                        float(metrics["mfcc0_mean"][i]),
                        float(metrics["env_p50"][i]),
                        float(metrics["env_nonzero_ratio"][i]),
                        float(metrics["peaks_sum"][i]),
                        float(metrics["beats_sum"][i]),
                    ]
                )
        # exit code: 0 => has some non-silent frames; 1 => all silent
        return 0 if np.any(~silent_mask) else 1

    silent, info = is_silent_feats(
        args.npy,
        mfcc0_mean_threshold=args.mfcc0_mean_threshold,
        env_p50_threshold=args.env_p50_threshold,
        env_nonzero_ratio_threshold=args.env_nonzero_ratio_threshold,
    )
    print(json.dumps(_to_jsonable(asdict(info)), ensure_ascii=False, indent=2))
    return 0 if not silent else 1


if __name__ == "__main__":
    raise SystemExit(main())


# python3 -m utils.silence_detect --npy tmp/zero_feats_32/null.npy --framewise --out_csv tmp/zero_feats_32/null_frame_silence.csv
# python3 -m utils.silence_detect --npy tmp/test_feats_32/1.npy --framewise --out_csv tmp/test_feats_32/1_frame_silence.csv