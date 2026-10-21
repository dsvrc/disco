#!/usr/bin/env python
"""
Evaluate M2D-style metrics on AIST++ ground-truth motions.

This script reads 272-dim motion representations (*.npy), recovers 3D joints,
and reports:
  - DIV_k_gt / DIV_g_gt (diversity of GT features)
  - FSR (optional, physics metric; computed on the provided motion)
  - BAS (optional, requires corresponding wav files)

Note on FID:
  M2DMetrics is designed for comparing generated vs GT distributions (FID_k/FID_g).
  For GT-only evaluation we feed GT as both "ref" and "rst", so FID is 0 by design.
  Use the DIV_*_gt/BAS/FSR outputs as the meaningful GT baselines.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch

# Ensure repo root is on sys.path (so we can import `metrics/` even if it lacks __init__.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.m2d import M2DMetrics  # noqa: E402
from utils.motion_process import convert_motion_to_joints  # noqa: E402


def _maybe_load(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return np.load(str(p))


def _find_wav(wav_dir: Optional[str], stem: str) -> Optional[str]:
    if not wav_dir:
        return None
    p = Path(wav_dir) / f"{stem}.wav"
    return str(p) if p.exists() else None


def _apply_ground_height_for_fsr(
    joints: np.ndarray,
    *,
    mode: str,
    ground_height: float,
) -> np.ndarray:
    """
    M2D's FSR implementation assumes the ground plane is at y=0.
    Many recovered motions have a positive y-offset (e.g. min foot y ~ 0.11m),
    which makes the "foot contact" mask never trigger and FSR becomes 0.0.

    This helper optionally shifts joints in y so that the estimated ground sits at y=0.
    """
    if mode == "none":
        return joints

    # indices used by calculate_foot_skating_ratio
    l_ankle, r_ankle, l_toe, r_toe = 7, 8, 10, 11
    foot_y = np.concatenate(
        [
            joints[:, l_ankle, 1],
            joints[:, r_ankle, 1],
            joints[:, l_toe, 1],
            joints[:, r_toe, 1],
        ],
        axis=0,
    )

    if mode == "auto":
        gh = float(np.min(foot_y))
    elif mode == "fixed":
        gh = float(ground_height)
    else:
        raise ValueError(f"Unknown fsr_ground_mode: {mode}")

    joints_adj = joints.copy()
    joints_adj[:, :, 1] -= gh
    return joints_adj


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--motions_dir",
        default="raw_data/aistpp/motions_272_30fps",
        help="Directory containing GT motion .npy files (T, 272).",
    )
    ap.add_argument(
        "--dim",
        type=int,
        default=272,
        choices=[263, 272],
        help="Motion representation dimensionality.",
    )
    ap.add_argument(
        "--mean_path",
        default="",
        help="Optional mean .npy for de-normalization (will apply motion*std+mean).",
    )
    ap.add_argument(
        "--std_path",
        default="",
        help="Optional std .npy for de-normalization (will apply motion*std+mean).",
    )
    ap.add_argument(
        "--auto_denorm_from",
        default="raw_data/aistpp",
        help="If mean/std not provided, try loading Mean_{dim}.npy and Std_{dim}.npy from here.",
    )
    ap.add_argument("--joints_num", type=int, default=22)
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument(
        "--evaluate_music",
        action="store_true",
        help="Enable BAS (requires wav files).",
    )
    ap.add_argument(
        "--wav_dir",
        default="",
        help="Directory containing wav files named <stem>.wav for BAS.",
    )
    ap.add_argument(
        "--evaluate_physics",
        action="store_true",
        help="Enable physics metric FSR.",
    )
    ap.add_argument(
        "--fsr_ground_mode",
        default="auto",
        choices=["auto", "fixed", "none"],
        help='How to set ground height for FSR contact detection. "auto" shifts y by min(ankle/toe).',
    )
    ap.add_argument(
        "--fsr_ground_height",
        type=float,
        default=0.0,
        help='Used when --fsr_ground_mode=fixed (ground plane at this y before shifting to 0).',
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help='Torch device for recovery/metric bookkeeping (e.g. "cpu", "cuda:0").',
    )
    ap.add_argument(
        "--output_json",
        default="",
        help="Optional path to write metrics JSON.",
    )

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    motions_dir = Path(args.motions_dir)
    if not motions_dir.exists():
        raise FileNotFoundError(f"motions_dir not found: {motions_dir}")

    npy_files = sorted(motions_dir.glob("*.npy"))
    if args.max_files and args.max_files > 0:
        npy_files = npy_files[: args.max_files]
    if len(npy_files) == 0:
        raise RuntimeError(f"No .npy files found under: {motions_dir}")

    # Resolve mean/std
    mean = _maybe_load(args.mean_path)
    std = _maybe_load(args.std_path)
    if mean is None and std is None and args.auto_denorm_from:
        base = Path(args.auto_denorm_from)
        mp = base / f"Mean_{args.dim}.npy"
        sp = base / f"Std_{args.dim}.npy"
        if mp.exists() and sp.exists():
            mean = np.load(str(mp))
            std = np.load(str(sp))

    cfg = SimpleNamespace(
        evaluate_music=bool(args.evaluate_music),
        evaluate_physics=bool(args.evaluate_physics),
    )
    metric = M2DMetrics(cfg=cfg).to(args.device)

    used = 0
    skipped_no_wav = 0

    for f in npy_files:
        motion = np.load(str(f))
        if motion.ndim != 2 or motion.shape[1] != args.dim:
            raise ValueError(f"Unexpected motion shape in {f}: got {motion.shape}, expected (T, {args.dim})")

        if cfg.evaluate_music:
            wav_path = _find_wav(args.wav_dir, f.stem)
            if wav_path is None:
                skipped_no_wav += 1
                continue
            music_paths = [wav_path]
        else:
            music_paths = None

        joints = convert_motion_to_joints(
            motion_data=motion,
            dim=args.dim,
            mean=mean,
            std=std,
            joints_num=args.joints_num,
        )  # (T, J, 3)
        if cfg.evaluate_physics:
            joints = _apply_ground_height_for_fsr(
                joints,
                mode=args.fsr_ground_mode,
                ground_height=args.fsr_ground_height,
            )
        if joints.ndim != 3 or joints.shape[1] != args.joints_num or joints.shape[2] != 3:
            raise ValueError(f"Recovered joints have unexpected shape for {f}: {joints.shape}")

        T = int(joints.shape[0])
        joints_t = torch.from_numpy(joints).float().unsqueeze(0).to(args.device)  # (1, T, J, 3)

        # GT-only baseline: feed GT as both ref and rst.
        metric.update(
            joints_ref=joints_t,
            joints_rst=joints_t,
            lengths_ref=[T],
            lengths_rst=[T],
            music_paths=music_paths,
        )
        used += 1

    metrics: Dict[str, float] = {
        k: float(v.detach().cpu().item()) for k, v in metric.compute(sanity_flag=False).items()
    }
    metrics["_files_total"] = int(len(npy_files))
    metrics["_files_used"] = int(used)
    metrics["_files_skipped_no_wav"] = int(skipped_no_wav)
    metrics["_motions_dir"] = str(motions_dir)
    if mean is not None and std is not None:
        metrics["_denorm"] = True
    else:
        metrics["_denorm"] = False

    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


