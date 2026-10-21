#!/usr/bin/env python3
"""
Render a (T, 22, 3) joints .npy file into an mp4 using existing skeleton utilities.

Example:
  python scripts/render_joints_npy.py \
    --joints_npy tmp/output/aistpp_transition_1_joints_smooth.npy \
    --out_mp4 tmp/output/aistpp_transition_1_joints_smooth.mp4 \
    --fps 30
"""

import argparse
import os
import sys

import numpy as np

# Make `utils.*` importable when running as `python scripts/xxx.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.render_skeleton import get_humanml3d_chains, render_simple_skeleton_video


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--joints_npy", type=str, required=True, help="Path to joints npy (T,22,3).")
    p.add_argument("--out_mp4", type=str, default=None, help="Output mp4 path.")
    p.add_argument("--fps", type=int, default=30, help="Video fps.")
    p.add_argument("--text", type=str, default=None, help="Optional overlay text (if supported).")
    return p.parse_args()


def main():
    args = _parse_args()
    joints = np.load(args.joints_npy)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"Expected joints shape (T,22,3), got {joints.shape}")

    out_mp4 = args.out_mp4
    if out_mp4 is None:
        base, _ = os.path.splitext(args.joints_npy)
        out_mp4 = base + ".mp4"

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)

    chains = get_humanml3d_chains()
    render_simple_skeleton_video(
        data=joints,
        chains=chains,
        out_path=out_mp4,
        fps=args.fps,
        text=args.text,
    )
    print(f"[ok] Rendered {args.joints_npy} -> {out_mp4}")


if __name__ == "__main__":
    main()


