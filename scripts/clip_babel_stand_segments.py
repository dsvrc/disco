#!/usr/bin/env python3
"""
Clip stand segments from BABEL 30FPS stream npy files based on text annotations.

Rule:
- Scan every *.txt in text_dir
- For ANY line containing needle (default "#stand/"), extract the trailing "#start#end" seconds
- Convert to frame indices at fps (default 30) using round(sec * fps)
- Clip motion[start_frame:end_frame] (end is exclusive) from corresponding *.npy in motion_dir
- Save each segment as an individual .npy file in output_dir
- Write a JSONL metadata file describing every saved segment
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm


TIME_RE = re.compile(r"#\s*(-?\d+(?:\.\d+)?)\s*#\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class StandSpan:
    seq_name: str
    line_idx: int  # 0-based within txt
    raw_line: str
    start_sec: float
    end_sec: float


def iter_stand_spans(txt_path: Path, needle: str) -> Iterable[StandSpan]:
    seq_name = txt_path.stem
    try:
        lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return
    for i, line in enumerate(lines):
        if needle not in line:
            continue
        m = TIME_RE.search(line)
        if not m:
            continue
        start_sec = float(m.group(1))
        end_sec = float(m.group(2))
        yield StandSpan(
            seq_name=seq_name,
            line_idx=i,
            raw_line=line,
            start_sec=start_sec,
            end_sec=end_sec,
        )


def sec_to_frame(sec: float, fps: float) -> int:
    # BABEL timestamps are typically aligned to 1/fps; round is robust to float printing (e.g. 2.166667)
    return int(round(sec * fps))


def safe_clip(arr: np.ndarray, start_f: int, end_f: int) -> Optional[np.ndarray]:
    n = int(arr.shape[0])
    s = max(0, min(start_f, n))
    e = max(0, min(end_f, n))
    if e <= s:
        return None
    return arr[s:e]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_dir",
        type=Path,
        default=Path("raw_data/babel_272/all_stream_text_30fps"),
        help="Directory containing *_seq_*.txt annotation files",
    )
    parser.add_argument(
        "--motion_dir",
        type=Path,
        default=Path("raw_data/babel_272/all_stream_30fps"),
        help="Directory containing *_seq_*.npy motion files (30FPS)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("raw_data/babel_272/all_stream_stand_30fps"),
        help="Output directory to write clipped stand segments (.npy)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Motion FPS (default 30)")
    parser.add_argument(
        "--needle",
        type=str,
        default="#stand/",
        help='Substring to match stand lines (default "#stand/")',
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=1,
        help="Minimum frames to keep a clip (after clamping). Default 1.",
    )
    parser.add_argument(
        "--limit_txt",
        type=int,
        default=0,
        help="If >0, only process the first N txt files (for quick testing).",
    )
    args = parser.parse_args()

    text_dir: Path = args.text_dir
    motion_dir: Path = args.motion_dir
    output_dir: Path = args.output_dir
    fps: float = args.fps
    needle: str = args.needle
    min_frames: int = args.min_frames
    limit_txt: int = args.limit_txt

    if not text_dir.is_dir():
        raise SystemExit(f"text_dir is not a directory: {text_dir}")
    if not motion_dir.is_dir():
        raise SystemExit(f"motion_dir is not a directory: {motion_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "stand_segments.jsonl"

    txt_files = sorted(text_dir.glob("*.txt"))
    if limit_txt and limit_txt > 0:
        txt_files = txt_files[:limit_txt]

    written = 0
    missing_motion = 0
    bad_spans = 0

    with meta_path.open("w", encoding="utf-8") as mf:
        for txt_path in tqdm(txt_files, desc="Clipping stand segments"):
            seq_name = txt_path.stem
            motion_path = motion_dir / f"{seq_name}.npy"
            if not motion_path.exists():
                # Still parse spans to count, but cannot clip
                for _ in iter_stand_spans(txt_path, needle):
                    missing_motion += 1
                continue

            # mmap to avoid loading full arrays when possible
            motion = np.load(motion_path, mmap_mode="r")

            seg_idx = 0
            for span in iter_stand_spans(txt_path, needle):
                if span.end_sec < span.start_sec:
                    bad_spans += 1
                    continue
                start_f = sec_to_frame(span.start_sec, fps)
                end_f = sec_to_frame(span.end_sec, fps)

                clip = safe_clip(motion, start_f, end_f)
                if clip is None or int(clip.shape[0]) < min_frames:
                    bad_spans += 1
                    continue

                out_name = f"{seq_name}_stand_{seg_idx:02d}_sf{start_f}_ef{end_f}.npy"
                out_path = output_dir / out_name
                np.save(out_path, np.asarray(clip))

                rec = {
                    "seq": seq_name,
                    "segment_idx": seg_idx,
                    "txt_file": str(txt_path),
                    "motion_file": str(motion_path),
                    "needle": needle,
                    "line_idx": span.line_idx,
                    "raw_line": span.raw_line,
                    "start_sec": span.start_sec,
                    "end_sec": span.end_sec,
                    "fps": fps,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "saved_file": str(out_path),
                    "num_frames": int(clip.shape[0]),
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                seg_idx += 1

    print(f"Output dir: {output_dir}")
    print(f"Metadata: {meta_path}")
    print(f"Saved clips: {written}")
    print(f"Missing motion files for spans: {missing_motion}")
    print(f"Skipped bad/empty spans: {bad_spans}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


