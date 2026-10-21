#!/usr/bin/env python3
"""
Find all *.txt under a directory whose 2nd line contains a given substring,
and write their basenames (without ".txt") to an output file.

Default use-case: scan BABEL stream text files and collect sequences whose
second line contains "#stand/" into raw_data/babel_272/stand.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _second_line_contains(path: Path, needle: str) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            _ = f.readline()
            line2 = f.readline()
    except OSError:
        return False
    return bool(line2) and (needle in line2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("raw_data/babel_272/all_stream_text_30fps"),
        help="Directory containing *_seq_*.txt files",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("raw_data/babel_272/stand.txt"),
        help="Output text file to write (one sequence per line, no .txt)",
    )
    parser.add_argument(
        "--needle",
        type=str,
        default="#stand/",
        help='Substring to search for in the second line (default: "#stand/")',
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_file: Path = args.output_file
    needle: str = args.needle

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir does not exist or is not a directory: {input_dir}")

    matches: list[str] = []
    for p in input_dir.glob("*.txt"):
        if _second_line_contains(p, needle):
            matches.append(p.stem)

    matches.sort()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(matches) + ("\n" if matches else ""), encoding="utf-8")

    print(f"Wrote {len(matches)} sequences to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




