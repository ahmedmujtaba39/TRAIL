"""Command-line SVO extraction for a verified small pilot before batch runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from trail.svo import extract_left_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract left-view JPEG frames from a JUMLA SVO file.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    extract_left_frames(args.source, args.output, args.ffmpeg, fps=args.fps, max_frames=args.max_frames)
    print(f"Extracted {len(list(args.output.glob('frame_*.jpg')))} frames to {args.output}")


if __name__ == "__main__":
    main()
