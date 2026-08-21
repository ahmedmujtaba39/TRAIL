"""Extract a normalized pose sequence from already-decoded frame images."""

from __future__ import annotations

import argparse
from pathlib import Path

from trail.pose import save_pose_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a TRAIL pose sequence from JPEG frames.")
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    coverage = save_pose_sequence(args.frames, args.model, args.output, fps=args.fps)
    print(f"Wrote {args.output} (detection coverage: {coverage:.1%})")


if __name__ == "__main__":
    main()
