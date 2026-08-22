"""Extract a hand-aware TRAIL landmark sequence from a folder of JPEG frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from trail.hand_pose import save_hand_pose_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    audit = save_hand_pose_sequence(args.frames, args.pose_model, args.hand_model, args.output, fps=args.fps)
    print("Saved hand-aware landmarks to " + str(args.output))
    print(", ".join(f"{key}={value:.1%}" for key, value in audit.items()))


if __name__ == "__main__":
    main()
