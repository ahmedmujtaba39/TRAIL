"""Build a weak Saudi transition-window training set from extracted poses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from trail.transitions import write_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak source transition windows.")
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/isharah_weak_transitions.npz"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--audit", type=Path, default=None, help="Optional pose_audit.csv for coverage filtering.")
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--windows-per-clip", type=int, default=2)
    parser.add_argument("--boundary-frames", type=int, default=6)
    parser.add_argument("--duration", type=int, default=8)
    args = parser.parse_args()
    poses = sorted(path for path in args.pose_root.glob("*.npy") if path.name != "pose_audit.npy")
    if args.audit is not None:
        with args.audit.open(encoding="utf-8", newline="") as handle:
            quality = {row["clip_id"]: float(row["coverage"]) for row in csv.DictReader(handle) if row["status"] == "ok" and row["coverage"]}
        poses = [path for path in poses if quality.get(path.stem, 0.0) >= args.min_coverage]
    if args.limit is not None:
        poses = poses[:args.limit]
    count = write_windows(poses, args.output, windows_per_clip=args.windows_per_clip, boundary_frames=args.boundary_frames, duration=args.duration)
    print(f"Wrote {count} weak Saudi transition windows to {args.output} from {len(poses)} qualified clips")


if __name__ == "__main__":
    main()
