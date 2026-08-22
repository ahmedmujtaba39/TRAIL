"""Build a weak Saudi transition-window training set from extracted poses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from trail.transitions import write_windows
from trail.handshape import load_handshape_classifier
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak source transition windows.")
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/isharah_weak_transitions.npz"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--audit", type=Path, default=None, help="Optional pose_audit.csv for coverage filtering.")
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--min-hand-coverage", type=float, default=0.0, help="Require at least this coverage for either detected hand when using hand_pose_audit.csv.")
    parser.add_argument("--windows-per-clip", type=int, default=2)
    parser.add_argument("--boundary-frames", type=int, default=6)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--handshape-checkpoint", type=Path, default=None, help="Optional soft handshape classifier; requires 75-joint inputs.")
    args = parser.parse_args()
    poses = sorted([*args.pose_root.glob("*.npy"), *args.pose_root.glob("*.npz")])
    if args.audit is not None:
        with args.audit.open(encoding="utf-8", newline="") as handle:
            quality = {}
            for row in csv.DictReader(handle):
                if row["status"] != "ok":
                    continue
                body = float(row.get("coverage") or row.get("body_coverage") or 0.0)
                hand = max(float(row.get("left_hand_coverage") or 0.0), float(row.get("right_hand_coverage") or 0.0))
                quality[row["clip_id"]] = (body, hand)
        poses = [path for path in poses if quality.get(path.stem, (0.0, 0.0))[0] >= args.min_coverage and quality.get(path.stem, (0.0, 0.0))[1] >= args.min_hand_coverage]
    if args.limit is not None:
        poses = poses[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handshape = load_handshape_classifier(args.handshape_checkpoint, device) if args.handshape_checkpoint else None
    count = write_windows(poses, args.output, windows_per_clip=args.windows_per_clip, boundary_frames=args.boundary_frames, duration=args.duration, handshape_model=handshape, device=device)
    print(f"Wrote {count} weak Saudi transition windows to {args.output} from {len(poses)} qualified clips")


if __name__ == "__main__":
    main()
