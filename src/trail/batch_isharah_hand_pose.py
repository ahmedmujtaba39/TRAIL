"""Disk-bounded hand-aware extraction from Isharah JPEG archives."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from trail.batch_isharah_pose import _materialize_frames, _records
from trail.hand_pose import save_hand_pose_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True); args.cache_root.mkdir(parents=True, exist_ok=True)
    rows = [row for row in _records(args.manifest) if row["split"] == args.split]
    pending = [row for row in rows if not (args.output_root / f"{row['clip_id']}.npz").exists()]
    if args.limit is not None: pending = pending[:args.limit]
    audit_path = args.output_root / "hand_pose_audit.csv"; new_audit = not audit_path.exists()
    with audit_path.open("a", encoding="utf-8", newline="") as audit:
        writer = csv.DictWriter(audit, fieldnames=["clip_id", "split", "status", "body_coverage", "left_hand_coverage", "right_hand_coverage", "detail"])
        if new_audit: writer.writeheader()
        for index, row in enumerate(pending, 1):
            working = args.cache_root / row["clip_id"]
            try:
                if working.exists(): shutil.rmtree(working)
                working.mkdir(parents=True); _materialize_frames(row, working)
                result = save_hand_pose_sequence(working, args.pose_model, args.hand_model, args.output_root / f"{row['clip_id']}.npz", fps=args.fps)
                writer.writerow({"clip_id": row["clip_id"], "split": row["split"], "status": "ok", **{key: f"{value:.6f}" for key, value in result.items()}, "detail": ""})
                print(f"[{index}/{len(pending)}] {row['clip_id']}: hands L={result['left_hand_coverage']:.0%} R={result['right_hand_coverage']:.0%}")
            except Exception as error:
                writer.writerow({"clip_id": row["clip_id"], "split": row["split"], "status": "failed", "body_coverage": "", "left_hand_coverage": "", "right_hand_coverage": "", "detail": str(error)})
                print(f"[{index}/{len(pending)}] {row['clip_id']}: FAILED — {error}")
            finally:
                if working.exists(): shutil.rmtree(working)


if __name__ == "__main__":
    main()
