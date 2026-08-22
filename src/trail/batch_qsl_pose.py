"""Disk-bounded pose extraction for JUMLA-QSL-22 manifests."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from trail.pose import save_pose_sequence
from trail.svo import extract_left_frames


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract QSL poses one clip at a time.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--participants", nargs="+", choices=["AS", "AT", "MA"], default=["AS", "AT", "MA"])
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many remaining clips.")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    args.pose_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    audit_path = args.pose_root / "pose_audit.csv"
    records = [
        row for row in _read_manifest(args.manifest)
        if row["available"] == "true" and row["participant"] in args.participants
    ]
    pending = [row for row in records if not (args.pose_root / f"{row['clip_id']}.npy").exists()]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print("No pending QSL clips for the requested participants.")
        return

    new_audit = not audit_path.exists()
    with audit_path.open("a", encoding="utf-8", newline="") as audit:
        writer = csv.DictWriter(audit, fieldnames=["clip_id", "participant", "status", "coverage", "detail"])
        if new_audit:
            writer.writeheader()
        for index, row in enumerate(pending, start=1):
            clip_id = row["clip_id"]
            working = args.cache_root / clip_id
            output = args.pose_root / f"{clip_id}.npy"
            try:
                if working.exists():
                    shutil.rmtree(working)
                extract_left_frames(Path(row["video_path"]), working, args.ffmpeg, fps=args.fps)
                coverage = save_pose_sequence(working, args.model, output, fps=args.fps)
                writer.writerow({"clip_id": clip_id, "participant": row["participant"], "status": "ok", "coverage": f"{coverage:.6f}", "detail": ""})
                print(f"[{index}/{len(pending)}] {clip_id}: {coverage:.1%}")
            except Exception as error:
                writer.writerow({"clip_id": clip_id, "participant": row["participant"], "status": "failed", "coverage": "", "detail": str(error)})
                print(f"[{index}/{len(pending)}] {clip_id}: FAILED — {error}")
            finally:
                if working.exists():
                    shutil.rmtree(working)


if __name__ == "__main__":
    main()
