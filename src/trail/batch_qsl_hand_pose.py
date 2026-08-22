"""Disk-bounded hand-aware extraction for JUMLA-QSL-22 clips."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from trail.hand_pose import save_hand_pose_sequence
from trail.svo import extract_left_frames


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--participants", nargs="+", choices=["AS", "AT", "MA"], default=["AS", "AT", "MA"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True); args.cache_root.mkdir(parents=True, exist_ok=True)
    records = [row for row in _read_manifest(args.manifest) if row["available"] == "true" and row["participant"] in args.participants]
    pending = [row for row in records if not (args.output_root / f"{row['clip_id']}.npz").exists()]
    if args.limit is not None:
        pending = pending[:args.limit]
    audit_path = args.output_root / "hand_pose_audit.csv"; new_audit = not audit_path.exists()
    with audit_path.open("a", encoding="utf-8", newline="") as audit:
        writer = csv.DictWriter(audit, fieldnames=["clip_id", "participant", "status", "body_coverage", "left_hand_coverage", "right_hand_coverage", "detail"])
        if new_audit: writer.writeheader()
        for index, row in enumerate(pending, 1):
            clip_id = row["clip_id"]; working = args.cache_root / clip_id
            try:
                if working.exists(): shutil.rmtree(working)
                extract_left_frames(Path(row["video_path"]), working, args.ffmpeg, fps=args.fps)
                result = save_hand_pose_sequence(working, args.pose_model, args.hand_model, args.output_root / f"{clip_id}.npz", fps=args.fps)
                writer.writerow({"clip_id": clip_id, "participant": row["participant"], "status": "ok", **{key: f"{value:.6f}" for key, value in result.items()}, "detail": ""})
                print(f"[{index}/{len(pending)}] {clip_id}: hands L={result['left_hand_coverage']:.0%} R={result['right_hand_coverage']:.0%}")
            except Exception as error:
                writer.writerow({"clip_id": clip_id, "participant": row["participant"], "status": "failed", "body_coverage": "", "left_hand_coverage": "", "right_hand_coverage": "", "detail": str(error)})
                print(f"[{index}/{len(pending)}] {clip_id}: FAILED — {error}")
            finally:
                if working.exists(): shutil.rmtree(working)


if __name__ == "__main__":
    main()
