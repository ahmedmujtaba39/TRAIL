"""Disk-bounded pose extraction directly from Isharah JPEG zip archives."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from zipfile import ZipFile

from trail.pose import save_pose_sequence


def _records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _materialize_frames(row: dict[str, str], working: Path) -> Path:
    prefix = row["member_prefix"]
    with ZipFile(row["archive_path"]) as archive:
        members = [name for name in archive.namelist() if name.startswith(prefix) and name.lower().endswith(".jpg")]
        if not members:
            raise RuntimeError(f"No JPEG frames found for {row['clip_id']}")
        for member in members:
            destination = working / Path(member).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    # Isharah names frames `frame0001.jpg`; standardize the glob used by pose.py.
    for index, frame in enumerate(sorted(working.glob("*.jpg"))):
        frame.rename(working / f"frame_{index:05d}.jpg")
    return working


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Isharah body poses directly from source archives.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    args.pose_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    rows = [row for row in _records(args.manifest) if row["split"] == args.split]
    pending = [row for row in rows if not (args.pose_root / f"{row['clip_id']}.npy").exists()]
    if args.limit is not None:
        pending = pending[: args.limit]
    audit_path = args.pose_root / "pose_audit.csv"
    new_audit = not audit_path.exists()
    with audit_path.open("a", encoding="utf-8", newline="") as audit:
        writer = csv.DictWriter(audit, fieldnames=["clip_id", "split", "status", "coverage", "detail"])
        if new_audit:
            writer.writeheader()
        for index, row in enumerate(pending, start=1):
            working = args.cache_root / row["clip_id"]
            output = args.pose_root / f"{row['clip_id']}.npy"
            try:
                if working.exists():
                    shutil.rmtree(working)
                working.mkdir(parents=True)
                _materialize_frames(row, working)
                coverage = save_pose_sequence(working, args.model, output, fps=args.fps)
                writer.writerow({"clip_id": row["clip_id"], "split": row["split"], "status": "ok", "coverage": f"{coverage:.6f}", "detail": ""})
                print(f"[{index}/{len(pending)}] {row['clip_id']}: {coverage:.1%}")
            except Exception as error:
                writer.writerow({"clip_id": row["clip_id"], "split": row["split"], "status": "failed", "coverage": "", "detail": str(error)})
                print(f"[{index}/{len(pending)}] {row['clip_id']}: FAILED — {error}")
            finally:
                if working.exists():
                    shutil.rmtree(working)


if __name__ == "__main__":
    main()
