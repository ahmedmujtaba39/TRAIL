"""Create a real-QSL oracle training manifest for pipeline sanity checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real continuous-QSL oracle manifest.")
    parser.add_argument("--qsl-manifest", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--participant", choices=["AT", "MA"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.qsl_manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if row["role"] != "heldout_continuous_test" or row["participant"] != args.participant or row["available"] != "true":
            continue
        pose_path = args.pose_root / f"{row['clip_id']}.npy"
        if pose_path.exists():
            selected.append({"sample_id": row["clip_id"], "tokens": row["tokens"], "pose_path": str(pose_path)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "tokens", "pose_path"])
        writer.writeheader(); writer.writerows(selected)
    print(f"Wrote {len(selected)} real {args.participant} oracle examples to {args.output}")


if __name__ == "__main__":
    main()
