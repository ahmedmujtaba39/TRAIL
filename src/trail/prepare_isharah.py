"""Create a source-data index for Isharah's published SI split."""

from __future__ import annotations

import argparse
from pathlib import Path

from trail.isharah import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an Isharah SI manifest.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/isharah_sequences.csv"))
    args = parser.parse_args()
    counts = build_manifest(args.source_root, args.output)
    print(f"Wrote {args.output}")
    print("Isharah SI clips: " + ", ".join(f"{split}={count}" for split, count in counts.items()))


if __name__ == "__main__":
    main()
