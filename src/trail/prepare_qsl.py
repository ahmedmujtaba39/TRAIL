"""Command-line entry point for the QSL data audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from trail.qsl import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a JUMLA-QSL-22 TRAIL manifest.")
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/jumla_qsl_manifest.csv"))
    args = parser.parse_args()
    counts = build_manifest(args.workbook, args.data_root, args.output)
    print(f"Wrote {args.output}")
    print(f"AS proxy isolated lexicon: {counts['isolated_available']}/{counts['isolated_total']} available")
    print(f"AT+MA held-out continuous evaluation: {counts['heldout_available']}/{counts['heldout_total']} available")
    if counts["isolated_available"] != counts["isolated_total"]:
        raise SystemExit("QSL lexicon is incomplete; download the missing AS rec0.svo clips before training.")


if __name__ == "__main__":
    main()
