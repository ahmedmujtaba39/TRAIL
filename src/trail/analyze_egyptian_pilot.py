"""Numerical safety checks for Egyptian TRAIL pilot outputs.

These diagnostics do not assess linguistic validity.  They only catch invalid
or implausibly large skeleton outputs before clips are rendered for experts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def motion(sequence: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(sequence, axis=0), axis=2).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.root / "articulatory" / "synthetic_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    details = []
    for row in rows:
        sample_id = row["sample_id"]
        trail = np.load(args.root / "articulatory" / f"{sample_id}.npy")
        linear = np.load(args.root / "interpolation" / f"{sample_id}.npy")
        jerk = np.load(args.root / "minimum_jerk" / f"{sample_id}.npy")
        details.append({
            "sample_id": sample_id,
            "trail_minus_linear_rms": float(np.sqrt(np.mean((trail - linear) ** 2))),
            "trail_max_abs_coordinate": float(np.abs(trail).max()),
            "trail_motion": motion(trail), "linear_motion": motion(linear), "minimum_jerk_motion": motion(jerk),
            "finite": bool(np.isfinite(trail).all()),
        })
    report = {
        "samples": len(details),
        "all_finite": all(item["finite"] for item in details),
        "mean_trail_minus_linear_rms": float(np.mean([item["trail_minus_linear_rms"] for item in details])),
        "max_trail_minus_linear_rms": float(np.max([item["trail_minus_linear_rms"] for item in details])),
        "max_abs_coordinate": float(np.max([item["trail_max_abs_coordinate"] for item in details])),
        "mean_motion": {name: float(np.mean([item[name] for item in details])) for name in ("trail_motion", "linear_motion", "minimum_jerk_motion")},
        "warning": "Numerical checks only; Deaf-expert assessment is required for linguistic plausibility.",
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
