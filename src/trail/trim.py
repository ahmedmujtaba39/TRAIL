"""Motion-energy trimming for isolated-sign proxy clips."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ACTIVE_JOINTS = np.asarray([13, 14, 15, 16])  # elbows and wrists


def activity_bounds(pose: np.ndarray, *, padding: int = 6, min_frames: int = 16) -> tuple[int, int, np.ndarray]:
    """Find a conservative active interval using smoothed elbow/wrist velocity."""
    if len(pose) < min_frames:
        return 0, len(pose), np.zeros(len(pose), dtype=np.float32)
    velocity = np.linalg.norm(np.diff(pose[:, ACTIVE_JOINTS], axis=0), axis=-1).mean(axis=1)
    energy = np.concatenate([[velocity[0]], velocity])
    energy = np.convolve(energy, np.ones(5, dtype=np.float32) / 5, mode="same")
    median = float(np.median(energy)); mad = float(np.median(np.abs(energy - median)))
    threshold = max(0.006, median + 0.75 * mad)
    active = np.flatnonzero(energy >= threshold)
    if not len(active):
        return 0, len(pose), energy
    start = max(0, int(active[0]) - padding); end = min(len(pose), int(active[-1]) + padding + 1)
    if end - start < min_frames:
        center = (start + end) // 2
        start = max(0, center - min_frames // 2); end = min(len(pose), start + min_frames)
    return start, end, energy


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim AS proxy-lexicon pose clips to active signing intervals.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--pattern", default="AS_*.npy")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True); args.audit.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.input_root.glob(args.pattern)):
        pose = np.load(path); start, end, energy = activity_bounds(pose)
        output = args.output_root / path.name; np.save(output, pose[start:end])
        rows.append({"clip_id": path.stem, "input_frames": len(pose), "start": start, "end": end, "output_frames": end-start, "retained_fraction": f"{(end-start)/len(pose):.5f}", "mean_energy": f"{float(energy.mean()):.6f}"})
    with args.audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["clip_id"]); writer.writeheader(); writer.writerows(rows)
    print(f"Trimmed {len(rows)} clips to {args.output_root}")


if __name__ == "__main__":
    main()
