"""Build classifier examples from hand-aware isolated-sign landmark clips.

The input CSV is intentionally explicit: ``pose_path,handshape_label,split``.
``split`` must be a signer-held-out train/test assignment supplied by the
dataset metadata; this tool never guesses signer identities from file names.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from trail.handshape import dominant_hand_features
from trail.landmarks import load_landmarks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"pose_path", "handshape_label", "split"}
    if not rows or not required.issubset(rows[0]): raise SystemExit(f"Manifest requires columns: {sorted(required)}")
    labels = {label: index for index, label in enumerate(sorted({row["handshape_label"] for row in rows}))}
    features, targets, splits = [], [], []
    for row in rows:
        landmarks = load_landmarks(Path(row["pose_path"]))
        features.append(dominant_hand_features(landmarks, use_last=len(landmarks) > 1))
        targets.append(labels[row["handshape_label"]]); splits.append(row["split"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack(features), labels=np.asarray(targets, dtype=np.int64), split=np.asarray(splits, dtype="U8"))
    args.output.with_suffix(".labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(features)} examples / {len(labels)} handshape classes to {args.output}")


if __name__ == "__main__": main()
