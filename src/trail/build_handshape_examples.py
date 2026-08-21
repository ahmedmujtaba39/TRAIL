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

from trail.handshape import dominant_hand, handshape_features
from trail.landmarks import load_landmarks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=8, help="Stable middle frames used as training views; clip IDs remain available for evaluation aggregation.")
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"pose_path", "handshape_label", "split"}
    if not rows or not required.issubset(rows[0]): raise SystemExit(f"Manifest requires columns: {sorted(required)}")
    labels = {label: index for index, label in enumerate(sorted({row["handshape_label"] for row in rows}))}
    features, targets, splits, clip_ids = [], [], [], []
    for row in rows:
        landmarks = load_landmarks(Path(row["pose_path"]))
        hand = dominant_hand(landmarks)
        # Citation clips include rest at either end. Restrict supervision to the
        # stable middle portion, then sample several appearance/motion views.
        low, high = int(0.30 * len(hand)), max(int(0.30 * len(hand)) + 1, int(0.70 * len(hand)))
        indices = np.linspace(low, high - 1, args.frames_per_clip, dtype=int)
        for index in indices:
            features.append(handshape_features(hand[index]))
            targets.append(labels[row["handshape_label"]]); splits.append(row["split"]); clip_ids.append(row["sample_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack(features), labels=np.asarray(targets, dtype=np.int64), split=np.asarray(splits, dtype="U8"), clip_id=np.asarray(clip_ids, dtype="U64"))
    args.output.with_suffix(".labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(features)} examples / {len(labels)} handshape classes to {args.output}")


if __name__ == "__main__": main()
