"""Extract 60-D hand geometry and HamNoSys-derived h labels from 3DZ renders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from trail.handshape import handshape_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="render_manifest.csv from trail-render-3dz-handshapes.")
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install mediapipe before extracting rendered handshape examples.") from error
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8", newline="")))
    rows = [row for row in rows if row["handshape_h"] != "unknown"]
    labels = {name: index for index, name in enumerate(sorted({row["handshape_h"] for row in rows}))}
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(args.hand_model)), running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.25, min_hand_presence_confidence=0.25)
    features, targets, sign_ids = [], [], []
    with vision.HandLandmarker.create_from_options(options) as detector:
        for row in rows:
            image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
            if not result.hand_landmarks: continue
            hand = max(result.hand_landmarks, key=lambda points: max(p.x for p in points) - min(p.x for p in points))
            landmarks = np.asarray([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
            features.append(handshape_features(landmarks)); targets.append(labels[row["handshape_h"]]); sign_ids.append(row["sign_id"])
    if not features: raise SystemExit("MediaPipe found no hands in the rendered examples.")
    # Entire lexical entries are held out, never neighbouring frames.
    unique = sorted(set(sign_ids)); heldout = {sign for index, sign in enumerate(unique) if index % 5 == 0}
    split = np.asarray(["test" if sign in heldout else "train" for sign in sign_ids], dtype="U8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack(features), labels=np.asarray(targets, dtype=np.int64), split=split, clip_id=np.asarray(sign_ids, dtype="U64"))
    args.output.with_suffix(".labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Wrote {len(features)} detected avatar hands across {len(labels)} HamNoSys-derived classes; {len(heldout)} lexical signs held out.")


if __name__ == "__main__":
    main()
