"""Extract TRAIL-compatible hand geometry from public labelled reference sets.

This prepares *reference* supervision only.  The labels retain their source
inventory (LSA16/RWTH/JSL); they must not be represented as Arabic handshape
ground truth.  A later multi-source encoder uses them for visual-geometry
pretraining and is calibrated separately on KArSL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from trail.handshape import handshape_features


def _records(root: Path, source: str):
    if source == "lsa16":
        folder = root / "lsa16" / "lsa16_images" / "lsa32x32_nr_rgb_black_background"
        for image in sorted(folder.glob("*.png")):
            yield image, int(image.stem.split("_")[0]) - 1
        return
    if source == "rwth":
        data = np.load(root / "rwth" / "rwth.npz")
        for image, label in zip(data["x"], data["y"], strict=True):
            yield np.asarray(image), int(label)
        return
    if source == "jsl":
        folder = root / "jsl" / "jsl_images"
        for image in sorted(folder.glob("*.png")):
            # Dataset naming uses two digits at positions 9--10 for its class.
            yield image, int(image.name[9:11]) - 1
        return
    raise ValueError(f"Unsupported source: {source}")


def _hand_detector(model: Path):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
    )
    return mp, vision.HandLandmarker.create_from_options(options)


def _image(record: object) -> np.ndarray:
    if isinstance(record, Path):
        image = cv2.imread(str(record), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {record}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.asarray(record)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
    # All three public reference sets are distributed at thumbnail resolution.
    # Upscaling does not invent detail, but lets the detector operate in its
    # intended pixel range and avoids treating a 32px hand as a failed sample.
    height, width = image.shape[:2]
    scale = max(1, int(np.ceil(256 / min(height, width))))
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Root produced by the public reference-data download.")
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["lsa16", "rwth", "jsl"])
    parser.add_argument("--max-per-source", type=int, default=None, help="Optional debug cap; never use for final evaluation.")
    args = parser.parse_args()
    if not args.hand_model.exists():
        raise SystemExit(f"Missing MediaPipe hand-landmarker model: {args.hand_model}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[str] = []
    mp, detector = _hand_detector(args.hand_model)
    try:
        for source in args.sources:
            total = found = 0
            for record, label in _records(args.root, source):
                if args.max_per_source is not None and total >= args.max_per_source:
                    break
                total += 1
                try:
                    image = _image(record)
                    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
                    if not result.hand_landmarks:
                        continue
                    # Use the largest detected hand; reference images contain
                    # citation forms, so this is more stable than handedness.
                    hand = max(
                        result.hand_landmarks,
                        key=lambda points: max(p.x for p in points) - min(p.x for p in points),
                    )
                    landmarks = np.asarray([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
                    features.append(handshape_features(landmarks)); labels.append(label); sources.append(source); found += 1
                except Exception:
                    continue
            print(f"{source}: {found}/{total} images yielded a hand geometry")
    finally:
        detector.close()
    if not features:
        raise SystemExit("No hand geometries were detected.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack(features), labels=np.asarray(labels, dtype=np.int64), source=np.asarray(sources, dtype="U16"))
    print(f"Wrote {len(features)} reference geometries to {args.output}")


if __name__ == "__main__":
    main()
