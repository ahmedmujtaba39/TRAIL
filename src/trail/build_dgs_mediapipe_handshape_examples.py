"""Create real-human HamNoSys handshape examples using MediaPipe landmarks.

Unlike the OpenPose smoke extractor, this reads the official DGS front video
and extracts the same MediaPipe 21-hand representation used by KArSL/IshaRah.
That representation match is required before testing cross-corpus transfer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from trail.handshape import handshape_features


def frame_number(timecode: str, fps: int) -> int:
    hours, minutes, seconds, frames = (int(part) for part in timecode.split(":"))
    return (((hours * 60 + minutes) * 60 + seconds) * fps) + frames


def sampled_frames(start: int, end: int, count: int) -> list[int]:
    if end <= start:
        return [start]
    count = min(count, end - start + 1)
    return sorted({round(start + (end - start) * (index + 1) / (count + 1)) for index in range(count)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--tier-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=int, default=50)
    parser.add_argument("--samples-per-segment", type=int, default=3)
    args = parser.parse_args()
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install mediapipe before extracting DGS hand examples.") from error

    rows = [row for row in csv.DictReader(args.segments.open(encoding="utf-8")) if row["tier_id"] == args.tier_id and row["handshape_h"] != "unknown"]
    if not rows:
        raise SystemExit("No labelled DGS segments match the requested tier.")
    labels_map = {name: index for index, name in enumerate(sorted({row["handshape_h"] for row in rows}))}
    wanted: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for frame in sampled_frames(frame_number(row["timecode_start"], args.source_fps), frame_number(row["timecode_end"], args.source_fps), args.samples_per_segment):
            wanted[frame].append(row)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    if round(video_fps) != args.source_fps:
        raise SystemExit(f"Video FPS {video_fps} does not match annotation FPS {args.source_fps}.")
    features: list[np.ndarray] = []
    labels: list[int] = []
    clip_ids: list[str] = []
    selected_hands: Counter[str] = Counter()
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.hand_model)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.35,
    )
    max_frame = max(wanted)
    with vision.HandLandmarker.create_from_options(options) as detector:
        for frame_index in range(max_frame + 1):
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index not in wanted:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.hand_landmarks:
                continue
            candidates = []
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name.lower()
                confidence = float(handedness[0].score)
                points = np.asarray([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
                candidates.append((confidence, label, points))
            _, side, hand = max(candidates, key=lambda item: item[0])
            feature = handshape_features(hand)
            for row in wanted[frame_index]:
                features.append(feature)
                labels.append(labels_map[row["handshape_h"]])
                clip_ids.append(row["type_id"])
                selected_hands[side] += 1
    capture.release()
    if not features:
        raise SystemExit("MediaPipe did not recover hands for any selected DGS sign segment.")
    unique = sorted(set(clip_ids))
    heldout = {identifier for index, identifier in enumerate(unique) if index % 5 == 0}
    split = np.asarray(["test" if identifier in heldout else "train" for identifier in clip_ids], dtype="U8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack(features), labels=np.asarray(labels, dtype=np.int64), split=split, clip_id=np.asarray(clip_ids, dtype="U64"))
    args.output.with_suffix(".labels.json").write_text(json.dumps(labels_map, indent=2), encoding="utf-8")
    audit = {
        "examples": len(features), "classes": labels_map, "lexical_types": len(unique),
        "heldout_lexical_types": len(heldout), "selected_hands": dict(selected_hands),
        "video_fps": video_fps,
        "label_contract": "Real-human DGS video with MediaPipe hand geometry, aligned to first explicit HamNoSys base handshape. Dominant hand is approximated by detector confidence.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {len(features)} MediaPipe DGS examples across {len(labels_map)} HamNoSys handshape classes.")


if __name__ == "__main__":
    main()
