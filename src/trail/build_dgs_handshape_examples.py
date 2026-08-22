"""Build real-human handshape examples from Public DGS Corpus OpenPose tracks.

This joins `trail.prepare_dgs_hamnosys` output to a matching OpenPose archive.
Only sparse middle frames of a time-aligned lexical segment are used, so the
output is a handshape training set rather than an over-counted frame corpus.
The selected hand is the detected hand with strongest OpenPose confidence;
this is an auditable baseline and not a substitute for manual dominance tags.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import ijson
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


def as_hand(values: list[float], minimum_confidence: float) -> tuple[np.ndarray, float] | None:
    if len(values) != 63:
        return None
    points = np.asarray(values, dtype=np.float32).reshape(21, 3)
    confidence = float(points[:, 2].mean())
    if confidence < minimum_confidence:
        return None
    # OpenPose has image-plane x/y only.  Palm-aligned geometry remains useful
    # for a supervised calibration baseline; 3-D depth is not invented.
    return np.column_stack([points[:, :2], np.zeros(21, dtype=np.float32)]), confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True, help="CSV emitted by trail.prepare_dgs_hamnosys.")
    parser.add_argument("--openpose", type=Path, required=True, help="Matching Public DGS .json.gz archive.")
    parser.add_argument("--camera", default="a1", help="Camera stream to use, e.g. a1.")
    parser.add_argument("--tier-id", required=True, help="Lexical tier corresponding to the selected camera/signer.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--samples-per-segment", type=int, default=3)
    parser.add_argument("--minimum-confidence", type=float, default=0.25)
    args = parser.parse_args()

    rows = [row for row in csv.DictReader(args.segments.open(encoding="utf-8")) if row["tier_id"] == args.tier_id and row["handshape_h"] != "unknown"]
    if not rows:
        raise SystemExit("No labelled lexical segments match the requested tier.")
    classes = {name: index for index, name in enumerate(sorted({row["handshape_h"] for row in rows}))}
    wanted: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for frame in sampled_frames(frame_number(row["timecode_start"], args.fps), frame_number(row["timecode_end"], args.fps), args.samples_per_segment):
            wanted[frame].append(row)

    features: list[np.ndarray] = []
    labels: list[int] = []
    clip_ids: list[str] = []
    selected_hands: Counter[str] = Counter()
    camera = None
    current_frame = None
    people: list[dict[str, list[float]]] = []
    person: dict[str, list[float]] | None = None
    active_field: str | None = None
    target_camera_seen = False

    with gzip.open(args.openpose, "rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if prefix == "item.camera" and event == "string":
                if target_camera_seen and camera == args.camera and value != args.camera:
                    break
                camera = value
                target_camera_seen |= camera == args.camera
            elif prefix == "item.frames" and event == "map_key":
                current_frame = int(value)
                people = []
            elif current_frame is not None and prefix == f"item.frames.{current_frame}.people.item" and event == "start_map":
                person = {"hand_left_keypoints_2d": [], "hand_right_keypoints_2d": []}
            elif person is not None and event == "start_array" and prefix.endswith(("hand_left_keypoints_2d", "hand_right_keypoints_2d")):
                active_field = prefix.rsplit(".", 1)[-1]
            elif person is not None and active_field is not None and event == "number" and prefix.endswith(f"{active_field}.item"):
                person[active_field].append(float(value))
            elif person is not None and active_field is not None and event == "end_array" and prefix.endswith(active_field):
                active_field = None
            elif person is not None and current_frame is not None and prefix == f"item.frames.{current_frame}.people.item" and event == "end_map":
                people.append(person)
                person = None
            elif current_frame is not None and prefix == f"item.frames.{current_frame}" and event == "end_map":
                if camera == args.camera and current_frame in wanted:
                    candidates = []
                    for candidate in people:
                        for side in ("left", "right"):
                            converted = as_hand(candidate[f"hand_{side}_keypoints_2d"], args.minimum_confidence)
                            if converted is not None:
                                hand, confidence = converted
                                candidates.append((confidence, side, hand))
                    if candidates:
                        _, side, hand = max(candidates, key=lambda item: item[0])
                        feature = handshape_features(hand)
                        for row in wanted[current_frame]:
                            features.append(feature)
                            labels.append(classes[row["handshape_h"]])
                            clip_ids.append(row["type_id"])
                            selected_hands[side] += 1
                current_frame = None

    if not features:
        raise SystemExit("No confident OpenPose hands were recovered for the requested labelled segments.")
    unique = sorted(set(clip_ids))
    heldout = {identifier for index, identifier in enumerate(unique) if index % 5 == 0}
    split = np.asarray(["test" if identifier in heldout else "train" for identifier in clip_ids], dtype="U8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack(features), labels=np.asarray(labels, dtype=np.int64), split=split, clip_id=np.asarray(clip_ids, dtype="U64"))
    args.output.with_suffix(".labels.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    audit = {
        "examples": len(features), "classes": classes, "lexical_types": len(unique),
        "heldout_lexical_types": len(heldout), "selected_hands": dict(selected_hands),
        "label_contract": "Real-human OpenPose hand geometry matched to first explicit HamNoSys base handshape.  Dominant hand is approximated by confidence.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {len(features)} real-human DGS examples across {len(classes)} HamNoSys handshape classes.")


if __name__ == "__main__":
    main()
