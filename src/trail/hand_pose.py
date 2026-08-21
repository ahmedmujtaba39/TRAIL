"""Hand-aware landmark extraction for TRAIL.

The output is a fixed 75-joint array: MediaPipe's 33 body joints followed by
21 left-hand and 21 right-hand joints. Coordinates are centered and scaled by
the body shoulders, making hand/body geometry comparable across clips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trail.pose import LEFT_SHOULDER, RIGHT_SHOULDER, _interpolate_missing, normalize_pose


BODY_JOINTS, HAND_JOINTS, TOTAL_JOINTS = 33, 21, 75


def _fill_optional_hand(hand: np.ndarray) -> tuple[np.ndarray, float]:
    valid = ~np.isnan(hand).all(axis=(1, 2))
    coverage = float(valid.mean())
    if not valid.any():
        return np.zeros_like(hand), coverage
    return _interpolate_missing(hand), coverage


def extract_hand_pose_sequence(
    frame_dir: Path, pose_model: Path, hand_model: Path, *, fps: int = 12
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract `[frames, 75, 3]` body and handedness-assigned hand landmarks.

    Input frames must be unmirrored. MediaPipe's handedness labels then map to
    anatomical left/right; clips where this assumption is not true should be
    marked in the dataset card rather than silently flipped.
    """

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install mediapipe and opencv-python before hand-aware extraction.") from error
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No extracted JPEG frames in {frame_dir}")
    if not hand_model.exists():
        raise FileNotFoundError(f"Missing MediaPipe hand-landmarker model: {hand_model}")
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(pose_model)), running_mode=vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(hand_model)), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,
    )
    body = np.full((len(frames), BODY_JOINTS, 3), np.nan, dtype=np.float32)
    left = np.full((len(frames), HAND_JOINTS, 3), np.nan, dtype=np.float32)
    right = np.full((len(frames), HAND_JOINTS, 3), np.nan, dtype=np.float32)
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_detector, vision.HandLandmarker.create_from_options(hand_options) as hand_detector:
        for index, frame in enumerate(frames):
            image = mp.Image.create_from_file(str(frame)); timestamp = round(index * 1000 / fps)
            pose_result = pose_detector.detect_for_video(image, timestamp)
            hand_result = hand_detector.detect_for_video(image, timestamp)
            if pose_result.pose_landmarks:
                body[index] = np.asarray([[p.x, p.y, p.z] for p in pose_result.pose_landmarks[0]], dtype=np.float32)
            for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name.lower()
                target = left if label == "left" else right if label == "right" else None
                if target is not None:
                    target[index] = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    body_coverage = float((~np.isnan(body).all(axis=(1, 2))).mean())
    body = _interpolate_missing(body)
    left, left_coverage = _fill_optional_hand(left)
    right, right_coverage = _fill_optional_hand(right)
    center = (body[:, LEFT_SHOULDER] + body[:, RIGHT_SHOULDER]) / 2
    scale = np.maximum(np.linalg.norm(body[:, LEFT_SHOULDER] - body[:, RIGHT_SHOULDER], axis=1), 1e-4)
    hands = np.concatenate([left, right], axis=1)
    hands = (hands - center[:, None, :]) / scale[:, None, None]
    return np.concatenate([normalize_pose(body), hands], axis=1), {"body_coverage": body_coverage, "left_hand_coverage": left_coverage, "right_hand_coverage": right_coverage}


def save_hand_pose_sequence(frame_dir: Path, pose_model: Path, hand_model: Path, output: Path, *, fps: int = 12) -> dict[str, float]:
    landmarks, audit = extract_hand_pose_sequence(frame_dir, pose_model, hand_model, fps=fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, landmarks=landmarks, **audit)
    return audit
