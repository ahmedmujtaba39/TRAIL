"""MediaPipe pose extraction for TRAIL's reproducible pose representation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def _interpolate_missing(poses: np.ndarray) -> np.ndarray:
    """Linearly fill failed frames; fail rather than invent an empty sequence."""
    valid = ~np.isnan(poses).all(axis=(1, 2))
    if not valid.any():
        raise RuntimeError("No pose was detected in any frame.")
    indices = np.arange(len(poses))
    for joint in range(poses.shape[1]):
        for axis in range(poses.shape[2]):
            values = poses[:, joint, axis]
            known = ~np.isnan(values)
            values[~known] = np.interp(indices[~known], indices[known], values[known])
    return poses


def normalize_pose(poses: np.ndarray) -> np.ndarray:
    """Center at the shoulder midpoint and scale by shoulder width per frame."""
    center = (poses[:, LEFT_SHOULDER] + poses[:, RIGHT_SHOULDER]) / 2
    scale = np.linalg.norm(poses[:, LEFT_SHOULDER] - poses[:, RIGHT_SHOULDER], axis=1)
    scale = np.maximum(scale, 1e-4)
    return (poses - center[:, None, :]) / scale[:, None, None]


def extract_pose_sequence(frame_dir: Path, model_path: Path, *, fps: int = 12) -> tuple[np.ndarray, float]:
    """Return normalized `[frames, 33, 3]` landmarks and detection coverage."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:  # pragma: no cover - environment guidance
        raise RuntimeError("Install mediapipe and opencv-python before pose extraction.") from error
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No extracted JPEG frames in {frame_dir}")
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    poses = np.full((len(frames), 33, 3), np.nan, dtype=np.float32)
    with vision.PoseLandmarker.create_from_options(options) as detector:
        for index, frame in enumerate(frames):
            image = mp.Image.create_from_file(str(frame))
            result = detector.detect_for_video(image, round(index * 1000 / fps))
            if result.pose_landmarks:
                poses[index] = np.asarray([[point.x, point.y, point.z] for point in result.pose_landmarks[0]], dtype=np.float32)
    coverage = float((~np.isnan(poses).all(axis=(1, 2))).mean())
    return normalize_pose(_interpolate_missing(poses)), coverage


def save_pose_sequence(frame_dir: Path, model_path: Path, output: Path, *, fps: int = 12) -> float:
    poses, coverage = extract_pose_sequence(frame_dir, model_path, fps=fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, poses)
    return coverage
