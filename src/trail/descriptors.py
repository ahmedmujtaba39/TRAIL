"""Pose-derived articulatory endpoint descriptors for the workshop pilot.

These features are deliberately not called expert phonological annotations.
They are reproducible functions of body pose and provide the structured-input
condition until hand landmarks and human-verified phonological categories exist.
"""

from __future__ import annotations

import numpy as np


LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HAND_START, RIGHT_HAND_START = 33, 54
FINGERTIPS = np.asarray([4, 8, 12, 16, 20])
HAND_AWARE_DESCRIPTOR_DIM = 77


def _body_descriptor(window: np.ndarray, *, use_last: bool) -> np.ndarray:
    frame = window[-1] if use_last else window[0]
    previous = window[max(0, len(window) - 3)] if use_last else window[min(len(window) - 1, 2)]
    left_vector = frame[LEFT_WRIST] - frame[LEFT_ELBOW]
    right_vector = frame[RIGHT_WRIST] - frame[RIGHT_ELBOW]
    left_motion = frame[LEFT_WRIST] - previous[LEFT_WRIST]
    right_motion = frame[RIGHT_WRIST] - previous[RIGHT_WRIST]
    # wrist positions (6), forearm directions (6), local wrist motion (6)
    return np.concatenate([frame[LEFT_WRIST], frame[RIGHT_WRIST], left_vector, right_vector, left_motion, right_motion]).astype(np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def _hand_features(hand: np.ndarray, wrist: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """26D handshape, palm-orientation, spread, and local-motion descriptor."""
    fingertips = hand[FINGERTIPS] - wrist
    palm_normal = _unit(np.cross(hand[5] - hand[17], hand[9] - hand[0]))
    spread = np.linalg.norm(fingertips, axis=1)
    motion = hand[0] - previous[0]
    return np.concatenate([fingertips.ravel(), palm_normal, spread, motion])


def endpoint_descriptor(window: np.ndarray, *, use_last: bool) -> np.ndarray:
    """Return 18D body or 77D hand-aware articulatory endpoint features."""
    if window.ndim != 3 or window.shape[-1] != 3 or window.shape[1] not in (33, 75):
        raise ValueError("Expected normalized [frames, 33|75, 3] landmarks.")
    body = _body_descriptor(window[:, :33], use_last=use_last)
    if window.shape[1] == 33:
        return body
    frame = window[-1] if use_last else window[0]
    previous = window[max(0, len(window) - 3)] if use_last else window[min(len(window) - 1, 2)]
    left, right = frame[LEFT_HAND_START:RIGHT_HAND_START], frame[RIGHT_HAND_START:]
    left_previous, right_previous = previous[LEFT_HAND_START:RIGHT_HAND_START], previous[RIGHT_HAND_START:]
    left_hand = _hand_features(left, frame[LEFT_WRIST], left_previous)
    right_hand = _hand_features(right, frame[RIGHT_WRIST], right_previous)
    bilateral = np.concatenate([
        left[0] - right[0],
        np.asarray([np.linalg.norm(left[0] - right[0])], dtype=np.float32),
        (left[0] - right[0]) - (left_previous[0] - right_previous[0]),
    ])
    descriptor = np.concatenate([body, left_hand, right_hand, bilateral]).astype(np.float32)
    if descriptor.shape != (HAND_AWARE_DESCRIPTOR_DIM,):
        raise AssertionError(f"Expected {HAND_AWARE_DESCRIPTOR_DIM}D descriptor, got {descriptor.shape}")
    return descriptor
