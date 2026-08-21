"""Pose-derived articulatory endpoint descriptors for the workshop pilot.

These features are deliberately not called expert phonological annotations.
They are reproducible functions of body pose and provide the structured-input
condition until hand landmarks and human-verified phonological categories exist.
"""

from __future__ import annotations

import numpy as np


LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16


def endpoint_descriptor(window: np.ndarray, *, use_last: bool) -> np.ndarray:
    """Return an 18D body-articulatory feature vector from a pose boundary."""
    if window.ndim != 3 or window.shape[1:] != (33, 3):
        raise ValueError("Expected [frames, 33, 3] normalized body pose.")
    frame = window[-1] if use_last else window[0]
    previous = window[max(0, len(window) - 3)] if use_last else window[min(len(window) - 1, 2)]
    left_vector = frame[LEFT_WRIST] - frame[LEFT_ELBOW]
    right_vector = frame[RIGHT_WRIST] - frame[RIGHT_ELBOW]
    left_motion = frame[LEFT_WRIST] - previous[LEFT_WRIST]
    right_motion = frame[RIGHT_WRIST] - previous[RIGHT_WRIST]
    # wrist positions (6), forearm directions (6), local wrist motion (6)
    return np.concatenate([frame[LEFT_WRIST], frame[RIGHT_WRIST], left_vector, right_vector, left_motion, right_motion]).astype(np.float32)
