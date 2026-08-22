"""Weakly supervised transition windows from continuous source pose sequences.

Isharah releases sequence-level gloss strings but no temporal gloss boundaries.
For the workshop pilot, this module samples masked central motion spans from
Saudi continuous pose sequences.  They are explicitly *weak transition windows*;
they must not be reported as gold coarticulation boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from trail.descriptors import endpoint_descriptor
from trail.landmarks import load_landmarks
from trail.handshape import HandshapeClassifier


@dataclass(frozen=True)
class TransitionWindow:
    left: np.ndarray
    right: np.ndarray
    target: np.ndarray
    duration: int


def sample_window(pose: np.ndarray, *, boundary_frames: int, duration: int, rng: np.random.Generator) -> TransitionWindow:
    """Sample a center span with context on both sides from `[T, J, 3]` pose."""
    if pose.ndim != 3 or pose.shape[-1] != 3:
        raise ValueError("Expected pose shape [frames, joints, 3].")
    minimum = 2 * boundary_frames + duration
    if len(pose) < minimum:
        raise ValueError(f"Sequence has {len(pose)} frames; needs at least {minimum}.")
    start = int(rng.integers(boundary_frames, len(pose) - boundary_frames - duration + 1))
    return TransitionWindow(
        left=pose[start - boundary_frames:start],
        target=pose[start:start + duration],
        right=pose[start + duration:start + duration + boundary_frames],
        duration=duration,
    )


def write_windows(pose_paths: list[Path], output: Path, *, windows_per_clip: int = 2, boundary_frames: int = 6, duration: int = 8, seed: int = 42, handshape_model: HandshapeClassifier | None = None, device: torch.device | None = None) -> int:
    """Materialize compact weak transition windows into one compressed NPZ file."""
    rng = np.random.default_rng(seed)
    windows: list[TransitionWindow] = []
    for path in pose_paths:
        pose = load_landmarks(path)
        for _ in range(windows_per_clip):
            try:
                windows.append(sample_window(pose, boundary_frames=boundary_frames, duration=duration, rng=rng))
            except ValueError:
                continue
    if not windows:
        raise RuntimeError("No source clips were long enough to form transition windows.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        left=np.stack([item.left for item in windows]),
        right=np.stack([item.right for item in windows]),
        target=np.stack([item.target for item in windows]),
        left_descriptor=np.stack([endpoint_descriptor(item.left, use_last=True, handshape_model=handshape_model, device=device) for item in windows]),
        right_descriptor=np.stack([endpoint_descriptor(item.right, use_last=False, handshape_model=handshape_model, device=device) for item in windows]),
        duration=np.asarray([item.duration for item in windows], dtype=np.int64),
    )
    return len(windows)
