"""Backward-compatible loading of body-only and hand-aware TRAIL sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_landmarks(path: Path) -> np.ndarray:
    """Load legacy ``.npy`` body pose or ``.npz`` hand-aware landmarks."""
    values = np.load(path)
    if isinstance(values, np.ndarray):
        return values.astype(np.float32, copy=False)
    try:
        return values["landmarks"].astype(np.float32, copy=False)
    finally:
        values.close()
