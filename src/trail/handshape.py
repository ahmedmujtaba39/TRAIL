"""Soft handshape classifier used by TRAIL's phonological descriptor.

The classifier operates on wrist-centred, palm-aligned 21-joint hand geometry.
It is intentionally language-neutral at its interface; class inventories and
training examples are supplied by the source isolated-sign resource.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def handshape_features(hand: np.ndarray) -> np.ndarray:
    """Return 60D palm-aligned coordinates, excluding the wrist origin."""
    if hand.shape != (21, 3):
        raise ValueError("Expected a [21, 3] hand landmark array.")
    wrist = hand[0]
    across = hand[5] - hand[17]
    across /= max(float(np.linalg.norm(across)), 1e-6)
    forward = hand[9] - wrist
    normal = np.cross(across, forward)
    normal /= max(float(np.linalg.norm(normal)), 1e-6)
    up = np.cross(normal, across)
    rotation = np.stack([across, up, normal], axis=1)
    return ((hand[1:] - wrist) @ rotation).astype(np.float32).ravel()


def dominant_hand_features(window: np.ndarray, *, use_last: bool) -> np.ndarray:
    """Use the more active hand in a boundary window, deterministically."""
    if window.ndim != 3 or window.shape[1:] != (75, 3):
        raise ValueError("Handshape features require [frames, 75, 3] landmarks.")
    hand = dominant_hand(window)
    hand = hand[-1 if use_last else 0]
    return handshape_features(hand)


def dominant_hand(landmarks: np.ndarray) -> np.ndarray:
    """Select the active hand once for an isolated clip."""
    if landmarks.ndim != 3 or landmarks.shape[1:] != (75, 3):
        raise ValueError("Expected [frames, 75, 3] landmarks.")
    left, right = landmarks[:, 33:54], landmarks[:, 54:75]
    left_energy = float(np.linalg.norm(np.diff(left[:, 0], axis=0), axis=-1).mean())
    right_energy = float(np.linalg.norm(np.diff(right[:, 0], axis=0), axis=-1).mean())
    return left if left_energy >= right_energy else right


class HandshapeClassifier(nn.Module):
    """Small MLP whose softmax is the proposal's handshape token h."""

    def __init__(self, classes: int, width: int = 128):
        super().__init__()
        self.classes = classes
        self.network = nn.Sequential(nn.Linear(60, width), nn.ReLU(), nn.Dropout(0.15), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ReferenceHandshapeClassifier(nn.Module):
    """Shared geometry encoder with a source-specific soft reference codebook.

    The selected output head is a *reference handshape inventory*, not a claim
    that its language-specific label names are Arabic phonological categories.
    Its soft posterior is used as an automatic hand-configuration token and is
    evaluated separately before it may condition TRAIL.
    """

    def __init__(self, head_sizes: dict[str, int], reference_source: str, width: int = 128):
        super().__init__()
        if reference_source not in head_sizes:
            raise ValueError(f"Unknown reference source {reference_source!r}")
        self.head_sizes = dict(head_sizes); self.reference_source = reference_source; self.width = width
        self.encoder = nn.Sequential(nn.Linear(60, width), nn.ReLU(), nn.Dropout(0.10), nn.Linear(width, width), nn.ReLU())
        self.heads = nn.ModuleDict({source: nn.Linear(width, classes) for source, classes in self.head_sizes.items()})

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def logits(self, features: torch.Tensor, source: str) -> torch.Tensor:
        return self.heads[source](self.encode(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(features, self.reference_source)


def load_handshape_classifier(path: Path, device: torch.device) -> nn.Module:
    values = torch.load(path, map_location=device, weights_only=False)
    if values.get("checkpoint_type") == "reference_handshape":
        model = ReferenceHandshapeClassifier(
            {str(key): int(value) for key, value in values["head_sizes"].items()},
            str(values["reference_source"]), int(values.get("width", 128)),
        ).to(device)
        model.load_state_dict(values["state_dict"]); model.eval()
        return model
    model = HandshapeClassifier(int(values["classes"]), int(values.get("width", 128))).to(device)
    model.load_state_dict(values["state_dict"]); model.eval()
    return model


def handshape_distribution(window: np.ndarray, model: nn.Module, *, use_last: bool, device: torch.device) -> np.ndarray:
    feature = torch.from_numpy(dominant_hand_features(window, use_last=use_last)).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(model(feature), dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
