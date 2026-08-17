"""Synthetic smoke test; this does not use or claim real sign-language results."""

from __future__ import annotations

import numpy as np
import torch

from trail.decision import decide
from trail.model import TransitionTransformer, interpolate


def main() -> None:
    torch.manual_seed(7)
    model = TransitionTransformer(joints=8, descriptor_dim=16, width=64, heads=4, layers=2)
    left = torch.randn(2, 6, 8, 3)
    right = torch.randn(2, 6, 8, 3)
    descriptor_left = torch.randn(2, 16)
    descriptor_right = torch.randn(2, 16)
    duration = torch.tensor([8, 8])
    generated = model(left, right, descriptor_left, descriptor_right, duration, use_phonology=True)
    masked = model(left, right, descriptor_left, descriptor_right, duration, use_phonology=False)
    baseline = interpolate(left, right, 8)
    assert generated.shape == masked.shape == baseline.shape == (2, 8, 8, 3)

    rng = np.random.default_rng(7)
    within_pose = rng.normal(60, 2, 200)
    within_phon = within_pose + rng.normal(0, 0.4, 200)
    transfer_pose = rng.normal(70, 2, 200)
    transfer_phon = transfer_pose - rng.normal(5, 0.6, 200)
    result = decide(within_phon, within_pose, transfer_phon, transfer_pose)
    assert result.go, result
    print("Smoke test passed.")
    print(result.rationale)
    print(f"transfer difference={result.transfer_difference:.2f} pp, interaction={result.interaction:.2f} pp")


if __name__ == "__main__":
    main()
