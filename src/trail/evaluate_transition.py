"""Evaluate one shared TRAIL Model T checkpoint on held-out source windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trail.model import TransitionTransformer, interpolate


def load_model(path: Path, device: torch.device):
    values = torch.load(path, map_location=device, weights_only=False)
    model = TransitionTransformer(int(values["joints"]), int(values["descriptor_dim"]), int(values["width"]), heads=4, layers=int(values["layers"])).to(device)
    model.load_state_dict(values["state_dict"]); model.eval()
    return model, np.asarray(values["descriptor_mean"], dtype=np.float32), np.maximum(np.asarray(values["descriptor_std"], dtype=np.float32), 1e-4)


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    position = float(torch.mean((prediction - target) ** 2))
    velocity = float(torch.mean(((prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])) ** 2))
    return {"position_mse": position, "velocity_mse": velocity, "combined": position + 0.25 * velocity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    values = np.load(args.windows)
    left = torch.from_numpy(values["left"]).float()
    right = torch.from_numpy(values["right"]).float()
    target = torch.from_numpy(values["target"]).float()
    duration = torch.from_numpy(values["duration"]).long()
    left_raw = torch.from_numpy(values["left_descriptor"]).float()
    right_raw = torch.from_numpy(values["right_descriptor"]).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = load_model(args.checkpoint, device)
    predictions = {"interpolation": [], "pose_only": [], "articulatory": []}
    for start in range(0, len(left), args.batch_size):
        end = min(start + args.batch_size, len(left))
        batch_left, batch_right, batch_duration = left[start:end].to(device), right[start:end].to(device), duration[start:end].to(device)
        left_descriptor = ((left_raw[start:end].numpy() - mean) / std)
        right_descriptor = ((right_raw[start:end].numpy() - mean) / std)
        with torch.no_grad():
            predictions["interpolation"].append(interpolate(batch_left, batch_right, int(batch_duration[0])).cpu())
            predictions["pose_only"].append(model(batch_left, batch_right, torch.zeros_like(torch.from_numpy(left_descriptor), device=device), torch.zeros_like(torch.from_numpy(right_descriptor), device=device), batch_duration, use_descriptors=False).cpu())
            predictions["articulatory"].append(model(batch_left, batch_right, torch.from_numpy(left_descriptor).to(device), torch.from_numpy(right_descriptor).to(device), batch_duration, use_descriptors=True).cpu())
    report = {condition: metrics(torch.cat(chunks), target) for condition, chunks in predictions.items()}
    report["windows"] = len(target)
    report["ablation_contract"] = "one shared checkpoint; descriptor-present versus descriptor-masked"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
