"""Paired bootstrap and endpoint-distance analysis for DGS Model T controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trail.evaluate_transition import load_model
from trail.model import interpolate, minimum_jerk


def per_example_error(prediction: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    position = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    velocity = torch.mean(((prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])) ** 2, dim=(1, 2, 3))
    return (position + 0.25 * velocity).cpu().numpy()


def predict(values: np.lib.npyio.NpzFile, checkpoint: Path, handshape_dim: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, z_clip = load_model(checkpoint, device)
    left = torch.from_numpy(values["left"]).float().to(device)
    right = torch.from_numpy(values["right"]).float().to(device)
    target = torch.from_numpy(values["target"]).float().to(device)
    duration = torch.from_numpy(values["duration"]).long().to(device)
    left_descriptor = np.clip((values["left_descriptor"] - mean) / std, -z_clip, z_clip)
    right_descriptor = np.clip((values["right_descriptor"] - mean) / std, -z_clip, z_clip)
    left_descriptor = torch.from_numpy(left_descriptor).float().to(device)
    right_descriptor = torch.from_numpy(right_descriptor).float().to(device)
    with torch.no_grad():
        result = {
            "interpolation": per_example_error(interpolate(left, right, int(duration[0])), target),
            "minimum_jerk": per_example_error(minimum_jerk(left, right, int(duration[0])), target),
            "masked": per_example_error(model(left, right, torch.zeros_like(left_descriptor), torch.zeros_like(right_descriptor), duration, use_descriptors=False), target),
            "full": per_example_error(model(left, right, left_descriptor, right_descriptor, duration, use_descriptors=True), target),
        }
        left_h, right_h = left_descriptor.clone(), right_descriptor.clone()
        left_h[:, handshape_dim:] = 0.0; right_h[:, handshape_dim:] = 0.0
        result["handshape_only"] = per_example_error(model(left, right, left_h, right_h, duration, use_descriptors=True), target)
    geometry = np.concatenate([left_descriptor.cpu().numpy()[:, handshape_dim:], right_descriptor.cpu().numpy()[:, handshape_dim:]], axis=1)
    handshape = np.concatenate([left_descriptor.cpu().numpy()[:, :handshape_dim], right_descriptor.cpu().numpy()[:, :handshape_dim]], axis=1)
    # Endpoint distance uses equal group weighting after train-set standardization.
    distance = np.linalg.norm(handshape[:, :handshape_dim] - handshape[:, handshape_dim:], axis=1) / np.sqrt(handshape_dim)
    width = geometry.shape[1] // 2
    distance += np.linalg.norm(geometry[:, :width] - geometry[:, width:], axis=1) / np.sqrt(width)
    return result, distance


def bootstrap_improvement(reference: np.ndarray, candidate: np.ndarray, draws: int, rng: np.random.Generator) -> dict[str, float]:
    delta = reference - candidate
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        samples[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return {"mean_error_reduction": float(delta.mean()), "relative_reduction_percent": float(100 * delta.mean() / reference.mean()), "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handshape-dim", type=int, default=6)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    values = np.load(args.windows)
    per_seed, distances = zip(*(predict(values, path, args.handshape_dim) for path in args.checkpoints))
    conditions = per_seed[0].keys()
    mean_errors = {name: np.mean([result[name] for result in per_seed], axis=0) for name in conditions}
    summary = {name: {"mean": float(values_.mean()), "seed_std": float(np.asarray([result[name].mean() for result in per_seed]).std(ddof=1))} for name, values_ in mean_errors.items()}
    rng = np.random.default_rng(args.seed)
    summary["paired_bootstrap_vs_interpolation"] = {name: bootstrap_improvement(mean_errors["interpolation"], mean_errors[name], args.bootstrap_draws, rng) for name in conditions if name != "interpolation"}
    distance = np.mean(distances, axis=0)
    edges = np.quantile(distance, [0, 1 / 3, 2 / 3, 1])
    bins = []
    for index, label in enumerate(("low", "medium", "high")):
        mask = (distance >= edges[index]) & (distance <= edges[index + 1] if index == 2 else distance < edges[index + 1])
        baseline = mean_errors["interpolation"][mask].mean()
        full = mean_errors["full"][mask].mean()
        bins.append({"bin": label, "n": int(mask.sum()), "distance_range": [float(edges[index]), float(edges[index + 1])], "interpolation": float(baseline), "full": float(full), "relative_full_improvement_percent": float(100 * (baseline - full) / baseline)})
    summary["endpoint_articulatory_distance"] = {"definition": "standardized endpoint h distance / sqrt(6) plus standardized endpoint l/mu/o distance / sqrt(77)", "bins": bins}
    summary["split_note"] = "Temporal-block, segment-instance-disjoint split; lexical types overlap because the source recording's transition graph is connected."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
