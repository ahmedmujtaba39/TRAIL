"""Compose synthetic QSL pose sequences from AS proxy-lexicon clips."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from trail.descriptors import endpoint_descriptor
from trail.landmarks import load_landmarks
from trail.model import TransitionTransformer, interpolate


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_model(checkpoint: Path, device: torch.device) -> tuple[TransitionTransformer, np.ndarray, np.ndarray]:
    values = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TransitionTransformer(
        joints=int(values["joints"]), descriptor_dim=int(values["descriptor_dim"]),
        width=int(values["width"]), heads=4, layers=int(values["layers"]),
    ).to(device)
    model.load_state_dict(values["state_dict"])
    model.eval()
    mean = np.asarray(values.get("descriptor_mean", np.zeros(int(values["descriptor_dim"]))), dtype=np.float32)
    std = np.asarray(values.get("descriptor_std", np.ones(int(values["descriptor_dim"]))), dtype=np.float32)
    return model, mean, np.maximum(std, 1e-4)


def _compose(units: list[np.ndarray], model: TransitionTransformer | None, descriptor_mean: np.ndarray | None, descriptor_std: np.ndarray | None, *, duration: int, device: torch.device, articulatory: bool) -> np.ndarray:
    output = [units[0]]
    with torch.no_grad():
        for left, right in zip(units, units[1:]):
            left_boundary = torch.from_numpy(left[-6:]).unsqueeze(0).to(device)
            right_boundary = torch.from_numpy(right[:6]).unsqueeze(0).to(device)
            if model is None:
                transition = interpolate(left_boundary, right_boundary, duration)
            else:
                if articulatory:
                    left_raw = endpoint_descriptor(left[-6:], use_last=True)
                    right_raw = endpoint_descriptor(right[:6], use_last=False)
                    if len(left_raw) != model.descriptor_projection.in_features:
                        raise ValueError("Descriptor/data mismatch: use a hand-aware checkpoint with hand-aware units, or body-only for both.")
                    left_descriptor = torch.from_numpy((left_raw - descriptor_mean) / descriptor_std).unsqueeze(0).to(device)
                    right_descriptor = torch.from_numpy((right_raw - descriptor_mean) / descriptor_std).unsqueeze(0).to(device)
                else:
                    left_descriptor = right_descriptor = torch.zeros((1, model.descriptor_projection.in_features), device=device)
                transition = model(left_boundary, right_boundary, left_descriptor, right_descriptor, torch.tensor([duration], device=device), use_descriptors=articulatory)
            output.append(transition.squeeze(0).cpu().numpy())
            output.append(right)
    return np.concatenate(output, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose synthetic QSL sequences from AS units.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", choices=["pose_only", "articulatory", "interpolation"], required=True)
    parser.add_argument("--checkpoint", type=Path, help="Required for pose_only/articulatory.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants-per-template", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=8)
    args = parser.parse_args()
    if args.condition != "interpolation" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for pose_only/articulatory synthesis.")
    rows = _read_csv(args.manifest)
    lexicon: dict[str, list[Path]] = defaultdict(list)
    for row in rows:
        if row["role"] == "proxy_isolated_lexicon":
            npz_path = args.pose_root / f"{row['clip_id']}.npz"
            path = npz_path if npz_path.exists() else args.pose_root / f"{row['clip_id']}.npy"
            if path.exists():
                lexicon[row["tokens"]].append(path)
    templates: dict[str, tuple[str, ...]] = {}
    for row in rows:
        tokens = tuple(row["tokens"].split())
        if len(tokens) > 1 and all(token in lexicon for token in tokens):
            templates.setdefault(" ".join(tokens), tokens)
    selected = [(label, tokens, variant) for label, tokens in templates.items() for variant in range(args.variants_per_template)]
    if args.limit is not None:
        selected = selected[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.condition != "interpolation":
        model, descriptor_mean, descriptor_std = _load_model(args.checkpoint, device)
    else:
        model = descriptor_mean = descriptor_std = None
    destination = args.output_root / args.condition
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "synthetic_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "tokens", "pose_path", "condition", "source_units"])
        writer.writeheader()
        for index, (label, tokens, variant) in enumerate(selected):
            # Cycle across independent isolated exemplars deterministically.
            paths = [lexicon[token][(variant * 17 + position * 7 + args.seed) % len(lexicon[token])] for position, token in enumerate(tokens)]
            sequence = _compose([load_landmarks(path) for path in paths], model, descriptor_mean, descriptor_std, duration=args.duration, device=device, articulatory=args.condition == "articulatory")
            sample_id = f"syn_{index:04d}"
            pose_path = destination / f"{sample_id}.npy"
            np.save(pose_path, sequence)
            writer.writerow({"sample_id": sample_id, "tokens": label, "pose_path": str(pose_path), "condition": args.condition, "source_units": "|".join(map(str, paths))})
    print(f"Wrote {len(selected)} synthetic {args.condition} sequences to {destination}")


if __name__ == "__main__":
    main()
