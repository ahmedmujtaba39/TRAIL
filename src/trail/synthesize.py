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
from trail.handshape import HandshapeClassifier, load_handshape_classifier
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


def _compose(units: list[np.ndarray], model: TransitionTransformer | None, descriptor_mean: np.ndarray | None, descriptor_std: np.ndarray | None, handshape_model: HandshapeClassifier | None, *, duration: int, device: torch.device, articulatory: bool) -> np.ndarray:
    output = [units[0]]
    with torch.no_grad():
        for left, right in zip(units, units[1:]):
            left_boundary = torch.from_numpy(left[-6:]).unsqueeze(0).to(device)
            right_boundary = torch.from_numpy(right[:6]).unsqueeze(0).to(device)
            if model is None:
                transition = interpolate(left_boundary, right_boundary, duration)
            else:
                if articulatory:
                    left_raw = endpoint_descriptor(left[-6:], use_last=True, handshape_model=handshape_model, device=device)
                    right_raw = endpoint_descriptor(right[:6], use_last=False, handshape_model=handshape_model, device=device)
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
    parser.add_argument("--handshape-checkpoint", type=Path, default=None, help="Required when checkpoint descriptor width includes soft handshape probabilities.")
    parser.add_argument("--hand-audit", type=Path, default=None, help="Optional hand_pose_audit.csv used to exclude low-coverage units.")
    parser.add_argument("--min-hand-coverage", type=float, default=0.0, help="Require this coverage for at least one detected hand when --hand-audit is supplied.")
    args = parser.parse_args()
    if args.condition != "interpolation" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for pose_only/articulatory synthesis.")
    rows = _read_csv(args.manifest)
    hand_quality: dict[str, float] = {}
    if args.hand_audit is not None:
        for row in _read_csv(args.hand_audit):
            if row.get("status") != "ok":
                continue
            hand_quality[row["clip_id"]] = max(float(row.get("left_hand_coverage") or 0.0), float(row.get("right_hand_coverage") or 0.0))
    lexicon: dict[str, list[Path]] = defaultdict(list)
    for row in rows:
        if row["role"] == "proxy_isolated_lexicon":
            if args.hand_audit is not None and hand_quality.get(row["clip_id"], 0.0) < args.min_hand_coverage:
                continue
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
        handshape_model = load_handshape_classifier(args.handshape_checkpoint, device) if args.handshape_checkpoint else None
    else:
        model = descriptor_mean = descriptor_std = handshape_model = None
    destination = args.output_root / args.condition
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "synthetic_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "tokens", "pose_path", "condition", "source_units"])
        writer.writeheader()
        for index, (label, tokens, variant) in enumerate(selected):
            # Cycle across independent isolated exemplars deterministically.
            paths = [lexicon[token][(variant * 17 + position * 7 + args.seed) % len(lexicon[token])] for position, token in enumerate(tokens)]
            sequence = _compose([load_landmarks(path) for path in paths], model, descriptor_mean, descriptor_std, handshape_model, duration=args.duration, device=device, articulatory=args.condition == "articulatory")
            sample_id = f"syn_{index:04d}"
            pose_path = destination / f"{sample_id}.npy"
            np.save(pose_path, sequence)
            writer.writerow({"sample_id": sample_id, "tokens": label, "pose_path": str(pose_path), "condition": args.condition, "source_units": "|".join(map(str, paths))})
    print(f"Wrote {len(selected)} synthetic {args.condition} sequences to {destination}")


if __name__ == "__main__":
    main()
