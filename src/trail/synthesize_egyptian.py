"""Generate auditable Egyptian Sign Language transition-only synthesis samples.

This is intentionally a *local-transition* pilot: it composes two real
isolated Egyptian clips and asks whether the intervening motion is plausible.
It does not assert that the two labels form an Egyptian Sign Language sentence;
grammar and semantic acceptability remain for Deaf-expert evaluation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from trail.handshape import load_handshape_classifier
from trail.landmarks import load_landmarks
from trail.model import interpolate, minimum_jerk
from trail.synthesize import _load_model
from trail.descriptors import endpoint_descriptor


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def best_clips(audit: Path, pose_root: Path, threshold: float) -> dict[str, tuple[Path, dict[str, str]]]:
    """Select one high-quality independently recorded clip per target label."""
    choices: dict[str, tuple[float, Path, dict[str, str]]] = {}
    for row in read_csv(audit):
        if row.get("status") != "ok":
            continue
        score = min(float(row["left_hand_coverage"]), float(row["right_hand_coverage"]))
        path = pose_root / f"{row['clip_id']}.npz"
        if not path.exists() or score < threshold:
            continue
        old = choices.get(row["label"])
        if old is None or score > old[0]:
            choices[row["label"]] = (score, path, row)
    return {label: (path, row) for label, (_, path, row) in choices.items()}


def generate_transition(
    left: np.ndarray, right: np.ndarray, *, condition: str, duration: int,
    model, mean: np.ndarray | None, std: np.ndarray | None, z_clip: float,
    handshape_model, device: torch.device,
) -> np.ndarray:
    left_context = torch.from_numpy(left[-6:]).unsqueeze(0).to(device)
    right_context = torch.from_numpy(right[:6]).unsqueeze(0).to(device)
    with torch.no_grad():
        if condition == "interpolation":
            transition = interpolate(left_context, right_context, duration)
        elif condition == "minimum_jerk":
            transition = minimum_jerk(left_context, right_context, duration)
        else:
            left_raw = endpoint_descriptor(left[-6:], use_last=True, handshape_model=handshape_model, device=device)
            right_raw = endpoint_descriptor(right[:6], use_last=False, handshape_model=handshape_model, device=device)
            if len(left_raw) != model.descriptor_projection.in_features:
                raise ValueError(f"Expected {model.descriptor_projection.in_features}D descriptors, got {len(left_raw)}D")
            left_desc = torch.from_numpy(np.clip((left_raw - mean) / std, -z_clip, z_clip)).unsqueeze(0).to(device)
            right_desc = torch.from_numpy(np.clip((right_raw - mean) / std, -z_clip, z_clip)).unsqueeze(0).to(device)
            transition = model(left_context, right_context, left_desc, right_desc, torch.tensor([duration], device=device), use_descriptors=True)
    return np.concatenate([left, transition.squeeze(0).cpu().numpy(), right], axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="CSV: sample_id,left_label,right_label")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", choices=["interpolation", "minimum_jerk", "articulatory"], required=True)
    parser.add_argument("--checkpoint", type=Path, help="Required for articulatory generation")
    parser.add_argument("--handshape-checkpoint", type=Path, help="Required for full 83D descriptor checkpoint")
    parser.add_argument("--min-both-hand-coverage", type=float, default=0.4)
    parser.add_argument("--duration", type=int, default=8)
    args = parser.parse_args()
    if args.condition == "articulatory" and (not args.checkpoint or not args.handshape_checkpoint):
        raise SystemExit("--checkpoint and --handshape-checkpoint are required for articulatory generation")

    candidates = best_clips(args.audit, args.pose_root, args.min_both_hand_coverage)
    pairs = read_csv(args.pairs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.condition == "articulatory":
        model, mean, std, z_clip = _load_model(args.checkpoint, device)
        handshape_model = load_handshape_classifier(args.handshape_checkpoint, device)
    else:
        model = mean = std = handshape_model = None
        z_clip = 5.0
    destination = args.output_root / args.condition
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "synthetic_manifest.csv"
    fields = ["sample_id", "left_label", "right_label", "left_clip_id", "right_clip_id", "condition", "pose_path", "note"]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in pairs:
            if row["left_label"] not in candidates or row["right_label"] not in candidates:
                missing = [label for label in (row["left_label"], row["right_label"]) if label not in candidates]
                raise ValueError(f"No clip meeting quality threshold for: {', '.join(missing)}")
            left_path, left_meta = candidates[row["left_label"]]
            right_path, right_meta = candidates[row["right_label"]]
            sequence = generate_transition(load_landmarks(left_path), load_landmarks(right_path), condition=args.condition, duration=args.duration, model=model, mean=mean, std=std, z_clip=z_clip, handshape_model=handshape_model, device=device)
            pose_path = destination / f"{row['sample_id']}.npy"; np.save(pose_path, sequence)
            writer.writerow({"sample_id": row["sample_id"], "left_label": row["left_label"], "right_label": row["right_label"], "left_clip_id": left_meta["clip_id"], "right_clip_id": right_meta["clip_id"], "condition": args.condition, "pose_path": str(pose_path), "note": "Transition-only sample; no grammatical-sentence claim."})
    print(f"Wrote {len(pairs)} {args.condition} Egyptian transition samples to {destination}")


if __name__ == "__main__":
    main()
