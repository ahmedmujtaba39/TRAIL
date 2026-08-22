"""Pretrain a hand-geometry encoder on public reference inventories and calibrate on KArSL.

The KArSL labels are isolated-sign identities, not expert handshape labels.  We
therefore report their held-out accuracy only as a domain-calibration check.
The descriptor exposed to TRAIL is a posterior over the chosen public reference
inventory (normally RWTH), and must be named an automatic reference-code token
in reports until expert phonological annotation is available.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trail.handshape import ReferenceHandshapeClassifier


def _batches(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=True, drop_last=False)


def _accuracy_per_clip(logits: torch.Tensor, labels: torch.Tensor, clip_ids: np.ndarray) -> float:
    grouped: dict[str, list[int]] = {}
    for index, clip_id in enumerate(clip_ids.astype(str)): grouped.setdefault(clip_id, []).append(index)
    correct = [bool(logits[indexes].mean(0).argmax() == labels[indexes[0]]) for indexes in grouped.values()]
    return float(np.mean(correct)) if correct else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="NPZ from trail-prepare-reference-handshapes.")
    parser.add_argument("--karsl", type=Path, required=True, help="NPZ from trail-build-handshape-examples.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", default="rwth", help="Source head whose posterior becomes h.")
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--calibration-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); torch.manual_seed(args.seed)
    ref = np.load(args.reference, allow_pickle=False); target = np.load(args.karsl, allow_pickle=False)
    ref_x = torch.from_numpy(ref["features"]).float(); ref_y = torch.from_numpy(ref["labels"]).long(); ref_source = ref["source"].astype(str)
    target_x = torch.from_numpy(target["features"]).float(); target_y = torch.from_numpy(target["labels"]).long(); target_split = target["split"].astype(str)
    clip_ids = target["clip_id"].astype(str) if "clip_id" in target else np.asarray([str(i) for i in range(len(target_y))])
    if ref_x.ndim != 2 or ref_x.shape[1] != 60 or target_x.ndim != 2 or target_x.shape[1] != 60:
        raise SystemExit("Both inputs must contain [N,60] features.")
    head_sizes = {source: int(ref_y[ref_source == source].max()) + 1 for source in sorted(set(ref_source))}
    head_sizes["karsl_proxy"] = int(target_y.max()) + 1
    if args.reference_source not in head_sizes:
        raise SystemExit(f"Reference source {args.reference_source!r} has no detected examples; have {sorted(head_sizes)}")
    train = np.flatnonzero(target_split == "train"); test = np.flatnonzero(target_split == "test")
    if not len(train) or not len(test): raise SystemExit("KArSL examples need explicit train/test groups.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReferenceHandshapeClassifier(head_sizes, args.reference_source, args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Phase 1: each external source keeps its own label head, sharing only
    # geometry representation.  This prevents false cross-language label maps.
    for _ in range(args.pretrain_epochs):
        model.train()
        for source in sorted(set(ref_source)):
            subset = np.flatnonzero(ref_source == source)
            for values, labels in _batches(ref_x[subset], ref_y[subset], args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model.logits(values.to(device), source), labels.to(device))
                loss.backward(); optimizer.step()
    # Phase 2: target-domain calibration uses only the provided recording-group
    # split. It is a domain-shift diagnostic, not phonological ground truth.
    for _ in range(args.calibration_epochs):
        model.train()
        for values, labels in _batches(target_x[train], target_y[train], args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model.logits(values.to(device), "karsl_proxy"), labels.to(device))
            loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): target_logits = model.logits(target_x[test].to(device), "karsl_proxy").cpu()
    score = _accuracy_per_clip(target_logits, target_y[test], clip_ids[test])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint_type": "reference_handshape", "state_dict": model.state_dict(), "head_sizes": head_sizes, "reference_source": args.reference_source, "width": args.width}, args.output)
    audit = {"karsl_proxy_heldout_clip_accuracy": score, "reference_examples": int(len(ref_x)), "reference_examples_by_source": dict(Counter(ref_source)), "target_train_frames": int(len(train)), "target_test_frames": int(len(test)), "target_test_clips": int(len(set(clip_ids[test]))), "reference_source": args.reference_source, "descriptor_classes": head_sizes[args.reference_source], "warning": "KArSL sign IDs are calibration proxies, not expert phonological handshape labels."}
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"KArSL recording-group-held-out proxy accuracy: {score:.2%}")
    print(f"Reference descriptor: {args.reference_source} ({head_sizes[args.reference_source]} soft classes)")


if __name__ == "__main__":
    main()
