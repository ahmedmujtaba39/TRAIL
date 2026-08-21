"""Train and evaluate a signer-held-out soft handshape classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trail.handshape import HandshapeClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True, help="NPZ: features [N,60], labels [N], split [N] (train/test).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); torch.manual_seed(args.seed)
    values = np.load(args.examples, allow_pickle=False)
    features = torch.from_numpy(values["features"]).float(); labels = torch.from_numpy(values["labels"]).long(); split = values["split"].astype(str)
    if features.ndim != 2 or features.shape[1] != 60 or len(features) != len(labels): raise SystemExit("Expected matching [N,60] features and labels.")
    train = np.flatnonzero(split == "train"); test = np.flatnonzero(split == "test")
    if not len(train) or not len(test): raise SystemExit("Examples must provide non-empty signer-held-out train and test splits.")
    classes = int(labels.max()) + 1; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandshapeClassifier(classes, args.width).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(features[train], labels[train]), batch_size=args.batch_size, shuffle=True)
    for _ in range(args.epochs):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True); loss = nn.functional.cross_entropy(model(batch_features.to(device)), batch_labels.to(device)); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(features[test].to(device)).argmax(-1).cpu()
    accuracy = float((prediction == labels[test]).float().mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "width": args.width}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({"signer_heldout_accuracy": accuracy, "n_train": int(len(train)), "n_test": int(len(test)), "classes": classes}, indent=2), encoding="utf-8")
    print(f"Signer-held-out handshape accuracy: {accuracy:.2%}")


if __name__ == "__main__": main()
