"""Train TRAIL Model T on fixed-duration source transition windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trail.model import TransitionTransformer


def transition_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reconstruction = nn.functional.smooth_l1_loss(prediction, target)
    velocity = nn.functional.smooth_l1_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
    return reconstruction + 0.25 * velocity


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRAIL's transition model on source pose windows.")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--descriptor-dim", type=int, default=18)
    parser.add_argument("--condition", choices=["pose_only", "articulatory"], default="pose_only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    values = np.load(args.windows)
    left = torch.from_numpy(values["left"]).float()
    right = torch.from_numpy(values["right"]).float()
    target = torch.from_numpy(values["target"]).float()
    duration = torch.from_numpy(values["duration"]).long()
    if args.condition == "articulatory":
        if "left_descriptor" not in values or "right_descriptor" not in values:
            raise SystemExit("Window file has no articulatory descriptors; rebuild it with trail.prepare_transitions.")
        left_descriptor = torch.from_numpy(values["left_descriptor"]).float()
        right_descriptor = torch.from_numpy(values["right_descriptor"]).float()
    else:
        left_descriptor = torch.zeros((len(left), args.descriptor_dim), dtype=torch.float32)
        right_descriptor = torch.zeros((len(left), args.descriptor_dim), dtype=torch.float32)
    loader = DataLoader(TensorDataset(left, right, target, duration, left_descriptor, right_descriptor), batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitionTransformer(joints=left.shape[2], descriptor_dim=args.descriptor_dim, width=args.width, heads=4, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.train()
    history: list[float] = []
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        count = 0
        for batch_left, batch_right, batch_target, batch_duration, batch_left_descriptor, batch_right_descriptor in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch_left.to(device), batch_right.to(device), batch_left_descriptor.to(device), batch_right_descriptor.to(device),
                batch_duration.to(device), use_phonology=args.condition == "articulatory",
            )
            loss = transition_loss(prediction, batch_target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch_left)
            count += len(batch_left)
        history.append(running / count)
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            print(f"epoch={epoch} loss={history[-1]:.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "joints": left.shape[2], "descriptor_dim": args.descriptor_dim, "width": args.width, "layers": args.layers, "history": history},
        args.output,
    )
    args.output.with_suffix(".json").write_text(json.dumps({"final_loss": history[-1], "epochs": args.epochs, "windows": len(left), "condition": args.condition}, indent=2), encoding="utf-8")
    print(f"Saved {args.condition} Model T checkpoint to {args.output}")


if __name__ == "__main__":
    main()
