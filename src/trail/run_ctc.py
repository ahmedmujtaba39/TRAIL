"""Train/evaluate a matched CTC recognizer on synthetic QSL pose sequences.

This is the downstream-utility measurement in the TRAIL pilot.  Every
condition uses the identical recognizer, optimizer, seed, target vocabulary,
and real AT/MA evaluation set; only synthetic transitions differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from trail.landmarks import load_landmarks


@dataclass(frozen=True)
class Example:
    pose_path: Path
    tokens: tuple[str, ...]
    clip_id: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def synthetic_examples(path: Path) -> list[Example]:
    return [Example(Path(row["pose_path"]), tuple(row["tokens"].split()), row["sample_id"]) for row in read_csv(path)]


def heldout_examples(manifest: Path, pose_root: Path, participants: set[str]) -> list[Example]:
    result: list[Example] = []
    for row in read_csv(manifest):
        if row["role"] != "heldout_continuous_test" or row["available"] != "true" or row["participant"] not in participants:
            continue
        path = pose_root / f"{row['clip_id']}.npy"
        if path.exists():
            result.append(Example(path, tuple(row["tokens"].split()), row["clip_id"]))
    return result


class PoseDataset(Dataset):
    def __init__(self, examples: list[Example], vocab: dict[str, int]):
        self.examples, self.vocab = examples, vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        pose = torch.from_numpy(load_landmarks(example.pose_path)).float().flatten(1)
        labels = torch.tensor([self.vocab[token] for token in example.tokens], dtype=torch.long)
        return pose, labels, example


def collate(batch):
    poses, labels, examples = zip(*batch)
    lengths = torch.tensor([len(pose) for pose in poses], dtype=torch.long)
    return pad_sequence(poses, batch_first=True), torch.cat(labels), lengths, torch.tensor([len(label) for label in labels]), examples


class CTCRecognizer(nn.Module):
    def __init__(self, features: int, classes: int, hidden: int = 96):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(features, hidden, 5, stride=2, padding=2), nn.ReLU(), nn.Dropout(0.15))
        self.encoder = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden, classes)

    def forward(self, pose: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.front(pose.transpose(1, 2)).transpose(1, 2)
        output, _ = self.encoder(output)
        new_lengths = (lengths + 1) // 2
        return self.classifier(output).log_softmax(-1), new_lengths


def wer(reference: list[int], hypothesis: list[int]) -> float:
    table = list(range(len(hypothesis) + 1))
    for i, value in enumerate(reference, 1):
        previous, table[0] = table[0], i
        for j, other in enumerate(hypothesis, 1):
            old = table[j]
            table[j] = min(table[j] + 1, table[j - 1] + 1, previous + (value != other))
            previous = old
    return table[-1] / max(1, len(reference))


def decode(logits: torch.Tensor, length: int) -> list[int]:
    values = logits[:length].argmax(-1).tolist()
    result, previous = [], 0
    for value in values:
        if value != 0 and value != previous:
            result.append(value)
        previous = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched synthetic-to-real QSL CTC experiment.")
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--qsl-manifest", type=Path, required=True)
    parser.add_argument("--qsl-pose-root", type=Path, required=True)
    parser.add_argument("--eval-participants", nargs="+", choices=["AT", "MA"], default=["AT", "MA"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train = synthetic_examples(args.synthetic_manifest)
    test = heldout_examples(args.qsl_manifest, args.qsl_pose_root, set(args.eval_participants))
    tokens = sorted({token for item in train + test for token in item.tokens})
    vocab = {token: index + 1 for index, token in enumerate(tokens)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(PoseDataset(train, vocab), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(PoseDataset(test, vocab), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = CTCRecognizer(99, len(vocab) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0
        for pose, labels, lengths, label_lengths, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, output_lengths = model(pose.to(device), lengths.to(device))
            loss = criterion(logits.transpose(0, 1), labels.to(device), output_lengths, label_lengths)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            loss_sum += float(loss.detach())
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(f"epoch={epoch} loss={loss_sum / len(loader):.5f}")
    model.eval(); scores: list[float] = []; participant_scores: dict[str, list[float]] = {"AT": [], "MA": []}; per_clip: dict[str, float] = {}
    with torch.no_grad():
        for pose, labels, lengths, label_lengths, examples in test_loader:
            logits, output_lengths = model(pose.to(device), lengths.to(device))
            offset = 0
            for index, example in enumerate(examples):
                reference = labels[offset:offset + label_lengths[index]].tolist(); offset += label_lengths[index]
                score = wer(reference, decode(logits[index].cpu(), int(output_lengths[index])))
                scores.append(score); participant_scores[example.clip_id[:2]].append(score)
                per_clip[example.clip_id] = score
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"mean_wer": float(np.mean(scores)), "n_test": len(scores), "seed": args.seed, "participant_wer": {key: float(np.mean(value)) if value else None for key, value in participant_scores.items()}, "n_train": len(train), "vocab_size": len(vocab), "per_clip_wer": per_clip}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
