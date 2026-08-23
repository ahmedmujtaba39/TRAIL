"""Create a lexical-type-disjoint DGS reconstruction split.

Each reconstruction pair connects two lexical types.  We form the undirected
type-transition graph and assign whole connected components to train or test.
Consequently, neither target lexical type nor either source lexical type in a
test example occurs in training.  The matching NPZ rows retain the manifest
order produced by ``build_dgs_direct_windows``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def connected_components(rows: list[dict[str, str]]) -> list[set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        left, right = row["left_type_id"], row["right_type_id"]
        graph[left].add(right)
        graph[right].add(left)
    seen: set[str] = set()
    components: list[set[str]] = []
    for root in sorted(graph):
        if root in seen:
            continue
        stack, component = [root], set()
        seen.add(root)
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbour in graph[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def choose_test_components(components: list[set[str]], rows: list[dict[str, str]], fraction: float, seed: int) -> set[int]:
    component_of = {token: index for index, component in enumerate(components) for token in component}
    counts = np.zeros(len(components), dtype=int)
    for row in rows:
        left, right = component_of[row["left_type_id"]], component_of[row["right_type_id"]]
        if left != right:
            raise AssertionError("A graph edge crossed connected components.")
        counts[left] += 1
    target = round(len(rows) * fraction)
    rng = np.random.default_rng(seed)
    best: tuple[int, set[int]] | None = None
    # Component sizes vary.  Multiple shuffled greedy passes find a near-target
    # held-out set without ever splitting a lexical type across partitions.
    for _ in range(2000):
        order = rng.permutation(len(components))
        selected: set[int] = set()
        total = 0
        for index in order:
            candidate = total + int(counts[index])
            if abs(candidate - target) < abs(total - target):
                selected.add(int(index))
                total = candidate
        score = abs(total - target)
        if best is None or score < best[0]:
            best = (score, selected)
    assert best is not None
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.pairs.open(encoding="utf-8")))
    windows = np.load(args.windows)
    if len(rows) != len(windows["left"]):
        raise SystemExit("Pair manifest and window tensor have different row counts.")
    components = connected_components(rows)
    test_components = choose_test_components(components, rows, args.test_fraction, args.seed)
    component_of = {token: index for index, component in enumerate(components) for token in component}
    test_indices = np.asarray([
        index for index, row in enumerate(rows)
        if component_of[row["left_type_id"]] in test_components
    ], dtype=int)
    train_indices = np.asarray([
        index for index, row in enumerate(rows)
        if component_of[row["left_type_id"]] not in test_components
    ], dtype=int)
    train_types = {rows[index][key] for index in train_indices for key in ("left_type_id", "right_type_id")}
    test_types = {rows[index][key] for index in test_indices for key in ("left_type_id", "right_type_id")}
    overlap = train_types & test_types
    if overlap:
        raise AssertionError(f"Lexical leakage detected: {sorted(overlap)[:5]}")
    for output, indices in ((args.train_output, train_indices), (args.test_output, test_indices)):
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **{key: windows[key][indices] for key in windows.files})
    audit = {
        "split": "lexical-type-disjoint connected-component split",
        "seed": args.seed,
        "components": len(components),
        "train_pairs": int(len(train_indices)),
        "test_pairs": int(len(test_indices)),
        "train_types": len(train_types),
        "test_types": len(test_types),
        "type_overlap": len(overlap),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
