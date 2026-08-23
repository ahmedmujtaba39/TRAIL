"""Create a temporal-block, segment-instance-disjoint DGS split.

Lexical types form one giant connected component in the available DGS recording,
which makes a usable type-disjoint split impossible.  This split instead holds
out a contiguous final block of real boundaries and removes every training pair
that reuses any target or source segment instance from that block.  Lexical
types may recur, but no exact annotated segment instance leaks across splits.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEGMENT_COLUMNS = (
    "left_target_segment_id", "right_target_segment_id",
    "left_source_segment_id", "right_source_segment_id",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.pairs.open(encoding="utf-8")))
    windows = np.load(args.windows)
    if len(rows) != len(windows["left"]):
        raise SystemExit("Pair manifest and window tensor have different row counts.")
    cutoff = len(rows) - round(len(rows) * args.test_fraction)
    test_indices = np.arange(cutoff, len(rows), dtype=int)
    test_segments = {rows[index][column] for index in test_indices for column in SEGMENT_COLUMNS}
    train_indices = np.asarray([
        index for index in range(cutoff)
        if not any(rows[index][column] in test_segments for column in SEGMENT_COLUMNS)
    ], dtype=int)
    train_segments = {rows[index][column] for index in train_indices for column in SEGMENT_COLUMNS}
    overlap = train_segments & test_segments
    if overlap:
        raise AssertionError(f"Segment leakage detected: {sorted(overlap)[:5]}")
    for output, indices in ((args.train_output, train_indices), (args.test_output, test_indices)):
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **{key: windows[key][indices] for key in windows.files})
    train_types = {rows[index][column] for index in train_indices for column in ("left_type_id", "right_type_id")}
    test_types = {rows[index][column] for index in test_indices for column in ("left_type_id", "right_type_id")}
    audit = {
        "split": "chronological temporal block with source-and-target segment-instance disjointness",
        "train_pairs": int(len(train_indices)),
        "test_pairs": int(len(test_indices)),
        "train_segments": len(train_segments),
        "test_segments": len(test_segments),
        "segment_overlap": len(overlap),
        "lexical_type_overlap": len(train_types & test_types),
        "limitation": "lexical types are not disjoint because 566/567 pairs form one connected lexical transition graph.",
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
