"""Audit conservative KArSL-to-Isharah lexical coverage for the Saudi control."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from trail.karsl import exact_normalized_index, load_labels, normalize_arabic


def main() -> None:
    # Windows PowerShell may still default to cp1252; the report contains Arabic.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--isharah-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    label_index = exact_normalized_index(labels)
    token_counts: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    with args.isharah_manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tokens = row["glosses"].split()
            token_counts.update(tokens)
            matched = [label_index.get(normalize_arabic(token)) for token in tokens]
            rows.append(
                {
                    "clip_id": row["clip_id"],
                    "split": row["split"],
                    "glosses": row["glosses"],
                    "all_tokens_match": str(all(item is not None for item in matched)),
                    "karsl_sign_ids": " ".join(str(item) for item in matched if item is not None),
                }
            )

    matched_tokens = {
        token: label_index[normalize_arabic(token)]
        for token in token_counts
        if normalize_arabic(token) in label_index
    }
    complete_by_split = Counter(
        row["split"] for row in rows if row["all_tokens_match"] == "True"
    )
    report = {
        "matching_policy": "exact match after conservative Arabic orthographic normalization",
        "karsl_labels": len(labels),
        "isharah_unique_tokens": len(token_counts),
        "matched_unique_tokens": len(matched_tokens),
        "token_coverage_by_occurrence": sum(token_counts[token] for token in matched_tokens) / sum(token_counts.values()),
        "complete_sentence_clips_by_split": dict(complete_by_split),
        "matched_tokens": [
            {"isharah_token": token, "karsl_sign_id": sign_id, "karsl_label": labels[sign_id], "occurrences": token_counts[token]}
            for token, sign_id in sorted(matched_tokens.items())
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".clips.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
