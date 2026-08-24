"""Create a blinded local TRAIL-versus-interpolation expert-review packet."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Folder safe to share with reviewers")
    parser.add_argument("--condition-key", type=Path, required=True, help="Private condition mapping; do not share")
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    trail = {path.stem: path for path in (args.rendered_root / "articulatory").glob("eg_*.mp4")}
    linear = {path.stem: path for path in (args.rendered_root / "interpolation").glob("eg_*.mp4")}
    sample_ids = sorted(set(trail) & set(linear))
    if not sample_ids:
        raise SystemExit("No matched articulatory/interpolation rendered clips found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    public_rows, private_rows = [], []
    for index, sample_id in enumerate(sample_ids, 1):
        pair_id = f"pair_{index:02d}"
        items = [("TRAIL", trail[sample_id]), ("baseline", linear[sample_id])]
        rng.shuffle(items)
        for presentation, (condition, source) in zip(("A", "B"), items):
            destination = args.output_root / f"{pair_id}_{presentation}.mp4"
            shutil.copy2(source, destination)
            public_rows.append({"pair_id": pair_id, "clip": presentation, "file": destination.name})
            private_rows.append({"pair_id": pair_id, "clip": presentation, "sample_id": sample_id, "condition": condition, "source": str(source)})
    with (args.output_root / "review_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "clip", "file"]); writer.writeheader(); writer.writerows(public_rows)
    with args.condition_key.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "clip", "sample_id", "condition", "source"]); writer.writeheader(); writer.writerows(private_rows)
    (args.output_root / "README.txt").write_text(
        "Egyptian Sign Language transition review pilot\n\n"
        "For each pair, watch Clip A and Clip B. The clips show the same two isolated signing units with different automatically generated intervening motion.\n"
        "Please do not infer that a pair is a grammatical sentence. Rate only the local transition: which clip has the more natural, plausible, and less abrupt connection between units?\n"
        "A separate response form should record: pair ID, preferred clip (A/B/tie), confidence (1-5), and optional comment on handshape, movement, location, orientation, or other issue.\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(sample_ids)} blinded pairs in {args.output_root}")


if __name__ == "__main__":
    main()
