"""Isharah sequence manifest construction.

Isharah stores JPEG frames inside numbered zip archives.  A manifest keeps the
archive/member reference intact so the raw archive is never copied or committed
just to prepare an experiment.
"""

from __future__ import annotations

import csv
from pathlib import Path


SPLITS = ("train", "dev", "test")


def build_manifest(source_root: Path, output_csv: Path) -> dict[str, int]:
    records: list[dict[str, str]] = []
    for split in SPLITS:
        annotation = source_root / "Annotations" / "SI" / f"{split}.txt"
        if not annotation.exists():
            raise FileNotFoundError(f"Missing Isharah annotation file: {annotation}")
        with annotation.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            for row in reader:
                clip_id = row["id"].strip()
                archive_id = clip_id.split("_", maxsplit=1)[0]
                archive = source_root / f"{archive_id}.zip"
                if not archive.exists():
                    raise FileNotFoundError(f"Missing Isharah archive: {archive}")
                records.append(
                    {
                        "clip_id": clip_id,
                        # SI is a predefined split; signer ids are unavailable in
                        # the released annotation file, so no signer-disjoint claim
                        # is made here.
                        "split": split,
                        "glosses": row["gloss"].strip(),
                        "text": row["text"].strip(),
                        "archive_path": str(archive),
                        "member_prefix": f"{archive_id}/{clip_id}/",
                    }
                )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip_id", "split", "glosses", "text", "archive_path", "member_prefix"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return {split: sum(item["split"] == split for item in records) for split in SPLITS}
