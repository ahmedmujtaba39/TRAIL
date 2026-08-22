"""Convert ArabSign Kinect skeleton recordings into TRAIL pose sequences.

ArabSign's skeleton release stores one MATLAB ``body`` struct per frame.  Each
frame provides a 25-joint Kinect body layout, joint tracking states, and coarse
left/right hand state flags.  It does *not* provide finger landmarks, so this
converter deliberately records the data as a body-only source benchmark rather
than pretending it supplies the full handshape descriptor.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_ground_truth(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    # The official release uses a UTF-16 little-endian text file.  Retain a
    # UTF-8 fallback for mirrors that have been re-saved by users.
    encoding = "utf-16" if path.read_bytes()[:2] in {bytes.fromhex("fffe"), bytes.fromhex("feff")} else "utf-8-sig"
    with path.open(encoding=encoding) as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            sentence_id, arabic, glosses = (item.strip() for item in row[:3])
            rows[sentence_id.zfill(4)] = {
                "arabic_text": arabic,
                "glosses": " ".join(glosses.split()),
            }
    if not rows:
        raise ValueError(f"No ArabSign labels found in {path}")
    return rows


def load_kinect_pose(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return position, tracking-state, and hand-state arrays from one MAT file."""
    body = loadmat(path, simplify_cells=True)["body"]
    if not isinstance(body, list) or not body:
        raise ValueError("Expected a non-empty frame list in MAT key 'body'.")
    positions = np.stack([np.asarray(frame["Position"], dtype=np.float32).T for frame in body])
    tracking = np.stack([np.asarray(frame["TrackingState"], dtype=np.int8) for frame in body])
    hand_state = np.asarray(
        [[frame["LeftHandState"], frame["RightHandState"]] for frame in body], dtype=np.int8
    )
    if positions.ndim != 3 or positions.shape[1:] != (25, 3):
        raise ValueError(f"Unexpected Kinect pose shape {positions.shape}")
    return positions, tracking, hand_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    labels = load_ground_truth(args.ground_truth)
    files = sorted(args.raw_root.glob("*/*/*/*.mat"))
    if args.limit is not None:
        files = files[:args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    sentence_pattern = re.compile(r"/(?:train|test)/(\d{4})/")
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(args.raw_root).as_posix()
        match = sentence_pattern.search(f"/{relative}")
        if match is None:
            raise ValueError(f"Could not identify sentence ID from {relative}")
        sentence_id = match.group(1)
        label = labels.get(sentence_id)
        if label is None:
            raise ValueError(f"No ground-truth entry for sentence {sentence_id}")
        split = path.parts[-3]
        set_id = path.parts[-4]
        clip_id = f"{set_id}_{split}_{sentence_id}_{path.stem}"
        output_path = args.output_root / f"{clip_id}.npz"
        # Conversion can take several minutes on a laptop.  Reuse verified
        # existing files so interrupted runs are safe to resume.
        if output_path.exists():
            values = np.load(output_path)
            try:
                pose = values["landmarks"]
                tracking = values["tracking_state"]
            finally:
                values.close()
        else:
            pose, tracking, hand_state = load_kinect_pose(path)
            np.savez_compressed(
                output_path,
                landmarks=pose,
                tracking_state=tracking,
                hand_state=hand_state,
            )
        rows.append({
            "clip_id": clip_id,
            "set_id": set_id,
            "split": split,
            "sentence_id": sentence_id,
            "frames": len(pose),
            "body_coverage": float((tracking == 2).mean()),
            "glosses": label["glosses"],
            "arabic_text": label["arabic_text"],
            "raw_path": relative,
        })
        if index % 100 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] {clip_id}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ArabSign pose clips and {args.manifest}")


if __name__ == "__main__":
    main()
