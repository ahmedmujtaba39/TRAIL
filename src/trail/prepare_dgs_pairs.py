"""Build lexically controlled DGS coarticulation reconstruction pairs.

This is a *same-language mechanism control*, not a claim that DGS segments are
clean citation-form clips. Inputs are temporally disjoint occurrences of the
same annotated DGS lexical types; the target is real boundary motion between
adjacent annotated types in a held-out occurrence. The manifest retains all
segment IDs and frame spans for audit.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def frame_number(timecode: str, fps: int) -> int:
    hours, minutes, seconds, frames = (int(part) for part in timecode.split(":"))
    return (((hours * 60 + minutes) * 60 + seconds) * fps) + frames


def independent_example(row: dict[str, str], candidates: list[dict[str, str]], *, fps: int) -> dict[str, str] | None:
    """Choose the most temporally distant occurrence of the same lexical type."""
    start = frame_number(row["timecode_start"], fps)
    options = [item for item in candidates if item["segment_id"] != row["segment_id"]]
    return max(options, key=lambda item: abs(frame_number(item["timecode_start"], fps) - start), default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--tier-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--boundary-frames", type=int, default=6)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--max-gap-frames", type=int, default=25, help="Reject discontinuous neighbours separated by more than this many frames.")
    args = parser.parse_args()

    rows = [row for row in csv.DictReader(args.segments.open(encoding="utf-8")) if row["tier_id"] == args.tier_id and row["handshape_h"] != "unknown"]
    rows.sort(key=lambda row: frame_number(row["timecode_start"], args.fps))
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[row["type_id"]].append(row)

    examples: list[dict[str, str | int]] = []
    half = args.duration // 2
    for left_target, right_target in zip(rows, rows[1:]):
        left_end = frame_number(left_target["timecode_end"], args.fps)
        right_start = frame_number(right_target["timecode_start"], args.fps)
        gap = right_start - left_end
        if gap < 0 or gap > args.max_gap_frames:
            continue
        left_source = independent_example(left_target, by_type[left_target["type_id"]], fps=args.fps)
        right_source = independent_example(right_target, by_type[right_target["type_id"]], fps=args.fps)
        if left_source is None or right_source is None:
            continue
        source_left_end = frame_number(left_source["timecode_end"], args.fps)
        source_right_start = frame_number(right_source["timecode_start"], args.fps)
        boundary = round((left_end + right_start) / 2)
        if source_left_end < args.boundary_frames or boundary < half:
            continue
        examples.append({
            "example_id": f"{left_target['segment_id']}_{right_target['segment_id']}",
            "tier_id": args.tier_id,
            "left_target_segment_id": left_target["segment_id"],
            "right_target_segment_id": right_target["segment_id"],
            "left_source_segment_id": left_source["segment_id"],
            "right_source_segment_id": right_source["segment_id"],
            "left_type_id": left_target["type_id"],
            "right_type_id": right_target["type_id"],
            "left_h": left_target["handshape_h"],
            "right_h": right_target["handshape_h"],
            "left_source_start": source_left_end - args.boundary_frames,
            "left_source_end": source_left_end,
            "right_source_start": source_right_start,
            "right_source_end": source_right_start + args.boundary_frames,
            "target_start": boundary - half,
            "target_end": boundary - half + args.duration,
            "gap_frames": gap,
            "duration": args.duration,
        })
    if not examples:
        raise SystemExit("No usable DGS paired examples. Check tier, frame rate, and gap threshold.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(examples[0]))
        writer.writeheader()
        writer.writerows(examples)
    print(f"Wrote {len(examples)} DGS lexical reconstruction pairs to {args.output}")


if __name__ == "__main__":
    main()
