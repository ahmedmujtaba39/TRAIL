"""Extract TRAIL-ready pose sequences from the nested Egyptian Sign Language archive.

The release contains labelled isolated-sign MP4 clips inside an inner ZIP.  To
keep this practical on machines with limited disk, this command materializes
only one video and its sampled JPEG frames at a time, writes a compressed
75-landmark pose sequence, then removes the temporary files.  Raw videos are
never copied into the repository.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

import cv2

from trail.hand_pose import save_hand_pose_sequence


def _materialize_inner(outer: Path, destination: Path) -> Path:
    """Copy the single nested data ZIP only when it is not already cached."""
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer) as archive:
        candidates = [item for item in archive.infolist() if item.filename.lower().endswith(".zip")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one nested ZIP in {outer}, found {len(candidates)}")
        with archive.open(candidates[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
    return destination


def _write_frames(video: Path, frame_dir: Path, sample_fps: int) -> tuple[int, float]:
    """Decode a video to sampled JPEGs without requiring a system FFmpeg install."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or sample_fps)
    stride = max(1, round(source_fps / sample_fps))
    frame_dir.mkdir(parents=True, exist_ok=True)
    read_index = written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if read_index % stride == 0:
            destination = frame_dir / f"frame_{written:05d}.jpg"
            if not cv2.imwrite(str(destination), frame):
                raise RuntimeError(f"Could not write {destination}")
            written += 1
        read_index += 1
    capture.release()
    if not written:
        raise RuntimeError("Video decoded to zero frames")
    return written, source_fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Outer 'Egyptian Sign Language (1).zip' archive")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="Process only this many pending clips")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    inner = _materialize_inner(args.archive, args.cache_root / "egyptian_rebirth_inner.zip")
    with zipfile.ZipFile(inner) as archive:
        members = sorted(item.filename for item in archive.infolist() if item.filename.lower().endswith(".mp4"))
        pending = [member for member in members if not (args.output_root / f"{Path(member).stem}.npz").exists()]
        if args.limit is not None:
            pending = pending[: args.limit]
        audit_path = args.output_root / "hand_pose_audit.csv"
        write_header = not audit_path.exists()
        fields = ["clip_id", "label", "archive_member", "status", "sampled_frames", "source_fps", "body_coverage", "left_hand_coverage", "right_hand_coverage", "detail"]
        with audit_path.open("a", encoding="utf-8", newline="") as audit:
            writer = csv.DictWriter(audit, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for index, member in enumerate(pending, 1):
                clip_id = Path(member).stem
                label = Path(member).parts[-2]
                video = args.cache_root / "clip.mp4"
                frames = args.cache_root / "frames"
                try:
                    with archive.open(member) as source, video.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    sampled_frames, source_fps = _write_frames(video, frames, args.fps)
                    coverage = save_hand_pose_sequence(frames, args.pose_model, args.hand_model, args.output_root / f"{clip_id}.npz", fps=args.fps)
                    writer.writerow({"clip_id": clip_id, "label": label, "archive_member": member, "status": "ok", "sampled_frames": sampled_frames, "source_fps": f"{source_fps:.3f}", **{name: f"{value:.6f}" for name, value in coverage.items()}, "detail": ""})
                    print(f"[{index}/{len(pending)}] {clip_id}: L={coverage['left_hand_coverage']:.0%} R={coverage['right_hand_coverage']:.0%}")
                except Exception as error:
                    writer.writerow({"clip_id": clip_id, "label": label, "archive_member": member, "status": "failed", "sampled_frames": "", "source_fps": "", "body_coverage": "", "left_hand_coverage": "", "right_hand_coverage": "", "detail": str(error)})
                    print(f"[{index}/{len(pending)}] {clip_id}: FAILED — {error}")
                finally:
                    video.unlink(missing_ok=True)
                    shutil.rmtree(frames, ignore_errors=True)


if __name__ == "__main__":
    main()
