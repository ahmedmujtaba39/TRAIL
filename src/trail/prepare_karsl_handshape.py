"""Create a signer-held-out KArSL digit/alphabet handshape dataset.

KArSL archive member names begin ``<signer>_<view>_<class>_...``.  The caller
must explicitly provide train/test signer prefixes; the script records them in
the output manifest rather than inferring a split from clip order.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from trail.hand_pose import save_hand_pose_sequence
from trail.karsl import load_labels


def _members(archive: Path) -> list[str]:
    result = subprocess.run(["tar", "-tf", str(archive)], text=True, capture_output=True, check=True)
    return [line for line in result.stdout.splitlines() if line.endswith(".mp4")]


def _extract_frames(video: Path, destination: Path, ffmpeg: Path, fps: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"fps={fps}", "-frames:v", "80", str(destination / "frame_%05d.jpg")], text=True, capture_output=True)
    if result.returncode or not any(destination.glob("frame_*.jpg")):
        raise RuntimeError(result.stderr.strip() or "FFmpeg produced no JPEG frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True, help="Explicit split assigned by the outer KArSL recording group.")
    parser.add_argument("--signer", default=None, help="Optional inner recording ID filter; omit to retain every member in this archive.")
    parser.add_argument("--per-class-per-signer", type=int, default=5)
    parser.add_argument("--max-class", type=int, default=69, help="1-31 digits, 32-69 Arabic alphabet variants.")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    labels = load_labels(args.labels); args.output_root.mkdir(parents=True, exist_ok=True); args.cache_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[int, str], list[str]] = defaultdict(list)
    for member in _members(args.archive):
        class_id = int(member.split("/", 1)[0]); basename = Path(member).name; signer = basename.split("_", 1)[0]
        if class_id <= args.max_class and (args.signer is None or signer == args.signer): grouped[(class_id, signer)].append(member)
    selected = [(class_id, signer, member) for (class_id, signer), members in sorted(grouped.items()) for member in members[:args.per_class_per_signer]]
    manifest = args.output_root / "karsl_handshape_manifest.csv"; existing = set()
    if manifest.exists():
        with manifest.open(encoding="utf-8", newline="") as handle: existing = {row["sample_id"] for row in csv.DictReader(handle)}
    new = not manifest.exists()
    with manifest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "pose_path", "handshape_label", "class_id", "signer_id", "split", "body_coverage", "left_hand_coverage", "right_hand_coverage"])
        if new: writer.writeheader()
        for index, (class_id, signer, member) in enumerate(selected, 1):
            sample_id = f"{class_id:04d}_{signer}_{Path(member).stem[-8:]}"; output = args.output_root / f"{sample_id}.npz"
            if sample_id in existing or output.exists(): continue
            video_dir, frame_dir = args.cache_root / "video" / sample_id, args.cache_root / "frames" / sample_id
            try:
                video_dir.mkdir(parents=True)
                subprocess.run(["tar", "-xf", str(args.archive), "-C", str(video_dir), member], check=True)
                video = video_dir / member; _extract_frames(video, frame_dir, args.ffmpeg, args.fps)
                audit = save_hand_pose_sequence(frame_dir, args.pose_model, args.hand_model, output, fps=args.fps)
                writer.writerow({"sample_id": sample_id, "pose_path": str(output), "handshape_label": labels[class_id], "class_id": class_id, "signer_id": signer, "split": args.split, **{key: f"{value:.6f}" for key, value in audit.items()}})
                handle.flush()
                print(f"[{index}/{len(selected)}] class={class_id} signer={signer}: saved")
            except Exception as error:
                print(f"[{index}/{len(selected)}] class={class_id} signer={signer}: FAILED - {error}")
            finally:
                # On Windows an antivirus/indexer can briefly retain a decoded
                # MP4 handle.  Cleanup must never abort the resumable run.
                if video_dir.exists(): shutil.rmtree(video_dir, ignore_errors=True)
                if frame_dir.exists(): shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == "__main__": main()
