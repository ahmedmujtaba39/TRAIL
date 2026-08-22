"""CPU-only extraction of the front camera from JUMLA ``rec0.svo`` files."""

from __future__ import annotations

import subprocess
from pathlib import Path


class SvoDecodeError(RuntimeError):
    """Raised when FFmpeg cannot decode a JUMLA SVO file."""


def extract_left_frames(
    source: Path,
    output_dir: Path,
    ffmpeg: Path,
    *,
    fps: int = 12,
    max_frames: int | None = None,
) -> None:
    """Extract the left view of a side-by-side H.264 SVO recording.

    JUMLA's `rec0.svo` starts with a 296-byte SVO header.  The underlying H.264
    stream is stereo side-by-side, therefore only its left half is used.  This
    uses FFmpeg's software decoder and does not require the NVIDIA-only ZED SDK.
    """
    if not source.exists():
        raise FileNotFoundError(source)
    if not ffmpeg.exists():
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%05d.jpg"
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "h264", "-skip_initial_bytes", "296", "-i", str(source),
        "-vf", f"crop=iw/2:ih:0:0,fps={fps}",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(pattern))
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SvoDecodeError(completed.stderr.strip() or "FFmpeg failed without an error message.")
    if not any(output_dir.glob("frame_*.jpg")):
        raise SvoDecodeError("FFmpeg completed but wrote no frames.")
