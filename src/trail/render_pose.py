"""Render normalized TRAIL pose sequences as neutral 2D skeleton MP4 clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


# Expert review focuses on signing space.  Keeping the upper body, arms, and
# hands makes the neutral visualization legible without distracting lower-body
# pose-estimation noise.
BODY_EDGES = [(0, 2), (0, 5), (2, 7), (5, 8), (0, 11), (0, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def project(sequence: np.ndarray, size: int, padding: int) -> np.ndarray:
    xy = sequence[..., :2].copy()
    valid = np.isfinite(xy).all(axis=-1)
    values = xy[valid]
    low, high = np.percentile(values, [1, 99], axis=0)
    span = max(float(np.max(high - low)), 1e-4)
    xy = (xy - (low + high) / 2) / span
    xy[..., 0] = xy[..., 0] * (size - 2 * padding) + size / 2
    xy[..., 1] = xy[..., 1] * (size - 2 * padding) + size / 2
    return xy


def draw_edge(canvas: np.ndarray, points: np.ndarray, first: int, second: int, color: tuple[int, int, int], width: int) -> None:
    a, b = points[first], points[second]
    if np.isfinite(a).all() and np.isfinite(b).all():
        cv2.line(canvas, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, width, cv2.LINE_AA)


def render(sequence: np.ndarray, output: Path, *, fps: int, size: int, label: str | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    points = project(sequence, size, max(20, size // 12))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        for frame in points:
            canvas = np.full((size, size, 3), 250, dtype=np.uint8)
            for edge in BODY_EDGES:
                draw_edge(canvas, frame, *edge, (65, 65, 65), 3)
            if len(frame) == 75:
                for start, wrist, color in ((33, 15, (29, 120, 220)), (54, 16, (40, 170, 80))):
                    draw_edge(canvas, frame, wrist, start, color, 3)
                    for first, second in HAND_EDGES:
                        draw_edge(canvas, frame, start + first, start + second, color, 2)
            for joint in (0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
                cv2.circle(canvas, tuple(np.round(frame[joint]).astype(int)), 4, (35, 35, 35), -1, cv2.LINE_AA)
            if label:
                cv2.putText(canvas, label, (14, size - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--label", default=None, help="Optional visible label; omit for blinded clips.")
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG containing the first rendered frame.")
    args = parser.parse_args()
    render(np.load(args.input), args.output, fps=args.fps, size=args.size, label=args.label)
    if args.preview is not None:
        capture = cv2.VideoCapture(str(args.output)); ok, frame = capture.read(); capture.release()
        if not ok:
            raise RuntimeError(f"Could not read rendered preview from {args.output}")
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview), frame):
            raise RuntimeError(f"Could not write {args.preview}")
    print(f"Rendered {args.output}")


if __name__ == "__main__":
    main()
