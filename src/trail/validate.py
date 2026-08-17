"""Validate required manifests before a real experiment is launched."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


REQUIRED = {"clip_id", "signer_id", "glosses", "pose_path", "split"}


def validate_manifest(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    errors: list[str] = []
    if missing:
        errors.append(f"{path.name}: missing columns {sorted(missing)}")
        return errors
    if not set(frame["split"].dropna()) <= {"train", "dev", "test"}:
        errors.append(f"{path.name}: split must be train/dev/test")
    signer_splits = frame.groupby("signer_id")["split"].nunique()
    leaked = signer_splits[signer_splits > 1]
    if not leaked.empty:
        errors.append(f"{path.name}: signer leakage across splits: {', '.join(map(str, leaked.index.tolist()))}")
    for pose_path in frame["pose_path"].dropna():
        if not Path(pose_path).exists():
            errors.append(f"{path.name}: missing pose file {pose_path}")
            break
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    errors = [error for path in args.manifests for error in validate_manifest(path)]
    if errors:
        raise SystemExit("\n".join(errors))
    print("All manifests passed schema, path, and signer-leakage checks.")
