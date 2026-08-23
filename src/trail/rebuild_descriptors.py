"""Recompute pose-derived descriptor suffixes for existing transition windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trail.descriptors import endpoint_descriptor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handshape-dim", type=int, default=6)
    args = parser.parse_args()
    values = np.load(args.input)
    if "left_descriptor" not in values or values["left_descriptor"].shape[1] < args.handshape_dim:
        raise SystemExit("Input requires a handshape prefix in left_descriptor/right_descriptor.")
    left_h = values["left_descriptor"][:, :args.handshape_dim]
    right_h = values["right_descriptor"][:, :args.handshape_dim]
    left_geometry = np.stack([endpoint_descriptor(window, use_last=True) for window in values["left"]])
    right_geometry = np.stack([endpoint_descriptor(window, use_last=False) for window in values["right"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{key: values[key] for key in values.files if key not in {"left_descriptor", "right_descriptor"}},
        left_descriptor=np.concatenate([left_h, left_geometry], axis=1),
        right_descriptor=np.concatenate([right_h, right_geometry], axis=1),
    )
    audit = {"rows": int(len(left_h)), "handshape_dim": args.handshape_dim, "geometry_dim": int(left_geometry.shape[1]), "descriptor_dim": int(args.handshape_dim + left_geometry.shape[1])}
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
