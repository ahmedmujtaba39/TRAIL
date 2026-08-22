"""Report whether a handshape model degenerates on a target landmark corpus.

This is a transfer *sanity gate*, not a target-language accuracy metric: KArSL
does not supply HamNoSys handshape ground truth.  A near-single-class output
is grounds to reject a descriptor model before it conditions Model T.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trail.handshape import load_handshape_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True, help="Target feature NPZ with features [N,60].")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = np.load(args.examples, allow_pickle=False)
    features = torch.from_numpy(values["features"]).float()
    model = load_handshape_classifier(args.checkpoint, torch.device("cpu"))
    with torch.no_grad():
        probabilities = torch.softmax(model(features), dim=-1).numpy()
    predictions = probabilities.argmax(axis=1)
    histogram = np.bincount(predictions, minlength=probabilities.shape[1])
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
    report = {
        "examples": int(len(probabilities)),
        "classes": int(probabilities.shape[1]),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "mean_entropy": float(entropy.mean()),
        "max_entropy": float(np.log(probabilities.shape[1])),
        "prediction_histogram": histogram.tolist(),
        "largest_class_share": float(histogram.max() / len(probabilities)),
        "gate": "pass" if histogram.max() / len(probabilities) < 0.80 else "fail_prediction_collapse",
        "warning": "This checks output diversity only. It is not target-language handshape accuracy without target HamNoSys labels.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
