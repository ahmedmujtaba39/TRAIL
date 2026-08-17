"""Pre-registered decision rule for the TRAIL go/no-go experiment."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Decision:
    go: bool
    transfer_difference: float
    interaction: float
    transfer_ci: tuple[float, float]
    interaction_ci: tuple[float, float]
    within_difference: float
    rationale: str


def _bootstrap_mean(values: np.ndarray, resamples: int, rng: np.random.Generator) -> np.ndarray:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Expected a non-empty one-dimensional array.")
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    return values[indices].mean(axis=1)


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return tuple(np.quantile(samples, [alpha / 2, 1 - alpha / 2]).tolist())


def decide(
    within_phon_wer: np.ndarray,
    within_pose_wer: np.ndarray,
    transfer_phon_wer: np.ndarray,
    transfer_pose_wer: np.ndarray,
    *,
    resamples: int = 2000,
    tolerance_wer_pp: float = 2.0,
    seed: int = 42,
) -> Decision:
    """Apply a paired-bootstrap, lower-WER-is-better go/no-go rule.

    Inputs are matched per evaluation unit (for example, sentence or signer).
    Differences are measured in WER percentage points.
    """
    arrays = [within_phon_wer, within_pose_wer, transfer_phon_wer, transfer_pose_wer]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("All WER arrays must be paired and have identical shape.")

    within = within_phon_wer - within_pose_wer
    transfer = transfer_phon_wer - transfer_pose_wer
    interaction = transfer - within
    rng = np.random.default_rng(seed)
    transfer_ci = percentile_ci(_bootstrap_mean(transfer, resamples, rng))
    interaction_ci = percentile_ci(_bootstrap_mean(interaction, resamples, rng))
    transfer_mean = float(transfer.mean())
    interaction_mean = float(interaction.mean())
    within_mean = float(within.mean())
    go = (
        transfer_ci[1] < 0
        and interaction_ci[1] < 0
        and abs(within_mean) <= tolerance_wer_pp
    )
    rationale = (
        "GO: phonological conditioning improves transfer and the interaction is negative."
        if go
        else "NO-GO: the pre-registered transfer and interaction conditions were not both met."
    )
    return Decision(go, transfer_mean, interaction_mean, transfer_ci, interaction_ci, within_mean, rationale)
