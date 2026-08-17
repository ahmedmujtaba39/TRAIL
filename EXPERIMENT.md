# Saudi-to-Qatari TRAIL go/no-go experiment

This package implements the first TRAIL experiment: train a transition model on
Saudi Sign Language and test whether explicit phonological conditioning transfers
better to Qatari Sign Language than raw-pose conditioning alone.

## Experimental design

Three transition conditions are evaluated in two regimes:

| Regime | Phonological-T | Pose-only-T | Interpolation |
|---|---:|---:|---:|
| Saudi -> Saudi control | yes | yes | yes |
| Saudi -> Qatari transfer | yes | yes | yes |

The confirmatory statistic is the difference-in-differences interaction:

`(WER_phon,Qatar - WER_pose,Qatar) - (WER_phon,Saudi - WER_pose,Saudi)`.

Lower WER is better. The experiment is a **GO** only if the paired bootstrap
confidence interval for the Qatari phonological-vs-pose-only difference and for
the interaction are both below zero, while the Saudi difference stays within the
pre-registered tolerance in `configs/saudi_to_qatari.yaml`.

## Data required

Raw datasets are never committed. Place them under `data/raw/` and create the
four manifests named in the config. Each manifest is a CSV with these columns:

`clip_id,signer_id,glosses,pose_path,split`

- `pose_path`: path to a NumPy `.npy` array shaped `[frames, joints, 3]`.
- `glosses`: whitespace-separated gloss sequence.
- `split`: `train`, `dev`, or `test`; splits must be signer-disjoint.

`isharah_transitions.csv` additionally requires:

`left_boundary_path,right_boundary_path,transition_path,transition_length`

Before running the real experiment, validate QSL dictionary-to-JUMLA gloss
coverage. The target isolated dictionary must cover every gloss used for a
synthetic target sentence, or the sentence must be excluded and coverage reported.

## Quick start

```powershell
cd C:\Users\cp\Downloads\TRAIL
uv sync
uv run trail-smoke
```

The smoke test is synthetic and only verifies that masking, interpolation,
transition generation, and the pre-registered decision rule execute correctly.
It is not a scientific result.
