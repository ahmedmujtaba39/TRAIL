# Dataset setup

TRAIL keeps all raw and processed datasets out of version control. This is
intentional: the source datasets are large and subject to their own access and
licensing terms.

## Isharah-500 (Saudi Sign Language)

This is the continuous Saudi source dataset for the first Saudi-to-Qatari
go/no-go experiment. Download the following from the official Isharah release:

- `Annotations/SI/{train,dev,test}.txt` for the signer-independent protocol.
- Archives `00.zip` through `14.zip`.

Each annotation row is pipe-separated as `id|gloss|text`. Use the `gloss`
field as the CSLR target. Do not use the unseen-sentence (`US`) split for the
confirmatory signer-independent experiment unless it is reported separately.

Archive location is configurable. A typical local layout is:

```text
data/raw/isharah500/
  archives/00.zip ... 14.zip
  Annotations/SI/train.txt
  Annotations/SI/dev.txt
  Annotations/SI/test.txt
```

## KArSL-502 (Saudi Sign Language)

Use only the raw-video release for the isolated Saudi control lexicon. RGB
frames, depth frames, and released skeleton files are not required because the
pipeline extracts a consistent pose representation itself.

## JUMLA-QSL-22 (Qatari Sign Language)

For the initial pseudo-zero-resource validation, first obtain the dataset
spreadsheet and all participant CSV files. Select a signer-disjoint subset of
front-view recordings only after the metadata audit. Do not download every
camera view by default. JUMLA sentence labels do not replace a human isolated
Qatari lexicon; any target units derived from it must be labelled as a
continuous-data-derived proxy lexicon.

## Data governance

- Never commit raw videos, pose arrays, extracted frames, credentials, or
  participant-identifying metadata.
- Preserve the original dataset licence and citation requirements.
- Record exact download versions and preprocessing commands in an experiment
  run directory, not in Git.
