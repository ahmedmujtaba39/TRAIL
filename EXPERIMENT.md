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

## First real-data gate: QSL audit

Run this once after downloading the front-view (`rec0.svo`) clips. It creates a
reproducible manifest without copying the raw videos:

```powershell
python -m trail.prepare_qsl `
  --workbook C:\Users\cp\Downloads\JUMLA_QSL-2022.xlsx `
  --data-root "C:\Users\cp\Downloads\Student-learning-hub-main\Qatari Sign Language"
```

The protocol treats AS one-token clips as a **proxy isolated lexicon** and keeps
AT and MA continuous clips strictly held out for evaluation. Report the result
as Arabic intent-token WER, not formal QSL-gloss WER: JUMLA provides Arabic
intent sequences rather than an expert temporal QSL gloss tier.

Create the matching Saudi source index with:

```powershell
python -m trail.prepare_isharah `
  --source-root C:\Users\cp\Downloads\isharah500_compressed\isharah500_compressed
```

This produces a sequence-level source manifest from Isharah's published SI
split. It is not yet a transition manifest: transition windows are created only
after pose extraction and source-side temporal segmentation. The phonological
condition additionally requires an auditable descriptor table; it must not be
silently substituted with labels from held-out QSL continuous clips.

### QSL SVO decode check

JUMLA's `.svo` front recordings can be decoded with FFmpeg on this Intel GPU
machine; ZED Explorer is not required. Test one clip first and visually inspect
the frames before any batch extraction:

```powershell
python -m trail.extract_svo `
  "C:\Users\cp\Downloads\Student-learning-hub-main\Qatari Sign Language\Participant_AS\f_AS059\rec0.svo" `
  data\cache\qsl_decode_check `
  --ffmpeg "C:\path\to\ffmpeg.exe" --max-frames 12
```

The extractor strips the 296-byte SVO header, uses FFmpeg's CPU H.264 decoder,
and retains only the left half of the stereo front recording. `data/cache/` is
ignored by Git.

Then turn the decoded frames into a normalized `[frames, 33, 3]` pose array:

```powershell
python -m trail.extract_pose data\cache\qsl_decode_check data\cache\qsl_decode_check.npy `
  --model assets\models\pose_landmarker_full.task
```

This first representation is 33 body landmarks, centered at the shoulder
midpoint and scaled by shoulder width. The experiment code will report it as a
body-pose baseline; hand landmarks are a required later extension before making
strong claims about handshape or orientation.

For a disk-bounded pilot, process a small number of clips at a time. The command
creates only the final `.npy` pose arrays and deletes temporary JPEG frames after
each clip:

```powershell
python -m trail.batch_qsl_pose `
  --manifest data\processed\jumla_qsl_manifest.csv `
  --pose-root data\processed\qsl_body_pose `
  --cache-root data\cache\qsl_work `
  --ffmpeg "C:\path\to\ffmpeg.exe" `
  --model assets\models\pose_landmarker_full.task `
  --limit 9
```

Isharah needs no video decoder: its source archives already contain JPEG frame
sequences. Extract Saudi poses directly from a small source pilot with:

```powershell
python -m trail.batch_isharah_pose `
  --manifest data\processed\isharah_sequences.csv `
  --pose-root data\processed\isharah_body_pose `
  --cache-root data\cache\isharah_work `
  --model assets\models\pose_landmarker_full.task --split train --limit 9
```

For the initial Model-T engineering pilot, create fixed-length **weak** Saudi
transition windows after source poses exist:

```powershell
python -m trail.prepare_transitions `
  --pose-root data\processed\isharah_body_pose `
  --audit data\processed\isharah_body_pose\pose_audit.csv --min-coverage 0.70 `
  --output data\processed\isharah_weak_transitions.npz
```

Because Isharah's released annotations are sequence-level rather than temporal
gloss boundaries, these windows supervise generic continuous motion completion.
They are useful for the pose-only baseline and pipeline validation, but they are
not gold coarticulation labels. The final phonological claim requires the
separate descriptor table and a clearly reported alignment protocol.

Train the pose-only Model-T engineering baseline (the phonological flag is
intentionally refused until audited descriptors are available):

```powershell
python -m trail.train_transition `
  --windows data\processed\isharah_weak_transitions.npz `
  --output runs\pilot\pose_only_model_t.pt
```

After AS poses and a Model-T checkpoint exist, generate matched target
sequences with two non-phonological conditions:

```powershell
python -m trail.synthesize --manifest data\processed\jumla_qsl_manifest.csv `
  --pose-root data\processed\qsl_body_pose --output-root data\processed\synthetic_qsl `
  --condition interpolation

python -m trail.synthesize --manifest data\processed\jumla_qsl_manifest.csv `
  --pose-root data\processed\qsl_body_pose --output-root data\processed\synthetic_qsl `
  --condition pose_only --checkpoint runs\pilot\pose_only_model_t.pt
```

### Structured-conditioning pilot

The workshop pilot also includes an **articulatory Model T** condition. Its
18D descriptors are automatically derived from pose boundaries: bilateral wrist
locations, forearm directions, and local wrist motion. This is a reproducible
structured-conditioning test, not expert phonological annotation. The paper must
reserve “phonological” for a later descriptor table validated by sign-language
experts and enriched with handshape/orientation features.

### Downstream utility measurement

Train the same CTC recognizer for each synthetic condition, then evaluate on
real AT/MA recordings that were never used to train Model T. The current
prototype evaluates Arabic intent-token WER. Its templates overlap with the
synthetic set, so the first result is a controlled transition-utility result,
not a claim of open-vocabulary sentence generalization.

Before interpreting synthetic results, run the **real-data oracle**: train the
same recognizer on real AT continuous pose sequences and evaluate only on MA.
This is a diagnostic upper bound for the current pose/CTC implementation, not a
zero-resource result.

### Active-sign trimming

The QSL proxy lexicon clips include leading/trailing rest. Before synthetic
composition, use conservative elbow/wrist motion-energy trimming and inspect its
audit before treating the outputs as sign units:

```powershell
python -m trail.trim --input-root data\processed\qsl_body_pose `
  --output-root data\processed\qsl_body_pose_trimmed `
  --audit data\processed\qsl_body_pose_trimmed\trim_audit.csv
```

### Saudi source-control feasibility audit

The planned Saudi control needs isolated KArSL signs to compose the same Saudi
intent-token sequences that are evaluated on held-out Isharah continuous clips.
Do not infer lexical identity merely because two Arabic words look similar.  The
first reproducible screen is an exact match after conservative orthographic
normalization; any synonym or dialectal match must be reviewed by a signer or
other qualified annotator before it is used.

```powershell
python -m trail.audit_karsl_overlap `
  --labels C:\path\to\KARSL-502_Labels.xlsx `
  --isharah-manifest data\processed\isharah_sequences.csv `
  --output data\processed\karsl_isharah_overlap.json
```

On the currently supplied releases this screen finds 57 of 388 Isharah tokens
(12.4% by occurrence), but just one complete sequence template (15 signer/split
instances).  This is insufficient for a WER-based Saudi control.  Treat it as
an extraction sanity check only, not a source-control result.  A publishable
Saudi control requires a larger manually verified KArSL--Isharah lexical map,
or a source corpus with aligned isolated and continuous vocabulary.

### Hand-aware articulatory descriptors (main-paper upgrade)

The workshop body-only condition is an 18D endpoint feature.  It is not enough
to support a strong handshape-level claim.  The upgraded descriptor pipeline
uses 75 normalized landmarks: 33 body joints plus 21 joints for each hand.  It
creates a 77D endpoint descriptor consisting of body/wrist motion (18D), each
hand's fingertip geometry, palm normal, finger spread, local motion (26D per
hand), and bilateral hand geometry/motion (7D).  Descriptors are standardized
on the source training set and independently feature-masked while training
Model T.

First run a small extraction-quality audit.  The hand-landmarker model must be
supplied locally; it is intentionally excluded from Git.

```powershell
python -m trail.extract_hand_pose `
  --frames data\cache\example_frames `
  --pose-model assets\models\pose_landmarker_full.task `
  --hand-model assets\models\hand_landmarker.task `
  --output data\processed\hand_descriptor_smoke\example.npz
```

For the actual experiment, extract source continuous and target isolated clips
into separate hand-aware roots, filter clips using `hand_pose_audit.csv`, build
new 77D transition windows, then run the same three-way comparison:
interpolation, pose-only Model T, and hand-aware Model T.  The recognizer must
use landmarks of the same dimensionality as the generated clips.

The two disk-bounded batch commands are `trail.batch_isharah_hand_pose` for
Saudi source clips and `trail.batch_qsl_hand_pose` for Qatari isolated/eval
clips.  Run a 10-clip quality audit first, inspect hand coverage, and only then
launch the full extraction; hand-aware landmark files are intentionally stored
outside Git.

### Proposal-level phonological token

The full TRAIL condition prefixes the 77D deterministic descriptor with a soft
handshape distribution from `HandshapeClassifier`.  The classifier must be
trained on an isolated, **signer-held-out** alphabet/digit or handshape-labelled
set (KArSL Arabic letters/digits are the intended Saudi source).  Its accuracy
and uncertainty must be reported per language.  Do not use a generic ASL or
German pretrained classifier as ground truth for Arabic handshapes.

`trail.build_handshape_examples` turns a reviewed isolated-clip manifest
(`pose_path,handshape_label,split`) into 60D wrist-centred, palm-aligned inputs.
`trail.train_handshape` then produces the soft classifier checkpoint.  Pass that
checkpoint to both `trail.prepare_transitions --handshape-checkpoint` and
`trail.synthesize --handshape-checkpoint`; this yields a descriptor of
`77 + H` dimensions, where `H` is the source handshape inventory size.

For the causal ablation, train **one** articulatory/phonological Model T with
`--descriptor-token-drop-prob 0.5`.  This masks both descriptor tokens on half
of training examples.  At inference, synthesize `articulatory` (tokens present)
and `pose_only` (the exact same checkpoint, tokens masked).  Separate training
runs do not satisfy the shared-weight causal contract in the proposal.
