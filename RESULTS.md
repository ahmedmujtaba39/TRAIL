# Current descriptor-pipeline evidence

## Real-human HamNoSys supervision

The Public DGS Corpus smoke recording joins real signer video, time-aligned
lexical segments, and corpus HamNoSys citation forms.  We export the first
explicit HamNoSys base hand configuration as the initial `h` target, retaining
the full notation for future location, movement, and orientation parsing.

The MediaPipe-matched extraction contains 3,137 sampled examples, six observed
base-handshape classes, and 419 lexical types.  Evaluation holds out lexical
types, rather than neighbouring video frames.

| Seed | Held-out lexical-type accuracy | Chance |
| --- | ---: | ---: |
| 42 | 42.86% | 16.67% |
| 43 | 38.10% | 16.67% |
| 44 | 41.67% | 16.67% |

Mean accuracy is 40.87%.  This supports that the real-human, MediaPipe-based
classifier learns a non-trivial `h` signal on held-out DGS lexical types.  It
does not establish Arabic handshape accuracy.

## Arabic transfer sanity gate

Applying the OpenPose-trained DGS model to MediaPipe KArSL features collapsed
into one class (100% largest-class share), so it was rejected.  With the
representation-matched DGS **MediaPipe** model (seed 42), KArSL predictions
had a 46.49% largest-class share and the following counts across 1,736 target
examples: `[408, 807, 20, 0, 486, 15]`.  It therefore passes the *non-collapse*
gate, but this is not a KArSL accuracy result because KArSL has no matched
HamNoSys labels in this experiment.

The same gate passed for seeds 43 and 44 as well (57.03% and 56.97%
largest-class share, respectively).  These are still uneven distributions, so
the soft descriptor must be used with uncertainty-aware ablations rather than
treated as a hard linguistic label.

## Interpretation

This makes `h` eligible for a controlled Model T ablation only after the
remaining descriptor fields and transition experiment are run.  It does not
yet justify claims of linguistic correctness for Egyptian Sign Language;
those require independent expert review of generated clips.

## Shared-checkpoint Model T preliminary ablation

We built a disjoint Saudi split from 215 hand-qualified IshaRah clips: 180
source clips (360 weak windows) train Model T and 35 source clips (70 windows)
are held out.  Because IshaRah does not release temporal gloss boundaries, the
primary weak-window variant samples high wrist-velocity-change spans; these are
explicitly a coarticulation *proxy*, not gold boundaries.

The same descriptor-dropout checkpoint (83D descriptor, 50% full-token dropout)
was evaluated three ways on held-out windows:

| Condition | Combined position/velocity error |
| --- | ---: |
| Interpolation | 0.09380 |
| Shared Model T, descriptors masked | 0.10814 |
| Shared Model T, descriptors present | 0.09749 |

Descriptors improve the shared Model T by 9.8% relative to its masked version,
but interpolation remains better.  Therefore this is evidence for a useful
structured-conditioning signal, **not** evidence that the complete weak-source
Model T pipeline already surpasses interpolation.  The Qatari CTC utility run
is currently underpowered (11 quality-filtered synthetic templates) and is not
used as a positive result.

## DGS controlled coarticulation reconstruction

The DGS experiment is a same-language mechanism control, not a cross-language
transfer result. It uses 567 annotated adjacent-sign pairs from one Public DGS
Corpus recording. Each input context is drawn from a temporally independent
occurrence of the same lexical type, while the target is real motion at a
time-aligned boundary. A lexical-type-disjoint split is not feasible in this
recording because 566 of 567 pairs form one connected lexical-transition
component. We therefore use a chronological temporal block with all source and
target **segment instances** disjoint: 225 train pairs and 113 test pairs.

The descriptor is 83D: a six-class oracle HamNoSys handshape token in this
control plus 77D pose-derived location, local motion, orientation, and
bilateral geometry. Three seed-42/43/44 checkpoints use the same architecture
and descriptor-group dropout, allowing masked, handshape-only, and full modes
to be evaluated from each checkpoint.

| Condition | Mean combined error | Seed SD | Paired 95% CI of reduction vs. interpolation |
| --- | ---: | ---: | ---: |
| Linear interpolation | 0.027204 | 0.000000 | -- |
| Minimum-jerk interpolation | 0.029574 | 0.000000 | [-0.002801, -0.001972] |
| Shared Model T, descriptors masked | 0.025578 | 0.001964 | [0.000157, 0.003108] |
| Shared Model T, handshape only | 0.026264 | 0.001592 | [-0.000486, 0.002385] |
| Shared Model T, full 83D | **0.023569** | **0.001516** | **[0.001034, 0.006360]** |

The full descriptor reduces mean error by 13.36% relative to linear
interpolation; its paired bootstrap interval excludes zero. The handshape-only
benefit is not statistically stable in this small control. The result supports
the full structured transition model, but does not by itself prove a
cross-language phonological-transfer claim.

An exploratory endpoint-articulatory-distance analysis did not show a monotonic
relative-gain trend across low/medium/high distance tertiles (20.0%, 10.2%, and
11.1%, respectively). It must therefore be reported as a diagnostic null
result, not as evidence for the proposed language-level distance law.
