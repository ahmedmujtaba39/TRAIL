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
