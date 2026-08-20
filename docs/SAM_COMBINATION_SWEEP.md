# Which SAM combination works best — 78 variants, leave-one-image-out

**Answer: the combination already deployed.** Nothing tested beat it by more than measurement
noise, and the two things that looked like wins died under controls. The useful output is the
mechanism — *why* the deployed shape is right — plus a list of where the real headroom is not.

Run 2026-08-18. 78 variants across five families, then adversarial verification. Harness:
`leave-one-image-out over the four ground-truth images`, threshold chosen on training data
only, test rows sampled uniformly so class balance matches the real frames.

> **Every number here is at a 90,000-row training cap.** Absolute IoU therefore sits below
> the 0.821 the shipped baseline gets from the full training set. These numbers **rank
> architectures**; they are not deployment estimates.

Harness validation: the deployed recipe scores **0.8310** here against the **0.821** that
the README reports from the full-scale leave-one-image-out study.

## The ranking

| architecture | LOO IoU@0.5 | fold sd |
|---|---|---|
| **17-feature MLP + 273-d hybrid MLP, mean probability (deployed)** | **0.8310** | 0.0445 |
| 273-d hybrid MLP alone | 0.8057 | 0.0610 |
| 17 hand-crafted features alone | 0.7397 | **0.0164** |
| 256-d SAM embedding alone | 0.7261 | 0.1253 |

Neither half is usable alone and the pair beats the hybrid that contains both. Best variant
found across all 78: 0.8392, which did not survive reseeding (below).

## Why the deployed shape is right

**The ensemble's +0.025 is feature-set diversity, not variance reduction.** Averaging two
273-d hybrids that differ only in random seed scores 0.8071 — barely above one of them. The
gain needs a genuinely different feature set, and the two models' errors are only 0.24–0.47
correlated.

**Equal weights are a real optimum, not a convention.** Sweeping the 17-feature model's
weight: `0.8057 / 0.8145 / 0.8221 / 0.8310 / 0.8111 / 0.7803 / 0.7397` at w = 0, .25, .4,
**.5**, .6, .75, 1. w=0.5 wins on 4 of 4 folds against both neighbours. A learned stacker is
*worse* (0.8150), because it weights the hybrid ~7.5 against ~2.8 for the 17-feature arm and
so discards the insurance exactly when it is needed.

**The 17 hand-crafted features are the cross-image stabiliser.** They are the weakest single
model but by far the most stable across folds (sd 0.0164 against 0.061–0.125 for anything
containing SAM). On the one out-of-distribution image, SAM-only collapses to **0.5231** while
17-only holds **0.7184**. They also must sit *inside* the same model as the SAM dims, not
merely beside it: pairing 17-only with a pure-SAM model scores 0.7956 versus 0.8310 for
pairing it with the hybrid.

**One image decides every ranking.** `LARGE_343_75` is genuinely out of distribution —
centroid displacement from its training images is 1.130 sd in 17-feature space against
0.251–0.384 for the others. It is the hardest fold for all 78 variants, its across-run range
is 0.474–0.809 against 0.021–0.031 sd for the other three, and essentially every mean
difference in the table is that one fold moving.

## What does not work, and why

**PCA cannot shrink the SAM block.** The whitened-PCA ladder is flat from k=8 to k=256
(0.782 / 0.771 / 0.788 / 0.764 / 0.772 / 0.789), which suggests SAM's usable signal is
low-dimensional. But the decisive control is full-rank PCA-256, which discards *no*
information and differs from the raw 273-d input only by a rotation plus rescaling — it still
scores 0.7889 against 0.8057. So the ~0.02 deficit is **the transform, not the truncation**:
`StandardScaler` forces unit variance per column, so any PCA basis arrives whitened, which
inflates a long tail of near-noise directions (32 dims = 89% of variance, 127 needed for 99%)
and destroys the high-variance-first prior that L2-regularised training has on correlated raw
inputs. Inside the production ensemble, swapping the hybrid for `[17 | PCA-32]` costs 0.0132
with the baseline ahead on 4/4 folds. PCA *is* finding real structure — it beats a random
32-d projection on 4/4 folds, 0.7884 vs 0.7587 — it just cannot be cashed in.

**The estimator barely matters on the 273-d design.** LogisticRegression 0.7840, ExtraTrees
0.7910, HGB 0.7997, RandomForest 0.8039, MLP(64,32) 0.7941 — a 0.020 band, *narrower than
the MLP's own reseeding spread of 0.026*. The SAM dims make the boundary close to linearly
separable; capacity is not the binding constraint. On the 17-d design the estimator matters
enormously (MLP 0.7397 vs HGB 0.6757, and HGB collapses to 0.5463 on the unseen image because
trees threshold on absolute values that shift between specimens).

**Rebalancing SAM against the 17 features is a provable no-op.** The motivating arithmetic is
real — after per-column standardisation the 17 features hold 17/273 = 6.2% of input energy —
but a diagonal per-block gain is inside every estimator's hypothesis space, so it just
rescales its first-layer weights. Proof without noise: logistic regression is convex and
deterministic, and gives 0.7840 at the production ratio versus 0.7837 at ratio 1.0, a delta
of 0.0003.

**Calibration and thresholding are a dead end.** Total oracle headroom — threshold chosen on
the answer sheet — is **+0.0045**, one third of this project's reseeding noise, and three of
the four folds have *literally zero*. Raw 0.5 is structurally near-optimal because two biases
cancel: balanced 50/50 training inflates crack posteriors (pushing the honest cut up to an
equivalent raw 0.75, which costs −0.034), while IoU is monotone in F1 so the optimum sits
near F1*/2 = 0.455 (pushing it down). Calibration does fix the area bias — predicted area
matches true area to 1–2% relative, against ~7% over-prediction raw — and buys no IoU.

## The one candidate, and why it is not recommended

Adding a **273-d HistGradientBoosting model as a third ensemble member** was the only variant
to claim a win: 0.8392 against 0.8310, on 4/4 folds.

It survived a leakage audit and it is not a single-fold artifact — paired against identical
base fits it gives +0.0082 / +0.0083 / +0.0076 across seeds (gap sd 0.0003), positive on
12/12 fold comparisons. An attribution control confirms the third member must be a *different
estimator family*: swapping the HGB for a reseeded 273-d MLP scores 0.8260, **below**
baseline, so "a third vote helps" is false.

It still should not be adopted, for three independent reasons.

1. **The level is not reproducible.** Reseeded, the claim averages 0.8322 and 3 of 5 fresh
   seeds fall *below* the 0.8310 baseline headline. Excluding its own lucky seed its
   remaining seeds average 0.8308 — i.e. −0.0002. The baseline's own reseeds span
   0.8164–0.8359.
2. **It is worse on the image that matters most.** On confirmed crack-free specimen it looked
   excellent — false positives 0.146% → 0.086%, a 41% relative cut, lower on 4/4 seeds with
   non-overlapping ranges. But HGB is the highest-precision member, so averaging it in lowers
   every probability, which cuts false positives simply by predicting less. At **matched
   false-positive rate** the advantage is +0.018 / +0.008 / +0.019 on the three
   in-distribution folds and **−0.025 on the out-of-distribution one** — mean +0.005. The
   273-d HGB scores 0.6692 on `LARGE_343_75` against 0.86–0.88 elsewhere, so giving it a
   one-third vote drags exactly the fold that resembles a new specimen.
3. **It costs a third model** at inference across 71 images of 3–32 MP, and a new estimator
   family in the artifact.

For the question actually asked — which combination works best *for all* the images,
including ones not yet seen — reason 2 decides it.

## Two code-level findings

**The retrain and the shipped baseline disagree on MLP shape, and it does not matter.**
`pipeline.py` fits MLP(128,64) max_iter=400 while the shipped models are (64,32) max_iter=300.
Measured on the 273-d design: 0.7923 for (128,64) versus 0.7941 for (64,32) as a 3-seed mean —
indistinguishable. The larger net buys nothing and costs ~50% more fit time, so aligning the
retrain down to (64,32) is a free simplification, not an accuracy change.

**A float32/float64 summation difference moved IoU by 0.021.** A hand-rolled
`(x - x.mean(0)) / x.std(0)` accumulates in float32 on these arrays; sklearn's
`StandardScaler` accumulates in float64. Max discrepancy was 2.1e-3 on a unit-variance
column — 0.2% of a standard deviation — and it was enough to move MLP early stopping into a
different basin: 0.8057 → 0.7843, with the hard fold dropping 0.735 → 0.674. **Anyone
comparing architectures across two scripts that both "standardise then fit an MLP" may be
reading a numerics artefact rather than an architecture difference.**

## Where the headroom actually is

1. ~~**Spatial post-processing**~~ — **TESTED, and it was right.** See the section below;
   a minimum-area filter is now on by default. It was the largest remaining lever and it
   beat every architecture change in this document.
2. **More ground truth.** Four `_gt.npy` files against 71 labelled images, and one atypical,
   different-magnification frame decides every ranking above. A fifth and sixth ground-truth
   image would be worth more than any architecture change in this document.
3. **Vary the training-set size.** All 78 runs used a 90,000-row cap, never varied once. HGB
   and the 273-d MLP are the members most likely to be data-starved there, so the third-member
   effect could grow or vanish at full scale.

**Do not spend more effort on** feature-set selection, SAM dimensionality reduction, block
rescaling, estimator choice on the 273-d design, calibration, or thresholding. Each is
measured above and each is a no-op or a loss.

## Reproducing

The harness and fold builders live in the session scratch directory, not the repo, because
they read the 2.1 GB reference feature stacks and are not part of the app. `exp_harness.py`
carries the protocol in its docstring; `build_folds.py` samples the folds;
`build_fp_holdout.py` builds the crack-free false-positive set — sampling only pixels ≥256 px
inside its window, because a cropped feature stack disagrees with the full-image stack out to
exactly that distance (4.4e-2 at the edge, 1.1e-3 at 128 px, 0 at 256 px).

The owner's 71 correction masks were SHA-256 fingerprinted before the run and verified
**71/71 byte-identical** afterwards. No agent read or wrote anything under `app_data/`
except the derived SAM embedding cache.


---

# Follow-up: spatial post-processing, the one lever this sweep could not test

The sweep flagged this as its biggest blind spot, and correctly: its test rows were 120,000
pixels sampled uniformly and scattered, so no test pixel had neighbours. **You cannot
measure a neighbourhood operation on data with the neighbourhoods removed.**

Retested properly on **full-resolution probability maps** from models that never saw the
image they are scored on — four leave-one-image-out refits, predicting every pixel of each
ground-truth frame. Parameters are chosen by **nested** cross-validation: for each held-out
image the parameter is picked on the other three, so nothing is tuned on the answer sheet.

| rule | held-out IoU | vs none | folds won | crack-free FP |
|---|---|---|---|---|
| none | 0.8317 | — | — | 0.264% |
| **minimum area** | **0.8420** | **+0.0103** | **4 of 4** | **0.037%** |
| elongation filter | 0.8386 | +0.0068 | 4 of 4 | 0.199% |
| morphological closing | 0.8315 | −0.0002 | 2 of 4 | 0.267% |
| hysteresis linking | 0.8313 | −0.0005 | 1 of 4 | 0.502% |
| hysteresis + minimum area | 0.8219 | −0.0099 | 1 of 4 | 1.105% |

Two conclusions, and the second is the interesting one.

**Hysteresis linking does not work here.** It was the untested rule with the most appeal —
trust confident pixels, let them recruit weaker neighbours — and it is the *worst* of the
six, roughly doubling false positives on crack-free specimen. A weak cluster with no
confident core still grows if it is connected to itself.

**The size filter was never the harmful part of the legacy rule.** The app's existing
post-processing measured −0.084 IoU and is off by default, and it bundles a blur, a closing,
ring rejection via Euler number, an eccentricity test *and* hysteresis growth. Isolating the
pieces shows minimum-area alone is a clear win. The compound rule was hiding it.

## Choosing the threshold, using the owner's own labels

Every metric improves monotonically with area, so IoU alone would say 5000. But all four
ground-truth images are wide-open cracks — 65 px median width — and are structurally
incapable of revealing what this filter could destroy: a thin crack whose every component is
small. The legacy rule carries exactly that warning (stroke recall 0.869 raw vs 0.14–0.40
post-processed).

The owner's 30.2 M hand-drawn crack pixels across 56 images, including specimen groups with
no ground truth at all, are the only data that can answer it:

| min area | held-out IoU | crack-free FP | worst image's confirmed crack kept |
|---|---|---|---|
| none | 0.8317 | 0.264% | 100.0% |
| 1000 | 0.8371 | 0.144% | 97.3% |
| **2000 (shipped)** | **0.8391** | **0.106%** | **97.3%** |
| 5000 | 0.8420 | 0.037% | **87.6%** ← cliff |

**2000, not the top-scoring 5000.** At 5000 the worst single image loses 12.4% of the crack
its owner confirmed; 2000 costs the same 2.7% that 1000 does. Trading 0.003 IoU for a
researcher's work is not a trade worth making.

## What it does in the app

`pipeline.MIN_BLOB_PX = 2000`, applied in `effective_mask` **before** the correction layer,
which is the safety property that matters: **pruning can never remove crack you painted
yourself**, because your labels are re-applied on top of the pruned mask. `selftest.py`
asserts this, and it is the check that fires if those two lines are ever swapped.

Measured across all 71 loaded images, mean predicted area moves 8.713% → 8.552% — barely,
because real cracks are orders of magnitude too large to prune. The change lands almost
entirely where it should: on `B2_amb_mosaic_2`, a confirmed crack-free specimen, 100% of the
predicted area was specks and the frame now reads 0.000%.


## Follow-up, 2026-08-19: three things the sweep did not cover

An independent seven-dimension audit and a literature review challenged three of this
document's conclusions. All three were re-measured on full held-out probability maps
(a model refit per fold, predicting the whole frame, not sampled pixels).

**Spatial post-processing DOES help — the sweep could not see it.** The original sweep's
test rows were 120k uniformly scattered pixels, so no test pixel had neighbours and a
neighbourhood operation was unmeasurable. Under nested cross-validation on whole masks,
with the parameter chosen on the other three images and applied to the held-out one:

| rule | held-out IoU | vs none | folds won | false positives on crack-free specimen |
|---|---|---|---|---|
| none | 0.8317 | — | — | 0.264% |
| **min-area 2000 px** | **0.8391** | **+0.0074** | **4/4** | **0.106%** |
| elongation filter | 0.8386 | +0.0068 | 4/4 | 0.199% |
| closing | 0.8315 | −0.0002 | 2/4 | 0.267% |
| hysteresis (dual threshold) | 0.8313 | −0.0005 | 1/4 | 0.502% |
| hysteresis + min-area | 0.8219 | −0.0099 | 1/4 | 1.105% |

Hysteresis linking — the rule the app had never tried and the one this document guessed
might be the win — is the *worst* of the six, and it doubles false positives. Dropping
small components is the win, and it is the same move the published 3D concrete-CT work
reached for to fix "many false positives in the areas without crack".

**The min-area filter's justification was pixel-weighted, which cannot see the harm it
might do.** Per COMPONENT rather than per pixel: 98.7% of ground-truth crack objects are
below 2000 px. That sounds alarming and is not, for a reason worth recording — with **no
filter at all** the model recovers only **28.0%** of ground-truth objects (984/3517), so
those fragments are overwhelmingly never detected in the first place rather than destroyed
by the filter. The filter costs 1.9 points of object recall (67 objects, median width
**1.0 px**, 90th percentile 2.4 px) and buys +0.0074 IoU and a 2.5x reduction in false
alarms. A variant that spares small-but-elongated components was tested to protect thin
cracks specifically: it keeps 38 more objects but scores 0.8332 and 0.212% — giving up most
of both gains. The shipped rule stands.

Caveat that cannot be resolved with this data: the ground truth contains no thin cracks
(median width ~65 px), so "the lost fragments are noise" and "the lost fragments are the
thin cracks we miss" are not distinguishable here. Thin-crack labels would settle it.

**Orientation-selective features do NOT help, and this document's "features are a dead
end" was previously too broad.** The sweep only ever recombined the existing 17 features
with SAM; it never added a new feature family. Every one of the 17 is isotropic by
construction — Gaussian smooths, gradient magnitudes, Laplacians, local std — while a crack
is a thin oriented structure, and Fiji's Trainable Weka Segmentation ships Hessian,
structure tensor and ridge filters as standard. So this was a real gap in the earlier
conclusion.

Tested: 18 added features — Hessian eigenvalues and their difference at sigma 1/2/4/8,
structure-tensor coherence at the same scales, plus Frangi and Sato vesselness — appended
to both branches, on the SAME rows (verified pixel-identical by matching the raw-intensity
column and the labels). Result **0.8270 against 0.8308, −0.0038, winning 1 of 4 folds.**
No help. The same caveat applies: ridge filters should matter most at the thin widths this
ground truth does not contain, so this refutes "orientation features help here", not
"orientation features help".


## 2026-08-19: the first evaluation outside B2 — and it fails

Until now every number in this project came from four dense ground-truth images, all
specimen group B2. Sampling the owner's 30.2 M hand-drawn labels into a fold cache
(`code/build_label_folds.py`) makes leave-one-specimen-GROUP-out constructible for the
first time: 71 labelled images across AM/HC (27), B2 (17), B3 (13), wrought (14).

**Sparse labels need a different metric.** `corr == 0` means "no opinion", not "not crack".
Measured on six images, treating corrections as dense ground truth gives mean IoU 0.06,
because every crack the model found correctly and the owner never painted over counts as a
false positive; restricted to judged pixels the same images give 0.70–0.997. So this reports
**agreement on judged pixels**, never a whole-image IoU.

Train on the other three groups plus the four dense ground-truth images; test on the
held-out group's judged pixels only.

| held-out group | images | judged px | crack recall | not-crack agreement |
|---|---|---|---|---|
| **AM/HC** | 27 | 367,499 | **0.397** | 0.973 |
| B2 | 17 | 294,817 | 0.836 | 0.943 |
| B3 | 13 | 212,010 | 0.795 | 0.818 |
| wrought | 14 | 159,994 | 0.763 | 0.866 |
| mean | | | 0.698 | 0.900 |

**The model does not transfer to AM/HC.** Held out entirely it recovers 40% of the crack
the owner marked there, against 76–84% for every other group. Note the direction: AM/HC has
the *highest* not-crack agreement (0.973) and the lowest crack recall, so the failure is
UNDER-marking, not over-marking — consistent with the thin-crack misses the README figure
already shows on an AM/HC frame.

**Controlled for training-set size**, because holding out AM/HC also removes 27 of 71
images:

| held out | crack recall |
|---|---|
| the 27 AM/HC images (by group) | **0.397** |
| 27 random images, seed 0 | 0.782 |
| 27 random images, seed 1 | 0.816 |
| 27 random images, seed 2 | 0.755 |

Same number of images, same training budget, twice the recall. The collapse is the specimen
group, not the data volume.

**Caveats that matter.** These are sparse labels, so this is agreement with the owner's
judgement and not accuracy against physical truth. 98.3% of the force-crack labels sit on
pixels the model already called crack (Flip region confirms a blob in one click), which
makes crack recall partly circular — but that circularity would *inflate* recall, and AM/HC
still reads 0.397, so the failure is real and if anything understated. The not-crack side is
dominated by imported research negatives covering large background regions.

**What this changes.** The honest scope of the deployed model is B2-like material. Any claim
about AM/HC needs either dense annotation there or a model trained to transfer. This is the
measurement that was missing, and it argues the top priority is annotation in AM/HC rather
than any further work on the architecture.


## Launch-readiness fixes, measured

| what | before | after |
|---|---|---|
| retrain peak memory, assembling the training matrix | ~17.7 GB transient (three live copies) | **8.55 GB resident**, 6.77 GB footprint, matrix 4.58 GB |
| `display.png` on the 32 MP mosaic, revisited | 30,034,329 bytes, 0.39 s | **304, 0 bytes, 0.001 s** |
| sidebar labels, 71 real filenames | 26 distinct, one string repeated 27x | **71/71 distinct**, 13-45 chars |
| `DELETE /api/image/%2e%2e` | `shutil.rmtree` on all of `app_data` | **404**, all 71 images intact |
| SAM unreachable | red job error on every image | **17-feature model**, reason shown per image |
| truncated `emb.npz` | image permanently unusable | **detected, deleted, recomputed** |
| server dies mid-retrain | green "Retraining complete" | **red, "did not finish", buttons restored** |

The retrain matrix is 4,191,206 rows x 273 float32 = 4.58 GB, verified contiguous, finite,
and 0.5000 crack fraction, built in 714 s. Peak resident 8.55 GB rather than 4.58 GB because
the per-image 17-feature stacks are transient and the largest is 2.18 GB on its own; that one
is deliberately left alone, since banding a Gaussian filter bank needs a halo sized to the
largest sigma and getting it wrong changes predictions silently.
