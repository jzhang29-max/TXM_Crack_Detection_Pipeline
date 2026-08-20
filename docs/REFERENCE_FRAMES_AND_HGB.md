# Two changes: the reference frames left training, and the classifier changed

Both were measured before being made. `code/experiment_no_gt.py` reproduces everything here.

## 1. The four reference frames are the test set now

Every model in this project used to train on the four dense B2 frames in `dataset_cache/`
and then be validated against those same four. The gate was grading with part of the answer
key in the training set, so its number could be improved by fitting the evaluation set
harder — which is the one thing a gate must not reward.

They are now excluded from training entirely, **by specimen, not by image**. All four are
also loaded in the app carrying the owner's own corrections, and `b2_343_75` and
`b2_343_75_LARGE` are two fields of view of one specimen, so training on one while testing
on the other leaks. Five of 71 images are held out this way (`pipeline.REFERENCE_SPECIMENS`).

### What it cost

Leave-one-group-out over AM/HC, B2, B3 and wrought, 3 independent row samples per cell,
deployed ensemble, 150 k-row budget for every arm:

| training | AUC | recall @5% FPR | recall @10% FPR |
|---|---|---|---|
| corrections only (the change) | **0.871** | **0.478** | **0.642** |
| reference frames + corrections (before) | 0.863 | 0.452 | 0.612 |
| reference frames alone | 0.826 | 0.426 | 0.483 |
| *noise floor* | ±0.008 | ±0.041 | ±0.032 |

Cross-group cost: none. 0.871 against 0.863 is exactly the noise floor. On the B2 frames
themselves IoU falls 0.768 → 0.741, which is real at 2.5× the noise floor and is the honest
price: B2-specific training data was removed and B2 was then tested.

Two things that look like results and are not. `reference frames alone` topping the B2 test
(IoU 0.818) is home-group bias — it is B2-trained and B2-tested, and it is the worst arm
cross-group. Its cross-group recall of 0.904 at threshold 0.5 is not skill either: its FPR
at that threshold is 0.424. It simply calls more of the image crack. That is the whole reason
the matched-FPR columns exist, and why arms must not be compared at a fixed threshold when
they are calibrated differently.

### A dead end, recorded

Raising the corrections-only budget from 150 k rows to 500 k made both tests slightly worse
(cross-group AUC 0.856, B2 IoU 0.730). More labelled pixels are not what this model is short
of, so the exclusion of five images costs nothing on that axis either.

### The gate had to change too

Every model produced before this recipe trained on the reference frames, so its score on
them is in-sample. Comparing a clean candidate against an in-sample incumbent rejects the
candidate for being honest — the two numbers are not the same quantity, and no tolerance
widening fixes that. So models now carry a `recipe` tag:

- same recipe → the usual no-regression test against the incumbent
- different recipe → the comparison is void, and the candidate must clear an absolute floor
  (`MIN_ABS_IOU`), with the reason recorded in `gate_detail.iou_basis` rather than the check
  silently skipped

## 2. Single HistGradientBoosting, not a mean-probability MLP ensemble

Measured inside this project's own grouped-by-image protocol, 5 folds, **identical rows and
identical folds** for all three (1,274,320 rows over 75 image groups):

| model | IoU @0.5 | AUC | best IoU | at threshold |
|---|---|---|---|---|
| HistGradientBoosting(17+SAM) | **0.7642** | **0.9634** | **0.7798** | 0.33 |
| mean probability of MLP(17) and MLP(17+SAM) | 0.7541 | 0.9529 | 0.7645 | 0.43 |
| MLP(17+SAM) alone | 0.7398 | 0.9552 | 0.7574 | 0.20 |

And cross-group, from the leave-one-group-out run above:

| model | B2 frames AUC | cross-group AUC |
|---|---|---|
| HistGradientBoosting(17+SAM) | 0.981 | **0.897** |
| MLP(17+SAM) | 0.972 | 0.893 |
| the old ensemble | 0.977 | 0.863 |
| MLP(17) alone | 0.959 | 0.782 |

The +0.034 cross-group AUC over the ensemble is four times the ±0.008 noise floor, and
nothing is given up on B2. The cause is the 17-feature member: at 0.782 cross-group it is far
the weakest of the four, and averaging it into the ensemble drags the mean down. Dropping it
also halves inference, because the old ensemble ran both models over every band, and removes
the 5.9 GB in-place `StandardScaler` pass — a tree ensemble is invariant to feature scaling.

### A measurement error worth recording

The first comparison here looked like a regression: a live grouped-CV run with
HistGradientBoosting scored 0.759 against a **recorded** 0.815 for the old ensemble, and that
0.815 was treated as a baseline. It was not one. It had been measured on 2026-08-19 against a
smaller label corpus, so the two numbers were never comparable. Re-measured on identical rows
the ensemble scores 0.7541, below HistGradientBoosting's 0.7642. A stored number from an
earlier corpus is not a baseline for a fresh run, however tempting the direct comparison is.

### Still open: the threshold

HistGradientBoosting's best threshold on this protocol is **0.33**, not the 0.50 the app
defaults to, worth about 0.016 IoU. The default was calibrated for the old ensemble (whose
own optimum measures 0.43). Recalibrating means either changing the default sensitivity or,
better, storing a per-model calibrated threshold in the registry so each model is served at
its own operating point. Not done here.
