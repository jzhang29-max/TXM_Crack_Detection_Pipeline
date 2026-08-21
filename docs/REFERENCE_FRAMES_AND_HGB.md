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

## 2. HistGradientBoosting looked better on every labelled metric, and was reverted

It was tried, deployed, caught by the gate, and rolled back. The sequence is the point of
this section, so it is recorded in the order it happened rather than tidied up.

**What was measured first.** Inside this project's own grouped-by-image protocol, 5 folds,
identical rows and identical folds for all three (1,274,320 rows over 75 image groups):

| model | IoU @0.5 | AUC | best IoU | at threshold |
|---|---|---|---|---|
| HistGradientBoosting(17+SAM) | **0.7642** | **0.9634** | **0.7798** | 0.33 |
| mean probability of MLP(17) and MLP(17+SAM) | 0.7541 | 0.9529 | 0.7645 | 0.43 |
| MLP(17+SAM) alone | 0.7398 | 0.9552 | 0.7574 | 0.20 |

And cross-group, leave-one-group-out over AM/HC, B2, B3 and wrought, 3 repeats:

| model | B2 frames AUC | cross-group AUC |
|---|---|---|
| HistGradientBoosting(17+SAM) | 0.981 | **0.897** |
| MLP(17+SAM) | 0.972 | 0.893 |
| the old ensemble | 0.977 | 0.863 |
| MLP(17) alone | 0.959 | 0.782 |

+0.034 cross-group AUC over the ensemble, four times the ±0.008 noise floor, giving up
nothing on B2. On that basis it was deployed.

**What the gate then measured.** Predicted area on the six specimens confirmed to contain no
crack rose from **0.26% to 1.98%**, all six worse, worst 0.41% → 3.49%. The retrain was
refused and the model kept for inspection.

**Attribution.** Four arms at a fixed 400,000-row budget, crossing {reference frames in, out}
with {HGB, the ensemble}, each scored on both gate axes (`research/fp_attribution.json`):

| arm | reference IoU | crack-free FP |
|---|---|---|
| no-GT + HGB | 0.748 | 2.141% |
| no-GT + MLP ensemble | 0.714 | **0.451%** |
| with-GT + HGB | 0.869 \* | 2.540% |
| with-GT + MLP ensemble (old recipe) | 0.921 \* | 0.544% |

\* contaminated: those arms train on the frames they are scored on.

So the classifier owns it, not the training composition: HGB is ~4.7x worse on crack-free
material in **both** compositions, and removing the reference frames slightly *lowers* FP
(0.544% → 0.451%). The classifier was reverted to the ensemble; the reference-frame change
was kept.

### Why every metric that favoured HGB was blind to this

All of them — IoU, AUC, matched-FPR recall, cross-group AUC — are computed over **labelled**
pixels. Crack-free specimen is exactly the material nobody labels: there is nothing to find,
so there is nothing to mark. HGB separates the labelled distribution better and behaves far
worse off it, and no amount of AUC over labels can see that.

This is the argument for keeping a false-positive axis measured on unlabelled material known
to contain nothing. It is not a redundant second opinion on the same evidence; it is the only
axis reading material the training distribution does not cover. It caught a 22.4% model
before, and it caught this one.

### A measurement error worth recording

The first comparison looked like a regression for the wrong reason: a live grouped-CV run
with HGB scored 0.759 against a **recorded** 0.815 for the old ensemble, and that 0.815 was
treated as a baseline. It was not one — it had been measured on 2026-08-19 against a smaller
label corpus. Re-measured on identical rows the ensemble scores 0.7541, *below* the 0.7589 it
had just rejected. Both gate axes are now recipe-aware for this reason; a stored number from
another architecture or another corpus is not a baseline.

### Still open: the threshold

The ensemble's own best threshold on this protocol measures **0.43**, not the 0.50 the app
serves. Worth ~0.010 IoU. The clean fix is a per-model calibrated threshold stored in the
registry, so each model is served at its own operating point rather than a shared constant.
Not done here.
