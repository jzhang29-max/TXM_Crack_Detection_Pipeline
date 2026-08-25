# Ridge and vesselness filters: they win by image and lose by specimen

All 17 hand-crafted features are isotropic — intensity, Gaussian smooths, gradient magnitude,
Laplacian, local std — while a crack is a curvilinear structure. Frangi, Sato and Meijering
filters are built for exactly that, so this looked like a real gap. It was tested twice.

## Grouped by image, it works

Adding ridge channels to the 17, 5 MLP seeds, `GroupKFold(5)` over images. First against the
old wide labels, then against the thin-core labels that ship now:

| arm | wide labels | thin-core labels |
|---|---|---|
| baseline_17 | 0.6624 ±0.0041 | 0.7437 ±0.0156 |
| + meijering | **+0.0159** | **+0.0136** |
| + all ridge (frangi, sato, meijering, hessian eigenvalues) | **+0.0204** | **+0.0066** |
| + 9 random columns *(control)* | −0.0040 | −0.0039 |

The control is what made this worth chasing: nine random channels are worth about −0.004, so the
ridge gain was information rather than the MLP getting extra width. Narrowing the labels had
already absorbed most of it, though — the all-ridge arm fell from +0.020 to +0.007.

## Held out by specimen, it fails

60 frames come from four specimens (AM/HC 25, B2 13, wrought 12, B3 10). Grouping by *image*
still leaves a frame's siblings in training. Holding out a whole specimen group:

| held out | baseline | + meijering | + all ridge | + noise9 *(control)* |
|---|---|---|---|---|
| AM/HC | 0.5072 | −0.0175 | −0.0188 | 0.0000 |
| B2 | 0.5676 | −0.0396 | −0.0536 | −0.0714 |
| B3 | 0.6275 | −0.0470 | −0.1022 | −0.0310 |
| wrought | 0.6729 | **−0.1740** | **−0.2304** | −0.0130 |

Not one group improves. Part of that is generic — nine columns of *noise* also cost 0.03–0.07
on the specimen axis, so simply widening the feature vector hurts on a small, specimen-clustered
training set, and on B2 the ridge arms beat junk. But on `wrought` meijering is 13× worse than
noise and all-ridge 18× worse, which noise cannot explain. Those channels carry something
specimen-specific and it does not transfer.

**Not shipped.** A feature that helps only when the same specimen appears on both sides of the
split is not a feature that helps on the next specimen, which is the only case that matters.

## The finding that outlives the experiment

**Grouped-by-image cross-validation is generous on this dataset**, and every number this project
publishes is measured that way — including v5's held-out IoU of 0.811. Same rows, same model,
two splits:

| split | IoU |
|---|---|
| grouped by image | **0.744** |
| held out by specimen | **0.507 – 0.673** |

Both are honest answers to different questions. Grouped by image answers *"another frame of a
specimen I have labelled"* — which is the normal use of this tool, tracking a crack across a load
series. Held out by specimen answers *"a specimen I have never labelled"*, and there the number
is far lower. Quote the by-image figure for the first question only, and do not let it stand in
for the second.

(The 0.744 here is the ridge harness's own baseline — 8k+8k rows per image, MLP (64,32) — not the
0.811 the retrain gate reports, which uses the deployed architecture. The gap between splits is
the point, not the absolute values.)

## Notes

- Two agent runs were lost to the machine sleeping mid-response before this was finished; the
  work was moved into a detached job that writes each stage's JSON as it completes.
- The first version of the control here appended nothing — it passed the noise array *alone*,
  measuring "noise only" and scoring 0.276. A control that extreme is a harness bug, not a
  result. Corrected, it lands at −0.0039 and reproduces the earlier run's −0.0040.
- `black_ridges=True` throughout, since cracks are dark. Frangi's auto-gamma varies 14.3× across
  these frames, so a fixed-gamma arm was included to separate the filter from its adaptivity.

Scripts and raw results: `research/ridge/`.
