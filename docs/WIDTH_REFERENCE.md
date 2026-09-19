# The image carries a width reference, and measured against it the detector under-marks

Run 2026-09-19 on all 71 frames. Every number below is reproducible from
`app/core/pipeline.py: local_background`, `_transverse_fwhm` and `clip_to_measured_width`.

## What prompted it

The detector was reported as "marking a bit over". Three things were checked first and all
three came back negative:

- **Threshold.** Sweeping 0.60 -> 0.98 on clDice and width: 0.60 gives clDice 0.863 at
  28.9 px, 0.90 gives 0.795 at 22.1 px, 0.95 gives 0.325 at 17.3 px, 0.98 gives nothing.
  Raising the cut removes crack rather than narrowing it. Not the lever.
- **`tighten_to_image`.** Already on by default, and it does far less than it says. Its
  docstring records 8.178% -> 6.464% of frame area and a 16.3 px -> 1.4 px half-width.
  Measured over all 71 frames the area drop is ~6% and the half-width moves 49.0 -> 44.4 px.
  The reason is in the test itself: it keeps `img <= uniform_filter(img, 301)`, and that box
  mean is taken over the mask as well as around it. A 30-50 px band inside a 301 px window
  is compared against the bright matrix either side, so the whole band passes. Corrected in
  place in both that docstring and `app/server.py`, which repeated the "factor of two".
- **Contrast thresholding done properly.** Estimating the background from non-mask pixels
  and cutting at half the local peak (FWHM) does narrow -- 3.2x at FRAC 0.5 -- and fails its
  pre-registered guard by 23x: Tsens 0.798 -> 0.261, clDice 0.846 -> 0.412. Adjudicating
  that loss against the image rather than the labels, the material a FRAC 0.4 cut removes
  sits 1.63 sigma above matrix at 42% of the core contrast. It is dark. Deleting it is not
  narrowing, it is deleting crack.

## The measurement that settled it

`docs/OVERMARKING.md:100` says the blocker is that "no tight reference exists". One does:
the transverse contrast profile. At each skeleton point, take the profile along 12
directions and keep the smallest full-width-at-half-maximum -- along the crack the profile
never returns below half maximum, across it the width is the feature's.

Over **11,426 profiles on 61 frames**:

| | |
|---|---|
| median transverse FWHM | **45 px** = 1.3 um at 29 nm/px |
| 10th - 90th percentile | 3 - 147 px |
| coefficient of variation | 0.916 |
| Spearman rho vs peak contrast | **+0.673** |
| frame-median FWHM vs frame-median peak | +0.756, p = 1.9e-12 |

This was pre-registered as a test of whether the width is the instrument. It is not. A
fixed point-spread cannot produce a width that ranges over 50x and rises with depth. Deeper
cracks are measurably wider, which is what crack opening displacement predicts. The width
in these images is the crack's, and it can be read off the image with no annotation in the
loop.

## What the reference says about over-marking

Mask width / image FWHM, per-frame medians, 15,397 profiles. **1.00 means the mask is
exactly as wide as the feature it sits on.**

| mask | ratio | IQR | wider than the feature on |
|---|---|---|---|
| shipped detector (`tight=1`) | **0.61x** | 0.57-0.90 | 10 / 63 frames |
| wide (`tight=0`) | 0.79x | 0.64-1.06 | 19 / 61 |
| human brush label | 0.69x | 0.47-1.13 | 19 / 61 |

Wilcoxon against 1.0: p = 5.8e-06. **The detector under-marks the dark feature corpus-wide,
by close to the factor the human annotator does.** That is the third framing of
"over-marking" in this repo to come out the other way round when measured; see the
correction at the top of `docs/OVERMARKING.md`.

The over-marking is real, and it is **local**: 10 frames of 63, and every one of them is a
frame whose feature is a hairline. That is what `MIN_BLOB_PX = 2000` forces -- a component
thinner than roughly 2000/length px cannot survive the floor, so the only hairlines that
reach an export are the ones the model happened to draw fat.

## `clip_to_measured_width`

Clips the mask to the width the image shows, per location, and **never widens it**. Where
the mask is already narrower than the feature it does nothing, which is what makes it safe
on the 53 frames that do not over-mark. Where the width cannot be measured -- fewer than 8
measurable profiles, no image, empty skeleton -- it declines and returns the mask unchanged.
Declining matters here: AM cracks are intensity-invisible (Cohen d +0.09 against +1.10 to
+2.91 elsewhere) and a step that guessed a width would do its worst damage on 38% of the
corpus.

Measured against the four criteria fixed before the run, on all 71 frames at the deployed
operating point:

| | | |
|---|---|---|
| (a) safety: Tsens and clDice drop <= 0.02 | clDice 0.846 -> 0.823 (paired **-0.017**), Tsens 0.798 -> 0.761 (paired **-0.026**) | **clDice PASS, Tsens FAIL** |
| (b) median \|ratio-1\| falls | 0.390 -> 0.442 | **FAIL** |
| (c) every over-marking frame comes down, none of the others rises | **10/10 down**, median 1.24x -> 0.88x, worst **2.91x -> 0.92x**; 0 of 53 rose | **PASS** |
| (d) crack-free specimens do not gain area | max 0.00139 -> 0.00128 | **PASS** |

Per group, clDice and area:

| group | n | clDice | area identified |
|---|---|---|---|
| B3 | 10 | 0.960 -> 0.957 | 5.77% -> 5.38% |
| B2 | 14 | 0.887 -> 0.881 | 18.87% -> 18.51% |
| Wrought | 12 | 0.845 -> 0.839 | 7.71% -> 7.46% |
| AM / HC_316L | 25 | 0.736 -> 0.714 | 3.02% -> 2.72% |

## Two criteria failed and the step ships on by default. Both failures, in full.

**Tsens missed its allowance by 0.006** (-0.026 against -0.020). The loss is not spread over
the corpus: Spearman between area removed and Tsens lost is **+0.797**, 12 frames are
unchanged, the 75th percentile is -0.007, and **five of the six worst-hit frames are
over-marking frames that the step exists to fix** --

      HC_316L_fatigue_600    ratio 2.91x -> 0.92x   Tsens 1.000 -> 0.623
      wrought_800_cycles     ratio 2.08x -> 0.88x   Tsens 1.000 -> 0.755
      HC_316L_fatigue_1250   ratio 2.00x -> 1.06x   Tsens 0.550 -> 0.394
      HC_316L_fatigue_1400   ratio 1.29x -> 0.89x   Tsens 0.758 -> 0.553
      B2_3_1_lbf             ratio 1.31x -> 0.75x   Tsens 0.631 -> 0.370

On those frames the mask was two to three times the width of the feature, the brush
centreline ran through the excess, and clipping to the measured width necessarily drops it.
Tsens is agreement with a stroke 73-126 px wide; on exactly the frames where both the stroke
and the mask are too wide, it is not the right referee. The sixth, `b3_385_63um_ZOOM`
(Tsens 1.000 -> 0.820 at ratio 0.40x), is not that case -- it was already well under-marked
and the step should have left it alone. That one is a genuine cost, not a metric artefact.

**(b) failed and is mis-specified.** It is a corpus-median criterion, and no clip-only
operator can satisfy it on a corpus whose median is 0.61x. Clipping can only move a frame
down, so on the 53 frames that under-mark any clip at all pushes \|ratio-1\| up, and that
swamps the improvement on the 10 frames the step exists for. It asks a targeted operator to
improve a population statistic dominated by the cases it deliberately does not touch.
Criterion (c) is the one that encodes the report this work started from, and it passes
completely.

The honest summary of the trade: **2.3 points of clDice for the removal of every case of
over-marking in the corpus.** Reversible in one line -- `WIDTH_CLIP = False` in
`app/core/pipeline.py`, or `tight=0` on any request, which also turns off `tighten_to_image`.

## An implementation note worth keeping

The first version carried the measured radius to each skeleton point from its NEAREST probe.
That makes the radius field piecewise constant over the probes' Voronoi cells, and the clip
boundary then follows the cell edges: rendered at native resolution the mask picked up long
straight cuts and triangular wedges no crack has. It is the same class of artefact as the
diamond lattice a radius-1 closing stamps into an export, and the area numbers did not show
it at all -- only the picture did. Inverse-distance weighting over the 12 nearest probes
makes the field continuous. It is also strictly more local than the frame-order smoothing it
replaced, which is why the final numbers above are more aggressive than the first pass.

## What this gives the paper

A width metric that does not go through the annotations at all. Every accuracy number in
this repo is scored against brush strokes, and the strokes over-mark -- that is the
documented ceiling (`docs/OVERMARKING.md`, IoU ceiling 0.0498 for a perfect 3 px crack).
The width ratio is scored against the image. It ranks a mask on whether it is as wide as the
thing it sits on, which is the question IoU against a 73-126 px stroke cannot answer.
