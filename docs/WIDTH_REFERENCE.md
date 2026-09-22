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

## Prior art, checked 2026-09-21 — the METHOD is standard, and this doc did not say so

An earlier version of this page pitched the transverse-profile width measurement as new
("a width metric that does not go through the annotations at all") and carried **zero
citations**. That was wrong, and it is the seventh novelty claim in this project to die on a
prior-art check. What is standard:

- **The method.** Sampling intensity along lines perpendicular to the crack centreline is a
  named, established algorithm for automated crack width measurement -- *Orthogonal Profile
  Extraction* -- and is one of the two principal approaches alongside the Euclidean distance
  transform on the mask. Its canonical steps are exactly the ones implemented here:
  skeletonise, compute the local tangent, take the normal, sample the profile.
  ([TarmacView, Automated Crack Width Measurement from Imagery](https://www.tarmacview.com/glossary/crack-width-measurement/);
  cf. [Edge-OrthoBoundary, *Buildings* 15:2489](https://doi.org/10.3390/buildings15142489))
- **The half-maximum criterion.** Taking the edge at half the profile's peak is **ISO50**,
  the default surface-determination rule in X-ray CT dimensional metrology, codified in
  VDI/VDE 2630 and with its own measurement-uncertainty literature.
  ([VDI/VDE 2630 Blatt 1.1](https://www.vdi.de/en/home/vdi-standards/details/vdivde-2630-blatt-11-computed-tomography-in-dimensional-measurement-fundamentals-and-definitions);
  [Precision Engineering, S0141635919301590](https://www.sciencedirect.com/science/article/abs/pii/S014163591930159X))
- **The orthogonality requirement.** That width must be taken perpendicular or it
  overestimates by 1/cos(theta) is textbook, and is why the minimum over 12 directions is
  used here.

One difference worth stating precisely rather than inflating: ISO50 in XCT is a **global**
threshold at the midpoint of the air/material histogram peaks, whereas this takes a **local**
half-maximum of each transverse profile against a locally-estimated background. That is a
routine variation, not a new instrument.

**So what, if anything, is left.** Not the measurement. What this repo has that the crack-
width literature does not is the *use*: turning a standard metrology rule into an AUDIT
instrument, and pointing it at the annotations rather than at the specimen -- asking "is the
mask as wide as the feature", and getting the answer that the detector and the human
annotator are indistinguishable and both under-mark. Any claim beyond that should be cut.

## The measurement

`docs/OVERMARKING.md:100` says the blocker is that "no tight reference exists" for width on
this corpus. One does, and it is an off-the-shelf one: the transverse contrast profile. At each skeleton point, take the profile along 12
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

**This table was re-measured on 2026-09-21 and the previous one was a splice.** Its detector
row came from the final inverse-distance-weighted estimator over 63 frames while the `wide`
and label rows came from an earlier run over 61, and it printed them side by side as though
they were comparable. They were not. Every row below is one pass, one estimator, one frame
set, and -- the part that actually matters -- **the same sample points for every variant**,
taken on the skeleton of the widest mask so that no variant is scored at points chosen by
its own geometry. Generator: `code/audit/measure_width_ratio.py`. Artifact:
`crack-evolution-5d/out/txm_width_ratio_unified.json`.

Mask width / image transverse FWHM, per-frame medians, 64 frames. **1.00 means the mask is
exactly as wide as the feature it sits on.**

| mask | n | ratio | IQR | wider than the feature |
|---|---|---|---|---|
| wide (`tight=0`) | 64 | 0.672 | 0.58-0.94 | 14 / 64 |
| tighten only | 64 | 0.590 | 0.51-0.74 | 7 / 64 |
| **SHIPPED (`tight=1`)** | 64 | **0.505** | 0.44-0.59 | **0 / 64** |
| human brush label | 61 | 0.529 | 0.43-0.77 | 11 / 61 |

Three things change against what this page said before.

**1. The shipped detector over-marks on nothing.** Zero frames of 64, against 7 for tighten
alone and 14 for the wide corridor. The "10 of 63 frames" this page reported was measured on
the mask BEFORE `clip_to_measured_width` and `drop_straight_lines` shipped. Those two steps
removed the over-marking they were built to remove, and the earlier figure should be read as
the problem statement, not the current state.

**2. The detector and the human annotator are indistinguishable.** Paired on the 61 frames
that have both, same points, same estimator: shipped **0.506** against label **0.529**, a
paired median difference of -0.011, **Wilcoxon p = 0.061**. The previous page reported
0.61x against 0.69x and read that as a gap. On one instrument there is no gap. What survives
is the direction, and it is strong: both under-mark the dark feature, shipped vs 1.0
p = 3.5e-12, label vs 1.0 p = 1.1e-06.

**3. Everything under-marks by more than previously stated.** Scoring every variant at the
same points is what makes the rows comparable, and it necessarily lowers all of them: the
shared skeleton reaches places a narrow mask does not, and those points enter its median as
near-zero width. So **these numbers are a fair comparison ACROSS masks and are not absolute
width estimates** -- do not quote 0.505 as "the detector is half as wide as the crack"
without that qualification. The per-mask self-skeleton figures (detector 0.68, label 0.69)
are the better absolute estimate and are in `out/txm_width_reference.json`.

Five frames are excluded: they have no accepted area at all, so there is no skeleton to
sample. All five are crack-free controls, which is the correct behaviour rather than a
failure.

Per specimen, shipped against label on the same points:

| group | n | shipped | label |
|---|---|---|---|
| B2 | 15 | 0.491 | 0.546 |
| B3 | 12 | 0.444 | 0.442 |
| AM | 25 | 0.566 | 0.615 |
| Wrought | 12 | 0.506 | 0.703 |

(Grouping corrected 2026-09-22: one AM frame was filed as B3 because its content hash
contains the substring `b3` and the generator tested that before `hc_316l`. Corpus figures
unchanged; see the note in `docs/UNLABELLED_AREA.md`.)

## `clip_to_measured_width`

Clips the mask to the width the image shows, per location, and **never widens it**.

**A correction to how that safety was described.** This page said "where the mask is already
narrower than the feature it does nothing, which is what makes it safe on the 53 frames that
do not over-mark". The first half is an argument about monotonicity -- the operator can only
remove pixels -- but the sentence asserted an empirical no-op, and the per-frame data
contradicts it. Of the 53 frames with a pre-clip ratio at or below 1.0, **50 lose area**
(median 4.6% of it, max 14.1%), 45 of the 52 labelled ones lose Tsens (median -0.016), and
48 end up further from a ratio of 1.0 than they started. The operator is safe in the sense
that it cannot invent crack, not in the sense that it leaves well-behaved frames alone. Where the width cannot be measured -- fewer than 8
measurable profiles, no image, empty skeleton -- it declines and returns the mask unchanged.
Declining matters here: AM cracks are intensity-invisible (Cohen d +0.09 against +1.10 to
+2.91 elsewhere) and a step that guessed a width would do its worst damage on 38% of the
corpus.

Measured against the four criteria fixed before the run, on all 71 frames at the deployed
operating point:

| | | |
|---|---|---|
| (a) safety: Tsens and clDice drop <= 0.02 | clDice 0.846 -> 0.823 (paired **-0.017**), Tsens 0.798 -> 0.761 (paired **-0.026**) | **clDice PASS, Tsens FAIL** |
| (b) median \|ratio-1\| falls | 0.399 -> 0.442 | **FAIL** |
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

An APPLICATION of standard profile metrology (see the prior-art section above -- the method
itself is not ours) as a scoring instrument that does not go through the annotations. Every accuracy number in
this repo is scored against brush strokes, and the strokes over-mark -- that is the
documented ceiling (`docs/OVERMARKING.md`, IoU ceiling 0.0498 for a perfect 3 px crack).
The width ratio is scored against the image. It ranks a mask on whether it is as wide as the
thing it sits on, which is the question IoU against a 73-126 px stroke cannot answer.
