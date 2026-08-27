# Physics-constrained self-registration and a label-free consistency metric

A result found by testing an idea rather than assuming it. Written up because it is, on current
evidence, the most defensible novel contribution this project has.

## The idea

Fatigue damage is irreversible. A pixel that is crack at N cycles cannot be sound at N+1. So for
any two frames of the same region, the earlier mask must be **contained** in the later one.

Every published crack-segmentation method for X-ray tomography segments frames independently
(XCT-SAM, Material-SAM, Segment Any Crack, and the 2026 in-situ evolution papers all analyse
evolution *after* per-frame segmentation). None uses irreversibility as a constraint. That gives
three things at once, and all of them are orthogonal to whether the encoder is frozen or
LoRA-tuned:

1. **A registration method** for data with no usable texture.
2. **A label-free error metric** — the residual violation after optimal alignment.
3. **A bad-frame detector** — pairs that cannot be reconciled at all.

## Why registration is the whole problem, and how the constraint solves it

Two standard methods were tried and both failed on this imagery:

| method | result |
|---|---|
| phase correlation on gradient magnitude, 8x downsampled | peak-to-second-peak ratio **1.00-1.02** on every pair: no peak at all |
| ORB keypoints + RANSAC | **0-3 matches** per pair |

The ORB failure had an obvious cause once checked: ORB finds only **32 keypoints in a 22 MP
frame**. This material is nearly textureless at this scale. A positive control confirmed the
matcher itself works — a frame against a shifted copy of itself gives 22 matches of 32, and
against a 90%-overlap crop 23 of 32 — so the between-frame collapse is real, not a bug.

An edge-based registration (locate the specimen boundary per row) gave shifts with an
interquartile spread of 36-80 px. A crack is 5-25 px wide, so that is useless: registration
accuracy must be **finer than the feature**.

The constraint supplies its own registration. Search the shift that maximises containment of the
earlier mask in the later one. **The crack is its own fiducial.** No texture, no fiducial
markers, no digital volume correlation.

## Measured

Coarse-to-fine search, +/-512 px, on the two series where frames image the same region.

    wrought 316L, >=800 cycles, constant 22 MP field
      pair            shift (dy,dx)     unregistered   registered   residual
      900->1000        (-4, -470)            67.2%        79.2%       20.8%
      1000->1100       (182, -100)           12.8%        97.3%        2.7%

    HC 316L, main views, >=1300 cycles
      1300->1350       (-12, 52)             72.4%        79.2%       20.8%
      1350->1400       (342, 592)             3.0%        73.2%       26.8%   at search edge
      1450->1500       (-152, 8)             15.6%        88.0%       12.0%
      1600->1650       (52, 46)              42.2%        84.9%       15.1%
      1650->1750       (-592, -48)            0.0%        26.1%       73.9%   at search edge
      1750->1790       (480, -100)            6.8%        71.3%       28.7%

The best-registered pair reaches **97.3% containment, a 2.7% residual** — the segmenter agreeing
with physics to within three percent, established with **no ground truth**. Unregistered, the
same pair reads 12.8%, so essentially all of the apparent violation was misalignment.

## Outright violations, and why they are the point

Three pairs show the later crack SMALLER than the earlier one, which no alignment can fix:

    wrought  800->900    39,313 -> 36,742 px   (-6.5%)
    HC      1400->1450  328,293 -> 320,517 px  (-2.4%)
    HC      1500->1600  388,236 -> 387,869 px  (-0.1%)

These are physically impossible for the same region, so each is a measurable segmentation or
acquisition error. Their size (0.1-6.5%) is a **calibration of the segmenter's frame-to-frame
area error**, again with no labels.

And the method independently flagged the frames that a separate per-set analysis had already
flagged from unrelated evidence (region orientation showing a different feature in the HC
900-1250 block). Two methods, no shared assumptions, same suspect frames.

## Honest limits

- Demonstrated cleanly on **one pair** (97.3%). Six more land at 71-88%, two are still
  unconverged at the +/-512 px search edge.
- Translation only. No rotation, no scale, no non-rigid deformation.
- The residual mixes registration error with segmentation error and this analysis does not
  separate them. That separation is the obvious next experiment: apply a known synthetic shift
  to a single frame and measure the residual floor.
- 2D exported slices, not reconstructed volumes. The field registers time-resolved tomography
  with digital volume correlation on volumes, which is strictly better where volumes exist.
- Only 4 and 9 usable frames in the two series after excluding tip and ZOOM sub-views.

## What would settle it

1. The **residual floor** under a known shift, to separate registration from segmentation error.
2. Rotation and scale in the search, then a non-rigid stage.
3. Run it on the reconstructed volumes rather than exported slices.
4. Record stage coordinates at acquisition. The method does not need them, but they would bound
   the search and make the two unconverged pairs converge.
