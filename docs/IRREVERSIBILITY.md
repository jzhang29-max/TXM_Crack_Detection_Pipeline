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

---

## CORRECTION: the synthetic control overturned the headline number

Everything above was measured on real data, where the true alignment is unknown. A synthetic
control -- known shift, known growth -- was then run, and it changes the conclusion.

    case                          true shift     found        containment
    translation only              (0, 0)         (0, 0)          100.0%
    translation only              (12, -30)      (12, -30)       100.0%
    translation only              (-120, 240)    (-128, -38)      38.1%
    translation only              (300, -500)    (316, -30)       19.7%
    translation + growth          (0, 0)         (0, 0)          100.0%
    translation + growth          (40, -80)      (44, 120)        43.9%
    a DIFFERENT crack entirely    n/a            (-148, 174)      46.0%

Two things follow.

**The objective is degenerate along the crack.** Small shifts are recovered exactly; large ones
are not, and the failures are not random. A roughly linear crack slid along its own axis still
overlaps itself, so containment is nearly flat in that direction. The method registers ACROSS a
crack and cannot register ALONG it. This is intrinsic to the objective, not a bug in the search.

**So the 97.3% figure is an upper bound, not a measurement.** A large along-axis registration
error would score just as well as a correct alignment. The real-data test could not have
revealed this -- it has no ground truth to check against -- and I reported 97.3% as evidence of
physical consistency before running the control. That was wrong.

**And the metric does not cleanly discriminate.** An entirely different synthetic crack still
reaches 46% containment, so no threshold on this metric is currently defensible.

## CORRECTION: monotone repair does not improve the masks

Measured against the operator's own corrections across 11 consecutive pairs:

    series             mean recall change     leak into painted NOT-crack
    wrought >=800cyc      +0.00 pp             0.000% -> 0.000-0.151%
    HC >=1300cyc          +1.98 pp             0.000% -> 0.178-5.823%

Repair added 29,000-363,000 pixels per frame. On the wrought series it changed recall by
exactly zero. On HC it bought 2 points of recall while putting up to 5.8% of explicitly
marked not-crack material into the crack mask. At this registration accuracy the constraint is
a diagnostic, not a correction.

## What survives

- The **area-decrease detector** survives untouched: three pairs show the later crack smaller
  than the earlier one (-6.5%, -2.4%, -0.1%), which no alignment can explain and which needs no
  registration to detect. That is a genuine label-free error signal.
- The **observation** that no published method uses irreversibility survives.
- The **registration difficulty** is real and documented: phase correlation gives no peak, ORB
  gives 0-3 matches on 32 keypoints in 22 MP, edge tracking gives 36-80 px scatter.

## What would make the constraint usable

1. ~~Break the along-axis degeneracy~~ — **done, see the next section.** Mostly solved on a
   synthetic control; it lowered the real-data numbers rather than rescuing them.
2. Register on the IMAGE with the crack masked out, using the constraint only to validate.
3. Recorded stage coordinates. The whole difficulty disappears if the shift is known.

## The degeneracy is broken by an anchor, and the anchor makes the results worse

`register_anchored()` in `app/core/sequence.py`. The physics: a fatigue crack has a **fixed
root**. It initiates at a free surface and grows inward, so the mouth where it meets the
specimen boundary does not move between load steps while the tip does. A point anchor constrains
both axes where a self-overlap objective constrains only one.

It needs **two different features**, one per axis, and that is the non-obvious part:

| axis | taken from | why not the other feature |
|---|---|---|
| across-crack (here *y*) | the crack mouth's centroid | a genuine crack property, fixed by the physics |
| along-crack (here *x*) | median leading edge of the specimen | the mouth's along-edge coordinate is merely wherever the boundary happens to be, so it carries no crack information |

Taking both axes from the mouth was tried first and fixed only one of them — *dy* came back −112
against a true −120, while *dx* came back 530 against a true 240. That failure is what forced
the split.

### Synthetic control, rebuilt with a specimen edge and a root fixed by construction

Earlier frame: crack length 600 px. Later frame: 800 px of real growth, plus a known shift.

| true shift | containment only | anchored | |
|---|---|---|---|
| (0, 0) | (0, 0) | **(0, 0)** | exact, 100% containment |
| (12, −30) | (12, −30) | **(12, −30)** | exact, 100% containment |
| (−120, 240) | (−124, 152) — wrong | **(−120, 240)** | exact, 100% containment |
| (−40, 620) | (−48, 338) — wrong | **(−40, 620)** | exact |
| (300, −500) | (316, −16) — wrong | (309, −152) | **still wrong** |

Four of five, including both cases containment alone got wrong. The one remaining failure is
understood rather than mysterious: a −500 px shift carries the specimen boundary off the frame,
so the along-edge reference does not exist to be measured.

The **negative control improves too** — an entirely different synthetic crack, root at a
different height, falls from 46% containment to 23.5%. The metric now separates consistent from
inconsistent pairs far better than it did. No threshold on it has been validated against enough
pairs to be defensible, though.

### And on the real data it lowers the headline

Over the 13 consecutive pairs in the two usable series:

- **3 of 13 pairs have the later crack smaller than the earlier one.** No registration rescues
  that; it is an outright violation, detected with no labels. Unchanged by the anchor.
- On the **wide-FOV wrought frames the mouth is not locatable** and the method falls back to
  containment only. The anchor is not available everywhere.
- Where the anchor does fire it agrees with containment-only on 4 of 6 HC pairs and **cuts the
  other two hard: 78.7% → 46.8%, and 71.3% → 25.5%**.

That last line is the result. The anchored number is *lower* because containment-only had been
finding along-axis shifts that scored well and were wrong. **So 97.3% stays an upper bound, and
the properly constrained estimate of sequence consistency in this data is materially worse than
the unconstrained one.** The anchor did not rescue the claim — it measured how much the claim
was inflated. `monotone_repair()` remains something not to use for producing masks.

One bug worth recording because it silenced the method completely: the first `specimen_mask()`
thresholded on brightness, and **the crack is dark**, so a plain threshold carved the crack out
of the specimen support — and the mouth, which lies exactly on the boundary, fell outside it.
Every frame returned `None` and every pair silently fell back to containment-only. Closing and
hole-filling the support fixed it. A fallback that reports which path it took is what made this
visible at all.
