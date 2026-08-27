"""Sequence consistency for time-lapse crack imaging: irreversibility as a constraint.

WHAT THIS IS FOR. Fatigue damage does not heal. For two frames of the same region at increasing
load cycles, the earlier crack mask must be CONTAINED in the later one. Every crack-segmentation
method for X-ray tomography the author is aware of segments frames independently, so none of
them can use this. It gives three things a single-frame method cannot have:

  register_by_containment()  registration for imagery with no usable texture
  pair_consistency()         a label-free error signal -- the residual violation after
                             optimal alignment
  monotone_repair()          the constraint applied as a correction, not just a check

WHY REGISTRATION IS THE WHOLE DIFFICULTY, and why the usual tools fail here. Measured on this
project's wrought 316L series:

  phase correlation on gradient magnitude   peak-to-second-peak ratio 1.00-1.02: no peak at all
  ORB keypoints + RANSAC                    0-3 matches per pair
  specimen-edge tracking                    36-80 px interquartile scatter

ORB finds 32 keypoints in a 22 megapixel frame -- this material is nearly textureless at this
scale. A positive control confirmed the matcher works (22 of 32 against a shifted copy of the
same frame, 23 of 32 against a 90% overlap crop), so the between-frame collapse is a property
of the data. And 36-80 px scatter is useless for a feature 5-25 px wide: registration has to be
finer than the thing being registered.

The constraint supplies its own registration. Search the shift that maximises the containment
physics requires -- the crack is its own fiducial. Measured on the best-registered pair:
containment 12.8% unregistered, 97.3% registered, so nearly all of the apparent violation was
misalignment rather than segmentation error.

THE OBJECTIVE IS DEGENERATE ALONG THE CRACK, and that is the finding that matters. A synthetic
control -- known shift, known growth -- recovers small shifts exactly ((0,0) and (12,-30) both
found exactly, 100% containment) and FAILS on large ones: true (-120, 240) came back as
(-128, -38), true (300, -500) as (316, -30). The cause is not a coding error. A roughly linear
crack slid ALONG its own axis still overlaps itself, so containment is nearly flat in that
direction. The method can register ACROSS a crack and cannot register ALONG it.

That degeneracy invalidates the headline real-data number. The 97.3% containment measured on the
wrought 1000->1100 pair cannot be read as evidence that the segmenter agrees with physics,
because a large along-axis error would score just as well. It has to be treated as an UPPER
BOUND on agreement, not a measurement of it. The real-data test could never have revealed this;
only the synthetic control could, which is the argument for always having one.

Also: a deliberately inconsistent pair -- an entirely DIFFERENT synthetic crack -- still reaches
46% containment. So the metric does not cleanly separate consistent from inconsistent pairs at
this stage, and no threshold on it is currently defensible.

WHAT THIS MODULE DOES NOT DO. Translation only: no rotation, no scale, no non-rigid warp. It
operates on 2D masks, where the field's proper tool for time-resolved tomography is digital
volume correlation on reconstructed volumes. The residual MIXES registration and segmentation
error and does not separate them.

AND MONOTONE REPAIR DOES NOT IMPROVE THE MASKS. Measured against the operator's own labels over
11 consecutive pairs: mean recall change +0.00 pp on the wrought series and +1.98 pp on HC, while
leak into material the operator explicitly marked NOT crack rose from 0.000% to as much as
5.823%. It adds hundreds of thousands of pixels for almost no recall and a real precision cost.
Use this module as a diagnostic. Do not use monotone_repair to produce masks.
"""
import numpy as np

# Search half-width in pixels for the coarse stage. The observed true shifts in this project's
# series reach 592 px, and two pairs were still unconverged at +/-512, so this is deliberately
# generous. Cost is quadratic in the range but the coarse stage runs on a 16x decimation.
COARSE_RANGE = 768
COARSE_STEP = 16
REFINE = ((4, 64), (2, 16))          # (decimation, half-width) for each refinement stage


def _crop_common(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def _shift_into(mask, dy, dx, shape):
    """Move `mask` by (-dy, -dx) onto a canvas of `shape`, plus the validity region.

    NOT np.roll. Rolling WRAPS, which does two damaging things: a shift aliases modulo the
    image size, so a true (0, 0) can be reported as (-700, 0) on a 700-row frame; and crack
    wrapped around from the opposite edge scores as agreement. Both were found by the
    synthetic control in docs/IRREVERSIBILITY.md, which is the whole reason that control
    exists. Everything here is scored inside the validity region only.
    """
    h, w = shape
    out = np.zeros((h, w), bool)
    valid = np.zeros((h, w), bool)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = mask[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        valid[ys0:ys1, xs0:xs1] = True
    return out, valid


def register_by_containment(earlier, later, coarse_range=COARSE_RANGE):
    """Shift (dy, dx) applied to `later` that maximises containment of `earlier` within it.

    Coarse-to-fine so the generous search range stays affordable: the coarse pass runs on a
    16x decimation, then two refinements at 4x and 2x around the current best.

    Returns (dy, dx, containment, containment_unregistered).
    """
    a, b = _crop_common(np.asarray(earlier, bool), np.asarray(later, bool))
    if not a.any():
        return 0, 0, 1.0, 1.0                     # nothing to contain: trivially satisfied

    best_y = best_x = 0
    for ds, half in ((COARSE_STEP, coarse_range),) + REFINE:
        ad = a[::ds, ::ds]
        bd = b[::ds, ::ds]
        tot = int(ad.sum())
        if tot == 0:
            break
        best_v = -1
        cy, cx = best_y, best_x
        for dy in range(cy - half, cy + half + 1, ds):
            for dx in range(cx - half, cx + half + 1, ds):
                shifted, valid = _shift_into(bd, dy // ds, dx // ds, ad.shape)
                # score only where the shifted frame HAS data, and against only the part of
                # the earlier crack that lies there -- otherwise a shift that pushes most of
                # the frame out of view scores well on the sliver that remains
                v = int((ad & shifted).sum()) - int((ad & ~valid).sum())
                if v > best_v:
                    best_v, best_y, best_x = v, dy, dx

    ds = REFINE[-1][0]
    ad, bd = a[::ds, ::ds], b[::ds, ::ds]
    tot = max(int(ad.sum()), 1)
    shifted, valid = _shift_into(bd, best_y // ds, best_x // ds, ad.shape)
    zero, _ = _shift_into(bd, 0, 0, ad.shape)
    return best_y, best_x, int((ad & shifted).sum()) / tot, int((ad & zero).sum()) / tot


def pair_consistency(earlier, later):
    """Label-free consistency of one consecutive pair.

    `violation` is the fraction of the earlier crack absent from the later one after optimal
    alignment. A perfect segmenter on perfectly registered data gives 0.

    `area_decreased` is the blunt case no alignment can rescue: the later mask is smaller than
    the earlier one, so the crack shrank. That is physically impossible for the same region and
    is therefore a measured segmentation or acquisition error.
    """
    a = np.asarray(earlier, bool)
    b = np.asarray(later, bool)
    dy, dx, cont, cont_unreg = register_by_containment(a, b)
    ac, bc = _crop_common(a, b)
    moved, valid = _shift_into(bc, dy, dx, ac.shape)
    av = ac & valid
    return dict(
        dy=int(dy), dx=int(dx),
        containment=round(float(cont), 4),
        containment_unregistered=round(float(cont_unreg), 4),
        violation=round(float(1.0 - cont), 4),
        earlier_px=int(a.sum()), later_px=int(b.sum()),
        area_decreased=bool(int(b.sum()) < int(a.sum())),
        vanished_px=int((av & ~moved).sum()),
        new_px=int((moved & ~av).sum()),
    )


def monotone_repair(earlier, later):
    """Apply the constraint: the later mask must contain the registered earlier one.

    Returns (repaired_later, info). This can only ADD pixels, so it can only raise recall and
    can only lower precision -- which is exactly why it has to be measured against labels
    rather than assumed to help. See docs/IRREVERSIBILITY.md for that measurement.
    """
    a = np.asarray(earlier, bool)
    b = np.asarray(later, bool)
    dy, dx, cont, cont_unreg = register_by_containment(a, b)
    ac, bc = _crop_common(a, b)
    # move the EARLIER mask forward onto the later frame's grid: the inverse of the shift that
    # brought the later frame back onto the earlier one
    moved, valid = _shift_into(ac, -dy, -dx, bc.shape)
    repaired = bc | (moved & valid)
    return repaired, dict(dy=int(dy), dx=int(dx),
                          containment=round(float(cont), 4),
                          added_px=int((repaired & ~bc).sum()),
                          before_px=int(bc.sum()), after_px=int(repaired.sum()))


def sequence_report(masks_in_order):
    """Consistency across a whole ordered series. `masks_in_order` is [(label, mask), ...]."""
    out = []
    for i in range(1, len(masks_in_order)):
        (l0, m0), (l1, m1) = masks_in_order[i - 1], masks_in_order[i]
        r = pair_consistency(m0, m1)
        r["pair"] = f"{l0}->{l1}"
        out.append(r)
    return out
