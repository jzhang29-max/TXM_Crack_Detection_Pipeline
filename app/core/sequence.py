"""Sequence consistency for time-lapse crack imaging: irreversibility as a constraint.

WHAT THIS IS FOR. Fatigue damage does not heal. For two frames of the same region at increasing
load cycles, the earlier crack mask must be CONTAINED in the later one. Every crack-segmentation
method for X-ray tomography the author is aware of segments frames independently, so none of
them can use this. It gives three things a single-frame method cannot have:

  register_by_containment()  registration for imagery with no usable texture
  register_anchored()        the same, with the crack's fixed root as an anchor, which is what
                             breaks the along-axis degeneracy described below
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
control -- known shift, known growth -- recovers small shifts exactly and FAILS on large ones.
The cause is not a coding error: a roughly linear crack slid ALONG its own axis still overlaps
itself, so containment is nearly flat in that direction. Containment can register ACROSS a crack
and cannot register ALONG it. The real-data test could never have revealed this; only the
synthetic control could, which is the argument for always having one.

THE FIX IS AN ANCHOR, and it needs TWO DIFFERENT features because the two axes carry their
information in different places. A fatigue crack has a fixed root: it initiates at a free
surface and grows inward, so the mouth where it meets the specimen boundary stays put while the
tip advances. register_anchored() therefore takes

  the ACROSS-crack shift from the crack mouth       a genuine crack property
  the ALONG-crack shift from the specimen boundary  because the mouth's along-edge coordinate
                                                    is merely wherever the boundary is and
                                                    carries no crack information at all

and then refines locally with containment. Taking BOTH axes from the mouth was tried first and
fixed only one of them (dy -112 against a true -120, while dx came back 530 against a true 240),
which is what forced the split. Measured on a synthetic control rebuilt with a specimen edge and
a crack root fixed by construction, earlier length 600 px growing to 800 px:

  true shift     containment only      anchored
  (0, 0)         (0, 0)                (0, 0)          exact, 100% containment
  (12, -30)      (12, -30)             (12, -30)       exact, 100% containment
  (-120, 240)    (-124, 152)   wrong   (-120, 240)     exact, 100% containment
  (-40, 620)     (-48, 338)    wrong   (-40, 620)      exact
  (300, -500)    (316, -16)    wrong   (309, -152)   STILL WRONG

Four of five, including both cases containment alone got wrong. The remaining failure is
understood rather than mysterious: a -500 px shift carries the specimen boundary off the frame
entirely, so the along-edge reference does not exist to be measured. And the negative control --
an entirely DIFFERENT synthetic crack, root at a different height -- falls from 46% containment
to 23.5%, so the metric now separates consistent from inconsistent pairs far better than it did.
Still, no threshold on it has been validated against enough pairs to be defensible.

WHAT THE ANCHOR DOES TO THE REAL NUMBERS, which is the part that matters for the paper. Over the
13 consecutive pairs in this project's two usable series:

  3 of 13 pairs have the LATER crack SMALLER than the earlier one. No registration can rescue
    that. It is an outright irreversibility violation and therefore a measured segmentation or
    acquisition error, found without any labels.
  On the wide-FOV wrought frames the mouth is not locatable and the method falls back to
    containment only, so the anchor is not available everywhere.
  Where the anchor does fire it agrees with containment-only on 4 of 6 HC pairs and cuts the
    other two hard: 78.7% -> 46.8%, and 71.3% -> 25.5%.

That last line is the honest headline. The anchored number is LOWER because containment-only was
finding along-axis shifts that scored well and were wrong. So the previously reported 97.3%
remains an UPPER BOUND rather than a measurement, and the properly constrained estimate of
sequence consistency in this data is materially worse than the unconstrained one. The anchor did
not rescue the claim; it measured how much the claim was inflated.

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


# ---------------------------------------------------------------- anchored registration
# Containment alone cannot resolve displacement along a crack's own axis (see the module
# docstring). The fix is physical rather than numerical: a fatigue crack has a FIXED ROOT. It
# initiates at a free surface and grows inward, so the mouth where it meets the specimen
# boundary does not move between load steps while the tip does. A point anchor constrains both
# axes, which is exactly what the degenerate direction was missing.

MOUTH_BAND = 60          # px from the specimen boundary counted as "the mouth"


def _edge_position(img):
    """Median column at which the specimen begins, i.e. the boundary's along-edge position.

    A one-dimensional estimate of the one thing the crack cannot supply. Robust because it is
    a median over every row that has a boundary at all, and because the boundary is the single
    highest-contrast feature in these frames.
    """
    sp = specimen_mask(img)
    if not sp.any():
        return None
    has = sp.any(axis=1)
    if has.sum() < 20:
        return None
    first = np.argmax(sp[has], axis=1).astype(float)
    return float(np.median(first))


def specimen_mask(img, thresh_pct=45):
    """Rough specimen support: the bright side of the frame.

    Deliberately crude. It only has to locate the boundary well enough to pick out which crack
    pixels are at the mouth, and a percentile split does that on this imagery. pipeline.py has
    a far more careful specimen_support() for anything that needs to be right.
    """
    from scipy.ndimage import binary_closing, binary_fill_holes
    a = np.asarray(img, np.float32)
    lo, hi = np.percentile(a, [2, 98])
    n = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    bright = n > (thresh_pct / 100.0)
    # CLOSE AND FILL. The crack is DARK, so a plain brightness threshold carves it out of the
    # specimen -- and then the crack mouth, which lies exactly on the boundary, falls outside
    # the support and the anchor can never be found. The first version of this returned None
    # for every frame for that reason. Closing bridges the crack and filling seals it.
    return binary_fill_holes(binary_closing(bright, np.ones((9, 9), bool)))


def crack_mouth(mask, spec):
    """Centroid of the crack pixels lying within MOUTH_BAND of the specimen boundary.

    Returns (y, x) or None. This is the anchor: physically fixed across load steps, whereas
    the crack's centroid, bounding box and tip all move as it grows.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    m = np.asarray(mask, bool)
    sp = np.asarray(spec, bool)
    if not m.any() or not sp.any():
        return None
    # boundary of the specimen: dilate the outside and intersect with the inside
    outside = binary_dilation(~sp, iterations=3)
    band = outside & sp
    for _ in range(max(1, MOUTH_BAND // 8)):
        band = binary_dilation(band, iterations=4) & sp
    at_mouth = m & band
    if at_mouth.sum() < 20:
        return None
    ys, xs = np.nonzero(at_mouth)
    return float(ys.mean()), float(xs.mean())


def register_anchored(earlier, later, img_earlier, img_later, refine=48):
    """Register by crack-mouth correspondence, then refine with containment locally.

    The anchor fixes the along-axis direction that containment cannot see; the local
    containment refinement then recovers the last few pixels. Falls back to pure containment
    if either mouth cannot be located, and says which path it took.
    """
    a = np.asarray(earlier, bool)
    b = np.asarray(later, bool)
    ma = crack_mouth(a, specimen_mask(img_earlier))
    mb = crack_mouth(b, specimen_mask(img_later))
    if ma is None or mb is None:
        dy, dx, cont, unreg = register_by_containment(a, b)
        return dict(dy=dy, dx=dx, containment=round(float(cont), 4),
                    containment_unregistered=round(float(unreg), 4), method="containment_only",
                    reason="crack mouth not locatable in one or both frames")
    # mouth of `later` sits at mb; to bring it onto ma we move later by (ma - mb), and
    # _shift_into moves by (-dy,-dx), so dy = mb - ma
    dy0 = int(round(mb[0] - ma[0]))
    # THE TWO AXES CARRY INFORMATION IN DIFFERENT PLACES, so they are estimated separately.
    # The crack root's ACROSS-crack coordinate (here y) is a genuine crack property and the
    # mouth gives it directly. Its ALONG-crack coordinate is not: the mouth sits wherever the
    # specimen boundary is, so its x says nothing about the crack. Measured on the synthetic
    # control, taking both axes from the mouth fixed dy (-112 against a true -120) and left dx
    # badly wrong (530 against 240). So dx comes from the boundary itself.
    ex, ey = _edge_position(img_earlier), _edge_position(img_later)
    dx0 = int(round(ey - ex)) if (ex is not None and ey is not None) else \
        int(round(mb[1] - ma[1]))
    ac, bc = _crop_common(a, b)
    best = (-1, dy0, dx0)
    ds = 2
    ad, bd = ac[::ds, ::ds], bc[::ds, ::ds]
    tot = max(int(ad.sum()), 1)
    for dy in range(dy0 - refine, dy0 + refine + 1, ds):
        for dx in range(dx0 - refine, dx0 + refine + 1, ds):
            shifted, valid = _shift_into(bd, dy // ds, dx // ds, ad.shape)
            v = int((ad & shifted).sum()) - int((ad & ~valid).sum())
            if v > best[0]:
                best = (v, dy, dx)
    _, dy, dx = best
    shifted, _ = _shift_into(bd, dy // ds, dx // ds, ad.shape)
    zero, _ = _shift_into(bd, 0, 0, ad.shape)
    return dict(dy=int(dy), dx=int(dx),
                containment=round(int((ad & shifted).sum()) / tot, 4),
                containment_unregistered=round(int((ad & zero).sum()) / tot, 4),
                method="anchored", anchor_shift=(dy0, dx0),
                mouth_earlier=(round(ma[0], 1), round(ma[1], 1)),
                mouth_later=(round(mb[0], 1), round(mb[1], 1)))
