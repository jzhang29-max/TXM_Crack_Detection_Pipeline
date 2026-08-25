"""Feature definitions for the RIDGE / VESSELNESS arms.

The gap this tests: the shipped 17 features are all isotropic -- intensity, Gaussian
smooths, gradient magnitude, Laplacian, local std. None of them can tell a dark BLOB from
a dark LINE. A crack is a line, and ridge/vesselness filters are the purpose-built
detectors for exactly that, so their absence is a real hole rather than an oversight we
can assume was already tested.

Nothing here writes to code/txm_features.py. The shipped 17 are imported and used as-is,
every arm is a COLUMN SUBSET of one cached matrix, and baseline_17 is recomputed here
rather than quoted from docs/CONTRAST.md -- that is what makes the arms self-comparable.

WHAT IS ALREADY IN THE BASELINE, AND WHAT THESE CHANNELS ACTUALLY ADD
The Laplacian is the TRACE of the Hessian, l1 + l2, and the shipped 17 already carry it
at sigma 1,2,4,8. So the baseline is not blind to second-order structure; it is blind to
how that structure is SPLIT between the two directions. A dark blob and a dark line can
have identical Laplacians and completely different eigenvalue pairs (blob: l1 ~ l2 > 0;
line: l1 >> l2 ~ 0). The hessian_eigs arm hands the classifier precisely that splitting,
which is the one genuinely new ingredient here. Everything the packaged filters do is a
fixed nonlinear recipe over the same eigenvalues.

Because of that, `hess_ratio` matters more than it looks. `aniso = l1 - l2` is deliberately
NOT a column: StandardScaler and the MLP's first layer are both linear, so any linear
combination of columns already present is free to the model and adding it measures nothing.
The RATIO is nonlinear and therefore is not free.

A TRAP THAT IS BUILT INTO TWO OF THE THREE PACKAGED FILTERS
skimage's `frangi` sets its structuredness constant to `gamma = s.max() / 2` when gamma is
None (the default) -- i.e. from the single largest Hessian norm IN THAT FRAME. Measured
over these 60 labelled frames that auto-gamma spans 14.3x (0.0079 to 0.1119). `meijering`
is worse: it divides each scale's response by that scale's own per-frame max
(skimage/filters/ridges.py, `vals /= max_val`), so its output is always [0,1] no matter how
faint the strongest ridge in the frame actually is.

That is the same defect that made local contrast normalisation the biggest loss in
docs/CONTRAST.md: training POOLS frames, so a channel rescaled by each frame's own
statistics means something different in every row-block and cannot be cashed in. `sato`
does NOT normalise per frame, and the raw Hessian eigenvalues do not either -- which is
why the arm list deliberately spans both kinds, and why `frangi_fixed_gamma` exists as a
controlled de-adaptivised version of frangi rather than as a tuning knob.

Cracks are DARK, so every packaged filter is called with black_ridges=True. Verified on a
synthetic dark line rather than assumed: black_ridges=True gives an on-line/background
response ratio of 2199x (frangi), 75x (sato), 90x (meijering), and black_ridges=False
gives <=0.6x, i.e. nothing.

Column layout of the cached matrix (34 columns):
     0..16  the 17 shipped features, from code/txm_features.compute_feature_stack
    17..18  frangi, auto gamma, sigma sets (1,2,3) and (2,4,6,8)
    19..20  sato,   same two sigma sets
    21..22  meijering, same two sigma sets
    23..24  frangi with gamma FIXED at 0.0757 (median of the auto value over these frames)
    25..33  Hessian eigenvalues at sigma 1,2,4: (l_maj, l_min, ratio) per scale
"""

import numpy as np
from scipy import ndimage as ndi

from txm_features import compute_feature_stack, FEATURE_NAMES, N_FEATURES

# Two sigma sets, both fixed in PIXELS. FINE spans the thin-crack regime the owner cares
# about (thin = median half-width <= 3 px); COARSE reaches the wide dominant cracks. Kept
# to two so that "which scale" is not confounded with "which filter".
SIGMAS_FINE = (1, 2, 3)
SIGMAS_COARSE = (2, 4, 6, 8)

# Median of frangi's own auto gamma (s.max()/2 at sigma 1) over the 60 labelled frames,
# measured before this file was written. Using the median means the fixed-gamma arm sits
# at the centre of the adaptive arm's range, so the comparison isolates the ADAPTIVITY
# rather than testing a different sensitivity.
GAMMA_FIXED = 0.0757

HESS_SIGMAS = (1, 2, 4)

BLACK_RIDGES = True   # cracks are dark

RIDGE_NAMES = [
    "frangi_fine", "frangi_coarse",
    "sato_fine", "sato_coarse",
    "meijering_fine", "meijering_coarse",
]
FRANGI_FIX_NAMES = ["frangi_fix_fine", "frangi_fix_coarse"]
HESS_NAMES = []
for _s in HESS_SIGMAS:
    HESS_NAMES += [f"hess_lmaj_s{_s}", f"hess_lmin_s{_s}", f"hess_ratio_s{_s}"]

ALL_NAMES = list(FEATURE_NAMES) + RIDGE_NAMES + FRANGI_FIX_NAMES + HESS_NAMES

_n = N_FEATURES
COL_BASE17 = list(range(0, _n))                                  # 17
COL_FRANGI = [_n + 0, _n + 1]
COL_SATO = [_n + 2, _n + 3]
COL_MEIJ = [_n + 4, _n + 5]
COL_PACKAGED = COL_FRANGI + COL_SATO + COL_MEIJ                  # 6
COL_FRANGI_FIX = [_n + 6, _n + 7]
COL_HESS = list(range(_n + 8, _n + 8 + 3 * len(HESS_SIGMAS)))    # 9
N_COLS = len(ALL_NAMES)

ARMS = {
    # the shipped features, recomputed here so every number below is self-comparable
    "baseline_17":            COL_BASE17,
    # one packaged filter at a time -- the headline question
    "17_plus_frangi":         COL_BASE17 + COL_FRANGI,
    "17_plus_sato":           COL_BASE17 + COL_SATO,
    "17_plus_meijering":      COL_BASE17 + COL_MEIJ,
    # the raw ingredients instead of a packaged score
    "17_plus_hessian_eigs":   COL_BASE17 + COL_HESS,
    # de-adaptivised frangi: same recipe, gamma no longer set by each frame's own max
    "17_plus_frangi_fixed":   COL_BASE17 + COL_FRANGI_FIX,
    # everything at once, to catch a combination no single arm shows
    "17_plus_all_ridge":      COL_BASE17 + COL_PACKAGED + COL_HESS,
    # DIAGNOSTIC, not a deployment candidate: the ridge channels with the 17 taken AWAY.
    # This is the framing under which the sibling SEM project found "Frangi families
    # collapse recall", so it says whether that result reproduces here -- and separates
    # "carries no information" from "carries information the 17 already have".
    "ridge_only_6":           COL_PACKAGED,
    "hessian_only_9":         COL_HESS,
}


def _ridge(fn, img01, sigmas, **kw):
    return np.asarray(fn(img01, sigmas=sigmas, black_ridges=BLACK_RIDGES,
                         **kw)).astype(np.float32)


def hessian_eigs(img01, sigma):
    """(l_maj, l_min, ratio) at one scale, sorted by MAGNITUDE not sign.

    skimage returns eigenvalues in decreasing ALGEBRAIC order, which for our sign
    convention is nearly always (positive-across-crack, ~0-along-crack) -- but not at
    every pixel, and a column whose meaning flips sign partway through the frame is
    noise. Sorting by |.| makes l_maj mean "the strong direction" everywhere.

    Verified on a synthetic dark line: for a DARK ridge the large-magnitude eigenvalue is
    POSITIVE (+0.042 across the line against +0.0002 on plain background), the other is
    ~0. So a crack should show l_maj > 0, l_min ~ 0, ratio ~ 0.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    ev = hessian_matrix_eigvals(
        hessian_matrix(img01, sigma=sigma, mode="reflect",
                       use_gaussian_derivatives=True))
    order = np.abs(ev).argsort(0)          # 0 -> smaller |.|, 1 -> larger |.|
    lmin = np.take_along_axis(ev, order[0][None], 0).squeeze(0)
    lmaj = np.take_along_axis(ev, order[1][None], 0).squeeze(0)
    # Frangi's blobness ingredient r_b, kept SIGNED so the model can tell a dark line
    # (ratio ~ 0, l_maj > 0) from a saddle (ratio < 0) from a blob (ratio ~ +1).
    ratio = lmin / np.where(np.abs(lmaj) < 1e-12, 1e-12, lmaj)
    return (lmaj.astype(np.float32), lmin.astype(np.float32),
            np.clip(ratio, -4.0, 4.0).astype(np.float32))


def ridge_stack(img01):
    """(H, W, 17) -> the 17 non-baseline channels as a list of full-frame float32 planes.

    Returned as a list rather than one array so each plane can be freed as soon as its
    sampled rows are taken; the largest labelled frame is 32 Mpx and holding 17 float32
    planes of it at once is 2.2 GB per worker.
    """
    from skimage.filters import frangi, sato, meijering
    planes = [
        _ridge(frangi, img01, SIGMAS_FINE),
        _ridge(frangi, img01, SIGMAS_COARSE),
        _ridge(sato, img01, SIGMAS_FINE),
        _ridge(sato, img01, SIGMAS_COARSE),
        _ridge(meijering, img01, SIGMAS_FINE),
        _ridge(meijering, img01, SIGMAS_COARSE),
        _ridge(frangi, img01, SIGMAS_FINE, gamma=GAMMA_FIXED),
        _ridge(frangi, img01, SIGMAS_COARSE, gamma=GAMMA_FIXED),
    ]
    for s in HESS_SIGMAS:
        planes.extend(hessian_eigs(img01, s))
    assert len(planes) == N_COLS - N_FEATURES, (len(planes), N_COLS - N_FEATURES)
    return planes


def sample_rows(img01, flat_idx):
    """Return (len(flat_idx), 34) float32 for the given flat pixel indices.

    Every filter is computed on the WHOLE frame and only then sampled: all of them are
    non-local, so cropping to a neighbourhood of the sampled pixels would silently change
    their values. Only the sampled rows are kept, which is what makes 60 frames totalling
    642 Mpx affordable.
    """
    h, w = img01.shape
    rr, cc = np.unravel_index(flat_idx, (h, w))
    out = np.empty((len(flat_idx), N_COLS), dtype=np.float32)

    base = compute_feature_stack(img01)
    out[:, COL_BASE17] = base[rr, cc, :]
    del base

    from skimage.filters import frangi, sato, meijering
    col = N_FEATURES
    for fn, sg, kw in [
        (frangi, SIGMAS_FINE, {}), (frangi, SIGMAS_COARSE, {}),
        (sato, SIGMAS_FINE, {}), (sato, SIGMAS_COARSE, {}),
        (meijering, SIGMAS_FINE, {}), (meijering, SIGMAS_COARSE, {}),
        (frangi, SIGMAS_FINE, dict(gamma=GAMMA_FIXED)),
        (frangi, SIGMAS_COARSE, dict(gamma=GAMMA_FIXED)),
    ]:
        p = _ridge(fn, img01, sg, **kw)
        out[:, col] = p[rr, cc]
        col += 1
        del p
    for s in HESS_SIGMAS:
        for p in hessian_eigs(img01, s):
            out[:, col] = p[rr, cc]
            col += 1
            del p
    assert col == N_COLS, (col, N_COLS)
    return out


def thin_half_width(img01, corr):
    """Median crack half-width in px, by the protocol in the task brief.

    Inside correction==1, keep pixels darker than the 20th percentile of img01 within the
    strokes, drop objects under 64 px, then take the median of the Euclidean distance
    transform sampled on the skeleton. Returns (half_width or None, n_dark_px).

    Byte-for-byte the same routine as research/contrast/augment_features.py, so the thin
    frame LIST is identical to the one the contrast sweep reported and the two experiments
    are talking about the same 33 frames.
    """
    from skimage.morphology import remove_small_objects, skeletonize

    strokes = corr == 1
    if int(strokes.sum()) == 0:
        return None, 0
    p20 = float(np.percentile(img01[strokes], 20))
    dark = strokes & (img01 <= p20)
    # max_size=63 keeps objects >= 64 px -- exactly what the deprecated min_size=64 did,
    # without the skimage 0.26 FutureWarning.
    dark = remove_small_objects(dark, max_size=63)
    nd = int(dark.sum())
    if nd == 0:
        return None, 0
    skel = skeletonize(dark)
    if not skel.any():
        return None, nd
    edt = ndi.distance_transform_edt(dark)
    return float(np.median(edt[skel])), nd
