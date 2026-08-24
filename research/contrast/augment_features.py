"""Feature definitions for the ADD-contrast-alongside arms.

The question this file exists to answer: flat-fielding the model INPUT cost 0.169 IoU on
this dataset, because the large-radius smoothed-intensity features carry ~41% of the
model's importance and flat-fielding removes exactly those (docs/MARKUP_GUIDE.md). That
is an argument against REPLACING the intensity signal. It says nothing about ADDING a
local-contrast signal in extra columns while leaving the original 17 untouched -- which is
what this builds.

Nothing here writes to code/txm_features.py. The original 17 are imported and used as-is
so every arm is a column subset of one cached matrix, and therefore self-comparable.

Column layout of the cached matrix (37 columns):
    0..16   the 17 shipped features, from code/txm_features.compute_feature_stack
    17      lcn_w51   local contrast normalisation, uniform window 51 px
    18      lcn_w151  local contrast normalisation, uniform window 151 px
    19      dog_g8    img - gaussian(img, 8)
    20..36  the same 17 features recomputed on the CLAHE-equalised image
            (so col 20 = clahe intensity, col 27 = clahe gradmag_s1, col 28 = clahe
             gradmag_s2, ... same order as FEATURE_NAMES)
"""

import numpy as np
from scipy import ndimage as ndi

from txm_features import compute_feature_stack, FEATURE_NAMES, N_FEATURES

# CLAHE settings. kernel_size is fixed in PIXELS rather than left at skimage's default
# (image_shape // 8) so that a 1693 px frame and a 6367 px frame get the same physical
# amount of local adaptation -- the shipped 17 features are also fixed in pixels (largest
# Gaussian sigma 64, so a reach of ~256 px), and 256 matches that reach.
CLAHE_KERNEL = 256
CLAHE_CLIP = 0.01
CLAHE_NBINS = 256

LCN_WINDOWS = [51, 151]
DOG_SIGMA = 8

CONTRAST_NAMES = [f"lcn_w{w}" for w in LCN_WINDOWS] + [f"dog_g{DOG_SIGMA}"]
CLAHE_NAMES = [f"clahe_{n}" for n in FEATURE_NAMES]
ALL_NAMES = list(FEATURE_NAMES) + CONTRAST_NAMES + CLAHE_NAMES

# Column index groups, used to slice arms out of the cached matrix.
COL_BASE17 = list(range(0, N_FEATURES))                      # 17
COL_CONTRAST = list(range(N_FEATURES, N_FEATURES + 3))       # 3
COL_CLAHE17 = list(range(N_FEATURES + 3, N_FEATURES + 3 + N_FEATURES))  # 17
COL_CLAHE_INT = [COL_CLAHE17[FEATURE_NAMES.index("intensity")]]
COL_CLAHE_GRAD = [COL_CLAHE17[FEATURE_NAMES.index("gradmag_s2")]]

ARMS = {
    "baseline_17":            COL_BASE17,
    "17_plus_contrast3":      COL_BASE17 + COL_CONTRAST,
    "17_plus_clahe_int":      COL_BASE17 + COL_CLAHE_INT,
    "17_plus_clahe_int_grad": COL_BASE17 + COL_CLAHE_INT + COL_CLAHE_GRAD,
    "34_dup_clahe_stack":     COL_BASE17 + COL_CLAHE17,
}


def local_contrast(img01, w):
    """(img - mean_w) / std_w on a uniform (box) window of side w.

    float64 throughout: the sum-of-squares variance trick cancels two large numbers, and
    in float32 that cancellation goes negative often enough on smooth regions to matter.
    """
    x = img01.astype(np.float64)
    m = ndi.uniform_filter(x, size=w, mode="reflect")
    m2 = ndi.uniform_filter(x * x, size=w, mode="reflect")
    sd = np.sqrt(np.clip(m2 - m * m, 0.0, None))
    return ((x - m) / (sd + 1e-6)).astype(np.float32)


def clahe_image(img01):
    from skimage.exposure import equalize_adapthist
    return equalize_adapthist(img01.astype(np.float64), kernel_size=CLAHE_KERNEL,
                              clip_limit=CLAHE_CLIP, nbins=CLAHE_NBINS).astype(np.float32)


def sample_rows(img01, flat_idx):
    """Return (len(flat_idx), 37) float32 for the given flat pixel indices.

    Filters are computed on the whole frame -- they are non-local, so cropping to the
    sampled pixels would change their values -- but only the sampled rows are kept, which
    is what makes 60 frames up to 23 Mpx affordable.
    """
    h, w = img01.shape
    rr, cc = np.unravel_index(flat_idx, (h, w))
    out = np.empty((len(flat_idx), len(ALL_NAMES)), dtype=np.float32)

    base = compute_feature_stack(img01)
    out[:, COL_BASE17] = base[rr, cc, :]
    del base

    for j, win in enumerate(LCN_WINDOWS):
        out[:, COL_CONTRAST[j]] = local_contrast(img01, win)[rr, cc]
    out[:, COL_CONTRAST[-1]] = (img01 - ndi.gaussian_filter(img01, sigma=DOG_SIGMA))[rr, cc]

    ce = clahe_image(img01)
    cstack = compute_feature_stack(ce)
    out[:, COL_CLAHE17] = cstack[rr, cc, :]
    del cstack, ce

    return out


def thin_half_width(img01, corr):
    """Median crack half-width in px, by the protocol in the task brief.

    Inside correction==1, keep pixels darker than the 20th percentile of img01 within the
    strokes, drop objects under 64 px, then take the median of the Euclidean distance
    transform sampled on the skeleton. Returns (half_width or None, n_dark_px).
    """
    from skimage.morphology import remove_small_objects, skeletonize

    strokes = corr == 1
    n = int(strokes.sum())
    if n == 0:
        return None, 0
    p20 = float(np.percentile(img01[strokes], 20))
    dark = strokes & (img01 <= p20)
    # max_size=63 removes objects of size <= 63, i.e. keeps >= 64 -- exactly what the
    # deprecated min_size=64 did, without the skimage 0.26 FutureWarning.
    dark = remove_small_objects(dark, max_size=63)
    nd = int(dark.sum())
    if nd == 0:
        return None, 0
    skel = skeletonize(dark)
    if not skel.any():
        return None, nd
    edt = ndi.distance_transform_edt(dark)
    return float(np.median(edt[skel])), nd
