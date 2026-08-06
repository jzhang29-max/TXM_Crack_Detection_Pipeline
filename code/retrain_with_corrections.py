"""
Fold manual corrections made in the paint tool (paint_server.py) back into
the pixel classifier's training data and retrain, closing the loop the same
way the CBS SEM project's active-learning system does (ingest_labels.py ->
train_interior_model.py) -- just without a separate discrete "ingest" step,
since a TXM correction is already just labeled pixels, not a candidate
region needing to be merged into a review CSV.

What counts as training signal:
  1. The original 4-image Ilastik-derived ground truth (dataset_cache/),
     sampled at BOOTSTRAP_N_PER_CLASS_PER_IMAGE (default 100,000/class/image).
  2. Every pixel a human explicitly corrected in the paint tool
     (paint/corrections/<name>_correction.npy: 1=forced crack,
     2=forced not-crack), for ANY image that's been opened in the tool --
     not just the original 4 training images. Capped separately at
     CORRECTION_N_PER_CLASS_PER_IMAGE (default 30,000/class/image) and
     weighted by --correction-weight (default 1.0, i.e. equal footing with
     bootstrap pixels).

These two defaults (bootstrap=100k, correction weight=1.0) are NOT
arbitrary -- they're the result of a real regression chase, worth knowing
before changing them:
  - An earlier version used correction_weight=5.0 (treating a human
    correction as 5x more trustworthy per-pixel than the Ilastik
    bootstrap). Combined with the correction pool now spanning 12 images
    instead of 4, that 5x weight let corrections dominate the loss so
    heavily that the model's fit on images' own *original* ground truth
    measurably regressed -- verified via IoU against a "corrected ground
    truth" (Ilastik + overrides) that isn't just stale-label bias.
  - Lowering the weight alone only partially fixed it. The real fix was
    also raising BOOTSTRAP_N_PER_CLASS_PER_IMAGE from 30k to 100k, so the
    original clean 4-image signal isn't so heavily outnumbered in raw
    pixel count by the (inherently noisier, hand-painted) correction data
    regardless of weighting. This combination (bootstrap=100k, weight=1.0)
    is what actually closed most of the gap: mean IoU vs corrected ground
    truth went from ~0.638 (5x weight, 30k bootstrap) to ~0.742.
  - CORRECTION_N_PER_CLASS_PER_IMAGE stayed at 30k throughout -- raising
    it (tried at 50k alongside a lower weight) partially cancels the
    weight reduction instead of compounding it, since the cap directly
    controls how many correction pixels get fed to the class-balance
    weighting regardless of the explicit weight multiplier.
  - Watch the border/edge region specifically after any change here --
    boosting bootstrap volume can pull in more edge-adjacent pixels from
    the largest image and reintroduce vignetting-correlated false
    positives near tile boundaries. Always re-check row/col pixel-density
    profiles near the image edges on the largest image after retraining,
    not just aggregate IoU (a border artifact barely moves an aggregate
    metric on a 24-megapixel image but is very visible).

Usage:
    python3 retrain_with_corrections.py [--correction-weight 1.0] [--out models/pixel_hgb_v2.joblib]

Never overwrites the current production model (models/pixel_hgb_final.joblib)
by default -- writes a new versioned file so you can compare before
switching paint_common.MODEL_PATH / apply_pixel_model.py's default over.
"""

import argparse
import glob
import os
import sys

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txm_features import compute_feature_stack, robust_normalize, N_FEATURES
import paint_common as pc

import tifffile

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_CACHE_DIR = os.path.join(PROJECT_DIR, "dataset_cache")

# --- Current champion architecture: MLPClassifier (neural network) -----
# Switched from HistGradientBoostingClassifier after
# evaluate_mlp_production_candidate.py showed a consistent, gate-passing
# accuracy improvement on the REAL production recipe (bootstrap=100k +
# corrections=30k/class/image across all 12 images): mean IoU vs corrected
# ground truth 0.778 vs 0.742 (+0.036) across all 4 GT images, with zero
# border/spontaneous-artifact/degenerate-output flags. See
# results/mlp_candidate_gate_report.json and benchmark_figures/fig_l_* for
# the evidence. build_classifier() below is the SINGLE place this project
# decides which architecture to train -- retrain_and_deploy.py calls this
# same function rather than constructing a classifier itself, specifically
# so the automated retrain loop can't silently drift back to a different
# architecture than whatever this file says is current.
MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=300,
    early_stopping=True, random_state=0,
)

# Previous champion (kept for reference/rollback only -- build_classifier()
# below no longer constructs this). Was the winning variant from the
# original model comparison, results/pixel_hgb_results.json.
HGB_PARAMS = dict(
    max_iter=300, max_depth=8, learning_rate=0.1,
    class_weight="balanced", random_state=0,
)

BOOTSTRAP_N_PER_CLASS_PER_IMAGE = 100000
CORRECTION_N_PER_CLASS_PER_IMAGE = 30000


def build_classifier():
    """Single source of truth for "what model architecture do we currently
    train" -- both this module's main() and retrain_and_deploy.py's
    train_candidate() call this instead of constructing a classifier
    inline, so there is exactly one place to change if the champion
    architecture changes again."""
    return Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(**MLP_PARAMS))])


def fit_with_sample_weight(clf, X, y, sample_weight):
    """Fits clf on (X, y, sample_weight), routing the weight to the right
    step if clf is a Pipeline (the current MLP champion) or passing it
    directly if clf is a plain estimator (the previous HGB champion) --
    keeps callers architecture-agnostic. Verified against this project's
    installed sklearn (1.7.2): MLPClassifier.fit genuinely uses
    sample_weight in its loss computation, not just accepts and ignores it."""
    if isinstance(clf, Pipeline):
        final_step_name = clf.steps[-1][0]
        clf.fit(X, y, **{f"{final_step_name}__sample_weight": sample_weight})
    else:
        clf.fit(X, y, sample_weight=sample_weight)
    return clf


def load_bootstrap_samples(rng):
    """Same balanced-sampling recipe as train_pixel_hgb.py, from the cached
    Ilastik-derived ground truth."""
    X_parts, y_parts, w_parts = [], [], []
    for feat_path in sorted(glob.glob(os.path.join(DATASET_CACHE_DIR, "*_features.npy"))):
        name = os.path.basename(feat_path)[: -len("_features.npy")]
        gt_path = os.path.join(DATASET_CACHE_DIR, f"{name}_gt.npy")
        if not os.path.exists(gt_path):
            continue
        feats = np.load(feat_path)
        gt = np.load(gt_path)

        crack_idx = np.flatnonzero(gt)
        bg_idx = np.flatnonzero(~gt)
        n_crack = min(BOOTSTRAP_N_PER_CLASS_PER_IMAGE, len(crack_idx))
        n_bg = min(BOOTSTRAP_N_PER_CLASS_PER_IMAGE, len(bg_idx))
        crack_sample = rng.choice(crack_idx, size=n_crack, replace=False)
        bg_sample = rng.choice(bg_idx, size=n_bg, replace=False)

        flat_feats = feats.reshape(-1, feats.shape[-1])
        idx = np.concatenate([crack_sample, bg_sample])
        X_parts.append(flat_feats[idx])
        y_parts.append(np.concatenate([np.ones(n_crack, bool), np.zeros(n_bg, bool)]))
        w_parts.append(np.ones(len(idx)))
        print(f"  bootstrap[{name}]: {n_crack} crack + {n_bg} background")
    return X_parts, y_parts, w_parts


def load_correction_samples(correction_weight, rng, max_per_class_per_image=CORRECTION_N_PER_CLASS_PER_IMAGE):
    """Every human-corrected pixel across every image ever opened in the
    paint tool, regardless of whether that image was one of the original 4
    training images.

    Capped at `max_per_class_per_image` (same density as the bootstrap
    sampling) rather than using every corrected pixel verbatim. A brush
    correction easily touches hundreds of thousands of pixels once you've
    painted over a broad over-marked region -- taking all of them let 6
    corrected images outnumber the original 4-image bootstrap by ~15x in
    raw pixel count. Combined with the correction weight on top of that,
    the retrained model's decision boundary was being set almost entirely
    by those 6 (smaller, differently-lit) images: LARGE's own bootstrap
    contribution dropped to ~1.7% of the full training set, and the model
    lost its calibration for LARGE's vignetting specifically, causing a
    large new false-positive band across its top edge that wasn't there
    before. Capping keeps every image's *contribution* comparable while
    still trusting each individual correction pixel more via the weight.
    """
    X_parts, y_parts, w_parts = [], [], []
    pattern = os.path.join(pc.CORRECTIONS_DIR, "*_correction.npy")
    for corr_path in sorted(glob.glob(pattern)):
        name = os.path.basename(corr_path)[: -len("_correction.npy")]
        correction = np.load(corr_path)
        n_crack_total = int((correction == 1).sum())
        n_bg_total = int((correction == 2).sum())
        if n_crack_total == 0 and n_bg_total == 0:
            continue

        # Reuse cached features if this image already has them (it's one of
        # the 4 original training images); otherwise compute fresh from the
        # raw source image -- corrections can come from ANY image opened in
        # the paint tool, not just the original training set.
        feat_path = os.path.join(DATASET_CACHE_DIR, f"{name}_features.npy")
        if os.path.exists(feat_path):
            feats = np.load(feat_path)
        else:
            raw_path = pc._find_path(name)
            raw = tifffile.imread(raw_path).astype(np.float64)
            img01 = robust_normalize(raw, 1.0, 99.0)
            feats = compute_feature_stack(img01)

        flat_feats = feats.reshape(-1, feats.shape[-1])
        flat_corr = correction.reshape(-1)
        crack_idx = np.flatnonzero(flat_corr == 1)
        bg_idx = np.flatnonzero(flat_corr == 2)

        n_crack = min(max_per_class_per_image, len(crack_idx))
        n_bg = min(max_per_class_per_image, len(bg_idx))
        crack_sample = rng.choice(crack_idx, size=n_crack, replace=False) if n_crack else crack_idx
        bg_sample = rng.choice(bg_idx, size=n_bg, replace=False) if n_bg else bg_idx
        idx = np.concatenate([crack_sample, bg_sample])
        if len(idx) == 0:
            continue

        X_parts.append(flat_feats[idx])
        y_parts.append(np.concatenate([np.ones(n_crack, bool), np.zeros(n_bg, bool)]))
        w_parts.append(np.full(len(idx), correction_weight, dtype=np.float64))
        print(f"  corrections[{name}]: {n_crack}/{n_crack_total} forced-crack + "
              f"{n_bg}/{n_bg_total} forced-background px sampled (weight x{correction_weight})")
    return X_parts, y_parts, w_parts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--correction-weight", type=float, default=1.0,
                     help="sample-weight multiplier for human-corrected pixels relative to bootstrapped Ilastik pixels "
                          "(1.0 = equal footing; see module docstring for why higher values caused a real regression)")
    ap.add_argument("--out", default=os.path.join(PROJECT_DIR, "models", "pixel_model_retrained.joblib"),
                     help="output path for the retrained model (never overwrites pixel_hgb_final.joblib by default)")
    args = ap.parse_args()

    rng = np.random.RandomState(0)

    print("Loading bootstrapped Ilastik-derived samples (dataset_cache/)...")
    X_boot, y_boot, w_boot = load_bootstrap_samples(rng)

    print("Loading human correction samples (paint/corrections/)...")
    X_corr, y_corr, w_corr = load_correction_samples(args.correction_weight, rng)

    if not X_corr:
        print("\nNo corrections found yet -- nothing to add. Open the paint tool, make some "
              "corrections, click 'Save corrections', then run this again.")
        sys.exit(0)

    X = np.concatenate(X_boot + X_corr, axis=0)
    y = np.concatenate(y_boot + y_corr, axis=0)
    base_weight = np.concatenate(w_boot + w_corr, axis=0)

    # class_weight='balanced' handles crack/background imbalance; multiply in
    # the correction emphasis on top of that rather than picking one or the
    # other. (The MLP champion has no built-in class_weight param the way
    # the tree-ensemble champions did, so this balanced-weight computation
    # is what actually supplies that half of the behavior now.)
    class_weight = compute_sample_weight("balanced", y)
    sample_weight = class_weight * base_weight

    n_total = len(y)
    n_correction_px = sum(len(w) for w in w_corr)
    print(f"\nTraining on {n_total} pixels total ({n_correction_px} from human corrections, "
          f"weighted {args.correction_weight}x)...")

    clf = build_classifier()
    fit_with_sample_weight(clf, X, y, sample_weight)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(clf, args.out)
    print(f"\nSaved retrained model: {args.out}")
    print("This does NOT replace the production model automatically. To switch to it:")
    print(f"  cp {args.out} {os.path.join(PROJECT_DIR, 'models', 'pixel_hgb_final.joblib')}")
    print("(back up the old one first if you want to be able to revert)")


if __name__ == "__main__":
    main()
