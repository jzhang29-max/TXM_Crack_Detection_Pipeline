"""
Train a crack classifier on FLATFIELDED images.

Bootstrap labels come from the 4 Ilastik-derived ground-truth masks
(reused unchanged -- flatfielding is a per-pixel intensity correction and
moves nothing geometrically) plus every existing paint-tool correction,
whose features are recomputed from the FLATFIELDED counterpart of each
corrected image rather than the raw one.

Why flatfielded (measured, not assumed):
  - The raw-trained model's dominant rule is "broad dark region = crack"
    (large-sigma smoothed intensity ~41% of feature importance), and raw
    median brightness varies 2.6x across specimen groups (Wrought 0.575
    vs B2-training 1.518). Result: median predicted crack area 68.7%
    (Wrought) and 59.4% (AM) vs 28.3% on the training group, and an
    UNDAMAGED zero-cycle specimen predicted at 41% crack / 256 regions.
  - Flatfielding collapses every group's median brightness to ~0.997
    (AM 0.998, B2 0.997, B3 0.999, Wrought 0.996), and visual comparison
    confirms it also removes the mosaic-tile grid pattern and the broad
    illumination gradient -- two of the three main false-positive drivers.
    The specimen/off-specimen boundary becomes crisp.

Writes models/pixel_flatfield.joblib. Does NOT touch production.

Usage:
    python3 train_flatfield_model.py [--correction-weight 1.0]
"""

import argparse
import glob
import json
import os
import sys
import time

import joblib
import numpy as np
import tifffile
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
import retrain_with_corrections as rc
from txm_features import compute_feature_stack, robust_normalize
import build_flatfield_dataset as bf

CACHE_DIR = bf.CACHE_DIR
OUT_PATH = os.path.join(pc.PROJECT_DIR, "models", "pixel_flatfield.joblib")
FEATCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_featcache")
os.makedirs(FEATCACHE, exist_ok=True)

BOOTSTRAP_N = rc.BOOTSTRAP_N_PER_CLASS_PER_IMAGE
CORRECTION_N = rc.CORRECTION_N_PER_CLASS_PER_IMAGE


def flat_features_for(name):
    """Feature stack for `name` computed from its FLATFIELDED image, cached
    to disk since recomputing 17 features on a 24MP image is slow."""
    cp = os.path.join(FEATCACHE, f"{name}_features.npy")
    if os.path.exists(cp):
        return np.load(cp, mmap_mode="r")
    raw_path = pc._find_path(name)
    ff = bf.flatfield_path_for(raw_path)
    if ff is None:
        return None
    img01 = robust_normalize(tifffile.imread(ff).astype(np.float64), 1.0, 99.0)
    feats = compute_feature_stack(img01)
    np.save(cp, feats)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correction-weight", type=float, default=1.0)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--crack-cap", type=int, default=CORRECTION_N,
                     help="max force-crack pixels per corrected image. Must be tuned TOGETHER with "
                          "--neg-cap: once ~50 images carry positive labels, leaving this at 30000 while "
                          "capping negatives lower inverts the balance (measured: 72.6%% crack), which "
                          "swings the model back toward over-prediction just as the opposite imbalance "
                          "(27.5%% crack) did in v2.")
    ap.add_argument("--neg-cap", type=int, default=CORRECTION_N,
                     help="max force-not-crack pixels to take per corrected image. Matters because the "
                          "auto-written off-specimen corrections are negative-ONLY: at the default 30000 "
                          "across 59 new images they added 1.77M negatives and zero positives, dropping the "
                          "training set from 50%% crack to 27.5%%. class_weight='balanced' then upweighted "
                          "crack ~2.6x to compensate, which measurably made the model MORE trigger-happy "
                          "(a held-in B2 image went 30.2%% -> 36.1%% predicted crack vs ~29.7%% ground truth, "
                          "and an undamaged B2 specimen went 26.9%% -> 40.1%%). Lower this to keep the class "
                          "balance near 50/50 and avoid that.")
    args = ap.parse_args()
    rng = np.random.RandomState(0)

    X_parts, y_parts, w_parts = [], [], []

    # --- bootstrap: the 4 Ilastik GT images, flatfielded features ---
    with open(os.path.join(CACHE_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    for img in manifest["images"]:
        feats = np.load(img["feat_path"], mmap_mode="r")
        gt = np.load(img["gt_path"])
        flat = np.asarray(feats).reshape(-1, feats.shape[-1])
        fg = gt.reshape(-1)
        ci, bi = np.flatnonzero(fg), np.flatnonzero(~fg)
        nc, nb = min(BOOTSTRAP_N, len(ci)), min(BOOTSTRAP_N, len(bi))
        idx = np.concatenate([rng.choice(ci, nc, replace=False), rng.choice(bi, nb, replace=False)])
        X_parts.append(flat[idx])
        y_parts.append(np.concatenate([np.ones(nc, bool), np.zeros(nb, bool)]))
        w_parts.append(np.ones(len(idx)))
        print(f"  bootstrap[{img['name']}]: {nc} crack + {nb} background (flatfielded)")
        del feats, flat, gt

    # --- corrections: every existing paint-tool correction, flatfielded features ---
    for cp in sorted(glob.glob(os.path.join(pc.CORRECTIONS_DIR, "*_correction.npy"))):
        name = os.path.basename(cp)[: -len("_correction.npy")]
        corr = np.load(cp)
        if not (corr != 0).any():
            continue
        feats = flat_features_for(name)
        if feats is None:
            print(f"  [skip] corrections[{name}]: no flatfielded counterpart")
            continue
        if feats.shape[:2] != corr.shape:
            print(f"  [SKIP] corrections[{name}]: shape {feats.shape[:2]} != correction {corr.shape}")
            continue
        flat = np.asarray(feats).reshape(-1, feats.shape[-1])
        fc = corr.reshape(-1)
        ci, bi = np.flatnonzero(fc == 1), np.flatnonzero(fc == 2)
        nc, nb = min(args.crack_cap, len(ci)), min(args.neg_cap, len(bi))
        parts = []
        if nc: parts.append(rng.choice(ci, nc, replace=False))
        if nb: parts.append(rng.choice(bi, nb, replace=False))
        if not parts:
            continue
        idx = np.concatenate(parts)
        X_parts.append(flat[idx])
        y_parts.append(np.concatenate([np.ones(nc, bool), np.zeros(nb, bool)]))
        w_parts.append(np.full(len(idx), args.correction_weight))
        print(f"  corrections[{name[:56]}]: {nc} +crack / {nb} -crack (flatfielded)")
        del feats, flat

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    base_w = np.concatenate(w_parts)
    sw = compute_sample_weight("balanced", y) * base_w
    print(f"\nTraining on {len(y)} pixels ({int(y.sum())} crack)...")

    t0 = time.time()
    clf = rc.build_classifier()
    rc.fit_with_sample_weight(clf, X, y, sw)
    print(f"  fit time: {time.time()-t0:.1f}s")

    joblib.dump(clf, args.out)
    print(f"\nSaved {args.out} (production untouched)")


if __name__ == "__main__":
    main()
