"""
Train a DEPLOYABLE SAM+17 hybrid model on exactly the labels the current
champion (raw_v4) was trained on.

The point of this file is that the FEATURE SET is the only thing that changes.
It mirrors retrain_with_corrections.py's sampling recipe verbatim -- the same
bootstrap cap (100k/class/image from the 4 Ilastik masks), the same correction
cap (--neg-cap, default 2000, which is the value that produced raw_v4), the
same correction weight, the same RNG seed, the same architecture from
build_classifier() -- and then appends SAM's 256 embedding channels to each
sampled pixel's 17 hand-crafted features.

Everything measured so far was leave-one-image-out EVALUATION, which produces
no saved model. This produces one, so the hybrid can be run on all 71 images
and compared against the current outputs on the owner's own data.

Never overwrites models/pixel_hgb_final.joblib. Default output is
models/pixel_sam_hybrid.joblib.

Requires the embedding cache: python3 cache_sam_embeddings.py

Usage:
    python3 train_sam_hybrid.py [--neg-cap 2000] [--out models/pixel_sam_hybrid.joblib]
"""

import argparse
import glob
import os
import sys

import joblib
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sam_embeddings as EC
import paint_common as pc
from retrain_with_corrections import (BOOTSTRAP_N_PER_CLASS_PER_IMAGE,
                                      CORRECTION_N_PER_CLASS_PER_IMAGE,
                                      build_classifier, fit_with_sample_weight)
from txm_features import compute_feature_stack, robust_normalize

PROJECT_DIR = pc.PROJECT_DIR
DATASET_CACHE_DIR = os.path.join(PROJECT_DIR, "dataset_cache")
TILE = EC.TILE
EMB_STRIDE = 16
SEED = 0


def interp_tile(emb_tile, rr, cc):
    """Bilinear lookup of TILE-LOCAL pixel coords in one C x 64 x 64 grid.

    Vectorised across all 256 channels in one gather rather than looping
    scipy.ndimage.map_coordinates per channel. Same arithmetic -- verified equal
    to the per-channel version to float32 precision -- but the loop version was
    256 separate calls per batch, and prediction over 71 images calls this
    thousands of times, which made it the dominant cost of the whole run.
    """
    e = np.ascontiguousarray(emb_tile, dtype=np.float32)
    C, H, W = e.shape
    r = np.clip(rr / EMB_STRIDE - 0.5, 0, H - 1)
    c = np.clip(cc / EMB_STRIDE - 0.5, 0, W - 1)
    r0 = np.floor(r).astype(np.intp)
    c0 = np.floor(c).astype(np.intp)
    r1 = np.minimum(r0 + 1, H - 1)
    c1 = np.minimum(c0 + 1, W - 1)
    dr = (r - r0).astype(np.float32)
    dc = (c - c0).astype(np.float32)
    w00 = ((1 - dr) * (1 - dc))[:, None]
    w01 = ((1 - dr) * dc)[:, None]
    w10 = (dr * (1 - dc))[:, None]
    w11 = (dr * dc)[:, None]
    flat = e.reshape(C, H * W)
    g00 = flat[:, r0 * W + c0].T
    g01 = flat[:, r0 * W + c1].T
    g10 = flat[:, r1 * W + c0].T
    g11 = flat[:, r1 * W + c1].T
    return g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11


def _interp_tile_reference(emb_tile, rr, cc):
    """The per-channel scipy version, kept only to validate the fast path."""
    from scipy import ndimage as ndi
    e = emb_tile.astype(np.float32)
    r = np.clip(rr / EMB_STRIDE - 0.5, 0, e.shape[1] - 1)
    c = np.clip(cc / EMB_STRIDE - 0.5, 0, e.shape[2] - 1)
    out = np.empty((len(rr), e.shape[0]), np.float32)
    coords = np.stack([r, c])
    for k in range(e.shape[0]):
        out[:, k] = ndi.map_coordinates(e[k], coords, order=1, mode="nearest")
    return out


def sam_features_at(coords, emb, rows, cols):
    """SAM embedding vectors for arbitrary global (rows, cols) in one image.

    Tiles overlap where S.tiles clamped them inward at the right/bottom edges,
    so a pixel can fall in more than one. Later tiles are offered first and
    earlier ones only fill what is still unassigned -- deterministic, and it
    prefers the tile whose interior the pixel sits in rather than its margin.
    """
    out = np.zeros((len(rows), emb.shape[1]), np.float32)
    todo = np.ones(len(rows), bool)
    for k in range(len(coords) - 1, -1, -1):
        y0, x0 = int(coords[k][0]), int(coords[k][1])
        m = todo & (rows >= y0) & (rows < y0 + TILE) & (cols >= x0) & (cols < x0 + TILE)
        if not m.any():
            continue
        out[m] = interp_tile(emb[k], rows[m] - y0, cols[m] - x0)
        todo &= ~m
    if todo.any():
        print(f"    [warn] {int(todo.sum())} sampled px fell outside every tile")
    return out


def feats17_for(name):
    """The 17-feature stack, from cache when available, else computed the same way."""
    p = os.path.join(DATASET_CACHE_DIR, f"{name}_features.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    raw = tifffile.imread(pc._find_path(name)).astype(np.float64)
    return compute_feature_stack(robust_normalize(raw, 1.0, 99.0))


def hybrid_at(name, rows, cols):
    """Concatenated [17 hand-crafted | 256 SAM] for the given pixels."""
    f17 = feats17_for(name)
    a = np.asarray(f17[rows, cols, :], np.float32)
    del f17
    coords, emb = EC.ensure(name)
    b = sam_features_at(coords, emb, rows, cols)
    return np.concatenate([a, b], axis=1)


def load_bootstrap(rng):
    """4 Ilastik masks. Same caps and RNG order as retrain_with_corrections."""
    X, y, w = [], [], []
    for feat_path in sorted(glob.glob(os.path.join(DATASET_CACHE_DIR, "*_features.npy"))):
        stem = os.path.basename(feat_path)[: -len("_features.npy")]
        gt_path = os.path.join(DATASET_CACHE_DIR, f"{stem}_gt.npy")
        if not os.path.exists(gt_path):
            continue
        gt = np.load(gt_path)
        crack_idx = np.flatnonzero(gt)
        bg_idx = np.flatnonzero(~gt)
        n_c = min(BOOTSTRAP_N_PER_CLASS_PER_IMAGE, len(crack_idx))
        n_b = min(BOOTSTRAP_N_PER_CLASS_PER_IMAGE, len(bg_idx))
        idx = np.concatenate([rng.choice(crack_idx, n_c, replace=False),
                              rng.choice(bg_idx, n_b, replace=False)])
        rr, cc = np.unravel_index(idx, gt.shape)

        # The dataset_cache stems ("336_25") are abbreviations of the full paint
        # tool names, and the embedding cache is keyed by the full name.
        full = resolve_full_name(stem)
        if full is None:
            print(f"  bootstrap[{stem}]: SKIPPED, no matching image for SAM cache")
            continue
        X.append(hybrid_at(full, rr, cc))
        y.append(np.concatenate([np.ones(n_c, bool), np.zeros(n_b, bool)]))
        w.append(np.ones(n_c + n_b))
        print(f"  bootstrap[{stem}]: {n_c} crack + {n_b} background  -> {X[-1].shape[1]} features")
    return X, y, w


_NAME_CACHE = None


def resolve_full_name(stem):
    """dataset_cache stem -> full paint-tool image name."""
    global _NAME_CACHE
    if _NAME_CACHE is None:
        _NAME_CACHE = [i["name"] for i in pc.list_images()]
    key = stem.replace("LARGE_343_75", "343_75_LARGE")
    hits = [n for n in _NAME_CACHE if key in n]
    if not hits:
        low = key.lower()
        hits = [n for n in _NAME_CACHE if low in n.lower()]
    return hits[0] if hits else None


def load_corrections(neg_cap, correction_weight, rng):
    X, y, w = [], [], []
    for corr_path in sorted(glob.glob(os.path.join(pc.CORRECTIONS_DIR, "*_correction.npy"))):
        name = os.path.basename(corr_path)[: -len("_correction.npy")]
        corr = np.load(corr_path)
        crack_idx = np.flatnonzero(corr.reshape(-1) == 1)
        bg_idx = np.flatnonzero(corr.reshape(-1) == 2)
        if len(crack_idx) == 0 and len(bg_idx) == 0:
            continue
        if not os.path.exists(EC.cache_path(name)):
            print(f"  corrections[{name[:36]}]: SKIPPED, no SAM embedding cached")
            continue
        n_c = min(CORRECTION_N_PER_CLASS_PER_IMAGE, len(crack_idx))
        n_b = min(neg_cap, len(bg_idx))
        idx = np.concatenate([
            rng.choice(crack_idx, n_c, replace=False) if n_c else crack_idx[:0],
            rng.choice(bg_idx, n_b, replace=False) if n_b else bg_idx[:0]])
        if len(idx) == 0:
            continue
        rr, cc = np.unravel_index(idx, corr.shape)
        X.append(hybrid_at(name, rr, cc))
        y.append(np.concatenate([np.ones(n_c, bool), np.zeros(n_b, bool)]))
        w.append(np.full(n_c + n_b, correction_weight))
        print(f"  corrections[{name[:36]}]: {n_c} crack + {n_b} bg")
    return X, y, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-cap", type=int, default=2000,
                    help="max force-not-crack px per corrected image. 2000 is the "
                         "value that produced raw_v4 (43.9%% crack); the printed "
                         "crack fraction below must stay near 50%%.")
    ap.add_argument("--correction-weight", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(PROJECT_DIR, "models", "pixel_sam_hybrid.joblib"))
    args = ap.parse_args()

    rng = np.random.RandomState(SEED)
    print("Bootstrap samples (4 Ilastik masks), hybrid features...")
    Xb, yb, wb = load_bootstrap(rng)
    print("\nCorrection samples (paint tool), hybrid features...")
    Xc, yc, wc = load_corrections(args.neg_cap, args.correction_weight, rng)

    if not (Xb or Xc):
        sys.exit("no training data -- run cache_sam_embeddings.py first")
    X = np.concatenate(Xb + Xc).astype(np.float32)
    y = np.concatenate(yb + yc)
    w = np.concatenate(wb + wc)

    frac = float(y.mean())
    print(f"\nTraining set: {len(y):,} px, {X.shape[1]} features, {frac*100:.1f}% crack")
    if not (0.42 <= frac <= 0.58):
        print(f"  *** WARNING: crack fraction {frac*100:.1f}% is outside 42-58%. "
              f"class_weight='balanced' will skew the boundary; adjust --neg-cap. ***")

    clf = build_classifier()
    print("Fitting...")
    fit_with_sample_weight(clf, X, y, w)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(dict(model=clf, n_features=int(X.shape[1]), kind="sam17_hybrid",
                     neg_cap=args.neg_cap, correction_weight=args.correction_weight,
                     crack_fraction=frac, sam_model=EC.MODEL_ID, tile=TILE,
                     emb_stride=EMB_STRIDE), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
