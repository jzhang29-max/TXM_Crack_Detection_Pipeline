"""
Rebuild the feature/ground-truth cache from FLATFIELDED images instead of
raw ones, reusing the existing externally-derived ground-truth masks.

Why: the current raw-trained model generalizes badly across specimen
types. Measured root cause -- its dominant learned rule is "broad dark
region = crack" (large-sigma smoothed intensity is ~41% of total feature
importance), and median raw brightness varies 2.6x across specimen groups
(Wrought 0.575 vs B2-training 1.518). So on the darker groups the whole
frame reads as "dark => crack": median predicted crack area 68.7% for
Wrought and 59.4% for AM, versus 28.3% on the training group. The
negative controls confirm it's failure rather than genuine damage -- an
UNDAMAGED zero-fatigue-cycle specimen is predicted 41% crack / 256
regions, and a zero-load specimen 87.6% crack.

Flatfielding collapses every group's median brightness to ~0.997
(measured: AM 0.998, B2 0.997, B3 0.999, Wrought 0.996), removing exactly
that confound. A model trained on flatfielded data cannot lean on the
brightness crutch, so it should transfer across specimen types far better.

The ground-truth masks are reused unchanged: flatfielding is a per-pixel
intensity correction that does not move anything geometrically, so a mask
drawn on the raw image is still pixel-aligned to the flatfielded one.
Shapes are asserted identical rather than assumed.

Writes to dataset_cache_flatfield/ -- deliberately a SEPARATE directory
from dataset_cache/, so the existing raw-trained pipeline keeps working
and the two can be compared instead of one clobbering the other.

Usage:
    python3 build_flatfield_dataset.py
"""

import json
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txm_features import compute_feature_stack, robust_normalize
import build_dataset as bd
import paint_common as pc

CACHE_DIR = os.path.join(pc.PROJECT_DIR, "dataset_cache_flatfield")
RAW_CACHE_DIR = os.path.join(pc.PROJECT_DIR, "dataset_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def flatfield_path_for(raw_path):
    """Map a raw image path to its flatfielded counterpart. The new data
    lives under 'TXM DATA/<group>/...' with flatfielded output mirrored at
    'TXM DATA processed/flatfielded/<group>/...'; the four original
    ground-truth images predate that layout and live in standalone Desktop
    folders, so fall back to matching on basename across the whole
    flatfielded tree."""
    import glob
    if "/TXM DATA/" in raw_path:
        cand = raw_path.replace("/TXM DATA/", "/TXM DATA processed/flatfielded/")
        if os.path.exists(cand):
            return cand
    base = os.path.basename(raw_path)
    hits = glob.glob(os.path.join("/Users/jiamingzhang/Desktop/TXM DATA processed/flatfielded",
                                   "**", base), recursive=True)
    return hits[0] if hits else None


def main():
    manifest = {"images": [], "note": "flatfielded features + reused externally-derived GT masks"}

    for pair in bd.PAIRS:
        name = pair["name"]
        gt_path_existing = os.path.join(RAW_CACHE_DIR, f"{name}_gt.npy")
        if not os.path.exists(gt_path_existing):
            print(f"[skip] {name}: no existing ground-truth mask at {gt_path_existing}")
            continue

        ff = flatfield_path_for(pair["raw"])
        if ff is None:
            print(f"[skip] {name}: no flatfielded counterpart found for {os.path.basename(pair['raw'])}")
            continue

        gt = np.load(gt_path_existing)
        raw_ff = tifffile.imread(ff).astype(np.float64)
        if raw_ff.shape != gt.shape:
            print(f"[SKIP] {name}: flatfielded shape {raw_ff.shape} != GT shape {gt.shape} "
                  f"-- misaligned ground truth is worse than none")
            continue

        img01 = robust_normalize(raw_ff, 1.0, 99.0)
        print(f"[{name}] flatfielded {raw_ff.shape}, GT coverage {gt.mean():.4f} -- computing 17 features...")
        feats = compute_feature_stack(img01)

        feat_path = os.path.join(CACHE_DIR, f"{name}_features.npy")
        gt_out = os.path.join(CACHE_DIR, f"{name}_gt.npy")
        img_out = os.path.join(CACHE_DIR, f"{name}_img.npy")
        np.save(feat_path, feats)
        np.save(gt_out, gt)
        np.save(img_out, img01)
        manifest["images"].append(dict(
            name=name, shape=list(gt.shape), gt_coverage=float(gt.mean()),
            feat_path=feat_path, gt_path=gt_out, img_path=img_out,
            raw_source=ff, raw_source_original=pair["raw"],
        ))
        del feats, img01, raw_ff, gt
        print(f"  saved {feat_path}")

    with open(os.path.join(CACHE_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest['images'])} images to {CACHE_DIR}/manifest.json")


if __name__ == "__main__":
    main()
