"""
Build and cache the dense multi-scale feature stack + ground-truth crack
mask for every TXM image that has a matching, pixel-aligned external
probability map. Run once; downstream training/evaluation scripts just load
the cached .npy files (fast) instead of recomputing features every time.

Ground truth: threshold the external crack-class probability channel at 0.5.
Pairs whose raw image and probability map don't have identical (H, W) are
skipped outright (misaligned ground truth is worse than no ground truth).
"""

import json
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txm_features import compute_feature_stack, robust_normalize, FEATURE_NAMES

PAIRS = [
    dict(
        name="LARGE_343_75",
        raw="/Users/jiamingzhang/Desktop/260618_b2_343_75_LARGE/Average_mosaic_260618_b2_343_75_LARGE_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif",
        prob="/Users/jiamingzhang/Desktop/260618_b2_343_75_LARGE/Result of Average_mosaic_260618_b2_343_75_LARGE_idx00000_mosaictileAA_img001of010.xrm.bim.bim_Probabilities.tif - C=1.tif",
        channel=None,
    ),
    dict(
        name="333_75_um_zoom",
        raw="/Users/jiamingzhang/Desktop/260618_b2_333_75_to_339_06/Average_mosaic_260618_B2_333_75_um_zoom_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif",
        prob="/Users/jiamingzhang/Desktop/260618_b2_333_75_to_339_06/Result of Average_mosaic_260618_B2_333_75_um_zoom_idx00000_mosaictileAA_img001of010.xrm.bim.bim_Probabilities.tif",
        channel=1,
    ),
    dict(
        name="336_25",
        raw="/Users/jiamingzhang/Desktop/260618_b2_333_75_to_339_06/Average_mosaic_260618_b2_336_25_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif",
        prob="/Users/jiamingzhang/Desktop/260618_b2_333_75_to_339_06/Result of Average_mosaic_260618_b2_336_25_idx00000_mosaictileAA_img001of010.xrm.bim.bim_Probabilities.tif",
        channel=1,
    ),
    dict(
        name="338_13",
        raw="/Users/jiamingzhang/Desktop/260618_b2_333_75_to_339_06/Average_mosaic_260618_b2_338_13_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif",
        prob="/Users/jiamingzhang/Desktop/260618_b2_333_75_to_339_06/Result of Average_mosaic_260618_b2_338_13_idx00000_mosaictileAA_img001of010.xrm.bim.bim_Probabilities.tif",
        channel=1,
    ),
]

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_cache")


def load_probability_channel(path, channel):
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[..., channel]
    arr = arr.astype(np.float64)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest = {"feature_names": FEATURE_NAMES, "images": []}

    for pair in PAIRS:
        name = pair["name"]
        raw = tifffile.imread(pair["raw"]).astype(np.float64)
        prob_raw = load_probability_channel(pair["prob"], pair["channel"])

        if raw.shape != prob_raw.shape:
            print(f"SKIP {name}: shape mismatch raw={raw.shape} prob={prob_raw.shape}")
            continue

        img01 = robust_normalize(raw, 1.0, 99.0)
        print(f"{name}: computing feature stack for shape {img01.shape} ...")
        feats = compute_feature_stack(img01)
        gt = (prob_raw >= 0.5)

        feat_path = os.path.join(CACHE_DIR, f"{name}_features.npy")
        gt_path = os.path.join(CACHE_DIR, f"{name}_gt.npy")
        img_path = os.path.join(CACHE_DIR, f"{name}_img.npy")
        np.save(feat_path, feats)
        np.save(gt_path, gt)
        np.save(img_path, img01)

        coverage = float(gt.mean())
        print(f"{name}: saved. shape={feats.shape} gt_coverage={coverage*100:.2f}%")

        manifest["images"].append(
            dict(name=name, shape=list(img01.shape), gt_coverage=coverage,
                 feat_path=feat_path, gt_path=gt_path, img_path=img_path,
                 raw_source=pair["raw"])
        )

    with open(os.path.join(CACHE_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest with {len(manifest['images'])} image(s) to {CACHE_DIR}/manifest.json")


if __name__ == "__main__":
    main()
