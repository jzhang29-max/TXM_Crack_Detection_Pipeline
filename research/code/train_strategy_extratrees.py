"""
Strategy-search candidate: ExtraTreesClassifier, trained on the bootstrapped
Ilastik samples PLUS the human paint-tool corrections from all 12 opened
images, using the corrected-sampling recipe in retrain_with_corrections.py
(load_bootstrap_samples / load_correction_samples) -- i.e. this is the
"ExtraTrees, but with the capping fix that avoided the LARGE-vignetting
regression seen in the very first, uncapped comparison" run.

This script is intentionally standalone (does not touch models/pixel_hgb_v3
or pixel_hgb_final) so it can be compared side-by-side against both.

Evaluation is against a *corrected* ground truth per image: the original
Ilastik dataset_cache/<key>_gt.npy, overridden pixel-by-pixel wherever the
user's paint/corrections/<raw_name>_correction.npy says so (1 -> force
True/crack, 2 -> force False/not-crack). This is the only fair comparison
once corrections exist for an image -- see project background.

Usage:
    python3 train_strategy_extratrees.py
"""

import glob
import os
import sys
import time

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrain_with_corrections import load_bootstrap_samples, load_correction_samples
from apply_pixel_model import postprocess_mask
from txm_features import N_FEATURES

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_CACHE_DIR = os.path.join(PROJECT_DIR, "dataset_cache")
CORRECTIONS_DIR = os.path.join(PROJECT_DIR, "paint", "corrections")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "strategy_search")

STRATEGY_NAME = "extratrees_v2_capped_corrections"
MODEL_OUT = os.path.join(PROJECT_DIR, "models", "pixel_extratrees_v2.joblib")
OVERLAY_IMAGE_KEY = "338_13"
OVERLAY_OUT = os.path.join(RESULTS_DIR, "extratrees_v2_338_13_overlay.png")

CORRECTION_WEIGHT = 3.0
MAX_PER_CLASS_PER_IMAGE = 40000

ET_PARAMS = dict(
    n_estimators=300, max_depth=16, min_samples_leaf=3,
    class_weight="balanced", n_jobs=-1, random_state=0,
)

# dataset_cache key -> raw filename stem used in paint/corrections/*_correction.npy
KEY_TO_RAW_STEM = {
    "333_75_um_zoom": "Average_mosaic_260618_B2_333_75_um_zoom_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "336_25": "Average_mosaic_260618_b2_336_25_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "338_13": "Average_mosaic_260618_b2_338_13_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "LARGE_343_75": "Average_mosaic_260618_b2_343_75_LARGE_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
}


def build_corrected_gt(key):
    gt = np.load(os.path.join(DATASET_CACHE_DIR, f"{key}_gt.npy")).astype(bool)
    corr_path = os.path.join(CORRECTIONS_DIR, f"{KEY_TO_RAW_STEM[key]}_correction.npy")
    corrected = gt.copy()
    n_forced_crack = n_forced_bg = 0
    if os.path.exists(corr_path):
        correction = np.load(corr_path)
        assert correction.shape == gt.shape, f"{key}: correction shape {correction.shape} != gt shape {gt.shape}"
        forced_crack = correction == 1
        forced_bg = correction == 2
        n_forced_crack = int(forced_crack.sum())
        n_forced_bg = int(forced_bg.sum())
        corrected[forced_crack] = True
        corrected[forced_bg] = False
    else:
        print(f"  WARNING: no correction file found for {key} at {corr_path}")
    return corrected, n_forced_crack, n_forced_bg


def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.reshape(-1)
    gt = gt_mask.reshape(-1)
    inter = np.count_nonzero(pred & gt)
    union = np.count_nonzero(pred | gt)
    pred_sum = np.count_nonzero(pred)
    gt_sum = np.count_nonzero(gt)
    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    return iou, dice


def predict_full_image(clf, feats, chunk=2_000_000):
    h, w, f = feats.shape
    flat = feats.reshape(-1, f)
    pos_idx = list(clf.classes_).index(True)
    proba = np.empty(flat.shape[0], dtype=np.float32)
    for start in range(0, flat.shape[0], chunk):
        end = min(start + chunk, flat.shape[0])
        proba[start:end] = clf.predict_proba(flat[start:end])[:, pos_idx]
    return proba.reshape(h, w)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    rng = np.random.RandomState(0)

    print("Loading bootstrapped Ilastik-derived samples (dataset_cache/)...")
    X_boot, y_boot, w_boot = load_bootstrap_samples(rng)

    print(f"Loading human correction samples (weight={CORRECTION_WEIGHT}, "
          f"cap={MAX_PER_CLASS_PER_IMAGE}/class/image)...")
    X_corr, y_corr, w_corr = load_correction_samples(
        CORRECTION_WEIGHT, rng, MAX_PER_CLASS_PER_IMAGE
    )

    X = np.concatenate(X_boot + X_corr, axis=0)
    y = np.concatenate(y_boot + y_corr, axis=0)
    base_weight = np.concatenate(w_boot + w_corr, axis=0)

    class_weight = compute_sample_weight("balanced", y)
    sample_weight = class_weight * base_weight

    n_total = len(y)
    n_correction_px = sum(len(w) for w in w_corr)
    print(f"\nTraining on {n_total} pixels total ({n_correction_px} from human "
          f"corrections, weighted {CORRECTION_WEIGHT}x)...")
    print(f"ExtraTrees params: {ET_PARAMS}")

    t0 = time.time()
    clf = ExtraTreesClassifier(**ET_PARAMS)
    clf.fit(X, y, sample_weight=sample_weight)
    train_seconds = time.time() - t0
    print(f"Training done in {train_seconds:.1f}s")

    joblib.dump(clf, MODEL_OUT)
    model_size_mb = os.path.getsize(MODEL_OUT) / 1e6
    print(f"Saved model to {MODEL_OUT} ({model_size_mb:.1f} MB)")

    # Also load the current production model, to sanity-check that our
    # predictions actually DIFFER from it on corrected regions (i.e. we
    # aren't just reproducing production and ignoring the corrections).
    prod_model_path = os.path.join(PROJECT_DIR, "models", "pixel_hgb_final.joblib")
    prod_model = joblib.load(prod_model_path)

    per_image = []
    keys = ["333_75_um_zoom", "336_25", "338_13", "LARGE_343_75"]
    for key in keys:
        print(f"\n=== Evaluating on {key} ===")
        feats = np.load(os.path.join(DATASET_CACHE_DIR, f"{key}_features.npy"))
        corrected_gt, n_forced_crack, n_forced_bg = build_corrected_gt(key)
        print(f"  corrected_gt built (forced-crack={n_forced_crack}, forced-bg={n_forced_bg})")

        t1 = time.time()
        proba = predict_full_image(clf, feats)
        pred_mask = proba >= 0.5
        pred_seconds = time.time() - t1
        print(f"  predicted full image ({feats.shape[0]}x{feats.shape[1]}) in {pred_seconds:.1f}s")

        iou, dice = compute_metrics(pred_mask, corrected_gt)
        print(f"  IoU={iou:.4f} Dice={dice:.4f}")

        # Sanity check: does the prediction actually differ from production
        # in the specifically-corrected pixels for this image?
        corr_path = os.path.join(CORRECTIONS_DIR, f"{KEY_TO_RAW_STEM[key]}_correction.npy")
        if os.path.exists(corr_path):
            correction = np.load(corr_path)
            corrected_pixels = correction != 0
            if corrected_pixels.any():
                prod_proba = predict_full_image(prod_model, feats)
                prod_mask = prod_proba >= 0.5
                n_changed = int(np.count_nonzero(
                    (pred_mask != prod_mask) & corrected_pixels
                ))
                frac_changed = n_changed / int(corrected_pixels.sum())
                print(f"  sanity check: {n_changed}/{int(corrected_pixels.sum())} "
                      f"({frac_changed:.1%}) corrected pixels changed vs production model")
                del prod_proba, prod_mask

        per_image.append(dict(image=key, iou=float(iou), dice=float(dice)))

        if key == OVERLAY_IMAGE_KEY:
            img01 = np.load(os.path.join(DATASET_CACHE_DIR, f"{key}_img.npy"))
            final_mask = postprocess_mask(proba)
            gray = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
            overlay = np.stack([gray, gray, gray], axis=-1)
            overlay[final_mask] = [255, 0, 0]
            Image.fromarray(overlay, mode="RGB").save(OVERLAY_OUT)
            print(f"  saved overlay: {OVERLAY_OUT}")

        del feats, proba, pred_mask, corrected_gt

    mean_iou = float(np.mean([p["iou"] for p in per_image]))
    mean_dice = float(np.mean([p["dice"] for p in per_image]))

    print("\n=== SUMMARY ===")
    print(f"strategy: {STRATEGY_NAME}")
    for p in per_image:
        print(f"  {p['image']}: IoU={p['iou']:.4f} Dice={p['dice']:.4f}")
    print(f"mean IoU={mean_iou:.4f} mean Dice={mean_dice:.4f}")
    print(f"train_seconds={train_seconds:.1f} model_size_mb={model_size_mb:.2f}")
    print(f"model saved: {MODEL_OUT}")
    print(f"overlay saved: {OVERLAY_OUT}")

    import json
    summary_path = os.path.join(RESULTS_DIR, "extratrees_v2_results.json")
    with open(summary_path, "w") as f:
        json.dump(dict(
            strategy_name=STRATEGY_NAME,
            config=dict(correction_weight=CORRECTION_WEIGHT,
                        max_per_class_per_image=MAX_PER_CLASS_PER_IMAGE,
                        et_params={k: (v if not callable(v) else str(v)) for k, v in ET_PARAMS.items()}),
            per_image=per_image,
            mean_iou=mean_iou,
            mean_dice=mean_dice,
            train_seconds=train_seconds,
            model_path=MODEL_OUT,
            model_size_mb=model_size_mb,
            overlay_path=OVERLAY_OUT,
        ), f, indent=2)
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
