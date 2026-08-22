"""
Strategy: "tuned HistGradientBoostingClassifier" -- directly targets the
dilution problem diagnosed in the v3 postmortem (see retrain_with_corrections.py
docstring / this project's v3 comparison notes): v3 used correction_weight=5.0
with a 30000/class/image cap and regressed hard (mean IoU ~0.638 vs corrected
GT, down from production's ~0.773). A quick test showed lowering
correction_weight recovered some accuracy but not all of it. This strategy
tries a more moderate weight (2.0, down from 5.0) combined with a BIGGER
per-image correction cap (50000, up from 30000) so corrections still
contribute more total signal (more sampled pixels per image) without
individually overpowering the bootstrap via an extreme per-pixel weight
multiplier -- i.e. push the "more signal" lever on cap instead of on weight.

Also tunes the HGB hyperparameters themselves (deeper trees, more iterations,
slightly lower learning rate) relative to both the production model's
HGB_PARAMS (max_iter=300, max_depth=8, lr=0.1) and v3's, to give the model
more capacity to actually fit the added correction signal instead of
smoothing over it:
    HistGradientBoostingClassifier(max_iter=400, max_depth=10,
                                    learning_rate=0.08,
                                    class_weight='balanced', random_state=0)

Evaluation protocol (identical across all strategies being compared, so
results are comparable):
  1. load_bootstrap_samples(rng), rng = np.random.RandomState(0)
  2. load_correction_samples(correction_weight=2.0, rng, max_per_class_per_image=50000)
     -- same rng instance, called second, per the shared protocol.
  3. Combine exactly like retrain_with_corrections.py's main(): sample_weight
     = compute_sample_weight('balanced', y) * base_weight.
  4. Train the HGB model above.
  5. For each of the 4 ground-truth images, build corrected_gt = external_gt
     with correction==1 forced True / correction==2 forced False, predict
     on the full dense feature array (predict_proba, threshold 0.5, RAW
     pre-postprocess prediction), compute IoU/Dice against corrected_gt.
  6. Save the model to models/pixel_hgb_tuned.joblib. Save one full
     postprocessed (via apply_pixel_model.postprocess_mask, unmodified)
     overlay for the 338_13 image to
     results/strategy_search/tuned_hgb_338_13_overlay.png.
  7. Report training wall-clock time and model file size.

Usage:
    python3 train_strategy_tuned_hgb.py
"""

import glob
import json
import os
import sys
import time

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txm_features import robust_normalize, N_FEATURES  # noqa: F401  (kept for parity with other scripts)
from retrain_with_corrections import load_bootstrap_samples, load_correction_samples
from apply_pixel_model import predict_probability_map, postprocess_mask
import paint_common as pc

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_CACHE_DIR = os.path.join(PROJECT_DIR, "dataset_cache")
CORRECTIONS_DIR = os.path.join(PROJECT_DIR, "paint", "corrections")
MODEL_OUT = os.path.join(PROJECT_DIR, "models", "pixel_hgb_tuned.joblib")
OVERLAY_OUT = os.path.join(PROJECT_DIR, "results", "strategy_search", "tuned_hgb_338_13_overlay.png")
PRODUCTION_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "pixel_hgb_final.joblib")

STRATEGY_NAME = "tuned_hgb"
CORRECTION_WEIGHT = 2.0
MAX_PER_CLASS_PER_IMAGE = 50000
HGB_PARAMS = dict(
    max_iter=400, max_depth=10, learning_rate=0.08,
    class_weight="balanced", random_state=0,
)

# GT image key -> raw filename stem used for paint/corrections/<name>_correction.npy
GT_IMAGES = {
    "333_75_um_zoom": "Average_mosaic_260618_B2_333_75_um_zoom_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "336_25": "Average_mosaic_260618_b2_336_25_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "338_13": "Average_mosaic_260618_b2_338_13_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "LARGE_343_75": "Average_mosaic_260618_b2_343_75_LARGE_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
}


def iou_dice(pred_bool, gt_bool):
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    iou = inter / union if union > 0 else 1.0
    dice = (2 * inter) / (pred_bool.sum() + gt_bool.sum()) if (pred_bool.sum() + gt_bool.sum()) > 0 else 1.0
    return float(iou), float(dice)


def build_corrected_gt(key, corr_name):
    gt = np.load(os.path.join(DATASET_CACHE_DIR, f"{key}_gt.npy")).astype(bool)
    corr_path = os.path.join(CORRECTIONS_DIR, f"{corr_name}_correction.npy")
    corrected = gt.copy()
    if os.path.exists(corr_path):
        correction = np.load(corr_path)
        corrected[correction == 1] = True
        corrected[correction == 2] = False
    else:
        print(f"  WARNING: no correction file found for {key} ({corr_path}); using raw external GT")
    return corrected, (correction if os.path.exists(corr_path) else None)


def main():
    rng = np.random.RandomState(0)

    print("=" * 70)
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"  correction_weight={CORRECTION_WEIGHT}, max_per_class_per_image={MAX_PER_CLASS_PER_IMAGE}")
    print(f"  HGB params: {HGB_PARAMS}")
    print("=" * 70)

    print("\nLoading bootstrapped externally-derived samples (dataset_cache/)...")
    X_boot, y_boot, w_boot = load_bootstrap_samples(rng)

    print("\nLoading human correction samples (paint/corrections/)...")
    X_corr, y_corr, w_corr = load_correction_samples(
        CORRECTION_WEIGHT, rng, max_per_class_per_image=MAX_PER_CLASS_PER_IMAGE
    )

    X = np.concatenate(X_boot + X_corr, axis=0)
    y = np.concatenate(y_boot + y_corr, axis=0)
    base_weight = np.concatenate(w_boot + w_corr, axis=0)

    class_weight = compute_sample_weight("balanced", y)
    sample_weight = class_weight * base_weight

    n_total = len(y)
    n_correction_px = sum(len(w) for w in w_corr)
    print(f"\nTraining on {n_total} pixels total ({n_correction_px} from human corrections, "
          f"weighted {CORRECTION_WEIGHT}x)...")

    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    t0 = time.time()
    clf.fit(X, y, sample_weight=sample_weight)
    train_seconds = time.time() - t0
    print(f"Training took {train_seconds:.2f}s")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    model_size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)
    print(f"Saved model: {MODEL_OUT} ({model_size_mb:.2f} MB)")

    # Load production model once, for the "did anything actually change"
    # sanity check on corrected regions.
    prod_model = None
    if os.path.exists(PRODUCTION_MODEL_PATH):
        prod_model = joblib.load(PRODUCTION_MODEL_PATH)

    print("\nEvaluating against corrected ground truth (external GT + human corrections overlaid)...")
    per_image = []
    ious, dices = [], []
    for key, corr_name in GT_IMAGES.items():
        feats = np.load(os.path.join(DATASET_CACHE_DIR, f"{key}_features.npy"))
        h, w, c = feats.shape
        flat = feats.reshape(-1, c)

        corrected_gt, correction = build_corrected_gt(key, corr_name)

        proba = clf.predict_proba(flat)[:, 1].reshape(h, w)
        pred_bool = proba >= 0.5

        iou, dice = iou_dice(pred_bool, corrected_gt)
        per_image.append({"image": key, "iou": iou, "dice": dice})
        ious.append(iou)
        dices.append(dice)
        print(f"  {key}: IoU={iou:.4f} Dice={dice:.4f}")

        # Sanity check: did the model's prediction actually change from
        # production on the specifically-corrected pixels?
        if prod_model is not None and correction is not None and correction.any():
            prod_proba = prod_model.predict_proba(flat)[:, 1].reshape(h, w)
            prod_pred = prod_proba >= 0.5
            corr_mask = correction != 0
            n_corr = int(corr_mask.sum())
            n_changed = int((pred_bool[corr_mask] != prod_pred[corr_mask]).sum())
            # of the corrected pixels, how many now AGREE with the correction
            # (vs. production's prediction)?
            target = np.zeros_like(corrected_gt)
            target[correction == 1] = True
            target[correction == 2] = False
            n_agree_new = int((pred_bool[corr_mask] == target[corr_mask]).sum())
            n_agree_prod = int((prod_pred[corr_mask] == target[corr_mask]).sum())
            print(f"    [sanity] {n_corr} corrected px: {n_changed} changed vs production "
                  f"({100.0*n_changed/max(n_corr,1):.1f}%); "
                  f"agreement-with-correction new={n_agree_new}/{n_corr} "
                  f"({100.0*n_agree_new/max(n_corr,1):.1f}%) vs "
                  f"production={n_agree_prod}/{n_corr} ({100.0*n_agree_prod/max(n_corr,1):.1f}%)")

    mean_iou = float(np.mean(ious))
    mean_dice = float(np.mean(dices))
    print(f"\nMean IoU={mean_iou:.4f}  Mean Dice={mean_dice:.4f}")

    # Save one full overlay for 338_13, using the UNMODIFIED postprocess_mask.
    print("\nGenerating 338_13 overlay (raw proba -> postprocess_mask, unmodified)...")
    img01_338 = np.load(os.path.join(DATASET_CACHE_DIR, "338_13_img.npy"))
    feats_338 = np.load(os.path.join(DATASET_CACHE_DIR, "338_13_features.npy"))
    h, w, c = feats_338.shape
    proba_338 = clf.predict_proba(feats_338.reshape(-1, c))[:, 1].reshape(h, w)
    final_mask_338 = postprocess_mask(proba_338)

    os.makedirs(os.path.dirname(OVERLAY_OUT), exist_ok=True)
    from PIL import Image
    gray = (np.clip(img01_338, 0, 1) * 255).astype(np.uint8)
    overlay = np.stack([gray, gray, gray], axis=-1)
    overlay[final_mask_338] = [255, 0, 0]
    Image.fromarray(overlay, mode="RGB").save(OVERLAY_OUT)
    print(f"Saved overlay: {OVERLAY_OUT}")

    result = {
        "strategy_name": STRATEGY_NAME,
        "config": f"HGB(max_iter=400,max_depth=10,lr=0.08,class_weight=balanced) "
                   f"correction_weight={CORRECTION_WEIGHT} max_per_class_per_image={MAX_PER_CLASS_PER_IMAGE}",
        "per_image": per_image,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "train_seconds": train_seconds,
        "model_path": MODEL_OUT,
        "model_size_mb": model_size_mb,
        "overlay_path": OVERLAY_OUT,
    }
    result_path = os.path.join(PROJECT_DIR, "results", "strategy_search", f"{STRATEGY_NAME}_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result summary: {result_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
