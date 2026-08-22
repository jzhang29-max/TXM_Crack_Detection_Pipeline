"""
Strategy search: MLPClassifier (sklearn feed-forward neural net) -- a
genuinely different model family from every tree ensemble tried so far
(pixel_hgb_final.joblib, pixel_hgb_v2/v3.joblib, pixel_rf.joblib,
pixel_extratrees.joblib). Trees produce axis-aligned, sometimes-discontinuous
decision surfaces in feature space; a smooth neural network function may be
less prone to isolated single-pixel confidence flips on its own, independent
of the hysteresis postprocessing fix already merged into apply_pixel_model.py.

Training signal is identical in spirit to retrain_with_corrections.py's
main(): the 4-image external bootstrap sample plus every human correction
across all 12 images, capped per-class-per-image so no single image's
correction volume can numerically drown out the rest.

MLPClassifier needs feature scaling (StandardScaler) and does not accept a
`sample_weight` argument to .fit(), unlike the HGB/RF models used elsewhere
in this project. We emulate the same "correction pixels count more" idea by
oversampling instead: every row is repeated round(sample_weight) times,
where sample_weight = compute_sample_weight('balanced', y) * base_weight
(base_weight = 1.0 for bootstrap rows, correction_weight for correction
rows) -- exactly the same weight computation retrain_with_corrections.py's
main() uses, just converted from a continuous per-row weight into an
integer repeat count. In practice the capped-sampling recipe already yields
an exactly class-balanced y (600000 crack / 600000 background pixels for
this strategy's settings), so compute_sample_weight('balanced') is ~1.0
everywhere and this reduces to the intuitive "bootstrap rows x1, correction
rows x3" the task asked for.

Usage:
    python3 code/train_strategy_mlp.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_pixel_model import postprocess_mask  # noqa: E402  (reuse as-is, do not reimplement)
from retrain_with_corrections import load_bootstrap_samples, load_correction_samples  # noqa: E402
import paint_common as pc  # noqa: E402

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_CACHE_DIR = os.path.join(PROJECT_DIR, "dataset_cache")
MODEL_OUT = os.path.join(PROJECT_DIR, "models", "pixel_mlp.joblib")
OVERLAY_OUT_DIR = os.path.join(PROJECT_DIR, "results", "strategy_search")
STRATEGY_LABEL = "mlp"

CORRECTION_WEIGHT = 3.0
MAX_PER_CLASS_PER_IMAGE = 40000

MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=300,
    early_stopping=True, random_state=0,
)

# key -> raw filename (for locating paint/corrections/<raw>_correction.npy)
GT_IMAGES = {
    "333_75_um_zoom": "Average_mosaic_260618_B2_333_75_um_zoom_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "336_25": "Average_mosaic_260618_b2_336_25_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "338_13": "Average_mosaic_260618_b2_338_13_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "LARGE_343_75": "Average_mosaic_260618_b2_343_75_LARGE_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
}


def build_training_set():
    rng = np.random.RandomState(0)

    print("Loading bootstrapped externally-derived samples (dataset_cache/)...")
    X_boot, y_boot, w_boot = load_bootstrap_samples(rng)

    print("Loading human correction samples (paint/corrections/)...")
    X_corr, y_corr, w_corr = load_correction_samples(
        CORRECTION_WEIGHT, rng, MAX_PER_CLASS_PER_IMAGE
    )

    X = np.concatenate(X_boot + X_corr, axis=0)
    y = np.concatenate(y_boot + y_corr, axis=0)
    base_weight = np.concatenate(w_boot + w_corr, axis=0)

    class_weight = compute_sample_weight("balanced", y)
    sample_weight = class_weight * base_weight

    print(f"Combined (pre-oversample): {len(y)} rows, crack frac={y.mean():.4f}")
    print(f"sample_weight range: min={sample_weight.min():.4f} max={sample_weight.max():.4f}")

    repeat_counts = np.maximum(1, np.round(sample_weight)).astype(np.int64)
    X_rep = np.repeat(X, repeat_counts, axis=0)
    y_rep = np.repeat(y, repeat_counts, axis=0)
    print(f"After weight-proportional oversampling: {len(y_rep)} rows "
          f"(from {len(y)} rows pre-oversample)")

    return X_rep, y_rep


def build_corrected_gt(key):
    """original external gt, overridden by the human correction wherever one
    exists: correction==1 forces True (crack), correction==2 forces False."""
    gt = np.load(os.path.join(DATASET_CACHE_DIR, f"{key}_gt.npy")).astype(bool)
    raw_name = GT_IMAGES[key]
    corr_path = os.path.join(pc.CORRECTIONS_DIR, f"{raw_name}_correction.npy")
    corrected = gt.copy()
    correction = None
    if os.path.exists(corr_path):
        correction = np.load(corr_path)
        corrected[correction == 1] = True
        corrected[correction == 2] = False
    else:
        print(f"WARNING: no correction file found for {key} at {corr_path}")
    return corrected, correction


def predict_proba_map(bundle, feats):
    h, w, c = feats.shape
    flat = feats.reshape(-1, c)
    flat_scaled = bundle["scaler"].transform(flat)
    proba = bundle["model"].predict_proba(flat_scaled)[:, 1]
    return proba.reshape(h, w)


def iou_dice(pred_mask, gt_mask):
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    iou = inter / union if union > 0 else 1.0
    denom = int(pred_mask.sum()) + int(gt_mask.sum())
    dice = (2 * inter / denom) if denom > 0 else 1.0
    return float(iou), float(dice)


def main():
    os.makedirs(OVERLAY_OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    X, y = build_training_set()

    print("Fitting StandardScaler on training X...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Training MLPClassifier on {X_scaled.shape[0]} rows x {X_scaled.shape[1]} features "
          f"({MLP_PARAMS})...")
    clf = MLPClassifier(**MLP_PARAMS)
    t0 = time.time()
    clf.fit(X_scaled, y)
    train_seconds = time.time() - t0
    print(f"Training done in {train_seconds:.1f}s, n_iter_={clf.n_iter_}, "
          f"loss_={clf.loss_:.5f}")

    bundle = {"model": clf, "scaler": scaler}
    joblib.dump(bundle, MODEL_OUT)
    model_size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)
    print(f"Saved model bundle: {MODEL_OUT} ({model_size_mb:.2f} MB)")

    # Production model, loaded once, for the "did anything actually change in
    # the corrected regions" sanity check per image.
    prod_model = joblib.load(os.path.join(PROJECT_DIR, "models", "pixel_hgb_final.joblib"))

    per_image = []
    for key in ["333_75_um_zoom", "336_25", "338_13", "LARGE_343_75"]:
        feats = np.load(os.path.join(DATASET_CACHE_DIR, f"{key}_features.npy"))
        corrected_gt, correction = build_corrected_gt(key)

        proba = predict_proba_map(bundle, feats)
        pred_mask = proba >= 0.5

        iou, dice = iou_dice(pred_mask, corrected_gt)
        per_image.append({"image": key, "iou": iou, "dice": dice})
        print(f"{key}: IoU={iou:.4f} Dice={dice:.4f}")

        if correction is not None:
            flat = feats.reshape(-1, feats.shape[-1])
            prod_proba = prod_model.predict_proba(flat)[:, 1].reshape(correction.shape)
            prod_mask = prod_proba >= 0.5
            corr_bool = correction != 0
            n_corr_px = int(corr_bool.sum())
            n_changed = int(np.logical_xor(pred_mask, prod_mask)[corr_bool].sum())
            frac_changed = n_changed / n_corr_px if n_corr_px else 0.0
            print(f"  sanity check: {n_changed}/{n_corr_px} ({frac_changed:.2%}) of "
                  f"human-corrected pixels differ from the production model's prediction")

        if key == "338_13":
            img01 = np.load(os.path.join(DATASET_CACHE_DIR, f"{key}_img.npy"))
            final_mask = postprocess_mask(proba)
            gray = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
            overlay = np.stack([gray, gray, gray], axis=-1)
            overlay[final_mask] = [255, 0, 0]
            overlay_path = os.path.join(OVERLAY_OUT_DIR, f"{STRATEGY_LABEL}_338_13_overlay.png")
            Image.fromarray(overlay, mode="RGB").save(overlay_path)
            print(f"Saved overlay: {overlay_path}")

    mean_iou = float(np.mean([r["iou"] for r in per_image]))
    mean_dice = float(np.mean([r["dice"] for r in per_image]))

    summary = {
        "strategy_name": "mlp",
        "config": (
            "MLPClassifier(hidden_layer_sizes=(64,32), alpha=1e-4, max_iter=300, "
            "early_stopping=True, random_state=0) + StandardScaler; "
            f"correction_weight={CORRECTION_WEIGHT}, max_per_class_per_image={MAX_PER_CLASS_PER_IMAGE}; "
            "sample_weight emulated via round()-based row oversampling (no native "
            "sample_weight support in MLPClassifier.fit)"
        ),
        "per_image": per_image,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "train_seconds": train_seconds,
        "model_path": MODEL_OUT,
        "model_size_mb": model_size_mb,
        "overlay_path": os.path.join(OVERLAY_OUT_DIR, f"{STRATEGY_LABEL}_338_13_overlay.png"),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(OVERLAY_OUT_DIR, f"{STRATEGY_LABEL}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
