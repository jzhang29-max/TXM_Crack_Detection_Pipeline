"""
Train + evaluate a per-pixel HistGradientBoostingClassifier (scikit-learn)
crack detector for the TXM dataset, using the cached 17-feature-per-pixel
stack in dataset_cache/ (see txm_features.py for what the features mean).

Why HistGradientBoostingClassifier: it is sklearn's fast histogram-based
gradient boosting implementation (LightGBM-style binning), which scales far
better than plain RandomForest/GradientBoosting to the ~23.5M-pixel full
prediction pass needed for the LARGE_343_75 held-out fold, while still being
expressive enough to exploit the large-radius smoothed-intensity features
that a shallow model / single global threshold cannot.

Protocol (must match the other variants in this project so results are
apples-to-apples -- see project README / task description):
  - Leave-one-image-out (LOIO) across the 4 cached images, 4 folds.
  - Per fold, train on a class-balanced random sample (RandomState seed=0)
    of up to 30,000 crack + 30,000 background pixels from EACH of the 3
    non-held-out images (up to 180,000 pixels/fold total).
  - Predict on every pixel of the held-out image (no subsampling of eval).
  - Threshold the predicted probability at 0.5 for IoU/Dice/Precision/Recall
    against that image's real cached ground-truth mask.
  - Try >=2 hyperparameter configs, keep whichever gets the better mean IoU.
  - After LOIO, fit one FINAL model on all 4 images' sampled pixels combined
    (same per-image sampling recipe) and joblib.dump it -- this final model,
    not any fold model, is what the rest of the pipeline actually uses.

Usage:
    python3 train_pixel_hgb.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txm_features import FEATURE_NAMES, N_FEATURES

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_DIR, "dataset_cache")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "pixel_hgb_final.joblib")
RESULTS_PATH = os.path.join(PROJECT_DIR, "results", "pixel_hgb_results.json")

N_PER_CLASS_PER_IMAGE = 30_000
SAMPLE_SEED = 0

HGB_CONFIGS = {
    "config1_shallow_fast": dict(
        max_iter=300,
        max_depth=8,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=0,
        early_stopping=False,
    ),
    "config2_deep_slow": dict(
        max_iter=500,
        max_depth=None,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=0,
        early_stopping=False,
    ),
}


def load_manifest():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    assert manifest["feature_names"] == FEATURE_NAMES, (
        "Cached feature order does not match txm_features.FEATURE_NAMES -- "
        "dataset_cache is stale, rebuild it with build_dataset.py."
    )
    return manifest


def load_image_arrays(entry):
    feats = np.load(entry["feat_path"])
    gt = np.load(entry["gt_path"])
    assert feats.shape[:2] == gt.shape
    assert feats.shape[2] == N_FEATURES
    return feats, gt


def sample_balanced_pixels(feats, gt, n_per_class, seed):
    """Return (X, y) sampled from up to n_per_class crack and n_per_class
    background pixels of this single image, using a fresh RandomState(seed)
    (so every image draws the same fixed sample given the same seed, per
    the project protocol)."""
    rng = np.random.RandomState(seed)
    h, w = gt.shape
    flat_gt = gt.reshape(-1)
    crack_idx = np.flatnonzero(flat_gt)
    bg_idx = np.flatnonzero(~flat_gt)

    n_crack = min(n_per_class, crack_idx.size)
    n_bg = min(n_per_class, bg_idx.size)
    sel_crack = rng.choice(crack_idx, size=n_crack, replace=False)
    sel_bg = rng.choice(bg_idx, size=n_bg, replace=False)
    sel = np.concatenate([sel_crack, sel_bg])

    flat_feats = feats.reshape(-1, feats.shape[2])
    X = flat_feats[sel]
    y = flat_gt[sel]
    return X, y


def build_training_set(entries, n_per_class, seed):
    Xs, ys = [], []
    for entry in entries:
        feats, gt = load_image_arrays(entry)
        X, y = sample_balanced_pixels(feats, gt, n_per_class, seed)
        Xs.append(X)
        ys.append(y)
        del feats, gt
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def predict_full_image(clf, feats):
    h, w, c = feats.shape
    flat = feats.reshape(-1, c)
    proba = clf.predict_proba(flat)[:, 1]
    return proba.reshape(h, w)


def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum(dtype=np.int64)
    union = np.logical_or(pred, gt).sum(dtype=np.int64)
    pred_sum = pred.sum(dtype=np.int64)
    gt_sum = gt.sum(dtype=np.int64)

    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    precision = inter / pred_sum if pred_sum > 0 else 0.0
    recall = inter / gt_sum if gt_sum > 0 else 0.0
    return dict(iou=float(iou), dice=float(dice), precision=float(precision), recall=float(recall))


def run_loio(manifest, hgb_params, config_name):
    entries = manifest["images"]
    fold_results = []
    for held_out in entries:
        train_entries = [e for e in entries if e["name"] != held_out["name"]]

        t0 = time.time()
        X_train, y_train = build_training_set(train_entries, N_PER_CLASS_PER_IMAGE, SAMPLE_SEED)
        t1 = time.time()

        clf = HistGradientBoostingClassifier(**hgb_params)
        clf.fit(X_train, y_train)
        t2 = time.time()

        held_feats, held_gt = load_image_arrays(held_out)
        proba = predict_full_image(clf, held_feats)
        pred_mask = proba >= 0.5
        t3 = time.time()

        metrics = compute_metrics(pred_mask, held_gt)
        metrics["held_out_image"] = held_out["name"]
        fold_results.append(metrics)

        print(
            f"[{config_name}] held_out={held_out['name']:16s} "
            f"n_train={X_train.shape[0]:7d} "
            f"sample_t={t1 - t0:5.1f}s fit_t={t2 - t1:5.1f}s pred_t={t3 - t2:5.1f}s "
            f"IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}",
            flush=True,
        )

        del held_feats, held_gt, proba, pred_mask, clf, X_train, y_train

    mean_metrics = {
        k: float(np.mean([f[k] for f in fold_results]))
        for k in ("iou", "dice", "precision", "recall")
    }
    return fold_results, mean_metrics


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    manifest = load_manifest()

    all_config_results = {}
    for config_name, hgb_params in HGB_CONFIGS.items():
        print(f"\n=== Running LOIO for {config_name}: {hgb_params} ===", flush=True)
        fold_results, mean_metrics = run_loio(manifest, hgb_params, config_name)
        all_config_results[config_name] = dict(
            hgb_params=hgb_params, fold_results=fold_results, mean_metrics=mean_metrics
        )
        print(f"[{config_name}] MEAN over {len(fold_results)} folds: {mean_metrics}", flush=True)

    best_config_name = max(
        all_config_results, key=lambda name: all_config_results[name]["mean_metrics"]["iou"]
    )
    best = all_config_results[best_config_name]
    print(f"\n=== Best config: {best_config_name} (mean IoU = {best['mean_metrics']['iou']:.4f}) ===")

    # Final model: fit on ALL 4 images' sampled pixels combined, using the
    # winning hyperparameters.
    print("\n=== Fitting FINAL model on all 4 images combined ===", flush=True)
    t0 = time.time()
    X_all, y_all = build_training_set(manifest["images"], N_PER_CLASS_PER_IMAGE, SAMPLE_SEED)
    final_clf = HistGradientBoostingClassifier(**best["hgb_params"])
    final_clf.fit(X_all, y_all)
    t1 = time.time()
    print(f"Final model fit on {X_all.shape[0]} pixels in {t1 - t0:.1f}s", flush=True)

    joblib.dump(final_clf, MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}", flush=True)

    output = dict(
        variant_name=f"pixel_hgb_{best_config_name}",
        model_path=MODEL_PATH,
        best_config_name=best_config_name,
        all_configs=all_config_results,
        final_model_n_train_pixels=int(X_all.shape[0]),
        final_model_fit_seconds=t1 - t0,
    )
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results JSON to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
