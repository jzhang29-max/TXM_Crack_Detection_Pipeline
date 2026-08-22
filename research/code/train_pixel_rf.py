"""
Train + evaluate a per-pixel RandomForestClassifier for TXM crack detection.

This is the pixel-classifier fix for the failed region-candidate-classifier
approach (see txm_features.py docstring for the full story): instead of
accepting/rejecting Otsu-found candidate regions (which structurally can't
recover the ~18-31% true crack area from an initial mask that only covers
~1.1-1.3%), we train a classifier directly on a 17-feature-per-pixel stack
against externally-derived ground truth, and evaluate honestly with
leave-one-image-out (LOIO) cross-validation across the 4 labeled images.

Usage:
    python train_pixel_rf.py

Reads the cached feature/gt arrays from ../dataset_cache (see manifest.json),
runs 4-fold LOIO evaluation for two RandomForest hyperparameter configs,
picks whichever gets the better mean IoU, refits that config on all 4
images combined, and saves the final model to ../models/pixel_rf.joblib.
"""

import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from txm_features import FEATURE_NAMES, N_FEATURES

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_cache")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
MODEL_PATH = os.path.join(MODEL_DIR, "pixel_rf.joblib")

N_PER_CLASS = 30_000  # up to 30k crack + 30k background pixels per training image
RNG_SEED = 0

CONFIGS = {
    "config1_deep_unbounded": dict(
        n_estimators=300, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=0
    ),
    "config2_wide_capped": dict(
        n_estimators=500, max_depth=20, class_weight="balanced", n_jobs=-1, random_state=0
    ),
}


def load_manifest():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    assert manifest["feature_names"] == FEATURE_NAMES, "feature order mismatch vs txm_features.py"
    return manifest["images"]


def load_image_arrays(entry):
    feats = np.load(entry["feat_path"])
    gt = np.load(entry["gt_path"])
    return feats, gt


def sample_balanced_pixels(feats, gt, n_per_class, rng):
    """Return (X, y) sampled from a single image: up to n_per_class crack
    pixels (gt==True) and up to n_per_class background pixels (gt==False)."""
    h, w, c = feats.shape
    flat_gt = gt.reshape(-1)
    flat_feats = feats.reshape(-1, c)

    pos_idx = np.flatnonzero(flat_gt)
    neg_idx = np.flatnonzero(~flat_gt)

    n_pos = min(n_per_class, pos_idx.size)
    n_neg = min(n_per_class, neg_idx.size)

    pos_sel = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_sel = rng.choice(neg_idx, size=n_neg, replace=False)

    sel = np.concatenate([pos_sel, neg_sel])
    X = flat_feats[sel]
    y = flat_gt[sel]
    return X, y


def build_training_set(entries, n_per_class):
    """Sample balanced pixels from each of `entries` using a fresh
    RandomState(seed=0) per image (per protocol: seed=0 sampling)."""
    Xs, ys = [], []
    for entry in entries:
        feats, gt = load_image_arrays(entry)
        rng = np.random.RandomState(RNG_SEED)
        X, y = sample_balanced_pixels(feats, gt, n_per_class, rng)
        Xs.append(X)
        ys.append(y)
        del feats, gt
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def evaluate_on_full_image(clf, entry):
    feats, gt = load_image_arrays(entry)
    h, w, c = feats.shape
    flat_feats = feats.reshape(-1, c)

    t0 = time.time()
    proba = clf.predict_proba(flat_feats)
    # class order follows clf.classes_; find index of the True (crack) class
    crack_col = list(clf.classes_).index(True)
    prob_crack = proba[:, crack_col].reshape(h, w)
    predict_time = time.time() - t0

    pred = prob_crack >= 0.5

    inter = np.logical_and(pred, gt).sum(dtype=np.int64)
    union = np.logical_or(pred, gt).sum(dtype=np.int64)
    pred_sum = pred.sum(dtype=np.int64)
    gt_sum = gt.sum(dtype=np.int64)

    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    precision = inter / pred_sum if pred_sum > 0 else 0.0
    recall = inter / gt_sum if gt_sum > 0 else 0.0

    del feats, gt, flat_feats, proba, prob_crack, pred
    return dict(iou=float(iou), dice=float(dice), precision=float(precision),
                recall=float(recall), predict_seconds=predict_time)


def run_loio(images, rf_kwargs, label):
    """4-fold leave-one-image-out evaluation for one hyperparameter config."""
    folds = []
    for i, held_out in enumerate(images):
        train_entries = [e for j, e in enumerate(images) if j != i]

        t0 = time.time()
        X_train, y_train = build_training_set(train_entries, N_PER_CLASS)
        clf = RandomForestClassifier(**rf_kwargs)
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        metrics = evaluate_on_full_image(clf, held_out)
        metrics["held_out_image"] = held_out["name"]
        metrics["train_seconds"] = train_time
        metrics["n_train_pixels"] = int(X_train.shape[0])
        folds.append(metrics)

        print(
            f"[{label}] fold held_out={held_out['name']:16s} "
            f"n_train={X_train.shape[0]:7d} train_s={train_time:6.1f} "
            f"predict_s={metrics['predict_seconds']:6.1f} "
            f"IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}",
            flush=True,
        )
        del X_train, y_train, clf

    mean_iou = float(np.mean([f["iou"] for f in folds]))
    mean_dice = float(np.mean([f["dice"] for f in folds]))
    mean_precision = float(np.mean([f["precision"] for f in folds]))
    mean_recall = float(np.mean([f["recall"] for f in folds]))
    print(
        f"[{label}] MEAN  IoU={mean_iou:.4f} Dice={mean_dice:.4f} "
        f"P={mean_precision:.4f} R={mean_recall:.4f}",
        flush=True,
    )
    return folds, mean_iou, mean_dice, mean_precision, mean_recall


def main():
    import joblib

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    images = load_manifest()
    print(f"Loaded manifest: {[im['name'] for im in images]}", flush=True)

    all_results = {}
    for label, kwargs in CONFIGS.items():
        print(f"\n===== Running LOIO for {label}: {kwargs} =====", flush=True)
        folds, mean_iou, mean_dice, mean_precision, mean_recall = run_loio(images, kwargs, label)
        all_results[label] = dict(
            kwargs=kwargs, folds=folds, mean_iou=mean_iou, mean_dice=mean_dice,
            mean_precision=mean_precision, mean_recall=mean_recall,
        )

    best_label = max(all_results, key=lambda k: all_results[k]["mean_iou"])
    best = all_results[best_label]
    print(f"\n===== BEST CONFIG: {best_label} (mean IoU={best['mean_iou']:.4f}) =====", flush=True)

    # Save the full LOIO comparison for the record.
    with open(os.path.join(RESULTS_DIR, "pixel_rf_loio_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Fit the FINAL model on all 4 images combined (same sampling recipe).
    print("\nFitting final model on all 4 images combined...", flush=True)
    t0 = time.time()
    X_final, y_final = build_training_set(images, N_PER_CLASS)
    final_clf = RandomForestClassifier(**CONFIGS[best_label])
    final_clf.fit(X_final, y_final)
    final_train_time = time.time() - t0
    print(
        f"Final model fit on {X_final.shape[0]} pixels in {final_train_time:.1f}s",
        flush=True,
    )

    joblib.dump(final_clf, MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}", flush=True)

    summary = dict(
        variant_name=f"RandomForestClassifier ({best_label})",
        model_path=MODEL_PATH,
        hyperparams=str(CONFIGS[best_label]),
        best_config_label=best_label,
        all_configs=CONFIGS,
        folds=[
            dict(
                held_out_image=f["held_out_image"],
                iou=f["iou"],
                dice=f["dice"],
                precision=f["precision"],
                recall=f["recall"],
            )
            for f in best["folds"]
        ],
        mean_iou=best["mean_iou"],
        mean_dice=best["mean_dice"],
        mean_precision=best["mean_precision"],
        mean_recall=best["mean_recall"],
        final_train_seconds=final_train_time,
        final_n_train_pixels=int(X_final.shape[0]),
    )
    with open(os.path.join(RESULTS_DIR, "pixel_rf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
