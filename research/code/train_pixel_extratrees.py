"""
ExtraTreesClassifier pixel-classifier variant for TXM crack detection.

Rationale for this variant (see PROJECT CONTEXT / dataset_cache/manifest.json
and txm_features.py docstring for the full story): the earlier region-
candidate-classifier approach failed because it could only accept/reject
Otsu-derived candidate regions that already covered ~1-1.3% of the image,
against a true crack extent of 18-31% (per external reference). Skipping
region proposal entirely and training a genuine per-pixel classifier on a
17-feature multi-scale stack (raw intensity, Gaussian smoothing at sigma =
2..64, gradient magnitude, Laplacian, local texture) lets the model use the
same "is this pixel embedded in a broad dark region" signal a human would,
at every pixel, not just inside a handful of pre-selected regions.

Why ExtraTrees specifically (vs RandomForest/GradientBoosting variants
tried elsewhere in this project): with 17 features drawn from heavily
overlapping/correlated smoothing radii (e.g. smooth_s16 vs smooth_s32 are
strongly correlated), ExtraTrees' fully-randomized split-threshold selection
(instead of searching for the locally optimal split like RandomForest does)
tends to decorrelate trees more and can be more robust in this regime, at
the cost of slightly higher bias per tree -- hence trying both a deep/
unconstrained config and a depth/leaf-regularized config below.

Evaluation protocol: strict leave-one-image-out (LOIO) across the 4 cached
images. For each fold, train on a balanced random sample (<=30k crack +
<=30k background pixels per non-held-out image, seed=0) from the 3 other
images, and predict on *every* pixel of the held-out image (no subsampling
at eval time -- that would make the IoU dishonest). Metrics are computed at
the standard 0.5 probability threshold. After LOIO, a final model is fit on
all 4 images' sampled pixels combined and saved for downstream use.
"""

import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txm_features import FEATURE_NAMES, N_FEATURES  # noqa: F401  (kept for reference/debug)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_cache")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "pixel_extratrees_final.joblib")

MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
SAMPLES_PER_CLASS = 30_000
SEED = 0

CONFIGS = {
    "config1_deep_unconstrained": dict(
        n_estimators=400, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=0,
    ),
    "config2_depth25_leaf5": dict(
        n_estimators=400, max_depth=25, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0,
    ),
}


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_image_arrays(entry):
    feats = np.load(entry["feat_path"])
    gt = np.load(entry["gt_path"])
    return feats, gt


def sample_balanced_pixels(feats, gt, rng, n_per_class=SAMPLES_PER_CLASS):
    """Sample up to n_per_class crack-labeled and n_per_class background-labeled
    pixel locations (flat indices) from a (H, W, F) feature stack + (H, W) gt mask."""
    h, w, f = feats.shape
    flat_gt = gt.reshape(-1)
    pos_idx = np.flatnonzero(flat_gt)
    neg_idx = np.flatnonzero(~flat_gt)

    n_pos = min(n_per_class, pos_idx.size)
    n_neg = min(n_per_class, neg_idx.size)
    pos_sel = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_sel = rng.choice(neg_idx, size=n_neg, replace=False)

    sel = np.concatenate([pos_sel, neg_sel])
    flat_feats = feats.reshape(-1, f)
    X = flat_feats[sel]
    y = flat_gt[sel]
    return X, y


def build_training_set(entries, seed=SEED):
    """entries: list of manifest image dicts to draw training pixels from.
    Uses a fresh RandomState(seed) per image (matches protocol: seed=0 sampling
    per image), and concatenates across images."""
    X_parts, y_parts = [], []
    for entry in entries:
        feats, gt = load_image_arrays(entry)
        rng = np.random.RandomState(seed)
        X, y = sample_balanced_pixels(feats, gt, rng)
        X_parts.append(X)
        y_parts.append(y)
        print(f"    sampled {X.shape[0]} pixels ({int(y.sum())} pos / {int((~y).sum())} neg) from {entry['name']}")
        del feats, gt
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.reshape(-1)
    gt = gt_mask.reshape(-1)
    inter = np.count_nonzero(pred & gt)
    union = np.count_nonzero(pred | gt)
    pred_sum = np.count_nonzero(pred)
    gt_sum = np.count_nonzero(gt)

    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    precision = inter / pred_sum if pred_sum > 0 else 0.0
    recall = inter / gt_sum if gt_sum > 0 else 0.0
    return dict(iou=iou, dice=dice, precision=precision, recall=recall)


def run_loio(manifest, clf_params):
    entries = manifest["images"]
    fold_results = []

    for held_out in entries:
        train_entries = [e for e in entries if e["name"] != held_out["name"]]
        print(f"\n=== Fold: held-out = {held_out['name']} ===")
        t0 = time.time()
        X_train, y_train = build_training_set(train_entries)
        print(f"  training set: X={X_train.shape}, y_pos={int(y_train.sum())}, y_neg={int((~y_train).sum())}")

        clf = ExtraTreesClassifier(**clf_params)
        clf.fit(X_train, y_train)
        t_fit = time.time() - t0
        print(f"  fit done in {t_fit:.1f}s")

        t1 = time.time()
        feats_ho, gt_ho = load_image_arrays(held_out)
        h, w, f = feats_ho.shape
        flat_feats = feats_ho.reshape(-1, f)

        # predict in chunks to bound peak memory on the LARGE image
        proba = np.empty(flat_feats.shape[0], dtype=np.float32)
        chunk = 2_000_000
        pos_class_idx = list(clf.classes_).index(True)
        for start in range(0, flat_feats.shape[0], chunk):
            end = min(start + chunk, flat_feats.shape[0])
            proba[start:end] = clf.predict_proba(flat_feats[start:end])[:, pos_class_idx]
        pred_mask = (proba >= 0.5).reshape(h, w)
        t_pred = time.time() - t1
        print(f"  predicted full held-out image {held_out['name']} ({h}x{w}={h*w} px) in {t_pred:.1f}s")

        metrics = compute_metrics(pred_mask, gt_ho)
        print(f"  IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f} "
              f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f}")

        fold_results.append(dict(held_out_image=held_out["name"], **metrics,
                                  fit_seconds=t_fit, predict_seconds=t_pred))
        del feats_ho, gt_ho, flat_feats, proba, pred_mask, X_train, y_train, clf

    return fold_results


def mean_of(fold_results, key):
    return float(np.mean([r[key] for r in fold_results]))


def main():
    manifest = load_manifest()
    entries = manifest["images"]

    all_config_results = {}
    for cfg_name, cfg_params in CONFIGS.items():
        print(f"\n\n##### Running LOIO for {cfg_name}: {cfg_params} #####")
        t0 = time.time()
        fold_results = run_loio(manifest, cfg_params)
        elapsed = time.time() - t0
        mean_iou = mean_of(fold_results, "iou")
        mean_dice = mean_of(fold_results, "dice")
        mean_precision = mean_of(fold_results, "precision")
        mean_recall = mean_of(fold_results, "recall")
        print(f"\n{cfg_name} summary: mean_iou={mean_iou:.4f} mean_dice={mean_dice:.4f} "
              f"mean_precision={mean_precision:.4f} mean_recall={mean_recall:.4f} "
              f"(total {elapsed:.1f}s)")
        all_config_results[cfg_name] = dict(
            params=cfg_params, fold_results=fold_results,
            mean_iou=mean_iou, mean_dice=mean_dice,
            mean_precision=mean_precision, mean_recall=mean_recall,
            elapsed_seconds=elapsed,
        )

    best_cfg_name = max(all_config_results, key=lambda k: all_config_results[k]["mean_iou"])
    best = all_config_results[best_cfg_name]
    print(f"\n\n=== BEST CONFIG: {best_cfg_name} (mean_iou={best['mean_iou']:.4f}) ===")
    for other_name, other in all_config_results.items():
        if other_name != best_cfg_name:
            print(f"    (comparison) {other_name}: mean_iou={other['mean_iou']:.4f} "
                  f"mean_dice={other['mean_dice']:.4f}")

    # Final model: fit on ALL 4 images' sampled pixels combined, using the
    # winning config's hyperparameters.
    print(f"\n=== Fitting FINAL model on all {len(entries)} images with {best_cfg_name} params ===")
    t0 = time.time()
    X_final, y_final = build_training_set(entries)
    print(f"  final training set: X={X_final.shape}, y_pos={int(y_final.sum())}, y_neg={int((~y_final).sum())}")
    final_clf = ExtraTreesClassifier(**CONFIGS[best_cfg_name])
    final_clf.fit(X_final, y_final)
    print(f"  final fit done in {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final_clf, MODEL_PATH)
    print(f"  saved final model to {MODEL_PATH}")

    # Dump a small JSON summary alongside for humans/other scripts.
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "pixel_extratrees_results.json")
    with open(summary_path, "w") as f:
        json.dump(
            dict(
                best_config=best_cfg_name,
                all_configs={
                    k: dict(params=v["params"], fold_results=v["fold_results"],
                             mean_iou=v["mean_iou"], mean_dice=v["mean_dice"],
                             mean_precision=v["mean_precision"], mean_recall=v["mean_recall"],
                             elapsed_seconds=v["elapsed_seconds"])
                    for k, v in all_config_results.items()
                },
                model_path=MODEL_PATH,
            ),
            f, indent=2,
        )
    print(f"  wrote results summary to {summary_path}")


if __name__ == "__main__":
    main()
