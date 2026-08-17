"""
Per-image, honestly-held-out numbers for the DEPLOYED 17-feature pixel model
architecture, for the SAM comparison table.

Why this file exists: models/pixel_hgb_final.joblib was trained on all four
ground-truth images' pixels AND on human corrections painted on those same
images. Running that joblib on those four images and quoting the result would
be training-set leakage, and would hand the deployed model an advantage the
SAM embedding conditions (embed_lr / embed_mlp / embed_plus17) do not get --
those are leave-one-image-out. So the baseline is RE-TRAINED under LOIO here,
same architecture, and scored on the held-out image only.

Parity, item by item, with code/generate_benchmark_report.py's run_loio:
  - same 4 images, same order, from dataset_cache/manifest.json (IMAGES)
  - same np.random.RandomState(SEED=0) created fresh per fold, consumed by
    the same sample_pixels() in the same image order
  - sample_pixels() and metrics_from_pred() are IMPORTED from
    generate_benchmark_report, not reimplemented, so they cannot drift
  - same 0.5 probability threshold, evaluated on every pixel of the held-out
    image, NO post-processing (matches both run_loio and sam_experiments.py's
    `prob > 0.5` scoring)

The ONE deliberate implementation difference from run_loio: the held-out
image's features are memory-mapped and predicted in row chunks instead of one
23.5-megapixel call. That is numerically identical (StandardScaler and
MLPClassifier are per-sample maps) but keeps peak RSS near 1 GB instead of
~20 GB, which matters because a SAM benchmark is using the same machine's
unified memory. Verified: reproduces benchmark_figures/extended_summary.json's
MLP LOIO folds.

Architecture = whatever the deployed model actually is, read off
models/pixel_hgb_final.joblib (despite the filename it is not HGB any more):
Pipeline(StandardScaler, MLPClassifier(hidden_layer_sizes=(64,32), alpha=1e-4,
max_iter=300, early_stopping=True, random_state=0)) -- i.e.
retrain_with_corrections.MLP_PARAMS.

CPU-only sklearn. Does not touch torch, SAM, or the GPU.

Usage:
    python3 baseline_loio_for_sam.py [--n-per-class 30000] [--out PATH]
"""
import argparse
import json
import os
import sys
import time

# Leave cores for the SAM benchmark sharing this machine.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_benchmark_report import IMAGES, SEED, metrics_from_pred, sample_pixels

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Deployed architecture (retrain_with_corrections.MLP_PARAMS / the params
# inside models/pixel_hgb_final.joblib).
MLP_PARAMS = dict(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=300,
                  early_stopping=True, random_state=0)
CHUNK_ROWS = 1_000_000  # pixels per predict_proba call


def build_deployed_architecture():
    return Pipeline([("scaler", StandardScaler()),
                     ("mlp", MLPClassifier(**MLP_PARAMS))])


def predict_chunked(clf, feat_path, n_pixels, n_feat):
    """Full-image crack probability, streamed. Same math as one big call."""
    feats = np.load(feat_path, mmap_mode="r")
    flat = feats.reshape(-1, n_feat)
    out = np.empty(n_pixels, np.float32)
    for a in range(0, n_pixels, CHUNK_ROWS):
        b = min(a + CHUNK_ROWS, n_pixels)
        out[a:b] = clf.predict_proba(np.asarray(flat[a:b], np.float32))[:, 1]
    del flat, feats
    return out


def run(n_per_class):
    rows = []
    for i, held in enumerate(IMAGES):
        # rng lifecycle copied from run_loio: fresh per fold, then consumed by
        # sample_pixels over the non-held-out images in IMAGES order.
        rng = np.random.RandomState(SEED)
        X_list, y_list = [], []
        for j, img in enumerate(IMAGES):
            if j == i:
                continue
            feats = np.load(img["feat_path"], mmap_mode="r")
            gt = np.load(img["gt_path"])
            X, y = sample_pixels(feats, gt, n_per_class, rng)
            X_list.append(np.asarray(X, np.float32))
            y_list.append(y)
            del feats, gt
        X_train = np.concatenate(X_list)
        y_train = np.concatenate(y_list)
        del X_list, y_list

        clf = build_deployed_architecture()
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_s = time.time() - t0
        n_train = int(len(y_train))
        n_feat = int(X_train.shape[1])
        del X_train, y_train

        gt_h = np.load(held["gt_path"]).astype(bool)
        gt_flat = gt_h.reshape(-1)
        t0 = time.time()
        proba = predict_chunked(clf, held["feat_path"], gt_flat.size, n_feat)
        pred_s = time.time() - t0
        pred_flat = proba >= 0.5

        m = metrics_from_pred(pred_flat, gt_flat)
        m.update(image=held["name"],
                 pred_area_frac=float(pred_flat.mean()),
                 gt_area_frac=float(gt_flat.mean()),
                 megapixels=round(gt_flat.size / 1e6, 1),
                 n_train_pixels=n_train,
                 fit_seconds=round(fit_s, 1),
                 predict_seconds=round(pred_s, 1))
        rows.append(m)
        print(f"    {held['name']:16s} IoU={m['iou']:.4f} dice={m['dice']:.4f} "
              f"prec={m['precision']:.4f} rec={m['recall']:.4f} "
              f"area={m['pred_area_frac']*100:.1f}% (gt {m['gt_area_frac']*100:.1f}%) "
              f"[fit {fit_s:.0f}s predict {pred_s:.0f}s]", flush=True)
        del gt_h, gt_flat, proba, pred_flat, clf
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=30000,
                    help="pixels per class per TRAINING image (benchmark default 30000; "
                         "sam_experiments.py's embedding conditions use 20000)")
    ap.add_argument("--out", default=os.path.join(PROJECT_DIR, "results", "sam",
                                                  "baseline_pixel17_loio.json"))
    args = ap.parse_args()

    print(f"LOIO baseline | arch=MLP{MLP_PARAMS['hidden_layer_sizes']} "
          f"(deployed) | n_per_class_per_training_image={args.n_per_class}", flush=True)
    t0 = time.time()
    rows = run(args.n_per_class)
    mean = {k: float(np.mean([r[k] for r in rows]))
            for k in ("iou", "dice", "precision", "recall", "pred_area_frac")}
    std = {k: float(np.std([r[k] for r in rows])) for k in ("iou", "dice", "precision", "recall")}
    print(f"  => mean IoU {mean['iou']:.4f} dice {mean['dice']:.4f} "
          f"prec {mean['precision']:.4f} rec {mean['recall']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    payload = dict(
        condition="pixel17_mlp_loio",
        deployable=True,
        architecture="Pipeline(StandardScaler, MLPClassifier(hidden_layer_sizes=(64,32), "
                     "alpha=1e-4, max_iter=300, early_stopping=True, random_state=0))",
        deployed_model_reference=os.path.join(PROJECT_DIR, "models", "pixel_hgb_final.joblib"),
        protocol="leave-one-image-out over the 4 GT images; train on n_per_class crack + "
                 "n_per_class background pixels sampled from each of the 3 non-held-out "
                 "images; predict every pixel of the held-out image; threshold 0.5; no "
                 "post-processing; metrics_from_pred from generate_benchmark_report.py",
        n_per_class=args.n_per_class, n_features=17,
        rows=rows, mean=mean, std=std,
        mean_iou=mean["iou"], mean_dice=mean["dice"],
        mean_precision=mean["precision"], mean_recall=mean["recall"],
        secs=round(time.time() - t0, 1),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
