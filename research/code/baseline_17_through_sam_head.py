"""
Control run: the 17 hand-crafted features through the EXACT SAME classifier
head the SAM-embedding conditions use.

Why this exists. sam_experiments.run_embed_loio trains MLP(128,64), max_iter=400,
no early stopping. The baseline in results/sam/baseline_pixel17_loio.json trains
MLP(64,32), max_iter=300, early_stopping=True. Comparing those two directly
confounds TWO variables -- the feature set and the head -- so a difference
cannot be attributed to the features, which is the entire question being asked.

This script holds the head fixed at SAM's version and swaps only the features.
Together with the existing baseline it gives all four cells:

                        MLP(64,32)+ES        MLP(128,64)
    17 features         baseline_*.json      THIS SCRIPT
    SAM embeddings      (not run)            embed_mlp

Usage:
    python3 baseline_17_through_sam_head.py
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_benchmark_report import metrics_from_pred

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(PROJECT_DIR, "dataset_cache")
OUT_DIR = os.path.join(PROJECT_DIR, "results", "sam")
STEMS = ["333_75_um_zoom", "336_25", "338_13", "LARGE_343_75"]
SEED = 0

# Verbatim from sam_experiments.run_embed_loio's `head == "mlp"` branch.
SAM_HEAD = dict(hidden_layer_sizes=(128, 64), max_iter=400, random_state=SEED)


def sample(stem, n_per_class, rng):
    feats = np.load(os.path.join(CACHE, f"{stem}_features.npy"), mmap_mode="r")
    gt = np.load(os.path.join(CACHE, f"{stem}_gt.npy")).astype(bool)
    Xs, ys = [], []
    for cls in (1, 0):
        idx = np.nonzero((gt == bool(cls)).ravel())[0]
        take = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        rr, cc = np.unravel_index(take, gt.shape)
        Xs.append(np.asarray(feats[rr, cc, :], np.float32))
        ys.append(np.full(len(rr), cls, np.int8))
    del feats
    return np.concatenate(Xs), np.concatenate(ys)


def run(n_per_class):
    rows = []
    for held in STEMS:
        rng = np.random.RandomState(SEED)
        Xtr, ytr = [], []
        for stem in STEMS:
            if stem == held:
                continue
            X, y = sample(stem, n_per_class, rng)
            Xtr.append(X)
            ytr.append(y)
        Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
        clf = Pipeline([("s", StandardScaler()), ("m", MLPClassifier(**SAM_HEAD))])
        t0 = time.time()
        clf.fit(Xtr, ytr)

        feats = np.load(os.path.join(CACHE, f"{held}_features.npy"), mmap_mode="r")
        gt = np.load(os.path.join(CACHE, f"{held}_gt.npy")).astype(bool)
        prob = np.zeros(gt.shape, np.float32)
        for r0 in range(0, gt.shape[0], 256):          # band-chunked: LARGE is 1.6 GB
            r1 = min(r0 + 256, gt.shape[0])
            block = np.asarray(feats[r0:r1], np.float32).reshape(-1, feats.shape[2])
            prob[r0:r1] = clf.predict_proba(block)[:, 1].reshape(r1 - r0, -1)
        del feats

        m = metrics_from_pred(prob > 0.5, gt)
        m.update(image=held, n_train=int(len(ytr)), secs=round(time.time() - t0, 1),
                 pred_area_frac=float((prob > 0.5).mean()))
        rows.append(m)
        print(f"  {held:16s} IoU={m['iou']:.3f} rec={m['recall']:.3f} "
              f"prec={m['precision']:.3f} area={m['pred_area_frac']*100:.1f}%", flush=True)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for n in (20000, 30000):
        print(f"\n17 features through SAM's head MLP(128,64), n_per_class={n}:")
        rows = run(n)
        mean = float(np.mean([r["iou"] for r in rows]))
        print(f"  => mean IoU {mean:.4f}  recall {np.mean([r['recall'] for r in rows]):.4f}")
        out = os.path.join(OUT_DIR, f"baseline_pixel17_samhead_n{n}.json")
        with open(out, "w") as f:
            json.dump(dict(condition="pixel17_through_sam_head", deployable=True,
                           architecture=f"Pipeline(StandardScaler, MLPClassifier({SAM_HEAD}))",
                           protocol="leave-one-image-out over the 4 GT images",
                           n_per_class=n, n_features=17, rows=rows,
                           mean_iou=mean), f, indent=2)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
