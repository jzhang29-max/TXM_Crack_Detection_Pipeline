"""How well does the model actually generalise? k-fold, grouped by image.

    python3 code/crossval.py                # the honest number
    python3 code/crossval.py --demo-leak    # also show what random k-fold would tell you

WHY THIS EXISTS. The retrain gate reports IoU around 0.94 and recall around 0.98, and both
are measured on the same four ground-truth images the candidate trains on. They are floors,
not estimates of performance on a new specimen, and reading them as accuracy is how a model
that marks 22% of confirmed crack-free specimen as crack came to be deployed here.

WHY NOT ORDINARY k-FOLD -- this is the important part. Shuffling pixels into k folds leaks,
and it leaks in the flattering direction:

  * the 17 hand-crafted features come from neighbourhood filters that reach 256 px (measured:
    a cropped feature stack matches the full-image one only beyond a 256 px inset)
  * a SAM embedding is a bilinear lookup into a 64x64 grid per 1024 px tile, so a 16x16 block
    of pixels shares essentially one embedding vector

So a randomly held-out pixel almost always has a training pixel a few pixels away carrying
the same measurement. Measured on this data with the deployed architecture:

    random 4-fold, pixels shuffled     IoU 0.930   fold sd 0.003
    grouped 4-fold, split by image     IoU 0.824   fold sd 0.050

Random k-fold does not fail to reveal the overfitting -- it hides it, inflating the score by
0.106, and it reports an implausibly tight fold spread while doing so. That tight spread is
the tell: four folds of genuinely different specimens cannot agree to 0.003.

Grouping by image is the coarsest split this data supports and the only one where train and
test share no neighbourhood. With four ground-truth images it is leave-one-image-out.

WHAT THE NUMBER MEANS. It scores THE ARCHITECTURE PLUS THE GROUND TRUTH, refit from scratch
per fold at a capped row count. It is not the deployed model's own score: that model has
seen all four images, so it has no honest score and never can. Read it as "what a model
built this way scores on an image it has not seen".

CAVEAT THAT WILL NOT GO AWAY. n = 4, all one specimen group (B2), and the fold sd is ~0.05.
Differences under ~0.015 IoU are reseeding noise. A fifth and sixth ground-truth image would
do more for confidence here than any change to the model.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import pipeline as P         # noqa: E402


def demo_leak():
    """Run BOTH protocols on identical rows, so the only difference is the split."""
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(0)
    Xs, ys, gs, n17 = [], [], [], None
    for gi, stem in enumerate(P.GT_STEMS):
        got = P._gt_rows(stem, P.CV_ROWS_PER_IMAGE, rng)
        if got is None:
            continue
        block, y, n17 = got
        Xs.append(block); ys.append(y); gs.append(np.full(len(y), gi))
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    del Xs, ys, gs
    print(f"{len(y):,} rows from {len(np.unique(g))} images, {100*y.mean():.1f}% crack\n")

    def clf():
        return Pipeline([("scaler", StandardScaler()),
                         ("mlp", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                               early_stopping=True, n_iter_no_change=8))])

    def score(tr, te):
        if len(tr) > P.CV_TRAIN_CAP:
            tr = np.random.RandomState(7).choice(tr, P.CV_TRAIN_CAP, replace=False)
        probs = [clf().fit(X[tr, c], y[tr]).predict_proba(X[te, c])[:, 1]
                 for c in (slice(0, n17), slice(0, X.shape[1]))]
        pred = np.mean(probs, axis=0) >= 0.5
        tp = int((pred & y[te]).sum()); fp = int((pred & ~y[te]).sum())
        fn = int((~pred & y[te]).sum())
        return tp / max(tp + fp + fn, 1)

    out = {}
    for label, splits in (
        ("RANDOM k-fold (pixels shuffled, image boundaries ignored)",
         list(KFold(4, shuffle=True, random_state=0).split(X))),
        ("GROUPED k-fold (split by image, no image on both sides)",
         list(GroupKFold(4).split(X, groups=g))),
    ):
        print(f"=== {label} ===")
        ious = []
        for k, (tr, te) in enumerate(splits, 1):
            v = score(tr, te)
            ious.append(v)
            who = P.GT_STEMS[int(g[te][0])] if "GROUPED" in label else f"fold {k}"
            print(f"  {who:<18} IoU {v:.4f}")
        print(f"  MEAN {np.mean(ious):.4f}   fold sd {np.std(ious):.4f}\n")
        out["random" if "RANDOM" in label else "grouped"] = ious
    gap = np.mean(out["random"]) - np.mean(out["grouped"])
    print(f"Random k-fold is inflated by {gap:+.4f} IoU, and its fold sd is "
          f"{np.std(out['random']):.4f} against {np.std(out['grouped']):.4f} -- "
          f"agreement that tight across four different specimens is the tell.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-leak", action="store_true",
                    help="also run random pixel-level k-fold, to show why it must not be used")
    ap.add_argument("--json", metavar="PATH", help="write the result as JSON")
    a = ap.parse_args()

    if a.demo_leak:
        demo_leak()
        print()

    r = P.crossval_grouped(progress=lambda s, k, n: print(f"  {s}", flush=True))
    if r is None:
        raise SystemExit("no ground truth available -- run code/unpack_package.py first")
    print(f"\nHELD-OUT IoU {r['mean_iou']:.4f}   sd {r['std_iou']:.4f}   "
          f"worst image {r['min_iou']:.4f}")
    print(f"  precision {r['mean_precision']:.4f}   recall {r['mean_recall']:.4f}")
    print(f"  {r['k']}-fold grouped by {r['grouped_by']}, "
          f"{r['rows_per_image']:,} px sampled per image, "
          f"fits capped at {r['train_cap']:,} rows\n")
    for f in r["per_fold"]:
        print(f"    held out {f['held_out']:<16} IoU {f['iou']:.4f}  "
              f"P {f['precision']:.3f}  R {f['recall']:.3f}")
    print("\nCompare against the retrain gate's in-sample IoU, which is measured on these")
    print("same images and runs about 0.11 higher. This is the number to quote.")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(r, fh, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
