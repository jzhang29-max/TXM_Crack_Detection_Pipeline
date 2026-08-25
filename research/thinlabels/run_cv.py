"""Grouped-by-image cross-validation of four label-width arms on identical folds.

Every arm is trained on rows drawn from the same four disjoint pools (see
extract_rows.py) and scored on ONE fixed evaluation row set, so the numbers are
comparable across arms. Folds are derived once, from the eval rows, so all arms see the
same image partition.

    baseline          crack = the painted stroke as-is        not-crack = corr==2
    thin_ring_neg     crack = dark core                       not-crack = corr==2 + ring
    thin_plus_margin  crack = dark core                       not-crack = corr==2 only
                      (the ring is UNLABELLED -- excluded from both classes)
    core_dilate3      crack = dark core dilated 3 px          not-crack = corr==2 only
                      (outer ring unlabelled)

Under the deployed row-sampling recipe (gather_training_data samples negatives from
corr==2 only) "narrow the crack labels and leave not-crack rows unchanged" IS
thin_plus_margin -- the ring is already unsampled. thin_ring_neg is the arm that
actually calls the ring background, which is the contrast worth measuring.

Two scorings per arm:
  same-target  y* = pixel is in the dark core. The same target for every arm; the best
               available proxy for the physical crack. This is the comparable number.
  own-target   each arm scored against its own label definition, on the same rows, with
               rows its recipe calls unlabelled dropped. NOT comparable across arms.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
OUT = os.path.join(P0, "research", "thinlabels")

CRACK_CAP = 8000
NEG_CAP = 8000
EVAL_PER_CLASS = 4000
TRAIN_CAP = 90000          # pipeline.CV_TRAIN_CAP -- measured better than 400k on this data
TEST_CAP = 250000          # pipeline.CV_TEST_CAP
K = 5
N17 = 17

ARMS = {
    "baseline":         dict(pos=[1, 2, 3], neg=[4]),
    "thin_ring_neg":    dict(pos=[1],       neg=[4, 2, 3]),
    "thin_plus_margin": dict(pos=[1],       neg=[4]),
    "core_dilate3":     dict(pos=[1, 2],    neg=[4]),
    # A POWERED version of "the ring is background". thin_ring_neg is what editing
    # correction.npy would actually do, and it is area-faithful -- corr==2 covers 10-1000x
    # more area than the ring, so the ring wins only ~5 of 8000 negative rows per image and
    # the arm cannot answer whether ring-as-background does damage. This one hands the ring
    # half the negative budget so the question has statistical power. It is a stress test,
    # not a deployable recipe.
    "ring_neg_forced":  dict(pos=[1],       neg=[4], neg_forced_frac=0.5),
}
# Own-target label for each arm, as a function of the pool a row came from.
OWN_POS = {"baseline": {1, 2, 3}, "thin_ring_neg": {1}, "thin_plus_margin": {1},
           "core_dilate3": {1, 2}, "ring_neg_forced": {1}}
OWN_DROP = {"baseline": set(), "thin_ring_neg": set(), "thin_plus_margin": {2, 3},
            "core_dilate3": {3}, "ring_neg_forced": set()}


def clf():
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def draw(rng, want, pool_rows, weights):
    """`want` rows split across pools in proportion to `weights` (true pixel areas).

    Sampling k rows from a uniform sample of a set is a uniform sample of that set, so
    splitting a fixed budget by true area reproduces what sampling the union directly
    would have given -- without a second feature pass.
    """
    avail = np.array([len(r) for r in pool_rows])
    w = np.array(weights, float)
    if w.sum() <= 0:
        return np.empty(0, np.int64)
    want = int(min(want, avail.sum()))
    n = np.floor(want * w / w.sum()).astype(int)
    n = np.minimum(n, avail)
    # hand any shortfall (from rounding or from an exhausted pool) to pools with headroom
    for _ in range(4):
        short = want - int(n.sum())
        if short <= 0:
            break
        head = avail - n
        if head.sum() <= 0:
            break
        add = np.minimum(head, np.maximum(1, np.floor(short * head / head.sum())).astype(int))
        room = short
        for i in np.argsort(-head):
            t = int(min(add[i], room, head[i]))
            n[i] += t; room -= t
            if room <= 0:
                break
    out = []
    for r, k in zip(pool_rows, n):
        if k <= 0:
            continue
        out.append(rng.choice(r, k, replace=False) if k < len(r) else r)
    return np.concatenate(out) if out else np.empty(0, np.int64)


def build():
    pool = np.load(os.path.join(OUT, "pool.npy"))
    grp = np.load(os.path.join(OUT, "grp.npy"))
    meta = json.load(open(os.path.join(OUT, "rows_meta.json")))
    area = {d["gi"]: [d["n_core"], d["n_inner"], d["n_outer"], d["n_notcrack"]]
            for d in meta["images"]}
    gis = sorted(set(grp.tolist()))
    rows_by = {}
    for gi in gis:
        m = grp == gi
        rows_by[gi] = {p: np.flatnonzero(m & (pool == p)) for p in (1, 2, 3, 4)}

    arm_rows = {}
    for name, spec in ARMS.items():
        idx = []
        for gi in gis:
            rb, ar = rows_by[gi], area[gi]
            idx.append(draw(np.random.RandomState(0), CRACK_CAP,
                            [rb[p] for p in spec["pos"]], [ar[p - 1] for p in spec["pos"]]))
            frac = spec.get("neg_forced_frac")
            if frac:
                idx.append(draw(np.random.RandomState(2), int(NEG_CAP * frac),
                                [rb[2], rb[3]], [ar[1], ar[2]]))
                idx.append(draw(np.random.RandomState(1), NEG_CAP - int(NEG_CAP * frac),
                                [rb[4]], [ar[3]]))
            else:
                idx.append(draw(np.random.RandomState(1), NEG_CAP,
                                [rb[p] for p in spec["neg"]],
                                [ar[p - 1] for p in spec["neg"]]))
        arm_rows[name] = np.concatenate(idx)

    ev = []
    for gi in gis:
        rb, ar = rows_by[gi], area[gi]
        ev.append(draw(np.random.RandomState(7), EVAL_PER_CLASS,
                       [rb[1], rb[2], rb[3]], ar[:3]))
        ev.append(draw(np.random.RandomState(8), EVAL_PER_CLASS, [rb[4]], [ar[3]]))
    eval_rows = np.concatenate(ev)
    return pool, grp, meta, arm_rows, eval_rows


def score(pred, t):
    tp = int((pred & t).sum()); fp = int((pred & ~t).sum()); fn = int((~pred & t).sum())
    return dict(iou=round(tp / max(tp + fp + fn, 1), 4),
                precision=round(tp / max(tp + fp, 1), 4),
                recall=round(tp / max(tp + fn, 1), 4), n=int(len(t)), pos=int(t.sum()))


def main():
    t00 = time.time()
    X = np.load(os.path.join(OUT, "X.npy"), mmap_mode="r")
    pool, grp, meta, arm_rows, eval_rows = build()

    # Folds from the eval rows -> identical image partition for every arm.
    fold_of = {}
    for f, (_, te) in enumerate(GroupKFold(K).split(eval_rows, groups=grp[eval_rows])):
        for gi in np.unique(grp[eval_rows][te]):
            fold_of[int(gi)] = f
    json.dump({str(k): v for k, v in fold_of.items()},
              open(os.path.join(OUT, "folds.json"), "w"), indent=1)

    mdir = os.path.join(OUT, "models")
    os.makedirs(mdir, exist_ok=True)
    import joblib

    results = {}
    for name in ARMS:
        rows = arm_rows[name]
        y_arm = np.isin(pool[rows], list(OWN_POS[name]))
        per_fold = []
        for f in range(K):
            tr_mask = np.array([fold_of[int(g)] != f for g in grp[rows]])
            tr = rows[tr_mask]
            ytr = y_arm[tr_mask]
            te = eval_rows[np.array([fold_of[int(g)] == f for g in grp[eval_rows]])]
            rng = np.random.RandomState(0)
            if len(tr) > TRAIN_CAP:
                sel = rng.choice(len(tr), TRAIN_CAP, replace=False)
                tr, ytr = tr[sel], ytr[sel]
            if len(te) > TEST_CAP:
                te = te[rng.choice(len(te), TEST_CAP, replace=False)]
            order = np.argsort(tr)                    # mmap reads like sorted access
            tr, ytr = tr[order], ytr[order]
            Xtr = np.asarray(X[tr], np.float32)
            Xte = np.asarray(X[np.sort(te)], np.float32)
            te = np.sort(te)

            probs, fits = [], []
            for cols in (slice(0, N17), slice(0, X.shape[1])):
                p = clf().fit(Xtr[:, cols], ytr)
                fits.append(p)
                probs.append(p.predict_proba(Xte[:, cols])[:, 1])
            prob = np.mean(probs, axis=0)
            joblib.dump(fits[0], os.path.join(mdir, f"{name}_f{f}_17.joblib"))
            joblib.dump(fits[1], os.path.join(mdir, f"{name}_f{f}_273.joblib"))

            tp_ = pool[te]
            same = tp_ == 1                                    # FIXED target: the dark core
            keep = ~np.isin(tp_, list(OWN_DROP[name]))
            own = np.isin(tp_, list(OWN_POS[name]))
            row = dict(fold=f, n_train=int(len(tr)),
                       train_crack_fraction=round(float(ytr.mean()), 4),
                       n_images_heldout=int(len(np.unique(grp[te]))),
                       same=score(prob > 0.5, same),
                       same_17=score(probs[0] > 0.5, same),
                       same_273=score(probs[1] > 0.5, same),
                       own=score((prob > 0.5)[keep], own[keep]))
            per_fold.append(row)
            print(f"{name:17s} fold{f} same IoU {row['same']['iou']:.4f} "
                  f"P {row['same']['precision']:.4f} R {row['same']['recall']:.4f}  "
                  f"own IoU {row['own']['iou']:.4f}  ntr={len(tr)} "
                  f"crackfrac={row['train_crack_fraction']:.3f}", flush=True)
        agg = {}
        for k in ("same", "own", "same_17", "same_273"):
            for met in ("iou", "precision", "recall"):
                v = [pf[k][met] for pf in per_fold]
                agg[f"{k}_{met}"] = round(float(np.mean(v)), 4)
                agg[f"{k}_{met}_sd"] = round(float(np.std(v, ddof=1)), 4)
        results[name] = dict(per_fold=per_fold, mean=agg,
                             n_rows_total=int(len(rows)),
                             crack_fraction=round(float(y_arm.mean()), 4))
        print(f"== {name}: same IoU {agg['same_iou']:.4f} +-{agg['same_iou_sd']:.4f} "
              f"P {agg['same_precision']:.4f} R {agg['same_recall']:.4f} | "
              f"own IoU {agg['own_iou']:.4f}", flush=True)

    json.dump(dict(arms=results, folds=fold_of, k=K, train_cap=TRAIN_CAP,
                   crack_cap=CRACK_CAP, neg_cap=NEG_CAP,
                   eval_per_class=EVAL_PER_CLASS,
                   n_eval_rows=int(len(eval_rows)),
                   seconds=round(time.time() - t00, 1)),
              open(os.path.join(OUT, "cv_results.json"), "w"), indent=1)
    print("done in", round(time.time() - t00, 1), "s")


if __name__ == "__main__":
    main()
