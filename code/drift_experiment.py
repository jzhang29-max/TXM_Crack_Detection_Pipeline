"""Does the human-in-the-loop segmentation cycle silently degrade? A controlled demonstration.

    python3 code/build_label_folds.py          # once, builds the feature cache
    python3 code/drift_experiment.py --generations 6

THE OBSERVATION THIS FORMALISES. Three consecutive real retrains in this project moved the
false-call rate on confirmed crack-free specimens 0.137% -> 0.238% -> 0.264% while
ground-truth IoU sat flat at 0.936-0.940. Nothing in the tool, and nothing in any comparable
tool, would have shown that: the metric being watched did not move, and the metric that moved
was not being watched.

THE MECHANISM, measured. Correcting a model is cheap when you agree with it and expensive
when you do not: one click on a connected region accepts a whole blob, while disagreeing means
drawing. In this project's own label set, **98.3% of force-crack pixels lie on pixels the
model had already called crack**. So the labels a human produces inside the loop are
overwhelmingly *confirmations*, and training on them is a form of self-training -- which is
known to amplify a model's existing bias, but has not to our knowledge been quantified for
interactive microscopy segmentation, nor connected to a measurable degradation.

WHAT THIS SCRIPT DOES. It runs the loop deliberately, with the human replaced by a
perfectly-agreeable one, so the effect can be isolated from any real annotator's judgement:

  generation 0   train on the dense ground truth only
  generation k   take generation k-1's predictions on the unlabelled pool, accept the
                 confident ones as crack labels (the machine equivalent of clicking
                 Flip region), add them to the training set, refit

and after each generation measures three things:

  * ground-truth IoU IN-SAMPLE -- what this project's gate actually watched
  * ground-truth IoU HELD OUT  -- leave-one-image-out over the four ground-truth images
  * false-call rate on six specimens confirmed to contain NO crack, which are held out of
    every generation's training set, so this number is never contaminated

The prediction being tested: the first stays flat while the third climbs. If so, a
human-in-the-loop cycle guarded only by an in-sample overlap metric degrades in a direction
its guard cannot see -- and the negative-control false-call rate is what makes it visible.

This is a null-effect-possible experiment. If the false-call rate stays flat, the drift
observed in production was caused by something else and this document should say so.
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

import store as S            # noqa: E402
import pipeline as P         # noqa: E402

CACHE = os.path.join(PROJECT, "paint", "label_folds.npz")
GT_PER_IMAGE = 25000
POOL_PER_GEN = 40000          # pseudo-labels accepted per generation
CONF = 0.80                   # only confident predictions are "confirmed", as a human would


def clf():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args()

    if not os.path.exists(CACHE):
        raise SystemExit("run python3 code/build_label_folds.py first")
    z = np.load(CACHE, allow_pickle=False)
    ids = sorted({k.split("|")[0] for k in z.files})

    # Partition the images. The crack-free specimens are held out of ALL training so the
    # false-call number can never be contaminated by its own measurement set.
    clean_ids, pool_ids = [], []
    by_id = {m["id"]: m for m in S.list_images()}
    for iid in ids:
        fn = (by_id.get(iid, {}) or {}).get("filename", "") or ""
        if any(k.lower() in fn.lower() for k in P.CLEAN_SPECIMENS):
            clean_ids.append(iid)
        else:
            pool_ids.append(iid)

    def block(iid):
        kx, ks = f"{iid}|x17", f"{iid}|xsam"
        if kx not in z.files or ks not in z.files:
            return None
        return np.concatenate([z[kx], z[ks]], axis=1).astype(np.float32)

    # The false-call measurement set must be sampled UNIFORMLY over the specimen, not drawn
    # from the labelled pixels in the cache.
    #
    # A second attempt used those labelled pixels and measured a 4.97% false-call rate where
    # the real model measures 0.106% -- 47x too high. The cause: the not-crack labels on
    # these specimens are largely imported research negatives, and HANDOFF.md records that
    # batch as "false-positive cleanup (wedge margin, edge ring, round speckle)". They sit
    # exactly on the pixels a model is most likely to fire on, so a rate estimated from them
    # is an adversarial sample, not a false-call rate. Any conclusion drawn from it -- drift
    # or no drift -- would have been an artifact of the sampling.
    fp6 = os.environ.get("FP_HOLDOUT") or os.path.join(
        os.environ.get("SP", "/tmp"), "fp_holdout6.npz")
    if not os.path.exists(fp6):
        raise SystemExit(f"need a uniform false-call set at {fp6} "
                         f"(build it with scratch/build_fp_holdout.py)")
    zf = np.load(fp6)
    names = sorted({k.rsplit("_", 1)[0] for k in zf.files})
    clean_X = np.concatenate([np.concatenate([zf[f"{n}_x17"], zf[f"{n}_xsam"]], axis=1)
                              for n in names]).astype(np.float32)
    clean_per = {n: np.concatenate([zf[f"{n}_x17"], zf[f"{n}_xsam"]], axis=1).astype(np.float32)
                 for n in names}
    print(f"false-call set: {len(names)} specimens, {len(clean_X):,} pixels sampled "
          f"uniformly inside the specimen")
    _unused_clean = [b for b in (block(i) for i in clean_ids) if b is not None]
    pool_X = [b for b in (block(i) for i in pool_ids) if b is not None]
    if not pool_X:
        raise SystemExit("cache lacks SAM blocks; rebuild it")
    pool_X = np.concatenate(pool_X)
    print(f"{len(clean_ids)} crack-free specimens held out of all training "
          f"({len(clean_X):,} px for the false-call measurement)")
    print(f"{len(pool_ids)} other images form the unlabelled pool ({len(pool_X):,} px)")

    # dense ground truth, and the per-image split for the held-out score
    rng = np.random.RandomState(5)
    gt_by_stem = {}
    for stem in P.GT_STEMS_SHIPPED:
        got = P._gt_rows(stem, GT_PER_IMAGE, rng)
        if got is not None:
            gt_by_stem[stem] = (got[0], got[1])
    gtX = np.concatenate([v[0] for v in gt_by_stem.values()])
    gtY = np.concatenate([v[1] for v in gt_by_stem.values()])
    print(f"{len(gt_by_stem)} dense ground-truth images ({len(gtY):,} px, "
          f"{100*gtY.mean():.1f}% crack)\n")

    def iou(pred, y):
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
        return tp / max(tp + fp + fn, 1)

    def heldout_iou(X, Y):
        """Leave-one-ground-truth-image-out, refit per fold."""
        vals = []
        for held in gt_by_stem:
            trX = [X]; trY = [Y]
            for s2, (bx, by) in gt_by_stem.items():
                if s2 == held:
                    continue
            # X,Y already contain every GT stem, so remove the held one by rebuilding
            keep = [np.concatenate([bx for s2, (bx, _) in gt_by_stem.items() if s2 != held]),
                    np.concatenate([by for s2, (_, by) in gt_by_stem.items() if s2 != held])]
            extraX = X[len(gtX):] if len(X) > len(gtX) else np.empty((0, X.shape[1]), X.dtype)
            extraY = Y[len(gtY):] if len(Y) > len(gtY) else np.empty(0, bool)
            fX = np.concatenate([keep[0], extraX]); fY = np.concatenate([keep[1], extraY])
            if len(fY) > 150000:
                i = np.random.RandomState(7).choice(len(fY), 150000, replace=False)
                fX, fY = fX[i], fY[i]
            mh = clf().fit(fX, fY)
            bx, by = gt_by_stem[held]
            vals.append(iou(mh.predict_proba(bx)[:, 1] >= 0.5, by))
        return float(np.mean(vals))

    # GENERATION 0 MUST BE THE REAL DEPLOYED CONDITION, or the experiment measures nothing.
    #
    # A first attempt trained generation 0 on the four ground-truth images ALONE. Those are
    # 25% crack by area and carry no negatives from any other specimen, so that model marked
    # 25.1% of confirmed crack-free material as crack -- the same regime as the two models
    # this project already measured at 22% and rejected. The false-call rate was saturated
    # before the loop started and could not climb, so "no drift" was a statement about a
    # broken baseline, not about the loop.
    #
    # Generation 0 is now what a real retrain sees: the dense ground truth PLUS the owner's
    # existing labels on the pool images. Successive generations then add the model's own
    # confident predictions on top, which isolates the effect of accumulating CONFIRMATIONS
    # from the effect of having labels at all.
    pool_Y = []
    for iid in pool_ids:
        ky = f"{iid}|y"
        if ky in z.files and block(iid) is not None:
            pool_Y.append(z[ky].astype(bool))
    pool_Y = np.concatenate(pool_Y) if pool_Y else np.empty(0, bool)
    assert len(pool_Y) == len(pool_X), (len(pool_Y), len(pool_X))
    r0 = np.random.RandomState(4)
    base = r0.choice(len(pool_Y), min(120000, len(pool_Y)), replace=False)
    X = np.concatenate([gtX, pool_X[base]])
    Y = np.concatenate([gtY, pool_Y[base]])
    print(f"generation 0 = {len(gtY):,} dense ground-truth rows + "
          f"{len(base):,} of the owner's own labels ({100*pool_Y[base].mean():.1f}% crack)\n")
    rows = []
    for gen in range(a.generations):
        fit = (X, Y) if len(Y) <= 150000 else None
        if fit is None:
            i = np.random.RandomState(7).choice(len(Y), 150000, replace=False)
            fit = (X[i], Y[i])
        model = clf().fit(*fit)

        ins = iou(model.predict_proba(gtX)[:, 1] >= 0.5, gtY)
        ho = heldout_iou(X, Y)
        pers = {n: float((model.predict_proba(v)[:, 1] >= 0.5).mean())
                for n, v in clean_per.items()}
        fc = float(np.mean(list(pers.values())))
        rows.append(dict(generation=gen, train_rows=int(len(Y)),
                         gt_iou_in_sample=round(ins, 4), gt_iou_held_out=round(ho, 4),
                         false_call_rate=round(fc, 6),
                         false_call_range=[round(min(pers.values()), 6),
                                           round(max(pers.values()), 6)]))
        print(f"  gen {gen}: rows {len(Y):>7,}  in-sample IoU {ins:.4f}  "
              f"held-out IoU {ho:.4f}  false calls {fc*100:.3f}%", flush=True)

        if gen == a.generations - 1:
            break
        # THE LOOP: accept this model's own confident predictions as crack labels, exactly
        # as a one-click region flip does, and add them to the training set.
        p = model.predict_proba(pool_X)[:, 1]
        take_pos = np.flatnonzero(p >= CONF)
        take_neg = np.flatnonzero(p <= 1 - CONF)
        n = min(POOL_PER_GEN // 2, len(take_pos), len(take_neg))
        if n == 0:
            print("  pool exhausted"); break
        r = np.random.RandomState(100 + gen)
        pos = r.choice(take_pos, n, replace=False); neg = r.choice(take_neg, n, replace=False)
        X = np.concatenate([X, pool_X[pos], pool_X[neg]])
        Y = np.concatenate([Y, np.ones(n, bool), np.zeros(n, bool)])

    print("\n" + "=" * 74)
    f0, fL = rows[0]["false_call_rate"], rows[-1]["false_call_rate"]
    i0, iL = rows[0]["gt_iou_in_sample"], rows[-1]["gt_iou_in_sample"]
    print(f"in-sample ground-truth IoU   {i0:.4f} -> {iL:.4f}   ({iL-i0:+.4f})")
    print(f"false calls on clean material {f0*100:.3f}% -> {fL*100:.3f}%   "
          f"({fL/max(f0,1e-9):.1f}x)")
    print("=" * 74)
    if fL > f0 * 1.5 and abs(iL - i0) < 0.02:
        print("DRIFT REPRODUCED: the watched metric held while the unwatched one degraded.")
    elif fL <= f0 * 1.2:
        print("NO DRIFT: the false-call rate did not climb. The production drift had another")
        print("cause, and the write-up must say so.")
    else:
        print("PARTIAL: both moved. Report both columns and do not claim a clean dissociation.")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
