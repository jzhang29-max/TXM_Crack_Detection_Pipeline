"""Does the model still work if the four B2 reference images are dropped from TRAINING?

    python3 code/experiment_no_gt.py --eval gt       # test on the 4 dense B2 frames
    python3 code/experiment_no_gt.py --eval groups   # test on the owner's labels, by group

THE QUESTION. Every model in this project trains on the four dense B2 ground-truth images
plus the owner's corrections. Those four are one specimen group, so the worry is that they
anchor the model to B2. The owner has labelled all 71 images across four groups, which is
enough to train on alone -- so: drop the four, train on corrections only, and measure what
changes.

A SIDE EFFECT THAT MATTERS MORE THAN THE ANSWER. While the four are in training they cannot
honestly be a test set, and they are exactly what the retrain gate validates against. Drop
them from training and they become a genuine held-out set for the first time.

THREE TRAINING COMPOSITIONS, one fixed row budget so the comparison is about composition and
not about sample size:

    gt+corr    the current recipe: reference rows + the owner's corrections
    corr-only  the owner's corrections alone            <- the proposal
    gt-only    reference rows alone, for reference

FOUR MODELS on each, all on identical rows:

    ens   deployed: mean probability of MLP(17) and MLP(17+SAM)
    hyb   MLP(17+SAM) alone
    f17   MLP(17) alone
    hgb   HistGradientBoosting(17+SAM) -- a different model class, not just a different head

TWO TEST SETS, and the metrics differ between them on purpose.

  --eval gt: the four dense B2 frames, leave-one-SPECIMEN-out. Each frame's pixels are
  sampled UNIFORMLY, so IoU on the sample is an unbiased estimate of the whole frame's IoU.
  Specimen-level exclusion, not image-level: all four frames are also loaded in the app with
  the owner's corrections, and b2_343_75 and b2_343_75_LARGE are the same specimen, so
  training on one while testing the other would leak.

  --eval groups: the owner's labels, leave-one-GROUP-out. These rows are a sample of labels
  and not uniform over the frame, so an IoU computed on them would be biased by the sampling
  ratio. Reported instead are RECALL on crack rows and FALSE-POSITIVE RATE on not-crack rows
  -- both conditional on the true class, so both are unbiased under any sampling scheme.
  Note that when the held-out group is B2 the gt+corr arm loses its reference rows too,
  because all four reference images are B2. That is not a bug; it is the point.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import store as S            # noqa: E402
import pipeline as P         # noqa: E402

CACHE = os.path.join(PROJECT, "paint", "label_folds.npz")
TRAIN_CAP = 150000
GT_ROWS_TRAIN = 25000
GT_ROWS_TEST = 250000
ARMS = ("gt+corr", "corr-only", "gt-only", "corr-big")
BIG_CAP = 500000
# stem -> the specimen token that identifies every app image of the same specimen
STEM_SPECIMEN = {"333_75_um_zoom": "333_75", "336_25": "336_25",
                 "338_13": "338_13", "LARGE_343_75": "343_75"}


def group_of(fn):
    n = (fn or "").lower()
    return ("wrought" if "wrought" in n else "AM/HC" if "hc_316l" in n
            else "B3" if "_b3_" in n or "b3_" in n else "B2")


def mlp():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, early_stopping=True,
                                          random_state=0)


def fit_all(X, y, n17):
    """The four models, on identical rows."""
    t0 = time.time()
    m17 = mlp().fit(X[:, :n17], y)
    mhy = mlp().fit(X, y)
    mgb = hgb().fit(X, y)
    return dict(m17=m17, mhy=mhy, mgb=mgb, fit_s=time.time() - t0)


def probs(models, X, n17):
    p17 = models["m17"].predict_proba(X[:, :n17])[:, 1]
    phy = models["mhy"].predict_proba(X)[:, 1]
    pgb = models["mgb"].predict_proba(X)[:, 1]
    return {"ens": (p17 + phy) / 2.0, "hyb": phy, "f17": p17, "hgb": pgb}


def scores(p, y):
    """Threshold metrics plus two that do not depend on where the threshold sits.

    Comparing arms at a fixed 0.5 is misleading when they are calibrated differently: an arm
    that simply calls more of the image crack buys recall with false positives and looks
    better on recall alone. AUC is threshold-free, and recall-at-matched-FPR puts every arm
    at the same operating point, which is the comparison an inspector actually cares about.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    pred = p > 0.5
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    out = dict(iou=round(tp / max(tp + fp + fn, 1), 4),
               precision=round(tp / max(tp + fp, 1), 4),
               recall=round(tp / max(tp + fn, 1), 4),
               fpr=round(fp / max(fp + tn, 1), 4), n=int(len(y)))
    if y.any() and (~y).any():
        out["auc"] = round(float(roc_auc_score(y, p)), 4)
        fprs, tprs, _ = roc_curve(y, p)
        for tgt in (0.05, 0.10):
            out[f"r{int(tgt*100):02d}"] = round(float(np.interp(tgt, fprs, tprs)), 4)
    else:
        out["auc"] = float("nan"); out["r05"] = float("nan"); out["r10"] = float("nan")
    return out


def load_corrections():
    z = np.load(CACHE, allow_pickle=False)
    meta = {m["id"]: m for m in S.list_images()}
    ids = sorted({k.split("|")[0] for k in z.files})
    out = []
    for i in ids:
        kx, ks, ky = f"{i}|x17", f"{i}|xsam", f"{i}|y"
        if not (kx in z.files and ks in z.files and ky in z.files):
            continue
        fn = meta.get(i, {}).get("filename") or ""
        out.append(dict(iid=i, filename=fn, group=group_of(fn),
                        x17=z[kx], xsam=z[ks], y=z[ky].astype(bool)))
    return out


def corr_rows(recs, keep, rng, cap):
    """Concatenate [17|256] rows from records passing keep(), capped."""
    sel = [r for r in recs if keep(r)]
    if not sel:
        return None, None
    X = np.concatenate([np.concatenate([r["x17"], r["xsam"]], axis=1) for r in sel]).astype(np.float32)
    Y = np.concatenate([r["y"] for r in sel])
    if len(Y) > cap:
        i = rng.choice(len(Y), cap, replace=False)
        X, Y = X[i], Y[i]
    return X, Y


def gt_rows(stems, n, rng):
    X, Y, n17 = [], [], None
    for st in stems:
        got = P._gt_rows(st, n, rng)
        if got:
            X.append(got[0]); Y.append(got[1].astype(bool)); n17 = got[2]
    if not X:
        return None, None, None
    return np.concatenate(X).astype(np.float32), np.concatenate(Y), n17


def build(arm, gt_stems_train, keep, recs, rng, n17):
    """Rows for one arm at a fixed total budget."""
    if arm == "gt-only":
        X, Y, _ = gt_rows(gt_stems_train, TRAIN_CAP // max(len(gt_stems_train), 1), rng)
        return X, Y
    if arm == "corr-only":
        return corr_rows(recs, keep, rng, TRAIN_CAP)
    if arm == "corr-big":
        # the fixed budget exists to compare composition; but with the reference frames gone
        # there are 1.03 M correction rows available, so this asks what the proposal is
        # actually worth when it is not held to the smaller arm's sample size.
        return corr_rows(recs, keep, rng, BIG_CAP)
    gX, gY, _ = gt_rows(gt_stems_train, GT_ROWS_TRAIN, rng)
    room = TRAIN_CAP - (0 if gX is None else len(gY))
    cX, cY = corr_rows(recs, keep, rng, max(room, 1))
    parts = [(a, b) for a, b in ((gX, gY), (cX, cY)) if a is not None]
    if not parts:
        return None, None
    return (np.concatenate([a for a, _ in parts]),
            np.concatenate([b for _, b in parts]))


def eval_gt(recs, out_json, repeats=3):
    rng = np.random.RandomState(7)
    rows = []
    print(f"\nTEST: the 4 dense B2 reference frames, uniform pixel sample "
          f"({GT_ROWS_TEST:,} px each), leave-one-specimen-out")
    print(f"train budget {TRAIN_CAP:,} rows for every arm\n")
    print(f"{'held-out frame':<16} {'r':<2} {'arm':<10} {'model':<5} {'IoU':>7} {'AUC':>7} {'r@5%':>7}")
    print("-" * 72)
    for st in P.GT_STEMS_SHIPPED:
        spec = STEM_SPECIMEN[st]
        tX, tY, n17 = gt_rows([st], GT_ROWS_TEST, rng)
        if tX is None:
            print(f"  {st}: no features, skipped"); continue
        others = [s for s in P.GT_STEMS_SHIPPED if s != st]
        # exclude every app image of this specimen -- image-level would leak
        keep = lambda r, spec=spec: spec not in (r["filename"] or "").lower()
        n_excl = sum(1 for r in recs if not keep(r))
        for rep in range(repeats):
            for arm in ARMS:
                X, Y = build(arm, others, keep, recs, np.random.RandomState(100 + rep), n17)
                if X is None:
                    continue
                models = fit_all(X, Y, n17)
                for name, p in probs(models, tX, n17).items():
                    sc = scores(p, tY)
                    rows.append(dict(test=st, specimen=spec, arm=arm, model=name, rep=rep,
                                     train_n=int(len(Y)), excluded_images=n_excl, **sc))
                    print(f"{st[:16]:<16} r{rep} {arm:<10} {name:<5} {sc['iou']:>7.4f} "
                          f"{sc['auc']:>7.4f} {sc['r05']:>7.4f}", flush=True)
        print("-" * 72)
    summarise(rows, "IoU @0.5 on held-out B2 reference frames (unbiased, uniform sample)", "iou")
    summarise(rows, "AUC -- threshold-free, so calibration cannot flatter an arm", "auc")
    summarise(rows, "recall at matched 5% FPR -- same operating point for every arm", "r05")
    json.dump(rows, open(out_json, "w"), indent=1)
    print(f"\nwrote {out_json}")


def eval_groups(recs, out_json, repeats=3):
    rng = np.random.RandomState(11)
    rows = []
    groups = sorted({r["group"] for r in recs})
    print("\nTEST: the owner's labels, leave-one-GROUP-out")
    print("recall on crack rows and FPR on not-crack rows -- both conditional on the true")
    print("class, so both are unbiased even though the rows are a sample of labels.\n")
    print(f"{'group':<10} {'r':<2} {'arm':<10} {'model':<5} {'recall':>8} {'FPR':>8} {'AUC':>8} {'r@5%':>8}")
    print("-" * 62)
    n17 = 17
    for g in groups:
        keep = lambda r, g=g: r["group"] != g
        tX, tY = corr_rows(recs, lambda r, g=g: r["group"] == g, rng, 300000)
        if tX is None:
            continue
        # all four reference frames are B2, so holding out B2 must drop them too
        others = [] if g == "B2" else list(P.GT_STEMS_SHIPPED)
        for rep in range(repeats):
            for arm in ARMS:
                if arm == "gt-only" and not others:
                    continue
                if arm == "gt+corr" and not others:
                    continue   # identical to corr-only when there is no reference data to add
                X, Y = build(arm, others, keep, recs, np.random.RandomState(200 + rep), n17)
                if X is None:
                    continue
                models = fit_all(X, Y, n17)
                for name, p in probs(models, tX, n17).items():
                    sc = scores(p, tY)
                    rows.append(dict(test_group=g, arm=arm, model=name, rep=rep,
                                     train_n=int(len(Y)),
                                     gt_in_train=bool(others and arm != "corr-only"), **sc))
                    print(f"{g:<10} r{rep} {arm:<10} {name:<5} {sc['recall']:>8.4f} "
                          f"{sc['fpr']:>8.4f} {sc['auc']:>8.4f} {sc['r05']:>8.4f}", flush=True)
        print("-" * 62)
    summarise(rows, "crack recall @0.5 on held-out groups", "recall")
    summarise(rows, "false-positive rate @0.5 (lower is better) -- read WITH the recall above", "fpr")
    summarise(rows, "AUC -- threshold-free, the fair single number", "auc")
    summarise(rows, "recall at matched 5% FPR -- same operating point for every arm", "r05")
    summarise(rows, "recall at matched 10% FPR", "r10")
    json.dump(rows, open(out_json, "w"), indent=1)
    print(f"\nwrote {out_json}")


def summarise(rows, title, key):
    print(f"\n{title}")
    models = sorted({r["model"] for r in rows})
    print("  " + f"{'arm':<10}" + "".join(f"{m:>9}" for m in models))
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        line = f"  {arm:<10}"
        for m in models:
            v = [r[key] for r in sub if r["model"] == m and r[key] == r[key]]
            line += f"{np.mean(v):>9.4f}" if v else f"{'—':>9}"
        print(line)
    reps = sorted({r.get("rep", 0) for r in rows})
    if len(reps) > 1:
        spread = []
        for cell in {(r["arm"], r["model"], r.get("test") or r.get("test_group")) for r in rows}:
            v = [r[key] for r in rows if (r["arm"], r["model"],
                 r.get("test") or r.get("test_group")) == cell and r[key] == r[key]]
            if len(v) > 1:
                spread.append(np.std(v, ddof=1))
        if spread:
            print(f"  {'noise':<10}" + f"±{np.mean(spread):.4f} mean sd across "
                  f"{len(reps)} independent row samples -- differences smaller than this "
                  f"are not real")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=("gt", "groups"), required=True)
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent row samples per cell; the spread is the noise floor")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if not os.path.exists(CACHE):
        raise SystemExit("run python3 code/build_label_folds.py first")
    recs = load_corrections()
    print(f"loaded corrections from {len(recs)} images, "
          f"{sum(len(r['y']) for r in recs):,} rows")
    out = a.json or os.path.join(PROJECT, "research", f"no_gt_{a.eval}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    (eval_gt if a.eval == "gt" else eval_groups)(recs, out, a.repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
