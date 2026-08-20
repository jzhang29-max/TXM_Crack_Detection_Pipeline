"""Does the model generalise OUTSIDE B2? Leave-one-specimen-group-out on the owner's labels.

    python3 code/build_label_folds.py     # once, builds the cache
    python3 code/crossval_groups.py       # then this

WHY THIS IS THE MOST IMPORTANT NUMBER IN THE PROJECT. Every other accuracy figure rests on
four dense ground-truth images, all of specimen group B2, all wide-open cracks. Nothing
measured there can say whether the model works on AM/HC, B3 or wrought material, and a
reviewer or a colleague will ask exactly that. Until the owner's 30.2 M hand-drawn labels
were sampled into a fold cache, leave-one-specimen-GROUP-out was not constructible at all.
It is now: 71 labelled images across AM/HC (27), B2 (17), B3 (13) and wrought (14).

SPARSE LABELS NEED A DIFFERENT METRIC, and this is the part to get right. Dense ground
truth labels every pixel, which is what makes IoU meaningful -- you need to know the false
negatives. A correction mask is sparse: `corr == 0` means "the owner expressed no opinion
here", not "not crack". Measured on six images, treating corrections as dense ground truth
gives a mean IoU of 0.06, because every crack the model found CORRECTLY and the owner never
painted over is counted as a false positive. Restricted to pixels the owner actually
judged, the same images give 0.70-0.997.

So this reports AGREEMENT ON JUDGED PIXELS, never a whole-image IoU:
  * crack recall      -- of pixels the owner marked crack, what fraction does the model?
  * not-crack agreement -- of pixels the owner marked NOT crack, what fraction does the
                         model leave alone? (one minus this is a false-alarm rate on
                         material the owner has looked at and rejected)

WHAT IT CANNOT TELL YOU, stated up front. 98.3% of the force-crack labels sit on pixels the
model already called crack, because Flip region confirms a whole blob in one click. So
crack recall is partly circular: it measures whether a newly fitted model still agrees with
what an older one found and the owner accepted. The not-crack side is less circular but is
dominated by imported research negatives covering large background regions. Read a high
number here as "consistent with the owner's judgement", not as "accurate".

Training for each fold uses the other groups' labelled pixels PLUS the four dense
ground-truth images, which is what a real retrain sees. The held-out group contributes
nothing to training.
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

CACHE = os.path.join(PROJECT, "paint", "label_folds.npz")
TRAIN_CAP = 150000


def load():
    if not os.path.exists(CACHE):
        raise SystemExit(f"no {os.path.relpath(CACHE, PROJECT)} -- run "
                         f"python3 code/build_label_folds.py first")
    z = np.load(CACHE, allow_pickle=False)
    ids = sorted({k.split("|")[0] for k in z.files})
    per = {}
    for iid in ids:
        if f"{iid}|x17" not in z.files or f"{iid}|y" not in z.files:
            continue
        xs = z[f"{iid}|xsam"] if f"{iid}|xsam" in z.files else None
        if xs is None:
            continue
        per[iid] = dict(x17=z[f"{iid}|x17"], xsam=xs, y=z[f"{iid}|y"],
                        group=str(z[f"{iid}|group"][0]))
    return per


def gt_rows(n_per_image=25000):
    """The dense ground-truth rows a real retrain also trains on."""
    rng = np.random.RandomState(5)
    X, y = [], []
    for stem in P.GT_STEMS:
        got = P._gt_rows(stem, n_per_image, rng)
        if got is None:
            continue
        block, yy, _ = got
        X.append(block); y.append(yy)
    if not X:
        return None, None
    return np.concatenate(X), np.concatenate(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args()

    per = load()
    groups = {}
    for iid, d in per.items():
        groups.setdefault(d["group"], []).append(iid)
    print(f"{len(per)} labelled images, {len(groups)} specimen groups")
    for g in sorted(groups):
        n = sum(len(per[i]["y"]) for i in groups[g])
        c = sum(int(per[i]["y"].sum()) for i in groups[g])
        print(f"    {g:<9} {len(groups[g]):>3} images  {n:>8,} sampled px  "
              f"{100*c/max(n,1):4.1f}% crack")

    gX, gy = gt_rows()
    print(f"\nplus {0 if gX is None else len(gy):,} dense ground-truth rows in every fold "
          f"(all four are B2)")

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def clf():
        return Pipeline([("s", StandardScaler()),
                         ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                             early_stopping=True, n_iter_no_change=8))])

    print(f"\n{'held-out group':<12} {'images':>7} {'judged px':>11} "
          f"{'crack recall':>13} {'not-crack agree':>16}")
    print("-" * 68)
    out = []
    for held in sorted(groups):
        tr17, trS, trY = [], [], []
        for g, ids in groups.items():
            if g == held:
                continue
            for i in ids:
                tr17.append(per[i]["x17"]); trS.append(per[i]["xsam"]); trY.append(per[i]["y"])
        Xtr = np.concatenate([np.concatenate(tr17), np.concatenate(trS)], axis=1)
        ytr = np.concatenate(trY)
        if gX is not None and gX.shape[1] == Xtr.shape[1]:
            Xtr = np.concatenate([Xtr, gX]); ytr = np.concatenate([ytr, gy])
        if len(ytr) > TRAIN_CAP:
            sub = np.random.RandomState(7).choice(len(ytr), TRAIN_CAP, replace=False)
            Xtr, ytr = Xtr[sub], ytr[sub]

        te17 = np.concatenate([per[i]["x17"] for i in groups[held]])
        teS = np.concatenate([per[i]["xsam"] for i in groups[held]])
        teY = np.concatenate([per[i]["y"] for i in groups[held]])
        Xte = np.concatenate([te17, teS], axis=1)

        m17 = clf().fit(Xtr[:, :te17.shape[1]], ytr)
        mhy = clf().fit(Xtr, ytr)
        prob = (m17.predict_proba(Xte[:, :te17.shape[1]])[:, 1]
                + mhy.predict_proba(Xte)[:, 1]) / 2.0
        pred = prob >= 0.5
        rec = float(pred[teY].mean()) if teY.any() else float("nan")
        agr = float((~pred[~teY]).mean()) if (~teY).any() else float("nan")
        out.append(dict(group=held, images=len(groups[held]), judged=int(len(teY)),
                        crack_recall=round(rec, 4), not_crack_agreement=round(agr, 4),
                        crack_px=int(teY.sum()), not_crack_px=int((~teY).sum())))
        print(f"{held:<12} {len(groups[held]):>7} {len(teY):>11,} "
              f"{rec:>12.4f}  {agr:>15.4f}")
        del Xtr, ytr, Xte, m17, mhy

    print("-" * 68)
    r = [o["crack_recall"] for o in out if o["crack_recall"] == o["crack_recall"]]
    g_ = [o["not_crack_agreement"] for o in out if o["not_crack_agreement"] == o["not_crack_agreement"]]
    print(f"{'MEAN':<12} {'':>7} {'':>11} {np.mean(r):>12.4f}  {np.mean(g_):>15.4f}")
    print("\nThis is agreement with the owner's judgement on pixels the owner judged --")
    print("not IoU, and not accuracy against physical truth. See the module docstring.")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
