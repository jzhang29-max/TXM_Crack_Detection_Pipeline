"""Does SAM 3's encoder beat SAM 1's as a feature source for this classifier?

    python3 code/experiment_sam3.py                 # the comparison
    python3 code/experiment_sam3.py --check         # just say whether SAM 3 is reachable

PAIRED ON IDENTICAL PIXELS. For each reference frame the same uniformly-sampled pixel
indices are looked up in BOTH encoders, so the only thing that differs between the two arms
is the 256 embedding channels. Same 17 hand-crafted features, same classifier, same folds,
same seeds. Anything left is the encoder.

LEAVE-ONE-FRAME-OUT over the four dense reference frames, which nothing trains on in the
shipped recipe -- so this protocol matches how the deployed model is actually judged.

REPORTS THE HYBRID MEMBER ON ITS OWN, not just the ensemble. The ensemble averages a
17-feature MLP that is byte-identical between arms, which dilutes whatever the encoder
contributes by roughly half. `hyb` is the number that isolates it.

WHAT THIS CANNOT SETTLE, and the reason is measured. Every metric here is scored on labelled
pixels. HistGradientBoosting won every such metric on this data -- grouped IoU, AUC,
cross-group AUC by four times the noise floor -- and then marked 7.9x more crack-free
specimen as crack and was reverted (docs/REFERENCE_FRAMES_AND_HGB.md). Crack-free material
is exactly what nobody labels, so a win here is necessary and NOT sufficient. Pass --fp to
add that axis before drawing any conclusion about deploying.
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

import model as M            # noqa: E402
import pipeline as P         # noqa: E402
import sam3_encoder as S3    # noqa: E402

GT_CACHE = os.path.join(PROJECT, "dataset_cache")
ROWS_PER_FRAME = 120000
SEEDS = (0, 1)


def mlp():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def scores(p, y):
    from sklearn.metrics import roc_auc_score, roc_curve
    pred = p > 0.5
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    out = dict(iou=round(tp / max(tp + fp + fn, 1), 4),
               recall=round(tp / max(tp + fn, 1), 4))
    if y.any() and (~y).any():
        out["auc"] = round(float(roc_auc_score(y, p)), 4)
        f, t, _ = roc_curve(y, p)
        out["r05"] = round(float(np.interp(0.05, f, t)), 4)
    else:
        out["auc"] = out["r05"] = float("nan")
    return out


def build(stem, n, rng):
    """({'sam1': X1, 'sam3': X3}, y, n17) at the SAME pixels for both encoders."""
    gt = np.asarray(np.load(os.path.join(GT_CACHE, f"{stem}_gt.npy"))).astype(bool)
    feat_p = os.path.join(GT_CACHE, f"{stem}_features.npy")
    if not os.path.exists(feat_p):
        raise SystemExit(f"missing {feat_p} -- run a Retrain once so ensure_gt_features "
                         f"builds the reference feature stacks")
    feat = np.load(feat_p, mmap_mode="r")
    H, W = gt.shape
    idx = np.sort(rng.choice(H * W, min(n, H * W), replace=False))
    rr, cc = np.unravel_index(idx, (H, W))
    x17 = np.asarray(feat[rr, cc, :], np.float32)
    del feat
    y = gt.ravel()[idx]

    # SAM 1, through the project's own cache
    c1, e1 = P.gt_embedding(stem)
    if c1 is None:
        raise SystemExit(f"no SAM 1 embedding for {stem}")
    b1 = np.zeros((len(rr), e1.shape[1]), np.float32)
    todo = np.ones(len(rr), bool)
    for t in range(len(c1) - 1, -1, -1):
        y0, x0 = int(c1[t][0]), int(c1[t][1])
        sel = (todo & (rr >= y0) & (rr < y0 + M.TILE) & (cc >= x0) & (cc < x0 + M.TILE))
        if sel.any():
            b1[sel] = M.interp_tile(e1[t], rr[sel] - y0, cc[sel] - x0)
            todo &= ~sel

    # SAM 3, same pixels
    img = np.asarray(np.load(os.path.join(GT_CACHE, f"{stem}_img.npy")), np.float32)
    c3, e3, stride = S3.cached_embedding(stem, img)
    b3 = S3.rows_at(c3, e3, stride, rr, cc)
    del img
    return (dict(sam1=np.concatenate([x17, b1], axis=1),
                 sam3=np.concatenate([x17, b3], axis=1)),
            y, x17.shape[1], dict(sam1_dim=int(b1.shape[1]), sam3_dim=int(b3.shape[1]),
                                  sam3_stride=stride))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report reachability and exit")
    ap.add_argument("--rows", type=int, default=ROWS_PER_FRAME)
    ap.add_argument("--json", default=os.path.join(PROJECT, "research", "sam3_vs_sam1.json"))
    a = ap.parse_args()

    ok, why = S3.availability()
    print(f"SAM 3 reachable: {ok}")
    if not ok:
        print(f"  {why}")
        if a.check:
            return 0
        raise SystemExit("cannot run the comparison until SAM 3 can be downloaded")
    if a.check:
        return 0

    stems = list(P.GT_STEMS_SHIPPED)
    print(f"\nbuilding paired rows for {len(stems)} reference frames "
          f"({a.rows:,} px each, identical pixels in both arms)")
    data, meta = {}, None
    for s in stems:
        t0 = time.time()
        X, y, n17, info = build(s, a.rows, np.random.RandomState(hash(s) % 2**31))
        data[s] = (X, y)
        meta = info
        print(f"  {s:<18} {len(y):>8,} px  {y.mean()*100:5.1f}% crack  {time.time()-t0:>5.0f}s")
    print(f"\n  SAM 1: {meta['sam1_dim']} channels at stride {M.EMB_STRIDE}")
    print(f"  SAM 3: {meta['sam3_dim']} channels at stride {meta['sam3_stride']:g}")

    rows = []
    print(f"\n{'held-out frame':<18} {'arm':<6} {'model':<5} {'IoU':>7} {'AUC':>7} {'r@5%':>7}")
    print("-" * 62)
    for held in stems:
        tr = [s for s in stems if s != held]
        for seed in SEEDS:
            for arm in ("sam1", "sam3"):
                Xtr = np.concatenate([data[s][0][arm] for s in tr])
                ytr = np.concatenate([data[s][1] for s in tr])
                Xte, yte = data[held][0][arm], data[held][1]
                m17 = mlp(); m17.set_params(m__random_state=seed)
                mhy = mlp(); mhy.set_params(m__random_state=seed)
                p17 = m17.fit(Xtr[:, :17], ytr).predict_proba(Xte[:, :17])[:, 1]
                phy = mhy.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                del Xtr, ytr
                for name, p in (("ens", (p17 + phy) / 2), ("hyb", phy)):
                    sc = scores(p, yte)
                    rows.append(dict(frame=held, arm=arm, model=name, seed=seed, **sc))
                    print(f"{held[:18]:<18} {arm:<6} {name:<5} {sc['iou']:>7.4f} "
                          f"{sc['auc']:>7.4f} {sc['r05']:>7.4f}", flush=True)
        print("-" * 62)

    print("\nmean over frames and seeds")
    print(f"  {'arm':<6} {'model':<5} {'IoU':>8} {'AUC':>8} {'r@5%':>8}")
    for arm in ("sam1", "sam3"):
        for name in ("ens", "hyb"):
            sub = [r for r in rows if r["arm"] == arm and r["model"] == name]
            print(f"  {arm:<6} {name:<5} " + "".join(
                f"{np.mean([r[k] for r in sub]):>8.4f}" for k in ("iou", "auc", "r05")))
    # paired per-frame spread, which is the only honest noise estimate here
    for name in ("ens", "hyb"):
        d = []
        for f in stems:
            for s in SEEDS:
                a1 = [r for r in rows if r["frame"] == f and r["seed"] == s
                      and r["model"] == name and r["arm"] == "sam1"][0]["auc"]
                a3 = [r for r in rows if r["frame"] == f and r["seed"] == s
                      and r["model"] == name and r["arm"] == "sam3"][0]["auc"]
                d.append(a3 - a1)
        print(f"  {name}: SAM3 - SAM1 AUC per frame/seed  mean {np.mean(d):+.4f}  "
              f"sd {np.std(d, ddof=1):.4f}  n={len(d)}")

    print("\nA win here is NOT sufficient to deploy: every number above is scored on")
    print("labelled pixels, and that is exactly the blind spot that let HistGradientBoosting")
    print("pass while marking 7.9x more crack-free specimen as crack.")
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(dict(meta=meta, rows=rows), open(a.json, "w"), indent=1)
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
