"""Is a newer SAM encoder a better feature source for this classifier than SAM 1 ViT-H?

    python3 code/experiment_encoders.py                    # sam1 vs whatever is reachable
    python3 code/experiment_encoders.py --arms sam1,sam2
    python3 code/experiment_encoders.py --check            # reachability only

PAIRED ON IDENTICAL PIXELS. For each reference frame the same uniformly-sampled pixel indices
are looked up in every encoder, so the 17 hand-crafted columns are byte-identical across arms
and only the 256 embedding channels differ. Same classifier, same folds, same seeds.

LEAVE-ONE-FRAME-OUT over the four dense reference frames -- which the shipped recipe holds
out of training entirely, so this protocol matches how the deployed model is judged.

THE HYBRID MEMBER IS REPORTED SEPARATELY. The deployed ensemble averages a 17-feature MLP
that is identical across arms, which halves whatever the encoder contributes; `hyb` isolates
it and `ens` is what would actually ship.

WHAT THIS CANNOT SETTLE. Every metric here is scored on LABELLED pixels. On this data
HistGradientBoosting won every such metric -- grouped IoU, AUC, cross-group AUC by four times
the noise floor -- then marked 7.9x more crack-free specimen as crack and was reverted
(docs/REFERENCE_FRAMES_AND_HGB.md). Crack-free material is precisely what nobody labels. A
win here is necessary and not sufficient; the false-positive axis has to follow before any
deployment conclusion.
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
import encoders as E         # noqa: E402

GT_CACHE = os.path.join(PROJECT, "dataset_cache")
SEEDS = (0, 1)


def mlp(seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=seed,
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


def sam1_block(stem, rr, cc):
    c1, e1 = P.gt_embedding(stem)
    if c1 is None:
        raise SystemExit(f"no SAM 1 embedding cached for {stem}")
    out = np.zeros((len(rr), e1.shape[1]), np.float32)
    todo = np.ones(len(rr), bool)
    for t in range(len(c1) - 1, -1, -1):
        y0, x0 = int(c1[t][0]), int(c1[t][1])
        sel = (todo & (rr >= y0) & (rr < y0 + M.TILE) & (cc >= x0) & (cc < x0 + M.TILE))
        if sel.any():
            out[sel] = M.interp_tile(e1[t], rr[sel] - y0, cc[sel] - x0)
            todo &= ~sel
    return out, float(M.EMB_STRIDE)


def build(stem, arms, n, rng, progress=None):
    gt = np.asarray(np.load(os.path.join(GT_CACHE, f"{stem}_gt.npy"))).astype(bool)
    feat_p = os.path.join(GT_CACHE, f"{stem}_features.npy")
    if not os.path.exists(feat_p):
        raise SystemExit(f"missing {feat_p} -- run one Retrain so the reference feature "
                         f"stacks get built")
    feat = np.load(feat_p, mmap_mode="r")
    H, W = gt.shape
    idx = np.sort(rng.choice(H * W, min(n, H * W), replace=False))
    rr, cc = np.unravel_index(idx, (H, W))
    x17 = np.asarray(feat[rr, cc, :], np.float32)
    del feat
    y = gt.ravel()[idx]
    img = None
    blocks, meta = {}, {}
    for arm in arms:
        if arm == "sam1":
            b, stride = sam1_block(stem, rr, cc)
        else:
            if img is None:
                img = np.asarray(np.load(os.path.join(GT_CACHE, f"{stem}_img.npy")), np.float32)
            c, e, stride = E.cached_embedding(arm, stem, img, progress=progress)
            b = E.rows_at(c, e, stride, rr, cc)
            del c, e
        blocks[arm] = np.concatenate([x17, b], axis=1)
        meta[arm] = dict(channels=int(b.shape[1]), stride=stride)
        del b
    del img
    return blocks, y, x17.shape[1], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="sam1,sam2,sam3")
    ap.add_argument("--rows", type=int, default=120000)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default=os.path.join(PROJECT, "research", "encoder_compare.json"))
    a = ap.parse_args()

    wanted = [s.strip() for s in a.arms.split(",") if s.strip()]
    arms = []
    for arm in wanted:
        if arm == "sam1":
            arms.append(arm); print(f"  sam1: available (the shipped baseline)"); continue
        ok, why = E.availability(arm)
        print(f"  {arm}: {'available' if ok else 'SKIPPED'}" + ("" if ok else f" -- {why}"))
        if ok:
            arms.append(arm)
    if a.check:
        return 0
    if len(arms) < 2:
        raise SystemExit("need at least two reachable arms to compare")

    stems = list(P.GT_STEMS_SHIPPED)
    print(f"\nbuilding paired rows, {a.rows:,} identical pixels per frame, arms={arms}")
    data, meta = {}, None
    for s in stems:
        t0 = time.time()
        blocks, y, n17, m = build(
            s, arms, a.rows, np.random.RandomState(abs(hash(s)) % 2**31),
            progress=lambda k, n: None)
        data[s] = (blocks, y)
        meta = m
        print(f"  {s:<18} {len(y):>8,} px  {y.mean()*100:5.1f}% crack  {time.time()-t0:>6.0f}s")
    print()
    for arm in arms:
        print(f"  {arm}: {meta[arm]['channels']} channels at stride {meta[arm]['stride']:g}")

    rows = []
    print(f"\n{'held-out frame':<18} {'arm':<6} {'model':<5} {'IoU':>7} {'AUC':>7} {'r@5%':>7}")
    print("-" * 62)
    for held in stems:
        tr = [s for s in stems if s != held]
        for seed in SEEDS:
            for arm in arms:
                Xtr = np.concatenate([data[s][0][arm] for s in tr])
                ytr = np.concatenate([data[s][1] for s in tr])
                Xte, yte = data[held][0][arm], data[held][1]
                p17 = mlp(seed).fit(Xtr[:, :n17], ytr).predict_proba(Xte[:, :n17])[:, 1]
                phy = mlp(seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                del Xtr, ytr
                for name, p in (("ens", (p17 + phy) / 2), ("hyb", phy)):
                    sc = scores(p, yte)
                    rows.append(dict(frame=held, arm=arm, model=name, seed=seed, **sc))
                    print(f"{held[:18]:<18} {arm:<6} {name:<5} {sc['iou']:>7.4f} "
                          f"{sc['auc']:>7.4f} {sc['r05']:>7.4f}", flush=True)
        print("-" * 62)

    print("\nmean over frames and seeds")
    print(f"  {'arm':<6} {'model':<5} {'IoU':>8} {'AUC':>8} {'r@5%':>8}")
    for arm in arms:
        for name in ("ens", "hyb"):
            sub = [r for r in rows if r["arm"] == arm and r["model"] == name]
            print(f"  {arm:<6} {name:<5} " + "".join(
                f"{np.mean([r[k] for r in sub]):>8.4f}" for k in ("iou", "auc", "r05")))

    base = "sam1"
    print(f"\npaired difference against {base}, per frame and seed (the only honest noise here)")
    for arm in [x for x in arms if x != base]:
        for name in ("ens", "hyb"):
            d = []
            for f in stems:
                for s in SEEDS:
                    g = lambda A: [r for r in rows if r["frame"] == f and r["seed"] == s
                                   and r["model"] == name and r["arm"] == A][0]["auc"]
                    d.append(g(arm) - g(base))
            print(f"  {arm} - {base}  {name}:  AUC {np.mean(d):+.4f}  "
                  f"sd {np.std(d, ddof=1):.4f}  n={len(d)}  "
                  f"{'(within noise)' if abs(np.mean(d)) < np.std(d, ddof=1) else ''}")

    print("\nScored on labelled pixels only. See the module docstring: this is necessary and")
    print("not sufficient -- the crack-free false-positive axis must follow.")
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(dict(meta=meta, arms=arms, rows=rows), open(a.json, "w"), indent=1)
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
