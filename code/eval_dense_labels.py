"""IoU on the images the owner labelled densely — including, for the first time, AM/HC.

    python3 code/build_label_folds.py        # once
    python3 code/eval_dense_labels.py

WHAT THIS CORRECTS. This project repeatedly said "dense ground truth exists for four images,
all specimen group B2", and used that to caveat every number. That was wrong by omission. The
owner labelled all 71 images, and while most are sparse, **ten are 90-100% judged** — and five
of those are AM/HC frames that contain crack. Sparse labels cannot give an IoU because IoU
needs the false negatives; these can.

MASKED IoU, over judged pixels only. A pixel the owner never marked is not evidence of
background, so scoring it as background would count missing labels as false negatives. It is
excluded instead. That exclusion is safe here and was checked rather than assumed: on these
frames 67-75% of the unjudged pixels are off-specimen surround, and the unjudged-AND-on-
specimen remainder is only 1.4-3.3% of the frame. So the masked estimate covers essentially
all the material a crack could be in.

HELD OUT BY IMAGE. For each target frame, the model is refit on every OTHER labelled image
plus the four dense B2 ground-truth images, then predicts the target's full frame. The target
contributes nothing to its own score. Full-frame prediction, not sampled pixels, because a
masked IoU needs the real spatial mask.

WHY IT MATTERS. Every accuracy figure in this repo came from four B2 images of wide-open
cracks. This is the first IoU measured on additively-manufactured material, on frames the
owner judged themselves.
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
import model as M            # noqa: E402
from txm_features import compute_feature_stack   # noqa: E402

CACHE = os.path.join(PROJECT, "paint", "label_folds.npz")
MIN_JUDGED = 0.90
MIN_CRACK = 5000
TRAIN_CAP = 150000
BAND_BYTES = 400e6


def clf():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def group_of(fn):
    n = (fn or "").lower()
    return ("wrought" if "wrought" in n else "AM/HC" if "hc_316l" in n
            else "B3" if "_b3_" in n or "b3_" in n else "B2")


def dense_targets():
    out = []
    for m in S.list_images():
        c, n = S.correction_counts(m["id"])
        tot = (m.get("width") or 1) * (m.get("height") or 1)
        if c >= MIN_CRACK and (c + n) / tot >= MIN_JUDGED:
            out.append((m, (c + n) / tot, c))
    return sorted(out, key=lambda t: -t[1])


def predict_full(iid, m17, mhy, n17):
    """Banded full-frame prediction from a freshly computed feature stack."""
    img = np.asarray(S.load_npy(iid, "img.npy"))
    Hh, Ww = img.shape
    z = np.load(S.path(iid, "emb.npz"))
    coords, embs = z["coords"], z["emb"]
    out = np.zeros((Hh, Ww), np.float32)
    rows = max(16, int(BAND_BYTES / (Ww * (n17 + embs.shape[1]) * 4)))
    # The 17-feature stack must be computed on the WHOLE frame, not per band: its largest
    # smoothing sigma reaches ~256 px, and a banded stack disagrees with a full one out to
    # exactly that distance (measured elsewhere in this repo).
    feats = np.asarray(compute_feature_stack(img), np.float32)
    for r0 in range(0, Hh, rows):
        r1 = min(r0 + rows, Hh)
        blk = feats[r0:r1].reshape(-1, n17)
        rr = np.repeat(np.arange(r0, r1), Ww)
        cc = np.tile(np.arange(Ww), r1 - r0)
        sam = np.zeros((len(rr), embs.shape[1]), np.float32)
        todo = np.ones(len(rr), bool)
        for t in range(len(coords) - 1, -1, -1):
            y0, x0 = int(coords[t][0]), int(coords[t][1])
            sel = (todo & (rr >= y0) & (rr < y0 + M.TILE)
                   & (cc >= x0) & (cc < x0 + M.TILE))
            if sel.any():
                sam[sel] = M.interp_tile(embs[t], rr[sel] - y0, cc[sel] - x0)
                todo &= ~sel
        p = (m17.predict_proba(blk)[:, 1]
             + mhy.predict_proba(np.concatenate([blk, sam], axis=1))[:, 1]) / 2.0
        out[r0:r1] = p.reshape(r1 - r0, Ww)
        del blk, rr, cc, sam, p
    del feats, img
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args()
    if not os.path.exists(CACHE):
        raise SystemExit("run python3 code/build_label_folds.py first")
    z = np.load(CACHE, allow_pickle=False)

    tgt = dense_targets()
    print(f"{len(tgt)} image(s) at least {MIN_JUDGED*100:.0f}% judged with "
          f">= {MIN_CRACK:,} crack px:\n")
    for m, cov, c in tgt:
        print(f"  {group_of(m.get('filename')):<8} {cov*100:5.1f}% judged  "
              f"{c:>9,} crack px  {(m.get('filename') or '')[22:56]}")

    rng = np.random.RandomState(5)
    gtX, gtY = [], []
    for stem in P.GT_STEMS_SHIPPED:
        got = P._gt_rows(stem, 25000, rng)
        if got:
            gtX.append(got[0]); gtY.append(got[1]); n17 = got[2]
    gtX = np.concatenate(gtX); gtY = np.concatenate(gtY)

    print(f"\n{'image':<34} {'group':<8} {'masked IoU':>11} {'prec':>7} {'recall':>7} {'s':>5}")
    print("-" * 78)
    rows = []
    for m, cov, c in tgt:
        t0 = time.time()
        # train on every OTHER labelled image, plus the dense B2 ground truth
        X, Y = [gtX], [gtY]
        for iid in sorted({k.split("|")[0] for k in z.files}):
            if iid == m["id"]:
                continue
            kx, ks, ky = f"{iid}|x17", f"{iid}|xsam", f"{iid}|y"
            if not (kx in z.files and ks in z.files and ky in z.files):
                continue
            X.append(np.concatenate([z[kx], z[ks]], axis=1).astype(np.float32))
            Y.append(z[ky].astype(bool))
        X = np.concatenate(X); Y = np.concatenate(Y)
        if len(Y) > TRAIN_CAP:
            i = rng.choice(len(Y), TRAIN_CAP, replace=False)
            X, Y = X[i], Y[i]
        m17 = clf().fit(X[:, :n17], Y)
        mhy = clf().fit(X, Y)
        del X, Y

        prob = predict_full(m["id"], m17, mhy, n17)
        pred = P.prune_specks(prob > 0.5)
        corr = np.asarray(S.load_npy(m["id"], "correction.npy"))
        judged = corr != 0
        crack = corr == 1
        tp = int((pred & crack).sum())
        fp = int((pred & judged & ~crack).sum())
        fn = int((~pred & crack).sum())
        iou = tp / max(tp + fp + fn, 1)
        rows.append(dict(image=m.get("filename"), group=group_of(m.get("filename")),
                         judged_fraction=round(cov, 4), iou=round(iou, 4),
                         precision=round(tp / max(tp + fp, 1), 4),
                         recall=round(tp / max(tp + fn, 1), 4),
                         crack_px=int(crack.sum())))
        print(f"{(m.get('filename') or '')[22:56]:<34} {rows[-1]['group']:<8} "
              f"{iou:>11.4f} {rows[-1]['precision']:>7.3f} {rows[-1]['recall']:>7.3f} "
              f"{time.time()-t0:>5.0f}", flush=True)
        del prob, pred, corr, judged, crack, m17, mhy

    print("-" * 78)
    for g in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == g]
        print(f"{g:<34} {'mean':<8} {np.mean([r['iou'] for r in sub]):>11.4f} "
              f"{np.mean([r['precision'] for r in sub]):>7.3f} "
              f"{np.mean([r['recall'] for r in sub]):>7.3f}  (n={len(sub)})")
    print("\nMasked IoU: scored over pixels the owner judged. Unjudged pixels are excluded,")
    print("not counted as background -- on these frames they are 67-75% off-specimen and the")
    print("unjudged-on-specimen remainder is 1.4-3.3% of the frame.")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
