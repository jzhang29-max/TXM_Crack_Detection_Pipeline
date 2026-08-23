"""Does halving the SAM embedding stride tighten the predicted crack boundary?

A controlled A/B on 1024x1024 crops, ~40 SAM passes instead of the 4,200 a full re-embed
would need. If stride 8 does not measurably tighten the boundary here, it will not justify
7.2 hours and 8.4 GB there.

HOW STRIDE 8 IS OBTAINED. SAM gives a 64x64 grid per 1024-px tile: one 256-d vector per
16x16 block, sample centres at 16k+8. Running it again on the tile shifted by 8 px puts the
centres at 16k+16. Interleaving the four phase shifts (0,0) (0,8) (8,0) (8,8) yields a
128x128 grid with centres every 8 px -- genuinely finer sampling, not interpolation of the
coarse grid. The combined centres are at 8(n+1), so the lookup origin shifts too; getting
that wrong would blur the very thing being measured.
"""
import os, sys
import numpy as np
PROJECT = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, os.path.join(PROJECT, "code"))
import model as M, store as S, pipeline as P
from txm_features import compute_feature_stack
from skimage.morphology import medial_axis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CROP = 1024
CACHE = os.path.join(PROJECT, "research", "logs", "pilot_emb")
os.makedirs(CACHE, exist_ok=True)


def embed_grid(img, offs):
    """SAM on img[dy:dy+CROP, dx:dx+CROP] for each offset -> list of (dy,dx,64x64xC)."""
    out = []
    for dy, dx in offs:
        sub = img[dy:dy + CROP, dx:dx + CROP]
        coords, emb = M.embed_image(sub)
        g = np.asarray(emb[0], np.float32).transpose(1, 2, 0)     # 64,64,C
        out.append((dy, dx, g))
    return out


def lookup(grids, stride, origin, rr, cc, C):
    """Bilinear lookup into an interleaved grid with an explicit sample origin."""
    n = int(CROP // stride)
    big = np.zeros((n, n, C), np.float32)
    for dy, dx, g in grids:
        big[(dy // stride)::(16 // stride), (dx // stride)::(16 // stride)] = g
    r = np.clip((rr - origin) / stride, 0, n - 1)
    c = np.clip((cc - origin) / stride, 0, n - 1)
    r0 = np.floor(r).astype(np.intp); c0 = np.floor(c).astype(np.intp)
    r1 = np.minimum(r0 + 1, n - 1);   c1 = np.minimum(c0 + 1, n - 1)
    dr = (r - r0)[:, None].astype(np.float32); dc = (c - c0)[:, None].astype(np.float32)
    f = big.reshape(n * n, C)
    return (f[r0 * n + c0] * (1 - dr) * (1 - dc) + f[r0 * n + c1] * (1 - dr) * dc
            + f[r1 * n + c0] * dr * (1 - dc) + f[r1 * n + c1] * dr * dc)


def clf():
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def main():
    # thin-crack frames with real painting: where over-marking is visible
    cands = []
    for m in S.list_images():
        c = m.get("corrected_crack_px") or 0
        tot = (m.get("width") or 1) * (m.get("height") or 1)
        if c > 20000 and c / tot < 0.05 and "SELFTEST" not in (m.get("filename") or ""):
            cands.append(m)
    cands = cands[:8]
    print(f"  {len(cands)} crops of {CROP}x{CROP}\n", flush=True)

    data = []
    for k, m in enumerate(cands, 1):
        iid = m["id"]
        corr = np.asarray(S.load_npy(iid, "correction.npy"))
        img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
        H, W = img.shape
        ys, xs = np.nonzero(corr == 1)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        y0 = min(max(cy - CROP // 2, 0), H - CROP - 16)
        x0 = min(max(cx - CROP // 2, 0), W - CROP - 16)
        sub = img[y0:y0 + CROP + 16, x0:x0 + CROP + 16]
        lab = (corr[y0:y0 + CROP, x0:x0 + CROP] == 1)
        neg = (corr[y0:y0 + CROP, x0:x0 + CROP] == 2)
        cpath = os.path.join(CACHE, f"{iid}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath)
            g16 = [(0, 0, z["g0"])]
            g8 = [(0, 0, z["g0"]), (0, 8, z["g1"]), (8, 0, z["g2"]), (8, 8, z["g3"])]
        else:
            gs = embed_grid(sub, [(0, 0), (0, 8), (8, 0), (8, 8)])
            np.savez(cpath, g0=gs[0][2], g1=gs[1][2], g2=gs[2][2], g3=gs[3][2])
            g16 = [gs[0]]; g8 = gs
        feats = np.asarray(compute_feature_stack(sub[:CROP, :CROP]), np.float32)
        data.append(dict(iid=iid, name=(m.get("filename") or "")[22:46],
                         lab=lab, neg=neg, feats=feats, g16=g16, g8=g8))
        print(f"    {k}/{len(cands)} embedded {(m.get('filename') or '')[22:46]}", flush=True)

    C = data[0]["g16"][0][2].shape[2]
    rng = np.random.RandomState(0)
    print(f"\n  {'held-out crop':<26} {'stride':>7} {'IoU':>7} {'thickness':>10} {'label thick':>12}")
    res = {16: [], 8: []}
    for i, held in enumerate(data):
        for stride, key, origin in ((16, "g16", 8.0), (8, "g8", 8.0)):
            X, Y = [], []
            for j, d in enumerate(data):
                if j == i: continue
                pos = np.flatnonzero(d["lab"].ravel()); ng = np.flatnonzero(d["neg"].ravel())
                take = min(12000, len(pos)), min(12000, len(ng))
                idx = np.concatenate([rng.choice(pos, take[0], replace=False),
                                      rng.choice(ng, take[1], replace=False)])
                rr, cc = np.unravel_index(idx, d["lab"].shape)
                x17 = d["feats"][rr, cc, :]
                emb = lookup(d[key], stride, origin, rr, cc, C)
                X.append(np.concatenate([x17, emb], axis=1))
                Y.append(np.concatenate([np.ones(take[0], bool), np.zeros(take[1], bool)]))
            mdl = clf().fit(np.concatenate(X), np.concatenate(Y))
            # predict the whole held-out crop, banded
            hm = np.zeros(held["lab"].shape, bool)
            for r0 in range(0, CROP, 128):
                r1 = min(r0 + 128, CROP)
                rr = np.repeat(np.arange(r0, r1), CROP); cc = np.tile(np.arange(CROP), r1 - r0)
                blk = np.concatenate([held["feats"][rr, cc, :],
                                      lookup(held[key], stride, origin, rr, cc, C)], axis=1)
                hm[r0:r1] = (mdl.predict_proba(blk)[:, 1] > 0.5).reshape(r1 - r0, CROP)
            hm = P.prune_specks(hm)
            t = held["lab"]
            iou = (hm & t).sum() / max((hm | t).sum(), 1)
            def wid(mk):
                if mk.sum() < 50: return float("nan")
                sk, dd = medial_axis(mk, return_distance=True); return float(2 * dd[sk].mean())
            res[stride].append((iou, wid(hm)))
            print(f"  {held['name']:<26} {stride:>7} {iou:>7.4f} {wid(hm):>10.1f} "
                  f"{wid(t):>12.1f}", flush=True)
    print()
    for stride in (16, 8):
        a = np.array(res[stride])
        print(f"  stride {stride:>2}: mean IoU {np.nanmean(a[:,0]):.4f}   "
              f"mean predicted thickness {np.nanmean(a[:,1]):.1f} px")
    print("PILOT_DONE")


if __name__ == "__main__":
    main()
