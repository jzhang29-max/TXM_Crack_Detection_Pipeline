"""Is stride 8 on the thickness/accuracy frontier, or does thresholding dominate it?

stride 8 bought 38.9 -> 33.2 px of thickness for 0.1748 -> 0.1555 of IoU. If simply raising
the threshold on the CHEAP stride-16 features reaches 33 px at a smaller IoU cost, then the
7.2-hour re-embed buys nothing that a slider does not.
"""
import os, sys
import numpy as np
PROJECT = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, os.path.join(PROJECT, "code"))
sys.path.insert(0, os.path.join(PROJECT, "research", "logs"))
import model as M, store as S, pipeline as P
from txm_features import compute_feature_stack
from skimage.morphology import medial_axis
from pilot_stride8 import lookup, clf, CROP, CACHE

def main():
    cands = []
    for m in S.list_images():
        c = m.get("corrected_crack_px") or 0
        tot = (m.get("width") or 1) * (m.get("height") or 1)
        if c > 20000 and c / tot < 0.05 and "SELFTEST" not in (m.get("filename") or ""):
            cands.append(m)
    cands = cands[:8]
    data = []
    for m in cands:
        iid = m["id"]
        corr = np.asarray(S.load_npy(iid, "correction.npy"))
        img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
        H, W = img.shape
        ys, xs = np.nonzero(corr == 1)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        y0 = min(max(cy - CROP // 2, 0), H - CROP - 16)
        x0 = min(max(cx - CROP // 2, 0), W - CROP - 16)
        z = np.load(os.path.join(CACHE, f"{iid}.npz"))
        data.append(dict(name=(m.get("filename") or "")[22:46],
                         lab=(corr[y0:y0+CROP, x0:x0+CROP] == 1),
                         neg=(corr[y0:y0+CROP, x0:x0+CROP] == 2),
                         feats=np.asarray(compute_feature_stack(img[y0:y0+CROP, x0:x0+CROP]), np.float32),
                         g16=[(0, 0, z["g0"])]))
    C = data[0]["g16"][0][2].shape[2]
    rng = np.random.RandomState(0)
    THS = (0.5, 0.6, 0.7, 0.8, 0.9)
    acc = {t: [] for t in THS}
    def wid(mk):
        if mk.sum() < 50: return float("nan")
        sk, d = medial_axis(mk, return_distance=True); return float(2*d[sk].mean())
    for i, held in enumerate(data):
        X, Y = [], []
        for j, d in enumerate(data):
            if j == i: continue
            pos = np.flatnonzero(d["lab"].ravel()); ng = np.flatnonzero(d["neg"].ravel())
            tk = min(12000, len(pos)), min(12000, len(ng))
            idx = np.concatenate([rng.choice(pos, tk[0], replace=False),
                                  rng.choice(ng, tk[1], replace=False)])
            rr, cc = np.unravel_index(idx, d["lab"].shape)
            X.append(np.concatenate([d["feats"][rr, cc, :],
                                     lookup(d["g16"], 16, 8.0, rr, cc, C)], axis=1))
            Y.append(np.concatenate([np.ones(tk[0], bool), np.zeros(tk[1], bool)]))
        mdl = clf().fit(np.concatenate(X), np.concatenate(Y))
        prob = np.zeros(held["lab"].shape, np.float32)
        for r0 in range(0, CROP, 128):
            r1 = min(r0+128, CROP)
            rr = np.repeat(np.arange(r0, r1), CROP); cc = np.tile(np.arange(CROP), r1-r0)
            blk = np.concatenate([held["feats"][rr, cc, :],
                                  lookup(held["g16"], 16, 8.0, rr, cc, C)], axis=1)
            prob[r0:r1] = mdl.predict_proba(blk)[:, 1].reshape(r1-r0, CROP)
        t_ = held["lab"]
        for th in THS:
            hm = P.prune_specks(prob > th)
            acc[th].append(((hm & t_).sum()/max((hm | t_).sum(),1), wid(hm)))
        print(f"    fold {i+1}/8 done", flush=True)
    print(f"\n  {'threshold':>10} {'mean IoU':>9} {'mean thickness':>15}")
    for th in THS:
        a = np.array(acc[th])
        print(f"  {th:>10.2f} {np.nanmean(a[:,0]):>9.4f} {np.nanmean(a[:,1]):>14.1f} px")
    print(f"\n  for reference, from the stride pilot:")
    print(f"    stride 16 @0.50   IoU 0.1748   38.9 px")
    print(f"    stride  8 @0.50   IoU 0.1555   33.2 px")
    print("THRESH_DONE")


if __name__ == "__main__":
    main()
