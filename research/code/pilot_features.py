"""Is the fat boundary caused by the large-scale smoothing features rather than the embedding?

The other hypothesis. smooth_s32 and smooth_s64 average over 32 and 64 px, so they cannot
represent a sharp edge; if they are what widens the mask, dropping them should tighten it --
and unlike the stride-8 route this costs no SAM passes at all, just a refit on the features
already cached by the stride pilot.

Same 8 crops, same folds, same classifier as pilot_stride8.py, so the numbers are directly
comparable to its stride-16 arm (IoU 0.1748, thickness 38.9 px).
"""
import os, sys
import numpy as np
PROJECT = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, os.path.join(PROJECT, "code"))
sys.path.insert(0, os.path.join(PROJECT, "research", "logs"))
import store as S, pipeline as P
from txm_features import compute_feature_stack, FEATURE_NAMES
from skimage.morphology import medial_axis
from pilot_stride8 import lookup, clf, CROP, CACHE

DROP = {"smooth_s32": [FEATURE_NAMES.index("smooth_s32")],
        "smooth_s32+s64": [FEATURE_NAMES.index("smooth_s32"), FEATURE_NAMES.index("smooth_s64")],
        "s16+s32+s64": [FEATURE_NAMES.index(n) for n in ("smooth_s16","smooth_s32","smooth_s64")]}

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
        y0 = min(max(cy-CROP//2, 0), H-CROP-16); x0 = min(max(cx-CROP//2, 0), W-CROP-16)
        z = np.load(os.path.join(CACHE, f"{iid}.npz"))
        data.append(dict(lab=(corr[y0:y0+CROP, x0:x0+CROP]==1),
                         neg=(corr[y0:y0+CROP, x0:x0+CROP]==2),
                         feats=np.asarray(compute_feature_stack(img[y0:y0+CROP, x0:x0+CROP]), np.float32),
                         g16=[(0,0,z["g0"])]))
    C = data[0]["g16"][0][2].shape[2]
    def wid(mk):
        if mk.sum() < 50: return float("nan")
        sk, d = medial_axis(mk, return_distance=True); return float(2*d[sk].mean())
    print(f"  {'feature set':<18} {'n feats':>8} {'mean IoU':>9} {'thickness':>11}")
    for label, drop in [("all 17", [])] + list(DROP.items()):
        keep = [i for i in range(17) if i not in drop]
        rng = np.random.RandomState(0)
        out = []
        for i, held in enumerate(data):
            X, Y = [], []
            for j, d in enumerate(data):
                if j == i: continue
                pos = np.flatnonzero(d["lab"].ravel()); ng = np.flatnonzero(d["neg"].ravel())
                tk = min(12000, len(pos)), min(12000, len(ng))
                idx = np.concatenate([rng.choice(pos, tk[0], replace=False),
                                      rng.choice(ng, tk[1], replace=False)])
                rr, cc = np.unravel_index(idx, d["lab"].shape)
                X.append(np.concatenate([d["feats"][rr, cc, :][:, keep],
                                         lookup(d["g16"], 16, 8.0, rr, cc, C)], axis=1))
                Y.append(np.concatenate([np.ones(tk[0], bool), np.zeros(tk[1], bool)]))
            mdl = clf().fit(np.concatenate(X), np.concatenate(Y))
            hm = np.zeros(held["lab"].shape, bool)
            for r0 in range(0, CROP, 128):
                r1 = min(r0+128, CROP)
                rr = np.repeat(np.arange(r0, r1), CROP); cc = np.tile(np.arange(CROP), r1-r0)
                blk = np.concatenate([held["feats"][rr, cc, :][:, keep],
                                      lookup(held["g16"], 16, 8.0, rr, cc, C)], axis=1)
                hm[r0:r1] = (mdl.predict_proba(blk)[:,1] > 0.5).reshape(r1-r0, CROP)
            hm = P.prune_specks(hm)
            t = held["lab"]
            out.append(((hm & t).sum()/max((hm | t).sum(),1), wid(hm)))
        a = np.array(out)
        print(f"  {label:<18} {len(keep)+C:>8} {np.nanmean(a[:,0]):>9.4f} "
              f"{np.nanmean(a[:,1]):>10.1f} px", flush=True)
    print("\n  reference: stride 8 (7.2 h re-embed)  IoU 0.1555   33.2 px")
    print("FEAT_DONE")

main()
