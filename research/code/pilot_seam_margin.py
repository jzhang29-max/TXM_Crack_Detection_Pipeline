"""Pick the blend margin: seams removed, area not inflated, false positives not raised.

Lookup-time blending removes the tile seams (5.2x -> 1.3x vertical, 12.3x -> 1.4x
horizontal) but costs area (+0.88 pp) and roughly doubles prediction time, because past its
own edge a tile contributes extrapolated edge cells rather than a real embedding. The margin
controls how far that extrapolation reaches, so it is the knob that trades seam removal
against smearing. This sweeps it and checks the one axis that decides deployment: predicted
area on a specimen confirmed to contain no crack, where every positive is a false positive.
"""
import os, sys, time
import numpy as np
PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, os.path.join(PROJECT, "code"))
sys.path.insert(0, os.path.join(PROJECT, "research", "code"))
import model as M, store as S, pipeline as P
from txm_features import compute_feature_stack
from pilot_seams import blended_rows, lastwins_rows, seam_ratio

def predict(iid, fn, mdl):
    img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
    H, W = img.shape
    z = np.load(S.path(iid, "emb.npz")); coords, embs = z["coords"], z["emb"]
    feats = np.asarray(compute_feature_stack(img), np.float32)
    n17 = feats.shape[2]
    prob = np.zeros((H, W), np.float32)
    for r0 in range(0, H, 96):
        r1 = min(r0 + 96, H)
        rr = np.repeat(np.arange(r0, r1), W); cc = np.tile(np.arange(W), r1 - r0)
        blk = feats[r0:r1].reshape(-1, n17)
        emb = fn(coords, embs, rr, cc)
        p17 = mdl.m17.predict_proba(blk)[:, 1]
        ph = mdl.hybrid.predict_proba(np.concatenate([blk, emb], axis=1))[:, 1]
        prob[r0:r1] = ((p17 + ph) / 2).reshape(r1 - r0, W)
    return prob, (H, W)

def main():
    mdl = P.get_model()
    cracked = [m for m in S.list_images()
               if m.get("has_prob") and (m.get("width") or 0) > 2*M.TILE
               and (m.get("height") or 0) > 2*M.TILE
               and "SELFTEST" not in (m.get("filename") or "")]
    cracked.sort(key=lambda x: (x.get("megapixels") or 0))
    cr = cracked[0]
    clean = [m for m in S.list_images()
             if any(k.lower() in (m.get("filename") or "").lower() for k in P.CLEAN_SPECIMENS)]
    clean.sort(key=lambda x: (x.get("megapixels") or 0))
    cl = clean[0]
    print(f"  cracked frame : {(cr.get('filename') or '')[22:52]}")
    print(f"  crack-free    : {(cl.get('filename') or '')[22:52]}\n")
    print(f"  {'margin':>8} {'vert seam':>10} {'horz seam':>10} {'crack%':>8} {'FP% clean':>10} {'sec':>6}")
    for margin in (None, 64, 128, 192):
        fn = lastwins_rows if margin is None else (lambda c, e, r, x, mg=margin: blended_rows(c, e, r, x, mg))
        t0 = time.time()
        prob, shp = predict(cr["id"], fn, mdl)
        v = seam_ratio(prob, 1, shp); h = seam_ratio(prob, 0, shp)
        crack = P.prune_specks(prob > 0.5).mean() * 100
        pc, _ = predict(cl["id"], fn, mdl)
        fp = P.prune_specks(pc > 0.5).mean() * 100
        lbl = "none" if margin is None else str(margin)
        print(f"  {lbl:>8} {v[0]/v[1]:>9.1f}x {h[0]/h[1]:>9.1f}x {crack:>7.2f}% {fp:>9.3f}% "
              f"{time.time()-t0:>6.0f}", flush=True)
    print("MARGIN_DONE")


if __name__ == "__main__":
    main()
