"""
Run the trained SAM+17 hybrid on all 71 images and write masks + stats, so its
overlays can be put next to the current pipeline's on the owner's own data.

Memory is the whole design constraint. A 15-tile image is ~15 M pixels; at 273
float32 features that is 16 GB if materialised at once, and the LARGE mosaic is
23.5 M pixels. So prediction walks tile by tile and, within each tile, row band
by row band, never holding more than a band's worth of concatenated features.
The 17-feature stack is computed once per image (1-1.6 GB) and indexed.

Post-processing is deliberately IDENTICAL to the current pipeline's
(apply_pixel_model.postprocess_mask), so the comparison is feature-set versus
feature-set and not post-processing versus post-processing.

Usage:
    python3 apply_sam_hybrid.py [--model models/pixel_sam_hybrid.joblib]
                               [--only SUBSTR] [--out results/sam_hybrid_71]
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sam_embeddings as EC
import paint_common as pc
from train_sam_hybrid import interp_tile
from txm_features import compute_feature_stack, robust_normalize

try:
    from apply_pixel_model import postprocess_mask
except Exception:                                     # keep running if the import moves
    postprocess_mask = None

TILE = EC.TILE
BAND = 128


def predict_image(name, clf, n_expected, band=BAND):
    """Full-resolution crack probability for one image from hybrid features."""
    raw = tifffile.imread(pc._find_path(name)).astype(np.float64)
    img01 = robust_normalize(raw, 1.0, 99.0).astype(np.float32)
    del raw
    f17 = compute_feature_stack(img01)
    coords, emb = EC.ensure(name)

    H, W = img01.shape
    prob = np.zeros((H, W), np.float32)
    written = np.zeros((H, W), bool)
    # Later tiles first so interior wins over clamped edge margins, matching
    # sam_features_at's assignment order during training.
    for k in range(len(coords) - 1, -1, -1):
        y0, x0 = int(coords[k][0]), int(coords[k][1])
        y1, x1 = min(y0 + TILE, H), min(x0 + TILE, W)
        for b0 in range(y0, y1, band):
            b1 = min(b0 + band, y1)
            sub = ~written[b0:b1, x0:x1]
            if not sub.any():
                continue
            rr, cc = np.nonzero(sub)
            rr_g, cc_g = rr + b0, cc + x0
            a = np.asarray(f17[rr_g, cc_g, :], np.float32)
            b = interp_tile(emb[k], rr_g - y0, cc_g - x0)
            X = np.concatenate([a, b], axis=1)
            if X.shape[1] != n_expected:
                raise ValueError(f"{name}: built {X.shape[1]} features, model wants {n_expected}")
            p = clf.predict_proba(X)[:, 1].astype(np.float32)
            blk = prob[b0:b1, x0:x1]
            blk[rr, cc] = p
            prob[b0:b1, x0:x1] = blk
            wblk = written[b0:b1, x0:x1]
            wblk[rr, cc] = True
            written[b0:b1, x0:x1] = wblk
    if not written.all():
        print(f"    [warn] {int((~written).sum()):,} px never predicted (outside all tiles)")
    del f17
    return prob, img01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(pc.PROJECT_DIR, "models", "pixel_sam_hybrid.joblib"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=os.path.join(pc.PROJECT_DIR, "results", "sam_hybrid_71"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-postprocess", action="store_true")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"] if isinstance(bundle, dict) else bundle
    n_expected = bundle.get("n_features", 273) if isinstance(bundle, dict) else 273
    print(f"model {args.model}\n  kind={bundle.get('kind') if isinstance(bundle,dict) else '?'} "
          f"n_features={n_expected}")

    mask_dir = os.path.join(args.out, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    infos = [i for i in pc.list_images() if not args.only or args.only in i["name"]]
    print(f"{len(infos)} images -> {args.out}\n")

    rows, t_all = [], time.time()
    for k, info in enumerate(infos, 1):
        name = info["name"]
        if not os.path.exists(EC.cache_path(name)):
            print(f"  [{k:2d}/{len(infos)}] SKIP (no SAM cache) {name[:44]}")
            continue
        t0 = time.time()
        try:
            prob, img01 = predict_image(name, clf, n_expected)
        except Exception as e:
            print(f"  [{k:2d}/{len(infos)}] FAILED {name[:40]}: {type(e).__name__}: {str(e)[:70]}")
            rows.append(dict(name=name, group=info.get("group"), error=str(e)[:200]))
            continue
        # postprocess_mask takes the PROBABILITY MAP, not a boolean mask -- it
        # blurs, thresholds at its own PROB_THRESHOLD, then does shape
        # validation and restricted hysteresis. Handing it a bool array would
        # silently skip all of that and make the comparison unfair to the
        # hybrid, since the current pipeline's outputs DO get it.
        raw_frac = float((prob > args.threshold).mean())
        if postprocess_mask is not None and not args.no_postprocess:
            mask = postprocess_mask(prob)
        else:
            mask = prob > args.threshold
        np.save(os.path.join(mask_dir, f"{name}_mask.npy"), mask)

        from skimage.measure import label
        n_reg = int(label(mask, connectivity=2).max())
        r = dict(name=name, group=info.get("group"), area_frac=float(mask.mean()),
                 area_frac_prepost=raw_frac, regions=n_reg,
                 secs=round(time.time() - t0, 1))
        rows.append(r)
        print(f"  [{k:2d}/{len(infos)}] {info.get('group','?')[:18]:18s} "
              f"area {r['area_frac']*100:5.2f}%  regions {n_reg:5d}  "
              f"{r['secs']:5.1f}s  {name[:38]}")

    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(dict(model=args.model, threshold=args.threshold,
                       postprocess=bool(postprocess_mask) and not args.no_postprocess,
                       rows=rows), f, indent=2)
    ok = [r for r in rows if "error" not in r]
    print(f"\n{len(ok)}/{len(infos)} predicted in {time.time()-t_all:.0f}s -> {args.out}")
    if ok:
        import collections
        by = collections.defaultdict(list)
        for r in ok:
            by[r["group"]].append(r["area_frac"])
        print("\nmedian predicted crack area by group:")
        for g in sorted(by):
            print(f"  {g:24s} n={len(by[g]):2d}  {np.median(by[g])*100:5.2f}%")


if __name__ == "__main__":
    main()
