"""
The one measurement missing before the ensemble could be recommended:
does averaging the two models' probabilities KEEP the hybrid's 46x reduction in
false positives on the owner-confirmed crack-free specimens, or does averaging
with the current model reintroduce them?

This matters because the two models disagree wildly off B2 (agreement IoU 0.006
on AM) and the current model marks 6.67% of a crack-free specimen while the
hybrid marks 0.14%. A mean of 0.9 and 0.05 clears 0.5; a mean of 0.6 and 0.05
does not. Which happens is an empirical question about how confident the current
model is when it is wrong, and it decides whether the ensemble is deployable.

Evaluates, on the 6 crack-free specimens (HANDOFF's second axis) and reporting
predicted area, where every marked pixel is a false positive by definition:
    current alone, hybrid alone, mean-probability, and both binary combinations.

Usage:
    python3 check_ensemble_crackfree.py
"""

import json
import os
import sys

import joblib
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sam_embeddings as EC
import paint_common as pc
from apply_pixel_model import postprocess_mask, predict_probability_map
from apply_sam_hybrid import predict_image
from txm_features import robust_normalize

OUT = os.path.join(pc.PROJECT_DIR, "results", "improve", "ensemble_crackfree.json")


def main():
    import mark_zero_crack_images as mz          # module-level side effects are
    zero = list(mz.ZERO_CRACK)                   # idempotent; we only read the list

    cur_model = joblib.load(pc.MODEL_PATH)
    bundle = joblib.load(os.path.join(pc.PROJECT_DIR, "models", "pixel_sam_hybrid.joblib"))
    hyb_clf = bundle["model"]
    n_feat = bundle["n_features"]

    rows = []
    print(f"{'specimen':40s} {'cur':>7s} {'hyb':>7s} {'meanp':>7s} {'union':>7s} {'inter':>7s}")
    for frag in zero:
        hits = [i["name"] for i in pc.list_images() if frag in i["name"]]
        if not hits:
            print(f"  {frag[:38]:40s} (not found)")
            continue
        name = hits[0]
        if not os.path.exists(EC.cache_path(name)):
            print(f"  {name[:38]:40s} (no SAM cache)")
            continue

        raw = tifffile.imread(pc._find_path(name)).astype(np.float64)
        img01 = robust_normalize(raw, 1.0, 99.0)
        del raw
        p_cur = predict_probability_map(cur_model, img01).astype(np.float32)
        p_hyb, _ = predict_image(name, hyb_clf, n_feat)

        if p_cur.shape != p_hyb.shape:
            print(f"  {name[:38]:40s} SHAPE MISMATCH {p_cur.shape} vs {p_hyb.shape}")
            continue

        # postprocess_mask consumes a probability map, so every rule below gets
        # the SAME post-processing the deployed pipeline applies. Comparing a
        # post-processed baseline against a raw-threshold ensemble would flatter
        # whichever one skipped it.
        m_cur = postprocess_mask(p_cur)
        m_hyb = postprocess_mask(p_hyb)
        m_mean = postprocess_mask((p_cur + p_hyb) / 2.0)
        r = dict(name=name,
                 current=float(m_cur.mean()), hybrid=float(m_hyb.mean()),
                 meanprob=float(m_mean.mean()),
                 union=float(np.logical_or(m_cur, m_hyb).mean()),
                 intersection=float(np.logical_and(m_cur, m_hyb).mean()))
        rows.append(r)
        print(f"  {name[:38]:40s} {r['current']*100:6.2f}% {r['hybrid']*100:6.2f}% "
              f"{r['meanprob']*100:6.2f}% {r['union']*100:6.2f}% {r['intersection']*100:6.2f}%")
        del p_cur, p_hyb, m_cur, m_hyb, m_mean, img01

    if rows:
        print()
        for k in ("current", "hybrid", "meanprob", "union", "intersection"):
            v = [r[k] for r in rows]
            print(f"  {k:14s} mean {np.mean(v)*100:6.2f}%   median {np.median(v)*100:6.2f}%")
        base = np.mean([r["current"] for r in rows])
        for k in ("hybrid", "meanprob"):
            v = np.mean([r[k] for r in rows])
            print(f"  {k} reduces false-positive area {base/max(v,1e-9):.1f}x vs current")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(note="predicted area on owner-confirmed crack-free specimens; "
                            "every marked pixel is a false positive",
                       postprocessed=True, rows=rows), f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
