"""Where each arm's precision gain comes from: positive rate per label pool.

The fixed same-target metric calls the ring NEGATIVE, so an arm trained to reject the ring
is structurally favoured on it. This breaks the same held-out predictions down by pool so
the gain can be attributed rather than trusted:

    pool 1  dark core      -> recall on the thing we believe is the physical crack
    pool 2  inner ring     -> within 3 px of the core, most likely to be real crack edge
    pool 3  outer ring     -> brush overhang, most likely to be over-mark
    pool 4  corr==2        -> false positives on material the owner declared not-crack

Reuses the fold models saved by run_cv.py; forward passes only.
"""
import json
import os
import sys

import numpy as np
import joblib

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
OUT = os.path.join(P0, "research", "thinlabels")
sys.path.insert(0, OUT)
from run_cv import ARMS, build, N17    # noqa: E402


def main():
    X = np.load(os.path.join(OUT, "X.npy"), mmap_mode="r")
    pool, grp, meta, arm_rows, eval_rows = build()
    fold_of = {int(k): v for k, v in json.load(open(os.path.join(OUT, "folds.json"))).items()}
    ev_fold = np.array([fold_of[int(g)] for g in grp[eval_rows]])

    res = {}
    for name in ARMS:
        rate = {p: [] for p in (1, 2, 3, 4)}
        for f in range(5):
            te = np.sort(eval_rows[ev_fold == f])
            Xte = np.asarray(X[te], np.float32)
            m17 = joblib.load(os.path.join(OUT, "models", f"{name}_f{f}_17.joblib"))
            m273 = joblib.load(os.path.join(OUT, "models", f"{name}_f{f}_273.joblib"))
            p = 0.5 * (m17.predict_proba(Xte[:, :N17])[:, 1]
                       + m273.predict_proba(Xte)[:, 1])
            pos = p > 0.5
            for pl in (1, 2, 3, 4):
                sel = pool[te] == pl
                rate[pl].append(round(float(pos[sel].mean()), 4) if sel.any() else None)
            del Xte
        res[name] = {f"pool{p}": dict(per_fold=rate[p],
                                      mean=round(float(np.mean([v for v in rate[p]
                                                                if v is not None])), 4))
                     for p in (1, 2, 3, 4)}
        r = res[name]
        print(f"{name:18s} core={r['pool1']['mean']:.4f} inner_ring={r['pool2']['mean']:.4f} "
              f"outer_ring={r['pool3']['mean']:.4f} notcrack={r['pool4']['mean']:.4f}",
              flush=True)
    json.dump(res, open(os.path.join(OUT, "per_pool_rates.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
