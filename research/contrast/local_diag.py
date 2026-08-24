"""Why an arm helps or hurts: per-feature single-feature separability.

Reads the same cache as local_eval.py and, per arm, reports the AUC of each of
the 17 features taken alone at separating correction==1 from correction==2 --
pooled over all rows and over thin-frame rows only. Cheap (no fitting) and it
answers the mechanistic question directly: does a local/adaptive transform buy
local separability at the price of the large-radius ABSOLUTE intensity channels
that this project already measured as ~41% of the model's importance?

AUC is reported as max(a, 1-a) so a feature that separates in either direction
scores high; 0.5 means the feature alone is uninformative.

Writes research/contrast/local_diagnostics.json only.
"""

import json
import os
import sys

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))

import numpy as np
from sklearn.metrics import roc_auc_score

from txm_features import FEATURE_NAMES  # noqa: E402

OUT = os.path.join(P0, "research", "contrast")
CACHE = os.path.join(OUT, "local_cache")


def main():
    man = json.load(open(os.path.join(OUT, "local_manifest.json")))
    ids = [(m["id"], bool(m["thin"])) for m in man["images"]]
    res = {}
    for arm in man["arms"]:
        Xs, ys, th = [], [], []
        for iid, is_thin in ids:
            with np.load(os.path.join(CACHE, iid + ".npz")) as z:
                Xs.append(z["X_" + arm])
                ys.append(z["y"])
                th.append(np.full(len(z["y"]), is_thin, bool))
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        t = np.concatenate(th)
        del Xs, ys, th

        def aucs(Xa, ya):
            out = {}
            for j, nm in enumerate(FEATURE_NAMES):
                a = roc_auc_score(ya, Xa[:, j])
                out[nm] = float(max(a, 1.0 - a))
            return out

        res[arm] = dict(all_rows=aucs(X, y), thin_rows=aucs(X[t], y[t]))
        a = res[arm]["all_rows"]
        print(f"{arm:26s} intensity={a['intensity']:.4f} "
              f"smooth_s64={a['smooth_s64']:.4f} smooth_s32={a['smooth_s32']:.4f} "
              f"lap_s1={a['laplacian_s1']:.4f} grad_s1={a['gradmag_s1']:.4f} "
              f"best={max(a, key=a.get)}({max(a.values()):.4f})", flush=True)
        del X, y, t
    json.dump(res, open(os.path.join(OUT, "local_diagnostics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
