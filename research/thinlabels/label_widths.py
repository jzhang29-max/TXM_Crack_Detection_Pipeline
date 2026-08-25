"""The premise, measured on every labelled frame: how wide is the painted stroke, and how
wide is the dark core inside it?

No model, no features -- just correction.npy, img.npy and pipeline.tighten_to_image. Also
records the deployed model's own predicted half-width from the cached prob.npy, so the
"the model reproduces the label width" claim can be checked frame by frame rather than on
the four frames the hypothesis quotes.
"""
import json
import os
import sys
import warnings

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
import store as S          # noqa: E402
import pipeline as P       # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = os.path.join(P0, "research", "thinlabels")


def half_width(mask):
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    if not mask.any():
        return None
    skel = skeletonize(mask)
    if not skel.any():
        return None
    return round(float(np.median(distance_transform_edt(mask)[skel])), 2)


def main():
    rows = []
    for m in S.list_images():
        iid = m["id"]
        corr = S.load_npy(iid, "correction.npy")
        if corr is None:
            continue
        crack = corr == 1
        if not crack.any():
            del corr, crack
            continue
        core = P.tighten_to_image(iid, crack)
        prob = S.load_npy(iid, "prob.npy")
        pred = P.prune_specks(prob > 0.5) if prob is not None else None
        r = dict(id=iid, painted_px=int(crack.sum()), core_px=int(core.sum()),
                 core_frac=round(float(core.sum() / crack.sum()), 4),
                 hw_painted=half_width(crack), hw_core=half_width(core),
                 hw_deployed_pred=half_width(pred) if pred is not None else None,
                 declined=bool(np.array_equal(core, crack)))
        r["ratio"] = (round(r["hw_painted"] / r["hw_core"], 2)
                      if r["hw_painted"] and r["hw_core"] else None)
        rows.append(r)
        print(f"{iid[:50]:50s} painted_hw={r['hw_painted']:6} core_hw={r['hw_core']:6} "
              f"pred_hw={r['hw_deployed_pred']:6} ratio={r['ratio']:6} "
              f"core_frac={r['core_frac']:.3f} decl={int(r['declined'])}", flush=True)
        del corr, crack, core, prob, pred

    def med(k):
        v = [r[k] for r in rows if r[k] is not None]
        return round(float(np.median(v)), 2) if v else None

    summ = dict(n_frames=len(rows), median_hw_painted=med("hw_painted"),
                median_hw_core=med("hw_core"),
                median_hw_deployed_pred=med("hw_deployed_pred"),
                median_ratio=med("ratio"),
                median_core_frac=med("core_frac"),
                n_declined=sum(r["declined"] for r in rows))
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, frames=rows),
              open(os.path.join(OUT, "label_widths.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
