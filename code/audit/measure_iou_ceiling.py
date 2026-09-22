#!/usr/bin/env python3
"""What a physically correct crack trace scores in IoU against these labels -- and what the
shipped detector scores, measured the SAME way on the SAME frames.

WHY IT IS RECOMPUTED. out/txm_iou_ceiling.json carries the ceiling (0.0498 for a 3 px trace)
and, next to it, `deployed_model_IoU_same_frames` 0.5047 with `model_over_ceiling` 10.1.
The ceiling is a property of the LABELS and is still correct. The model column is not: its
per-frame values are byte-identical to out/cldice_centreline_per_frame.json, which was run
at 02:05 on 2026-09-19, before clip_to_measured_width shipped at 14:14 the same day. The
ratio "10.1x the ceiling" therefore compares a current ceiling against a superseded mask.

Both sides are computed here in one pass with one IoU convention, so the ratio means
something.

Emits: out/txm_iou_ceiling_v2.json

Usage:  python3 code/audit/measure_iou_ceiling.py [out.jsonl]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
os.chdir(HERE)

from app.core import pipeline as P                                    # noqa: E402
from app.core import store as S                                       # noqa: E402
from skimage.morphology import skeletonize, binary_dilation, disk     # noqa: E402

WIDTHS = (3, 5, 11, 21)


def group_of(iid_or_meta):
    """Specimen group. Matched most-specific-first, and on the FILENAME where available.

    The obvious version tests `"b3" in id.lower()` before `"hc_316l"`, and the stored id
    carries a content hash: `HC_316L_fatigue_1600_cycles..._`4cb30dc6`` contains the
    substring "b3" inside the hash, so that frame was filed as B3 in three artifacts
    emitted from this directory. One frame, but it moved AM's unlabelled share 51.06% ->
    50.16% and B3's 4.49% -> 2.83%. Specific tokens first, and prefer the filename.
    """
    if isinstance(iid_or_meta, dict):
        low = (iid_or_meta.get("filename") or iid_or_meta.get("id") or "").lower()
    else:
        low = str(iid_or_meta).lower()
    if "hc_316l" in low:
        return "AM"
    if "wrought" in low:
        return "WR"
    if "_b2_" in low or "_b2" in low.split("260618_")[-1][:3]:
        return "B2"
    if "b3_" in low or "_b3" in low:
        return "B3"
    return "other"


def iou(a, b, dom):
    """IoU inside the labelled domain. Outside it there is no truth to agree with, so
    scoring there would credit or penalise the mask against nothing."""
    inter = float((a & b & dom).sum())
    union = float(((a | b) & dom).sum())
    return inter / union if union else float("nan")


def main(outp):
    done = set()
    if os.path.exists(outp):
        for ln in open(outp):
            try:
                done.add(json.loads(ln)["id"])
            except Exception:
                pass
    f = open(outp, "a")
    for m in S.list_images():
        iid = m["id"]
        if iid in done or not (m.get("corrected_crack_px") or 0):
            continue
        rec = dict(id=iid, group=group_of(iid))
        try:
            corr = S.load_npy(iid, "correction.npy")
            lab = corr == 1
            dom = corr > 0
            del corr
            if not lab.any():
                raise RuntimeError("no painted crack")
            shipped = P.effective_mask(iid, corrections="none", tight=True)
            from scipy.ndimage import distance_transform_edt
            sk = skeletonize(lab)
            # WIDTH IS MEASURED AT THE SKELETON, not over every label pixel. 2*EDT at a
            # medial-axis point is the local full width of the stroke; the median over ALL
            # label pixels is dominated by pixels near the edge and reads ~60% low (41.2 px
            # against 73.0 px corpus-wide on the first attempt here). The 73 px figure this
            # project has always quoted is the skeleton one and it is the correct statistic.
            rec["label_width_px"] = float(np.median(distance_transform_edt(lab)[sk]) * 2.0)
            for w in WIDTHS:
                r = max((w - 1) // 2, 0)
                trace = binary_dilation(sk, disk(r)) if r else sk
                rec[f"iou_w{w}"] = iou(trace, lab, dom)
                del trace
            rec["iou_shipped"] = iou(shipped, lab, dom)
            del sk, lab, dom, shipped
        except Exception as e:
            rec["err"] = f"{type(e).__name__}: {e}"
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"{iid[:34]:34s} " + (rec.get("err") or
              f"3px {rec['iou_w3']:.4f}   shipped {rec['iou_shipped']:.4f}   "
              f"label {rec['label_width_px']:.0f} px"), flush=True)

    R = [json.loads(l) for l in open(outp) if l.strip()]
    R = [r for r in R if "err" not in r]
    med = lambda k: float(np.median([r[k] for r in R]))                # noqa: E731
    res = dict(
        what="IoU ceiling of a physically correct crack trace against these labels, and the "
             "SHIPPED detector scored the same way on the same frames. Run 2026-09-21.",
        why="out/txm_iou_ceiling.json pairs a still-correct ceiling with a model column whose "
            "per-frame values are byte-identical to the pre-clip run, so its 10.1x ratio "
            "compares a current ceiling against a superseded mask.",
        generator="code/audit/measure_iou_ceiling.py",
        iou_convention="intersection over union restricted to the labelled domain (painted "
                        "crack OR painted not-crack)",
        trace_construction="skeletonize(label) dilated by disk((w-1)//2). The 3 px and 5 px "
                           "ceilings reproduce the earlier run exactly (0.0498, 0.0822); the "
                           "11 px and 21 px values differ (0.1957 vs 0.1754, 0.3431 vs 0.3140) "
                           "because the earlier run's structuring element is not recorded. The "
                           "3 px figure is the one quoted, and it is unaffected.",
        n_frames=len(R),
        ceiling={f"perfect_{w}px_trace": med(f"iou_w{w}") for w in WIDTHS},
        shipped_detector_iou=med("iou_shipped"),
        shipped_over_3px_ceiling=med("iou_shipped") / med("iou_w3"),
        median_label_width_px=med("label_width_px"),
        supersedes=dict(file="out/txm_iou_ceiling.json",
                        stale_keys=["headline.deployed_model_IoU_same_frames",
                                    "headline.model_over_ceiling",
                                    "per_specimen.*.model_iou"],
                        reason="model column measured before clip_to_measured_width shipped"),
        per_group={g: dict(n=len([r for r in R if r["group"] == g]),
                           ceiling_3px=float(np.median([r["iou_w3"] for r in R if r["group"] == g])),
                           shipped=float(np.median([r["iou_shipped"] for r in R if r["group"] == g])),
                           label_width_px=float(np.median([r["label_width_px"] for r in R if r["group"] == g])))
                   for g in sorted({r["group"] for r in R})},
        per_frame=R)
    # HERE is the pipeline repo root; the analysis repo is its SIBLING. dirname twice went
    # up to the home directory and wrote nowhere.
    out_json = os.path.join(os.path.dirname(HERE), "crack-depth-3d", "out",
                            "txm_iou_ceiling_v2.json")
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"\nceiling 3px {res['ceiling']['perfect_3px_trace']:.4f}   "
          f"shipped {res['shipped_detector_iou']:.4f}   "
          f"ratio {res['shipped_over_3px_ceiling']:.1f}x   (n={len(R)})")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out_iou_ceiling.jsonl")
