#!/usr/bin/env python3
"""Per-frame accounting of accepted area the annotator never labelled either way.

WHY THIS FILE EXISTS. The headline table of docs/UNLABELLED_AREA.md -- corpus 24.7%,
AM 51.1%, B2 36.9%, Wrought 10.7%, B3 4.7%, area-weighted over 64 frames -- had no
supporting artifact. Three of those five numbers existed only inside an English sentence in
a JSON `why` field; 10.7%, 4.7% and the frame count appeared nowhere. It is the number the
whole section rests on.

Emits: out/txm_unlabelled_area_per_frame.json with one record per frame, so every figure in
that table is recomputable and the mask version it was measured on is recorded.

Usage:  python3 code/audit/measure_unlabelled_area.py [out.jsonl]
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
from scipy.ndimage import label as cclabel, distance_transform_edt    # noqa: E402

SMALL_FLOOR = 4000          # the sampling frame used by the blind adjudication
NEAR_STROKE = 25            # px: within this of a painted stroke is stroke rim, not a find


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
        if iid in done:
            continue
        rec = dict(id=iid, group=group_of(iid),
                   mask_version="effective_mask(corrections='none', tight=True) with "
                                "WIDTH_CLIP=True and drop_straight_lines active")
        try:
            mask = P.effective_mask(iid, corrections="none", tight=True)
            corr = S.load_npy(iid, "correction.npy")
            rec["frame_px"] = int(mask.size)
            rec["accepted_px"] = int(mask.sum())
            if not mask.any() or corr is None or corr.shape != mask.shape:
                rec["skip"] = "no mask or no correction map"
            else:
                un = mask & (corr == 0)
                rec.update(
                    painted_crack_px=int((mask & (corr == 1)).sum()),
                    painted_not_px=int((mask & (corr == 2)).sum()),
                    unlabelled_px=int(un.sum()),
                    frac_unlabelled=float(un.sum() / max(mask.sum(), 1)),
                    unlabelled_frac_of_frame=float(un.sum() / mask.size),
                )
                lab, n = cclabel(un)
                if n:
                    sizes = np.bincount(lab.ravel())
                    sizes[0] = 0
                    big = sizes >= SMALL_FLOOR
                    rec.update(n_unlab_components=int(n),
                               n_unlab_big=int(big.sum()),
                               unlab_px_in_big=int(sizes[big].sum()),
                               frac_unlab_in_big=float(sizes[big].sum() / max(sizes.sum(), 1)))
                    painted = corr == 1
                    if painted.any():
                        d = distance_transform_edt(~painted)
                        near = (d <= NEAR_STROKE) & un
                        rec["unlab_px_within_25px_of_stroke"] = int(near.sum())
                        rec["frac_unlab_near_stroke"] = float(near.sum() / max(un.sum(), 1))
                        del d, near
                    del lab, sizes
                del un
            del mask, corr
        except Exception as e:
            rec["err"] = f"{type(e).__name__}: {e}"
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"{iid[:34]:34s} " + (rec.get("err") or rec.get("skip") or
              f"accepted {rec['accepted_px']:9,}  unlabelled {100*rec['frac_unlabelled']:5.1f}%"
              f"  ({100*rec.get('frac_unlab_in_big', 0):4.0f}% of it in >={SMALL_FLOOR}px comps)"),
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out_unlabelled_area.jsonl")
