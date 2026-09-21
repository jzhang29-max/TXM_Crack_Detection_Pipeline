#!/usr/bin/env python3
"""Width ratio of every mask variant against the image's own transverse FWHM.

WHY THIS FILE EXISTS. The table it replaces spliced two runs: the detector row came from
the final inverse-distance-weighted estimator over 63 frames, the `wide` and label rows from
an earlier run over 61. Read on one instrument the detector is 0.679 and the human brush
0.692 -- a 2% difference, not the 12% the spliced table implied. Every row here is computed
in ONE pass, with ONE estimator, on ONE frame set, sampling the SAME skeleton points for
every variant, so the rows are comparable by construction.

Emits: out/txm_width_ratio_unified.json

Usage:  python3 code/audit/measure_width_ratio.py [out.jsonl]
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
from scipy.ndimage import distance_transform_edt                      # noqa: E402
from skimage.morphology import skeletonize                            # noqa: E402

NDIR, RAD, NSAMP, RECENTRE, PEAK_SIGMA = 12, 200, 300, 3, 2.0
DIRS = [(np.cos(t), np.sin(t)) for t in np.linspace(0, np.pi, NDIR, endpoint=False)]
OFF = np.arange(-RAD, RAD + 1)


def transverse_fwhm(contrast, y, x, sigma):
    """Smallest full-width-at-half-maximum over NDIR directions. Same rule as the shipped
    pipeline's `_transverse_fwhm`; duplicated here so the audit does not silently inherit a
    change to the production one."""
    best = np.inf
    for dy, dx in DIRS:
        yy = np.clip(np.rint(y + OFF * dy).astype(np.int32), 0, contrast.shape[0] - 1)
        xx = np.clip(np.rint(x + OFF * dx).astype(np.int32), 0, contrast.shape[1] - 1)
        p = contrast[yy, xx]
        c = RAD - RECENTRE + int(np.argmax(p[RAD - RECENTRE:RAD + RECENTRE + 1]))
        pk = float(p[c])
        if pk < PEAK_SIGMA * sigma:
            continue
        half = pk / 2.0
        i = c
        while i + 1 <= 2 * RAD and p[i + 1] >= half:
            i += 1
        j = c
        while j - 1 >= 0 and p[j - 1] >= half:
            j -= 1
        if i == 2 * RAD or j == 0:
            continue
        best = min(best, float(i - j + 1))
    return best


def group_of(iid):
    low = iid.lower()
    if "_b2_" in low:
        return "B2"
    if "b3" in low:
        return "B3"
    if "hc_316l" in low:
        return "AM"
    if "wrought" in low:
        return "WR"
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
    rng = np.random.default_rng(0)
    for m in S.list_images():
        iid = m["id"]
        if iid in done:
            continue
        rec = dict(id=iid, group=group_of(iid))
        try:
            wide = P.effective_mask(iid, corrections="none", tight=False)
            _real = P.drop_straight_lines
            P.drop_straight_lines = lambda mk, sp=None, report=None: mk
            _wc = P.WIDTH_CLIP
            P.WIDTH_CLIP = False
            tight_noclip = P.effective_mask(iid, corrections="none", tight=True)
            P.WIDTH_CLIP = _wc
            P.drop_straight_lines = _real
            shipped = P.effective_mask(iid, corrections="none", tight=True)
            img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
            bg = P.local_background(img, wide)
            contrast = (bg - img).astype(np.float32)
            del bg, img
            outside = contrast[~wide]
            sigma = float(np.percentile(outside, 84.1) - np.percentile(outside, 50)) or 1e-6
            del outside
            corr = S.load_npy(iid, "correction.npy")
            lab = (corr == 1) if corr is not None else None
            # SAME sample points for every variant: the skeleton of the widest mask, so no
            # variant is scored on points chosen by its own geometry.
            sk = skeletonize(wide)
            ys, xs = np.nonzero(sk)
            del sk
            if ys.size == 0:
                raise RuntimeError("empty skeleton")
            sel = rng.choice(ys.size, min(NSAMP, ys.size), replace=False)
            dts = {k: distance_transform_edt(v) for k, v in
                   (("wide", wide), ("tight_noclip", tight_noclip), ("shipped", shipped))}
            if lab is not None and lab.any():
                dts["label"] = distance_transform_edt(lab)
            acc = {k: [] for k in dts}
            on_label = 0
            for k in sel:
                y, x = int(ys[k]), int(xs[k])
                w = transverse_fwhm(contrast, y, x, sigma)
                if not np.isfinite(w) or w <= 0:
                    continue
                for name, dt in dts.items():
                    if name == "label" and not lab[y, x]:
                        continue          # the label row is only meaningful where painted
                    acc[name].append(2 * float(dt[y, x]) / w)
                if lab is not None and lab.any() and lab[y, x]:
                    on_label += 1
            rec.update(sigma=sigma, n_profiles=len(acc["wide"]), n_on_label=on_label,
                       **{f"ratio_{k}": (float(np.median(v)) if v else None)
                          for k, v in acc.items()},
                       **{f"n_{k}": len(v) for k, v in acc.items()})
            del contrast, wide, tight_noclip, shipped, corr, dts
        except Exception as e:
            rec["err"] = f"{type(e).__name__}: {e}"
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"{iid[:34]:34s} " + (rec.get("err") or
              "  ".join(f"{k} {rec.get('ratio_'+k) and round(rec['ratio_'+k],3)}"
                        for k in ("wide", "tight_noclip", "shipped", "label"))), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out_width_ratio_unified.jsonl")
