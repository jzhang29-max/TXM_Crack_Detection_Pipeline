#!/usr/bin/env python3
"""Is the crack's transverse width physical, or is it the instrument's point spread?

Pre-registered test (crack-evolution-5d/out/txm_width_prereg_and_failed_variants.md):
call it INSTRUMENT-LIMITED only if the pooled transverse FWHM has a coefficient of variation
below 0.35 AND |Spearman rho| between FWHM and peak contrast is below 0.3. Both thresholds
were missed by a wide margin -- CV 0.916 and rho +0.673 -- so the width is the crack's.

NOTE ON PRIOR ART, since this file implements it: the method is standard. Sampling intensity
along lines perpendicular to the centreline is Orthogonal Profile Extraction; taking the edge
at half the profile peak is ISO50, codified for X-ray CT dimensional metrology in VDI/VDE
2630. See the prior-art section of docs/WIDTH_REFERENCE.md. What is being done with it here
-- pointing it at the ANNOTATIONS rather than at the specimen -- is the only part that is not
off the shelf.

Emits: out/txm_transverse_fwhm.json

Usage:  python3 code/audit/measure_transverse_fwhm.py [out.jsonl]
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
from skimage.morphology import skeletonize                            # noqa: E402

NDIR, RAD, NSAMP, RECENTRE, PEAK_SIGMA = 12, 120, 250, 3, 2.0
DIRS = [(np.cos(t), np.sin(t)) for t in np.linspace(0, np.pi, NDIR, endpoint=False)]
OFF = np.arange(-RAD, RAD + 1)
NM_PER_PX = 29.0


def profile_fwhm(contrast, y, x, sigma):
    best, bestpk = np.inf, None
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
        if (i - j + 1) < best:
            best, bestpk = float(i - j + 1), pk
    return (None, None) if not np.isfinite(best) else (best, bestpk)


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
        rec = dict(id=iid)
        try:
            wide = P.effective_mask(iid, corrections="none", tight=False)
            cur = P.effective_mask(iid, corrections="none", tight=True)
            img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
            bg = P.local_background(img, wide)
            contrast = (bg - img).astype(np.float32)
            del bg, img
            outside = contrast[~wide]
            sigma = float(np.percentile(outside, 84.1) - np.percentile(outside, 50)) or 1e-6
            del outside, wide
            sk = skeletonize(cur)
            ys, xs = np.nonzero(sk)
            del sk, cur
            if ys.size == 0:
                raise RuntimeError("empty skeleton")
            sel = rng.choice(ys.size, min(NSAMP, ys.size), replace=False)
            ws, pks = [], []
            for k in sel:
                w, pk = profile_fwhm(contrast, int(ys[k]), int(xs[k]), sigma)
                if w is not None:
                    ws.append(w)
                    pks.append(pk)
            rec.update(sigma=sigma, n=len(ws),
                       fwhm_med=float(np.median(ws)) if ws else None,
                       peak_med=float(np.median(pks)) if pks else None,
                       fwhm=[float(v) for v in ws], peak=[float(v) for v in pks])
            del contrast
        except Exception as e:
            rec["err"] = f"{type(e).__name__}: {e}"
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"{iid[:34]:34s} " + (rec.get("err") or
              f"n={rec['n']:4d}  FWHM {rec['fwhm_med']:6.1f} px = {rec['fwhm_med']*NM_PER_PX:7.0f} nm"),
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out_transverse_fwhm.jsonl")
