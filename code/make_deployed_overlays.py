#!/usr/bin/env python3
"""Render the deployed detector over every ingested frame, at native resolution.

WHY THIS IS IN THE REPO NOW. The first version of these overlays was a one-off script that
thresholded `prob.npy` directly, which silently bypassed `effective_mask` and therefore both
narrowing steps -- the pictures showed a mask ~70% wider than what the app and every export
actually produce. Anything that renders the deliverable has to go through the same call the
deliverable does, so this does, and it lives next to the code it renders.

Usage:  python3 code/make_deployed_overlays.py [outdir]
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core import pipeline as P                                    # noqa: E402
from app.core import store as S                                       # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "deployed_v5_overlays")


def group_of(image_id):
    low = image_id.lower()
    if "_b2_" in low or low.startswith("average_mosaic_260618_b2"):
        return "B2"
    if "b3_" in low:
        return "B3"
    if "hc_316l" in low:
        return "AM_HC_316L"
    if "wrought" in low:
        return "Wrought"
    return "other"


def short(image_id):
    stem = image_id.split("__")[0]
    for cut in ("Average_mosaic_260618_", "Average_mosaic_260619_", "Average_mosaic_260620_"):
        stem = stem.replace(cut, "")
    return stem[:40]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for meta in S.list_images():
        iid = meta["id"]
        disp = S.load_npy(iid, "display.npy")
        if disp is None:
            disp = S.load_npy(iid, "img.npy")
        if disp is None:
            print("skip (no image):", iid)
            continue
        disp = np.asarray(disp, np.float32)
        lo, hi = np.percentile(disp, (0.5, 99.5))
        g = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
        del disp
        wide = P.effective_mask(iid, corrections="none", tight=False)
        tight = P.effective_mask(iid, corrections="none", tight=True)
        rgb = np.dstack([g, g, g])
        rgb[tight] = rgb[tight] * 0.35 + np.array([0.65, 0.0, 0.0], np.float32)
        name = f"{group_of(iid)}_{short(iid)}"
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            os.path.join(OUT, name + "_overlay.png"))
        Image.fromarray(np.where(tight, 0, 255).astype(np.uint8)).save(
            os.path.join(OUT, name + "_mask.png"))
        from scipy.ndimage import label as cclabel
        n_ind = int(cclabel(tight)[1])
        rows.append(dict(group=group_of(iid), file=meta.get("filename") or iid,
                         megapixels=round(g.size / 1e6, 2),
                         area_pct_wide=round(100 * float(wide.mean()), 3),
                         area_pct_tight=round(100 * float(tight.mean()), 3),
                         shrink_x=round(float(wide.mean()) / max(float(tight.mean()), 1e-12), 2),
                         indications=n_ind, overlay=name + "_overlay.png"))
        print(f"{name:52s} {rows[-1]['area_pct_wide']:7.3f}% -> "
              f"{rows[-1]['area_pct_tight']:7.3f}%  ({rows[-1]['shrink_x']}x)  "
              f"{n_ind} indications", flush=True)
        del rgb, g, wide, tight
    with open(os.path.join(OUT, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} frames -> {OUT}")


if __name__ == "__main__":
    main()
