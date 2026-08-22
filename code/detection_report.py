"""Probability of detection versus flaw size, and false calls per frame. The NDT view.

    python3 code/detection_report.py

WHY THIS EXISTS, and why no comparable tool has it. Every interactive segmentation tool
surveyed for this project -- external, Fiji/TWS, Labkit, Dragonfly, Avizo, VGSTUDIO MAX, ZEN
Intellesis, MONAI Label, micro-sam, SAMBA, FeatureForest, napari-convpaint -- reports pixel
overlap: IoU, Dice, sometimes a validation curve. Pixel overlap cannot answer the two
questions an engineer actually asks:

    1. Of the flaws that are present, how many did it FIND AT ALL?
    2. On material containing no flaws, how many spurious indications will I have to dismiss?

Those are probability of detection and false-call rate, the two axes of MIL-HDBK-1823A, and
they behave differently from IoU. A model can score a respectable IoU by tracing large cracks
beautifully while missing most small ones, because pixel counts are dominated by the large
ones. This report separates the two.

HELD OUT. Detection is measured on full probability maps from models that never saw the image
being scored (build_loo_probmaps.py). In-sample detection rates would be meaningless.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))

import pipeline as P         # noqa: E402

SP = os.environ.get("SP", "/tmp")
BINS = [(0, 500, "under 500 px"), (500, 2000, "500 - 2 k"),
        (2000, 20000, "2 k - 20 k"), (20000, 10 ** 12, "over 20 k")]


def main():
    from skimage import measure
    have = [s for s in P.GT_STEMS
            if os.path.exists(os.path.join(SP, f"loo_prob_{s}.npy"))]
    if not have:
        raise SystemExit(f"no held-out probability maps in {SP}. Build them with "
                         f"scratch/build_loo_probmaps.py, or set SP to where they live.")

    print("PROBABILITY OF DETECTION vs flaw size  (held out: each image scored by a model")
    print("that never saw it; a flaw counts as detected if ANY of it is marked)\n")
    print(f"{'flaw size':<16} {'found':>7} {'of':>7} {'detected':>10}  {'pixels of that size':>20}")
    print("-" * 68)
    tot = {b: [0, 0, 0] for b in BINS}
    for s in have:
        pm = P.prune_specks(np.load(os.path.join(SP, f"loo_prob_{s}.npy"))
                            .astype(np.float32) > 0.5)
        gt = np.asarray(np.load(os.path.join(P.GT_CACHE, f"{s}_gt.npy"),
                                mmap_mode="r")).astype(bool)
        lab = measure.label(gt)
        for p in measure.regionprops(lab):
            for b in BINS:
                if b[0] <= p.area < b[1]:
                    hit = pm[p.slice][lab[p.slice] == p.label].any()
                    tot[b][0] += int(hit); tot[b][1] += 1; tot[b][2] += p.area
                    break
        del pm, gt, lab
    allpx = sum(t[2] for t in tot.values()) or 1
    for b in BINS:
        f, n, px = tot[b]
        print(f"{b[2]:<16} {f:>7} {n:>7} {100*f/max(n,1):>9.1f}%  {100*px/allpx:>19.1f}%")
    big = tot[BINS[-1]]
    print(f"\n  Read it this way: {100*big[0]/max(big[1],1):.0f}% of flaws over 20 k px are")
    print(f"  found, against {100*tot[BINS[0]][0]/max(tot[BINS[0]][1],1):.0f}% of those under")
    print(f"  500 px -- and the small ones are {100*tot[BINS[0]][2]/allpx:.1f}% of crack area,")
    print("  which is why a pixel-overlap score barely moves when they are missed.")

    fi = P.false_indications()
    if fi:
        print(f"\nFALSE CALLS on {fi['n_specimens']} specimens confirmed to contain no crack")
        print(f"{'specimen':<34} {'indications':>12} {'area':>9}")
        print("-" * 58)
        for p in fi["per_specimen"]:
            print(f"{(p['image'] or '')[22:54]:<34} {p['indications']:>12} "
                  f"{p['area_fraction']*100:>8.3f}%")
        print("-" * 58)
        print(f"{'MEAN':<34} {fi['mean_indications']:>12} "
              f"{fi['mean_area_fraction']*100:>8.3f}%")
        print(f"\n  {fi['zero_specimens']} of {fi['n_specimens']} specimens are completely "
              f"clean; the worst has {fi['max_indications']}.")
        print("  MIL-HDBK-1823A treats <=1% probability of false calls as the NDT yardstick.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
