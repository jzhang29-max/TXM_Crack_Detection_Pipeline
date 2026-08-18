"""
Add force-CRACK labels to the corrections, closing the gap left by
write_corrections_from_review.py (which only ever wrote force-NOT-crack,
so every one of the 59 new images contributed "0 +crack" and all crack
training pixels still came from the original B2 group).

Crack definition used here -- deliberately the same one the visual review
of all 71 images applied, implemented numerically:

    a crack is DARK RELATIVE TO ITS LOCAL SURROUNDINGS (not merely dark),
    ELONGATED rather than round, and lies INSIDE the specimen.

Each clause kills a specific known false positive:
  - local rather than global darkness: kills the off-specimen empty field
    and the broad illumination gradient, which is exactly what made the
    raw-trained model call 41% of an undamaged specimen "crack".
  - elongation: kills pores, inclusions and dust, which are round.
  - inside-specimen: kills the frame-edge / out-of-field regions.

Only HIGH-CONFIDENCE cores are labelled: the local-contrast threshold is
set well past the noise floor (LOCAL_CONTRAST_K sigma below the local
background) and small or round components are dropped. Everything else is
left as 0/untouched on purpose -- an untouched pixel contributes no
training signal, whereas a confidently-wrong one actively teaches the
model the wrong thing, and faint-thinning-vs-hairline-crack is precisely
the call this script is not qualified to make.

Never overwrites an existing non-zero correction value, so the 12
hand-labelled images and the off-specimen negatives both survive intact.

Usage:
    python3 write_positive_crack_labels.py [--dry-run] [--only NAME] [--preview N]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
from write_corrections_from_review import detect_offspecimen

PREDCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache")

LOCAL_BG_SIGMA = 40.0        # scale of "local surroundings"; well above crack width
LOCAL_CONTRAST_K = 2.2       # how many local-sigma below background to count as crack core
MIN_AREA = 400               # px; smaller than this is noise/dust, not a crack we can trust
MIN_ECCENTRICITY = 0.85      # strongly elongated only -- pores/inclusions are rounder
SPECIMEN_ERODE = 100         # keep well clear of the specimen edge transition ring.
                             # 25 was measurably too small: at that value the labels landed
                             # on the specimen boundary (itself locally dark against bright
                             # material) instead of the crack. 100 moved them onto the real crack.
MAX_PLAUSIBLE_FRAC = 0.015   # if the detector claims more than this fraction of the frame is
                             # crack CORE, it is not finding a crack -- it is outlining a thick
                             # one. Local-background subtraction is a band-pass filter, so inside
                             # a wide dark crack the local background is dark too and the
                             # contrast vanishes; only the perimeter survives. Training on a
                             # perimeter would teach 'crack = ring around a dark area' and yield
                             # hollow predictions, so those images are SKIPPED (left untouched)
                             # rather than mislabelled. Review put real crack near ~1% of frame.


def crack_cores(img01):
    """High-confidence crack cores: locally dark + elongated + inside specimen."""
    img = img01.astype(np.float32)
    bg = ndi.gaussian_filter(img, sigma=LOCAL_BG_SIGMA)
    resid = img - bg                                   # negative => darker than surroundings

    # Local scale of the residual, measured only over specimen interior so
    # the huge off-specimen step doesn't inflate it.
    offspec = detect_offspecimen(img01)
    specimen = ~offspec
    if SPECIMEN_ERODE:
        specimen = ndi.binary_erosion(specimen, np.ones((3, 3)), iterations=SPECIMEN_ERODE)
    if not specimen.any():
        return np.zeros_like(specimen), specimen

    sigma = float(np.std(resid[specimen]))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros_like(specimen), specimen

    dark = (resid < -LOCAL_CONTRAST_K * sigma) & specimen
    dark = ndi.binary_closing(dark, np.ones((3, 3)), iterations=2)

    keep = np.zeros_like(dark)
    lab = label(dark, connectivity=2)
    for r in regionprops(lab):
        if r.area < MIN_AREA:
            continue
        if r.eccentricity < MIN_ECCENTRICITY:
            continue
        keep[lab == r.label] = True
    return keep, specimen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="substring; restrict to matching image names")
    ap.add_argument("--preview", type=int, default=0, help="write N preview PNGs to results/positive_label_preview/")
    args = ap.parse_args()

    prev_dir = os.path.join(pc.PROJECT_DIR, "results", "positive_label_preview")
    if args.preview:
        os.makedirs(prev_dir, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    rows, n_prev = [], 0
    for info in pc.list_images():
        name = info["name"]
        if args.only and args.only not in name:
            continue
        ip = os.path.join(PREDCACHE, f"{name}_img.npy")
        if not os.path.exists(ip):
            continue
        img01 = np.load(ip)
        cores, specimen = crack_cores(img01)

        cpath = os.path.join(pc.CORRECTIONS_DIR, f"{name}_correction.npy")
        corr = np.load(cpath) if os.path.exists(cpath) else np.zeros(img01.shape, np.uint8)
        if corr.shape != img01.shape:
            print(f"  [SHAPE] {name[:56]}")
            del img01
            continue

        frac = float(cores.sum()) / cores.size
        if frac > MAX_PLAUSIBLE_FRAC:
            print(f"  [SKIP outline-failure {frac*100:5.2f}%] {info.get('group','?')[:20]:20s} {name[:44]}")
            rows.append(dict(name=name, group=info.get("group", "?"), forced_crack=0,
                             frac=0.0, skipped_outline_failure=True, detector_frac=frac))
            del img01, cores, specimen
            continue

        target = cores & (corr == 0)          # never clobber existing labels
        n_new = int(target.sum())
        corr[target] = 1

        rows.append(dict(name=name, group=info.get("group", "?"),
                         forced_crack=n_new, frac=float(n_new) / img01.size,
                         specimen_frac=float(specimen.mean())))
        if n_new:
            print(f"  {n_new:9,d} px -> CRACK ({n_new/img01.size*100:5.2f}% of frame)  "
                  f"[{info.get('group','?')[:20]:20s}] {name[:44]}")
        if not args.dry_run:
            np.save(cpath, corr)

        if args.preview and n_prev < args.preview and n_new:
            g = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
            rgb = np.stack([g] * 3, -1)
            rgb[cores] = [0, 255, 0]
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
            ax[0].imshow(g, cmap="gray", aspect="auto"); ax[0].set_title("flatfielded", fontsize=9)
            ax[1].imshow(rgb, aspect="auto"); ax[1].set_title(f"force-CRACK cores ({n_new/img01.size*100:.2f}%)", fontsize=9)
            for a in ax: a.set_xticks([]); a.set_yticks([])
            fig.suptitle(name[:70], fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(prev_dir, f"{n_prev:02d}_{info.get('group','x')[:12]}.png"), dpi=95, bbox_inches="tight")
            plt.close(fig)
            n_prev += 1
        del img01, cores, specimen, corr

    tot = sum(r["forced_crack"] for r in rows)
    print(f"\n{'DRY RUN' if args.dry_run else 'Wrote'}: {tot:,} px force-CRACK across {len(rows)} images")
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r)
    for g, rs in sorted(by.items()):
        t = sum(x["forced_crack"] for x in rs)
        nz = sum(1 for x in rs if x["forced_crack"])
        print(f"  {g:26s} {t:11,d} px across {nz}/{len(rs)} images")
    if not args.dry_run:
        with open(os.path.join(pc.PROJECT_DIR, "results", "positive_label_summary.json"), "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
