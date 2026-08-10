"""
Force-NOT-crack on the false positives visible in results/final_71 overlays.

Everything here removes predictions; nothing adds any. That asymmetry is
deliberate -- the user corrected two attempts at automated POSITIVE labelling
(elongated inclusions mislabelled as crack; the large dark wedge mislabelled
as a thick crack), so positive labelling is left entirely to the paint tool.
Removing a prediction that is demonstrably wrong is a different and much
safer operation.

The three false-positive classes, each grounded in a stated fact rather than
in my reading of crack morphology:

  1. THE DARK WEDGE AND ITS RIM. The user stated plainly that the large dark
     wedge in the Wrought frames is NOT the crack (the real cracks there are
     thin, very faint, and in the centre of the frame). The model paints a
     rim around that wedge -- visible throughout
     _montage_Wrought_316L_H_Fatigue.png -- so the wedge plus a dilated
     margin is forced to not-crack. The margin is what catches the rim,
     which is where the local-contrast response actually peaks.

  2. THE SPECIMEN EDGE TRANSITION RING. Not the off-specimen field itself
     (already handled by write_corrections_from_review.py) but the soft
     bright/dark transition band just inside it, which the model also traces.

  3. ISOLATED ROUND SPECKLE. Small, low-eccentricity, isolated predicted
     regions -- surface texture and pores. Kept conservative: elongated
     components are NEVER removed, because a thin crack is elongated and
     the whole point is not to delete real signal. Since this model
     demonstrably does not detect the faint centre cracks at all, the
     scattered round specks it does emit are texture.

Never touches an existing non-zero correction value, so hand-drawn work and
the confirmed crack-free labels both survive.

Usage:
    python3 clean_false_positives.py [--dry-run] [--preview N]
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
from write_corrections_from_review import detect_offspecimen

PREDCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache")

DARK_MAX = 0.30           # flatfielded: below this is "dark" (wedge / void / empty)
WEDGE_MIN_AREA_FRAC = 0.004   # a wedge is a big feature; ignore small dark specks here
WEDGE_MARGIN = 45         # dilate the wedge by this to also clear the rim the model traces
EDGE_RING = 130           # width of the specimen-edge transition band to clear. 55 was
                          # measurably too narrow: a red vertical strip survived right along
                          # the specimen boundary (see results/fp_cleanup_preview/00.png at
                          # the previous setting). Widening is acceptable here because the
                          # cracks of interest are, per the user, in the CENTRE of the frame,
                          # so clearing a generous band at the edge does not cost real signal.
SPECKLE_MAX_AREA = 6000   # only small components are candidates for speckle removal
SPECKLE_MAX_ECC = 0.90    # ... and only if NOT elongated (never delete thin-crack-like shapes)


def false_positive_mask(img01, pred):
    """Regions where a predicted crack pixel is not credible."""
    fp = np.zeros(pred.shape, dtype=bool)

    # --- 1. dark wedge + margin (user: the wedge is NOT the crack) ---
    dark = ndi.gaussian_filter(img01.astype(np.float32), 3.0) < DARK_MAX
    lab = label(dark, connectivity=2)
    min_area = WEDGE_MIN_AREA_FRAC * dark.size
    wedge = np.zeros_like(dark)
    for r in regionprops(lab):
        if r.area >= min_area:
            wedge[lab == r.label] = True
    if wedge.any():
        fp |= ndi.binary_dilation(wedge, np.ones((3, 3)), iterations=WEDGE_MARGIN)

    # --- 2. specimen-edge transition ring ---
    off = detect_offspecimen(img01)
    if off.any():
        grown = ndi.binary_dilation(off, np.ones((3, 3)), iterations=EDGE_RING)
        fp |= (grown & ~off) | off

    # --- 3. isolated ROUND speckle (never elongated) ---
    plab = label(pred & ~fp, connectivity=2)
    for r in regionprops(plab):
        if r.area <= SPECKLE_MAX_AREA and r.eccentricity < SPECKLE_MAX_ECC:
            fp[plab == r.label] = True
    return fp, wedge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()
    pdir = os.path.join(pc.PROJECT_DIR, "results", "fp_cleanup_preview")
    if args.preview:
        os.makedirs(pdir, exist_ok=True)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    rows, npv = [], 0
    for info in pc.list_images():
        nm, grp = info["name"], info.get("group", "?")
        mp = os.path.join(PREDCACHE, f"{nm}_mask.npy")
        ip = os.path.join(PREDCACHE, f"{nm}_img.npy")
        if not (os.path.exists(mp) and os.path.exists(ip)):
            continue
        pred = np.load(mp); img01 = np.load(ip)
        fp, wedge = false_positive_mask(img01, pred)

        cp = os.path.join(pc.CORRECTIONS_DIR, f"{nm}_correction.npy")
        corr = np.load(cp) if os.path.exists(cp) else np.zeros(pred.shape, np.uint8)
        if corr.shape != pred.shape:
            del pred, img01, fp; continue

        tgt = pred & fp & (corr == 0)      # only where the model predicted crack, and unlabelled
        n = int(tgt.sum()); corr[tgt] = 2
        kept = int((pred & ~fp).sum())
        rows.append(dict(name=nm, group=grp, removed_px=n,
                         pred_px=int(pred.sum()), kept_px=kept,
                         removed_frac_of_pred=(n / max(int(pred.sum()), 1))))
        if n:
            print(f"  -{n:9,d} px ({n/max(pred.sum(),1)*100:5.1f}% of predicted) kept {kept/pred.size*100:5.2f}% "
                  f"[{grp[:20]:20s}] {nm[:40]}")
        if not args.dry_run:
            np.save(cp, corr)

        if args.preview and npv < args.preview and n:
            g = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
            a = np.stack([g]*3, -1); a[pred] = [255, 0, 0]
            b = np.stack([g]*3, -1); b[pred & ~fp] = [255, 0, 0]; b[pred & fp] = [0, 128, 255]
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
            ax[0].imshow(a, aspect="auto"); ax[0].set_title("current prediction (all red)", fontsize=9)
            ax[1].imshow(b, aspect="auto"); ax[1].set_title("BLUE = removed as false positive, RED = kept", fontsize=9)
            for x in ax: x.set_xticks([]); x.set_yticks([])
            fig.suptitle(nm[:66], fontsize=8); fig.tight_layout()
            fig.savefig(os.path.join(pdir, f"{npv:02d}.png"), dpi=95, bbox_inches="tight"); plt.close(fig)
            npv += 1
        del pred, img01, fp, corr

    tot = sum(r["removed_px"] for r in rows)
    print(f"\n{'DRY RUN' if args.dry_run else 'Wrote'}: removed {tot:,} false-positive px across {len(rows)} images")
    by = {}
    for r in rows: by.setdefault(r["group"], []).append(r)
    for g, rs in sorted(by.items()):
        fr = np.mean([r["removed_frac_of_pred"] for r in rs]) * 100
        print(f"  {g:26s} {sum(x['removed_px'] for x in rs):12,d} px  (mean {fr:4.1f}% of prediction removed)")
    if not args.dry_run:
        json.dump(rows, open(os.path.join(pc.PROJECT_DIR, "results", "fp_cleanup_summary.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
