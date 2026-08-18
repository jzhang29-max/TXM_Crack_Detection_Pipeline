"""
Mark visually-confirmed CRACK-FREE images: clear any force-crack label and
force the whole specimen interior to not-crack.

These are the highest-value labels in the whole training set, because every
one of the 16 originally-labelled images contains 19-30% real crack -- the
model had never seen a single example of crack-free material and therefore
had no way to learn that "no crack" is a possible answer. That is why an
UNDAMAGED zero-fatigue-cycle specimen was predicted at 41% crack.

The set below was determined by viewing each candidate flatfielded, with
the specimen interior contrast-stretched so faint features are visible.
Each listed image shows mottled microstructure with scattered small dark
blobs (inclusions / porosity) but NO elongated connected crack.

This also repairs a real error: write_positive_crack_labels.py wrote
force-CRACK on 12 of 13 B3 images. On these frames what its
elongation filter actually caught was elongated INCLUSIONS, not cracks --
visible directly in results/positive_label_preview and in the montage that
drove this list. Those labels are cleared here.

Deliberately NOT included (left fully untouched, contributing no training
signal either way):
  B2_3_1_lbf   -- one short dark line mid-right; inclusion or early initiation, unclear
  B2_3_2_lbf   -- several elongated dark specks centre-right; same ambiguity
  b3_3_0lbf    -- degenerate frame: large black bands are stitching/processing
                  artifacts, not material features
"""
import os, sys, numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
from write_corrections_from_review import detect_offspecimen

PREDCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache")
INTERIOR_ERODE = 40   # stay clear of the specimen edge transition ring

ZERO_CRACK = [
 "Average_mosaic_260618_b3_amb_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
 "Average_mosaic_260618_B2_amb_mosaic_2_idx00000_mosaictileAA_img001of005.xrm.bim.bim",
 "Average_mosaic_260618_B2_2_1_lbf_idx00000_mosaictileAA_img001of005.xrm.bim.bim",
 "Average_mosaic_260618_B2_2_9_lbf_idx00000_mosaictileAA_img001of005.xrm.bim.bim",
 "Average_mosaic_260620_b3_3_18lbf_348_13um_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
 "Average_mosaic_260620_wrought_316L_fatigue_0_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
]

tot_cleared = tot_neg = 0
for nm in ZERO_CRACK:
    ip = os.path.join(PREDCACHE, f"{nm}_img.npy")
    cp = os.path.join(pc.CORRECTIONS_DIR, f"{nm}_correction.npy")
    if not os.path.exists(ip):
        print(f"  [missing prediction] {nm[:58]}"); continue
    img = np.load(ip)
    corr = np.load(cp) if os.path.exists(cp) else np.zeros(img.shape, np.uint8)
    if corr.shape != img.shape:
        print(f"  [SHAPE] {nm[:58]}"); del img; continue

    cleared = int((corr == 1).sum())
    corr[corr == 1] = 0                       # no crack here, so no force-crack label

    interior = ~detect_offspecimen(img)
    interior = ndi.binary_erosion(interior, np.ones((3, 3)), iterations=INTERIOR_ERODE)
    corr[interior] = 2                        # entire specimen interior is not-crack

    n_neg = int((corr == 2).sum())
    np.save(cp, corr)
    tot_cleared += cleared; tot_neg += n_neg
    print(f"  cleared {cleared:8,d} bogus force-crack | {n_neg:9,d} px force-not-crack "
          f"({interior.mean()*100:4.1f}% of frame)  {nm[:44]}")
    del img, corr, interior

print(f"\n{len(ZERO_CRACK)} crack-free images: cleared {tot_cleared:,} bogus crack px, "
      f"set {tot_neg:,} not-crack px")
