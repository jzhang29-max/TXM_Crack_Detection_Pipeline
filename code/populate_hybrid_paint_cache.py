"""
Pre-fill the paint tool's HYBRID prediction cache from the masks already
computed by apply_sam_hybrid.py, so opening an image in hybrid mode is instant
instead of a ~70 s SAM prediction pass.

Without this, the first visit to each of 71 images costs ~70 s and the tool
feels broken. The masks already exist; this only copies them into the layout
get_state() reads, alongside the RAW normalised image (hybrid mode serves raw
for every group, so the displayed image must be raw too or the picture and the
mask come from different inputs).

Writes paint/predicted_cache_hybrid/{name}_{mask,img}.npy. Never touches the
default cache at paint/predicted_cache_pergroup/, so switching back is instant.

Usage:
    python3 populate_hybrid_paint_cache.py
    TXM_PAINT_HYBRID=1 python3 code/paint_server.py      # then serve it
"""

import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TXM_PAINT_HYBRID", "1")     # so pc points at the hybrid cache
import paint_common as pc
from txm_features import robust_normalize

MASK_SRC = os.path.join(pc.PROJECT_DIR, "results", "sam_hybrid_71", "masks")


def main():
    if not pc.USE_HYBRID:
        sys.exit("hybrid mode did not engage -- is models/pixel_sam_hybrid.joblib present?")
    os.makedirs(pc.HYBRID_CACHE_DIR, exist_ok=True)
    infos = pc.list_images()
    print(f"source masks: {MASK_SRC}")
    print(f"target cache: {pc.HYBRID_CACHE_DIR}")
    print(f"{len(infos)} images\n")

    done = skipped = missing = 0
    for k, info in enumerate(infos, 1):
        name = info["name"]
        src = os.path.join(MASK_SRC, f"{name}_mask.npy")
        if not os.path.exists(src):
            print(f"  [{k:2d}] MISSING hybrid mask: {name[:48]}")
            missing += 1
            continue
        mask_path, img_path = pc._cache_paths(name)
        if os.path.exists(mask_path) and os.path.exists(img_path):
            skipped += 1
            continue
        mask = np.load(src).astype(bool)
        # DISPLAY image, not the model's input. The hybrid is fed raw (that is
        # what it was trained on); the human needs the contrast-enhanced version
        # because the real cracks are thin and faint. Both processing steps are
        # geometry-preserving, so the raw-derived mask still registers.
        src_path = pc._find_path(name)
        dp = pc.display_path_for(src_path)
        used = pc.DISPLAY_SET if dp else "raw"
        img01 = robust_normalize(tifffile.imread(dp or src_path).astype(np.float64), 1.0, 99.0)
        if img01.shape != mask.shape:
            img01 = robust_normalize(tifffile.imread(src_path).astype(np.float64), 1.0, 99.0)
            used = "raw (shape mismatch on processed)"
        if img01.shape != mask.shape:
            print(f"  [{k:2d}] SHAPE MISMATCH img{img01.shape} mask{mask.shape}: {name[:40]}")
            missing += 1
            continue
        np.save(mask_path, mask)
        np.save(img_path, img01)
        done += 1
        print(f"  [{k:2d}/{len(infos)}] {info.get('group','?')[:18]:18s} "
              f"{mask.mean()*100:5.2f}% crack  disp={used:12s} {name[:32]}")
        del mask, img01

    print(f"\npopulated {done}, already present {skipped}, unusable {missing}")
    if missing:
        print("  images without a cached mask will be predicted on demand (~70 s each)")
    print(f"\nStart the tool in hybrid mode with:\n"
          f"  TXM_PAINT_HYBRID=1 python3 code/paint_server.py")


if __name__ == "__main__":
    main()
