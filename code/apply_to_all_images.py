"""
Applies whatever model is currently at models/pixel_hgb_final.joblib (the
production path) to all 12 images the paint tool knows about, and writes
each one's mask/overlay/stats to results/corrected/ via paint_common's own
regenerate_outputs -- so the output reflects EXACTLY what the paint tool
would show (fresh prediction under the current model, with each image's
saved corrections layered on top), not a separate ad-hoc code path.

Forces a fresh prediction for every image (bypassing any stale in-memory
or on-disk cache) so this is safe to run immediately after a model swap,
before anyone has opened the paint tool again.

Usage:
    python3 apply_to_all_images.py
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc

RESULTS_SUMMARY_PATH = os.path.join(pc.PROJECT_DIR, "results", "all_images_summary.json")


def main():
    # Wipe any on-disk prediction cache up front (belt-and-suspenders on
    # top of paint_common's own mtime-based auto-invalidation) so every
    # image below is guaranteed to be predicted fresh under whatever model
    # is currently at pc.MODEL_PATH.
    for f in glob.glob(os.path.join(pc.PREDICTED_CACHE_DIR, "*.npy")):
        os.remove(f)
    pc._mem_cache.clear()
    pc._model = None
    pc._model_mtime = None

    images = pc.list_images()
    print(f"Applying current production model ({pc.MODEL_PATH}) to {len(images)} images...\n")

    summary = []
    for info in images:
        name = info["name"]
        t0 = time.time()
        state = pc.get_state(name)
        mask = pc.effective_mask(state)
        pc.regenerate_outputs(name)
        elapsed = time.time() - t0

        from skimage.measure import label
        n_regions = int(label(mask, connectivity=2).max())
        area_fraction = float(mask.mean())
        n_corrected = int((state["correction"] != 0).sum())
        summary.append({
            "name": name, "shape": list(state["img01"].shape),
            "area_fraction": area_fraction, "n_regions": n_regions,
            "n_corrected_px": n_corrected, "seconds": elapsed,
        })
        print(f"  [{elapsed:5.1f}s] {name}: area_fraction={area_fraction:.4f} n_regions={n_regions} "
              f"(includes {n_corrected} previously-corrected px)")

    with open(RESULTS_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {RESULTS_SUMMARY_PATH}")
    print(f"Overlays/masks/stats written to {pc.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
