"""
Apply the flatfielded model (models/pixel_flatfield.joblib) to every image's
FLATFIELDED counterpart, and report area fraction / region count per image
plus a group summary and an explicit negative-control check.

The negative-control check is the decisive one: specimens labelled with
zero fatigue cycles or zero load are UNDAMAGED and must come out near
crack-free. The raw-trained model failed this badly (an undamaged
zero-cycle specimen predicted at 41% crack / 256 regions), which is what
motivated moving to flatfielded input in the first place.

Writes results/flatfield_all_summary.json and caches each image's
flatfielded prediction to paint/flatfield_predcache/ so the markup step
doesn't have to recompute them.

Usage:
    python3 apply_flatfield_all.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import tifffile
from skimage.measure import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
import build_flatfield_dataset as bf
from apply_pixel_model import postprocess_mask, predict_probability_map
from txm_features import robust_normalize

MODEL_PATH = os.path.join(pc.PROJECT_DIR, "models", "pixel_flatfield.joblib")
PREDCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache")
SUMMARY = os.path.join(pc.PROJECT_DIR, "results", "flatfield_all_summary.json")
os.makedirs(PREDCACHE, exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)

# Substrings marking specimens that are undamaged / unloaded and therefore
# must be essentially crack-free -- the objective sanity check.
NEGATIVE_CONTROLS = ["_0_cycles", "0lbf", "_amb", "2_1_lbf"]


def is_negative_control(name):
    return any(k in name for k in NEGATIVE_CONTROLS)


def main():
    model = joblib.load(MODEL_PATH)
    images = pc.list_images()
    print(f"Applying {os.path.basename(MODEL_PATH)} to {len(images)} FLATFIELDED images...\n")

    rows = []
    for info in images:
        name = info["name"]
        ff = bf.flatfield_path_for(info["path"])
        if ff is None:
            print(f"  [skip] {name[:60]}: no flatfielded counterpart")
            continue
        t0 = time.time()
        img01 = robust_normalize(tifffile.imread(ff).astype(np.float64), 1.0, 99.0)
        prob = predict_probability_map(model, img01)
        mask = postprocess_mask(prob)
        np.save(os.path.join(PREDCACHE, f"{name}_mask.npy"), mask)
        np.save(os.path.join(PREDCACHE, f"{name}_img.npy"), img01.astype(np.float32))
        n_regions = int(label(mask, connectivity=2).max())
        rows.append(dict(name=name, group=info.get("group", "?"),
                         area_fraction=float(mask.mean()), n_regions=n_regions,
                         negative_control=is_negative_control(name),
                         seconds=time.time() - t0))
        print(f"  [{time.time()-t0:5.1f}s] {mask.mean()*100:5.1f}% {n_regions:4d}rg  "
              f"{'[NEG-CTRL] ' if is_negative_control(name) else ''}{name[:58]}")
        del img01, prob, mask

    with open(SUMMARY, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'='*76}\nGROUP SUMMARY (flatfielded model)\n{'='*76}")
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r)
    print(f"{'group':26s} {'n':>3s} {'median area':>12s} {'min':>7s} {'max':>7s} {'med rg':>7s}")
    for g, rs in sorted(by.items()):
        a = np.array([r["area_fraction"] for r in rs])
        n = np.array([r["n_regions"] for r in rs])
        print(f"{g:26s} {len(rs):3d} {np.median(a):11.3f} {a.min():7.3f} {a.max():7.3f} {np.median(n):7.0f}")

    print(f"\n{'='*76}\nNEGATIVE CONTROLS -- undamaged/unloaded, must be near crack-free\n{'='*76}")
    ncs = [r for r in rows if r["negative_control"]]
    for r in sorted(ncs, key=lambda r: -r["area_fraction"]):
        print(f"  {r['area_fraction']*100:5.1f}% crack, {r['n_regions']:4d} regions   {r['name'][:62]}")
    if ncs:
        print(f"\n  median negative-control crack area: "
              f"{np.median([r['area_fraction'] for r in ncs])*100:.1f}%  "
              f"(raw-trained model gave 26-41% on these -- lower is better)")
    print(f"\nSaved {SUMMARY}")


if __name__ == "__main__":
    main()
