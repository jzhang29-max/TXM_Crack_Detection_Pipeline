"""
Per-group input + model selection -- the resolution of the raw-vs-flatfielded
trade-off, which is real and structural.

MEASURED FACTS behind this design:

  1. Flatfielding DOES remove the mosaic tile grid, ~10x inside the specimen
     (column-profile periodicity over specimen pixels only: wrought_800
     0.943->0.088, wrought_900 0.947->0.071, b2_338_13 0.858->0.097).
     Measuring over the whole frame instead gives a misleading ~0.96 because
     the dark off-specimen band dominates the profile -- that error is what
     briefly made it look as though flatfielding did nothing.

  2. Flatfielding also destroys what B2 detection relies on. The largest
     feature group is large-radius smoothed intensity (~41% of total
     importance) which encodes "is this pixel inside a broad dark region";
     flatfielding's entire purpose is to remove broad trends. Measured cost on
     the 4 ground-truth images: IoU 0.779 (raw) -> 0.610 (flatfielded).

  3. On raw input the AM and Wrought groups trace the tile seams -- median
     predicted area 11.5% and 23.6% with region counts RISING (112->188,
     120->204), i.e. the mask fragments along the grid.

So neither input serves both groups, and no single model should be forced to.

  B2, B3  -> RAW input + models/pixel_hgb_final.joblib (raw_v4)
             verified: IoU 0.773, recall 0.881 on ground truth
  AM, Wrought -> FLATFIELDED input + models/pixel_flatfield_hgb.joblib
             the grid is gone there, which is the dominant failure on raw

Honest limitation: there is NO pixel-level ground truth for AM or Wrought, so
their assignment rests on grid suppression plus the crack-free negative
controls, not on IoU. Their outputs remain less trustworthy than B2/B3 and are
labelled as such.

Usage:
    python3 build_outputs_per_group.py
"""
import csv, json, os, sys, time
import joblib, numpy as np, tifffile
from PIL import Image
from skimage.measure import label, regionprops
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
import build_flatfield_dataset as bf
from apply_pixel_model import postprocess_mask, predict_probability_map
from txm_features import robust_normalize

OUT = os.path.join(pc.PROJECT_DIR, "results", "final_71_pergroup")
RAW_MODEL  = os.path.join(pc.PROJECT_DIR, "models", "pixel_hgb_final.joblib")        # raw_v4
FLAT_MODEL = os.path.join(pc.PROJECT_DIR, "models", "pixel_flatfield_hgb.joblib")
FLAT_GROUPS = {"AM 316LH Fatigue", "Wrought 316L H Fatigue"}
VERIFIED    = {"B2 316L H Tension", "B3 316L Amb Tension"}

def main():
    os.makedirs(OUT, exist_ok=True)
    raw_m, flat_m = joblib.load(RAW_MODEL), joblib.load(FLAT_MODEL)
    print(f"RAW  groups -> {os.path.basename(RAW_MODEL)}")
    print(f"FLAT groups -> {os.path.basename(FLAT_MODEL)}  ({', '.join(sorted(FLAT_GROUPS))})\n")
    rows = []
    for info in pc.list_images():
        nm, grp = info["name"], info.get("group", "?")
        use_flat = grp in FLAT_GROUPS
        path = bf.flatfield_path_for(info["path"]) if use_flat else info["path"]
        if path is None:
            print(f"  [skip, no flatfielded counterpart] {nm[:52]}"); continue
        t0 = time.time()
        img01 = robust_normalize(tifffile.imread(path).astype(np.float64), 1.0, 99.0)
        mask = postprocess_mask(predict_probability_map(flat_m if use_flat else raw_m, img01))
        Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), "L").save(os.path.join(OUT, f"{nm}_crack_mask.png"))
        g = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
        ov = np.stack([g] * 3, -1); ov[mask] = [255, 0, 0]
        Image.fromarray(ov, "RGB").save(os.path.join(OUT, f"{nm}_overlay.png"))
        lab = label(mask, connectivity=2)
        with open(os.path.join(OUT, f"{nm}_stats.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id","area_px","solidity","eccentricity","centroid_row","centroid_col"])
            for r in regionprops(lab):
                w.writerow([r.label, r.area, f"{r.solidity:.4f}", f"{r.eccentricity:.4f}",
                            f"{r.centroid[0]:.1f}", f"{r.centroid[1]:.1f}"])
        rows.append(dict(name=nm, group=grp, input="flatfielded" if use_flat else "raw",
                         area_fraction=float(mask.mean()), n_regions=int(lab.max()),
                         verified_vs_gt=grp in VERIFIED))
        print(f"  [{time.time()-t0:5.1f}s] {mask.mean()*100:5.1f}% {lab.max():4d}rg  "
              f"{'FLAT' if use_flat else 'raw '}  [{grp[:18]:18s}] {nm[:32]}")
        del img01, mask, ov, lab
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name","group","input","area_fraction","n_regions","verified_vs_gt"])
        for r in rows: w.writerow([r["name"], r["group"], r["input"], f"{r['area_fraction']:.5f}",
                                    r["n_regions"], r["verified_vs_gt"]])
    json.dump(rows, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    by = {}
    for r in rows: by.setdefault(r["group"], []).append(r)
    print(f"\n{len(rows)} images -> {OUT}\n")
    print(f"{'group':26s} {'n':>3s} {'input':>12s} {'median area':>12s} {'med rg':>7s}  trust")
    for g, rs in sorted(by.items()):
        print(f"{g:26s} {len(rs):3d} {rs[0]['input']:>12s} "
              f"{np.median([r['area_fraction'] for r in rs])*100:11.1f}% "
              f"{np.median([r['n_regions'] for r in rs]):7.0f}  "
              f"{'VERIFIED vs GT' if g in VERIFIED else 'no GT exists - weaker evidence'}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for g, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["name"]); cols = 5; rws = (len(rs)+cols-1)//cols
        fig, axes = plt.subplots(rws, cols, figsize=(3.0*cols, 2.1*rws)); axes = np.atleast_2d(axes)
        for k, r in enumerate(rs):
            ax = axes[k//cols, k%cols]
            p = os.path.join(OUT, f"{r['name']}_overlay.png")
            if os.path.exists(p):
                im = Image.open(p); im.thumbnail((420,420)); ax.imshow(np.array(im), aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{r['name'].split('_idx')[0][-26:]}\n{r['area_fraction']*100:.1f}%, {r['n_regions']}rg", fontsize=6)
        for k in range(len(rs), rws*cols): axes[k//cols, k%cols].axis("off")
        fig.suptitle(f"{g} -- {rs[0]['input'].upper()} input -- "
                     f"{'VERIFIED vs ground truth' if g in VERIFIED else 'no ground truth exists'}", fontsize=12)
        fig.tight_layout(rect=[0,0,1,0.97])
        fig.savefig(os.path.join(OUT, f"_montage_{g.replace(' ','_')}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("montages written")

if __name__ == "__main__":
    main()
