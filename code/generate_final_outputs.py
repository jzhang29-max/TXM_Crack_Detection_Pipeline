"""
Final deliverable: apply the chosen flatfielded model to all 71 images and
write, for each one, the same three artifacts the original pipeline
produced -- black-and-white crack mask (crack = BLACK, background = WHITE),
red-on-grayscale overlay, and a per-region stats CSV.

Input is the FLATFIELDED image, matching what the model was trained on.
Feeding it raw would be an input-distribution mismatch: raw median
brightness varies 2.6x across these specimen groups, which is the exact
confound that made the previous raw-trained model report 41% crack on an
undamaged specimen.

Outputs to results/final_71/ plus a summary CSV and a contact-sheet montage
per specimen group for quick review.

Usage:
    python3 generate_final_outputs.py --model models/pixel_flatfield_v5.joblib
"""
import argparse, csv, json, os, sys, time
import joblib, numpy as np, tifffile
from PIL import Image
from skimage.measure import label, regionprops
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
import build_flatfield_dataset as bf
from apply_pixel_model import predict_probability_map, postprocess_mask
from txm_features import robust_normalize

OUT = os.path.join(pc.PROJECT_DIR, "results", "final_71")
PREDCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(pc.PROJECT_DIR, "models", "pixel_flatfield_v5.joblib"))
    ap.add_argument("--montage", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    model = joblib.load(args.model)
    print(f"model: {os.path.basename(args.model)}  ->  {OUT}\n")

    rows = []
    for info in pc.list_images():
        nm, grp = info["name"], info.get("group", "?")
        t0 = time.time()
        # Reuse the cached flatfielded normalized image when present.
        ip = os.path.join(PREDCACHE, f"{nm}_img.npy")
        if os.path.exists(ip):
            img01 = np.load(ip).astype(np.float64)
        else:
            ff = bf.flatfield_path_for(info["path"])
            if ff is None:
                print(f"  [skip] no flatfielded counterpart: {nm[:56]}"); continue
            img01 = robust_normalize(tifffile.imread(ff).astype(np.float64), 1.0, 99.0)

        mask = postprocess_mask(predict_probability_map(model, img01))

        # crack = BLACK (0), background = WHITE (255) -- same convention as
        # the original apply_pixel_model.save_outputs deliverable.
        Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L").save(
            os.path.join(OUT, f"{nm}_crack_mask.png"))
        g = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
        ov = np.stack([g] * 3, -1); ov[mask] = [255, 0, 0]
        Image.fromarray(ov, mode="RGB").save(os.path.join(OUT, f"{nm}_overlay.png"))

        lab = label(mask, connectivity=2)
        with open(os.path.join(OUT, f"{nm}_stats.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "area_px", "solidity", "eccentricity", "centroid_row", "centroid_col"])
            for r in regionprops(lab):
                w.writerow([r.label, r.area, f"{r.solidity:.4f}", f"{r.eccentricity:.4f}",
                            f"{r.centroid[0]:.1f}", f"{r.centroid[1]:.1f}"])

        rows.append(dict(name=nm, group=grp, area_fraction=float(mask.mean()),
                         n_regions=int(lab.max()), shape=list(mask.shape)))
        print(f"  [{time.time()-t0:5.1f}s] {mask.mean()*100:5.1f}% {lab.max():4d}rg  [{grp[:20]:20s}] {nm[:44]}")
        del img01, mask, ov, lab

    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "group", "area_fraction", "n_regions"])
        for r in rows: w.writerow([r["name"], r["group"], f"{r['area_fraction']:.5f}", r["n_regions"]])
    json.dump(rows, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    print(f"\n{len(rows)} images -> {OUT}")
    by = {}
    for r in rows: by.setdefault(r["group"], []).append(r)
    print(f"\n{'group':26s} {'n':>3s} {'median area':>12s} {'median regions':>15s}")
    for g, rs in sorted(by.items()):
        a = np.median([r["area_fraction"] for r in rs]); n = np.median([r["n_regions"] for r in rs])
        print(f"{g:26s} {len(rs):3d} {a:11.3f} {n:15.0f}")

    if args.montage:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for g, rs in sorted(by.items()):
            rs = sorted(rs, key=lambda r: r["name"])
            cols = 5; rws = (len(rs) + cols - 1) // cols
            fig, axes = plt.subplots(rws, cols, figsize=(3.0 * cols, 2.1 * rws))
            axes = np.atleast_2d(axes)
            for k, r in enumerate(rs):
                ax = axes[k // cols, k % cols]
                p = os.path.join(OUT, f"{r['name']}_overlay.png")
                if os.path.exists(p):
                    im = Image.open(p); im.thumbnail((420, 420))
                    ax.imshow(np.array(im), aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{r['name'].split('_idx')[0][-26:]}\n{r['area_fraction']*100:.1f}%, {r['n_regions']}rg", fontsize=6)
            for k in range(len(rs), rws * cols):
                axes[k // cols, k % cols].axis("off")
            fig.suptitle(f"{g} -- final crack overlays ({len(rs)} images)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            safe = g.replace(" ", "_")
            fig.savefig(os.path.join(OUT, f"_montage_{safe}.png"), dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"  montage: _montage_{safe}.png")

if __name__ == "__main__":
    main()
