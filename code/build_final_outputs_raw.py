"""
Final outputs from the ORIGINAL RAW-trained model -- the best model by the only
metric that has ground truth behind it.

Measured on the 4 Ilastik ground-truth images, IoU:
    ORIG raw MLP   0.779   <- this script uses it
    ORIG raw HGB   0.764
    flatfield MLP  0.610
    flatfield HGB  0.607
    flatfield MLP + geometric post-processing (final_71_v2)   0.524

The flatfielded pipeline was adopted on the strength of the NEW specimen groups
(Wrought over-prediction 68.7%->28.7%, an undamaged specimen 41%->1.3%). That
evidence is real but it only measures FALSE POSITIVES. IoU against ground truth
was never checked before the switch, and it is 0.17 worse. Restoring raw.

NO geometric post-processing here beyond apply_pixel_model.postprocess_mask.
The extra geometric/curvilinearity filters cost a further 0.086 IoU
(0.610 -> 0.524) on top of the flatfielding loss, so they are not applied.

Known and unfixed: this model floods the AM and Wrought groups (predicting
crack over 40-70% of some frames, and 41% on an undamaged specimen). That is a
LABELLING gap -- the model has never seen an AM or Wrought crack -- not
something preprocessing fixed. Outputs for those groups should be treated as
unreliable; see HANDOFF.md. B2 and B3 are the trustworthy groups.

Usage:
    python3 build_final_outputs_raw.py
"""
import csv, json, os, sys, time
import joblib, numpy as np, tifffile
from PIL import Image
from skimage.measure import label, regionprops
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
from apply_pixel_model import postprocess_mask, predict_probability_map
from txm_features import robust_normalize

MODEL = os.path.join(pc.PROJECT_DIR, "models", "pixel_hgb_final.joblib")   # the ORIG raw MLP
OUT = os.path.join(pc.PROJECT_DIR, "results", "final_71_raw")
# Groups where this model is verified against ground truth vs. where it is known to flood.
TRUSTED = {"B2 316L H Tension", "B3 316L Amb Tension"}

def main():
    os.makedirs(OUT, exist_ok=True)
    model = joblib.load(MODEL)
    print(f"model: {os.path.basename(MODEL)} (ORIG raw MLP, mean IoU 0.779 on GT)\n")
    rows = []
    for info in pc.list_images():
        nm, grp = info["name"], info.get("group", "?")
        t0 = time.time()
        img01 = robust_normalize(tifffile.imread(info["path"]).astype(np.float64), 1.0, 99.0)
        mask = postprocess_mask(predict_probability_map(model, img01))
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
        rows.append(dict(name=nm, group=grp, area_fraction=float(mask.mean()),
                         n_regions=int(lab.max()), trusted_group=grp in TRUSTED))
        flag = "" if grp in TRUSTED else "   [UNRELIABLE GROUP]"
        print(f"  [{time.time()-t0:5.1f}s] {mask.mean()*100:5.1f}% {lab.max():4d}rg  [{grp[:18]:18s}] {nm[:34]}{flag}")
        del img01, mask, ov, lab
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name","group","area_fraction","n_regions","trusted_group"])
        for r in rows: w.writerow([r["name"], r["group"], f"{r['area_fraction']:.5f}", r["n_regions"], r["trusted_group"]])
    json.dump(rows, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    by = {}
    for r in rows: by.setdefault(r["group"], []).append(r)
    print(f"\n{len(rows)} images -> {OUT}\n")
    print(f"{'group':26s} {'n':>3s} {'median area':>12s} {'median rg':>10s}  trust")
    for g, rs in sorted(by.items()):
        t = "VERIFIED vs GT" if g in TRUSTED else "floods - unreliable"
        print(f"{g:26s} {len(rs):3d} {np.median([r['area_fraction'] for r in rs])*100:11.1f}% "
              f"{np.median([r['n_regions'] for r in rs]):10.0f}  {t}")
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
        tag = "VERIFIED vs ground truth" if g in TRUSTED else "KNOWN TO FLOOD - unreliable"
        fig.suptitle(f"{g} -- ORIG raw model ({len(rs)} images) -- {tag}", fontsize=12)
        fig.tight_layout(rect=[0,0,1,0.97])
        fig.savefig(os.path.join(OUT, f"_montage_{g.replace(' ','_')}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("montages written")

if __name__ == "__main__":
    main()
