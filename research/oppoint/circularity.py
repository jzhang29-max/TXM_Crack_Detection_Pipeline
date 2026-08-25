"""How circular are the owner's correction labels, and does dense GT align with prob?

Two questions that decide whether a threshold sweep on corrections means anything:
  1. What fraction of correction==1 pixels already have prob > 0.50?
     build_label_folds.py claims 98.3% (for an older model). If true for v4, then
     recall-on-corrections at 0.50 is high BY CONSTRUCTION and the sweep is rigged.
  2. Do the 4 dense GT frames in dataset_cache line up with app prob.npy shapes?

Read-only. Writes research/oppoint/circularity.json.
"""
import sys, os, json, glob
import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
import store as S, pipeline as P

OUT = os.path.join(P0, "research", "oppoint")
inv = json.load(open(os.path.join(OUT, "inventory.json")))

# ---- 1. circularity of positive labels -------------------------------------
tot1 = above = 0
per = []
for r in inv["rows"]:
    if not r["n_crack"]:
        continue
    prob = S.load_npy(r["iid"], "prob.npy")
    corr = S.load_npy(r["iid"], "correction.npy")
    if prob is None or corr is None or prob.shape != corr.shape:
        continue
    p1 = prob[corr == 1]
    n = p1.size
    a = int((p1 > 0.50).sum())
    tot1 += n; above += a
    per.append(dict(f=r["filename"][:55], n=n, frac_above_050=round(a / max(n, 1), 4),
                    med=round(float(np.median(p1)), 4)))
    del prob, corr, p1

print(f"correction==1 pixels: {tot1:,}   with prob>0.50: {above:,} "
      f"({above/max(tot1,1)*100:.2f}%)")
print("\nper-image fraction of force-crack labels already above 0.50:")
for d in sorted(per, key=lambda d: d["frac_above_050"]):
    print(f"  {d['frac_above_050']:.4f}  med_p={d['med']:.3f}  n={d['n']:>8}  {d['f']}")

# ---- 2. dense GT alignment -------------------------------------------------
gt_rows = []
for g in sorted(glob.glob(os.path.join(P0, "dataset_cache", "*_gt.npy"))):
    stem = os.path.basename(g)[:-7]
    gt = np.load(g, mmap_mode="r")
    match = [r for r in inv["rows"]
             if stem.lower().replace("large_", "").replace("_um_zoom", "")
             in r["filename"].lower().replace("_um_zoom", "")]
    # fall back to a looser token match
    if not match:
        tok = stem.lower().split("_")
        match = [r for r in inv["rows"]
                 if all(t in r["filename"].lower() for t in tok if t != "large")]
    info = dict(stem=stem, gt_shape=list(gt.shape), gt_dtype=str(gt.dtype),
                gt_uniq=[int(v) for v in np.unique(np.asarray(gt[::16, ::16]))],
                gt_frac=round(float((np.asarray(gt[::8, ::8]) > 0).mean()), 5),
                candidates=[m["filename"][:60] for m in match],
                prob_shapes=[m["prob_shape"] for m in match],
                iids=[m["iid"] for m in match])
    gt_rows.append(info)
    print(f"\n{stem}: gt{gt.shape} {gt.dtype} uniq={info['gt_uniq']} "
          f"crackfrac~{info['gt_frac']}")
    for m in match:
        print(f"    -> {m['filename'][:60]} prob={m['prob_shape']} "
              f"shape_match={m['prob_shape']==list(gt.shape)}")

json.dump(dict(pos_total=tot1, pos_above_050=above,
               frac=above / max(tot1, 1), per_image=per, gt=gt_rows),
          open(os.path.join(OUT, "circularity.json"), "w"), indent=2)
