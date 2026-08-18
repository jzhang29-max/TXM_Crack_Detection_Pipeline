"""Three-panel overlays for all 71: flatfielded image | deployed model | new model.

Separate from compare_hybrid_vs_current.py because that one compares the
17-feature pipeline against the hybrid; this compares two HYBRIDS -- the
deployed one against a retrain candidate -- which is the question after a
markup round.
"""
import os, sys, glob, json
import numpy as np, tifffile, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc, txm_preprocess as tp
from txm_features import robust_normalize

DEPLOYED = "results/sam_hybrid_71/masks"
CAND     = "results/sam_hybrid_cand16862/masks"
OUT      = os.path.join(pc.PROJECT_DIR, "results", "cand_vs_deployed")
_GB = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
if os.path.exists(_GB): fm.fontManager.addfont(_GB)
TF = fm.FontProperties(fname=_GB) if os.path.exists(_GB) else None
CUR = (0.10, 0.69, 0.48, 0.55); NEW = (0.16, 0.47, 0.84, 0.55)

def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    infos = pc.list_images()
    for k, info in enumerate(infos, 1):
        n = info["name"]
        dp, cp = f"{DEPLOYED}/{n}_mask.npy", f"{CAND}/{n}_mask.npy"
        if not (os.path.exists(dp) and os.path.exists(cp)):
            continue
        dep, cand = np.load(dp), np.load(cp)
        if dep.shape != cand.shape:
            continue
        img = robust_normalize(np.asarray(tp.get(pc._find_path(n), "flatfielded"), np.float64), 1.0, 99.0)
        u = np.logical_or(dep, cand).sum()
        ag = float(np.logical_and(dep, cand).sum() / u) if u else float("nan")
        rows.append(dict(name=n, group=info.get("group"), deployed=float(dep.mean()),
                         candidate=float(cand.mean()), agreement=ag))
        fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
        for a, (m, t, c) in zip(ax, [(None, "flatfielded image", None),
                (dep, f"DEPLOYED — {dep.mean()*100:.2f}%", CUR),
                (cand, f"NEW (your labels) — {cand.mean()*100:.2f}%", NEW)]):
            a.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="equal")
            if m is not None:
                o = np.zeros((*img.shape, 4), np.float32); o[m] = c; a.imshow(o, aspect="equal")
            a.set_title(t, fontsize=11); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{info.get('group','?')}  ·  {n[:64]}   agreement {ag:.3f}",
                     fontproperties=TF, fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{n}_cand.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{k:2d}/{len(infos)}] {info.get('group','?')[:18]:18s} "
              f"{dep.mean()*100:6.2f}% -> {cand.mean()*100:6.2f}%  ag {ag:.3f}  {n[:32]}", flush=True)
        del img, dep, cand
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(dict(deployed_dir=DEPLOYED, candidate_dir=CAND, rows=rows), f, indent=2)
    print(f"\n{len(rows)} overlays -> {OUT}")

if __name__ == "__main__":
    main()
