"""
Side-by-side overlays: current pipeline vs SAM+17 hybrid, for every image.

This is the artefact the deployment decision should actually rest on. The
numbers so far come from four B2 images with wide-open cracks (median width
65 px); 27 of the 71 images -- all of B3 and Wrought -- have never had a single
crack pixel labelled, and HANDOFF.md records their cracks as thin, faint and
central, the regime where the literature says SAM fails. No metric exists for
those images, so the only honest test is to look.

Renders one PNG per image (original | current | hybrid) with predicted area and
region count, plus a per-group contact sheet and a summary table sorted by how
much the two models DISAGREE -- disagreement is where looking pays off, and
sorting by it puts the informative cases first instead of burying them.

Usage:
    python3 compare_hybrid_vs_current.py
    python3 compare_hybrid_vs_current.py --group "Wrought 316L H Fatigue"
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc

PROJECT_DIR = pc.PROJECT_DIR
CURRENT_DIR = os.path.join(PROJECT_DIR, "results", "final_71_pergroup")
HYBRID_DIR = os.path.join(PROJECT_DIR, "results", "sam_hybrid_71")
OUT_DIR = os.path.join(PROJECT_DIR, "results", "hybrid_vs_current")

_GB = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
if os.path.exists(_GB):
    fm.fontManager.addfont(_GB)
TITLE_FONT = fm.FontProperties(fname=_GB) if os.path.exists(_GB) else None
plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 130,
                     "axes.grid": False, "font.size": 10.5})

CUR_COL = (0.10, 0.69, 0.48, 0.50)      # green, matching every existing figure
HYB_COL = (0.16, 0.47, 0.84, 0.50)      # blue, the SAM family colour


def find_mask(directory, name):
    """Locate a saved mask for `name`, whatever the layout.

    The two pipelines store masks differently and both must be readable: the
    current one writes `<name>_crack_mask.png` (8-bit, crack = BLACK per this
    project's convention), the hybrid writes `masks/<name>_mask.npy`.
    """
    for pat in (os.path.join(directory, "masks", f"{name}_mask.npy"),
                os.path.join(directory, f"{name}_mask.npy"),
                os.path.join(directory, f"{name}_crack_mask.png"),
                os.path.join(directory, "**", f"*{name}*mask*.npy"),
                os.path.join(directory, "**", f"*{name}*crack_mask*.png")):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def read_mask(path):
    """Load either storage form as a boolean crack mask."""
    if path.endswith(".npy"):
        return np.load(path).astype(bool)
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"))
    # Crack is drawn BLACK on white in this project's B&W outputs, so the dark
    # pixels are the mask. Guard against the inverse convention by checking
    # which polarity is the minority -- crack is never most of the frame.
    dark = a < 128
    return dark if dark.mean() <= 0.5 else ~dark


def load_img(name):
    p = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache", f"{name}_img.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    import tifffile
    from txm_features import robust_normalize
    raw = tifffile.imread(pc._find_path(name)).astype(np.float64)
    return robust_normalize(raw, 1.0, 99.0).astype(np.float32)


def overlay(ax, img, mask, rgba, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="equal")
    if mask is not None:
        o = np.zeros((*img.shape, 4), np.float32)
        o[mask] = rgba
        ax.imshow(o, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    infos = [i for i in pc.list_images()
             if not args.group or i.get("group") == args.group]

    rows = []
    for k, info in enumerate(infos, 1):
        name = info["name"]
        cp, hp = find_mask(CURRENT_DIR, name), find_mask(HYBRID_DIR, name)
        if hp is None:
            print(f"  [{k:2d}/{len(infos)}] no hybrid mask yet: {name[:44]}")
            continue
        hyb = read_mask(hp)
        cur = read_mask(cp) if cp else None
        img = np.asarray(load_img(name), np.float32)
        if cur is not None and cur.shape != hyb.shape:
            print(f"  [{k:2d}] shape mismatch cur{cur.shape} hyb{hyb.shape}, skipping current")
            cur = None

        inter = int(np.logical_and(cur, hyb).sum()) if cur is not None else 0
        union = int(np.logical_or(cur, hyb).sum()) if cur is not None else 0
        agree = inter / union if union else float("nan")
        r = dict(name=name, group=info.get("group"),
                 current_area=float(cur.mean()) if cur is not None else None,
                 hybrid_area=float(hyb.mean()),
                 agreement_iou=agree)
        rows.append(r)

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
        overlay(axes[0], img, None, None, "original")
        overlay(axes[1], img, cur, CUR_COL,
                "current pipeline" + (f"  —  {cur.mean()*100:.2f}% area" if cur is not None else "  —  (no mask)"))
        overlay(axes[2], img, hyb, HYB_COL,
                f"SAM+17 hybrid  —  {hyb.mean()*100:.2f}% area")
        fig.suptitle(f"{info.get('group','?')}   ·   {name[:64]}"
                     + (f"    agreement IoU {agree:.3f}" if np.isfinite(agree) else ""),
                     fontproperties=TITLE_FONT, fontsize=13)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"{name}_compare.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{k:2d}/{len(infos)}] {info.get('group','?')[:18]:18s} "
              f"cur {(cur.mean()*100 if cur is not None else float('nan')):5.2f}%  "
              f"hyb {hyb.mean()*100:5.2f}%  agree {agree:.3f}  {name[:34]}")
        if args.limit and len(rows) >= args.limit:
            break

    with open(os.path.join(OUT_DIR, "comparison.json"), "w") as f:
        json.dump(dict(current_dir=CURRENT_DIR, hybrid_dir=HYBRID_DIR, rows=rows), f, indent=2)

    if rows:
        print("\n--- where the two models DISAGREE most (look at these first) ---")
        ranked = sorted([r for r in rows if np.isfinite(r["agreement_iou"])],
                        key=lambda r: r["agreement_iou"])
        for r in ranked[:12]:
            print(f"  agree {r['agreement_iou']:.3f}  cur {r['current_area']*100:5.2f}%  "
                  f"hyb {r['hybrid_area']*100:5.2f}%  [{r['group'][:18]}] {r['name'][:38]}")
        import collections
        by = collections.defaultdict(list)
        for r in rows:
            by[r["group"]].append(r)
        print("\n--- median predicted area by group ---")
        print(f"{'group':26s} {'n':>3s} {'current':>9s} {'hybrid':>9s} {'agree':>7s}")
        for g in sorted(by):
            v = by[g]
            cm = [x["current_area"] for x in v if x["current_area"] is not None]
            print(f"{g:26s} {len(v):3d} "
                  f"{(np.median(cm)*100 if cm else float('nan')):8.2f}% "
                  f"{np.median([x['hybrid_area'] for x in v])*100:8.2f}% "
                  f"{np.nanmedian([x['agreement_iou'] for x in v]):7.3f}")
    print(f"\noverlays in {OUT_DIR}")


if __name__ == "__main__":
    main()
