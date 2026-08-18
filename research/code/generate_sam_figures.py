"""
Figures for the SAM-vs-pixel-classifier comparison, in the same visual
language as final_figures/ so they drop straight into the existing deck.

Reads results/sam/*.json (written by sam_experiments.py) plus the held-out
baseline in results/sam/baseline_loio.json, and produces:

  sam_iou_comparison.png    mean IoU, deployable vs oracle-prompted, against
                            the deployed 17-feature model
  sam_precision_recall.png  precision-recall plane -- the figure that actually
                            explains the result, because SAM's two failure
                            modes sit in opposite corners and a single IoU
                            number hides that
  sam_per_image.png         per-image IoU: SAM's variance across images is a
                            finding in its own right
  sam_resolution.png        why the embedding route is capped: SAM's ViT emits
                            a 64x64 grid for a 1024px tile, i.e. one feature
                            vector per 16x16 pixel block, against measured
                            crack widths
  sam_qualitative.png       image / ground truth / SAM / deployed model

Usage:
    python3 generate_sam_figures.py
"""

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

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM_DIR = os.path.join(PROJECT_DIR, "results", "sam")
OUT_DIR = os.path.join(PROJECT_DIR, "sam_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- style: identical to generate_final_comparison_suite.py -------------
for path in ["/System/Library/Fonts/Supplemental/Georgia.ttf",
             "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"]:
    if os.path.exists(path):
        fm.fontManager.addfont(path)
_GB = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
TITLE_FONT = fm.FontProperties(fname=_GB) if os.path.exists(_GB) else None

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#898781", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.labelsize": 12.5, "legend.fontsize": 10.5, "font.size": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 220, "figure.dpi": 110,
})

# Deployed model keeps the green it has in every existing figure
# (HistGradientBoosting/MLP family). SAM gets one hue family so it always
# reads as "the other approach"; oracle variants are the same hue lightened,
# since they are the SAME model with an unfair advantage, not new models.
COL_OURS = "#1baf7a"
COL_SAM_DEPLOY = "#2a78d6"
COL_SAM_ORACLE = "#9cc4ee"
COL_FAIL = "#c3c2b7"

PRETTY = {
    "amg_whole": "SAM auto-masks\n(whole frame)",
    "amg_tiled": "SAM auto-masks\n(1024px tiles)",
    "grid_points": "SAM point grid\n(per tile)",
    "embed_lr": "SAM features\n+ logistic reg.",
    "embed_mlp": "SAM features\n+ MLP",
    "embed_plus17": "SAM features\n+ our 17 + MLP",
    "embed_mlp_plus17": "SAM features\n+ our 17 + MLP",
    "amg_relaxed": "SAM auto-masks\n(relaxed thresholds)",
    "pts_oracle_group": "SAM prompted ON crack\n(one multi-point group)",
    "amg_oracle": "SAM auto-masks\n+ perfect picker",
    "amg_relaxed_oracle": "SAM relaxed masks\n+ perfect picker",
    "pts_oracle": "SAM prompted ON\nthe true crack",
    "box_oracle": "SAM boxed ON\nthe true crack",
}


def set_title(ax, text):
    ax.set_title(text, fontproperties=TITLE_FONT, fontsize=15.5, pad=12)


def savefig(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def load():
    """Collect every SAM run plus the held-out baseline.

    Results live in three files by necessity, not by accident:
      huge_gray.json         first pass (AMG at HuggingFace's natural-image
                             confidence defaults, 16x16 prompt grids)
      huge_gray_fixed.json   AMG/prompt conditions re-run after an audit found
                             those defaults discard ~99.8% of proposals
      huge_gray_oracle32.json oracle prompts at the documented budget of 32
                             plus the multi-point-group variant
    Later files win per condition. The embedding conditions appear only in the
    first file and are unaffected -- run_embed_loio and its helpers were not
    touched by the fixes -- so carrying them forward is sound.
    """
    order = ["huge_gray.json", "huge_gray_fixed.json", "huge_gray_oracle32.json"]
    merged, meta = {}, None
    for fn in order:
        fp = os.path.join(SAM_DIR, fn)
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            d = json.load(f)
        meta = meta or {k: v for k, v in d.items() if k != "results"}
        for cond, r in d.get("results", {}).items():
            if r.get("mean_iou") is None and cond in merged:
                continue                 # never let a failed re-run erase a good row
            r["source"] = fn
            merged[cond] = r
    runs = {}
    if merged:
        runs["huge_gray"] = dict(**meta, results=merged)
    # any other run files (e.g. a sam2 or clahe pass) stay available separately
    for p in sorted(glob.glob(os.path.join(SAM_DIR, "*.json"))):
        stem = os.path.basename(p)[:-5]
        if stem.startswith("baseline") or os.path.basename(p) in order:
            continue
        with open(p) as f:
            runs[stem] = json.load(f)
    base_path = os.path.join(SAM_DIR, "baseline_loio.json")
    baseline = None
    if os.path.exists(base_path):
        with open(base_path) as f:
            baseline = json.load(f)
    return runs, baseline


def rows_of(run, cond):
    r = run.get("results", {}).get(cond)
    if not r or "rows" not in r:
        return []
    return [x for x in r["rows"] if np.isfinite(x.get("iou", float("nan")))]


def fig_iou(runs, baseline, primary):
    """Mean IoU: deployable SAM, oracle SAM, and the deployed model.

    Horizontal bars because ten conditions with two-line names cannot be read
    on a vertical axis, and a colour legend rather than a divider line because
    a divider put the deployed model on the 'not deployable' side of it.
    """
    run = runs[primary]
    res = run["results"]
    deploy = [c for c in PRETTY if c in res and res[c].get("deployable") and res[c].get("mean_iou") is not None]
    oracle = [c for c in PRETTY if c in res and res[c].get("deployable") is False and res[c].get("mean_iou") is not None]
    if not (deploy or oracle):
        print("  [skip] fig_iou: no conditions with results")
        return

    # Pixel-weighted means alongside the unweighted ones. Not optional: three
    # of the four ground-truth images are 2.9 MP and the fourth is 23.5 MP, so
    # the unweighted mean gives 73% of the labelled pixels one quarter of the
    # vote. The headline "SAM+17 beats ours by +0.05" holds only unweighted and
    # collapses to +0.001 weighted -- showing one number alone would be the
    # same mistake this project has already made twice.
    pw = {}
    sp = os.path.join(SAM_DIR, "paired_stats.json")
    if os.path.exists(sp):
        with open(sp) as f:
            ps = json.load(f)
        for c, v in ps["comparisons"].items():
            pw[c] = v["cond_mean_pixelweighted"]
        pw["__base__"] = next(iter(ps["comparisons"].values()))["base_mean_pixelweighted"]

    items = []
    if baseline:
        items.append(("Our 17 features + MLP", baseline["mean_iou"], pw.get("__base__"), COL_OURS))
    items += [(PRETTY[c].replace("\n", " "), res[c]["mean_iou"], pw.get(c), COL_SAM_DEPLOY) for c in deploy]
    items += [(PRETTY[c].replace("\n", " "), res[c]["mean_iou"], pw.get(c), COL_SAM_ORACLE) for c in oracle]
    items.sort(key=lambda t: t[1])

    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    wvals = [i[2] for i in items]
    cols = [i[3] for i in items]

    fig, ax = plt.subplots(figsize=(10.8, 0.62 * len(items) + 2.5))
    y = np.arange(len(items))
    h = 0.36
    ax.barh(y + h / 2, vals, color=cols, edgecolor="white", linewidth=1.1,
            height=h, zorder=3, label="per-image mean")
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi + h / 2, f"{v:.3f}", va="center", ha="left", fontsize=10, color=INK)
    have_w = [i for i, v in enumerate(wvals) if v is not None]
    if have_w:
        ax.barh([y[i] - h / 2 for i in have_w], [wvals[i] for i in have_w],
                color=[cols[i] for i in have_w], edgecolor="white", linewidth=1.1,
                height=h, zorder=3, alpha=0.45, hatch="///",
                label="pixel-weighted mean")
        for i in have_w:
            ax.text(wvals[i] + 0.008, y[i] - h / 2, f"{wvals[i]:.3f}", va="center",
                    ha="left", fontsize=10, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("Mean IoU vs ground truth (4 held-out images)")
    ax.set_xlim(0, max(max(vals) * 1.16, 0.88))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COL_OURS, label="our model — deployable"),
        Patch(facecolor=COL_SAM_DEPLOY, label="SAM — deployable"),
        Patch(facecolor=COL_SAM_ORACLE, label="SAM — handed the ground truth (NOT deployable)"),
    ]
    if have_w:
        handles.append(Patch(facecolor="#888780", alpha=0.45, hatch="///",
                             label="hatched = pixel-weighted (23.5 MP image gets its real weight)"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5)
    set_title(ax, f"Segment Anything vs the deployed pixel classifier  ({run['model'].split('/')[-1]})")
    savefig(fig, "sam_iou_comparison.png")


def fig_pr(runs, baseline, primary):
    """Precision-recall plane: SAM's two failure modes are in opposite corners."""
    res = runs[primary]["results"]
    fig, ax = plt.subplots(figsize=(10.6, 6.4))

    # Iso-IoU contours first, so points and the legend sit on top of them.
    # IoU = 1/(1/p + 1/r - 1)  =>  p = 1/(1/IoU + 1 - 1/r)
    for f in (0.25, 0.5, 0.75):
        rc = np.linspace(0.02, 1.0, 400)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = 1.0 / (1.0 / f + 1.0 - 1.0 / rc)
        ok = np.isfinite(p) & (p > 0) & (p <= 1.0)
        ax.plot(rc[ok], p[ok], color="#cfcec6", lw=1.1, zorder=1)
        if ok.any():
            ax.annotate(f"IoU {f}", (rc[ok][0], p[ok][0]), fontsize=9.5, color=MUTED,
                        textcoords="offset points", xytext=(6, -2), zorder=2)

    # Numbered markers + a legend: ten inline annotations collide no matter how
    # the offsets are tuned.
    handles, n = [], 0
    for cond in PRETTY:
        if cond not in res or res[cond].get("mean_iou") is None:
            continue
        rs = rows_of(runs[primary], cond)
        if not rs:
            continue
        n += 1
        pr = float(np.mean([r["precision"] for r in rs]))
        rc = float(np.mean([r["recall"] for r in rs]))
        dep = res[cond].get("deployable")
        col = COL_SAM_DEPLOY if dep else COL_SAM_ORACLE
        ax.scatter(rc, pr, s=340, color=col, edgecolor="white", linewidth=1.5, zorder=4)
        ax.text(rc, pr, str(n), ha="center", va="center", fontsize=9.5,
                color="white", zorder=5)
        handles.append(plt.Line2D([], [], marker="o", ls="", markersize=9, color=col,
                                  label=f"{n}. {PRETTY[cond].replace(chr(10), ' ')}"
                                        f"{'' if dep else '  (oracle)'}"))
    if baseline:
        bp = float(np.mean([r["precision"] for r in baseline["per_image"]]))
        ax.scatter(baseline["mean_recall"], bp, s=420, marker="*", color=COL_OURS,
                   edgecolor="white", linewidth=1.4, zorder=6)
        handles.append(plt.Line2D([], [], marker="*", ls="", markersize=15, color=COL_OURS,
                                  label="our 17 features + MLP"))

    ax.set_xlabel("Recall  (fraction of real crack found)")
    ax.set_ylabel("Precision  (fraction of marks that are real crack)")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=10)
    set_title(ax, "Precision and recall separate what one IoU number hides")
    savefig(fig, "sam_precision_recall.png")


def fig_per_image(runs, baseline, primary):
    """Per-image IoU. SAM's spread across images is itself the finding."""
    run = runs[primary]
    res = run["results"]
    conds = [c for c in PRETTY if c in res and res[c].get("mean_iou") is not None]
    stems = ["333_75_um_zoom", "336_25", "338_13", "LARGE_343_75"]
    if not conds:
        print("  [skip] fig_per_image")
        return
    fig, ax = plt.subplots(figsize=(max(9.5, 1.5 * len(conds)), 5.6))
    w = 0.8 / len(stems)
    for si, stem in enumerate(stems):
        vals, xs = [], []
        for ci, c in enumerate(conds):
            row = next((r for r in run["results"][c].get("rows", []) if r.get("image") == stem), None)
            v = row.get("iou") if row else None
            xs.append(ci + si * w - 0.4 + w / 2)
            vals.append(v if (v is not None and np.isfinite(v)) else 0.0)
        ax.bar(xs, vals, width=w * 0.92, label=stem,
               color=plt.cm.Blues(0.35 + 0.18 * si), edgecolor="white", linewidth=0.7, zorder=3)
    if baseline:
        ax.axhline(baseline["mean_iou"], color=COL_OURS, lw=2.2, ls="--", zorder=4)
        ax.text(len(conds) - 0.45, baseline["mean_iou"] + 0.015,
                f"our model, mean {baseline['mean_iou']:.3f}", ha="right",
                fontsize=10.5, color=COL_OURS)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([PRETTY[c] for c in conds], fontsize=9.5)
    ax.set_ylabel("IoU vs ground truth")
    ax.legend(title="held-out image", fontsize=9.5, title_fontsize=10, frameon=False, ncol=2)
    ax.set_axisbelow(True)
    set_title(ax, "Per-image IoU: SAM is inconsistent image to image")
    savefig(fig, "sam_per_image.png")


def fig_resolution():
    """Why SAM's features work HERE but fail in the crack literature.

    This figure originally argued the opposite -- that SAM's stride-16
    embedding was too coarse for the cracks -- with a hand-waved "~4 px" crack
    width. Measuring it killed that argument: the median crack width in these
    ground-truth masks is 65 px, four embedding cells across. The real story is
    a regime difference, and the published zero-shot failures are on hairline
    cracks this dataset does not contain at load.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    widths, per_image = [], []
    for stem in ["333_75_um_zoom", "336_25", "338_13", "LARGE_343_75"]:
        p = os.path.join(PROJECT_DIR, "dataset_cache", f"{stem}_gt.npy")
        if not os.path.exists(p):
            continue
        gt = np.load(p).astype(bool)
        w = 2.0 * ndi.distance_transform_edt(gt)[skeletonize(gt)]
        w = w[w > 0]
        widths.append(w)
        per_image.append((stem, float(np.median(w))))
    if not widths:
        print("  [skip] fig_resolution: no GT")
        return
    w = np.concatenate(widths)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))

    ax = axes[0]
    # Clip for readability, but say so: without a label the pile-up in the last
    # bin reads as a second mode at 300 px rather than "everything wider".
    over = float((w > 300).mean())
    ax.hist(np.clip(w, 0, 300), bins=60, color=COL_SAM_DEPLOY, edgecolor="white",
            linewidth=0.4, zorder=3)
    if over > 0.005:
        ax.annotate(f"last bin = all widths\n>300 px ({over*100:.0f}%)", (300, 0),
                    textcoords="offset points", xytext=(-8, 46), ha="right",
                    fontsize=9.5, color=MUTED,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.axvline(16, color="#d85a30", lw=2.0, zorder=4)
    ax.annotate("SAM embedding cell\n16 px", (16, ax.get_ylim()[1] * 0.82),
                textcoords="offset points", xytext=(12, 0), fontsize=10, color="#993c1d")
    ax.axvline(np.median(w), color=COL_OURS, lw=2.0, ls="--", zorder=4)
    ax.annotate(f"median crack\nwidth {np.median(w):.0f} px",
                (np.median(w), ax.get_ylim()[1] * 0.55),
                textcoords="offset points", xytext=(14, 0), fontsize=10, color="#0f6e56")
    ax.set_xlabel("crack width along the medial axis (px)")
    ax.set_ylabel("count")
    ax.set_axisbelow(True)
    set_title(ax, "These cracks are wide, not hairline")
    ax.text(0.98, 0.96, f"{(w < 16).mean()*100:.0f}% of the crack is narrower\n"
                        f"than one embedding cell",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=MUTED)

    ax = axes[1]
    labels = ["this dataset\n(median)", "SAM embedding\ncell", "hairline crack\n(literature)"]
    vals = [float(np.median(w)), 16.0, 3.0]
    cols = [COL_OURS, COL_SAM_DEPLOY, "#d85a30"]
    ax.bar(range(3), vals, color=cols, edgecolor="white", linewidth=1.2, width=0.62, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.4, f"{v:.0f} px", ha="center", fontsize=11.5, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("width (pixels)")
    ax.set_ylim(0, max(vals) * 1.24)
    ax.set_axisbelow(True)
    set_title(ax, "Why our result and the papers agree")
    fig.tight_layout()
    savefig(fig, "sam_resolution.png")


def fig_qualitative(stem="336_25"):
    """image / ground truth / SAM conditions / deployed model, same crop.

    Numbers persuade a reviewer; this panel is what persuades everyone else,
    because SAM's failure is obvious on sight and ambiguous in a table.
    """
    mask_dir = os.path.join(SAM_DIR, "masks")
    if not os.path.isdir(mask_dir):
        print("  [skip] fig_qualitative: no results/sam/masks/ (run with --save-masks)")
        return
    import glob as _g
    found = {}
    for p in sorted(_g.glob(os.path.join(mask_dir, f"*__{stem}.npy"))):
        cond = os.path.basename(p).split("__")[0]
        found[cond] = p
    if not found:
        print(f"  [skip] fig_qualitative: no masks for {stem}")
        return

    img_p = os.path.join(PROJECT_DIR, "dataset_cache", f"{stem}_img.npy")
    gt_p = os.path.join(PROJECT_DIR, "dataset_cache", f"{stem}_gt.npy")
    if not (os.path.exists(img_p) and os.path.exists(gt_p)):
        print("  [skip] fig_qualitative: image cache missing")
        return
    img = np.load(img_p)
    gt = np.load(gt_p).astype(bool)

    ours_p = os.path.join(SAM_DIR, "masks", f"ours__{stem}.npy")
    order = ["amg_whole", "amg_tiled", "amg_relaxed", "grid_points",
             "embed_mlp", "embed_mlp_plus17", "embed_plus17",
             "amg_oracle", "pts_oracle", "pts_oracle_group", "box_oracle"]
    conds = [c for c in order if c in found]
    panels = [("original", None), ("ground truth", gt)]
    if os.path.exists(ours_p):
        panels.append(("our model", np.load(ours_p).astype(bool)))
    panels += [(PRETTY.get(c, c).replace("\n", " "), np.load(found[c])) for c in conds]

    ncol = min(4, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.35 * nrow))
    axes = np.atleast_1d(axes).ravel()
    g8 = np.clip(img, 0, 1)
    for ax, (label, m) in zip(axes, panels):
        ax.imshow(g8, cmap="gray", vmin=0, vmax=1, aspect="equal")
        if m is not None:
            over = np.zeros((*g8.shape, 4), np.float32)
            over[m] = (0.85, 0.20, 0.10, 0.50)
            ax.imshow(over, aspect="equal")
            sc = metrics_of(m, gt)
            label = f"{label}\nIoU {sc['iou']:.3f}  recall {sc['recall']:.3f}"
        ax.set_title(label, fontsize=10.5, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Predicted crack on {stem}", fontproperties=TITLE_FONT, fontsize=15.5, y=0.995)
    fig.tight_layout()
    savefig(fig, "sam_qualitative.png")


def metrics_of(pred, gt):
    pred = np.asarray(pred, bool)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return dict(iou=inter / union if union else float("nan"),
                recall=inter / int(gt.sum()) if gt.sum() else float("nan"))


def main():
    runs, baseline = load()
    if not runs:
        print("no results/sam/*.json yet -- run sam_experiments.py first")
        return
    primary = ("huge_gray" if "huge_gray" in runs else sorted(runs)[0])
    print(f"primary run: {primary}  ({len(runs)} run file(s), baseline={'yes' if baseline else 'MISSING'})")
    fig_iou(runs, baseline, primary)
    fig_pr(runs, baseline, primary)
    fig_per_image(runs, baseline, primary)
    fig_resolution()
    fig_qualitative()
    print(f"\nfigures in {OUT_DIR}")


if __name__ == "__main__":
    main()
