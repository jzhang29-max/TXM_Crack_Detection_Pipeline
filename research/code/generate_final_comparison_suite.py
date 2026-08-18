"""
Final, unified model-comparison suite. Fixes two real problems with the
incremental figures in benchmark_figures/ (fig_a...fig_m), produced as
this project's comparisons were requested one at a time:

  1. The neural network only ever appeared in ONE bolted-on bar chart
     (fig_h) instead of every multi-model comparison. This script reruns
     the full leave-one-image-out protocol for all 4 real architectures
     (RandomForest, ExtraTrees, HistGradientBoosting, MLP) together, so
     every chart type -- bar comparison, parity plot, ROC, confusion
     matrix, learning curve -- treats all 4 symmetrically.
  2. Color meaning drifted across the old figures: yellow meant "logistic
     regression" in fig_j and "MLP" in fig_h -- the same hex meaning two
     different things in different charts is a real inconsistency, not
     just taste. This script fixes ONE global color assignment (below)
     used everywhere: the 4 real candidate models get the 4 vivid
     categorical colors; the interpretability baselines that are NOT full
     models (Otsu threshold, logistic regression, decision tree) get a
     light -> dark GRAY ramp instead, which also visually de-emphasizes
     them relative to the real candidates -- a deliberate, common
     technique in ML papers, not an accident.

Also drops the old "(a)/(b)/(c)..." title-lettering: that convention only
makes sense for sub-panels of ONE merged multi-panel figure, and every
image here is delivered as its own standalone file, so it gets a clean
title instead and the letter/caption belongs in the surrounding prose.

Outputs to final_figures/ (not benchmark_figures/, so the older
incremental figures remain as an honest record of the development
process rather than being silently overwritten).

Usage:
    python3 generate_final_comparison_suite.py [--quick]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_benchmark_report import IMAGES, metrics_from_pred, run_loio, sample_pixels
from txm_features import FEATURE_NAMES
from apply_pixel_model import postprocess_mask

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD_FIG_DIR = os.path.join(PROJECT_DIR, "benchmark_figures")
OUT_DIR = os.path.join(PROJECT_DIR, "final_figures")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OLD_FIG_DIR, "extended_summary.json")) as f:
    PRIOR_EXTENDED = json.load(f)

# ----------------------------------------------------------------- style
# Titles in a serif face (Georgia, matching this project's diagram style)
# so they read as a print-ready caption, not a plot label; everything
# else in the system sans, larger than matplotlib's tiny defaults.
for path in ["/System/Library/Fonts/Supplemental/Georgia.ttf",
             "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"]:
    if os.path.exists(path):
        fm.fontManager.addfont(path)
TITLE_FONT = fm.FontProperties(fname="/System/Library/Fonts/Supplemental/Georgia Bold.ttf") \
    if os.path.exists("/System/Library/Fonts/Supplemental/Georgia Bold.ttf") else None

INK, MUTED = "#0b0b0b", "#52514e"
GRID = "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#898781", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.labelsize": 12.5, "legend.fontsize": 10.5, "font.size": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 220, "figure.dpi": 110,
})


def set_title(ax, text):
    ax.set_title(text, fontproperties=TITLE_FONT, fontsize=15.5, pad=12)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------- ONE color mapping
# Vivid categorical colors: the 4 real candidate architectures, used
# identically in every figure below. Grays: the 3 interpretability
# baselines, which are not full competing models, ramped light->dark by
# increasing sophistication (Otsu has no fitting at all; the tree is the
# most flexible of the three) so they read as "baseline family", visually
# distinct from the 4 real candidates.
MODEL_COLORS = {
    "RandomForest": "#2a78d6",
    "ExtraTrees": "#eb6834",
    "HistGradientBoosting": "#1baf7a",
    "MLP": "#eda100",
}
TIER_GRAYS = {
    "Otsu threshold": "#c3c2b7",
    "Logistic regression": "#8a887f",
    "Decision tree": "#52514e",
}
MODEL_ORDER = ["RandomForest", "ExtraTrees", "HistGradientBoosting", "MLP"]

MODEL_SPECS = {
    "RandomForest": (RandomForestClassifier, dict(n_estimators=300, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=0)),
    "ExtraTrees": (ExtraTreesClassifier, dict(n_estimators=400, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=0)),
    "HistGradientBoosting": (HistGradientBoostingClassifier, dict(max_iter=300, max_depth=8, learning_rate=0.1, class_weight="balanced", random_state=0)),
    "MLP": (lambda **p: Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(**p))]),
             dict(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=300, early_stopping=True, random_state=0)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_per_class = 300 if args.quick else 30000

    confmat = {m: np.zeros((2, 2), dtype=np.int64) for m in MODEL_ORDER}
    parity_rows = []
    pooled = {}
    means, stds, fi_by_model = {}, {}, {}

    def make_on_fold(name):
        def _on_fold(image_name, proba_2d, gt_2d):
            final_mask = postprocess_mask(proba_2d)
            parity_rows.append(dict(model=name, image=image_name,
                                     actual_fraction=float(gt_2d.mean()), predicted_fraction=float(final_mask.mean())))
            gtf, pf = gt_2d.reshape(-1), final_mask.reshape(-1)
            tp = int(np.logical_and(pf, gtf).sum()); tn = int(np.logical_and(~pf, ~gtf).sum())
            fp = int(np.logical_and(pf, ~gtf).sum()); fn = int(np.logical_and(~pf, gtf).sum())
            confmat[name] += np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        return _on_fold

    for name in MODEL_ORDER:
        cls, params = MODEL_SPECS[name]
        print(f"\n=== {name} (LOIO, n_per_class={n_per_class}) ===")
        folds, probs, trues, fi = run_loio(cls, params, n_per_class, want_fi=True, on_fold=make_on_fold(name))
        means[name] = {k: float(np.mean([f[k] for f in folds])) for k in ("iou", "dice", "precision", "recall")}
        stds[name] = {k: float(np.std([f[k] for f in folds])) for k in ("iou", "dice", "precision", "recall")}
        pooled[name] = (probs, trues)
        fi_by_model[name] = fi
        print(f"  mean IoU={means[name]['iou']:.4f}")

    # ------------------------------------------------ Fig 1: bar chart
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    metric_keys, metric_labels = ["iou", "dice", "precision", "recall"], ["IoU", "Dice", "Precision", "Recall"]
    x = np.arange(len(metric_keys))
    width = 0.19
    for k, name in enumerate(MODEL_ORDER):
        vals = [means[name][m] for m in metric_keys]
        errs = [stds[name][m] for m in metric_keys]
        bars = ax.bar(x + (k - 1.5) * width, vals, width, yerr=errs, capsize=3, label=name,
                       color=MODEL_COLORS[name], edgecolor="white", linewidth=0.6)
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=7.3, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (mean over 4 leave-one-image-out folds)")
    set_title(ax, "Model Comparison Across Segmentation Metrics")
    ax.legend(frameon=False, loc="lower right", ncol=2)
    savefig(fig, "fig1_model_comparison.png")

    # ------------------------------------------------ Fig 2: parity plot
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    lims = [0, 0.4]
    ax.plot(lims, lims, "--", color=MUTED, linewidth=1.2, label="perfect prediction (y = x)")
    markers = {img["name"]: m for img, m in zip(IMAGES, ["o", "s", "^", "D"])}
    for row in parity_rows:
        ax.scatter(row["actual_fraction"], row["predicted_fraction"], s=85, color=MODEL_COLORS[row["model"]],
                   marker=markers[row["image"]], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual crack area fraction (ground truth)")
    ax.set_ylabel("Predicted crack area fraction (final mask)")
    set_title(ax, "Predicted vs. Actual Crack Coverage")
    model_handles = [Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[n], label=n, markersize=9) for n in MODEL_ORDER]
    image_handles = [Line2D([0], [0], marker=markers[i["name"]], linestyle="", color=MUTED, label=i["name"], markersize=9) for i in IMAGES]
    leg1 = ax.legend(handles=model_handles, loc="upper left", frameon=False, title="Model")
    ax.add_artist(leg1)
    ax.legend(handles=image_handles, loc="lower right", frameon=False, title="Image", fontsize=9)
    savefig(fig, "fig2_area_fraction_parity.png")

    # ------------------------------------------------ Fig 3: ROC
    # figsize wider than tall -- at a square 6.4x6.4 the bold serif title
    # ran past the right edge of the canvas and got clipped.
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.1, label="chance (AUC = 0.50)")
    roc_summary = {}
    for name in MODEL_ORDER:
        probs, trues = pooled[name]
        fpr, tpr, _ = roc_curve(trues, probs)
        a = float(auc(fpr, tpr))
        roc_summary[name] = a
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2.3, label=f"{name} (AUC = {a:.3f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    set_title(ax, "ROC Curves, Pooled Across Held-Out Images")
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, "fig3_roc_curves.png")

    # ------------------------------------------------ Fig 4: confusion matrices
    fig, axes = plt.subplots(1, 4, figsize=(4.4 * 4, 4.6), sharey=True)
    fig.subplots_adjust(wspace=0.15)
    conf_summary = {}
    for k, (ax, name) in enumerate(zip(axes, MODEL_ORDER)):
        cmap = LinearSegmentedColormap.from_list("ramp", ["#fcfcfb", MODEL_COLORS[name]])
        cm = confmat[name]; cm_norm = cm / cm.sum(axis=1, keepdims=True)
        conf_summary[name] = cm.tolist()
        ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["background", "crack"])
        if k == 0:
            ax.set_yticklabels(["background", "crack"]); ax.set_ylabel("Ground truth")
        ax.set_xlabel("Predicted"); ax.grid(False)
        set_title(ax, name)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]:,}\n({cm_norm[r, c]*100:.1f}%)", ha="center", va="center",
                        color="white" if cm_norm[r, c] > 0.5 else INK, fontsize=9.5)
    fig.suptitle("Pixel-Level Confusion Matrices (Final Post-Processed Mask)", fontproperties=TITLE_FONT, fontsize=15.5, y=1.03)
    savefig(fig, "fig4_confusion_matrices.png")

    # ------------------------------------------------ Fig 5: feature importance (RF)
    rf_fi = fi_by_model["RandomForest"]
    order = np.argsort(rf_fi)[::-1]
    groups = []
    for fname in FEATURE_NAMES:
        if fname == "intensity": groups.append("intensity")
        elif fname.startswith("smooth"): groups.append("smoothed intensity")
        elif fname.startswith("gradmag"): groups.append("gradient magnitude")
        elif fname.startswith("laplacian"): groups.append("Laplacian")
        else: groups.append("texture (local std)")
    group_color = {"intensity": "#4a3aa7", "smoothed intensity": MODEL_COLORS["RandomForest"],
                    "gradient magnitude": MODEL_COLORS["ExtraTrees"], "Laplacian": MODEL_COLORS["HistGradientBoosting"],
                    "texture (local std)": MODEL_COLORS["MLP"]}
    fig, ax = plt.subplots(figsize=(7.3, 6.3))
    y_pos = np.arange(len(FEATURE_NAMES))
    ax.barh(y_pos, rf_fi[order], color=[group_color[groups[i]] for i in order], edgecolor="white", linewidth=0.6)
    ax.set_yticks(y_pos); ax.set_yticklabels([FEATURE_NAMES[i] for i in order]); ax.invert_yaxis()
    ax.set_xlabel("Relative importance (Random Forest, mean decrease in impurity)")
    set_title(ax, "Feature Importance Across All 17 Pixel Features")
    handles = [Line2D([0], [0], marker="s", linestyle="", color=c, label=g, markersize=10) for g, c in group_color.items()]
    ax.legend(handles=handles, frameon=False, loc="lower right", title="Feature group")
    savefig(fig, "fig5_feature_importance.png")

    # ------------------------------------------------ Fig 6: learning curve, HGB + MLP
    print("\n=== Learning curve: HistGradientBoosting + MLP vs. bootstrap sample size ===")
    sizes = [200, 400] if args.quick else [5000, 15000, 30000, 60000, 100000]
    lc = {"HistGradientBoosting": ([], []), "MLP": ([], [])}
    for name in ["HistGradientBoosting", "MLP"]:
        cls, params = MODEL_SPECS[name]
        for n in sizes:
            folds, _, _, _ = run_loio(cls, params, n)
            ious = [f["iou"] for f in folds]
            lc[name][0].append(float(np.mean(ious))); lc[name][1].append(float(np.std(ious)))
            print(f"  {name} n={n}: mean IoU={lc[name][0][-1]:.4f}")
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    for name in ["HistGradientBoosting", "MLP"]:
        m, s = lc[name]
        ax.errorbar(sizes, m, yerr=s, marker="o", markersize=7, color=MODEL_COLORS[name], linewidth=2, capsize=4, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Bootstrap sample size (pixels per class per training image)")
    ax.set_ylabel("Mean IoU (4-fold leave-one-image-out)")
    set_title(ax, "Accuracy vs. Training Sample Size")
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, "fig6_learning_curve.png")

    # ------------------------------------------------ Fig 7: interpretability tiers
    print("\n=== Interpretability tiers (reusing Otsu/LogReg/Tree numbers, HGB+MLP freshly computed above) ===")
    interp = PRIOR_EXTENDED["interpretability_tiers"]
    tier_labels = ["Otsu threshold\n(classical, no fitting)", "Logistic regression\n(identified equation, top-4)",
                    "Decision tree\n(depth 3, top-4)", "HistGradientBoosting\n(full ML, 17 features)",
                    "MLP\n(full ML, 17 features)"]
    tier_colors = [TIER_GRAYS["Otsu threshold"], TIER_GRAYS["Logistic regression"], TIER_GRAYS["Decision tree"],
                    MODEL_COLORS["HistGradientBoosting"], MODEL_COLORS["MLP"]]
    tier_means = [interp["otsu"]["mean"], interp["logistic_regression"]["mean"], interp["decision_tree"]["mean"],
                   means["HistGradientBoosting"], means["MLP"]]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(metric_keys)); width = 0.16
    for k, (label, color, m) in enumerate(zip(tier_labels, tier_colors, tier_means)):
        vals = [m[mm] for mm in metric_keys]
        ax.bar(x + (k - 2) * width, vals, width, label=label, color=color, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (mean over 4 images / LOIO folds)")
    set_title(ax, "Interpretability Tiers: Classical Rule -> Identified Equation -> Full ML")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    savefig(fig, "fig7_interpretability_tiers.png")

    # ------------------------------------------------ Fig 8: decision boundary, HGB vs MLP
    print("\n=== Decision boundary comparison: HGB vs MLP, top-2 features ===")
    top2_idx = np.argsort(rf_fi)[::-1][:2]
    f1_name, f2_name = FEATURE_NAMES[top2_idx[0]], FEATURE_NAMES[top2_idx[1]]
    rng = np.random.RandomState(0)
    X2_list, y2_list = [], []
    for img in IMAGES:
        feats = np.load(img["feat_path"]); gt = np.load(img["gt_path"])
        X, y = sample_pixels(feats, gt, 15000, rng)
        X2_list.append(X[:, top2_idx]); y2_list.append(y)
        del feats, gt
    X2, y2 = np.concatenate(X2_list), np.concatenate(y2_list)
    x_min, x_max = np.percentile(X2[:, 0], [0.5, 99.5]); y_min, y_max = np.percentile(X2[:, 1], [0.5, 99.5])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    diverging = LinearSegmentedColormap.from_list("bgr", ["#2a78d6", "#f0efec", "#e34948"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    for ax, name in zip(axes, ["HistGradientBoosting", "MLP"]):
        cls, params = MODEL_SPECS[name]
        clf2 = cls(**params); clf2.fit(X2, y2)
        grid_proba = clf2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        cf = ax.contourf(xx, yy, grid_proba, levels=np.linspace(0, 1, 21), cmap=diverging, alpha=0.85)
        ax.contour(xx, yy, grid_proba, levels=[0.5], colors=INK, linewidths=1.4, linestyles="--")
        plot_idx = rng.choice(len(y2), size=min(3000, len(y2)), replace=False)
        ax.scatter(X2[plot_idx, 0][~y2[plot_idx]], X2[plot_idx, 1][~y2[plot_idx]], s=7, color="#2a78d6", alpha=0.4, edgecolor="none")
        ax.scatter(X2[plot_idx, 0][y2[plot_idx]], X2[plot_idx, 1][y2[plot_idx]], s=7, color="#e34948", alpha=0.4, edgecolor="none")
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_xlabel(f1_name); ax.set_ylabel(f2_name)
        set_title(ax, name)
        fig.colorbar(cf, ax=ax, label="P(crack)", shrink=0.85)
    handles = [Line2D([0], [0], marker="o", linestyle="", color="#2a78d6", label="background (true)", markersize=9),
               Line2D([0], [0], marker="o", linestyle="", color="#e34948", label="crack (true)", markersize=9)]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Decision Regions: Tree Ensemble vs. Neural Network", fontproperties=TITLE_FONT, fontsize=15.5, y=1.14)
    savefig(fig, "fig8_decision_boundary_comparison.png")

    # ------------------------------------------------------------- summary
    summary = {
        "models": {n: dict(mean=means[n], std=stds[n], roc_auc=roc_summary[n]) for n in MODEL_ORDER},
        "confusion_matrix": conf_summary,
        "learning_curve": {n: dict(sizes=sizes, mean=lc[n][0], std=lc[n][1]) for n in ["HistGradientBoosting", "MLP"]},
        "interpretability_tiers": interp,
        "decision_boundary_features": [f1_name, f2_name],
        "color_mapping": {**MODEL_COLORS, **TIER_GRAYS},
    }
    with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {os.path.join(OUT_DIR, 'final_summary.json')}")


if __name__ == "__main__":
    main()
