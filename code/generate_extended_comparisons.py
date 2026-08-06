"""
Two extensions to the MeltpoolNet-style benchmark report (generate_benchmark_report.py):

1. NEURAL NETWORK COMPARISON. The paper included a neural network in its
   model bake-off; we'd previously only tested an MLP under a DIFFERENT,
   less rigorous protocol (strategy_search/mlp_summary.json -- pooled
   bootstrap+corrections, not leave-one-image-out). This adds an
   MLPClassifier under the EXACT SAME leave-one-image-out protocol as
   fig_a_model_comparison.png, so its numbers are directly, fairly
   comparable to RandomForest/ExtraTrees/HistGradientBoosting for the
   first time.

2. INTERPRETABILITY TIERS. The paper's "model identification" study
   (Table 6-8) compares a classical physics equation (Rosenthal) vs. a
   data-fit equation vs. the best ML model, trading accuracy for
   interpretability. There's no literal physics equation for "is this
   pixel a crack," so this substitutes the closest faithful equivalents:
     - classical / no-fitting baseline: Otsu intensity threshold -- this
       is literally the method this project's very first approach used
       and rejected (see txm_features.py's docstring), not a strawman.
     - identified / interpretable-but-fit model: logistic regression over
       the top-4 RF-ranked features (fig_e) -- an actual quotable formula.
     - bonus interpretable ruleset: a depth-3 decision tree over the same
       top-4 features, rendered as an actual diagram.
     - full black-box ML: production HistGradientBoosting (17 features,
       reused from the cached benchmark, not refit).

Reuses generate_benchmark_report.py's LOIO infrastructure directly
(imported as a module) so every protocol detail (sampling, seed,
metrics) is identical to fig_a-g -- nothing here is a new, uncalibrated
comparison.

Usage:
    python3 generate_extended_comparisons.py
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_benchmark_report import (
    AQUA, BLUE, IMAGES, INK, MUTED, ORANGE, OUT_DIR, RED, VIOLET, YELLOW,
    metrics_from_pred, run_loio, sample_pixels,
)
from txm_features import FEATURE_NAMES

with open(os.path.join(OUT_DIR, "benchmark_summary.json")) as f:
    PRIOR = json.load(f)

MAGENTA = "#e87ba4"
N_PER_CLASS = 30000


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ============================================================ part 1: NN
def make_mlp(**params):
    return Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(**params))])


def part1_neural_network():
    print("\n=== Neural network (MLPClassifier), SAME LOIO protocol as fig_a ===")
    mlp_params = dict(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=300,
                       early_stopping=True, random_state=0)
    t0 = time.time()
    folds, probs, trues, _ = run_loio(make_mlp, mlp_params, N_PER_CLASS)
    elapsed = time.time() - t0
    mean_m = {k: float(np.mean([f[k] for f in folds])) for k in ("iou", "dice", "precision", "recall")}
    std_m = {k: float(np.std([f[k] for f in folds])) for k in ("iou", "dice", "precision", "recall")}
    fpr, tpr, _ = roc_curve(trues, probs)
    roc_auc = float(auc(fpr, tpr))
    print(f"  mean IoU={mean_m['iou']:.4f}  ROC AUC={roc_auc:.4f}  (total wall time {elapsed:.0f}s)")

    all_models = ["RandomForest", "ExtraTrees", "HistGradientBoosting", "MLP (neural network)"]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    metric_keys = ["iou", "dice", "precision", "recall"]
    means_by_model = [PRIOR["models"][n]["mean"] for n in all_models[:3]] + [mean_m]
    stds_by_model = [PRIOR["models"][n]["std"] for n in all_models[:3]] + [std_m]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metric_keys))
    width = 0.19
    for k, name in enumerate(all_models):
        vals = [means_by_model[k][m] for m in metric_keys]
        errs = [stds_by_model[k][m] for m in metric_keys]
        ax.bar(x + (k - 1.5) * width, vals, width, yerr=errs, capsize=3, label=name,
               color=colors[k], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["IoU", "Dice", "Precision", "Recall"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (mean over 4 leave-one-image-out folds)")
    ax.set_title("(h) Adding a neural network to the base-algorithm comparison")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    savefig(fig, "fig_h_neural_network_comparison.png")

    return dict(folds=folds, mean=mean_m, std=std_m, roc_auc=roc_auc,
                elapsed_seconds=elapsed, params=mlp_params)


# ================================================ part 2: interpretability
def make_top_k_model(base_cls, feature_idx):
    """Wraps an sklearn classifier so it only ever sees `feature_idx`
    columns -- lets run_loio's full-17-feature sampling/pooling machinery
    be reused unchanged for a reduced-feature interpretable model."""
    class TopKWrapped:
        def __init__(self, **params):
            self.clf = base_cls(**params)

        def fit(self, X, y):
            self.clf.fit(X[:, feature_idx], y)
            return self

        def predict_proba(self, X):
            return self.clf.predict_proba(X[:, feature_idx])
    return TopKWrapped


def part2_interpretability():
    print("\n=== Interpretability tiers: Otsu threshold -> identified equation -> full ML ===")
    rf_fi = PRIOR["feature_importance_random_forest"]
    top4_names = [n for n, _ in sorted(rf_fi.items(), key=lambda x: -x[1])[:4]]
    top4_idx = [FEATURE_NAMES.index(n) for n in top4_names]
    print(f"  top-4 features by RF importance: {top4_names}")

    # ---- Tier 1: Otsu threshold. No fitting, no held-out split needed --
    # this is literally the classical method this project's very first
    # approach used (Otsu on a flattened/normalized image) and rejected
    # for insufficient coverage; evaluated directly on all 4 images.
    otsu_results = []
    for img in IMAGES:
        img01 = np.load(img["img_path"])
        gt = np.load(img["gt_path"])
        t = threshold_otsu(img01)
        pred = img01 < t  # crack = dark
        m = metrics_from_pred(pred, gt)
        m.update(image=img["name"], otsu_threshold=float(t))
        otsu_results.append(m)
        print(f"    [Otsu] {img['name']}: IoU={m['iou']:.4f} threshold={t:.4f}")
    otsu_mean = {k: float(np.mean([r[k] for r in otsu_results])) for k in ("iou", "dice", "precision", "recall")}

    # ---- Tier 2: logistic regression "identified equation", top-4 features,
    # SAME LOIO protocol as every other tier for a fair comparison ----
    print("  Logistic regression (top-4 features), LOIO...")
    LogRegTop4 = make_top_k_model(LogisticRegression, top4_idx)
    logreg_folds, _, _, _ = run_loio(LogRegTop4, dict(max_iter=2000, class_weight="balanced", random_state=0),
                                       N_PER_CLASS)
    logreg_mean = {k: float(np.mean([f[k] for f in logreg_folds])) for k in ("iou", "dice", "precision", "recall")}

    # ---- Tier 2b (bonus): shallow decision tree, same top-4 features ----
    # depth 3, not 4 -- at depth 4 the diagram's 16 leaves overlapped
    # illegibly in a figure this size, and a tree that deep arguably isn't
    # "human-readable" anymore anyway, which is the whole point of this tier.
    print("  Decision tree (depth 3, top-4 features), LOIO...")
    TreeTop4 = make_top_k_model(DecisionTreeClassifier, top4_idx)
    tree_folds, _, _, _ = run_loio(TreeTop4, dict(max_depth=3, class_weight="balanced", random_state=0),
                                     N_PER_CLASS)
    tree_mean = {k: float(np.mean([f[k] for f in tree_folds])) for k in ("iou", "dice", "precision", "recall")}

    # ---- Final pooled fit (all 4 images) for the quotable headline
    # equation and tree diagram -- distinct from the LOIO folds above,
    # which exist purely to get a fair held-out accuracy number ----
    rng = np.random.RandomState(0)
    X_list, y_list = [], []
    for img in IMAGES:
        feats = np.load(img["feat_path"])
        gt = np.load(img["gt_path"])
        X, y = sample_pixels(feats, gt, N_PER_CLASS, rng)
        X_list.append(X[:, top4_idx])
        y_list.append(y)
        del feats, gt
    X_all, y_all = np.concatenate(X_list), np.concatenate(y_list)

    final_logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    final_logreg.fit(X_all, y_all)
    terms = " ".join(f"{c:+.3f}×{n}" for c, n in zip(final_logreg.coef_[0], top4_names))
    equation_str = f"logit(P(crack)) = {final_logreg.intercept_[0]:+.3f} {terms}"
    print(f"  Identified equation: {equation_str}")

    final_tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    final_tree.fit(X_all, y_all)

    # ---- Fig j: 3-tier (+bonus) accuracy bar chart ----
    tiers = ["Otsu threshold\n(classical, no fitting)",
             "Logistic regression\n(identified equation, top-4)",
             "Decision tree\n(depth 3, top-4)",
             "HistGradientBoosting\n(full ML, 17 features)"]
    tier_colors = [VIOLET, YELLOW, MAGENTA, AQUA]
    tier_means = [otsu_mean, logreg_mean, tree_mean, PRIOR["models"]["HistGradientBoosting"]["mean"]]
    metric_keys = ["iou", "dice", "precision", "recall"]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(metric_keys))
    width = 0.19
    for k, (label, color, means) in enumerate(zip(tiers, tier_colors, tier_means)):
        vals = [means[m] for m in metric_keys]
        ax.bar(x + (k - 1.5) * width, vals, width, label=label, color=color, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["IoU", "Dice", "Precision", "Recall"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (mean over 4 images / LOIO folds)")
    ax.set_title("(j) Interpretability tiers: classical rule → identified equation → full ML")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    savefig(fig, "fig_j_interpretability_tiers.png")

    # ---- Fig k: the decision tree, as an actual diagram ----
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(final_tree, feature_names=top4_names, class_names=["background", "crack"],
              filled=True, rounded=True, fontsize=10, ax=ax, impurity=False)
    ax.set_title("(k) Explicit interpretable ruleset: depth-3 decision tree over top-4 features")
    savefig(fig, "fig_k_decision_tree.png")

    return dict(
        top4_features=top4_names,
        otsu=dict(per_image=otsu_results, mean=otsu_mean),
        logistic_regression=dict(folds=logreg_folds, mean=logreg_mean, equation=equation_str,
                                    coefficients={n: float(c) for n, c in zip(top4_names, final_logreg.coef_[0])},
                                    intercept=float(final_logreg.intercept_[0])),
        decision_tree=dict(folds=tree_folds, mean=tree_mean),
        full_ml_reused_from="HistGradientBoosting (see benchmark_summary.json)",
    )


def main():
    nn_result = part1_neural_network()
    interp_result = part2_interpretability()

    extended = {
        "note": "Extends benchmark_summary.json -- see generate_extended_comparisons.py docstring.",
        "neural_network": nn_result,
        "interpretability_tiers": interp_result,
    }
    json_path = os.path.join(OUT_DIR, "extended_summary.json")
    with open(json_path, "w") as f:
        json.dump(extended, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    md_lines = ["# Extended comparisons -- neural network + interpretability tiers", "",
                "## Neural network, same LOIO protocol as fig_a", "",
                "| Model | IoU | Dice | Precision | Recall | ROC AUC |",
                "|---|---|---|---|---|---|"]
    for name in ["RandomForest", "ExtraTrees", "HistGradientBoosting"]:
        m = PRIOR["models"][name]["mean"]
        md_lines.append(f"| {name} | {m['iou']:.4f} | {m['dice']:.4f} | {m['precision']:.4f} | "
                         f"{m['recall']:.4f} | {PRIOR['roc_auc'][name]:.4f} |")
    m = nn_result["mean"]
    md_lines.append(f"| MLP (neural network) | {m['iou']:.4f} | {m['dice']:.4f} | {m['precision']:.4f} | "
                     f"{m['recall']:.4f} | {nn_result['roc_auc']:.4f} |")
    md_lines += ["", f"Neural network fit+predict wall time: {nn_result['elapsed_seconds']:.0f}s "
                       f"(4 folds, hidden_layer_sizes={nn_result['params']['hidden_layer_sizes']}).", "",
                 "## Interpretability tiers", "",
                 "| Tier | IoU | Dice | Precision | Recall |",
                 "|---|---|---|---|---|",
                 f"| Otsu threshold (classical) | {interp_result['otsu']['mean']['iou']:.4f} | "
                 f"{interp_result['otsu']['mean']['dice']:.4f} | {interp_result['otsu']['mean']['precision']:.4f} | "
                 f"{interp_result['otsu']['mean']['recall']:.4f} |",
                 f"| Logistic regression (identified equation) | {interp_result['logistic_regression']['mean']['iou']:.4f} | "
                 f"{interp_result['logistic_regression']['mean']['dice']:.4f} | "
                 f"{interp_result['logistic_regression']['mean']['precision']:.4f} | "
                 f"{interp_result['logistic_regression']['mean']['recall']:.4f} |",
                 f"| Decision tree (depth 3) | {interp_result['decision_tree']['mean']['iou']:.4f} | "
                 f"{interp_result['decision_tree']['mean']['dice']:.4f} | "
                 f"{interp_result['decision_tree']['mean']['precision']:.4f} | "
                 f"{interp_result['decision_tree']['mean']['recall']:.4f} |",
                 f"| HistGradientBoosting (full ML) | {PRIOR['models']['HistGradientBoosting']['mean']['iou']:.4f} | "
                 f"{PRIOR['models']['HistGradientBoosting']['mean']['dice']:.4f} | "
                 f"{PRIOR['models']['HistGradientBoosting']['mean']['precision']:.4f} | "
                 f"{PRIOR['models']['HistGradientBoosting']['mean']['recall']:.4f} |",
                 "", f"**Identified equation**: `{interp_result['logistic_regression']['equation']}`", ""]
    md_path = os.path.join(OUT_DIR, "extended_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
