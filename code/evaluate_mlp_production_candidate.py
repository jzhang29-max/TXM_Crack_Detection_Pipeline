"""
Tests whether the neural network's LOIO win (fig_h: MLP mean IoU 0.734 vs
HGB 0.692 -- bootstrap-only, no corrections, no post-processing) survives
contact with the ACTUAL production training recipe and the ACTUAL
production verification gate, not just the base-algorithm bake-off.

Three things that earlier test did NOT check, all of which matter for a
real "should this replace HistGradientBoosting" decision:

  1. Sample weighting. Production training needs per-pixel sample_weight
     for both class-balance (compute_sample_weight("balanced", ...)) and
     the correction_weight multiplier central to this project's dilution-
     regression fix (retrain_with_corrections.py's docstring). Verified
     directly against this installed sklearn version (1.7.2): unlike what
     an older sklearn version required, MLPClassifier.fit DOES genuinely
     accept and use sample_weight -- confirmed in its source, feeding into
     the actual loss computation, not silently ignored. So this candidate
     is trained with native sample_weight, identical treatment to the tree
     ensembles, no oversampling workaround needed.
  2. Full production training data: 100k/class/image bootstrap + up to
     30k/class/image corrections across all 12 images -- much bigger than
     the LOIO test's 30k/class/image-from-3-images, and the first real
     test of whether MLP training time stays practical for the automated
     retrain loop at real scale.
  3. Whether this candidate actually PASSES retrain_and_deploy.py's real
     5-check gate against current production -- reusing that gate's own
     check_accuracy/check_border_and_artifacts functions completely
     UNCHANGED (imported directly, not reimplemented), so this candidate
     clears (or doesn't) the exact same bar the HGB production model did.

Does NOT touch production. Saves the candidate to
models/pixel_mlp_candidate.joblib for inspection; actually deploying it
(if this comes back positive) is a separate, deliberate step -- this
script only ever reads models/pixel_hgb_final.joblib, never writes it.

Usage:
    python3 evaluate_mlp_production_candidate.py
"""
import json
import os
import sys
import time

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
import retrain_and_deploy as rd
import retrain_with_corrections as rc

PROJECT_DIR = rc.PROJECT_DIR
CANDIDATE_PATH = os.path.join(PROJECT_DIR, "models", "pixel_mlp_candidate.joblib")
REPORT_PATH = os.path.join(PROJECT_DIR, "results", "mlp_candidate_gate_report.json")
FIG_DIR = os.path.join(PROJECT_DIR, "benchmark_figures")

MLP_PARAMS = dict(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=300,
                   early_stopping=True, random_state=0)

AQUA, YELLOW, MUTED = "#1baf7a", "#eda100", "#898781"


def make_mlp_pipeline(**params):
    return Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(**params))])


def train_mlp_candidate(correction_weight=1.0):
    rng = np.random.RandomState(0)
    print("Loading bootstrapped Ilastik-derived samples...")
    X_boot, y_boot, w_boot = rc.load_bootstrap_samples(rng)
    print("Loading human correction samples...")
    X_corr, y_corr, w_corr = rc.load_correction_samples(correction_weight, rng)

    X = np.concatenate(X_boot + X_corr, axis=0)
    y = np.concatenate(y_boot + y_corr, axis=0)
    base_weight = np.concatenate(w_boot + w_corr, axis=0)
    class_weight = compute_sample_weight("balanced", y)
    sample_weight = class_weight * base_weight

    print(f"Training MLP candidate on {len(y)} pixels (native sample_weight, no workaround needed)...")
    t0 = time.time()
    clf = make_mlp_pipeline(**MLP_PARAMS)
    clf.fit(X, y, mlp__sample_weight=sample_weight)
    fit_s = time.time() - t0
    print(f"  fit time: {fit_s:.1f}s")

    joblib.dump(clf, CANDIDATE_PATH)
    print(f"Saved candidate (NOT deployed): {CANDIDATE_PATH}")
    return clf, fit_s, len(y)


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    candidate, fit_s, n_rows = train_mlp_candidate(correction_weight=1.0)
    production = joblib.load(pc.MODEL_PATH)

    print("\n=== Check 1: accuracy vs corrected ground truth (reusing retrain_and_deploy.py's own check, unchanged) ===")
    t0 = time.time()
    accuracy_check = rd.check_accuracy(candidate, production)
    accuracy_predict_s = time.time() - t0

    print("\n=== Checks 2-5: border, spontaneous artifacts, degenerate output, change-sanity (all 12 images) ===")
    t0 = time.time()
    other_checks = rd.check_border_and_artifacts(candidate, production)
    other_predict_s = time.time() - t0

    all_checks = [accuracy_check] + other_checks
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate": "MLPClassifier(64,32) + StandardScaler, native sample_weight, "
                       "same bootstrap(100k)+corrections(30k, weight=1.0) recipe as production HGB",
        "fit_seconds": fit_s,
        "n_training_pixels": n_rows,
        "accuracy_check_predict_seconds": accuracy_predict_s,
        "other_checks_predict_seconds": other_predict_s,
        "checks": all_checks,
        "all_passed": all(c["passed"] for c in all_checks),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== Would this candidate pass the ACTUAL production gate? ===")
    for c in all_checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}" + (f" -- {c['reason']}" if c["reason"] else ""))
    verdict = "WOULD DEPLOY" if report["all_passed"] else "WOULD NOT DEPLOY"
    print(f"\nOverall: {verdict} (production untouched either way -- this script never writes to {pc.MODEL_PATH})")
    print(f"Full report: {REPORT_PATH}")

    # ---- Fig L: per-image IoU, MLP candidate vs current production HGB,
    # using the SAME corrected-GT accuracy numbers as the gate check above
    # (not the earlier bootstrap-only LOIO numbers) ----
    per_image = accuracy_check["per_image"]
    names = [r["image"] for r in per_image]
    cand_ious = [r["candidate_iou"] for r in per_image]
    prod_ious = [r["production_iou"] for r in per_image]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(names))
    width = 0.32
    ax.bar(x - width / 2, prod_ious, width, label="Production (HistGradientBoosting)", color=AQUA,
           edgecolor="white", linewidth=0.6)
    ax.bar(x + width / 2, cand_ious, width, label="MLP candidate (neural network)", color=YELLOW,
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU vs. corrected ground truth (Ilastik + your paint corrections)")
    ax.set_title("(l) MLP candidate vs. current production, trained on the REAL production recipe")
    ax.legend(frameon=False, loc="upper right")
    savefig(fig, "fig_l_mlp_production_candidate.png")


if __name__ == "__main__":
    main()
