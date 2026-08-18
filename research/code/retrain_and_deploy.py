"""
Fully automated retrain -> verify -> deploy pipeline. No human judgment
required to run this -- every check below encodes a REAL regression this
project actually hit during manual review, turned into an objective,
automatic pass/fail test. If everything passes, the candidate model is
copied straight into production (models/pixel_hgb_final.joblib), and the
paint tool picks it up immediately on its own (paint_common.py auto-detects
the file change and invalidates its cache -- no restart needed). If
anything fails, NOTHING is deployed, and a clear report explains exactly
what regressed and by how much.

The checks, and which real incident each one guards against:
  1. ACCURACY: candidate's mean IoU vs "corrected ground truth" (Ilastik +
     your paint corrections merged) must not drop more than IOU_TOLERANCE
     below current production's IoU on the same images. Guards against the
     dilution regression (correction_weight=5.0 pooled across 12 images
     dropped mean IoU from ~0.77 to ~0.64).
  2. BORDER/EDGE ARTIFACT: compares candidate vs production's crack-pixel
     density in bands near every image edge. Guards against the vignetting-
     flood regression (boosting bootstrap volume reintroduced a dense
     false-positive band across LARGE's top edge).
  3. SPONTANEOUS ARTIFACTS: labels candidate vs production masks and flags
     if candidate has a suspiciously large amount of area in components
     with ZERO overlap with production's mask. Guards against the
     hysteresis regression (22 brand-new artifact blobs from nothing on
     one test image).
  4. DEGENERATE OUTPUT: rejects a candidate that produces 0 regions, or an
     absurd area fraction, on any image -- a basic sanity floor under
     everything else.
  5. DID-ANYTHING-CHANGE (report-only, never blocks): confirms the
     candidate's predictions actually differ from production on your
     corrected pixels. A retrain that changes nothing didn't fail any
     check, but also didn't accomplish anything -- worth knowing.

Usage:
    python3 retrain_and_deploy.py [--correction-weight 1.0] [--force]

--force deploys the candidate regardless of gate results (for a case
you've reviewed yourself and want to push through anyway). Without it, a
failing gate leaves the candidate at models/pixel_model_candidate.joblib and
changes nothing in production.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import joblib
import numpy as np
from skimage.measure import label
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrain_with_corrections as rc
import paint_common as pc
from apply_pixel_model import postprocess_mask
from txm_features import compute_feature_stack, robust_normalize

import tifffile

PROJECT_DIR = rc.PROJECT_DIR
DATASET_CACHE_DIR = rc.DATASET_CACHE_DIR
CANDIDATE_PATH = os.path.join(PROJECT_DIR, "models", "pixel_model_candidate.joblib")
REPORT_PATH = os.path.join(PROJECT_DIR, "results", "deploy_gate_report.json")

# Thresholds -- see module docstring for what real incident each one guards.
IOU_TOLERANCE = 0.03
BORDER_BANDS_PX = [(100, 150), (150, 250), (250, 400)]
BORDER_DENSITY_RATIO_MAX = 2.0
BORDER_DENSITY_ABS_MIN = 0.03
NEW_ARTIFACT_AREA_FRACTION_MAX = 0.02
MIN_AREA_FRACTION = 0.005
MAX_AREA_FRACTION = 0.80


def train_candidate(correction_weight):
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

    print(f"Training on {len(y)} pixels...")
    t0 = time.time()
    # Delegates to retrain_with_corrections.build_classifier() rather than
    # constructing a classifier here -- that module is the single source of
    # truth for "what architecture is current champion" (see its
    # module-level comment for why: this exact duplication used to be a
    # real risk of the automated retrain loop silently drifting back to a
    # superseded architecture).
    clf = rc.build_classifier()
    rc.fit_with_sample_weight(clf, X, y, sample_weight)
    print(f"  fit time: {time.time() - t0:.1f}s")

    joblib.dump(clf, CANDIDATE_PATH)
    print(f"Saved candidate: {CANDIDATE_PATH}")
    return clf


def load_gt_manifest():
    manifest_path = os.path.join(DATASET_CACHE_DIR, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest["images"]


def build_corrected_gt(gt_path, raw_source):
    gt = np.load(gt_path)
    name = os.path.splitext(os.path.basename(raw_source))[0]
    corr_path = os.path.join(pc.CORRECTIONS_DIR, f"{name}_correction.npy")
    corrected = gt.copy()
    if os.path.exists(corr_path):
        correction = np.load(corr_path)
        corrected[correction == 1] = True
        corrected[correction == 2] = False
    return corrected


def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union else float("nan")


def check_accuracy(candidate, production):
    """Check 1: accuracy vs corrected ground truth, candidate vs production."""
    results = []
    for img in load_gt_manifest():
        feats = np.load(img["feat_path"])
        flat = feats.reshape(-1, feats.shape[-1])
        corrected_gt = build_corrected_gt(img["gt_path"], img["raw_source"])

        cand_pred = (candidate.predict_proba(flat)[:, 1] >= 0.5).reshape(corrected_gt.shape)
        prod_pred = (production.predict_proba(flat)[:, 1] >= 0.5).reshape(corrected_gt.shape)

        cand_iou = iou(cand_pred, corrected_gt)
        prod_iou = iou(prod_pred, corrected_gt)
        results.append({"image": img["name"], "candidate_iou": cand_iou, "production_iou": prod_iou})
        print(f"  [{img['name']}] candidate IoU={cand_iou:.4f}  production IoU={prod_iou:.4f}")

    mean_cand = float(np.mean([r["candidate_iou"] for r in results]))
    mean_prod = float(np.mean([r["production_iou"] for r in results]))
    passed = mean_cand >= mean_prod - IOU_TOLERANCE
    return {
        "name": "accuracy_vs_corrected_gt",
        "passed": passed,
        "mean_candidate_iou": mean_cand,
        "mean_production_iou": mean_prod,
        "tolerance": IOU_TOLERANCE,
        "per_image": results,
        "reason": None if passed else (
            f"candidate mean IoU {mean_cand:.4f} is more than {IOU_TOLERANCE} below "
            f"production's {mean_prod:.4f}"
        ),
    }


def band_density(mask, band):
    lo, hi = band
    h, w = mask.shape
    ring = np.zeros(mask.shape, dtype=bool)
    ring[lo:hi, :] = True
    ring[h - hi:h - lo, :] = True
    ring[:, lo:hi] = True
    ring[:, w - hi:w - lo] = True
    if not ring.any():
        return 0.0
    return float(mask[ring].mean())


def check_border_and_artifacts(candidate, production):
    """Checks 2, 3, 4: run on every image the paint tool knows about (not
    just the 4 with ground truth) since border/artifact regressions can
    show up on any image, GT or not."""
    border_flags = []
    artifact_flags = []
    degenerate_flags = []
    did_anything_change = []

    for info in pc.list_images():
        name, path = info["name"], info["path"]
        raw = tifffile.imread(path).astype(np.float64)
        img01 = robust_normalize(raw, 1.0, 99.0)
        feats = compute_feature_stack(img01)
        flat = feats.reshape(-1, feats.shape[-1])

        cand_prob = candidate.predict_proba(flat)[:, 1].reshape(img01.shape)
        prod_prob = production.predict_proba(flat)[:, 1].reshape(img01.shape)
        cand_mask = postprocess_mask(cand_prob)
        prod_mask = postprocess_mask(prod_prob)
        del feats, flat  # these are large; free before moving to the next image

        # Check 2: border/edge density comparison.
        for band in BORDER_BANDS_PX:
            cand_d = band_density(cand_mask, band)
            prod_d = band_density(prod_mask, band)
            if cand_d >= BORDER_DENSITY_ABS_MIN and cand_d > prod_d * BORDER_DENSITY_RATIO_MAX:
                border_flags.append({
                    "image": name, "band_px": band, "candidate_density": cand_d, "production_density": prod_d,
                })
                print(f"  BORDER FLAG [{name}] band {band}: candidate={cand_d:.3f} vs production={prod_d:.3f}")

        # Check 3: spontaneous new-artifact area.
        cand_labeled = label(cand_mask, connectivity=2)
        new_area = 0
        for lbl in range(1, cand_labeled.max() + 1):
            region = cand_labeled == lbl
            if not (region & prod_mask).any():
                new_area += int(region.sum())
        new_fraction = new_area / cand_mask.size
        if new_fraction > NEW_ARTIFACT_AREA_FRACTION_MAX:
            artifact_flags.append({"image": name, "new_area_px": new_area, "new_area_fraction": new_fraction})
            print(f"  ARTIFACT FLAG [{name}]: {new_area}px ({new_fraction*100:.2f}%) with zero overlap with production")

        # Check 4: degenerate output.
        n_regions = int(cand_labeled.max())
        area_fraction = float(cand_mask.mean())
        if n_regions == 0 or area_fraction > MAX_AREA_FRACTION or area_fraction < MIN_AREA_FRACTION:
            degenerate_flags.append({"image": name, "n_regions": n_regions, "area_fraction": area_fraction})
            print(f"  DEGENERATE FLAG [{name}]: n_regions={n_regions} area_fraction={area_fraction:.4f}")

        # Check 5 (report-only): did corrected pixels actually change vs production?
        corr_path = os.path.join(pc.CORRECTIONS_DIR, f"{name}_correction.npy")
        if os.path.exists(corr_path):
            correction = np.load(corr_path)
            touched = correction != 0
            if touched.any():
                changed_frac = float((cand_mask[touched] != prod_mask[touched]).mean())
                did_anything_change.append({"image": name, "changed_fraction_of_corrected_px": changed_frac})

    border_ok = len(border_flags) == 0
    artifacts_ok = len(artifact_flags) == 0
    degenerate_ok = len(degenerate_flags) == 0

    return [
        {
            "name": "border_edge_artifact", "passed": border_ok,
            "flags": border_flags,
            "reason": None if border_ok else f"{len(border_flags)} image/band combination(s) show a border density spike",
        },
        {
            "name": "spontaneous_artifacts", "passed": artifacts_ok,
            "flags": artifact_flags,
            "reason": None if artifacts_ok else f"{len(artifact_flags)} image(s) have suspiciously large brand-new area",
        },
        {
            "name": "degenerate_output", "passed": degenerate_ok,
            "flags": degenerate_flags,
            "reason": None if degenerate_ok else f"{len(degenerate_flags)} image(s) produced a degenerate mask",
        },
        {
            "name": "did_anything_change", "passed": True,  # report-only, never blocks
            "per_image": did_anything_change,
            "reason": None,
        },
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--correction-weight", type=float, default=1.0)
    ap.add_argument("--force", action="store_true", help="deploy regardless of gate results")
    args = ap.parse_args()

    candidate = train_candidate(args.correction_weight)
    production = joblib.load(pc.MODEL_PATH)

    print("\n=== Check 1: accuracy vs corrected ground truth ===")
    accuracy_check = check_accuracy(candidate, production)

    print("\n=== Checks 2-5: border, spontaneous artifacts, degenerate output, change-sanity (all 12 images) ===")
    other_checks = check_border_and_artifacts(candidate, production)

    all_checks = [accuracy_check] + other_checks
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "correction_weight": args.correction_weight,
        "checks": all_checks,
        "all_passed": all(c["passed"] for c in all_checks),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== Gate result ===")
    for c in all_checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}" + (f" -- {c['reason']}" if c["reason"] else ""))

    if report["all_passed"] or args.force:
        if not report["all_passed"] and args.force:
            print("\nGate FAILED but --force given -- deploying anyway.")
        backup_path = os.path.join(PROJECT_DIR, "models", f"pixel_hgb_final_prev_{int(time.time())}.joblib")
        shutil.copy(pc.MODEL_PATH, backup_path)
        shutil.copy(CANDIDATE_PATH, pc.MODEL_PATH)
        print(f"\nDEPLOYED: {CANDIDATE_PATH} -> {pc.MODEL_PATH}")
        print(f"(previous production backed up to {backup_path})")
        print("The paint tool will pick this up automatically on its next request -- no restart needed.")
    else:
        print(f"\nGATE FAILED -- nothing deployed. Candidate left at {CANDIDATE_PATH}.")
        print(f"Full report: {REPORT_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
