"""
Runs the ACTUAL TXM pipeline (not a mock) on one worked-example image and
returns every intermediate array the diagram needs as thumbnails. Mirrors
the SEM project's pipeline_stages.py in spirit: real data at every stage,
nothing hand-drawn or simulated.
"""
import json
import os
import sys

import joblib
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txm_features import compute_feature_stack, robust_normalize, FEATURE_NAMES
from apply_pixel_model import predict_probability_map, postprocess_mask, PROB_THRESHOLD
import paint_common as pc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")


def naive_normalize(img):
    """Plain min-max stretch -- what you'd get WITHOUT the percentile-based
    robust_normalize, included purely to make (A)'s before/after visible:
    a few outlier pixels can otherwise wash out all the real contrast."""
    lo, hi = img.min(), img.max()
    return np.clip((img - lo) / max(hi - lo, 1e-8), 0, 1)


def to_rgb_overlay(img01, mask):
    gray = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[mask] = [255, 0, 0]
    return rgb


def compute_stages(image_key="338_13"):
    manifest_path = os.path.join(ROOT, "dataset_cache", "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    info = next(i for i in manifest["images"] if i["name"] == image_key)
    raw_path = info["raw_source"]
    name = os.path.splitext(os.path.basename(raw_path))[0]

    raw = tifffile.imread(raw_path).astype(np.float64)
    img01_naive = naive_normalize(raw)
    img01 = robust_normalize(raw, 1.0, 99.0)

    feats = compute_feature_stack(img01)
    feature_idx = FEATURE_NAMES.index("smooth_s32")
    feature_map = feats[..., feature_idx]

    model = joblib.load(pc.MODEL_PATH)
    prob_map = predict_probability_map(model, img01)
    raw_thresh_mask = prob_map >= PROB_THRESHOLD
    final_mask = postprocess_mask(prob_map)

    overlay_before_correction = to_rgb_overlay(img01, final_mask)

    corr_path = os.path.join(pc.CORRECTIONS_DIR, f"{name}_correction.npy")
    if os.path.exists(corr_path):
        correction = np.load(corr_path)
        effective_mask = final_mask.copy()
        effective_mask[correction == 1] = True
        effective_mask[correction == 2] = False
        n_corrected_px = int((correction != 0).sum())
    else:
        effective_mask = final_mask
        n_corrected_px = 0
    overlay_after_correction = to_rgb_overlay(img01, effective_mask)

    gate_report_path = os.path.join(RESULTS_DIR, "deploy_gate_report.json")
    gate_lines = None
    if os.path.exists(gate_report_path):
        with open(gate_report_path) as f:
            gate = json.load(f)
        label_map = {
            "accuracy_vs_corrected_gt": "Accuracy vs. corrected GT",
            "border_edge_artifact": "Border / edge artifact",
            "spontaneous_artifacts": "Spontaneous artifacts",
            "degenerate_output": "Degenerate output",
            "did_anything_change": "Did anything change",
        }
        gate_lines = []
        for c in gate["checks"]:
            label = label_map.get(c["name"], c["name"])
            sub = c.get("reason") or "no regression detected"
            gate_lines.append((label, sub, bool(c["passed"])))

    return dict(
        name=name,
        raw_display=raw,
        img01_naive=img01_naive,
        img01=img01,
        feature_map=feature_map,
        feature_name=FEATURE_NAMES[feature_idx],
        prob_map=prob_map,
        raw_thresh_mask=raw_thresh_mask,
        final_mask=final_mask,
        overlay_before_correction=overlay_before_correction,
        overlay_after_correction=overlay_after_correction,
        n_corrected_px=n_corrected_px,
        n_regions_final=int(final_mask.sum() > 0),
        gate_lines=gate_lines,
    )
