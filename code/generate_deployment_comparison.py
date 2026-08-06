"""
Qualitative before/after figure for the HistGradientBoosting -> MLP
production swap: raw image | previous production (HGB) overlay | new
production (MLP) overlay, for a few representative images, plus a final
deployment summary table for paper-comparison purposes.

Run AFTER retrain_and_deploy.py has actually deployed the MLP candidate
(so models/pixel_hgb_final.joblib IS the new MLP, and the most recent
models/pixel_hgb_final_prev_*.joblib is the backed-up previous HGB).

Usage:
    python3 generate_deployment_comparison.py
"""
import glob
import json
import os
import sys

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
from apply_pixel_model import predict_probability_map, postprocess_mask

FIG_DIR = os.path.join(pc.PROJECT_DIR, "benchmark_figures")
INK, MUTED = "#0b0b0b", "#898781"

# A representative spread: LARGE (biggest measured improvement), one
# "typical" small image from each of the two correction folders.
SHOWCASE_IMAGES = [
    "Average_mosaic_260618_b2_343_75_LARGE_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "Average_mosaic_260618_b2_338_13_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "Average_mosaic_260618_b2_342_81_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
]


def find_previous_hgb_backup():
    backups = sorted(glob.glob(os.path.join(pc.PROJECT_DIR, "models", "pixel_hgb_final_prev_*.joblib")))
    if not backups:
        raise FileNotFoundError("No pixel_hgb_final_prev_*.joblib backup found -- did the deploy actually run?")
    return backups[-1]  # most recent


def to_overlay(img01, mask):
    gray = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[mask] = [255, 0, 0]
    return rgb


def main():
    backup_path = find_previous_hgb_backup()
    print(f"Previous production (HGB) backup: {backup_path}")
    print(f"Current production (should be MLP): {pc.MODEL_PATH}")

    old_model = joblib.load(backup_path)
    new_model = joblib.load(pc.MODEL_PATH)

    fig, axes = plt.subplots(len(SHOWCASE_IMAGES), 3, figsize=(13, 4.2 * len(SHOWCASE_IMAGES)))
    if len(SHOWCASE_IMAGES) == 1:
        axes = axes[None, :]

    for row, name in enumerate(SHOWCASE_IMAGES):
        path = pc._find_path(name)
        import tifffile
        from txm_features import robust_normalize
        raw = tifffile.imread(path).astype(np.float64)
        img01 = robust_normalize(raw, 1.0, 99.0)

        old_prob = predict_probability_map(old_model, img01)
        old_mask = postprocess_mask(old_prob)
        new_prob = predict_probability_map(new_model, img01)
        new_mask = postprocess_mask(new_prob)

        # aspect="auto" -- these images span very different true aspect
        # ratios (LARGE is ~1.7:1, the others near-square); at aspect="equal"
        # (imshow's default) the wide ones letterbox inside their fixed-size
        # subplot cell, which was pushing a large blank gap in above row 1.
        gray = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
        axes[row, 0].imshow(gray, cmap="gray", aspect="auto")
        axes[row, 1].imshow(to_overlay(img01, old_mask), aspect="auto")
        axes[row, 2].imshow(to_overlay(img01, new_mask), aspect="auto")
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
        short_name = name.split("_idx00000")[0].replace("Average_mosaic_260618_", "")
        axes[row, 0].set_ylabel(short_name, fontsize=10)
        if row == 0:
            axes[row, 0].set_title("Raw (normalized)", fontsize=11)
            axes[row, 1].set_title("Previous production (HGB)", fontsize=11)
            axes[row, 2].set_title("New production (MLP)", fontsize=11)
        old_frac, new_frac = old_mask.mean(), new_mask.mean()
        axes[row, 1].text(0.02, 0.06, f"crack area {old_frac:.1%}", transform=axes[row, 1].transAxes,
                           color="white", fontsize=9, backgroundcolor="black")
        axes[row, 2].text(0.02, 0.06, f"crack area {new_frac:.1%}", transform=axes[row, 2].transAxes,
                           color="white", fontsize=9, backgroundcolor="black")
        print(f"  {name}: old_area={old_frac:.4f} new_area={new_frac:.4f}")

    fig.suptitle("(m) Qualitative before/after: HistGradientBoosting → MLP production swap")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(FIG_DIR, "fig_m_deployment_before_after.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
