"""Mechanism, not score: why each ridge channel does or does not earn its column.

Three questions the arm table cannot answer on its own.

1. POOLED vs WITHIN-FRAME AUC. docs/CONTRAST.md established the trap that matters here:
   local contrast normalisation lifted texture_s2 from AUC 0.585 to 0.926 INSIDE one frame
   and still failed, because training pools frames and a per-frame-rescaled channel means
   something different in every row-block. frangi's gamma and meijering's max-division are
   both per-frame, so they should show exactly that signature -- a good within-frame AUC
   that does not survive pooling. sato and the raw Hessian eigenvalues are absolutely
   scaled and should not.

2. REDUNDANCY WITH THE SHIPPED 17. The Laplacian is the trace of the Hessian and is
   already in the baseline at sigma 1,2,4,8. If a ridge channel correlates ~0.9 with a
   column that is already there, then a high importance score for it means re-encoding
   rather than new information, and flat arm IoU is exactly what you would expect.

3. Whether any of it is different on the THIN frames, which are the ones the owner cares
   about.

Reads only the cache built by ridge_extract.py. Usage:
    .venv/bin/python research/ridge/ridge_diag.py
"""

import argparse
import json
import os
import sys

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
sys.path.insert(0, os.path.join(P0, "research", "ridge"))

import ridge_features as RF          # noqa: E402
from ridge_eval import load_cache    # noqa: E402


def auc(x, y):
    """Rank AUC, sign-free: reported as max(a, 1-a) so a channel that is discriminative
    in the 'wrong' direction is not scored as uninformative. The classifier can flip a
    sign for free; it cannot manufacture separation."""
    if y.all() or not y.any():
        return np.nan
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64) + 1
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    a = (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(a, 1 - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        "/private/tmp/claude-501/-Users-jiamingzhang-Desktop-APP",
        "6c65cf52-47c9-4de6-a996-f3251ee258ff", "scratchpad", "ridgecache"))
    ap.add_argument("--out", default=os.path.join(P0, "research", "ridge",
                                                  "ridge_diagnostics.json"))
    a = ap.parse_args()

    X, y, g, frames = load_cache(a.cache)
    thin = np.array([f["thin"] for f in frames])
    thin_rows = thin[g]
    print("rows %d cols %d frames %d (%d thin)" % (len(y), X.shape[1], len(frames),
                                                   thin.sum()), flush=True)

    added = list(range(RF.N_FEATURES, RF.N_COLS))
    base = RF.COL_BASE17
    out = {}

    # --- 1/3: pooled AUC, pooled-on-thin AUC, and mean within-frame AUC -------------
    for col in base + added:
        name = RF.ALL_NAMES[col]
        x = X[:, col].astype(np.float64)
        wf, wf_thin = [], []
        for i in range(len(frames)):
            s = g == i
            v = auc(x[s], y[s])
            if np.isfinite(v):
                wf.append(v)
                if thin[i]:
                    wf_thin.append(v)
        out[name] = dict(
            pooled_auc=round(float(auc(x, y)), 4),
            pooled_auc_thin=round(float(auc(x[thin_rows], y[thin_rows])), 4),
            within_frame_auc=round(float(np.mean(wf)), 4),
            within_frame_auc_thin=round(float(np.mean(wf_thin)), 4) if wf_thin else None,
        )
        # The signature of a per-frame-normalised channel: within-frame >> pooled.
        out[name]["pool_loss"] = round(out[name]["within_frame_auc"]
                                       - out[name]["pooled_auc"], 4)

    # --- 2/3: redundancy of each ADDED channel with the shipped 17 ------------------
    #     float64 and mean-centred: several of these columns have tiny dynamic range
    #     (hessian eigenvalues at sigma 4 span ~0.01) and float32 corr is unreliable there.
    B = X[:, base].astype(np.float64)
    B -= B.mean(0)
    Bn = np.linalg.norm(B, axis=0)
    for col in added:
        name = RF.ALL_NAMES[col]
        v = X[:, col].astype(np.float64)
        v -= v.mean()
        nv = np.linalg.norm(v)
        r = (B.T @ v) / np.maximum(Bn * nv, 1e-30)
        j = int(np.abs(r).argmax())
        out[name]["max_abs_corr_with_17"] = round(float(abs(r[j])), 4)
        out[name]["most_correlated_baseline_feature"] = RF.FEATURE_NAMES[j]
        # Multiple R^2 of the channel against ALL 17 at once -- the honest redundancy
        # number, since a channel can be poorly correlated with every single baseline
        # feature and still be an exact linear combination of several.
        try:
            coef, *_ = np.linalg.lstsq(B, v, rcond=None)
            resid = v - B @ coef
            out[name]["linear_r2_from_17"] = round(
                float(1.0 - (resid @ resid) / max(nv * nv, 1e-30)), 4)
        except np.linalg.LinAlgError:
            out[name]["linear_r2_from_17"] = None

    with open(a.out, "w") as fh:
        json.dump(dict(n_rows=int(len(y)), n_frames=len(frames), n_thin=int(thin.sum()),
                       channels=out), fh, indent=1)

    print("\n%-18s %7s %7s %7s %7s | %7s %-14s %6s" %
          ("channel", "pooled", "pool_th", "within", "p_loss", "|r|max", "vs", "linR2"),
          flush=True)
    print("-" * 92, flush=True)
    for col in base + added:
        n = RF.ALL_NAMES[col]
        d = out[n]
        if col == RF.N_FEATURES:
            print("-" * 30 + " ADDED RIDGE CHANNELS " + "-" * 40, flush=True)
        print("%-18s %7.4f %7.4f %7.4f %+7.4f | %7s %-14s %6s" % (
            n, d["pooled_auc"], d["pooled_auc_thin"], d["within_frame_auc"],
            d["pool_loss"],
            ("%.3f" % d["max_abs_corr_with_17"]) if "max_abs_corr_with_17" in d else "-",
            d.get("most_correlated_baseline_feature", "-"),
            ("%.3f" % d["linear_r2_from_17"])
            if d.get("linear_r2_from_17") is not None else "-"), flush=True)
    print("\nwrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
