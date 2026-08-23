"""Runs the DEPLOYED pipeline on one worked example and returns every panel the
figure needs. Real arrays at every stage; nothing hand-drawn or simulated.

WHY THIS WAS REWRITTEN. The previous version ran the retired *research* path --
`apply_pixel_model.predict_probability_map` on a single joblib model, the legacy
`postprocess_mask`, and a `results/deploy_gate_report.json` from a five-check gate that no
longer exists. It therefore could not show SAM at all, and the figure it produced described
a system this project stopped shipping: no destitch, no flat-field, a lone MLP, hysteresis
post-processing that is now off by default, and bootstrap ground-truth labels that are now
held out of training entirely.

It now imports `app/core/model.py` and `app/core/pipeline.py` -- the same modules the running
app imports -- so the figure cannot drift from the deployed system again without the code
change showing up here first.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROJECT = os.path.abspath(os.path.join(ROOT, ".."))
for p in (os.path.join(PROJECT, "app", "core"), os.path.join(PROJECT, "code"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as M            # noqa: E402
import pipeline as P         # noqa: E402
from txm_features import compute_feature_stack, FEATURE_NAMES, robust_normalize  # noqa: E402

GT_CACHE = os.path.join(PROJECT, "dataset_cache")


def to_rgb_overlay(img01, mask, colour=(255, 0, 0)):
    gray = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1)
    rgb[mask] = colour
    return rgb


def sam_pc_rgb(coords, embs, shape):
    """The 256-d SAM embedding as an image: its first three principal components as RGB.

    Honest about what it is. Each 1024-px tile yields a 64x64 grid of 256-d vectors -- one
    vector per 16x16 block of pixels -- so this is that grid assembled at 1/16 resolution and
    projected to three dimensions. It is a view of the 256 channels, not one of them.
    """
    H, W = shape
    GH, GW = int(np.ceil(H / M.EMB_STRIDE)), int(np.ceil(W / M.EMB_STRIDE))
    C = embs.shape[1]
    grid = np.zeros((GH, GW, C), np.float32)
    for t, (y0, x0) in enumerate(np.asarray(coords)):
        g = np.asarray(embs[t], np.float32).transpose(1, 2, 0)      # 64,64,C
        gy, gx = int(y0) // M.EMB_STRIDE, int(x0) // M.EMB_STRIDE
        h, w = min(g.shape[0], GH - gy), min(g.shape[1], GW - gx)
        if h > 0 and w > 0:
            grid[gy:gy + h, gx:gx + w] = g[:h, :w]
    flat = grid.reshape(-1, C)
    flat = flat - flat.mean(axis=0, keepdims=True)
    # economy SVD on ~10k x 256 -- cheap, and avoids a sklearn import for three components
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    pcs = flat @ vt[:3].T
    out = np.zeros((pcs.shape[0], 3), np.float32)
    for i in range(3):
        lo, hi = np.percentile(pcs[:, i], [1, 99])
        out[:, i] = np.clip((pcs[:, i] - lo) / max(hi - lo, 1e-8), 0, 1)
    return (out.reshape(GH, GW, 3) * 255).astype(np.uint8)


def gate_lines_from_history():
    """The two real gate axes from the last recorded retrain, or None.

    There used to be a third, scoring candidates against four externally-labelled frames.
    Those labels are gone from the project, and this function kept emitting the axis anyway:
    it read the CROSS-VALIDATION IoU and captioned it "Reference frames (held out)", then
    coloured it from gate_detail["iou_ok"], a key the gate no longer sets -- so a passing
    retrain rendered with a failed axis and a mislabelled number.
    """
    try:
        hist = P.retrain_history()
    except Exception:                                            # noqa: BLE001
        return None
    if not hist:
        return None
    L = hist[-1]
    gd = L.get("gate_detail") or {}
    lines = []
    cf, if_ = L.get("candidate_clean_fp"), L.get("incumbent_clean_fp")
    lines.append((f"Crack-free specimens ({L.get('clean_specimens') or 0})",
                  (f"{if_*100:.2f}% -> {cf*100:.2f}% of area"
                   if cf is not None and if_ is not None else "not measured"),
                  bool(gd.get("fp_ok"))))
    ho = (L.get("heldout") or {}).get("mean_iou")
    lines.append(("Grouped-by-image cross-val",
                  f"IoU {ho:.3f}" if ho is not None else "not measured",
                  bool(gd.get("heldout_ok"))))
    return lines


def compute_stages(image_key="338_13"):
    img_p = os.path.join(GT_CACHE, f"{image_key}_img.npy")
    if not os.path.exists(img_p):
        raise SystemExit(f"no {img_p} -- pass a dataset_cache stem, e.g. 338_13")
    # The model input, exactly as pipeline._score feeds it when validating a retrain.
    img01 = np.asarray(np.load(img_p), np.float32)

    # (A)+(B) the display view. Geometry-preserving, and DELIBERATELY not the model input.
    import destitch
    import flatfield
    destitched, _ = destitch.destitch_image(img01.astype(np.float32))
    destitched = np.asarray(destitched, np.float32)
    ff = flatfield.flatfield(destitched)
    if isinstance(ff, tuple):
        ff = ff[0]
    display = robust_normalize(np.asarray(ff, np.float64), 1.0, 99.0).astype(np.float32)

    # (C) the 17 hand-crafted features, on the raw-normalised input
    feats = compute_feature_stack(img01)
    fi = FEATURE_NAMES.index("smooth_s32")
    feature_map = np.asarray(feats[..., fi], np.float32)
    del feats

    # (D) SAM ViT-H embedding
    coords, embs = M.embed_image(img01)
    sam_rgb = sam_pc_rgb(coords, embs, img01.shape)

    # (E) both members, so the figure can show what averaging buys
    m17 = M.CrackModel(path_17=M.DEFAULT_17, path_hybrid="", ensemble=False)
    p17 = m17.predict(img01)
    ens = P.get_model()
    p_ens = ens.predict(img01, emb=(coords, embs))

    # (F) the DEFAULT cleanup: a threshold and speck pruning. The legacy hysteresis
    # post-processing is off by default -- measured to cost 0.08 IoU on thin crack.
    raw_thresh = p_ens > 0.5
    final_mask = P.prune_specks(raw_thresh)


    return dict(
        name=image_key,
        img01=img01,
        destitched=destitched,
        display=display,
        feature_map=feature_map,
        feature_name=FEATURE_NAMES[fi],
        sam_rgb=sam_rgb,
        n_tiles=int(len(coords)),
        tile_stride=int(M.TILE_STRIDE),
        emb_channels=int(embs.shape[1]),
        p17=p17,
        p_ens=p_ens,
        raw_thresh_display=np.where(raw_thresh, 0, 255).astype(np.uint8),
        final_mask_display=np.where(final_mask, 0, 255).astype(np.uint8),
        final_mask=final_mask,
        overlay_model=to_rgb_overlay(display, final_mask),
        model_describe=ens.describe(),
        gate_lines=gate_lines_from_history(),
    )
