"""
The SAM experiment matrix: does Meta's Segment Anything beat the deployed
17-feature pixel classifier on TXM crack images?

Scored on the same 4 Ilastik ground-truth images, with the same
metrics_from_pred, as every other model in this project.

CONDITIONS -- split by whether they could ever run in deployment:

  DEPLOYABLE (no ground truth used at inference time)
    amg_whole      automatic mask generation on the whole frame, then keep
                   masks darker than the median. "Just run SAM on it."
    amg_tiled      same, but 1024px tiles at NATIVE resolution, so thin
                   structures are not destroyed by SAM's 1024 long-edge resize
    grid_points    32x32 point grid per tile, best-of-3 by SAM's own
                   confidence, darkness selection
    embed_lr       SAM ViT embeddings -> logistic regression, LOIO
    embed_mlp      SAM ViT embeddings -> MLP, LOIO
    embed_plus17   SAM embeddings CONCATENATED with the 17 hand-crafted
                   features -> MLP, LOIO. Asks the useful question: does the
                   foundation model ADD anything on top of what we have?

  ORACLE / NOT DEPLOYABLE (uses the ground truth it is scored against)
    amg_oracle     AMG proposals + a perfect mask picker. Ceiling on "SAM's
                   proposals + any selection rule I could invent."
    pts_oracle     prompt points sampled ON the true crack skeleton
    box_oracle     one prompt box per true crack component
    pts_oracle_group  ALL true-crack points as ONE multi-point prompt plus
                   negative points off the crack -- SAM's strongest prompting
                   mode, so "you prompted it badly" is not available as an out
    amg_relaxed    automatic masks with the confidence gates lowered
                   (pred_iou 0.88->0.5, stability 0.95->0.7), because the
                   natural-image defaults discard ~99.8% of proposals and a
                   filter must not be mistaken for a capability limit

The oracle rows exist so the result cannot be dismissed as bad prompting.
If SAM loses while being handed the answer, the answer is not the problem.

Usage:
    python3 sam_experiments.py --conditions amg_whole,amg_tiled --model huge
    python3 sam_experiments.py --conditions all --model huge --out results/sam/huge.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sam_common as S

MODELS = {
    "base": "facebook/sam-vit-base",
    "huge": "facebook/sam-vit-huge",        # strongest SAM 1
    "sam2": "facebook/sam2.1-hiera-large",  # current generation, so "we tried
                                            # the old version" is not an out
}
TILE = 1024
EMB_STRIDE = 16          # ViT patch stride: a 1024px tile -> 64x64 embedding grid
SEED = 0

# Prompt budgets. Previously hard-coded to 16 while the module docstring
# advertised 32 -- half SAM's own points_per_crop default, which quietly
# understated the prompted conditions. Named constants so the code, the
# docstring and the JSON payload cannot disagree again.
GRID_N = 32              # 32x32 = 1024 grid points per tile
ORACLE_N = 32            # oracle prompt points sampled on the true crack
ORACLE_BOXES = 32        # oracle boxes, one per true crack component

# When set (via --save-masks), each condition writes its predicted mask for
# this image so a qualitative panel can be rendered without re-running SAM.
SAVE_MASK_STEM = os.environ.get("TXM_SAVE_MASK_STEM", "336_25")
SAVE_MASKS = False
MASK_DIR = None


def maybe_save_mask(cond, stem, pred):
    if not (SAVE_MASKS and stem == SAVE_MASK_STEM and MASK_DIR):
        return
    os.makedirs(MASK_DIR, exist_ok=True)
    np.save(os.path.join(MASK_DIR, f"{cond}__{stem}.npy"), np.asarray(pred, bool))

# Matches the existing benchmark's sampling budget so the learned SAM
# conditions are trained on the same number of pixels as the 17-feature model.
N_PER_CLASS = 20000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


def free_gpu():
    """Release MPS cache between images; ViT-H masks at 23 MP fragment it fast."""
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def amg_masks(rgb, model_id, points_per_batch=None, relaxed=False):
    """Automatic mask generation on one RGB array -> list of bool masks.

    `relaxed` matters more than it looks. HuggingFace's mask-generation
    pipeline defaults to pred_iou_thresh=0.88 and stability_score_thresh=0.95,
    tuned for natural photographs, and those cull the overwhelming majority of
    proposals on this data -- only 8-13 masks survive per image, against 50-100
    typical on photos. Reporting that as "SAM does not resolve the structure"
    would confuse a CONFIDENCE FILTER with a CAPABILITY LIMIT. The relaxed
    setting lowers both gates so the question can actually be answered: does
    SAM propose a crack mask that the default thresholds then throw away?

    points_per_batch is scaled to the image by default: post_process_masks
    allocates points_per_batch x 3 x H x W, which overflows int32 above roughly
    8 MP at the default of 64. That overflow -- not a memory wall -- is what
    killed whole-frame inference on the 23.5 MP mosaic.
    """
    from transformers import pipeline
    from PIL import Image
    key = ("amgpipe", model_id)
    if key not in S._CACHE:
        S._CACHE[key] = pipeline("mask-generation", model=model_id, device=S.DEVICE)
    gen = S._CACHE[key]

    H, W = rgb.shape[:2]
    if points_per_batch is None:
        points_per_batch = max(1, min(64, (2 ** 31 - 1) // (3 * H * W)))

    kw = dict(points_per_batch=points_per_batch)
    if relaxed:
        kw.update(pred_iou_thresh=0.5, stability_score_thresh=0.7,
                  crops_nms_thresh=0.95)
    out = gen(Image.fromarray(rgb), **kw)
    return [np.asarray(m).astype(bool) for m in out["masks"]]


def best_of_multimask(masks, confs, gt=None):
    """Collapse SAM's 3 ambiguity-resolution masks to one per prompt.

    gt=None -> pick by SAM's own predicted IoU (deployable).
    gt given -> pick the truly best (oracle); prevents 'SAM lost because the
    wrong one of its three masks was read out'.
    """
    out = []
    for i in range(masks.shape[0]):
        if gt is None:
            j = int(np.argmax(confs[i]))
        else:
            j = int(np.argmax([S.metrics_from_pred(masks[i, k], gt)["iou"]
                               for k in range(masks.shape[1])]))
        out.append(masks[i, j])
    return out


def embed_tiled(img, model_id, mode="gray"):
    """Per-tile SAM ViT embeddings.

    Returns (list of (y0, x0), array [n_tiles, C, 64, 64]). Tiles are exactly
    1024x1024 (S.tiles clamps by shifting inward), so the pixel -> embedding
    mapping is a clean divide-by-16 with no padding to reason about.
    """
    coords, embs = [], []
    for (y0, y1, x0, x1) in S.tiles(img.shape, size=TILE, overlap=0):
        crop = img[y0:y1, x0:x1]
        if crop.shape[0] < TILE or crop.shape[1] < TILE:
            pad = np.zeros((TILE, TILE), np.float32)
            pad[:crop.shape[0], :crop.shape[1]] = crop
            crop = pad
        embs.append(S.embed(S.to_rgb(crop, mode), model_id=model_id))
        coords.append((y0, x0))
    return coords, np.stack(embs)


def sample_embed_features(img, gt, coords, embs, n_per_class, rng):
    """Sample pixels and bilinearly interpolate their SAM embedding vector.

    Also returns each sample's GLOBAL (row, col) so the +17 hybrid condition
    can look up the hand-crafted features at exactly the same pixels -- the
    two feature sets must describe identical samples or the comparison is
    meaningless.
    """
    Xs, ys, rows, cols = [], [], [], []
    per_tile = max(n_per_class // max(len(coords), 1), 1)
    for (y0, x0), emb in zip(coords, embs):
        gsub = gt[y0:y0 + TILE, x0:x0 + TILE]
        if gsub.size == 0:
            continue
        for cls in (1, 0):
            idx = np.nonzero((gsub == bool(cls)).ravel())[0]
            if len(idx) == 0:
                continue
            take = rng.choice(idx, size=min(per_tile, len(idx)), replace=False)
            rr, cc = np.unravel_index(take, gsub.shape)
            Xs.append(interp_embed(emb, rr, cc))
            ys.append(np.full(len(rr), cls, np.int8))
            rows.append(rr + y0)
            cols.append(cc + x0)
    if not Xs:
        z = np.zeros(0, np.int64)
        return np.zeros((0, embs.shape[1]), np.float32), np.zeros(0, np.int8), z, z
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(rows), np.concatenate(cols))


def interp_embed(emb, rr, cc):
    """Bilinear lookup of tile-local pixel coords (rr, cc) in a C x 64 x 64 grid."""
    C = emb.shape[0]
    r = np.clip(rr / EMB_STRIDE - 0.5, 0, emb.shape[1] - 1)
    c = np.clip(cc / EMB_STRIDE - 0.5, 0, emb.shape[2] - 1)
    out = np.empty((len(rr), C), np.float32)
    for k in range(C):
        out[:, k] = ndi.map_coordinates(emb[k], np.stack([r, c]), order=1, mode="nearest")
    return out


def predict_embed_image(img, coords, embs, clf, band=128, feat17=None):
    """Full-resolution probability map from SAM embeddings, chunked by row band.

    Full resolution on purpose: predicting on the coarse 64x64 embedding grid
    and upsampling afterwards would be ~250x cheaper but would quietly change
    what is being measured, and this comparison has to survive scrutiny.
    """
    prob = np.zeros(img.shape, np.float32)
    for (y0, x0), emb in zip(coords, embs):
        h = min(TILE, img.shape[0] - y0)
        w = min(TILE, img.shape[1] - x0)
        for b0 in range(0, h, band):
            b1 = min(b0 + band, h)
            rr, cc = np.meshgrid(np.arange(b0, b1), np.arange(w), indexing="ij")
            X = interp_embed(emb, rr.ravel(), cc.ravel())
            if feat17 is not None:
                extra = feat17[y0 + b0:y0 + b1, x0:x0 + w, :].reshape(-1, feat17.shape[2])
                X = np.concatenate([X, extra], axis=1)
            p = clf.predict_proba(X)[:, 1].astype(np.float32)
            prob[y0 + b0:y0 + b1, x0:x0 + w] = p.reshape(b1 - b0, w)
    return prob


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------
def run_amg(model_id, mode, tiled, oracle, relaxed=False):
    rows = []
    for stem in S.STEMS:
        img, gt = S.load_pair(stem)
        t0 = time.time()
        # best_prop: the highest IoU reached by ANY SINGLE raw proposal, before
        # selection. This is the measurement that separates "SAM never proposes
        # the crack" from "SAM proposes it and our selection rule misses it" --
        # without it a low score cannot be attributed to either.
        best_prop, n_masks = 0.0, 0
        try:
            if tiled:
                # Select WITHIN each tile and OR the result in. Promoting every
                # tile mask to a full-frame array first would allocate
                # ~280 x 23.5 MB = 6.6 GB on the LARGE mosaic.
                pred = np.zeros(gt.shape, bool)
                for (y0, y1, x0, x1) in S.tiles(img.shape, size=TILE, overlap=0):
                    sub, gsub = img[y0:y1, x0:x1], gt[y0:y1, x0:x1]
                    ms = [m[:y1 - y0, :x1 - x0]
                          for m in amg_masks(S.to_rgb(sub, mode), model_id, relaxed=relaxed)]
                    n_masks += len(ms)
                    if gsub.any():
                        for m in ms:
                            best_prop = max(best_prop, S.metrics_from_pred(m, gsub)["iou"])
                    pred[y0:y1, x0:x1] |= (S.select_oracle_union(ms, gsub) if oracle
                                           else S.select_by_darkness(ms, sub))
                    free_gpu()
            else:
                masks = amg_masks(S.to_rgb(img, mode), model_id, relaxed=relaxed)
                n_masks = len(masks)
                for m in masks:
                    best_prop = max(best_prop, S.metrics_from_pred(m, gt)["iou"])
                pred = (S.select_oracle_union(masks, gt) if oracle
                        else S.select_by_darkness(masks, img))
        except (RuntimeError, ValueError) as e:
            free_gpu()
            rows.append(dict(image=stem, iou=float("nan"), dice=float("nan"),
                             precision=float("nan"), recall=float("nan"),
                             pred_area_frac=float("nan"), gt_area_frac=float(gt.mean()),
                             error=f"{type(e).__name__}: {str(e)[:120]}",
                             megapixels=round(gt.size / 1e6, 1)))
            log(f"    {stem:16s} ERROR ({gt.size/1e6:.1f} MP): {str(e)[:70]}")
            continue
        tag = "amg_" + ("oracle" if oracle else ("relaxed" if relaxed else
                                                ("tiled" if tiled else "whole")))
        maybe_save_mask(tag, stem, pred)
        r = S.score(pred, gt, image=stem, n_masks=n_masks,
                    best_single_proposal_iou=round(best_prop, 4),
                    secs=round(time.time() - t0, 1))
        rows.append(r)
        log(f"    {stem:16s} masks={n_masks:5d} IoU={r['iou']:.3f} "
            f"rec={r['recall']:.3f} prec={r['precision']:.3f} "
            f"area={r['pred_area_frac']*100:.1f}%  bestProposal={best_prop:.3f}")
        free_gpu()
    return rows


def run_prompts(model_id, mode, kind):
    """kind: grid | pts_oracle | box_oracle"""
    rows = []
    rng = np.random.RandomState(SEED)
    for stem in S.STEMS:
        img, gt = S.load_pair(stem)
        t0 = time.time()
        pred = np.zeros(gt.shape, bool)
        for (y0, y1, x0, x1) in S.tiles(img.shape, size=TILE, overlap=0):
            free_gpu()
            sub, gsub = img[y0:y1, x0:x1], gt[y0:y1, x0:x1]
            rgb = S.to_rgb(sub, mode)
            if kind == "grid":
                pts = S.grid_points(sub.shape, n=GRID_N)
                if len(pts) == 0:
                    continue
                m, c = S.predict_prompted(rgb, points=pts.reshape(-1, 1, 2), model_id=model_id)
                chosen = best_of_multimask(m, c, gt=None)
                sel = S.select_by_darkness(chosen, sub)
            elif kind == "pts_oracle":
                pts = S.oracle_points_from_gt(gsub, n=ORACLE_N, rng=rng)
                if len(pts) == 0:
                    continue
                m, c = S.predict_prompted(rgb, points=pts.reshape(-1, 1, 2), model_id=model_id)
                sel = S.select_oracle_union(best_of_multimask(m, c, gt=gsub), gsub)
            elif kind == "pts_oracle_group":
                # ALL oracle points as ONE prompt group, plus negative points off
                # the crack. This is SAM's strongest mode for a single large
                # diffuse structure, and the mode that actually matches the
                # question "told where the crack is, can SAM trace its extent?".
                # Feeding N separate one-point prompts (pts_oracle) asks a
                # different, weaker question and understates SAM.
                pos = S.oracle_points_from_gt(gsub, n=ORACLE_N, rng=rng)
                if len(pos) == 0:
                    continue
                neg_ys, neg_xs = np.nonzero(~gsub)
                if len(neg_ys):
                    pick = rng.choice(len(neg_ys), size=min(ORACLE_N, len(neg_ys)), replace=False)
                    neg = np.stack([neg_xs[pick], neg_ys[pick]], axis=1).astype(np.float32)
                else:
                    neg = np.zeros((0, 2), np.float32)
                allp = np.concatenate([pos, neg]).reshape(1, -1, 2)
                lab = np.concatenate([np.ones(len(pos), np.int64),
                                      np.zeros(len(neg), np.int64)]).reshape(1, -1)
                m, c = S.predict_prompted(rgb, points=allp, labels=lab, model_id=model_id)
                sel = S.select_oracle_union(best_of_multimask(m, c, gt=gsub), gsub)
            else:
                bx = S.oracle_boxes_from_gt(gsub, max_boxes=ORACLE_BOXES)
                if len(bx) == 0:
                    continue
                m, c = S.predict_prompted(rgb, boxes=bx, model_id=model_id)
                sel = S.select_oracle_union(best_of_multimask(m, c, gt=gsub), gsub)
            pred[y0:y1, x0:x1] |= sel
        maybe_save_mask(kind if kind != "grid" else "grid_points", stem, pred)
        r = S.score(pred, gt, image=stem, secs=round(time.time() - t0, 1))
        rows.append(r)
        log(f"    {stem:16s} IoU={r['iou']:.3f} rec={r['recall']:.3f} "
            f"prec={r['precision']:.3f} area={r['pred_area_frac']*100:.1f}%")
    return rows


def run_embed_loio(model_id, mode, head, plus17):
    """Leave-one-image-out with SAM embeddings as the feature vector.

    LOIO not random k-fold, for the reason recorded in the benchmark: pixels
    from one image leak between random folds and inflate IoU by 0.14-0.18.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    log("      embedding all 4 images ...")
    cache = {}
    for stem in S.STEMS:
        img, gt = S.load_pair(stem)
        coords, embs = embed_tiled(img, model_id, mode)
        cache[stem] = (img, gt, coords, embs)
        log(f"        {stem:16s} {len(coords)} tiles, emb {embs.shape[1]}ch")

    rows = []
    for held in S.STEMS:
        rng = np.random.RandomState(SEED)
        Xtr, ytr = [], []
        for stem in S.STEMS:
            if stem == held:
                continue
            img, gt, coords, embs = cache[stem]
            X, y, rr, cc = sample_embed_features(img, gt, coords, embs, N_PER_CLASS, rng)
            if plus17:
                f17 = np.load(os.path.join(S.RAW_CACHE, f"{stem}_features.npy"), mmap_mode="r")
                X = np.concatenate([X, np.asarray(f17[rr, cc, :], np.float32)], axis=1)
                del f17
            Xtr.append(X)
            ytr.append(y)
        Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)

        if head == "lr":
            clf = Pipeline([("s", StandardScaler()),
                            ("m", LogisticRegression(max_iter=2000, C=1.0))])
        else:
            clf = Pipeline([("s", StandardScaler()),
                            ("m", MLPClassifier(hidden_layer_sizes=(128, 64),
                                                max_iter=400, random_state=SEED))])
        t0 = time.time()
        clf.fit(Xtr, ytr)
        img, gt, coords, embs = cache[held]
        f17h = (np.load(os.path.join(S.RAW_CACHE, f"{held}_features.npy"), mmap_mode="r")
                if plus17 else None)
        prob = predict_embed_image(img, coords, embs, clf, feat17=f17h)
        maybe_save_mask(f"embed_{head}{'_plus17' if plus17 else ''}", held, prob > 0.5)
        r = S.score(prob > 0.5, gt, image=held, n_train=int(len(ytr)),
                    n_features=int(Xtr.shape[1]), secs=round(time.time() - t0, 1))
        rows.append(r)
        log(f"    {held:16s} IoU={r['iou']:.3f} rec={r['recall']:.3f} "
            f"prec={r['precision']:.3f} area={r['pred_area_frac']*100:.1f}%")
    return rows


CONDITIONS = {
    "amg_whole":   dict(fn=run_amg, kw=dict(tiled=False, oracle=False), deployable=True),
    "amg_tiled":   dict(fn=run_amg, kw=dict(tiled=True, oracle=False), deployable=True),
    "amg_relaxed": dict(fn=run_amg, kw=dict(tiled=True, oracle=False, relaxed=True), deployable=True),
    "amg_oracle":  dict(fn=run_amg, kw=dict(tiled=True, oracle=True), deployable=False),
    "amg_relaxed_oracle": dict(fn=run_amg, kw=dict(tiled=True, oracle=True, relaxed=True), deployable=False),
    "grid_points": dict(fn=run_prompts, kw=dict(kind="grid"), deployable=True),
    "pts_oracle":  dict(fn=run_prompts, kw=dict(kind="pts_oracle"), deployable=False),
    "box_oracle":  dict(fn=run_prompts, kw=dict(kind="box_oracle"), deployable=False),
    "pts_oracle_group": dict(fn=run_prompts, kw=dict(kind="pts_oracle_group"), deployable=False),
    "embed_lr":    dict(fn=run_embed_loio, kw=dict(head="lr", plus17=False), deployable=True),
    "embed_mlp":   dict(fn=run_embed_loio, kw=dict(head="mlp", plus17=False), deployable=True),
    "embed_plus17": dict(fn=run_embed_loio, kw=dict(head="mlp", plus17=True), deployable=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="amg_whole,amg_tiled")
    ap.add_argument("--model", default="huge", choices=list(MODELS))
    ap.add_argument("--mode", default="gray", choices=["gray", "clahe", "invert"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-masks", action="store_true",
                    help=f"persist each condition's mask for {SAVE_MASK_STEM} (qualitative figure)")
    args = ap.parse_args()

    global SAVE_MASKS, MASK_DIR
    SAVE_MASKS = args.save_masks
    MASK_DIR = os.path.join(S.PROJECT_DIR, "results", "sam", "masks")

    names = list(CONDITIONS) if args.conditions == "all" else args.conditions.split(",")
    model_id = MODELS[args.model]
    log(f"SAM matrix | model={model_id} | render={args.mode} | device={S.DEVICE}")

    # Self-describing payload: prompt budgets and selection settings recorded
    # next to the numbers, so a reader never has to guess what was run.
    run_meta = dict(model=model_id, render=args.mode, device=S.DEVICE,
                    n_per_class=N_PER_CLASS, tile=TILE, emb_stride=EMB_STRIDE,
                    grid_points_n=GRID_N, oracle_points_n=ORACLE_N,
                    oracle_max_boxes=ORACLE_BOXES,
                    darkness_quantile=0.5, darkness_max_area_frac=0.5,
                    amg_relaxed_settings=dict(pred_iou_thresh=0.5,
                                              stability_score_thresh=0.7,
                                              crops_nms_thresh=0.95))
    results = {}
    for name in names:
        spec = CONDITIONS[name]
        log(f"\n  [{name}]  deployable={spec['deployable']}")
        t0 = time.time()
        try:
            rows = spec["fn"](model_id, args.mode, **spec["kw"])
        except Exception as e:
            log(f"    FAILED {type(e).__name__}: {e}")
            results[name] = dict(error=f"{type(e).__name__}: {e}")
            continue
        ok = [r for r in rows if np.isfinite(r["iou"])]
        results[name] = dict(
            deployable=spec["deployable"], rows=rows,
            n_images_scored=len(ok), images_scored=[r.get("image") for r in ok],
            mean_iou=float(np.mean([r["iou"] for r in ok])) if ok else None,
            mean_dice=float(np.mean([r["dice"] for r in ok])) if ok else None,
            mean_recall=float(np.mean([r["recall"] for r in ok])) if ok else None,
            mean_precision=float(np.mean([r["precision"] for r in ok])) if ok else None,
            secs=round(time.time() - t0, 1))
        mi, mr = results[name]["mean_iou"], results[name]["mean_recall"]
        log(f"    => mean IoU {'nan' if mi is None else f'{mi:.3f}'}  "
            f"recall {'nan' if mr is None else f'{mr:.3f}'}  "
            f"over {len(ok)}/{len(rows)} images  ({results[name]['secs']}s)")
        # dump after EVERY condition: a late crash previously discarded all
        # earlier work because the JSON was only written at the very end.
        _dump(args, run_meta, results)

    log(f"\nwrote {_dump(args, run_meta, results)}")


def _dump(args, run_meta, results):
    payload = dict(**run_meta, results=results)
    out = args.out or os.path.join(S.PROJECT_DIR, "results", "sam",
                                   f"{args.model}_{args.mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


if __name__ == "__main__":
    main()
