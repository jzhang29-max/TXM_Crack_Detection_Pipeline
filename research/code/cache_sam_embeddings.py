"""
One-time, reusable SAM ViT-H embedding cache for all 71 images.

Why cache. Every hybrid operation -- training, inference, and every future
retrain after new paint corrections -- needs SAM's image embedding for the same
images. Recomputing costs GPU seconds per tile and would make the paint tool's
interactive loop unusable. Embedding once and caching makes the hybrid's extra
cost a fixed setup charge rather than a per-retrain tax.

CRITICAL -- normalisation must match the 17-feature path exactly. The features
in dataset_cache and the ones computed on the fly by retrain_with_corrections
both come from `robust_normalize(raw, 1.0, 99.0)` applied to the raw TIFF. If
the embeddings were computed from a differently normalised image, the two halves
of the concatenated feature vector would describe subtly different images and
the hybrid comparison would be meaningless. So this script loads and normalises
by calling the SAME functions, not by reimplementing them.

RAW input, not flatfielded, for the same reason: the deployed model this is
being compared against (raw_v4) is a raw-input model, and HANDOFF.md records
that switching to flatfielded cost 0.17 IoU. Holding the input fixed makes the
FEATURE SET the only variable.

Stored float16: SAM embeddings are ViT activations with no meaningful precision
below ~1e-3, and float32 would double a cache already near 1 GB. Verified
lossless to 3 decimal places on a sample tile.

Usage:
    python3 cache_sam_embeddings.py                 # all 71, skips existing
    python3 cache_sam_embeddings.py --only b2_336   # substring filter
    python3 cache_sam_embeddings.py --force         # recompute
"""

import argparse
import os
import sys
import time

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
import sam_common as S
from txm_features import robust_normalize

CACHE_DIR = os.path.join(pc.PROJECT_DIR, "paint", "sam_embcache")
MODEL_ID = "facebook/sam-vit-huge"
TILE = 1024


def load_img01(name):
    """The identical image the 17-feature path sees."""
    raw = tifffile.imread(pc._find_path(name)).astype(np.float64)
    return robust_normalize(raw, 1.0, 99.0).astype(np.float32)


def embed_image(img01, model_id=MODEL_ID):
    """Tiled SAM embeddings -> (coords int32 [n,2], emb float16 [n,C,64,64])."""
    coords, embs = [], []
    for (y0, y1, x0, x1) in S.tiles(img01.shape, size=TILE, overlap=0):
        crop = img01[y0:y1, x0:x1]
        if crop.shape != (TILE, TILE):
            # Reflect, never zero: a zero pad renders as pure black, which SAM
            # reads as a hard object boundary and which corrupts the ViT
            # features of every patch near it.
            crop = np.pad(crop, ((0, TILE - crop.shape[0]), (0, TILE - crop.shape[1])),
                          mode="reflect")
        embs.append(S.embed(S.to_rgb(crop), model_id=model_id).astype(np.float16))
        coords.append((y0, x0))
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
    return np.asarray(coords, np.int32), np.stack(embs)


def cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}_samemb.npz")


def ensure(name, force=False):
    """Return (coords, emb) for one image, computing and caching if needed."""
    p = cache_path(name)
    if os.path.exists(p) and not force:
        d = np.load(p)
        return d["coords"], d["emb"]
    img01 = load_img01(name)
    coords, emb = embed_image(img01)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(p, coords=coords, emb=emb, shape=np.asarray(img01.shape, np.int32))
    return coords, emb


def cached_shape(name):
    p = cache_path(name)
    if not os.path.exists(p):
        return None
    return tuple(np.load(p)["shape"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on image name")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    infos = [i for i in pc.list_images() if not args.only or args.only in i["name"]]
    print(f"SAM embedding cache -> {CACHE_DIR}")
    print(f"model {MODEL_ID} on {S.DEVICE} | {len(infos)} images\n")

    t_all = time.time()
    done = skipped = 0
    total_mb = 0.0
    for k, info in enumerate(infos, 1):
        name = info["name"]
        p = cache_path(name)
        if os.path.exists(p) and not args.force:
            total_mb += os.path.getsize(p) / 1e6
            skipped += 1
            continue
        t0 = time.time()
        try:
            coords, emb = ensure(name, force=args.force)
        except Exception as e:
            print(f"  [{k:2d}/{len(infos)}] FAILED {name[:44]}: {type(e).__name__}: {str(e)[:80]}")
            continue
        mb = os.path.getsize(p) / 1e6
        total_mb += mb
        done += 1
        print(f"  [{k:2d}/{len(infos)}] {info.get('group','?')[:18]:18s} "
              f"{len(coords):3d} tiles  {mb:6.1f} MB  {time.time()-t0:5.1f}s  {name[:40]}")

    print(f"\ncomputed {done}, reused {skipped}, cache {total_mb/1000:.2f} GB, "
          f"{time.time()-t_all:.0f}s total")


if __name__ == "__main__":
    main()
