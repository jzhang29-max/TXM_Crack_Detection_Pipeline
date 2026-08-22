"""SAM 3's vision encoder as a drop-in replacement for SAM 1's, for measurement only.

    python3 code/experiment_sam3.py          # the comparison this exists to serve

WHY THIS IS A SEPARATE MODULE AND NOT A CHANGE TO app/core/model.py. Swapping the encoder
changes the width and the spatial stride of the 273-d feature vector, which invalidates both
shipped models, every cached prediction, the reference feature stacks, and every number in
the repo. None of that is worth doing on a hunch, so this module lets the question be
measured first and leaves the deployed path untouched.

WHAT THE PROJECT ACTUALLY USES SAM FOR. Not masks. `app/core/model.py` calls the image
encoder and never the prompt encoder or mask decoder, because zero-shot SAM prompted the way
SAM is designed to be prompted measures 0.23-0.36 IoU here against 0.82 for the trained
hybrid. SAM 3's headline feature -- open-vocabulary concept segmentation from a text prompt
-- is therefore the part of it this project has the least use for: there is no noun phrase
for a hairline fatigue crack in a grayscale X-ray of 316L that a web-trained model has a
prior on. The only question worth asking is whether its Perception Encoder produces more
discriminative DENSE FEATURES than SAM 1's ViT-H neck.

STRIDE AND WIDTH ARE DISCOVERED, NOT ASSUMED. SAM 1 gives one 256-d vector per 16x16 px
block (a 64x64 grid per 1024-px tile), which is why `model.interp_tile` can hardcode a
divide-by-16. SAM 3 returns a multi-level FPN, and its `backbone_feature_sizes` and
`scale_factors` default to None in transformers -- they arrive with the checkpoint config.
So this module reads the shapes off the tensors it actually gets and picks the level whose
stride is closest to SAM 1's 16, keeping the comparison like-for-like and the feature width
at 256 (`fpn_hidden_size` is 256, same as SAM 1's neck).
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (os.path.join(_PROJECT, "app", "core"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as M            # noqa: E402  (TILE, tiles())

SAM3_REPO = "facebook/sam3"
TARGET_STRIDE = M.EMB_STRIDE          # 16, so SAM 3 is compared at SAM 1's granularity
CACHE = os.path.join(_PROJECT, "paint", "sam3_embcache")

_sam3 = None


class Sam3Unavailable(RuntimeError):
    """SAM 3 cannot be used here, and str(self) says exactly why."""


def availability():
    """(ok, reason) -- checked in the order a user hits the problems."""
    try:
        import transformers
    except Exception as e:                                       # noqa: BLE001
        return False, f"transformers not importable: {e}"
    if not hasattr(transformers, "Sam3Model"):
        return False, (f"transformers {transformers.__version__} has no Sam3Model; "
                       f"SAM 3 support landed in transformers 5.x")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=SAM3_REPO, filename="config.json")
    except Exception as e:                                       # noqa: BLE001
        return False, (f"{SAM3_REPO} is gated ({type(e).__name__}). Accept Meta's SAM "
                       f"License on https://huggingface.co/{SAM3_REPO} with your own "
                       f"account, then `huggingface-cli login`. This step cannot be "
                       f"automated -- it is a licence agreement.")
    return True, "ok"


def _load():
    global _sam3
    if _sam3 is None:
        ok, why = availability()
        if not ok:
            raise Sam3Unavailable(why)
        import torch
        from transformers import Sam3Model, Sam3Processor
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        proc = Sam3Processor.from_pretrained(SAM3_REPO)
        mdl = Sam3Model.from_pretrained(SAM3_REPO).to(dev).eval()
        _sam3 = (proc, mdl, dev, torch)
    return _sam3


def _pick_level(fpn_maps, tile_px):
    """Index of the FPN level whose stride is closest to SAM 1's, plus that stride."""
    best, best_err, best_stride = 0, None, None
    for i, t in enumerate(fpn_maps):
        # levels arrive as (B, C, H, W); a square tile makes stride = tile / H
        h = int(t.shape[-2])
        if h <= 0:
            continue
        stride = tile_px / h
        err = abs(stride - TARGET_STRIDE)
        if best_err is None or err < best_err:
            best, best_err, best_stride = i, err, stride
    if best_stride is None:
        raise Sam3Unavailable("SAM 3 returned no usable FPN level")
    return best, best_stride


def embed_image(img01, progress=None, level=None):
    """Tiled SAM 3 features -> (coords int32 [n,2], emb float16 [n,C,h,w], stride).

    Mirrors model.embed_image's tiling exactly so the two encoders are compared on the same
    pixels, and returns the stride because -- unlike SAM 1 -- it is not known in advance.
    """
    proc, mdl, dev, torch = _load()
    tl = M.tiles(img01.shape)
    coords, embs, stride, chosen = [], [], None, level
    for k, (y0, y1, x0, x1) in enumerate(tl):
        crop = img01[y0:y1, x0:x1]
        if crop.shape != (M.TILE, M.TILE):
            crop = np.pad(crop, ((0, M.TILE - crop.shape[0]), (0, M.TILE - crop.shape[1])),
                          mode="reflect")
        u8 = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([u8] * 3, -1)
        inp = proc(images=rgb, return_tensors="pt")
        px = inp["pixel_values"]
        px = (px.float() if px.dtype == torch.float64 else px).to(dev)
        with torch.no_grad():
            out = mdl.get_vision_features(pixel_values=px)
        maps = list(out.fpn_hidden_states)
        if chosen is None:
            chosen, stride = _pick_level(maps, M.TILE)
        t = maps[chosen]
        embs.append(t.float().cpu().numpy()[0].astype(np.float16))
        coords.append((y0, x0))
        if dev == "mps":
            torch.mps.empty_cache()
        if progress:
            progress(k + 1, len(tl))
    return np.asarray(coords, np.int32), np.stack(embs), float(stride)


def interp_tile(emb_tile, rr, cc, stride):
    """Bilinear lookup into one C x h x w grid at an ARBITRARY stride.

    model.interp_tile hardcodes /16 because SAM 1's grid is always 64x64 per 1024-px tile.
    Same arithmetic, with the stride passed in.
    """
    e = np.ascontiguousarray(emb_tile, dtype=np.float32)
    C, H, W = e.shape
    r = np.clip(rr / stride - 0.5, 0, H - 1)
    c = np.clip(cc / stride - 0.5, 0, W - 1)
    r0 = np.floor(r).astype(np.intp); c0 = np.floor(c).astype(np.intp)
    r1 = np.minimum(r0 + 1, H - 1);   c1 = np.minimum(c0 + 1, W - 1)
    dr = (r - r0).astype(np.float32)[:, None]
    dc = (c - c0).astype(np.float32)[:, None]
    f = e.reshape(C, H * W)
    return (f[:, r0 * W + c0].T * (1 - dr) * (1 - dc)
            + f[:, r0 * W + c1].T * (1 - dr) * dc
            + f[:, r1 * W + c0].T * dr * (1 - dc)
            + f[:, r1 * W + c1].T * dr * dc)


def cached_embedding(key, img01, progress=None):
    """Embed once per image and keep it; SAM 3 is far too slow to recompute per fold."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{key}.npz")
    if os.path.exists(p):
        try:
            z = np.load(p)
            return z["coords"], z["emb"], float(z["stride"])
        except Exception:                                        # noqa: BLE001
            os.remove(p)                                         # truncated cache, redo it
    coords, emb, stride = embed_image(img01, progress=progress)
    tmp = f"{p}.{os.getpid()}.tmp"
    np.savez(tmp, coords=coords, emb=emb, stride=np.float32(stride))
    os.replace(tmp, p)
    return coords, emb, stride


def rows_at(coords, emb, stride, rr, cc):
    """The [n, C] SAM 3 block for pixel coordinates rr, cc -- last tile wins, as in model.py."""
    out = np.zeros((len(rr), emb.shape[1]), np.float32)
    todo = np.ones(len(rr), bool)
    for t in range(len(coords) - 1, -1, -1):
        y0, x0 = int(coords[t][0]), int(coords[t][1])
        sel = (todo & (rr >= y0) & (rr < y0 + M.TILE) & (cc >= x0) & (cc < x0 + M.TILE))
        if sel.any():
            out[sel] = interp_tile(emb[t], rr[sel] - y0, cc[sel] - x0, stride)
            todo &= ~sel
    return out
