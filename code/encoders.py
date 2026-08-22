"""Frozen vision encoders as interchangeable feature sources, for measurement only.

    python3 code/selftest_encoders.py        # the arithmetic, no weights needed
    python3 code/experiment_encoders.py      # the paired comparison

WHY THIS EXISTS. app/core/model.py uses SAM 1 ViT-H as a frozen image encoder and never
calls its prompt encoder or mask decoder -- zero-shot SAM prompted as designed measures
0.23-0.36 IoU here against 0.82 for the trained hybrid. So the only interesting question
about a newer SAM is whether its encoder produces more discriminative DENSE FEATURES. This
module makes the encoder swappable so that question can be measured without touching the
deployed path, which a real swap would invalidate wholesale (both shipped models, every
cached prediction, the reference feature stacks, every published number).

THE THREE ENCODERS LOOK REMARKABLY ALIKE from here, which is what makes a fair test cheap:

    SAM 1  ViT-H          256-ch neck, one 64x64 grid per 1024-px tile  -> stride 16
    SAM 2  Hiera-L        256-ch FPN, 3 levels                          -> stride discovered
    SAM 3  Perception Enc 256-ch FPN, multi-level                       -> stride discovered

All three are 256 channels, so the 273-d feature vector is unchanged and the downstream
classifier is untouched. Only SAM 1's stride is knowable in advance; the others report
`backbone_feature_sizes`/`scale_factors` as None in transformers because those arrive with
the checkpoint, so this module reads the shapes off the tensors it actually receives and
picks the level nearest SAM 1's granularity -- keeping the comparison like-for-like.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (os.path.join(_PROJECT, "app", "core"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as M            # noqa: E402

TARGET_STRIDE = M.EMB_STRIDE           # 16: SAM 1's granularity, the baseline to match
CACHE_ROOT = os.path.join(_PROJECT, "paint")

# repo, transformers model class, processor class, and how to get the feature pyramid out
SPECS = {
    "sam2": dict(repo="facebook/sam2.1-hiera-large", model="Sam2Model",
                 processor="Sam2Processor", call="get_image_embeddings"),
    "sam3": dict(repo="facebook/sam3", model="Sam3Model",
                 processor="Sam3Processor", call="get_vision_features"),
}
_loaded = {}


class EncoderUnavailable(RuntimeError):
    """This encoder cannot be used here, and str(self) says why."""


def availability(name):
    """(ok, reason). Ordered by the sequence a user actually hits."""
    spec = SPECS.get(name)
    if spec is None:
        return False, f"unknown encoder {name!r}; known: {sorted(SPECS)}"
    try:
        import transformers
    except Exception as e:                                        # noqa: BLE001
        return False, f"transformers not importable: {e}"
    if not hasattr(transformers, spec["model"]):
        return False, (f"transformers {transformers.__version__} has no "
                       f"{spec['model']}")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=spec["repo"], filename="config.json")
    except Exception as e:                                        # noqa: BLE001
        return False, (f"{spec['repo']} not downloadable ({type(e).__name__}). If it is "
                       f"gated, accept the licence on https://huggingface.co/{spec['repo']} "
                       f"with your own account and authenticate -- a licence agreement "
                       f"cannot be automated.")
    # The processor is a separate failure mode and it is NOT optional: Sam2ImageProcessor
    # needs torchvision, which this project deliberately does not require -- the app has no
    # use for it. Checking it here turns an ImportError traceback partway through a long run
    # into one line at the start.
    try:
        getattr(transformers, spec["processor"])
        import importlib
        if importlib.util.find_spec("torchvision") is None:
            return False, ("torchvision is not installed, and the image processor for this "
                           "encoder requires it. `pip install torchvision` (it pins the "
                           "torch version already present, so the app is unaffected). It is "
                           "intentionally absent from requirements.txt: the app never needs "
                           "it.")
    except Exception as e:                                        # noqa: BLE001
        return False, f"processor {spec['processor']} unusable: {type(e).__name__}: {e}"
    return True, "ok"


def _load(name):
    if name not in _loaded:
        ok, why = availability(name)
        if not ok:
            raise EncoderUnavailable(why)
        import torch
        import transformers
        spec = SPECS[name]
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        proc = getattr(transformers, spec["processor"]).from_pretrained(spec["repo"])
        # WITH LOADING INFO, because a silent partial load would be catastrophic here and
        # invisible in the results. facebook/sam2.1-hiera-large is a `sam2_video` checkpoint
        # loaded into `Sam2Model`, and transformers only warns. If the VISION ENCODER weights
        # had not come from the checkpoint, its features would be randomly initialised and
        # the comparison would be measuring noise while looking perfectly well-behaved.
        mdl, info = getattr(transformers, spec["model"]).from_pretrained(
            spec["repo"], output_loading_info=True)
        missing = [k for k in (info.get("missing_keys") or [])
                   if "vision" in k or "image_encoder" in k or "backbone" in k]
        if missing:
            raise EncoderUnavailable(
                f"{spec['repo']}: {len(missing)} vision-encoder tensor(s) were NOT loaded "
                f"from the checkpoint and would be random, e.g. {missing[:3]}. Refusing to "
                f"produce features that would look valid and mean nothing.")
        mdl = mdl.to(dev).eval()
        _loaded[name] = (proc, mdl, dev, torch, spec)
        _loaded[f"{name}__loadinfo"] = dict(
            missing=len(info.get("missing_keys") or []),
            unexpected=len(info.get("unexpected_keys") or []))
    return _loaded[name]


def _pyramid(out):
    """Normalise the several shapes these APIs return into a list of 4-D tensors."""
    if hasattr(out, "fpn_hidden_states") and out.fpn_hidden_states is not None:
        return list(out.fpn_hidden_states)
    if isinstance(out, (list, tuple)):
        return [t for t in out if hasattr(t, "shape") and len(t.shape) == 4]
    if hasattr(out, "last_hidden_state"):
        return [out.last_hidden_state]
    raise EncoderUnavailable(f"cannot find feature maps in {type(out).__name__}")


def _to_bchw(t, n_ch):
    """Channels-first, whichever way the model hands them over."""
    if t.shape[1] == n_ch:
        return t
    if t.shape[-1] == n_ch:
        return t.permute(0, 3, 1, 2)
    # fall back to the conventional layout rather than guessing from sizes
    return t


def _pick_level(maps, tile_px, n_ch):
    """(index, stride) of the level closest to SAM 1's stride, ignoring degenerate levels."""
    best = (None, None, None)
    for i, t in enumerate(maps):
        tt = _to_bchw(t, n_ch)
        h = int(tt.shape[-2])
        if h <= 1:
            continue
        stride = tile_px / h
        err = abs(stride - TARGET_STRIDE)
        if best[1] is None or err < best[1]:
            best = (i, err, stride)
    if best[0] is None:
        raise EncoderUnavailable("encoder returned no usable feature level")
    return best[0], best[2]


def embed_image(name, img01, progress=None):
    """(coords int32 [n,2], emb float16 [n,C,h,w], stride) using model.tiles' exact tiling."""
    proc, mdl, dev, torch, spec = _load(name)
    n_ch = getattr(getattr(mdl.config, "vision_config", mdl.config), "fpn_hidden_size", 256)
    tl = M.tiles(img01.shape)
    coords, embs, stride, level = [], [], None, None
    for k, (y0, y1, x0, x1) in enumerate(tl):
        crop = img01[y0:y1, x0:x1]
        if crop.shape != (M.TILE, M.TILE):
            crop = np.pad(crop, ((0, M.TILE - crop.shape[0]), (0, M.TILE - crop.shape[1])),
                          mode="reflect")
        u8 = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([u8] * 3, -1)
        px = proc(images=rgb, return_tensors="pt")["pixel_values"]
        px = (px.float() if px.dtype == torch.float64 else px).to(dev)
        with torch.no_grad():
            out = getattr(mdl, spec["call"])(pixel_values=px)
        maps = _pyramid(out)
        if level is None:
            level, stride = _pick_level(maps, M.TILE, n_ch)
        t = _to_bchw(maps[level], n_ch)
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
    Identical arithmetic with the stride passed in -- selftest_encoders asserts the two agree
    bit-for-bit at stride 16, which they must, being the same function there.
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


def cached_embedding(name, key, img01, progress=None):
    """Embed once per (encoder, image); these are far too slow to recompute per fold."""
    d = os.path.join(CACHE_ROOT, f"{name}_embcache")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{key}.npz")
    if os.path.exists(p):
        try:
            z = np.load(p)
            return z["coords"], z["emb"], float(z["stride"])
        except Exception:                                         # noqa: BLE001
            os.remove(p)
    coords, emb, stride = embed_image(name, img01, progress=progress)
    # np.savez APPENDS ".npz" unless the name already ends in it, so a temp name like
    # "x.npz.1234.tmp" is written as "x.npz.1234.tmp.npz" and the rename then fails on a
    # path that was never created. Keep the suffix last.
    tmp = f"{p}.{os.getpid()}.tmp.npz"
    np.savez(tmp, coords=coords, emb=emb, stride=np.float32(stride))
    os.replace(tmp, p)
    return coords, emb, stride


def rows_at(coords, emb, stride, rr, cc):
    """[n, C] block for pixel coords rr, cc -- last tile wins, as app/core/model.py does."""
    out = np.zeros((len(rr), emb.shape[1]), np.float32)
    todo = np.ones(len(rr), bool)
    for t in range(len(coords) - 1, -1, -1):
        y0, x0 = int(coords[t][0]), int(coords[t][1])
        sel = (todo & (rr >= y0) & (rr < y0 + M.TILE) & (cc >= x0) & (cc < x0 + M.TILE))
        if sel.any():
            out[sel] = interp_tile(emb[t], rr[sel] - y0, cc[sel] - x0, stride)
            todo &= ~sel
    return out
