"""
The deployed crack model, as ONE object with ONE method.

Everything the app does goes through `CrackModel.predict(img01)`. The app never
needs to know that this is really two models averaged, that one of them needs a
SAM ViT forward pass, or how the 273-dim feature vector is assembled -- which is
the point: the frontend can offer "predict", "correct", "retrain" as buttons
because all of that complexity is behind this class.

WHY AN ENSEMBLE RATHER THAN JUST THE SAM HYBRID. Cross-validation grouped by whole image
over all 71 labelled images -- train and test never share a frame -- which is the only split
this data supports honestly:

  approach                      mean IoU   per fold
  17 hand-crafted features        0.651     0.685 0.664 0.672 0.591 0.645
  SAM 256 + 17 (the hybrid)       0.778     0.773 0.716 0.803 0.787 0.810
  mean probability of the two     0.792     0.790 0.738 0.817 0.799 0.816

Averaging wins in EVERY fold, not on average -- which is the property worth having, because a
mean can be carried by one fold. It costs about 25% more inference time than the hybrid alone.
The 17-feature member is much weaker on its own here than the number this docstring used to
quote (0.651 against 0.744), and that is the point of the change of basis rather than a
regression: the old figure came from leave-one-image-out over 4 externally-labelled frames
(17 alone 0.744, hybrid 0.795, ensemble 0.821, crack-free FP 7.43% / 0.14% / 0.11%). Those
labels came from another tool and are used nowhere in this project now, so that table is why
the ensemble was originally chosen and this one is why it stays. crossval_on_rows records the
per-member split on every retrain, so this cannot go stale again.

The deployed ensemble holds IoU 0.789 +-0.039 with 0.209% predicted area on the 6
owner-confirmed crack-free specimens. That 0.789 and the 0.792 above differ because the two
runs resampled different rows; the deployed figure is the one the gate recorded.

A third member (SAM-only) was tested and adds +0.009 IoU, which is below the
measured 0.0070 retrain-noise floor, so it is not included -- it would cost a
third of the inference time for an effect indistinguishable from reseeding.

Set ensemble=False to run the hybrid alone (about 2x faster, measurably worse).
"""

import os
import sys
import threading
import zipfile
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CODE = os.path.join(_PROJECT, "code")
for p in (_CODE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

TILE = 1024
EMB_STRIDE = 16

# How far apart tile origins sit when embedding. Below TILE, adjacent tiles OVERLAP, and
# that overlap is the whole reason the embedding field can be made continuous: with
# TILE_STRIDE == TILE the tiles abut, a 3914 px frame sits at x = 0, 1024, 2048, 2890, and
# every boundary but the last has no shared data at all -- so the embedding STEPS across it.
# Measured on a 3044x2354 frame, the worst row carried 32.8x the median row-to-row
# probability change, all of it at the y=1023 tile boundary. At stride 896 the worst row is
# 7.8x and sits at y=677, which is crack, not a boundary. Lookup-time fixes were tried first
# and both failed: reaching past a tile's edge has to invent data and raised false positives
# on crack-free specimens 6.2x, while a window that stops at the edge cannot fix anything
# where only one tile covers the pixel -- it reduces to last-tile-wins exactly. Only real
# overlap works. Costs 13% more tiles (1190 vs 1050 over the corpus, ~37 min to rebuild).
TILE_STRIDE = 896
# THE SHIPPED MODEL: both members fitted on the owner's corrections and nothing else.
#
# Every earlier model in this project inherited its 17-feature member from a research-phase
# artifact trained on four pre-existing masks made with a different tool. Because the model
# is a mean-probability ensemble, that meant half of every prediction came from labels the
# owner did not draw. Both halves are now trained together, from the owner's own labelling of
# all 71 images, and no external label is used anywhere -- not in training, not in the gate.
#
# Measured under grouped-by-image cross-validation (train and test never share an image):
# IoU 0.776, sd 0.029, worst fold 0.733, precision 0.929, recall 0.824. On the six specimens
# confirmed to contain no crack: 0.188% of area predicted crack, 2.83 false indications per
# frame -- both better than the model it replaced (0.250%, 4.0).
DEFAULT_17 = os.path.join(_PROJECT, "models", "f17_v5_20260824.joblib")
DEFAULT_HYBRID = os.path.join(_PROJECT, "models", "hybrid_v5_20260824.joblib")
SAM_MODEL_ID = "facebook/sam-vit-huge"


# ---------------------------------------------------------------- SAM embedding
_sam = None


class SamUnavailable(RuntimeError):
    """SAM cannot be used in this process, and the reason is in str(self).

    Its own exception type because the caller has a real fallback -- predict with the
    17-feature model alone -- and needs to distinguish "SAM is not reachable" from a bug
    inside the embedding pass, which must still surface as an error.
    """


# Latched, not retried per image. A machine behind a firewall would otherwise pay one
# failed 2.4 GB download attempt per image, 71 times, before showing the same message.
sam_unavailable_reason = None


def sam_device(torch):
    """Which device to run the SAM encoder on: cuda -> mps -> cpu, unless overridden.

    TXM_SAM_DEVICE exists because the MPS path is not always trustworthy, and that is
    measured rather than defensive. On a GitHub macos-26-arm64 runner -- where Metal IS
    available, contrary to what the CI header first claimed -- the shipped ensemble produced a
    predicted area of 0.0925 on the reference frame against 0.1880 everywhere else, with a
    probability map of visibly lower confidence (mean 0.369 against 0.381, std 0.264 against
    0.300, six times as much mass within 0.05 of the decision threshold) and a mask shattered
    into 21,020 components instead of 968.

    Everything else was ruled out first, by measurement, not by elimination-by-assumption:
      - the decoded input is bit-identical there (same shape, same sha1, same statistics)
      - a doubled denominator would print 0.0940, not 0.0925
      - the BLAS is the same -- the stock PyPI arm64 wheels link Accelerate on both, and the
        predict path is two small GEMM stacks and otherwise BLAS-free
      - the tile layout is four exact 1024x1024 crops, so the reflect-pad path is never
        entered and the Hann blend is a weighted mean that coverage cannot bias
      - the SamProcessor backend differs between this project's Mac and BOTH runners, and the
        Linux runner reproduces 0.1880 with the same backend the macOS one used, so it cannot
        be the distinguishing variable. Measured directly: 9 flipped pixels out of 2,857,784.
    The one remaining difference is that the macOS runner ran the encoder on MPS and the Linux
    runner on CPU. On real Apple silicon the two agree closely -- measured, embeddings differ
    by at most one float16 storage quantum (4.9e-4), ZERO pixels change side of the 0.60
    threshold, and the predicted area is 0.187972 either way -- so this is a
    property of some GPU/driver stacks and not of MPS as such -- which is exactly why it needs
    an override rather than a blanket ban.

    Values: "auto" (default), "cpu", "mps", "cuda". An explicit device that is unavailable
    falls back with a warning rather than failing the image -- a wrong device name should not
    cost someone their ingest.
    """
    # MPS BEFORE CUDA, preserving the order this had before the override existed. Docs
    # elsewhere describe it as "cuda -> mps -> cpu"; the code has always checked mps first.
    # No machine has both, so the difference is unreachable in practice -- but flipping it
    # while adding an override would be an unreviewed behaviour change smuggled in beside a
    # documented one.
    want = os.environ.get("TXM_SAM_DEVICE", "auto").strip().lower()
    have = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
            "cpu": True}
    auto = "mps" if have["mps"] else "cuda" if have["cuda"] else "cpu"
    if want in ("", "auto"):
        return auto
    if want not in have:
        print(f"  TXM_SAM_DEVICE={want!r} is not a device name; using auto", flush=True)
        return auto
    if not have[want]:
        print(f"  TXM_SAM_DEVICE={want} requested but unavailable; using cpu", flush=True)
        return "cpu"
    return want


def sam_disabled_by_env():
    return os.environ.get("TXM_NO_SAM", "").strip().lower() in ("1", "true", "yes")


def _get_sam():
    """Load SAM once per process. Downloads ~2.4 GB on first ever use.

    Raises SamUnavailable instead of a bare ImportError/OSError, so ingest can degrade to
    the 17-feature model rather than failing the image. run_app.sh has always told users
    the app "falls back to the 17-feature model alone" when SAM is missing; until this
    existed that promise was false, and a researcher on a network that blocks
    huggingface.co got a red job error on every single image with no way to reach the
    17-feature model sitting in models/f17_v5_20260824.joblib.
    """
    global _sam, sam_unavailable_reason
    if sam_unavailable_reason:
        raise SamUnavailable(sam_unavailable_reason)
    if sam_disabled_by_env():
        sam_unavailable_reason = "disabled by TXM_NO_SAM=1"
        raise SamUnavailable(sam_unavailable_reason)
    if _sam is None:
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except Exception as e:                                  # noqa: BLE001
            sam_unavailable_reason = f"{type(e).__name__}: {e}".split("\n")[0][:180]
            raise SamUnavailable(sam_unavailable_reason) from e
        try:
            dev = sam_device(torch)
            proc = SamProcessor.from_pretrained(SAM_MODEL_ID)
            model = SamModel.from_pretrained(SAM_MODEL_ID).to(dev).eval()
        except Exception as e:                                  # noqa: BLE001
            sam_unavailable_reason = f"{type(e).__name__}: {e}".split("\n")[0][:180]
            raise SamUnavailable(sam_unavailable_reason) from e
        _sam = (proc, model, dev, torch)
    return _sam


def tiles(shape, size=TILE, stride=None):
    """Tiles of exactly `size`, stepped by `stride` and clamped inward at the edges.

    Clamping inward (rather than padding) keeps the pixel->embedding mapping a clean
    divide-by-16 with no padding to reason about. `stride` defaults to `size`, which makes
    the tiles abut -- see TILE_STRIDE for why that produces visible seams and why embedding
    passes should hand in a smaller stride so adjacent tiles overlap.
    """
    H, W = shape[:2]
    st = size if stride is None else int(stride)
    ys = sorted({max(min(y0 + size, H) - size, 0) for y0 in range(0, max(H - 1, 1), st)})
    xs = sorted({max(min(x0 + size, W) - size, 0) for x0 in range(0, max(W - 1, 1), st)})
    return [(y, min(y + size, H), x, min(x + size, W)) for y in ys for x in xs]


def embed_image(img01, progress=None):
    """Tiled SAM ViT embeddings -> (coords int32 [n,2], emb float16 [n,C,64,64])."""
    proc, model, dev, torch = _get_sam()
    tl = tiles(img01.shape, stride=TILE_STRIDE)
    coords, embs = [], []
    for k, (y0, y1, x0, x1) in enumerate(tl):
        crop = img01[y0:y1, x0:x1]
        if crop.shape != (TILE, TILE):
            crop = np.pad(crop, ((0, TILE - crop.shape[0]), (0, TILE - crop.shape[1])),
                          mode="reflect")
        u8 = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([u8] * 3, -1)
        inp = proc(rgb, return_tensors="pt")
        px = inp["pixel_values"]
        px = (px.float() if px.dtype == torch.float64 else px).to(dev)
        with torch.no_grad():
            e = model.get_image_embeddings(px)
        embs.append(e.float().cpu().numpy()[0].astype(np.float16))
        coords.append((y0, x0))
        if dev == "mps":
            torch.mps.empty_cache()
        if progress:
            progress(k + 1, len(tl))
    return np.asarray(coords, np.int32), np.stack(embs)


def interp_tile(emb_tile, rr, cc):
    """Bilinear lookup of tile-local coords in one C x 64 x 64 grid, vectorised
    across all channels in a single gather."""
    e = np.ascontiguousarray(emb_tile, dtype=np.float32)
    C, H, W = e.shape
    r = np.clip(rr / EMB_STRIDE - 0.5, 0, H - 1)
    c = np.clip(cc / EMB_STRIDE - 0.5, 0, W - 1)
    r0 = np.floor(r).astype(np.intp); c0 = np.floor(c).astype(np.intp)
    r1 = np.minimum(r0 + 1, H - 1);   c1 = np.minimum(c0 + 1, W - 1)
    dr = (r - r0).astype(np.float32)[:, None]
    dc = (c - c0).astype(np.float32)[:, None]
    f = e.reshape(C, H * W)
    return (f[:, r0 * W + c0].T * (1 - dr) * (1 - dc)
            + f[:, r0 * W + c1].T * (1 - dr) * dc
            + f[:, r1 * W + c0].T * dr * (1 - dc)
            + f[:, r1 * W + c1].T * dr * dc)


def emb_rows(coords, embs, rr, cc):
    """One embedding vector per (rr, cc) pixel, blended over every tile containing it.

    THE ONE PLACE the pixel -> embedding mapping is defined. Inference and training-row
    assembly both call this, and they have to: fit on one lookup and predict with another and
    the model sees a feature distribution it never trained on. That mismatch is measurable --
    serving blended embeddings to weights fitted on last-tile-wins raised false positives on
    a crack-free specimen from 0.019% to 0.080%.

    The weight is a Hann window over each tile's OWN extent, so it falls to ~2e-6 at that
    tile's edge. Inside a tile's interior its own embedding dominates; across an overlap band
    the two tiles trade off smoothly; nothing is ever read from outside a tile. A tile's
    weight MUST vanish at the edge of its support or the field jumps wherever a tile enters
    or leaves the blend -- an earlier version added a 1e-3 floor to keep the sum comfortably
    positive and that floor alone put a 0.4999 step back in, because along the frame's top row
    the window is ~0 for every tile, so the floor dominated and made it a flat unweighted
    average. Without the floor that row is still correct: the shared row factor cancels in
    the normalisation, leaving the right 1-D blend across x. The window never reaches exactly
    zero inside the support (2.35e-6 at the first and last pixel), so the sum stays positive.
    """
    C = embs.shape[1]
    acc = np.zeros((len(rr), C), np.float32)
    wsum = np.zeros(len(rr), np.float32)
    for t in range(len(coords)):
        y0, x0 = int(coords[t][0]), int(coords[t][1])
        ly = rr - y0
        lx = cc - x0
        sel = (ly >= 0) & (ly < TILE) & (lx >= 0) & (lx < TILE)
        if not sel.any():
            continue
        lys, lxs = ly[sel], lx[sel]
        wy = 0.5 * (1.0 - np.cos(2 * np.pi * (lys + 0.5) / TILE))
        wx = 0.5 * (1.0 - np.cos(2 * np.pi * (lxs + 0.5) / TILE))
        w = (wy * wx).astype(np.float32)
        acc[sel] += interp_tile(embs[t], lys, lxs) * w[:, None]
        wsum[sel] += w
    return acc / np.maximum(wsum, 1e-12)[:, None]


def emb_is_current(path):
    """Is this cache usable as-is? Reads the stride tag only, not the 8-59 MB embedding.

    False for a missing file, an unreadable one, or one built when tiles still abutted: a
    zero-overlap cache cannot be blended, so serving from it would silently reintroduce the
    seams it was rebuilt to remove.
    """
    if not os.path.exists(path):
        return False
    try:
        z = np.load(path)
        return "stride" in z.files and int(z["stride"]) == TILE_STRIDE
    except (zipfile.BadZipFile, ValueError, EOFError, KeyError, OSError):
        return False


def read_emb(path, note=None):
    """(coords, embs) from a cache, or None if it has to be rebuilt.

    A missing file, a truncated one and a stale-stride one all mean the same thing to every
    caller -- re-embed -- so they collapse to None here rather than each caller repeating the
    three checks. `note` is called with a short reason when a cache is rejected, so the app
    can say why it is spending a minute on SAM instead of appearing to hang.
    """
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        if "stride" not in z.files:
            if note:
                note("SAM cache predates overlapping tiles, recomputing")
            return None
        if int(z["stride"]) != TILE_STRIDE:
            if note:
                note(f"SAM cache built at stride {int(z['stride'])}, need {TILE_STRIDE}")
            return None
        return z["coords"], z["emb"]
    except (zipfile.BadZipFile, ValueError, EOFError, KeyError, OSError) as e:
        if note:
            note(f"SAM cache damaged ({type(e).__name__}), recomputing")
        return None


def write_emb(path, coords, embs):
    """Write an embedding cache atomically, tagged with the stride it was built at.

    Atomic because a SAM embedding is 8-59 MB: quit or lose power partway through a bare
    np.savez onto the final path and the file is a truncated zip forever. That used to BRICK
    the image with no route back from the UI -- Re-apply calls ingest() without force so it
    reused the broken file, re-dropping produced the same content-hash id, and the only
    escape discarded that image's corrections. The same corrupt file then killed the next
    retrain an hour in. Tagged so read_emb() can tell a blendable cache from an abutting one.
    """
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, coords=coords, emb=embs, stride=np.int32(TILE_STRIDE))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------- the model
class CrackModel:
    def __init__(self, path_17=DEFAULT_17, path_hybrid=DEFAULT_HYBRID, ensemble=True):
        import joblib
        self.ensemble = bool(ensemble)
        self.m17 = joblib.load(path_17) if os.path.exists(path_17) else None
        self.hybrid, self.n_hybrid = None, None
        if os.path.exists(path_hybrid):
            b = joblib.load(path_hybrid)
            self.hybrid = b["model"] if isinstance(b, dict) else b
            self.n_hybrid = (b.get("n_features", 273) if isinstance(b, dict) else 273)
        if self.m17 is None and self.hybrid is None:
            raise FileNotFoundError(
                "no model found -- expected models/f17_v5_20260824.joblib and/or "
                "models/hybrid_v5_20260824.joblib")
        if self.ensemble and (self.m17 is None or self.hybrid is None):
            self.ensemble = False   # fall back rather than silently averaging one thing

    def describe(self):
        parts = []
        if self.m17 is not None:
            parts.append("17-feature MLP")
        if self.hybrid is not None:
            parts.append(f"SAM+17 hybrid ({self.n_hybrid}d)")
        mode = "mean-probability ensemble" if self.ensemble else "single model"
        return f"{mode}: {' + '.join(parts)}"

    def needs_sam(self):
        return self.hybrid is not None

    def predict(self, img01, emb=None, band=128, progress=None):
        """Crack probability map for a normalised image.

        `emb` is (coords, embeddings) from embed_image(); computed here if not
        supplied. Chunked by tile and row band -- a 23.5 MP image at 273 float32
        features would be 26 GB if materialised at once.
        """
        from txm_features import compute_feature_stack
        f17 = compute_feature_stack(img01)
        H, W = img01.shape

        p17 = None
        if self.m17 is not None:
            p17 = np.zeros((H, W), np.float32)
            for r0 in range(0, H, 256):
                r1 = min(r0 + 256, H)
                blk = np.asarray(f17[r0:r1], np.float32).reshape(-1, f17.shape[2])
                p17[r0:r1] = self.m17.predict_proba(blk)[:, 1].reshape(r1 - r0, W)
            if progress:
                progress("17-feature model", 1, 1)

        ph = None
        if self.hybrid is not None:
            if emb is None:
                emb = embed_image(img01, progress=(lambda k, n: progress("SAM embedding", k, n))
                                  if progress else None)
            coords, embs = emb
            ph = np.zeros((H, W), np.float32)
            # Walks the OUTPUT in blocks rather than walking the tiles, because a blended
            # embedding needs every tile covering a pixel at once -- the old tile-by-tile
            # loop with a `done` mask was last-tile-wins by construction, which is the seam.
            # Block size is unchanged, so the peak footprint is too: 128 x 1024 pixels at 273
            # float32 features is 143 MB, against 26 GB for a 23.5 MP frame materialised whole.
            for b0 in range(0, H, band):
                b1 = min(b0 + band, H)
                for c0 in range(0, W, TILE):
                    c1 = min(c0 + TILE, W)
                    rr = np.repeat(np.arange(b0, b1), c1 - c0)
                    cc = np.tile(np.arange(c0, c1), b1 - b0)
                    X = np.concatenate([np.asarray(f17[rr, cc, :], np.float32),
                                        emb_rows(coords, embs, rr, cc)], axis=1)
                    ph[b0:b1, c0:c1] = self.hybrid.predict_proba(X)[:, 1] \
                        .astype(np.float32).reshape(b1 - b0, c1 - c0)
                if progress:
                    progress("hybrid model", b1, H)
        del f17

        if self.ensemble and p17 is not None and ph is not None:
            return (p17 + ph) / 2.0
        return ph if ph is not None else p17


def postprocess(prob):
    """The project's standard mask cleanup. Kept behind a function so the app can
    offer it as a toggle: it is under suspicion of deleting thin hand-painted
    crack (raw-threshold stroke recall 0.869 vs 0.14-0.40 post-processed), which
    is unresolved."""
    from apply_pixel_model import postprocess_mask
    return postprocess_mask(prob)
