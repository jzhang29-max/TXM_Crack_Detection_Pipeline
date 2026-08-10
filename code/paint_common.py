"""
Shared state/helpers for the TXM manual-correction paint tool.

Design, adapted from the CBS SEM project's interior_active_learning paint
tool (../../CBS_Crack_Detection_Pipeline/interior_active_learning/code/
paint_server.py + paint_frontend.py) but simplified for TXM's shape of
problem: there, corrections operate on a discrete list of candidate
regions from a region-classifier; here the pixel classifier already
outputs a dense mask directly, so a "correction" is just "force this pixel
to crack" or "force this pixel to not-crack" -- no candidate list, no
separate artifact class (a TXM pixel is either crack or it isn't).

Persistent state per image (under ../paint/):
  predicted_cache/<name>_mask.npy   bool (H,W)   -- the model's raw prediction
                                                     (post-processed, pre-correction),
                                                     cached because generating it
                                                     can take up to ~30s for the
                                                     largest images
  predicted_cache/<name>_img.npy    float32 (H,W) -- normalized grayscale, for
                                                      rendering the template
  corrections/<name>_correction.npy uint8 (H,W)  -- 0=none, 1=force crack,
                                                      2=force not-crack
  corrections/<name>_painted.png    RGB           -- raster snapshot of the
                                                      last saved paint session,
                                                      used to resume painting
                                                      and to diff out new
                                                      strokes (same technique
                                                      as the CBS tool)

Saving corrections regenerates the deliverable outputs directly into
../results/corrected/.
"""

import glob
import os
import sys

import numpy as np
import tifffile
from PIL import Image
from skimage.measure import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_pixel_model import (
    predict_probability_map,
    postprocess_mask,
    save_outputs,
)
from txm_features import robust_normalize

import joblib

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# --- Which model + which input the paint tool serves -------------------
# USE_FLATFIELD switches the tool to the flatfielded pipeline: flatfielded
# input images AND the flatfielded-trained model. Both must move together --
# serving raw pixels to a flatfielded-trained model (or vice versa) is an
# input-distribution mismatch that produces worse results than either
# combination alone, which is precisely the trap the raw-trained model fell
# into on the new specimen groups (raw median brightness varies 2.6x across
# them, so "broad dark = crack" misfired and an UNDAMAGED specimen came out
# at 41% crack).
#
# Correction files stay valid across the switch: flatfielding is a per-pixel
# intensity correction and moves nothing geometrically, so a mask drawn on
# the raw image is still pixel-aligned to the flatfielded one. Shapes are
# asserted in get_state rather than assumed.
USE_FLATFIELD = os.environ.get("TXM_PAINT_RAW", "") == ""   # set TXM_PAINT_RAW=1 to go back to raw

if USE_FLATFIELD:
    MODEL_PATH = os.path.join(PROJECT_DIR, "models", "pixel_flatfield_final.joblib")
    PREDICTED_CACHE_DIR = os.path.join(PROJECT_DIR, "paint", "predicted_cache_flatfield")
else:
    MODEL_PATH = os.path.join(PROJECT_DIR, "models", "pixel_hgb_final.joblib")
    PREDICTED_CACHE_DIR = os.path.join(PROJECT_DIR, "paint", "predicted_cache")
CORRECTIONS_DIR = os.path.join(PROJECT_DIR, "paint", "corrections")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "corrected")

for d in (PREDICTED_CACHE_DIR, CORRECTIONS_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# Folders to offer in the image picker, searched RECURSIVELY (the dataset
# is organized one subfolder per specimen type).
#
# Points at the RAW images, deliberately, not the flatfielded ones:
# measured directly on 338_13 (the one specimen present in both sets), the
# production model scores IoU 0.789 on raw vs 0.421 on flatfielded, and
# over-predicts crack area 28% -> 52% on flatfielded. That's expected
# rather than a bug -- the single most important feature group is
# large-radius smoothed intensity (~41% of total importance, i.e. "is this
# pixel inside a broad dark region"), and flatfielding's whole purpose is
# to remove broad illumination trends, so it erases what those features
# encode. A flatfielded-trained model is a separate, legitimate option
# (it may well fix LARGE's vignetting artifact), but it needs its own
# training run -- the current model cannot be pointed at flatfielded data.
#
# Correcting on raw loses nothing either way: corrections are pixel masks
# and the flatfielded images are pixel-identical in shape, so any
# correction made here is equally valid ground truth for training a
# flatfielded model later.
SOURCE_DIRS = [
    "/Users/jiamingzhang/Desktop/TXM DATA",
]
EXCLUDE_SUBSTRINGS = ("Result of", "Probabilities", "gaussian")

RED = np.array([255, 0, 0])
CYAN = np.array([0, 204, 255])
COLOR_TOLERANCE = 40

_model = None
_mem_cache = {}  # name -> dict(img01, predicted_mask, correction)


_model_mtime = None


def _invalidate_stale_predictions():
    """Auto-detect a swapped-in model file and drop every cached prediction
    (in-memory AND on-disk) so the next request recomputes fresh -- this is
    what makes 'swap a verified model into production' immediately visible
    in the running paint tool, with no manual cache-clear or server restart.
    Cheap to call on every request: just one stat() syscall in the common
    case (model unchanged).

    Deliberately does NOT auto-detect a change to apply_pixel_model.py's
    postprocess_mask code itself (e.g. a hysteresis-logic edit) -- only the
    model file's mtime is tracked, since that's what actually gets swapped
    as part of the retrain/verify/deploy cycle. A code change to the
    pipeline itself should still go through a deliberate cache clear, the
    same as it always has.
    """
    global _model_mtime
    if not os.path.exists(MODEL_PATH):
        return
    mtime = os.path.getmtime(MODEL_PATH)
    if _model_mtime is not None and mtime == _model_mtime:
        return  # unchanged, nothing to do

    changed = _model_mtime is not None  # False on first-ever load, not a "change"
    _model_mtime = mtime
    global _model
    _model = None
    _mem_cache.clear()
    if changed:
        removed = 0
        for f in glob.glob(os.path.join(PREDICTED_CACHE_DIR, "*_mask.npy")):
            os.remove(f)
            removed += 1
        print(f"[paint_common] Detected a new model file ({MODEL_PATH}) -- "
              f"invalidated {removed} cached prediction(s), will recompute on next request.")


def get_model():
    global _model
    _invalidate_stale_predictions()
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def list_images():
    """Returns [{name, path, group}] for every eligible raw TIFF found
    RECURSIVELY under SOURCE_DIRS. `group` is the immediate subfolder name
    (the specimen type, e.g. "B2 316L H Tension"), used to label images in
    the picker -- the dataset is organized one subfolder per specimen, and
    the filenames alone don't say which specimen they belong to."""
    out = []
    seen = set()
    for d in SOURCE_DIRS:
        if not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "**", "*.tif"), recursive=True)):
            fn = os.path.basename(path)
            if any(s in fn for s in EXCLUDE_SUBSTRINGS):
                continue
            name = os.path.splitext(fn)[0]
            if name in seen:
                continue
            seen.add(name)
            rel = os.path.relpath(os.path.dirname(path), d)
            out.append({"name": name, "path": path,
                        "group": "" if rel == "." else rel})
    return out


def _find_path(name):
    for info in list_images():
        if info["name"] == name:
            return info["path"]
    raise FileNotFoundError(f"No source image registered for '{name}'")


def _cache_paths(name):
    return (
        os.path.join(PREDICTED_CACHE_DIR, f"{name}_mask.npy"),
        os.path.join(PREDICTED_CACHE_DIR, f"{name}_img.npy"),
    )


def get_state(name):
    """Loads (from memory -> disk -> fresh computation, in that order) and
    returns the mutable per-image state dict: {img01, predicted_mask, correction}.
    The SAME dict instance is cached and mutated in place so repeated calls
    within one server process are cheap.

    Checks for a swapped-in model FIRST, unconditionally -- not just inside
    get_model() -- because an image whose prediction is already cached on
    disk never calls get_model() at all on a cache hit, so that alone would
    silently keep serving a stale prediction from the old model forever.
    """
    _invalidate_stale_predictions()

    if name in _mem_cache:
        return _mem_cache[name]

    mask_path, img_path = _cache_paths(name)
    if os.path.exists(mask_path) and os.path.exists(img_path):
        predicted_mask = np.load(mask_path)
        img01 = np.load(img_path)
    else:
        path = _find_path(name)
        if USE_FLATFIELD:
            # Serve the FLATFIELDED image, matching what the flatfielded
            # model was trained on. Fall back to raw only if this image has
            # no flatfielded counterpart, and say so rather than silently
            # feeding mismatched input to the model.
            import build_flatfield_dataset as _bf
            ffp = _bf.flatfield_path_for(path)
            if ffp is None:
                print(f"[paint_common] WARNING: no flatfielded counterpart for {name} -- "
                      f"falling back to RAW input, which mismatches the flatfielded model. "
                      f"Predictions for this image will be unreliable.")
            else:
                path = ffp
        raw = tifffile.imread(path).astype(np.float64)
        img01 = robust_normalize(raw, 1.0, 99.0)
        model = get_model()
        prob_map = predict_probability_map(model, img01)
        predicted_mask = postprocess_mask(prob_map)
        np.save(mask_path, predicted_mask)
        np.save(img_path, img01)

    correction_path = os.path.join(CORRECTIONS_DIR, f"{name}_correction.npy")
    if os.path.exists(correction_path):
        correction = np.load(correction_path)
    else:
        correction = np.zeros(predicted_mask.shape, dtype=np.uint8)

    state = {"img01": img01, "predicted_mask": predicted_mask, "correction": correction}
    _mem_cache[name] = state
    return state


def effective_mask(state):
    m = state["predicted_mask"].copy()
    m[state["correction"] == 1] = True
    m[state["correction"] == 2] = False
    return m


def build_template(name):
    state = get_state(name)
    mask = effective_mask(state)
    gray = (np.clip(state["img01"], 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[mask] = RED
    return Image.fromarray(rgb, mode="RGB")


def save_correction(name):
    state = get_state(name)
    np.save(os.path.join(CORRECTIONS_DIR, f"{name}_correction.npy"), state["correction"])


def color_mask(painted, template, color):
    """A pixel counts as freshly painted `color` if it's close to `color` in
    the saved composite AND wasn't already that color in the template --
    same technique as the CBS tool's apply_paint_annotations._color_mask."""
    dist = np.linalg.norm(painted.astype(np.float32) - color, axis=-1)
    mask = dist < COLOR_TOLERANCE
    template_dist = np.linalg.norm(template.astype(np.float32) - color, axis=-1)
    mask &= template_dist >= COLOR_TOLERANCE
    return mask


def apply_paint_layer(name, painted_rgb):
    """painted_rgb: the composited (template + user strokes) RGB array, same
    shape as the template. Diffs it against the CURRENT (pre-this-save)
    template to find freshly-painted red (-> force crack) / cyan (-> force
    not-crack) pixels, merges them into the persisted correction array,
    regenerates the deliverable outputs, and returns a small summary dict."""
    state = get_state(name)
    old_template = np.array(build_template(name))

    if painted_rgb.shape != old_template.shape:
        raise ValueError(f"painted layer shape {painted_rgb.shape} != old_template.shape {old_template.shape}")

    new_red = color_mask(painted_rgb, old_template, RED)
    new_cyan = color_mask(painted_rgb, old_template, CYAN)

    n_added = int(new_red.sum())
    n_removed = int(new_cyan.sum())

    state["correction"][new_red] = 1
    state["correction"][new_cyan] = 2
    save_correction(name)

    regenerate_outputs(name)
    resync_painted_snapshot(name)
    return {"pixels_added": n_added, "pixels_removed": n_removed}


def resync_painted_snapshot(name):
    """Overwrite the saved painted.png (if any) with the CURRENT template,
    pixel for pixel. Call this right after any server-side change to the
    correction array (a brush save, or a click-to-remove flip) -- at that
    moment nothing is "pending" anymore, everything just drawn has been
    committed into the correction array and the regenerated outputs, so the
    resume-snapshot should exactly equal the fresh template.

    Without this, painted.png keeps whatever raw stroke color was drawn at
    save time forever. An eraser (cyan) stroke over a crack region commits
    correctly (correction array and outputs are right), but the new
    template shows plain gray there, not cyan -- so the next diff (on
    resume, or the reload that happens right after this very save) sees
    "painted.png is cyan, template isn't" and treats it as a brand new
    stroke, redrawing the eraser mark indefinitely instead of it
    disappearing. The same staleness can happen in the other direction
    after a click-to-remove flip. Making painted.png always equal the
    latest template after every commit means there is never anything left
    over to misdetect: the next diff finds zero differences until the user
    actually paints something new.
    """
    painted_path = os.path.join(CORRECTIONS_DIR, f"{name}_painted.png")
    if os.path.exists(painted_path):
        build_template(name).save(painted_path)


def flip_region(name, x, y):
    """Click-to-remove: if (x,y) lands on a currently-crack pixel, force the
    WHOLE connected component it belongs to (in the effective mask) to
    not-crack in one action. This is the single highest-value tool for the
    scattered false-positive speckle regions on the LARGE image -- clicking
    each one individually is far faster than brushing over every one by
    hand. Clicking on background is a no-op (there's no discrete "candidate"
    to flip the other way -- use the Add brush for that)."""
    state = get_state(name)
    mask = effective_mask(state)
    h, w = mask.shape
    if not (0 <= y < h and 0 <= x < w):
        raise ValueError("click is outside the image")
    if not mask[y, x]:
        return {"changed": False, "message": "No crack region there -- click inside a red region, or use the Add brush."}

    labeled = label(mask, connectivity=2)
    region_label = labeled[y, x]
    region_mask = labeled == region_label
    area = int(region_mask.sum())

    state["correction"][region_mask] = 2
    save_correction(name)
    regenerate_outputs(name)
    resync_painted_snapshot(name)
    return {"changed": True, "area": area}


def regenerate_outputs(name):
    state = get_state(name)
    mask = effective_mask(state)
    save_outputs(OUTPUT_DIR, name, state["img01"], mask)


def region_stats(name):
    state = get_state(name)
    mask = effective_mask(state)
    n_regions = int(label(mask, connectivity=2).max())
    area_fraction = float(mask.mean())
    return {"n_regions": n_regions, "area_fraction": area_fraction}
