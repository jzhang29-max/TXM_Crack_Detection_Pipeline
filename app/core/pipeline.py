"""
The per-image pipeline, and the retrain job. Everything the frontend's buttons
actually trigger.

    ingest(image_id)   original -> normalised model input
                                -> destitched+flatfielded display view
                                -> cached SAM embedding
                                -> crack probability
    retrain()          gather every correction -> train a new hybrid ->
                       validate on both axes -> deploy or refuse

The validation gate on retrain is not optional decoration. This project has
adopted three changes that later proved to be regressions, and every one of them
passed a SINGLE metric: pseudo-flat-fielding looked good on false positives and
cost 0.169 IoU; a curvilinearity gate cut predicted area 8x, which reads as
artifact removal, while destroying 98% of the true crack on one image. An
over-aggressive filter and a good one BOTH reduce area -- only recall against
ground truth separates them. So a candidate must hold IoU AND not increase false
positives on known-clean specimens, or it is not deployed.
"""

import glob
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
CODE = os.path.join(PROJECT, "code")
for p in (CODE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import store as S            # noqa: E402
import model as M            # noqa: E402

# NO EXTERNAL LABELS, ANYWHERE.
#
# This project used to ship four densely-annotated B2 frames as pixel-level ground truth and
# train on them as well as grade against them, so the gate was marking its own work with part
# of the answer key in the training set. Measured cost of dropping them from training (3
# repeats, leave-one-group-out over AM/HC, B2, B3, wrought): cross-group AUC 0.871 without
# them against 0.863 with -- a difference at the +-0.008 noise floor, i.e. nothing. The
# contamination it removed was worth 0.207 IoU (0.921 on frames trained on, 0.714 held out).
#
# They are not a test set either, now: those labels came from another tool and are not used
# for training or for scoring. All 71 labelled images train, and generalisation is estimated
# by grouped-by-image cross-validation, where train and test never share an image -- the only
# split this data supports honestly. The four frames themselves are ordinary training images,
# labelled by the owner like every other.

# Tag on every model the retrainer produces, so the gate can tell whether the model it is
# comparing against was trained under the same rules. A model trained WITH the reference
# frames scores them in-sample; comparing a clean candidate to it on those frames would
# reject the clean one for being honest.
#
# Bumped to v4 for the overlapping-tile embedding. The label corpus and both architectures
# are unchanged, but 256 of the 273 columns now mean something different: they come from a
# Hann-weighted blend over overlapping tiles instead of whichever tile happened to be last,
# which is exactly the change of basis this tag exists to stop the gate from comparing
# across. v3's held-out IoU was measured on the old columns, so this run establishes its own
# baseline against MIN_ABS_IOU rather than being scored against a number from a basis it
# cannot be compared to. See docs/TILE_SEAMS.md.
RECIPE = "corrections_only_v4_overlap"
MIN_ABS_IOU = 0.60



IOU_TOL = 0.01
FP_TOL = 0.005

# Specimens the owner confirmed contain NO crack (docs/HANDOFF.md section 4). Anything a
# model marks here is a false positive by definition, which makes them the only check on
# over-prediction that does not need pixel-level ground truth -- and ground truth exists
# for exactly four images, all one specimen group.
CLEAN_SPECIMENS = ["b3_amb", "B2_amb_mosaic_2", "B2_2_1_lbf", "B2_2_9_lbf",
                   "b3_3_18lbf", "wrought_316L_fatigue_0_cycles"]


def specimen_support(raw, ds=4):
    """True where the specimen is; False on off-specimen background.

    Brightness alone cannot separate them. A crack is dark too, so a plain Otsu split
    calls the crack background -- which is how a first attempt at this scored every
    crack-heavy frame as 100% false positive. What distinguishes them is topology:
    background is one large dark region open to the frame edge, a crack is a thin
    structure in the interior.
    """
    from skimage import filters, measure, morphology
    from scipy import ndimage as ndi
    small = raw[::ds, ::ds].astype(np.float32)
    dark = small < filters.threshold_otsu(small)
    dark = morphology.binary_opening(dark, morphology.disk(3))
    lab = measure.label(dark)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    off = np.zeros_like(dark)
    for pr in measure.regionprops(lab):
        if pr.label in edge and pr.area > 0.005 * lab.size:
            off |= (lab == pr.label)
    off = morphology.binary_closing(off, morphology.disk(5))
    spec = ndi.binary_fill_holes(~off)
    return np.kron(spec, np.ones((ds, ds), bool))[:raw.shape[0], :raw.shape[1]]


def display_limits(iid):
    """(lo, hi) to map display.npy onto 0-255 for a HUMAN to look at.

    Why this is not just clip(0, 1): flat-fielding leaves the specimen in a narrow
    bright band -- measured across the loaded images, the specimen occupies a standard
    deviation of 7 to 14 grey levels out of 255. Rendered with a plain clip, a crack
    whose amplitude is a handful of counts is very nearly invisible, and the person
    labelling it is the one who most needs to see it. Stretching the specimen's own
    1st-99th percentile across the full range takes that standard deviation to 41-47.

    THE MODEL IS UNAFFECTED. It reads img.npy, the raw frame, and never this. Changing
    these limits changes what the screen shows and nothing about a prediction --
    flat-fielding as model INPUT was measured at a cost of 0.169 IoU, which is why the
    human view and the model view are deliberately different in the first place.

    Percentiles are taken over the specimen only. Off-specimen background is often 20-40%
    of a frame and sits at zero, so whole-image percentiles are dragged down by it and the
    stretch does almost nothing. Memoised in meta.json: the Otsu-plus-morphology support
    pass costs a second or two on a 32 MP mosaic and the answer never changes.
    """
    meta = S.read_meta(iid)
    lim = meta.get("display_limits")
    if isinstance(lim, (list, tuple)) and len(lim) == 2:
        return float(lim[0]), float(lim[1])
    disp = S.load_npy(iid, "display.npy")
    if disp is None:
        return 0.0, 1.0
    a = np.clip(np.asarray(disp), 0, 1)
    try:
        raw = S.load_npy(iid, "img.npy", mmap=True)
        sel = a[specimen_support(np.asarray(raw))] if raw is not None else a.ravel()
    except Exception:                                   # noqa: BLE001
        sel = a.ravel()
    if sel.size < 1000:
        sel = a.ravel()
    lo, hi = (float(v) for v in np.percentile(sel, (1.0, 99.0)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-4:
        lo, hi = 0.0, 1.0                               # degenerate: fall back to clip
    S.write_meta(iid, dict(display_limits=[lo, hi]))
    return lo, hi


def _score_clean(model, progress=None, cache_key=None):
    """(mean predicted area fraction, n images) over the loaded crack-free specimens.

    Lower is better; 0.0 is perfect. Returns (None, 0) when none of them are loaded, so
    the caller can say the check was unavailable rather than quietly treat it as passed.

    cache_key: if given and this model's prediction for an image is already in the
    per-model cache, read it instead of predicting again. The incumbent has by definition
    already been run over every loaded image, so recomputing its predictions here cost
    ~50 s per specimen for information already on disk -- enough to push a retrain past
    the self test's timeout. The candidate has no cache yet and must be computed.
    """
    fracs, detail = [], []
    todo = [m for m in S.list_images()
            if any(k.lower() in (m.get("filename") or "").lower() for k in CLEAN_SPECIMENS)]
    for i, m in enumerate(todo, 1):
        name = m.get("filename", "")
        if progress:
            progress(f"false-positive check {i}/{len(todo)}: {name[:30]}", i, len(todo))
        if cache_key:
            cached = S.load_npy_at(S.prob_cache_path(m["id"], cache_key), mmap=True)
            if cached is not None:
                f = float((np.asarray(cached) > 0.5).mean())
                fracs.append(f)
                detail.append(dict(image=name, fp=round(f, 6)))
                del cached
                continue
        img = S.load_npy(m["id"], "img.npy")
        if img is None:
            continue
        emb = None
        if model.needs_sam():
            # None if missing, damaged, or built back when tiles abutted; predict() then
            # embeds it itself rather than serving a lookup the weights never saw.
            emb = M.read_emb(S.path(m["id"], "emb.npz"))
        prob = model.predict(img, emb=emb)
        f = float((prob > 0.5).mean())
        fracs.append(f)
        detail.append(dict(image=name, fp=round(f, 6)))
        del img, prob, emb
    if not fracs:
        return None, 0, []
    return float(np.mean(fracs)), len(fracs), detail

# Extensions the uploader accepts. Kept here next to the reader that has to cope
# with them so the two cannot drift apart.
READABLE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def read_any_image(src):
    """Read a TIFF with tifffile and anything else with PIL.

    This used to be a bare tifffile.imread(), which meant a dropped PNG or JPEG was
    accepted by the uploader, copied into app_data, and only then failed the ingest
    job with "not a TIFF file: header=b'\\x89PNG'" -- while the drop zone was openly
    advertising .png. TXM data is TIFF in practice, but the UI offered the others.

    Extension picks the reader, with the other as a fallback so a mislabelled file
    (a TIFF named .png, which batch exporters do produce) still loads. If both fail,
    the error from the reader the extension implied is the one raised, since that is
    the one describing what the user actually handed us.
    """
    ext = os.path.splitext(src)[1].lower()

    def _tiff():
        import tifffile
        return tifffile.imread(src)

    def _pil():
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None      # these mosaics trip the decompression-bomb guard
        with Image.open(src) as im:
            # Leave 16-bit/float/grayscale modes alone -- robust_normalize downstream
            # handles the range, and converting would throw away bit depth.
            if im.mode in ("I", "I;16", "I;16B", "I;16L", "F", "L"):
                return np.asarray(im)
            if im.mode in ("P", "RGBA", "LA", "CMYK", "1"):
                im = im.convert("RGB")     # ndim==3 is collapsed by the caller
            return np.asarray(im)

    first, second = (_tiff, _pil) if ext in (".tif", ".tiff") else (_pil, _tiff)
    try:
        return first()
    except Exception as e_expected:
        try:
            return second()
        except Exception:
            raise e_expected


# ------------------------------------------------------------------- ingest
def ingest(image_id, progress=None, force=False, predict=True):
    """Prepare one uploaded image for viewing and correcting."""
    def rep(stage, k=1, n=1):
        S.write_meta(image_id, dict(status=stage))
        if progress:
            progress(stage, k, n)

    from txm_features import robust_normalize

    src = S.original_path(image_id)
    if src is None:
        raise FileNotFoundError(f"no original file for {image_id}")

    if force or S.load_npy(image_id, "img.npy", mmap=True) is None:
        rep("reading image")
        raw = np.asarray(read_any_image(src))
        if raw.ndim == 3:
            # A colour or multi-page TIFF: take the first plane / mean channel
            raw = raw.mean(axis=-1) if raw.shape[-1] in (3, 4) else raw[0]
        img01 = robust_normalize(raw.astype(np.float64), 1.0, 99.0).astype(np.float32)
        del raw
        S.save_npy(image_id, "img.npy", img01)
        S.write_meta(image_id, dict(height=int(img01.shape[0]), width=int(img01.shape[1]),
                                    megapixels=round(img01.size / 1e6, 2)))
    else:
        img01 = np.asarray(S.load_npy(image_id, "img.npy"))

    # Display view: destitch + flatfield. The model is fed RAW because that is
    # what it was trained on, but the real cracks are thin and faint and are
    # often only visible under local-contrast enhancement -- so what the human
    # sees and what the model sees are deliberately different images. Both steps
    # preserve geometry, so the raw-derived mask still registers on the display.
    if force or S.load_npy(image_id, "display.npy", mmap=True) is None:
        rep("destitching + flat-fielding")
        try:
            import destitch
            import flatfield
            d, _ = destitch.destitch_image(img01.astype(np.float32))
            ff = flatfield.flatfield(np.asarray(d, np.float32))
            if isinstance(ff, tuple):
                ff = ff[0]
            disp = robust_normalize(np.asarray(ff, np.float64), 1.0, 99.0).astype(np.float32)
            if disp.shape != img01.shape:
                raise ValueError(f"shape changed {img01.shape}->{disp.shape}")
            S.save_npy(image_id, "display.npy", disp)
            S.write_meta(image_id, dict(display="destitched+flatfielded"))
            del d, ff, disp
        except Exception as e:
            # Never fail ingest over the display view -- fall back to raw and say so.
            S.save_npy(image_id, "display.npy", img01)
            S.write_meta(image_id, dict(display=f"raw (preprocessing failed: {type(e).__name__})"))

    mdl = get_model()
    mkey = S.model_key(S.registry().get("current"))

    # A prediction is a pure function of (image, model), so if this model has already
    # been run on this image the answer is on disk. Adopting it turns switching
    # models from minutes of re-prediction into a hard link.
    S.migrate_prob_cache(image_id)
    if not force and S.adopt_prob(image_id, mkey):
        rep("using cached prediction")
        if S.load_npy(image_id, "correction.npy", mmap=True) is None:
            S.save_npy(image_id, "correction.npy", np.zeros(img01.shape, np.uint8))
        cached = S.load_npy(image_id, "prob.npy", mmap=True)
        S.write_meta(image_id, dict(status="ready", model=mdl.describe(),
                                    predicted_area=float(prune_specks(
                                        np.asarray(cached) > 0.5).mean()),
                                    ingested=time.time()))
        return True

    if not predict:
        # Annotation-only ingest: preprocess and stop. No SAM, no classifier, no prob.npy.
        # Deliberate -- a labeller shown the model's mask is being asked to agree with it,
        # and 98.3% of this project's existing crack labels are confirmations of exactly
        # that. Dense ground truth has to be drawn without it.
        S.write_meta(image_id, dict(status="ready", model="annotation only (no prediction)",
                                    annotation_only=True, predicted_area=0.0,
                                    ingested=time.time()))
        return
    embp = S.path(image_id, "emb.npz")
    sam_note = None
    if mdl.needs_sam() and (M.sam_unavailable_reason or M.sam_disabled_by_env()) \
            and not M.emb_is_current(embp):
        # Already known unreachable in this process: skip straight to the 17-feature model
        # rather than attempting a 2.4 GB download once per image.
        sam_note = M.sam_unavailable_reason or "disabled by TXM_NO_SAM=1"
        mdl = M.CrackModel(path_17=M.DEFAULT_17, path_hybrid="", ensemble=False)
    if mdl.needs_sam() and (force or not M.emb_is_current(embp)):
        rep("SAM embedding")
        try:
            coords, emb = M.embed_image(img01,
                                        progress=lambda k, n: rep("SAM embedding", k, n))
        except M.SamUnavailable as e:
            # The fallback run_app.sh has always promised. Predict with the 17-feature
            # model alone (mean IoU 0.744 against 0.821 for the ensemble) and record WHY in
            # meta, so the sidebar and the model line say so instead of it being silent.
            sam_note = str(e)
            rep(f"SAM unavailable ({sam_note}) -- using the 17-feature model")
            mdl = M.CrackModel(path_17=M.DEFAULT_17, path_hybrid="", ensemble=False)
            coords = emb = None
        if coords is not None:
            M.write_emb(embp, coords, emb)
        del coords, emb

    rep("predicting")
    emb = None
    if mdl.needs_sam():
        emb = M.read_emb(embp, note=rep)
        if emb is None:
            # An unusable cache must not be a dead end. Recompute rather than raising: this
            # file is derived data, so throwing it away costs seconds of GPU and nothing
            # else, while raising cost the user the whole image.
            coords, emb2 = M.embed_image(img01,
                                         progress=lambda k, n: rep("SAM embedding", k, n))
            M.write_emb(embp, coords, emb2)
            emb = (coords, emb2)
            del coords, emb2
    prob = mdl.predict(img01, emb=emb, progress=lambda st, k, n: rep(st, k, n))
    S.store_prob(image_id, mkey, prob.astype(np.float32))

    if S.load_npy(image_id, "correction.npy", mmap=True) is None:
        S.save_npy(image_id, "correction.npy", np.zeros(img01.shape, np.uint8))

    S.write_meta(image_id, dict(status="ready", model=mdl.describe() + (f"  [SAM unavailable: {sam_note}]" if sam_note else ""),
                                # The pruned figure, because that is the mask the user is
                                # shown. Reporting the raw area next to a pruned overlay
                                # makes the sidebar disagree with the picture beside it.
                                predicted_area=float(prune_specks(prob > 0.5).mean()),
                                ingested=time.time()))
    return True


# ------------------------------------------------------------------- model
_model_cache = {"key": None, "obj": None}


def get_model():
    r = S.registry()["current"]
    key = json.dumps(r, sort_keys=True)
    if _model_cache["key"] != key:
        # "path_17" present but empty means "no 17-feature member", which is different
        # from absent (legacy entry -> the shipped default). `or` cannot express that:
        # it turned an explicit opt-out back into the default, and CrackModel would then
        # run the 17-feature model on every band and discard the result.
        _model_cache["obj"] = M.CrackModel(
            path_17=(r["path_17"] if "path_17" in r else M.DEFAULT_17),
            path_hybrid=r.get("path_hybrid") or M.DEFAULT_HYBRID,
            ensemble=(r.get("kind", "ensemble") == "ensemble"))
        _model_cache["key"] = key
    return _model_cache["obj"]


# Drop predicted blobs smaller than this many pixels. MEASURED, and the measurement is
# the whole argument for it:
#
#   min area   held-out IoU   folds won   crack-free FP   worst image's confirmed crack kept
#   none         0.8317           -          0.264%                100.0%
#   1000         0.8371        4 of 4        0.144%                 97.3%
#   2000         0.8391        4 of 4        0.106%                 97.3%
#   5000         0.8420        4 of 4        0.037%                 87.6%   <- cliff
#
# IoU is leave-one-image-out on full-resolution probability maps from models that never saw
# the image they are scored on -- a neighbourhood operation cannot be evaluated on the
# sampled scattered pixels the architecture sweep used. Every metric improves monotonically
# up to 2000, and it wins on all four folds, not on the mean.
#
# 2000 rather than the top-scoring 5000 because of the third column, which came from the
# owner's own 30.2 M hand-drawn crack pixels: at 5000 the worst single image loses 12.4% of
# the crack its owner confirmed, while 2000 costs the same 2.7% as 1000 does. The extra
# 0.003 IoU is not worth deleting a researcher's work.
#
# THIS IS NOT THE LEGACY POST-PROCESSING. M.postprocess bundles a blur, a closing, ring
# rejection, an eccentricity test and hysteresis growth, and measured -0.084 IoU, which is
# why its toggle is off by default. Isolating the pieces shows the size filter was never the
# harmful part.
MIN_BLOB_PX = 2000


def _skimage_size_kw(fn, n):
    """Keyword and value for a size threshold, across scikit-image versions.

    0.26 deprecated `min_size` / `area_threshold` in favour of `max_size` AND changed the
    comparison with it: the old parameters dropped things strictly SMALLER than the value,
    `max_size` drops things smaller than OR EQUAL to it. Passing the same number through the
    new spelling would silently move the floor by one pixel, so it gets n-1. requirements.txt
    allows scikit-image >= 0.20, so both spellings have to keep working.
    """
    import inspect
    if "max_size" in inspect.signature(fn).parameters:
        return {"max_size": int(n) - 1}
    old = "area_threshold" if "area_threshold" in inspect.signature(fn).parameters else "min_size"
    return {old: int(n)}


def _binary_morph(name):
    """closing/opening under whichever name this scikit-image exposes.

    0.26 deprecated binary_closing/binary_opening in favour of closing/opening, which do the
    same thing on a boolean array.
    """
    from skimage import morphology
    return getattr(morphology, name, None) or getattr(morphology, "binary_" + name)


def prune_specks(mask, min_px=None):
    """Drop connected components below `min_px`. Never touches anything else."""
    from skimage import morphology
    n = MIN_BLOB_PX if min_px is None else min_px
    if not n or not mask.any():
        return mask
    fn = morphology.remove_small_objects
    return fn(mask, **_skimage_size_kw(fn, n))


# Inside a hand-painted crack stroke, accept the model at this probability instead of the
# display threshold. 0.20 keeps 89.6% of what the strokes were adding while dropping the
# brush-shaped remainder; 0.35 would keep only 65.3%.
CORRECTION_FLOOR = 0.20


# Fill enclosed not-crack islands up to this many pixels. MEASURED over the first 14
# images: 1,313 interior holes totalling 176,163 px, and the distribution is sharply
# bimodal -- 1,302 of them (99.2%) are under 1,024 px and account for only 19.4% of the
# hole AREA, while two holes of 59,194 and 65,505 px make up most of it. Those two are
# islands of intact material surrounded by crack, and filling them would be a lie about
# the specimen. So the cutoff keeps the noise and keeps the islands.
FILL_HOLES_MAX_PX = 1024

# Neighbourhood width for the tight boundary, in pixels. Wider keeps more faint crack and
# runs slightly thicker: measured on wrought_316L_fatigue_1200_cycles, w=151 keeps 76.6% of
# the painted crack at a 1.0 px half-width and w=301 keeps 83.7% at 1.4 px. 301 is chosen for
# the recall, since the width is already at the 1.0 px core either way.
TIGHTEN_WINDOW = 301

# Narrowing rests on an ASSUMPTION -- that crack is darker than its surroundings -- and on
# some frames it is false. Checked per frame against the darkest fifth of the corridor, the
# rule keeps a median of 97.0% of it, but 8 of 66 frames fall below 70% and 4 below 50%, worst
# 34.4% on B2_3_1_lbf: there the narrowing deletes the crack instead of trimming it. So the
# assumption is verified on each frame and the tightening is DECLINED where it fails, rather
# than silently throwing away the thing being measured. 0.60 sits below the 65.1% frame and
# above the 49.8% one, so it catches the four pathological frames and leaves the rest narrowed.
TIGHTEN_MIN_CORE = 0.60


def effective_mask(image_id, threshold=0.5, postprocess=False, prune=True,
                   corrections="paste", fill_holes=True, tight=False):
    """The model's prediction, combined with the user's corrections in one of three ways.

    THREE MODES, because the canvas and the export want different things and a single
    behaviour cannot serve both.

    "paste" (default, and what the canvas uses): crack where you painted, not-crack where you
    erased, model elsewhere. A brush stroke appears the instant you release the mouse. The
    cost is that the brush's own shape lands in the mask -- disc bumps and circular arcs that
    no crack has.

    "gate" (what exports use): inside a crack stroke the threshold drops to CORRECTION_FLOOR
    instead of being forced True, so the stroke means "believe weaker evidence here" and the
    boundary is still drawn by the image. Measured over every corrected image: of the 93,633
    px that "paste" forces on against the model's 0.5 verdict, 89.6% survive at 0.20. So it
    keeps almost all the recall and drops the remainder, which is the brush geometry.

    "none": the model's own output. The only way to see whether a retrain actually learned a
    region or is having your answer pasted back over it.

    WHY "gate" IS NOT THE DEFAULT, learned the hard way: you paint where the model is WRONG,
    which is where its probability is lowest. Under "gate" a stroke on a confidently-wrong
    region produces no visible change at all -- the selftest caught exactly this, as painting
    no longer invalidating the overlay and as hand-painted crack surviving pruning at only
    99.889%. Silently ignoring a label in the one case the label matters most is worse than an
    ugly boundary, so the canvas pastes and only the deliverable gates.

    Erase is absolute in every mode. "This is not crack" is a human statement about the
    specimen, and 28% of the area it removes is where the model is confident (p>0.8) -- which
    is precisely the false-positive case erasing exists to fix.
    """
    prob = S.load_npy(image_id, "prob.npy")
    if prob is None:
        # No prediction: an annotation-only image, ingested deliberately without one so the
        # labeller is not anchored by the model's opinion. Their own strokes must still be
        # visible, so start from an empty mask rather than bailing out -- returning None
        # here made the overlay endpoint render nothing and the painting invisible.
        corr0 = S.load_npy(image_id, "correction.npy", mmap=True)
        if corr0 is None:
            return None
        mask = np.zeros(np.asarray(corr0).shape, bool)
        del corr0
        corr = S.load_npy(image_id, "correction.npy")
        if corr is not None:
            mask[corr == 1] = True
        return mask
    mask = M.postprocess(prob) if postprocess else (prob > threshold)
    if prune and not postprocess:
        mask = prune_specks(mask)
    if fill_holes and not postprocess:
        # Speck pruning removes small ISLANDS of crack; this removes small islands of
        # not-crack inside crack. They are the same argument from opposite sides: a
        # 3-pixel gap in the middle of a crack is noise in the probability map, not a
        # feature of the material. Legacy post-processing did fill holes, but it is off by
        # default because of everything ELSE it did -- it measurably deleted thin crack.
        # This is the one part of it worth keeping, on its own, with a measured cutoff.
        from skimage.morphology import remove_small_holes
        mask = remove_small_holes(
            mask, **_skimage_size_kw(remove_small_holes, FILL_HOLES_MAX_PX))
    if corrections == "none":
        return mask
    corr = S.load_npy(image_id, "correction.npy")
    if corr is not None and corr.shape == mask.shape:
        mask = mask.copy()
        # See the docstring: paste stamps the brush, gate lets the image draw the edge.
        inside = corr == 1
        if inside.any():
            mask[inside] = (prob[inside] > CORRECTION_FLOOR if corrections == "gate"
                            else True)
        # Erase stays absolute. "This is not crack" is a statement about the specimen that no
        # amount of model confidence should override, and it can only ever remove area.
        mask[corr == 2] = False
        # PRUNE AGAIN, because corrections are applied after the first prune and undo its
        # guarantee. An eraser stroke does not just remove area: it cuts THROUGH blobs and
        # leaves the offcuts behind as separate components below the floor. Measured on
        # b2_343_75_LARGE, the worst case: 19 components with no corrections, 70 with them,
        # and 51 of those 70 under 200 px. Those specks are what reads as "tiny black dots
        # that do not look like crack" in an exported mask -- they are debris from erasing,
        # not anything the model or the user asserted.
        #
        # Under "paste" a component containing any painted pixel is spared regardless of
        # size, because there the stroke is an assertion: a small deliberate dab is a
        # statement, an offcut of the model's output is not, and silently deleting a stroke
        # smaller than the floor is the one thing painting must never do.
        #
        # Under "gate" there is no such exemption, and deliberately. A gated pixel is not an
        # assertion that this is crack -- it is "believe weaker evidence here", with the
        # boundary still drawn by the image. So a stroke laid over a region the model barely
        # likes shatters into slivers at the floor, and those slivers are exactly the specks
        # this is removing. Sparing them would preserve the artefact on the one path that
        # produces the deliverable. Measured on b2_340_94: 6 sub-200 px components survive
        # with the exemption, 0 without, and crack area moves by 0.003 pp.
        if corrections == "gate" and not postprocess:
            # SMOOTH THE BOUNDARY, on the deliverable path only. The brush stamps discs, so a
            # stroke is a chain of overlapping circles, and gating erodes the arcs where they
            # meet into ragged cusps -- which is what makes an exported mask look painted
            # rather than measured. A radius-2 close-then-open cuts boundary cusps 5-18% per
            # frame (10,012 -> 8,211 on b2_343_75_LARGE) for +0.003 pp of area.
            #
            # Then RE-ERASE, which is not optional. Closing bleeds crack into regions the user
            # erased -- measured at 0.038-0.062% of erased area before this line was added --
            # and "this is not crack" is a human statement about the specimen that no
            # morphology gets to overrule. Re-asserting it costs about a third of the cusp
            # reduction and keeps the invariant exact.
            #
            # Not on the canvas: there a stroke must appear as drawn the instant the mouse is
            # released, and smoothing what someone is actively painting reads as the tool
            # fighting them. Guardrails on this: predicted area over the six crack-free
            # specimens moves by +0.0000 pp, recall on painted crack by -0.02 pp at worst.
            from skimage.morphology import disk
            mask = _binary_morph("opening")(_binary_morph("closing")(mask, disk(2)), disk(2))
            mask[corr == 2] = False
        if prune and not postprocess:
            spare = inside if corrections == "paste" else None
            mask = prune_specks_keeping(mask, spare)
    if tight and mask.any():
        # On the canvas ("paste") an explicitly painted pixel is never narrowed. Otherwise a
        # stroke over a crack that is not among the darkest pixels -- a filled or bright one
        # -- would produce no visible change at all, which is exactly the failure that
        # disqualified "gate" as the canvas default. The deliverable still tightens fully.
        spare = None
        if corrections == "paste":
            corr_p = S.load_npy(image_id, "correction.npy")
            if corr_p is not None and corr_p.shape == mask.shape:
                spare = corr_p == 1
        mask = tighten_to_image(image_id, mask, prune=prune and not postprocess, spare=spare)
    return mask


def tighten_to_image(image_id, mask, prune=True, spare=None):
    """Narrow an accepted region to the dark core inside it, using the image.

    WHY THIS EXISTS. The exported mask is as wide as the brush that labelled it. Measured
    against the darkest fifth of each stroke -- the crack itself -- the strokes on these
    frames are 2.6x to 15x too wide (half-width 15.0 px against a 1.0 px core on
    wrought_316L_fatigue_1200_cycles). The model reproduces that width faithfully, because
    that is what it was trained on: raising the probability cut to 0.95 still leaves a 12 px
    half-width while throwing away two thirds of the area, so it removes crack rather than
    narrowing it. The width was never in the labels, so no threshold or morphology recovers
    it -- but it IS in the image.
    
    So the detection supplies a corridor and the image draws the boundary inside it: a pixel
    stays if it is darker than the average of its own neighbourhood.

    LOCAL, not global. The first version used Otsu over the whole corridor, which is
    parameter-free and looked reasonable on the numbers -- but one threshold for the entire
    frame deletes faint crack in a brighter region outright, and that is exactly what was
    reported as "a lot of the correct overlay disappeared". Comparing the two on the same
    frames:

      variant            % of frame   median half-width   painted crack kept
      wide, no tighten      8.178%          16.3 px             100.0%
      global Otsu           4.698%          10.0 px              59.7%
      local mean w=301      6.464%           1.4 px              83.7%

    The local threshold is better on BOTH axes at once: a 1.4 px half-width against the 1.0 px
    dark core actually present, while keeping 83.7% of the painted crack instead of 59.7%. It
    follows the crack ridge everywhere rather than comparing a faint branch against the
    darkest pixels elsewhere in the frame.

    ON by default since 2026-08-24. Predicted area on the six confirmed crack-free specimens
    holds or improves (0.0230% wide, 0.0209% here), so this is not bought with false
    positives. What it still costs is recall against the painted strokes -- 83.7%, 63.8% and
    78.7% on the three frames measured -- and some of that is faint crack rather than
    over-mark. The two cannot be separated without pixel-accurate reference annotation, which
    this project does not have. `tight=0` on any request returns the wider boundary.
    """
    from scipy.ndimage import uniform_filter
    img = S.load_npy(image_id, "img.npy")
    if img is None:
        return mask
    img = np.asarray(img, np.float32)
    if img.shape != mask.shape:
        return mask
    vals = img[mask]
    if vals.size < 64 or float(vals.min()) == float(vals.max()):
        return mask
    # window clamped to the frame: uniform_filter pads, but a window wider than the image
    # averages mostly padding and the comparison stops meaning anything
    win = min(TIGHTEN_WINDOW, (min(img.shape) // 2) * 2 + 1)
    out = mask & (img <= uniform_filter(img, size=win))
    # Does "crack is locally darker" hold on THIS frame? Score against the darkest fifth of
    # what the detector already accepted -- the crack itself, whatever its polarity -- and
    # decline to narrow if the rule is throwing that away instead of trimming around it.
    core = mask & (img <= np.percentile(img[mask], 20))
    if core.any() and (out & core).sum() / core.sum() < TIGHTEN_MIN_CORE:
        return mask
    if spare is not None:
        out |= mask & spare
    if not out.any():
        return mask
    if not prune:
        return out
    # A tighter boundary breaks a wide band into thinner threads, so the 2000 px speck floor
    # cannot simply be re-applied: it is calibrated for corridor-width blobs and would delete
    # real thread. Dropping to 200 px keeps the threads but let 32 ROUNDISH blobs through
    # across the corpus -- and a round 200 px blob is what reads as a black dot.
    #
    # So shape decides, not just size: below the full floor, keep what is elongated like a
    # crack and drop what is not. Measured over 66 frames, sub-2000 px components split 75
    # elongated (aspect >= 3) against 32 roundish, and only the second kind is unwanted.
    #
    # `spare` is exempt from BOTH the size floor and the shape rule. A single click is a
    # round blob of about 1250 px, so without that exemption the shape rule deleted it and a
    # click on the canvas did nothing at all -- the selftest for "painting invalidates the
    # cached overlay" caught it. An assertion the user drew is not judged on its shape.
    from skimage.measure import label as _label, regionprops as _rp
    keep_lab = set()
    pruned = prune_specks_keeping(out, spare, min_px=200)
    lab = _label(pruned, connectivity=2)
    if lab.max() == 0:
        return pruned
    if spare is not None and spare.any():
        keep_lab = set(np.unique(lab[spare & pruned]).tolist())
    kill = []
    for r in _rp(lab):
        if r.area >= MIN_BLOB_PX or r.label in keep_lab:
            continue
        if r.major_axis_length / max(r.minor_axis_length, 1e-6) < 3.0:
            kill.append(r.label)
    return pruned & ~np.isin(lab, kill) if kill else pruned


def prune_specks_keeping(mask, keep, min_px=None):
    """prune_specks, except components overlapping `keep` survive at any size."""
    from skimage.measure import label as _label
    n = MIN_BLOB_PX if min_px is None else min_px
    lab = _label(mask, connectivity=2)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    doomed = np.flatnonzero(sizes < n)
    if doomed.size == 0:
        return mask
    spared = np.unique(lab[keep & mask]) if keep is not None and keep.any() else np.array([])
    kill = np.setdiff1d(doomed, spared)
    if kill.size == 0:
        return mask
    return mask & ~np.isin(lab, kill)


# ------------------------------------------------------------------ retrain




GT_EMB_DIR = os.path.join(S.DATA, "gt_emb")






CLEAN_FP_CACHE = os.path.join(S.DATA, "models", "clean_fp.json")


def clean_fp_measured(model_key):
    """Mean predicted area on the crack-free specimens for a model, from cached probs only.

    Returns (fraction, n_specimens) or (None, 0) when this model has not been run over all
    of them. Never predicts -- it reads the per-model prediction cache, so it is cheap
    enough to call from /api/models on every page load.

    WHY THIS IS SURFACED IN THE UI. The picker used to list every model in the registry
    with nothing but a name and a cache count, and two of the entries in a real user's
    history mark 22% of crack-free specimen as crack -- they predate the false-positive
    gate, so they deployed legitimately and then stayed selectable forever. Someone
    switched to one of them, got visibly worse masks with no explanation available
    anywhere in the interface, and reported it as "why does it show older model". A number
    next to the name is the whole fix: the bad ones read 22%, the good ones read 0.1-0.2%.

    Memoised on disk, keyed by model_key plus the correction mtimes that could change an
    effective mask, so a stale figure cannot outlive the thing it measured.
    """
    todo = [m for m in S.list_images()
            if any(k.lower() in (m.get("filename") or "").lower() for k in CLEAN_SPECIMENS)]
    if not todo:
        return None, 0
    stamp = 0.0
    for m in todo:
        cp = S.path(m["id"], "correction.npy")
        if os.path.exists(cp):
            stamp = max(stamp, os.path.getmtime(cp))
    key = f"{model_key}|{len(todo)}|{stamp:.3f}"

    cache = {}
    if os.path.exists(CLEAN_FP_CACHE):
        try:
            with open(CLEAN_FP_CACHE) as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
    if key in cache:
        v = cache[key]
        return (v[0], v[1]) if v else (None, 0)

    fracs = []
    for m in todo:
        a = S.load_npy_at(S.prob_cache_path(m["id"], model_key), mmap=True)
        if a is None:
            return None, 0                      # incomplete: do not cache a partial answer
        fracs.append(float((np.asarray(a) > 0.5).mean()))
        del a
    val = [float(np.mean(fracs)), len(fracs)]
    cache[key] = val
    cache = {k: v for k, v in list(cache.items())[-40:]}        # bounded
    try:
        S.write_json(CLEAN_FP_CACHE, cache)
    except OSError:
        pass
    return val[0], val[1]


# 60k rather than 30k: the per-fold FITS are capped at CV_TRAIN_CAP either way, so a
# bigger sample costs only sampling and prediction time (a few seconds) and halves the
# sampling noise in the number people will read. At 30k this measured 0.8107 against
# 0.8241 from a 120k-row harness -- a 0.013 gap that is pure sample size.
CV_ROWS_PER_IMAGE = 60000
CV_TRAIN_CAP = 400000
CV_TEST_CAP = 250000
CV_MAX_FOLDS = 5
CV_TRAIN_CAP = 90000




def false_indications(model_key=None, threshold=0.5):
    """Spurious INDICATIONS per frame on confirmed crack-free specimens, not just area.

    The NDT question that a pixel-area fraction cannot answer. "0.106% of area" tells an
    engineer nothing about whether they will be chasing one artifact or thirty; "4.0
    indications per frame" does, and it is the quantity MIL-HDBK-1823A's false-call analysis
    is built around. No segmentation tool surveyed for this project reports it -- they report
    IoU or Dice, which is silent on material that contains nothing to find.

    Counts connected components of the pruned mask, since that is what a person would have to
    look at and dismiss. Reads cached predictions only, so it costs nothing to include in
    every retrain scorecard.
    """
    from skimage import measure
    key = model_key or S.model_key(S.registry().get("current"))
    per = []
    for m in S.list_images():
        fn = m.get("filename") or ""
        if not any(k.lower() in fn.lower() for k in CLEAN_SPECIMENS):
            continue
        a = S.load_npy_at(S.prob_cache_path(m["id"], key), mmap=True)
        if a is None:
            continue
        mk = prune_specks(np.asarray(a) > threshold)
        per.append(dict(image=fn, indications=int(measure.label(mk).max()),
                        area_fraction=round(float(mk.mean()), 6)))
        del a, mk
    if not per:
        return None
    n = [p["indications"] for p in per]
    return dict(per_specimen=per, n_specimens=len(per),
                mean_indications=round(float(np.mean(n)), 2),
                max_indications=int(max(n)), zero_specimens=int(sum(1 for v in n if v == 0)),
                mean_area_fraction=round(float(np.mean([p["area_fraction"] for p in per])), 6))


def crossval_on_rows(X, y, groups, progress=None):
    """Grouped-by-image k-fold on the rows the model was actually trained on.

    Replaces the gate's old externally-labelled axis. Nothing here
    touches dataset_cache: the rows are the owner's own corrections, and the grouping is the
    image each row came from, so train and test never share an image -- which is the property
    that matters, because the 17 features reach 256 px and a SAM embedding is one vector per
    16x16 block, so a randomly held-out PIXEL almost always has a training pixel next to it
    carrying the same measurement. Measured on this data: random pixel folds read IoU 0.930
    with a fold spread of 0.003, grouping by image reads 0.824 with a spread of 0.050.

    Reuses the training rows rather than resampling, so it costs one refit per fold and no
    extra feature computation -- and it describes the same data the deployed model saw.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return None
    k = min(CV_MAX_FOLDS, n_groups)
    n17 = 17

    def _clf():
        return Pipeline([("s", StandardScaler()),
                         ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                             early_stopping=True, n_iter_no_change=8))])

    # cap the rows per fold: the full matrix refit k times is not affordable inside a retrain
    rng = np.random.RandomState(0)
    per_fold = []
    for f, (tr, te) in enumerate(GroupKFold(k).split(X, groups=groups), 1):
        if progress:
            progress(f"cross-validation fold {f}/{k}", f, k)
        if len(tr) > CV_TRAIN_CAP:
            tr = rng.choice(tr, CV_TRAIN_CAP, replace=False)
        if len(te) > CV_TEST_CAP:
            te = rng.choice(te, CV_TEST_CAP, replace=False)
        # the deployed architecture: mean probability of a 17-feature model and the hybrid
        probs = []
        for cols in (slice(0, n17), slice(0, X.shape[1])):
            probs.append(_clf().fit(X[tr, cols], y[tr]).predict_proba(X[te, cols])[:, 1])
        prob = np.mean(probs, axis=0)
        t = y[te]

        def _score(p):
            pred = p > 0.5
            tp = int((pred & t).sum()); fp = int((pred & ~t).sum()); fn = int((~pred & t).sum())
            return (round(tp / max(tp + fp + fn, 1), 4),
                    round(tp / max(tp + fp, 1), 4),
                    round(tp / max(tp + fn, 1), 4))

        # Both members are already fitted and predicted here, so scoring them individually is
        # free. It used to be discarded, which left the choice of an ensemble over either
        # member alone resting on a leave-one-image-out run over externally-labelled frames
        # that this project no longer uses for anything -- an argument with no current
        # measurement behind it. Now every retrain re-checks it on the basis that remains.
        iou, prec, rec = _score(prob)
        i17, _, _ = _score(probs[0])
        ihy, _, _ = _score(probs[1])
        per_fold.append(dict(held_out=f"{len(np.unique(groups[te]))} images",
                             n=int(len(te)),
                             iou=iou, precision=prec, recall=rec,
                             iou_17_only=i17, iou_hybrid_only=ihy))
    ious = [f["iou"] for f in per_fold]
    return dict(k=k, per_fold=per_fold, grouped_by="image",
                labelled_images_used=int(n_groups),
                label_source="owner corrections only (no external)",
                mean_iou=round(float(np.mean(ious)), 4),
                std_iou=round(float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0, 4),
                min_iou=round(float(np.min(ious)), 4),
                mean_iou_17_only=round(float(np.mean([f["iou_17_only"] for f in per_fold])), 4),
                mean_iou_hybrid_only=round(
                    float(np.mean([f["iou_hybrid_only"] for f in per_fold])), 4),
                mean_precision=round(float(np.mean([f["precision"] for f in per_fold])), 4),
                mean_recall=round(float(np.mean([f["recall"] for f in per_fold])), 4))




RETRAIN_HISTORY = os.path.join(S.DATA, "models", "retrain_history.json")
HISTORY_KEEP = 20


def retrain_history():
    """Every retrain this app has scored, oldest first. [] if none yet."""
    if not os.path.exists(RETRAIN_HISTORY):
        return []
    try:
        with open(RETRAIN_HISTORY) as f:
            h = json.load(f)
        return h if isinstance(h, list) else []
    except (OSError, ValueError):
        return []


def record_retrain(result, stamp=None):
    """Append one retrain's scorecard to the on-disk history.

    WHY THIS EXISTS. retrain() already measured everything a person needs to judge the
    model -- ground-truth IoU and recall for both candidate and incumbent, false positives
    on every crack-free specimen, what it trained on, whether each half of the gate passed
    -- and then threw all of it away. It lived only in the in-memory JOBS dict, so a page
    reload, a server restart or simply the next retrain erased it, and the interface said
    nothing except "retrain complete". Three consecutive retrains here drifted 0.137% ->
    0.238% -> 0.264% on crack-free specimen while ground-truth IoU sat flat at 0.936-0.940,
    and nobody could have seen that from the app. A number you measure and discard is worse
    than one you never measured, because it feels like you checked.

    Kept to the last HISTORY_KEEP entries so the file cannot grow without bound.
    """
    info = result.get("info") or {}
    entry = dict(
        stamp=stamp,
        when=time.strftime("%Y-%m-%d %H:%M:%S"),
        deployed=bool(result.get("deployed")),
        passes_gate=bool(result.get("passes_gate")),
        reason=result.get("reason"),
        gate_detail=result.get("gate_detail"),
        candidate=result.get("candidate"),
        incumbent=result.get("incumbent"),
        candidate_clean_fp=result.get("candidate_clean_fp"),
        incumbent_clean_fp=result.get("incumbent_clean_fp"),
        clean_specimens=result.get("clean_specimens"),
        clean_fp_by_specimen=result.get("clean_fp_by_specimen"),
        seconds=result.get("seconds"),
        rows=info.get("n_px"),
        n_features=info.get("n_features"),
        crack_fraction=info.get("crack_fraction"),
        correction_crack_px=info.get("correction_crack_px"),
        labelled_images=sum(1 for m in S.list_images()
                            if (m.get("corrected_crack_px") or 0)
                            or (m.get("corrected_not_px") or 0)),
        iou_tol=IOU_TOL, fp_tol=FP_TOL,
        recipe=info.get("recipe"),
        reference_held_out=info.get("reference_held_out"),
        trained_on_images=info.get("trained_on_images"),
        heldout=result.get("heldout"),
        heldout_error=result.get("heldout_error"),
        false_indications=result.get("false_indications"),
    )
    hist = retrain_history()
    hist.append(entry)
    try:
        S.write_json(RETRAIN_HISTORY, hist[-HISTORY_KEEP:])
    except OSError:
        pass
    return entry


def gather_training_data(progress=None):
    """Every corrected pixel across every uploaded image, as hybrid [17 | 256] features.

    Every labelled image, with no exclusions: there is no external ground truth in this
    project and nothing is held back. See the note at the top of this file.

    Class balance is computed from what actually exists rather than hard-coded:
    it is the single knob that has caused four regressions in this project, and
    a fixed cap silently goes stale as more images get labelled.
    """
    import destitch  # noqa: F401  (ensures code/ is importable before heavy work)
    Xs, ys = [], []

    # EVERY labelled image trains. The four B2 frames used to be excluded because the gate
    # scored against their external masks, so training on them would have been training on the
    # test set. The gate no longer uses those masks at all -- external labels are not used
    # anywhere in this project now -- so the frames are just images, and the owner's own
    # corrections on them are as good as any other. Their specimens come back too.
    items = [m for m in S.list_images()
             if m.get("corrected_crack_px") or m.get("corrected_not_px")]
    held_out = []
    n_crack_total = sum(m.get("corrected_crack_px", 0) for m in items)
    per_img_cap = 30000
    corr_crack = sum(min(per_img_cap, m.get("corrected_crack_px", 0)) for m in items)
    n_bg_imgs = sum(1 for m in items if m.get("corrected_not_px", 0) > 0)

    neg_cap = max(500, int(round(corr_crack / n_bg_imgs))) if n_bg_imgs else per_img_cap

    rng = np.random.RandomState(0)

    # NO GROUND-TRUTH ROWS. They used to be sampled here, 100 k crack + 100 k background per
    # stem, and they were the largest single block in the training set. Those labels came from
    # another tool and are gone from this project entirely -- not training, not scoring. What
    # follows is the owner's own corrections and nothing else.


    # user corrections
    for k, m in enumerate(items, 1):
        iid = m["id"]
        corr = S.load_npy(iid, "correction.npy")
        img = S.load_npy(iid, "img.npy")
        if corr is None or img is None:
            continue
        ci = np.flatnonzero(corr.reshape(-1) == 1)
        bi = np.flatnonzero(corr.reshape(-1) == 2)
        nc = min(per_img_cap, len(ci)); nb = min(neg_cap, len(bi))
        if nc + nb == 0:
            continue
        idx = np.concatenate([
            rng.choice(ci, nc, replace=False) if nc else ci[:0],
            rng.choice(bi, nb, replace=False) if nb else bi[:0]])
        rr, cc = np.unravel_index(idx, corr.shape)
        from txm_features import compute_feature_stack
        f17 = compute_feature_stack(np.asarray(img))
        a = np.asarray(f17[rr, cc, :], np.float32)
        del f17
        zp = S.path(iid, "emb.npz")
        got = M.read_emb(zp)
        if got is None:
            # Re-embed rather than skip. Skipping drops this image's labels from training
            # without saying so, and a cache built when tiles abutted cannot be blended:
            # fitting on last-tile-wins while serving a blended lookup is exactly the
            # mismatch that took crack-free false positives from 0.019% to 0.080%.
            if progress:
                progress(f"re-embedding {k}/{len(items)}", k, len(items))
            try:
                coords, embs = M.embed_image(np.asarray(img, np.float32))
                M.write_emb(zp, coords, embs)
            except M.SamUnavailable:
                continue
        else:
            coords, embs = got
        b = M.emb_rows(coords, embs, rr, cc)
        Xs.append(np.concatenate([a, b], axis=1))
        ys.append(np.concatenate([np.ones(nc, bool), np.zeros(nb, bool)]))
        del corr, img, coords, embs, a, b
        if progress:
            progress(f"features {k}/{len(items)}", k, len(items))

    if not Xs:
        return None, None, dict(reason="no labelled data")
    # The ground-truth block is 17-dim; correction blocks are 273-dim. Only train
    # on the common width, and say which happened rather than crashing later.
    widths = {x.shape[1] for x in Xs}
    target = max(widths)
    keep_i = [i for i, x in enumerate(Xs) if x.shape[1] == target]
    dropped = len(Xs) - len(keep_i)

    # PREALLOCATE AND FILL, rather than np.concatenate(...).astype(np.float32).
    #
    # That one line held three copies of the whole training matrix alive at once. With 71
    # labelled images at ~65k rows each plus 0.8 M ground-truth rows, the matrix is ~5.4 M
    # rows x 273 float32 = 5.9 GB, and at the moment of the astype there were three: the
    # per-image blocks still referenced by Xs, the concatenate result, and the astype copy.
    # ~17.7 GB transient, on a machine that may have 16 GB total -- and it happened AFTER
    # the expensive SAM feature pass, so the cost was paid before the failure. The astype
    # was pure waste besides: every block is constructed float32 already.
    #
    # Filling a preallocated array and dropping each block as it is copied holds one copy
    # plus one block: ~5.9 GB + ~90 MB.
    n_rows = sum(Xs[i].shape[0] for i in keep_i)
    X = np.empty((n_rows, target), np.float32)
    y = np.empty(n_rows, bool)
    # Which IMAGE each row came from. The grouped cross-validation reuses exactly these rows,
    # so it needs the grouping -- and reusing them means the honest number costs nothing extra
    # and describes the same data the model was fitted on.
    groups = np.empty(n_rows, np.int32)
    at = 0
    for gi, i in enumerate(keep_i):
        blk, lab = Xs[i], ys[i]
        n = blk.shape[0]
        X[at:at + n] = blk
        y[at:at + n] = lab
        groups[at:at + n] = gi
        at += n
        Xs[i] = None                    # free this block now, not at function exit
        ys[i] = None
    assert at == n_rows, (at, n_rows)
    del Xs, ys, keep_i

    info = dict(n_px=int(len(y)), n_features=int(target), crack_fraction=float(y.mean()),
                neg_cap=neg_cap, correction_crack_px=int(n_crack_total),
                blocks_dropped_for_width=dropped,
                # recorded per retrain so a scorecard is self-describing: a number measured
                # under one recipe cannot be silently compared to one measured under another
                gt_in_training=False, recipe=RECIPE,
                trained_on_images=len(items), reference_held_out=held_out)
    # `w` is gone: it was built, concatenated and returned, and retrain() never passed it to
    # clf.fit -- 43 MB of sample weights computed and thrown away every retrain.
    return X, y, info, groups


def retrain(deploy=True, progress=None):
    """Train a new hybrid on all current corrections, validate, maybe deploy."""
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    # Built here rather than at startup, so a fresh clone is running in seconds and
    # only pays for this the first time it retrains. Must precede gather_training_data,
    # whose ground-truth loop skips stems with no feature stack.
    built = None
    if progress:
        progress("gathering labels", 0, 1)
    X, y, info, groups = gather_training_data(progress=progress)
    if X is None:
        return dict(ok=False, error="no labelled data yet -- paint some corrections first")

    # A degenerate balance is a REFUSAL, not a warning. Measured on the first real
    # retrain: a 100%-crack set (all 4 ground-truth blocks dropped, only force-crack
    # strokes left) produced IoU 0.003. Spending 5 minutes to train and validate
    # something that cannot possibly work, then reporting it as a regression, wastes
    # the user's time and buries the actual cause.
    frac = info["crack_fraction"]
    if info["blocks_dropped_for_width"]:
        return dict(ok=False, info=info,
                    error=(f"{info['blocks_dropped_for_width']} label block(s) had the "
                           f"wrong feature width and were dropped. This means the "
                           f"ground truth did not reach training. Not training a model "
                           f"on partial data."))
    if not (0.05 <= frac <= 0.95):
        return dict(ok=False, info=info,
                    error=(f"training set is {frac*100:.1f}% crack -- degenerate. "
                           f"Paint some of BOTH kinds: 'Add crack' on real cracks and "
                           f"'Eraser' on false positives. Refusing to train."))
    warn = None
    if not (0.42 <= frac <= 0.58):
        warn = (f"training set is {frac*100:.1f}% crack, outside the 42-58% band; "
                f"class_weight='balanced' will skew the boundary. Paint more of the "
                f"under-represented kind for a better result.")

    if progress:
        progress("fitting", 0, 1)
    # WHY NOT HistGradientBoosting, WHICH LOOKED BETTER.
    #
    # It measured better on every axis scored over LABELLED pixels -- grouped-by-image
    # IoU@0.5 0.7642 against 0.7541, AUC 0.9634 against 0.9529, and cross-group AUC 0.897
    # against 0.863, four times the noise floor. It was deployed on that basis and the gate
    # caught it: predicted area on the six specimens confirmed to contain no crack rose from
    # 0.26% to 1.98%, every one of the six worse.
    #
    # Attributed with four arms at a fixed 400 k-row budget (research/fp_attribution.json),
    # crossing {reference frames in, out} with {HGB, this ensemble}:
    #
    #   arm                        reference IoU   crack-free FP
    #   no-GT  + HGB                   0.748          2.141%
    #   no-GT  + this ensemble         0.714          0.451%
    #   with-GT + HGB                  0.869*         2.540%
    #
    #   * contaminated: that arm trains on the frames it is scored on.
    #
    # So it is the classifier, not the training composition. The reason is what the two are
    # scored on: every metric that favoured HGB is computed over labelled pixels, and
    # crack-free specimen is precisely the material nobody labels. HGB separates the labelled
    # distribution better and behaves far worse off it. An AUC over labels cannot see that,
    # which is the whole argument for keeping a false-positive axis measured on unlabelled
    # material that is known to contain nothing.
    clf = Pipeline([("scaler", StandardScaler(copy=False)),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                          random_state=0))])
    # copy=False so StandardScaler standardises in place instead of returning a second
    # 5.9 GB array inside fit.
    clf.fit(X, y)

    # THE 17-FEATURE MEMBER IS TRAINED HERE, NOT INHERITED.
    #
    # It used to be a research-phase artifact reused by
    # reference, and every script that produced it trained on masks this project no longer
    # uses. Because the deployed model is a mean-probability ensemble, that meant half of
    # every prediction came from a model fitted on labels the owner did not draw -- and the
    # file carries no provenance metadata, so nothing in the app said so.
    #
    # Fitting it on the same correction rows costs one extra MLP on 17 columns, cheap beside
    # the 273-column fit above, and makes BOTH members of the ensemble a function of the
    # owner's own labelling and nothing else.
    if progress:
        progress("fitting the 17-feature member", 0, 1)
    clf17 = Pipeline([("scaler", StandardScaler()),
                      ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                                            random_state=0))])
    clf17.fit(X[:, :17], y)

    # The gate's honest axis, on the same rows, before they are freed.
    if progress:
        progress("cross-validation, grouped by image", 0, 1)
    try:
        heldout = crossval_on_rows(X, y, groups, progress=progress)
    except Exception as e:                                      # noqa: BLE001
        heldout, heldout_err = None, f"{type(e).__name__}: {e}"
    else:
        heldout_err = None
    del X, y, groups

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(S.MODELS, f"hybrid_{stamp}.joblib")
    out17 = os.path.join(S.MODELS, f"f17_{stamp}.joblib")
    joblib.dump(clf17, out17)
    # info already carries n_features; splat it first and let explicit keys win,
    # rather than passing n_features twice (which is a TypeError, not a merge).
    bundle = dict(info)
    bundle.update(model=clf, kind="sam17_hybrid", trained=stamp, recipe=RECIPE)
    joblib.dump(bundle, out)

    result = dict(ok=True, path=out, path_17=out17, info=info, warning=warn,
                  built_features=built or None,
                  seconds=round(time.time() - t0, 1))

    result["heldout"] = heldout
    result["heldout_error"] = heldout_err
    if heldout is None:
        result.update(deployed=False,
                      reason=f"cross-validation could not run ({heldout_err}); model saved "
                             f"but not deployed")
        record_retrain(result, stamp=stamp)
        return result

    cand_entry = dict(kind="ensemble", path_17=out17, path_hybrid=out, recipe=RECIPE,
                      label=f"retrained {stamp}", created=stamp)
    inc = get_model()
    cand = M.CrackModel(path_17=cand_entry["path_17"], path_hybrid=out, ensemble=True)
    # There is no dedicated labelled test set any more, by design: nothing is held back from
    # training, and the grouped-by-image cross-validation above is the generalisation
    # estimate. It is reported under `candidate` so the scorecard keeps one shape.
    result.update(candidate=dict(iou=heldout["mean_iou"], recall=heldout["mean_recall"]),
                  incumbent=None)

    # The IoU half of the gate, and then the half that was documented but never
    # implemented. FP_TOL sat unused while the docstring above promised "a candidate must
    # hold IoU AND not increase false positives on known-clean specimens". It did not, and
    # a model that marks 22% of a confirmed crack-free specimen as crack was deployed on
    # that basis -- against a shipped baseline that marks 0.21%. IoU on four images of one
    # specimen group cannot see that: over-prediction on OTHER specimens costs it nothing.
    if progress:
        progress("false-positive check on crack-free specimens", 0, 1)
    fp_inc, n_clean, det_inc = _score_clean(
        inc, progress=progress, cache_key=S.model_key(S.registry().get('current')))
    fp_cand, _, det_cand = _score_clean(cand, progress=progress)
    # Per-specimen, not just the mean. A mean that moves 0.14% -> 0.26% says nothing about
    # WHERE, and in practice the rise concentrates on one or two specimens -- which is the
    # difference between "slightly softer everywhere" and "one specimen fell apart".
    by_spec = []
    lookup = {d["image"]: d["fp"] for d in det_inc}
    for d in det_cand:
        by_spec.append(dict(image=d["image"], before=lookup.get(d["image"]), after=d["fp"]))
    result.update(clean_specimens=n_clean,
                  incumbent_clean_fp=fp_inc, candidate_clean_fp=fp_cand,
                  clean_fp_by_specimen=by_spec)
    try:
        result["false_indications"] = false_indications()
    except Exception:                                           # noqa: BLE001
        result["false_indications"] = None

    # THE GATE, WITHOUT ANY EXTERNALLY-LABELLED TEST SET.
    #
    # There used to be a first axis scoring candidates against four pre-existing B2 masks that
    # came from another tool. Those are not used anywhere in this project any more -- not for
    # training, not for evaluation -- so the axis is gone rather than reinterpreted. What is
    # left are the two axes that depend only on the owner's own labels and on material
    # confirmed to contain nothing:
    #
    #   heldout : grouped-by-image cross-validation on the training rows. Train and test never
    #             share an image, which is the only split this data supports honestly.
    #   fp      : predicted area on the confirmed crack-free specimens. Needs no labels at all
    #             -- every prediction there is a false positive by construction -- and it is
    #             the axis that caught a model marking 22% of blank specimen as crack, and
    #             later caught HistGradientBoosting at 7.9x the baseline.
    ho_ok, ho_prev = True, None
    try:
        # SAME RECIPE ONLY. A stored number from another architecture or another label corpus
        # is not a baseline: an earlier run of this gate rejected an honest candidate for
        # "regressing" against a figure measured months earlier by a different model.
        _hist = [h for h in retrain_history()
                 if (h.get("heldout") or {}).get("mean_iou") and h.get("recipe") == RECIPE]
        ho_prev = _hist[-1]["heldout"]["mean_iou"] if _hist else None
    except Exception:                                           # noqa: BLE001
        ho_prev = None
    ho_now = heldout["mean_iou"]
    if ho_prev is not None:
        ho_ok = ho_now >= ho_prev - IOU_TOL
        ho_basis = (f"no-regression against the last {RECIPE} run "
                    f"({ho_prev:.3f}, tolerance {IOU_TOL})")
    else:
        ho_ok = ho_now >= MIN_ABS_IOU
        ho_basis = (f"absolute floor {MIN_ABS_IOU:.2f}: no previous {RECIPE} run exists, so "
                    f"this one establishes the baseline")

    if n_clean and fp_inc is not None and fp_cand is not None:
        fp_ok = fp_cand <= fp_inc + FP_TOL
    else:
        fp_ok = True                       # cannot check; reported below, not hidden

    passes = bool(fp_ok and ho_ok)
    result["passes_gate"] = passes
    result["gate_detail"] = dict(
        fp_ok=bool(fp_ok), heldout_ok=bool(ho_ok), heldout_basis=ho_basis,
        recipe=RECIPE,
        incumbent_recipe=(S.registry().get("current") or {}).get("recipe"),
        external_labels_used=False,
        heldout_prev=ho_prev, heldout_now=ho_now,
        fp_checked_on=n_clean,
        note=(None if n_clean else
              "no crack-free specimen is loaded, so over-prediction was NOT checked -- "
              "load one of " + ", ".join(CLEAN_SPECIMENS[:3]) + ", ... to enable it"))
    if deploy and passes:
        S.set_current(cand_entry)
        _model_cache["key"] = None
        result.update(deployed=True)
        # Re-predict every image inside this job, rather than leaving it to the
        # browser to call /api/reoverlay afterwards. If that call never happens --
        # the tab was closed, the laptop slept, the page was reloaded during the
        # retrain -- the registry says the new model is current while every mask on
        # screen is still the old model's output, with nothing indicating the
        # mismatch. The embeddings are cached, so this is the classifier pass only.
        ids = [m["id"] for m in S.list_images()]
        for k, iid in enumerate(ids, 1):
            if progress:
                progress(f"re-applying to {iid} ({k}/{len(ids)})", k, len(ids))
            try:
                ingest(iid)
            except Exception as e:                        # noqa: BLE001
                # One unreadable image must not abandon the rest with stale masks.
                result.setdefault("reapply_failed", []).append(f"{iid}: {e}")
        result["reapplied"] = len(ids) - len(result.get("reapply_failed", []))
    else:
        bits = []
        if not ho_ok:
            bits.append(f"grouped-by-image IoU {ho_now:.3f} failed [{ho_basis}]"
                        + (f", previously {ho_prev:.3f}" if ho_prev is not None else ""))
        if not fp_ok:
            bits.append(f"false positives on {n_clean} crack-free specimen(s) rose "
                        f"{fp_inc*100:.2f}% -> {fp_cand*100:.2f}% of area "
                        f"(tolerance {FP_TOL*100:.1f} points)")
        result.update(deployed=False,
                      reason=(None if passes else
                              "; ".join(bits) + ". Not deployed. The model file is kept "
                              "so you can inspect it."))
    # Record the scorecard whether or not it deployed. A REJECTED retrain is the most
    # useful entry in the history -- it is the one a person will want to look at again.
    record_retrain(result, stamp=stamp)
    return result
