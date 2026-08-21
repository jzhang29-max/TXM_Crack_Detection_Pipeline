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
import threading
import time
import zipfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
CODE = os.path.join(PROJECT, "code")
for p in (CODE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import store as S            # noqa: E402
import model as M            # noqa: E402

# Reference data that ships with the package: the only pixel-level ground truth
# that exists (4 images), used to validate a retrain. If it is absent the gate
# degrades to "warn and refuse to auto-deploy" rather than silently passing.
GT_CACHE = os.path.join(PROJECT, "dataset_cache")
# The four that ship with the repo. Kept as the floor rather than the whole list, because
# any new densely-annotated stem dropped into dataset_cache has to be picked up by the gate,
# the cross-validation and the feature builder at once -- and hardcoding meant adding ground
# truth required editing four call sites, which is how a stem gets used by one and missed by
# another.
GT_STEMS_SHIPPED = ["333_75_um_zoom", "336_25", "338_13", "LARGE_343_75"]


def _discover_gt_stems():
    """Every stem in dataset_cache that has BOTH an image and a mask."""
    found = list(GT_STEMS_SHIPPED)
    try:
        for f in sorted(os.listdir(GT_CACHE)):
            if not f.endswith("_gt.npy"):
                continue
            stem = f[: -len("_gt.npy")]
            if stem in found:
                continue
            if os.path.exists(os.path.join(GT_CACHE, f"{stem}_img.npy")):
                found.append(stem)
    except OSError:
        pass
    return found


GT_STEMS = _discover_gt_stems()

# THE REFERENCE FRAMES ARE THE TEST SET, NOT TRAINING DATA.
#
# Until this change every model trained on the four dense B2 frames and was then validated
# against those same four, so the gate was grading with part of the answer key in the
# training set. Measured cost of taking them out (code/experiment_no_gt.py, 3 repeats,
# leave-one-group-out over AM/HC, B2, B3, wrought): cross-group AUC 0.871 without them
# against 0.863 with, a difference at the +-0.008 noise floor -- i.e. nothing. Measured cost
# on the B2 frames themselves: IoU 0.741 against 0.768, real but small. What is bought is a
# gate whose number means something.
#
# Excluded by SPECIMEN, not by image. All four frames are also loaded in the app with the
# owner's own corrections, and b2_343_75 and b2_343_75_LARGE are two fields of view of one
# specimen -- so training on one while testing on the other leaks, and the leak flatters.
# Five of 71 images are held out this way. Measured to cost nothing: raising the training
# budget from 150 k rows to 500 k did not improve either test (0.856 vs 0.871 cross-group
# AUC), so this project is not short of labels.
REFERENCE_SPECIMENS = ("333_75", "336_25", "338_13", "343_75")

# Tag on every model the retrainer produces, so the gate can tell whether the model it is
# comparing against was trained under the same rules. A model trained WITH the reference
# frames scores them in-sample; comparing a clean candidate to it on those frames would
# reject the clean one for being honest.
RECIPE = "mlp_ens_nogt"
MIN_ABS_IOU = 0.60


def is_reference_image(filename):
    """True if this image belongs to a specimen the reference frames come from."""
    n = (filename or "").lower()
    return any(t in n for t in REFERENCE_SPECIMENS)

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
        zp = S.path(m["id"], "emb.npz")
        if model.needs_sam() and os.path.exists(zp):
            z = np.load(zp)
            emb = (z["coords"], z["emb"])
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
            and not os.path.exists(embp):
        # Already known unreachable in this process: skip straight to the 17-feature model
        # rather than attempting a 2.4 GB download once per image.
        sam_note = M.sam_unavailable_reason or "disabled by TXM_NO_SAM=1"
        mdl = M.CrackModel(path_17=M.DEFAULT_17, path_hybrid="", ensemble=False)
    if mdl.needs_sam() and (force or not os.path.exists(embp)):
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
        if coords is None:
            pass
        # Written atomically, the same temp+fsync+replace dance store.save_npy does twenty
        # lines away. This used to be a bare np.savez straight onto the final path, and a
        # SAM embedding is 8-59 MB: quit or lose power during that write and the file is a
        # truncated zip forever. The image was then BRICKED with no route back from the UI
        # -- Re-apply calls ingest() without force so it takes the existing broken file,
        # re-dropping produces the same content-hash id, and the only escape (Remove and
        # re-drop) silently discards that image's corrections. The same corrupt file then
        # killed the next retrain an hour in, while gathering features.
        if coords is not None:
            tmp = f"{embp}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
            try:
                with open(tmp, "wb") as fh:
                    np.savez(fh, coords=coords, emb=emb)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, embp)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        del coords, emb

    rep("predicting")
    emb = None
    if mdl.needs_sam() and os.path.exists(embp):
        try:
            z = np.load(embp)
            emb = (z["coords"], z["emb"])
        except (zipfile.BadZipFile, ValueError, EOFError, KeyError) as e:
            # A damaged cache must not be a dead end. Recompute it rather than raising:
            # this file is derived data, so throwing it away costs seconds of GPU and
            # nothing else, while raising cost the user the whole image.
            rep(f"SAM cache damaged ({type(e).__name__}), recomputing")
            try:
                os.unlink(embp)
            except OSError:
                pass
            coords, emb2 = M.embed_image(img01,
                                         progress=lambda k, n: rep("SAM embedding", k, n))
            tmp = f"{embp}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
            with open(tmp, "wb") as fh:
                np.savez(fh, coords=coords, emb=emb2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, embp)
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


def prune_specks(mask, min_px=None):
    """Drop connected components below `min_px`. Never touches anything else."""
    from skimage import morphology
    n = MIN_BLOB_PX if min_px is None else min_px
    if not n or not mask.any():
        return mask
    return morphology.remove_small_objects(mask, min_size=int(n))


def effective_mask(image_id, threshold=0.5, postprocess=False, prune=True):
    """Model prediction with the user's corrections applied on top."""
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
    corr = S.load_npy(image_id, "correction.npy")
    if corr is not None and corr.shape == mask.shape:
        mask = mask.copy()
        mask[corr == 1] = True
        mask[corr == 2] = False
    return mask


# ------------------------------------------------------------------ retrain
def _gt_available():
    return all(os.path.exists(os.path.join(GT_CACHE, f"{s}_{k}.npy"))
               for s in GT_STEMS for k in ("img", "gt"))


def ensure_gt_features(progress=None):
    """Build the reference 17-feature stacks if they are missing. Returns what it built.

    These are 2.1 GB (1.5 GB of it for the 23 MP mosaic) and a pure function of the
    images, so they are neither shipped nor built at startup -- doing it on first run
    added minutes to `./run_app.sh` for something only retraining ever reads. They are
    built here instead, once, with progress, on the first retrain.

    This has to happen BEFORE the bootstrap loop below, which skips any stem whose
    feature file is absent. That skip is silent, and dropping the ground truth from
    training is precisely the failure that once produced a 100%-crack model scoring
    IoU 0.003 -- so "build it on demand" and "skip it quietly" must not be confused.
    """
    from txm_features import compute_feature_stack
    built = []
    for stem in GT_STEMS:
        img_p = os.path.join(GT_CACHE, f"{stem}_img.npy")
        feat_p = os.path.join(GT_CACHE, f"{stem}_features.npy")
        if os.path.exists(feat_p) or not os.path.exists(img_p):
            continue
        if progress:
            progress(f"preparing reference features: {stem} (first retrain only)", 0, 1)
        img = np.load(img_p)
        tmp = feat_p + ".tmp.npy"
        np.save(tmp, compute_feature_stack(img).astype(np.float32))
        os.replace(tmp, feat_p)                 # never leave a partial stack behind
        del img
        built.append(stem)
    return built


GT_EMB_DIR = os.path.join(S.DATA, "gt_emb")


def gt_embedding(stem, progress=None):
    """SAM embedding for a shipped ground-truth image, computed once and cached.

    Also looks in the research cache (paint/sam_embcache) first, since those 4
    images were embedded there under their full original filenames -- reusing
    that avoids ~20s of GPU work per image on a user's first retrain.
    """
    os.makedirs(GT_EMB_DIR, exist_ok=True)
    p = os.path.join(GT_EMB_DIR, f"{stem}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["coords"], z["emb"]

    legacy = os.path.join(PROJECT, "paint", "sam_embcache")
    if os.path.isdir(legacy):
        key = stem.replace("LARGE_343_75", "343_75_LARGE")
        for f in sorted(os.listdir(legacy)):
            if key in f and f.endswith("_samemb.npz"):
                try:
                    z = np.load(os.path.join(legacy, f))
                    coords, emb = z["coords"], z["emb"]
                    np.savez(p, coords=coords, emb=emb)
                    return coords, emb
                except Exception:
                    break

    img = np.load(os.path.join(GT_CACHE, f"{stem}_img.npy"))
    if progress:
        progress(f"embedding ground truth {stem}", 0, 1)
    coords, emb = M.embed_image(img)
    np.savez(p, coords=coords, emb=emb)
    del img
    return coords, emb


def _score(model, progress=None):
    """(mean IoU, mean recall) on the four reference frames -- a HELD-OUT set since the
    recipe change: nothing trains on them, or on corrections from their specimens."""
    from generate_benchmark_report import metrics_from_pred
    ious, recs = [], []
    for i, stem in enumerate(GT_STEMS):
        img = np.load(os.path.join(GT_CACHE, f"{stem}_img.npy"))
        gt = np.load(os.path.join(GT_CACHE, f"{stem}_gt.npy")).astype(bool)
        prob = model.predict(img)
        s = metrics_from_pred(prob > 0.5, gt)
        ious.append(s["iou"]); recs.append(s["recall"])
        if progress:
            progress("validating", i + 1, len(GT_STEMS))
        del img, gt, prob
    return float(np.mean(ious)), float(np.mean(recs))


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
CV_MAX_FOLDS = 5
LABEL_FOLDS = os.path.join(PROJECT, "paint", "label_folds.npz")
CV_TRAIN_CAP = 90000


def _gt_rows(stem, n, rng):
    """n pixels sampled UNIFORMLY from one ground-truth image, as [17 | 256] features.

    Uniform, not class-balanced, so the sample's crack fraction equals the real image's and
    IoU measured on it is an unbiased estimate of the whole-image value.
    """
    gt = np.asarray(np.load(os.path.join(GT_CACHE, f"{stem}_gt.npy"), mmap_mode="r")).astype(bool)
    feat_p = os.path.join(GT_CACHE, f"{stem}_features.npy")
    if not os.path.exists(feat_p):
        return None
    feat = np.load(feat_p, mmap_mode="r")
    if feat.shape[:2] != gt.shape:
        return None
    H_, W_ = gt.shape
    idx = np.sort(rng.choice(H_ * W_, min(n, H_ * W_), replace=False))
    rr, cc = np.unravel_index(idx, (H_, W_))
    x17 = np.asarray(feat[rr, cc, :], np.float32)
    del feat
    block = x17
    if get_model().needs_sam():
        coords, embs = gt_embedding(stem)
        if coords is not None:
            b = np.zeros((len(rr), embs.shape[1]), np.float32)
            todo = np.ones(len(rr), bool)
            for t in range(len(coords) - 1, -1, -1):
                y0, x0 = int(coords[t][0]), int(coords[t][1])
                sel = (todo & (rr >= y0) & (rr < y0 + M.TILE)
                       & (cc >= x0) & (cc < x0 + M.TILE))
                if sel.any():
                    b[sel] = M.interp_tile(embs[t], rr[sel] - y0, cc[sel] - x0)
                    todo &= ~sel
            block = np.concatenate([x17, b], axis=1)
    return block, gt.ravel()[idx], x17.shape[1]


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


def crossval_grouped(progress=None):
    """Honest generalisation estimate: k-fold GROUPED BY IMAGE, k = number of GT images.

    WHY NOT ORDINARY k-FOLD. Splitting pixels at random leaks, badly. The 17 hand-crafted
    features are computed from neighbourhoods reaching 256 px, and a SAM embedding is a
    bilinear lookup into a 64x64 grid per 1024-px tile, so a 16x16 block of pixels shares
    essentially one embedding vector. A randomly held-out pixel therefore almost always has
    a training pixel a few pixels away carrying the same measurement. Measured on this data
    with the deployed architecture: random 4-fold gives IoU 0.930 with a fold sd of 0.003,
    grouping by image gives 0.824 with a fold sd of 0.050. Random k-fold does not merely
    fail to reveal overfitting, it inflates the score by 0.106 and reports a suspiciously
    tight spread while doing it.

    Grouping by image is the coarsest split the data supports and the only one where train
    and test share no neighbourhood. With four ground-truth images it is leave-one-image-out.

    WHAT THIS NUMBER IS AND IS NOT. It measures THE ARCHITECTURE PLUS THE GROUND TRUTH, at
    a capped row count, refit from scratch per fold. It is not the deployed model's own
    score -- that model saw all four images, so it has no honest score and never can.
    Read this as "what a model built this way scores on an image it has not seen".

    n = 4, and the fold sd is ~0.05. Differences under ~0.015 are reseeding noise.
    """
    if not _gt_available():
        return None
    from sklearn.model_selection import GroupKFold
    rng = np.random.RandomState(0)
    Xs, ys, gs = [], [], []
    for gi, stem in enumerate(GT_STEMS):
        if progress:
            progress(f"cross-validation: sampling {stem}", gi, len(GT_STEMS))
        got = _gt_rows(stem, CV_ROWS_PER_IMAGE, rng)
        if got is None:
            continue
        block, y, n17 = got
        Xs.append(block); ys.append(y); gs.append(np.full(len(y), gi))

    # The owner's own labels, when code/build_label_folds.py has been run. Without them this
    # number depends only on the four shipped ground-truth images, so it cannot move when
    # someone labels more -- which makes it useless as a gate. With them it responds to the
    # actual corpus, and each labelled image is its own GROUP so no image is ever on both
    # sides of a fold.
    extra_groups = 0
    if os.path.exists(LABEL_FOLDS):
        try:
            z = np.load(LABEL_FOLDS, allow_pickle=False)
            ids = sorted({k.split("|")[0] for k in z.files})
            base = len(GT_STEMS)
            for j, iid in enumerate(ids):
                kx, ks, ky = f"{iid}|x17", f"{iid}|xsam", f"{iid}|y"
                if not (kx in z.files and ks in z.files and ky in z.files):
                    continue
                blk = np.concatenate([z[kx], z[ks]], axis=1)
                if Xs and blk.shape[1] != Xs[0].shape[1]:
                    continue                       # width mismatch: skip, never pad
                Xs.append(blk.astype(np.float32))
                ys.append(z[ky].astype(bool))
                gs.append(np.full(len(z[ky]), base + j))
                extra_groups += 1
            if progress and extra_groups:
                progress(f"cross-validation: +{extra_groups} labelled images", 1, 1)
        except (OSError, ValueError, KeyError):
            extra_groups = 0                       # a damaged cache must not sink a retrain
    if len(Xs) < 2:
        return None
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    del Xs, ys, gs
    # Grouped by image, but capped: with 71 labelled images leave-one-image-out would mean
    # 75 refits per retrain. GroupKFold with k folds keeps whole images together while
    # holding the refit count fixed, which is the property that matters.
    n_groups = len(np.unique(g))
    k = min(CV_MAX_FOLDS, n_groups)

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def _clf():
        """Same architecture as the deployed model, so this number describes what ships.

        Kept deliberately in step: this was briefly a single HistGradientBoosting, to match a
        deployed model that then failed the false-positive gate and was reverted. A held-out
        figure measuring an architecture other than the one shipping is worse than none,
        because it reads as though it had been checked.
        """
        return Pipeline([("scaler", StandardScaler()),
                         ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                               random_state=0, early_stopping=True,
                                               n_iter_no_change=8))])

    per_fold = []
    for f, (tr, te) in enumerate(GroupKFold(k).split(X, groups=g), 1):
        # Name the fold by WHAT IT HOLDS, not by its first group. With 75 groups over 5
        # folds each fold holds ~15 images, so labelling it after one ground-truth stem
        # read as if that single image were the test set.
        gids = sorted(set(int(v) for v in g[te]))
        gt_in = [GT_STEMS[i] for i in gids if i < len(GT_STEMS)]
        held = (f"{len(gids)} images" + (f" incl. {', '.join(gt_in)}" if gt_in else ""))
        if progress:
            progress(f"cross-validation fold {f}/{k}: holding out {held}", f, k)
        if len(tr) > CV_TRAIN_CAP:
            tr = np.random.RandomState(7).choice(tr, CV_TRAIN_CAP, replace=False)
        # The deployed architecture: mean probability of a 17-feature model and the hybrid.
        probs = []
        for cols in (slice(0, n17), slice(0, X.shape[1])):
            probs.append(_clf().fit(X[tr, cols], y[tr]).predict_proba(X[te, cols])[:, 1])
        prob = np.mean(probs, axis=0)
        pred = prob >= 0.5
        tp = int((pred & y[te]).sum()); fp = int((pred & ~y[te]).sum())
        fn = int((~pred & y[te]).sum())
        per_fold.append(dict(held_out=held, n=int(len(te)),
                             iou=round(tp / max(tp + fp + fn, 1), 4),
                             precision=round(tp / max(tp + fp, 1), 4),
                             recall=round(tp / max(tp + fn, 1), 4)))
    ious = [f["iou"] for f in per_fold]
    # `mean_iou` is agreement on JUDGED pixels: labels are dense ground truth on the four
    # shipped images and the owner's own corrections elsewhere. It is not IoU against
    # physical truth, and the scorecard says so.
    return dict(k=k, per_fold=per_fold, labelled_images_used=extra_groups,
                label_source=("ground truth + owner corrections" if extra_groups
                              else "ground truth only"),
                mean_iou=round(float(np.mean(ious)), 4),
                std_iou=round(float(np.std(ious)), 4),
                min_iou=round(float(np.min(ious)), 4),
                mean_precision=round(float(np.mean([f["precision"] for f in per_fold])), 4),
                mean_recall=round(float(np.mean([f["recall"] for f in per_fold])), 4),
                rows_per_image=CV_ROWS_PER_IMAGE, train_cap=CV_TRAIN_CAP,
                grouped_by="image")


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

    NOT the shipped ground truth, and not corrections on the specimens it comes from:
    those four frames are the held-out test set. See REFERENCE_SPECIMENS.

    Class balance is computed from what actually exists rather than hard-coded:
    it is the single knob that has caused four regressions in this project, and
    a fixed cap silently goes stale as more images get labelled.
    """
    import destitch  # noqa: F401  (ensures code/ is importable before heavy work)
    Xs, ys = [], []

    labelled = [m for m in S.list_images()
                if m.get("corrected_crack_px") or m.get("corrected_not_px")]
    items = [m for m in labelled if not is_reference_image(m.get("filename"))]
    held_out = [m.get("filename") for m in labelled if is_reference_image(m.get("filename"))]
    n_crack_total = sum(m.get("corrected_crack_px", 0) for m in items)
    per_img_cap = 30000
    corr_crack = sum(min(per_img_cap, m.get("corrected_crack_px", 0)) for m in items)
    n_bg_imgs = sum(1 for m in items if m.get("corrected_not_px", 0) > 0)

    neg_cap = max(500, int(round(corr_crack / n_bg_imgs))) if n_bg_imgs else per_img_cap

    rng = np.random.RandomState(0)

    # NO GROUND-TRUTH ROWS. They used to be sampled here, 100 k crack + 100 k background
    # per stem, and they were the largest single block in the training set. They are the
    # test set now -- see REFERENCE_SPECIMENS above for what that cost and what it bought.
    # Corrections on the same specimens are excluded too, a few lines up, because those
    # would leak the test set back in through the side door.


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
        if not os.path.exists(zp):
            continue
        z = np.load(zp); coords, embs = z["coords"], z["emb"]
        b = np.zeros((len(rr), embs.shape[1]), np.float32)
        todo = np.ones(len(rr), bool)
        for t in range(len(coords) - 1, -1, -1):
            y0, x0 = int(coords[t][0]), int(coords[t][1])
            sel = todo & (rr >= y0) & (rr < y0 + M.TILE) & (cc >= x0) & (cc < x0 + M.TILE)
            if sel.any():
                b[sel] = M.interp_tile(embs[t], rr[sel] - y0, cc[sel] - x0)
                todo &= ~sel
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
    at = 0
    for i in keep_i:
        blk, lab = Xs[i], ys[i]
        n = blk.shape[0]
        X[at:at + n] = blk
        y[at:at + n] = lab
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
    return X, y, info


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
    built = ensure_gt_features(progress=progress)
    if progress:
        progress("gathering labels", 0, 1)
    X, y, info = gather_training_data(progress=progress)
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
    del X, y

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(S.MODELS, f"hybrid_{stamp}.joblib")
    # info already carries n_features; splat it first and let explicit keys win,
    # rather than passing n_features twice (which is a TypeError, not a merge).
    bundle = dict(info)
    bundle.update(model=clf, kind="sam17_hybrid", trained=stamp, recipe=RECIPE)
    joblib.dump(bundle, out)

    result = dict(ok=True, path=out, info=info, warning=warn,
                  built_features=built or None,
                  seconds=round(time.time() - t0, 1))

    if not _gt_available():
        result.update(deployed=False,
                      reason="no ground truth available to validate against; "
                             "model saved but not deployed")
        record_retrain(result, stamp=stamp)
        return result

    cand_entry = dict(kind="ensemble", path_17=M.DEFAULT_17, path_hybrid=out, recipe=RECIPE,
                      label=f"retrained {stamp}", created=stamp)
    inc = get_model()
    if progress:
        progress("validating incumbent", 0, 1)
    i0, r0 = _score(inc, progress=progress)
    cand = M.CrackModel(path_17=cand_entry["path_17"], path_hybrid=out, ensemble=True)
    if progress:
        progress("validating candidate", 0, 1)
    i1, r1 = _score(cand, progress=progress)
    result.update(incumbent=dict(iou=i0, recall=r0), candidate=dict(iou=i1, recall=r1))

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

    # WHETHER i0 IS EVEN A VALID BASELINE.
    #
    # i0 and i1 are both measured on the four reference frames. The candidate never trained
    # on them, so i1 is held out. But any model produced before this recipe trained ON them,
    # so its i0 is in-sample -- and comparing a clean candidate against an in-sample
    # incumbent rejects the candidate for being honest. That is not a tolerance to widen; the
    # two numbers are not the same quantity.
    #
    # So: same recipe -> the usual no-regression test. Different recipe -> the comparison is
    # void, and the candidate has to clear an absolute floor instead, with the reason
    # recorded rather than the check quietly skipped.
    inc_recipe = (S.registry().get("current") or {}).get("recipe")
    same_recipe = (inc_recipe == RECIPE)
    if same_recipe:
        iou_ok = i1 >= i0 - IOU_TOL
        iou_basis = f"no-regression against an incumbent measured the same way ({inc_recipe})"
    else:
        iou_ok = i1 >= MIN_ABS_IOU
        iou_basis = (f"absolute floor {MIN_ABS_IOU:.2f}: the incumbent's {i0:.3f} is IN-SAMPLE "
                     f"(recipe {inc_recipe or 'pre-recipe'}, trained on the reference frames) "
                     f"so it is not a valid baseline for a held-out {i1:.3f}")

    # The held-out half of the gate. i1 vs i0 above is IN-SAMPLE -- the candidate trains on
    # the same four ground-truth images it is scored on -- so "IoU did not drop" can be
    # satisfied by fitting them harder. This compares the candidate's grouped-by-image
    # cross-validated score against the last one recorded, which is the same protocol
    # measured the same way, and is the only IoU-like number here that cannot be gamed by
    # memorising the evaluation set.
    ho_ok, ho_prev, ho_now = True, None, None
    try:
        # SAME RECIPE ONLY. This axis had exactly the bug the reference-frame axis was fixed
        # for: it compared a fresh score against whatever was last recorded, regardless of
        # which architecture produced it or which label corpus it was measured on. The first
        # run under this recipe was rejected for "regressing" 0.8151 -> 0.7589 against a
        # figure measured on 2026-08-19 by a two-MLP average over a smaller corpus. Measured
        # on identical rows that architecture scores 0.7541, below the 0.7589 it rejected.
        # A stored number from another recipe or another corpus is not a baseline.
        _hist = [h for h in retrain_history()
                 if (h.get("heldout") or {}).get("mean_iou") and h.get("recipe") == RECIPE]
        ho_prev = _hist[-1]["heldout"]["mean_iou"] if _hist else None
    except Exception:                                           # noqa: BLE001
        ho_prev = None
    if n_clean and fp_inc is not None and fp_cand is not None:
        fp_ok = fp_cand <= fp_inc + FP_TOL
    else:
        fp_ok = True                       # cannot check; reported below, not hidden
    # crossval runs after the gate is decided elsewhere in this function, so compute it
    # here where its verdict can actually count.
    try:
        if progress:
            progress("cross-validation (grouped by image)", 0, 1)
        result["heldout"] = crossval_grouped(progress=progress)
    except Exception as e:                                      # noqa: BLE001
        result["heldout"] = None
        result["heldout_error"] = f"{type(e).__name__}: {e}"
    ho_now = (result.get("heldout") or {}).get("mean_iou")
    if ho_prev is not None and ho_now is not None:
        ho_ok = ho_now >= ho_prev - IOU_TOL
        ho_basis = f"no-regression against the last {RECIPE} run"
    else:
        ho_basis = (f"no previous {RECIPE} run to compare against -- this run establishes "
                    f"the baseline for this axis")
    passes = bool(iou_ok and fp_ok and ho_ok)
    result["passes_gate"] = passes
    result["gate_detail"] = dict(
        iou_ok=bool(iou_ok), fp_ok=bool(fp_ok), heldout_ok=bool(ho_ok),
        iou_basis=iou_basis, heldout_basis=ho_basis,
        recipe=RECIPE, incumbent_recipe=inc_recipe,
        reference_frames_held_out=True,
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
        if not iou_ok:
            bits.append(f"IoU {i1:.3f} on the held-out reference frames failed the check "
                        f"[{iou_basis}]")
        if not ho_ok:
            bits.append(f"held-out IoU regressed {ho_prev:.3f} -> {ho_now:.3f} "
                        f"(tolerance {IOU_TOL}, {ho_basis}) -- this one is grouped by "
                        f"image, so it cannot be recovered by fitting the evaluation "
                        f"set harder")
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
