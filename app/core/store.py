"""
Storage for the app. One directory per uploaded image, everything derived from
it alongside it, and a model registry with history.

Design constraint that drove this: the previous pipeline read images from a
hard-coded absolute path (~/Desktop/TXM DATA). When that folder went missing the
whole tool stopped working -- list_images() returned 0 and nothing could load.
So here the app OWNS its data: an uploaded image is copied inside app_data/ and
never referenced outside it again. Nothing breaks if you move or delete the
original.

Layout:
    app_data/
      images/<id>/original.<ext>   the file as uploaded, untouched
                  img.npy         normalised, MODEL input (raw path)
                  display.npy     destitched + flatfielded, HUMAN view
                  emb.npz         cached SAM embedding (the expensive part)
                  prob.npy        crack probability from the current model
                  correction.npy  uint8: 0 untouched, 1 force-crack, 2 force-not
                  meta.json
      models/registry.json        current model + retrain history
"""

import hashlib
import json
import os
import re
import shutil
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA = os.path.join(PROJECT, "app_data")
IMAGES = os.path.join(DATA, "images")
MODELS = os.path.join(DATA, "models")
REGISTRY = os.path.join(MODELS, "registry.json")

for d in (DATA, IMAGES, MODELS):
    os.makedirs(d, exist_ok=True)


def _slug(name):
    base = os.path.splitext(os.path.basename(name))[0]
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in base)[:80]
    return keep or "image"


def new_id(filename, content=None):
    """Stable id: slug + short content hash, so re-uploading the same file
    reuses its entry (and its expensive SAM embedding) instead of duplicating."""
    h = hashlib.sha1(content if content is not None else filename.encode()).hexdigest()[:8]
    return f"{_slug(filename)}__{h}"


# An image id reaches this module straight out of a URL segment, and every filesystem
# access in the app funnels through path(). Unvalidated, ".." was enough to walk out of
# app_data/images: path("..") resolves to app_data itself and path("../..") to the whole
# repository, both with os.path.isdir() true -- so DELETE /api/image/%2e%2e handed
# delete_image() a live directory and shutil.rmtree took every correction mask, the model
# registry and the retrain history with it. Werkzeug unquotes the path before routing and
# its default converter regex is [^/]+, which ".." matches, so the segment arrives intact.
#
# Validated HERE rather than at each route, because a single chokepoint cannot be forgotten
# by the next endpoint someone adds. Real ids are _slug() + a content hash, so they only
# ever contain [A-Za-z0-9._-]; every existing id passes unchanged.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,160}$")


def valid_id(image_id):
    sid = str(image_id)
    return bool(_SAFE_ID.match(sid)) and ".." not in sid


def path(image_id, *parts):
    if not valid_id(image_id):
        raise ValueError(f"bad image id: {image_id!r}")
    return os.path.join(IMAGES, image_id, *parts)


# --------------------------------------------------------- per-image locking
_locks = {}
_locks_guard = threading.Lock()


def image_lock(image_id):
    """Serialise read-modify-write of one image's correction array.

    server.py runs Flask with threaded=True, and every brush stroke is a
    load correction.npy -> modify -> save cycle. Two of those in flight at once --
    a fast painter whose strokes overlap, an undo pressed while a stroke is still
    saving, or two browser tabs on the same image -- both read the same array and the
    second save overwrites the first, which had already told the user "saved". The
    stroke is not just delayed, it is gone, and nothing anywhere reports a problem.

    Per image rather than one global lock, so painting one image never blocks
    predicting or exporting another.
    """
    with _locks_guard:
        lk = _locks.get(image_id)
        if lk is None:
            lk = _locks[image_id] = threading.Lock()
        return lk


def exists(image_id):
    return os.path.isdir(path(image_id))


def save_upload(filename, content):
    """Write an uploaded file into its own directory. Returns (id, is_new)."""
    iid = new_id(filename, content)
    d = path(iid)
    if os.path.isdir(d) and os.path.exists(path(iid, "meta.json")):
        return iid, False
    os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(filename)[1] or ".tif"
    with open(path(iid, "original" + ext), "wb") as f:
        f.write(content)
    write_meta(iid, dict(id=iid, filename=filename, ext=ext,
                         uploaded=time.time(), status="uploaded"))
    return iid, True


def read_meta(image_id):
    p = path(image_id, "meta.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def write_json(p, obj):
    """Atomic JSON write, for the same reason as save_npy.

    read_meta() swallows a parse error and returns {}, which is the right call for a
    missing file but means a half-written meta.json degrades silently into an image
    with no dimensions and no status. registry.json matters more still: it holds which
    model is current and the whole retrain history, and write_meta runs on every
    progress update during ingest, so these files are rewritten far more often than
    their importance suggests.
    """
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_meta(image_id, meta):
    cur = read_meta(image_id)
    cur.update(meta)
    write_json(path(image_id, "meta.json"), cur)
    return cur


def original_path(image_id):
    d = path(image_id)
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.startswith("original"):
            return os.path.join(d, f)
    return None


def load_npy(image_id, name, mmap=False):
    p = path(image_id, name)
    if not os.path.exists(p):
        return None
    return np.load(p, mmap_mode="r" if mmap else None)


def load_npy_at(p, mmap=False):
    """Load an .npy by explicit path, or None if it is not there.

    load_npy() addresses files by (image_id, name); the per-model prediction cache is
    addressed by path, and the retrain gate needs to read from it without pretending a
    cache entry is the image's live prob.npy.
    """
    if not p or not os.path.exists(p):
        return None
    return np.load(p, mmap_mode="r" if mmap else None)


def save_npy(image_id, name, arr):
    """Write atomically: full file to a temp name, fsync, then rename over the target.

    The app has no save button -- every brush stroke is a write to correction.npy --
    so this path runs constantly while someone works, and it rewrites the WHOLE array
    each time (2.9 MB for a typical image, 23 MB for the big mosaic). Writing in place
    meant a crash or power cut mid-write left a truncated file, and the loss is not the
    one stroke in flight, it is every correction ever painted on that image. rename(2)
    is atomic on the same filesystem, so a reader sees either the old file or the new
    one. Verified: SIGKILL immediately after a stroke, restart, corrections intact.
    """
    d = path(image_id)
    os.makedirs(d, exist_ok=True)
    final = path(image_id, name)
    # Unique staging name, not a fixed "<name>.tmp". The CLI tools
    # (import_research_corrections.py, backup_labels.py --restore) are separate
    # PROCESSES that write correction.npy for a live app on purpose -- they delete the
    # overlay caches afterwards so the labels appear immediately in the open browser --
    # and a process boundary is exactly where image_lock() stops helping, since it is an
    # in-interpreter threading.Lock. With one shared temp path, a CLI write and a brush
    # stroke land in the same file: one replace() wins and the loser keeps writing into
    # the inode that is now the live correction.npy, which defeats the whole point of
    # staging. pid+thread is already how _link_or_copy names its temp, for the same reason.
    tmp = f"{final}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
            fh.flush()
            os.fsync(fh.fileno())          # without this the rename can beat the data to disk
        os.replace(tmp, final)
    except BaseException:
        # Never leave a stray .tmp behind to be mistaken for real data later.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ------------------------------------------------------------------ undo
# Undo stores a DELTA per stroke, not a snapshot. A correction array is uint8 at
# full image resolution -- 2.9 MB for a 2.9 MP image, 23 MB for the big mosaic --
# so twenty snapshots of the large one would be half a gigabyte. A delta is the
# bounding box of the stroke plus the pixel values that were there before, which
# for a brush stroke is a few hundred kilobytes at most.
UNDO_DEPTH = 30


def _undo_dir(image_id):
    d = path(image_id, "undo")
    os.makedirs(d, exist_ok=True)
    return d


def push_undo(image_id, y0, y1, x0, x1, prev_block):
    """Record what a region looked like BEFORE a stroke overwrote it."""
    d = _undo_dir(image_id)
    n = len(os.listdir(d))
    np.savez_compressed(os.path.join(d, f"{time.time():.6f}_{n}.npz"),
                        box=np.asarray([y0, y1, x0, x1], np.int64), prev=prev_block)
    files = sorted(os.listdir(d))
    for f in files[:-UNDO_DEPTH]:            # keep the stack bounded
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass


def pop_undo(image_id):
    """Undo the most recent stroke. Returns True if something was undone."""
    d = _undo_dir(image_id)
    files = sorted(os.listdir(d))
    if not files:
        return False
    p = os.path.join(d, files[-1])
    try:
        z = np.load(p)
        y0, y1, x0, x1 = [int(v) for v in z["box"]]
        prev = z["prev"]
    except Exception:
        os.remove(p)
        return False
    corr = load_npy(image_id, "correction.npy")
    if corr is None:
        os.remove(p)
        return False
    corr = np.asarray(corr).copy()
    if corr[y0:y1, x0:x1].shape == prev.shape:
        corr[y0:y1, x0:x1] = prev
        save_npy(image_id, "correction.npy", corr)
    os.remove(p)
    return True


def undo_depth(image_id):
    d = path(image_id, "undo")
    return len(os.listdir(d)) if os.path.isdir(d) else 0


def clear_undo(image_id):
    d = path(image_id, "undo")
    if os.path.isdir(d):
        shutil.rmtree(d)


def correction_counts(image_id, meta=None):
    """(force-crack px, force-not px) for one image, memoised in meta.json.

    This used to sum the whole correction array on every call, and list_images() calls
    it once per image. At 2 images that was 9 ms; at 71 it was 400 ms of summing 205 MB,
    on an endpoint the frontend hits after EVERY stroke and every 900 ms while a job
    runs. The counts only change when correction.npy does, so they are cached against
    that file's mtime -- which means a write by any path (a stroke, an undo, ingest
    zeroing it, a script) invalidates them without having to remember to update.
    """
    p = path(image_id, "correction.npy")
    if not os.path.exists(p):
        return 0, 0
    mt = os.path.getmtime(p)
    m = meta if meta is not None else read_meta(image_id)
    c = m.get("corr_counts")
    if isinstance(c, dict) and c.get("mtime") == mt:
        return int(c.get("crack", 0)), int(c.get("not", 0))
    a = np.asarray(np.load(p, mmap_mode="r"))
    n_crack, n_not = int((a == 1).sum()), int((a == 2).sum())
    del a
    write_meta(image_id, dict(corr_counts=dict(crack=n_crack, **{"not": n_not}, mtime=mt)))
    return n_crack, n_not


def list_images():
    out = []
    # Which model the app claims to be using, so each image can be compared against it.
    cur_key = model_key(registry().get("current"))
    for iid in sorted(os.listdir(IMAGES)) if os.path.isdir(IMAGES) else []:
        if not os.path.isdir(path(iid)):
            continue
        m = read_meta(iid)
        if not m:
            continue
        n_crack, n_not = correction_counts(iid, meta=m)
        has_prob = os.path.exists(path(iid, "prob.npy"))
        # A model switch flips the registry first and predicts afterwards, so if that
        # job dies or the app is quit mid-pass, some images still hold the previous
        # model's prediction. Reporting them as plain "ready" is what made that
        # invisible: the picker said model B while the mask was model A's. This says
        # so per image, and is derived from what is on disk rather than from whether
        # a job reported success.
        img_key = m.get("model_key")
        m.update(has_prob=has_prob,
                 has_emb=os.path.exists(path(iid, "emb.npz")),
                 model_key=img_key, current_model_key=cur_key,
                 stale=bool(has_prob and img_key and img_key != cur_key),
                 corrected_crack_px=n_crack, corrected_not_px=n_not)
        out.append(m)
    return out


def reconcile_statuses():
    """Repair statuses left mid-flight by a killed job. Returns what it changed.

    meta["status"] doubles as the ingest progress line -- "SAM embedding",
    "predicting", "hybrid model" -- and is only set to "ready" at the very end. So a job
    that dies partway (server restarted, machine slept, model switch aborted) leaves the
    last stage string sitting there as if it were a state. The frontend treats anything
    other than "ready" as still-processing and hides the canvas, which means a
    fully-predicted image becomes unlabellable because of a stale word.

    Truth is on disk, not in the status field: if prob.npy exists the image is usable,
    whatever the meta says. If it does not, the ingest really did not finish, and saying
    "interrupted" is more use than a frozen progress line the user cannot distinguish
    from work still happening.
    """
    fixed = []
    for iid in sorted(os.listdir(IMAGES)) if os.path.isdir(IMAGES) else []:
        if not os.path.isdir(path(iid)):
            continue
        m = read_meta(iid)
        st = m.get("status")
        if not m or st in ("ready", "uploaded", None) or str(st).startswith("interrupted"):
            continue
        if os.path.exists(path(iid, "prob.npy")):
            write_meta(iid, dict(status="ready"))
            fixed.append((iid, st, "ready"))
        else:
            write_meta(iid, dict(status=f"interrupted at '{st}' -- re-drop the file to finish"))
            fixed.append((iid, st, "interrupted"))
    return fixed


def delete_image(image_id):
    d = path(image_id)                      # raises on a crafted id before we get here
    # Independent of the id check, and deliberately so: this is the only rmtree in the app,
    # and it should be impossible for it to run anywhere but inside images/ no matter what
    # future change reaches it.
    root = os.path.realpath(IMAGES) + os.sep
    if not os.path.realpath(d).startswith(root):
        raise ValueError(f"refusing to delete outside {IMAGES}: {d!r}")
    if os.path.isdir(d):
        shutil.rmtree(d)
        return True
    return False


# ------------------------------------------------------------- model registry
def _default_registry():
    # `recipe` must match pipeline.RECIPE. It is spelled out rather than imported because
    # pipeline imports this module, and the selftest asserts the two agree -- without the tag
    # a retrain would treat this entry as a foreign recipe and fall back to an absolute floor
    # instead of a proper no-regression comparison.
    return dict(current=dict(kind="ensemble",
                             path_17=os.path.join(PROJECT, "models", "f17_v5_20260824.joblib"),
                             path_hybrid=os.path.join(PROJECT, "models",
                                                     "hybrid_v5_20260824.joblib"),
                             recipe="thincore_v5",
                             label="shipped baseline",
                             created=None),
                history=[])


def registry():
    if not os.path.exists(REGISTRY):
        r = _default_registry()
        write_json(REGISTRY, r)
        return r
    try:
        with open(REGISTRY) as f:
            return json.load(f)
    except Exception:
        # A registry we cannot parse must not take the app down: fall back to the
        # shipped baseline, which is a real working model, rather than raising on
        # every request. Retrain history is lost, the ability to predict is not.
        return _default_registry()


def model_key(entry):
    """Short stable id for one model, used to name its cached predictions.

    A retrain stamp is unique per model, so it is the natural key. Entries without one
    (the shipped baseline, or a hand-configured entry) are keyed by their files'
    paths AND size+mtime -- not paths alone. Paths alone were wrong: the training
    scripts overwrite a shipped model file in place, so a genuinely new
    model kept the old key, every image looked "ready" for it, and the app served the
    previous model's predictions while reporting the new one as current.
    """
    entry = entry or {}
    base = entry.get("created") or ""
    if not base:
        ident = []
        for p in (entry.get("path_17"), entry.get("path_hybrid")):
            try:
                st = os.stat(p)
                ident.append([p, st.st_size, int(st.st_mtime)])
            except (OSError, TypeError):
                ident.append([p, None, None])
        ident.append(entry.get("kind"))
        base = "m" + hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:10]
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(base))[:40]


def set_current(entry, remember=True):
    """Make `entry` the current model.

    remember=False is for switching between models the user already has: the old
    current still goes into history, but the entry being selected is pulled OUT of
    history so it is not listed twice, and re-selecting the same model is a no-op.
    Without this, flipping between two models a few times grew the history list with
    duplicate copies of both.
    """
    r = registry()
    cur = r.get("current")
    if cur and model_key(cur) == model_key(entry):
        return r
    hist = r.setdefault("history", [])
    if cur and not any(model_key(h) == model_key(cur) for h in hist):
        hist.append(cur)
    if not remember:
        r["history"] = [h for h in hist if model_key(h) != model_key(entry)]
    r["current"] = entry
    write_json(REGISTRY, r)
    return r


def remember_model(entry):
    """Add `entry` to the pickable list WITHOUT making it current.

    A retrain that fails the gate used to leave its model on disk and unreachable: the files
    were written, the scorecard said why it was refused, and there was no way to look at what
    it actually predicted. The refusal to auto-deploy is the point of the gate and stays --
    but "not deployed" should mean "you have to choose it deliberately", not "you cannot see
    it". So the candidate is registered here, appears in the picker with its verdict, and the
    user decides.

    Deduped by model_key, and never touches `current`.
    """
    r = registry()
    k = model_key(entry)
    if r.get("current") and model_key(r["current"]) == k:
        return r
    hist = r.setdefault("history", [])
    for i, h in enumerate(hist):
        if model_key(h) == k:
            hist[i] = entry                      # refresh the verdict on a re-run
            write_json(REGISTRY, r)
            return r
    hist.append(entry)
    write_json(REGISTRY, r)
    return r


def available_models():
    """Every model the user can pick: the current one, everything in history, and
    the shipped baseline even if it has never been current.

    Ordered current-first, then newest retrain to oldest, with the unstamped shipped
    baseline last. History order depends on how the user happened to switch around,
    which is not a sensible order to present.
    """
    r = registry()
    others = sorted((r.get("history") or []),
                    key=lambda e: (e.get("created") or ""), reverse=True)
    out, seen = [], set()
    for e in [r.get("current")] + others + [_default_registry()["current"]]:
        if not e:
            continue
        k = model_key(e)
        if k in seen:
            continue
        # Do not offer a model whose files are not all present. CrackModel silently
        # degrades when a path is missing -- drop path_hybrid and it predicts with the
        # 17-feature model alone -- so offering a half-present entry would mean the
        # user picks "retrained X" and gets something else, cached under X's key.
        # Requiring EVERY declared path to exist, not just one of them, is the point.
        declared = [p for p in (e.get("path_17"), e.get("path_hybrid")) if p]
        if not declared or not all(os.path.exists(p) for p in declared):
            continue
        seen.add(k)
        out.append(dict(e, id=k, current=(k == model_key(r.get("current")))))
    return out


# ------------------------------------------------- per-model prediction cache
# Switching models has to feel instant, and a prediction is a pure function of
# (image, model), so it is cacheable forever. prob.npy stays "the current model's
# prediction" -- every other module already reads it -- and is HARD LINKED to the
# cache entry rather than copied, so keeping N models per image costs N predictions
# on disk, not 2N. save_npy writes via temp+rename, which replaces the link target
# instead of writing through it, so a later write can never corrupt a cache entry.
PROB_CACHE_KEEP = 6


def sweep_stale_temps(max_age_s=3600, dry_run=False):
    """Delete staging files that no process is going to finish writing.

    Every atomic write here stages to "<final>.<pid>.<thread>.tmp" and unlinks it on
    exception -- but not on SIGKILL, a power cut, or a `kill` from a shell, because no
    handler runs. Those files are invisible in the UI and never read, so they accumulate
    silently: this repo reached 271 of them totalling 9.1 GB after a day of interrupted
    retrains and killed servers, and a prob.npy for a 32 MP frame is 128 MB on its own.

    Age, not pid, decides. Checking whether the pid is alive looks more precise and is
    wrong: pids are recycled, and a temp written by a process that has since exited can
    share a number with something running now. Anything older than an hour cannot belong to
    a write still in flight -- the slowest single write in this app is a 128 MB array.

    Returns (files_removed, bytes_reclaimed).
    """
    import time as _t
    cutoff = _t.time() - max_age_s
    n = freed = 0
    for root, _dirs, files in os.walk(IMAGES):
        for fn in files:
            if ".tmp" not in fn:
                continue
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
                if st.st_mtime >= cutoff:
                    continue
                if not dry_run:
                    os.unlink(p)
                n += 1
                freed += st.st_size
            except OSError:
                continue
    return n, freed


def _prob_cache_dir(image_id):
    d = path(image_id, "probs")
    os.makedirs(d, exist_ok=True)
    return d


def prob_cache_path(image_id, key):
    return os.path.join(_prob_cache_dir(image_id), f"{key}.npy")


def has_prob_for(image_id, key):
    return os.path.exists(prob_cache_path(image_id, key))


def _link_or_copy(src, dst):
    """Publish src at dst atomically, sharing the inode where the filesystem allows.

    The staging name is unique per process AND thread. A single fixed "<dst>.tmp" was a
    real hazard: two threads publishing prob.npy at once (a predict job finishing while
    the user picks a model) could have one link its source and the other replace it, so
    prob.npy ended up holding the wrong model's array. Worse, the copy fallback would
    open that shared temp name "wb" while it was still a hard link to a cache entry --
    truncating that entry and poisoning it for every future switch.
    """
    tmp = f"{dst}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        os.link(src, tmp)                       # same inode: no extra disk
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        shutil.copyfile(src, tmp)               # different device, or links unsupported
    os.replace(tmp, dst)


def store_prob(image_id, key, arr):
    """Write a prediction into the cache and make it the live prob.npy."""
    p = prob_cache_path(image_id, key)
    tmp = p + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _link_or_copy(p, path(image_id, "prob.npy"))
    _prune_prob_cache(image_id, keep_key=key)
    write_meta(image_id, dict(model_key=key))


def adopt_prob(image_id, key):
    """Point prob.npy at an already-computed prediction. True if the cache had it."""
    p = prob_cache_path(image_id, key)
    if not os.path.exists(p):
        return False
    _link_or_copy(p, path(image_id, "prob.npy"))
    os.utime(p, None)                           # mark as recently used for pruning
    write_meta(image_id, dict(model_key=key))
    return True


def migrate_prob_cache(image_id):
    """Seed the cache from a pre-cache prob.npy, once, and only when we KNOW its author.

    Only migrates when meta.json records which model produced the file. It used to fall
    back to "assume the current model", which is a guess that silently becomes a lie:
    if the file was actually an older model's output, it got filed under the current
    model's key, and every later request adopted it -- so the app would report the new
    model as ready for that image while showing the old model's mask, and no retrain or
    re-apply would ever correct it. An unlabelled file is left alone instead; the cost
    is one honest re-prediction, which is strictly better than a wrong cache entry.
    """
    live = path(image_id, "prob.npy")
    if not os.path.exists(live) or os.listdir(_prob_cache_dir(image_id)):
        return False
    key = read_meta(image_id).get("model_key")
    if not key:
        return False
    _link_or_copy(live, prob_cache_path(image_id, key))
    return True


def _prune_prob_cache(image_id, keep_key=None):
    """Bound the cache: a 23 MP prediction is 94 MB, so this is not free."""
    d = _prob_cache_dir(image_id)
    files = [f for f in os.listdir(d) if f.endswith(".npy")]
    if len(files) <= PROB_CACHE_KEEP:
        return
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    for f in files[PROB_CACHE_KEEP:]:
        if keep_key and f == f"{keep_key}.npy":
            continue
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass


def rollback():
    """Restore the previous model. Returns True if there was one."""
    r = registry()
    hist = r.get("history") or []
    if not hist:
        return False
    prev = hist.pop()
    r.setdefault("history", []).append(r["current"])
    r["current"] = prev
    r["history"] = hist + [h for h in r["history"] if h is not prev][-20:]
    write_json(REGISTRY, r)
    return True
