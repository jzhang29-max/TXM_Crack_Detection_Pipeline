"""
Load every TXM image into the app and predict it, reusing the research SAM cache.

    python3 code/load_all_images.py                  # the 71 images shipped in images/
    python3 code/load_all_images.py --src /some/other/dir
    python3 code/load_all_images.py --src <dir> --dry-run

Why this exists rather than dragging 71 files onto the window: the SAM embedding is
the expensive part of ingest (~20 s for a 2.9 MP image, minutes for the 23 MP mosaic),
and all 71 were already embedded during the research phase into paint/sam_embcache
under their original filenames. Seeding each image's emb.npz from there before ingest
runs turns "re-embed everything" into "run the classifier", which is the difference
between an afternoon and a few minutes.

Uses the same store/pipeline entry points the web upload uses, so an image loaded this
way is indistinguishable from a dragged one: same content-hash id, same preprocessing,
same per-model prediction cache. Re-running is safe -- an image whose prediction already
exists for the current model is skipped.
"""

import argparse
import glob
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import store as S            # noqa: E402
import pipeline as P         # noqa: E402

EMB_CACHE = os.path.join(PROJECT, "paint", "sam_embcache")


def find_cached_embedding(filename):
    """The research cache is keyed by the original filename stem."""
    if not os.path.isdir(EMB_CACHE):
        return None
    stem = os.path.splitext(os.path.basename(filename))[0]
    exact = os.path.join(EMB_CACHE, f"{stem}_samemb.npz")
    if os.path.exists(exact):
        return exact
    for f in sorted(os.listdir(EMB_CACHE)):
        if f.endswith("_samemb.npz") and stem in f:
            return os.path.join(EMB_CACHE, f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(PROJECT, "images"),
                    help="directory to search recursively (default: the repo's images/)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (for testing)")
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    files = sorted(f for ext in ("tif", "tiff", "png", "jpg", "jpeg")
                   for f in glob.glob(os.path.join(src, "**", f"*.{ext}"), recursive=True))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"no images found under {src}")
        return 1

    # Guard against loading the same PICTURE twice under a different id. An image id is
    # a hash of file CONTENT, and images/ holds recompressed copies -- same pixels, other
    # bytes -- so a library already loaded from the uncompressed originals would gain 71
    # duplicates, each with its own predictions and none of the other's labels. Filename
    # is the stable identity across recompression, and it is what the correction masks
    # and the SAM cache key on too.
    by_name = {m.get("filename"): m for m in S.list_images()}

    cur_key = S.model_key(S.registry().get("current"))
    print(f"found {len(files)} image(s) under {src}")
    print(f"current model: {S.registry()['current'].get('label')}  (key {cur_key})")
    n_emb = sum(1 for f in files if find_cached_embedding(f))
    print(f"SAM embeddings available to reuse: {n_emb}/{len(files)}"
          + ("" if n_emb == len(files) else "  -- the rest will be embedded, which is slow"))
    if args.dry_run:
        # Apply the SAME skip rule the real run applies. It used to print "would load"
        # for every file regardless, so a dry run against a full library announced 71
        # loads and the real run then skipped all 71 -- a dry run that does not predict
        # the real run is worse than no dry run at all.
        todo = [f for f in files
                if (by_name.get(os.path.basename(f)) or {}).get("status") != "ready"]
        skip = len(files) - len(todo)
        print(f"  would load {len(todo)}, skip {skip} already in the app")
        for f in todo[:6]:
            print(f"    load {os.path.basename(f)[:58]}"
                  f"  emb={'cached' if find_cached_embedding(f) else 'MISSING'}")
        if len(todo) > 6:
            print(f"    ... and {len(todo) - 6} more")
        return 0

    t0 = time.time()
    loaded = skipped = failed = 0
    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        prior = by_name.get(name)
        if prior is not None and prior.get("status") == "ready":
            print(f"[{i}/{len(files)}] {name[:58]}  already in the app (same filename)")
            skipped += 1
            continue
        with open(f, "rb") as fh:
            content = fh.read()
        iid, is_new = S.save_upload(name, content)
        del content

        # Already predicted with this model? Nothing to do.
        S.migrate_prob_cache(iid)
        if S.has_prob_for(iid, cur_key) and S.read_meta(iid).get("status") == "ready":
            print(f"[{i}/{len(files)}] {name[:58]}  already done")
            skipped += 1
            continue

        # Seed the SAM embedding before ingest looks for it.
        emb_dst = S.path(iid, "emb.npz")
        if not os.path.exists(emb_dst):
            cached = find_cached_embedding(f)
            if cached:
                shutil.copyfile(cached, emb_dst)

        note = "reused emb" if os.path.exists(emb_dst) else "EMBEDDING (slow)"
        print(f"[{i}/{len(files)}] {name[:58]}  {note} ... ", end="", flush=True)
        t1 = time.time()
        try:
            P.ingest(iid)
        except Exception as e:                              # noqa: BLE001
            print(f"FAILED: {type(e).__name__}: {e}")
            S.write_meta(iid, dict(status=f"failed: {type(e).__name__}"))
            failed += 1
            continue
        m = S.read_meta(iid)
        print(f"{time.time()-t1:.1f}s  {(m.get('predicted_area') or 0)*100:.1f}% crack")
        loaded += 1

    print(f"\ndone in {time.time()-t0:.0f}s: {loaded} loaded, {skipped} already present, "
          f"{failed} failed")
    imgs = S.list_images()
    ready = sum(1 for m in imgs if m.get("status") == "ready")
    stale = sum(1 for m in imgs if m.get("stale"))
    print(f"app now holds {len(imgs)} image(s): {ready} ready, {stale} on an older model")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
