"""
Version your correction labels in git, without committing app_data.

    python3 code/backup_labels.py             # save app_data labels -> paint/app_labels.npz
    python3 code/backup_labels.py --restore   # write them back into app_data
    python3 code/backup_labels.py --status    # what is saved vs what is live

WHY. app_data/ is gitignored, and it should be: it holds 19 GB of predictions, SAM
embeddings and thumbnails, all of it regenerable from the images plus the labels. The
labels are the exception -- they are the only thing in there a person made by hand, they
cannot be regenerated, and losing the disk loses them.

They are also nearly free to version. A correction mask is uint8 at full image
resolution and overwhelmingly zero, so 850 MB of masks compresses to a few MB. That is
small enough to commit after a labelling session and keep forever.

KEYED BY FILENAME, not by image id. An id is a hash of file CONTENT, so it changes if an
image is ever recompressed or re-exported, and a backup keyed by id would silently fail
to restore onto the same picture. Filename survives that, and it is what
import_research_corrections.py and the SAM cache already key on.

RESTORE IS CONSERVATIVE. It refuses to overwrite labels already present in the app unless
--force, because the app's copy is normally the newer one -- restoring should recover lost
work, not quietly replace current work with an older snapshot.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import store as S            # noqa: E402

ARCHIVE = os.path.join(PROJECT, "paint", "app_labels.npz")
MANIFEST = os.path.join(PROJECT, "paint", "app_labels.json")


def _labelled():
    """(filename, id, array, crack, not_crack) for every image carrying labels."""
    out = []
    for m in S.list_images():
        c, n = S.correction_counts(m["id"])
        if c + n == 0:
            continue
        arr = S.load_npy(m["id"], "correction.npy")
        if arr is None:
            continue
        out.append((m.get("filename") or m["id"], m["id"], np.asarray(arr), c, n))
    return out


def save():
    items = _labelled()
    if not items:
        print("no labels in app_data yet -- nothing to save")
        return 0
    raw = sum(a.nbytes for _, _, a, _, _ in items)
    os.makedirs(os.path.dirname(ARCHIVE), exist_ok=True)
    # Write through a handle, not a path: np.savez_compressed APPENDS ".npz" to a path
    # that does not already end in it, so "app_labels.npz.tmp" became
    # "app_labels.npz.tmp.npz" and the rename below failed on a file that was never
    # created. Passing an open file writes exactly where told.
    tmp = ARCHIVE + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **{fn: a for fn, _, a, _, _ in items})
    os.replace(tmp, ARCHIVE)

    manifest = dict(
        saved=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_images=len(items),
        total_crack_px=int(sum(c for *_, c, _ in items)),
        total_not_crack_px=int(sum(n for *_, n in items)),
        images={fn: dict(crack=int(c), not_crack=int(n), shape=list(a.shape))
                for fn, _, a, c, n in items},
    )
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    size = os.path.getsize(ARCHIVE)
    print(f"saved {len(items)} image(s) of labels")
    print(f"  {manifest['total_crack_px']:,} force-crack + "
          f"{manifest['total_not_crack_px']:,} force-not-crack pixels")
    print(f"  {raw/1e6:.0f} MB of masks -> {size/1e6:.2f} MB compressed "
          f"({raw/max(size,1):.0f}x)")
    print(f"  {os.path.relpath(ARCHIVE, PROJECT)}  +  {os.path.relpath(MANIFEST, PROJECT)}")
    print("\ncommit them to version this session's work:")
    print("  git add paint/app_labels.npz paint/app_labels.json && git commit -m 'labels' && git push")
    return 0


def status():
    live = {fn: (c, n) for fn, _, _, c, n in _labelled()}
    saved = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            man = json.load(f)
        saved = {k: (v["crack"], v["not_crack"]) for k, v in man["images"].items()}
        print(f"archive saved {man['saved']}: {man['n_images']} image(s), "
              f"{man['total_crack_px']:,} crack + {man['total_not_crack_px']:,} not-crack")
    else:
        print("no archive yet")
    print(f"live in app_data: {len(live)} image(s), "
          f"{sum(c for c, _ in live.values()):,} crack + "
          f"{sum(n for _, n in live.values()):,} not-crack")
    only_live = [k for k in live if k not in saved]
    changed = [k for k in live if k in saved and live[k] != saved[k]]
    only_saved = [k for k in saved if k not in live]
    for label, xs in (("not yet saved", only_live), ("changed since saving", changed),
                      ("in the archive but not loaded", only_saved)):
        if xs:
            print(f"  {label}: {len(xs)}")
            for x in xs[:4]:
                print(f"    {x[:66]}")
    if not (only_live or changed):
        print("  archive is up to date with app_data")
    return 0


def restore(force=False):
    if not os.path.exists(ARCHIVE):
        print(f"no archive at {ARCHIVE}")
        return 1
    z = np.load(ARCHIVE)
    by_name = {m.get("filename"): m for m in S.list_images()}
    wrote = skipped = absent = 0
    for fn in z.files:
        m = by_name.get(fn)
        if m is None:
            absent += 1
            continue
        arr = z[fn]
        cur_c, cur_n = S.correction_counts(m["id"])
        if (cur_c + cur_n) and not force:
            print(f"  keeping app's own labels for {fn[:52]} "
                  f"({cur_c + cur_n:,} px) -- --force to overwrite")
            skipped += 1
            continue
        live = S.load_npy(m["id"], "correction.npy", mmap=True)
        if live is not None and np.asarray(live).shape != arr.shape:
            print(f"  SHAPE MISMATCH for {fn[:52]} -- skipped")
            skipped += 1
            continue
        with S.image_lock(m["id"]):
            S.save_npy(m["id"], "correction.npy", arr)
        # drop rendered caches so the restored labels show immediately
        ov = S.path(m["id"], "overlays")
        if os.path.isdir(ov):
            for g in os.listdir(ov):
                try:
                    os.remove(os.path.join(ov, g))
                except OSError:
                    pass
        for g in os.listdir(S.path(m["id"])):
            if g.startswith("thumb_"):
                try:
                    os.remove(S.path(m["id"], g))
                except OSError:
                    pass
        wrote += 1
    print(f"restored {wrote}, kept {skipped}, {absent} archived image(s) not loaded here")
    return 0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--restore", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="with --restore, overwrite labels the app already has")
    a = ap.parse_args()
    if a.status:
        return status()
    if a.restore:
        return restore(force=a.force)
    return save()


if __name__ == "__main__":
    sys.exit(main())
