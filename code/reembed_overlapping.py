"""Rebuild every SAM embedding cache at the overlapping tile stride.

Tiles used to abut, so the embedding stepped across every interior tile boundary and the
step showed up in the output as a seam -- on one 3044x2354 frame the worst row carried 32.8x
the median row-to-row probability change, all of it at y=1023. Blending at lookup time
cannot fix that: with no overlap there is no real data spanning the boundary, so reaching
across has to invent it (false positives on crack-free specimens rose 6.2x) and a window
that stops at the edge reduces to last-tile-wins exactly. Real overlap is the only fix, and
it has to be baked into the cache because that is where the tiling is decided.

The previous cache is RENAMED, not deleted, so this is reversible: if the retrain that
follows fails its gate, restore with --rollback and nothing was lost. Resumable -- rerun it
and it skips whatever is already current.
"""
import argparse
import os
import sys
import time

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, os.path.join(PROJECT, "code"))

import model as M          # noqa: E402
import store as S          # noqa: E402

OLD = "emb_pre_overlap.npz"


def rollback():
    n = 0
    for m in S.list_images():
        src, dst = S.path(m["id"], OLD), S.path(m["id"], "emb.npz")
        if os.path.exists(src):
            os.replace(src, dst)
            n += 1
    print(f"  restored {n} pre-overlap caches")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollback", action="store_true",
                    help="put the pre-overlap caches back and exit")
    ap.add_argument("--keep-old", action="store_true", default=True,
                    help="rename the old cache instead of overwriting it (default)")
    args = ap.parse_args()
    if args.rollback:
        return rollback()

    imgs = [m for m in S.list_images() if "SELFTEST" not in (m.get("filename") or "")]
    todo = [m for m in imgs if not M.emb_is_current(S.path(m["id"], "emb.npz"))]
    print(f"  {len(imgs)} images, {len(todo)} to rebuild at stride {M.TILE_STRIDE} "
          f"(tile {M.TILE}, overlap {M.TILE - M.TILE_STRIDE} px)\n", flush=True)

    t_all = time.time()
    tiles_done = 0
    for k, m in enumerate(todo, 1):
        iid = m["id"]
        name = (m.get("filename") or iid)[:52]
        img = S.load_npy(iid, "img.npy")
        if img is None:
            print(f"  [{k}/{len(todo)}] {name}: no img.npy, skipped", flush=True)
            continue
        img01 = np.asarray(img, np.float32)
        embp = S.path(iid, "emb.npz")
        n_before = 0
        if os.path.exists(embp):
            try:
                n_before = len(np.load(embp)["coords"])
            except Exception:                                   # noqa: BLE001
                n_before = 0
            if args.keep_old and not os.path.exists(S.path(iid, OLD)):
                os.replace(embp, S.path(iid, OLD))
        t0 = time.time()
        try:
            coords, embs = M.embed_image(img01)
        except M.SamUnavailable as e:
            print(f"  [{k}/{len(todo)}] {name}: SAM unavailable ({e}), stopping", flush=True)
            return 1
        M.write_emb(embp, coords, embs)
        dt = time.time() - t0
        tiles_done += len(coords)
        rate = (time.time() - t_all) / tiles_done
        left = sum(len(M.tiles((mm.get("height") or 0, mm.get("width") or 0),
                               stride=M.TILE_STRIDE)) for mm in todo[k:])
        print(f"  [{k}/{len(todo)}] {name}  {n_before} -> {len(coords)} tiles  "
              f"{dt:.0f}s  ({rate:.2f}s/tile, ~{left*rate/60:.0f} min left)", flush=True)
        del img, img01, coords, embs

    print(f"\n  rebuilt {len(todo)} caches, {tiles_done} tiles, "
          f"{(time.time()-t_all)/60:.1f} min")
    bad = [m for m in imgs if not M.emb_is_current(S.path(m["id"], "emb.npz"))]
    print(f"  still not current: {len(bad)}")
    print("REEMBED_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
