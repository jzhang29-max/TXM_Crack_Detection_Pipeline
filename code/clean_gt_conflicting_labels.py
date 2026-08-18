"""Clear correction labels that the shipped ground truth contradicts.

    python3 code/clean_gt_conflicting_labels.py --dry-run     # measure, change nothing
    python3 code/clean_gt_conflicting_labels.py               # clear not-crack on real crack
    python3 code/clean_gt_conflicting_labels.py --also-crack  # and crack on real background

WHY. import_research_corrections.py brought in 263 M not-crack pixels from the research
archive. On the four images where pixel-level ground truth exists, 22-28% of that
archive's not-crack labels sit on real crack -- the outlines it was derived from were
drawn tight to the crack, so the crack's margins came through as background. The app's
current numbers are lower, 3.4-8.0%, because force-crack strokes drawn later overwrote
much of it, but ~363 k pixels of real crack are still labelled not-crack and they went
into a retrain.

A label that says "background" on a pixel that is crack does not merely fail to help. It
teaches the model that crack margins are background, and margins are exactly where a
crack segmentation is decided.

CLEARS TO 0 RATHER THAN FLIPPING TO CRACK. Deliberately. gather_training_data already
samples 100 k crack and 100 k background pixels per ground-truth image straight from the
GT masks, so the correct signal for these images arrives by that path no matter what the
correction mask says. Zeroing removes the contradiction without writing a second source's
opinion into a label set that is otherwise the owner's own strokes.

ONLY THE FOUR GROUND-TRUTH IMAGES. The other 67 carry negatives from the same archive and
cannot be checked -- there is no truth outside B2. Those are left alone on purpose; see
the README's caveats. This tool cannot fix what it cannot measure, and guessing would be
the same mistake the import made.

Reversible: labels are versioned by code/backup_labels.py. Take a snapshot first.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import store as S            # noqa: E402

GT_CACHE = os.path.join(PROJECT, "dataset_cache")
# (ground-truth stem, fragment of the app filename it belongs to)
PAIRS = [("LARGE_343_75", "b2_343_75_LARGE"),
         ("336_25", "b2_336_25"),
         ("338_13", "b2_338_13"),
         ("333_75_um_zoom", "B2_333_75_um_zoom")]


def find_image(fragment):
    for m in S.list_images():
        if fragment.lower() in (m.get("filename") or "").lower():
            return m
    return None


def drop_render_caches(iid):
    """Overlays and thumbnails are keyed on correction.npy's mtime, but delete them
    anyway: an image whose labels just changed should not show a stale burn-in."""
    ov = S.path(iid, "overlays")
    if os.path.isdir(ov):
        for g in os.listdir(ov):
            try:
                os.remove(os.path.join(ov, g))
            except OSError:
                pass
    for g in os.listdir(S.path(iid)):
        if g.startswith("thumb_"):
            try:
                os.remove(S.path(iid, g))
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--also-crack", action="store_true",
                    help="also clear force-CRACK labels that ground truth calls background")
    a = ap.parse_args()

    print(f"{'image':<18} {'not-crack on crack':>19} {'crack on background':>20}")
    print("-" * 60)
    plan, tot_n, tot_c = [], 0, 0
    for stem, frag in PAIRS:
        gt_p = os.path.join(GT_CACHE, f"{stem}_gt.npy")
        m = find_image(frag)
        if m is None:
            print(f"{stem:<18} not loaded in the app -- skipped")
            continue
        if not os.path.exists(gt_p):
            print(f"{stem:<18} {gt_p} missing -- run code/unpack_package.py")
            continue
        corr = S.load_npy(m["id"], "correction.npy")
        if corr is None:
            print(f"{stem:<18} no labels")
            continue
        gt = np.asarray(np.load(gt_p, mmap_mode="r")).astype(bool)
        corr = np.asarray(corr)
        if corr.shape != gt.shape:
            print(f"{stem:<18} SHAPE MISMATCH {corr.shape} vs {gt.shape} -- skipped")
            continue
        bad_n = (corr == 2) & gt              # said background, is crack
        bad_c = (corr == 1) & ~gt             # said crack, is background
        n_n, n_c = int(bad_n.sum()), int(bad_c.sum())
        tot_n += n_n
        tot_c += n_c
        print(f"{stem:<18} {n_n:>19,} {n_c:>20,}")
        if n_n or (a.also_crack and n_c):
            plan.append((m["id"], stem, corr, bad_n, bad_c))

    print("-" * 60)
    print(f"{'TOTAL':<18} {tot_n:>19,} {tot_c:>20,}")
    if not a.also_crack and tot_c:
        print(f"\n  {tot_c:,} force-crack px sit where ground truth says background. Not "
              f"touched\n  without --also-crack: at a crack margin it is genuinely "
              f"arguable which is\n  right, and these are the owner's own strokes.")

    if a.dry_run:
        print("\n(dry run -- nothing written)")
        return 0
    if not plan:
        print("\nnothing to clear")
        return 0

    print()
    for iid, stem, corr, bad_n, bad_c in plan:
        out = corr.copy()
        cleared = int(bad_n.sum())
        out[bad_n] = 0
        if a.also_crack:
            cleared += int(bad_c.sum())
            out[bad_c] = 0
        before_c, before_n = int((corr == 1).sum()), int((corr == 2).sum())
        after_c, after_n = int((out == 1).sum()), int((out == 2).sum())
        with S.image_lock(iid):
            S.save_npy(iid, "correction.npy", out)
        drop_render_caches(iid)
        print(f"  {stem:<18} cleared {cleared:>9,} px   "
              f"crack {before_c:,} -> {after_c:,}, not-crack {before_n:,} -> {after_n:,}")

    print("\nsnapshot the change:  python3 code/backup_labels.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
