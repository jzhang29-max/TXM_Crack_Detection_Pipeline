"""Sample the owner's own corrections into a cross-validation fold file, all specimen groups.

    python3 code/build_label_folds.py                 # build (slow, one-off, cached)
    python3 code/build_label_folds.py --per-image 20000
    python3 code/build_label_folds.py --list          # what it would do, no work

WHY. Every accuracy number in this project rests on FOUR ground-truth images, all one
specimen group (B2), all wide-open cracks with ~65 px median width. That is the single
biggest limitation of the whole project: nothing measured on those four can say anything
about the AM/HC, B3 or wrought specimens, and the model demonstrably misses thin cracks
that the ground-truth set does not contain.

The owner has since hand-labelled 30.2 M crack pixels across 56 images spanning all four
groups. Those labels are the only data that can extend evaluation beyond B2.

SPARSE, NOT DENSE -- this is the distinction that decides everything. Ground truth labels
EVERY pixel, which is what makes IoU computable: you need to know the false negatives.
Corrections are sparse; `corr == 0` means "the owner expressed no opinion here", not "not
crack". Measured on six images: treating corrections as dense ground truth gives a mean
IoU of 0.06, because every crack the model found correctly and the owner never painted
over is counted as a false positive. Restricted to pixels the owner actually judged, the
same images give 0.70-0.997.

So this file samples ONLY labelled pixels, and the evaluation built on it reports agreement
on those pixels -- precision and recall against the owner's judgement -- never a
whole-image IoU.

WHAT IT DOES NOT FIX. 98.3% of the force-crack labels sit on pixels the model already
called crack, because Flip region confirms a blob in one click. Agreement on those is
partly circular: it measures whether a new model still agrees with what an older one found
and the owner accepted. The informative subsets are the DISAGREEMENTS -- crack the owner
added where the model had none, and model crack the owner struck out -- and the evaluation
reports those separately for exactly that reason.

READ-ONLY on the labels. Reads correction.npy, img.npy and emb.npz; writes one .npz into
paint/.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import store as S            # noqa: E402
import model as M            # noqa: E402
from txm_features import compute_feature_stack   # noqa: E402

OUT = os.path.join(PROJECT, "paint", "label_folds.npz")
# A 32 MP frame at 17 float32 features is 2.2 GB, so big images are processed in row bands.
# MARGIN is not a guess: compute_feature_stack's largest-radius features pull context from
# far outside a pixel, and comparing a cropped stack against a full-image one showed the
# disagreement reach exactly zero at a 256 px inset (4.4e-2 at the edge, 1.1e-3 at 128 px).
# Only pixels at least MARGIN inside a band carry production-identical features.
MARGIN = 256
BAND_BUDGET_BYTES = 700e6


def group_of(name):
    n = (name or "").lower()
    if "wrought" in n:
        return "wrought"
    if "hc_316l" in n:
        return "AM/HC"
    if "_b3_" in n or "b3_" in n:
        return "B3"
    return "B2"


def sample_image(iid, per_image, rng):
    """(x17, xsam, y) for up to `per_image` labelled pixels of one image, or None."""
    corr = S.load_npy(iid, "correction.npy", mmap=True)
    img = S.load_npy(iid, "img.npy", mmap=True)
    if corr is None or img is None:
        return None
    corr = np.asarray(corr)
    img = np.asarray(img)
    if corr.shape != img.shape:
        return None
    Hh, Ww = img.shape
    rows_per_band = max(2 * MARGIN + 64,
                        int(BAND_BUDGET_BYTES / max(Ww * 17 * 4, 1)))

    # Balance crack against not-crack so one class cannot swamp the sample: an image can
    # hold 18 M not-crack labels and 30 k crack ones.
    want_each = max(1, per_image // 2)
    picks = []
    for r0 in range(0, Hh, rows_per_band):
        r1 = min(r0 + rows_per_band, Hh)
        lo = r0 + (MARGIN if r0 > 0 else 0)
        hi = r1 - (MARGIN if r1 < Hh else 0)
        if hi - lo < 8:
            continue
        band = corr[lo:hi]
        for cls in (1, 2):
            idx = np.flatnonzero(band.ravel() == cls)
            if not len(idx):
                continue
            take = min(len(idx), max(1, want_each // max(1, Hh // rows_per_band + 1)))
            sel = rng.choice(idx, take, replace=False)
            rr, cc = np.unravel_index(sel, band.shape)
            picks.append((r0, r1, lo, rr + lo, cc, np.full(take, cls == 1)))
    if not picks:
        return None

    x17, xsam, y = [], [], []
    z = np.load(S.path(iid, "emb.npz")) if os.path.exists(S.path(iid, "emb.npz")) else None
    coords, embs = (z["coords"], z["emb"]) if z is not None else (None, None)
    by_band = {}
    for r0, r1, lo, rr, cc, yy in picks:
        by_band.setdefault((r0, r1), []).append((rr, cc, yy))
    for (r0, r1), items in by_band.items():
        feats = np.asarray(compute_feature_stack(np.asarray(img[r0:r1])), np.float32)
        for rr, cc, yy in items:
            x17.append(feats[rr - r0, cc, :])
            y.append(yy)
            if coords is not None:
                out = np.zeros((len(rr), embs.shape[1]), np.float32)
                todo = np.ones(len(rr), bool)
                for t in range(len(coords) - 1, -1, -1):
                    y0, x0 = int(coords[t][0]), int(coords[t][1])
                    s = (todo & (rr >= y0) & (rr < y0 + M.TILE)
                         & (cc >= x0) & (cc < x0 + M.TILE))
                    if s.any():
                        out[s] = M.interp_tile(embs[t], rr[s] - y0, cc[s] - x0)
                        todo &= ~s
                xsam.append(out)
        del feats
    if not x17:
        return None
    return (np.concatenate(x17), np.concatenate(xsam) if xsam else None,
            np.concatenate(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", type=int, default=20000)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    todo = []
    for m in S.list_images():
        c, n = S.correction_counts(m["id"])
        if c + n == 0:
            continue
        todo.append((m, c, n))
    groups = {}
    for m, c, n in todo:
        groups.setdefault(group_of(m.get("filename")), []).append(m.get("filename"))
    print(f"{len(todo)} labelled image(s) across {len(groups)} specimen group(s):")
    for g in sorted(groups):
        print(f"    {g:<9} {len(groups[g]):>3} images")
    print(f"\nsampling up to {a.per_image:,} labelled px per image "
          f"(balanced crack / not-crack)")
    if a.list:
        return 0

    rng = np.random.RandomState(3)
    blocks, t0 = {}, time.time()
    for i, (m, c, n) in enumerate(todo, 1):
        got = sample_image(m["id"], a.per_image, rng)
        if got is None:
            print(f"  [{i}/{len(todo)}] {(m.get('filename') or '')[22:54]:<34} skipped")
            continue
        x17, xsam, y = got
        key = m["id"]
        blocks[f"{key}|x17"] = x17
        if xsam is not None:
            blocks[f"{key}|xsam"] = xsam
        blocks[f"{key}|y"] = y
        blocks[f"{key}|group"] = np.array([group_of(m.get("filename"))])
        print(f"  [{i}/{len(todo)}] {(m.get('filename') or '')[22:54]:<34} "
              f"{len(y):>6,} px  {100*y.mean():4.1f}% crack  {time.time()-t0:.0f}s",
              flush=True)
        del got, x17, xsam, y

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **blocks)
    os.replace(tmp, OUT)
    n_img = len({k.split("|")[0] for k in blocks})
    print(f"\nwrote {os.path.relpath(OUT, PROJECT)}  {os.path.getsize(OUT)/1e6:.0f} MB, "
          f"{n_img} images, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
