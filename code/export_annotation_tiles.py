"""Export small tiles for dense annotation, so a usable AM/HC ground truth is hours not weeks.

    python3 code/export_annotation_tiles.py --list
    python3 code/export_annotation_tiles.py HC_316L_fatigue_1250_cycles --tiles 12
    python3 code/export_annotation_tiles.py --group AM/HC --tiles 30

WHY TILES AND NOT FRAMES. The project's binding limitation is measured: held out, the model
recovers 39.7% of the crack the owner marked on AM/HC against 75.5-81.6% for the same number
of random images. Fixing it needs DENSE annotation there -- every pixel judged, because IoU
needs false negatives. A 10 MP frame is weeks of that.

It does not need whole frames. A held-out IoU is a statistic, and a uniformly-random sample
of tiles estimates it without bias, with a confidence interval you can quote. Thirty
512x512 tiles is 7.9 M pixels -- about one 8 MP frame's worth of decisions, spread across
many frames and specimens, which is worth far more than one frame densely done.

TILES ARE SAMPLED UNIFORMLY AT RANDOM inside the specimen, and that is deliberate. Choosing
tiles where the model is uncertain would find more crack per tile and make the annotation
feel productive -- and it would also make the resulting ground truth a biased sample, so any
IoU computed on it would not estimate the frame's IoU. If you want active learning, do it
for TRAINING tiles and keep an untouched uniform set for evaluation. This script writes the
uniform set.

WHAT WAS TRIED AND REJECTED FIRST, both measured, so nobody rebuilds them:
  * Candidate regions from model-independent ridge filters (Sato/Frangi/Hessian): captured
    only 63% of the owner's existing crack pixels even when covering 15% of the frame, so
    37% would still need painting by hand. No saving.
  * A superpixel tessellation to turn painting into accept/reject: the CEILING -- perfect
    labelling of every superpixel -- was IoU 0.700 at 8216 SLIC segments (0.543 at 2771,
    0.184 at 859) against the owner's own pixels. These cracks are too fine for superpixel
    boundaries to follow, and a ceiling below the model's own held-out score is useless.
  * Machine-generated labels of any kind. docs/HANDOFF.md records two attempts that produced
    "confidently-wrong labels" and were reverted. Labels made by a model cannot validate that
    model; the number would measure agreement with itself.

Each tile is written as a lossless 16-bit PNG at native resolution plus a manifest recording
exactly where it came from, so a painted mask can be placed back into frame coordinates.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import store as S            # noqa: E402
import pipeline as P         # noqa: E402

OUT = os.path.join(PROJECT, "paint", "annotate")
TILE = 512
MIN_ON_SPECIMEN = 0.90       # a tile mostly off-specimen teaches nothing


def group_of(name):
    n = (name or "").lower()
    if "wrought" in n:
        return "wrought"
    if "hc_316l" in n:
        return "AM/HC"
    if "_b3_" in n or "b3_" in n:
        return "B3"
    return "B2"


def candidates(m, rng, want, tile=TILE):
    img = np.asarray(S.load_npy(m["id"], "img.npy"))
    spec = P.specimen_support(img)
    Hh, Ww = img.shape
    if Hh < tile or Ww < tile:
        return img, []
    out, tries = [], 0
    while len(out) < want and tries < want * 200:
        tries += 1
        y = rng.randint(0, Hh - tile); x = rng.randint(0, Ww - tile)
        if spec[y:y + tile, x:x + tile].mean() < MIN_ON_SPECIMEN:
            continue
        if any(abs(y - yy) < tile // 2 and abs(x - xx) < tile // 2 for yy, xx in out):
            continue                            # no heavy overlap between tiles
        out.append((y, x))
    return img, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fragment", nargs="?")
    ap.add_argument("--group", help="sample across a whole specimen group, e.g. AM/HC")
    ap.add_argument("--tiles", type=int, default=12)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    imgs = S.list_images()
    if a.list:
        by = {}
        for m in imgs:
            by.setdefault(group_of(m.get("filename")), []).append(m)
        print("specimen groups, and whether dense ground truth exists:\n")
        for g in sorted(by):
            dense = "yes (4 images)" if g == "B2" else "NONE"
            print(f"  {g:<9} {len(by[g]):>3} images   dense ground truth: {dense}")
        print("\nAM/HC is the measured gap: held-out crack recall 0.397 there against")
        print("0.755-0.816 for the same number of random images.")
        return 0

    if a.group:
        pool = [m for m in imgs if group_of(m.get("filename")) == a.group]
        if not pool:
            raise SystemExit(f"no images in group {a.group!r}")
    elif a.fragment:
        pool = [m for m in imgs
                if a.fragment.lower() in (m.get("filename") or "").lower()]
        if not pool:
            raise SystemExit(f"no loaded image matches {a.fragment!r}")
    else:
        raise SystemExit("give a frame fragment, or --group AM/HC, or --list")

    rng = np.random.RandomState(17)
    per = max(1, a.tiles // len(pool)) if len(pool) > 1 else a.tiles
    os.makedirs(OUT, exist_ok=True)
    from PIL import Image
    man, n = [], 0
    print(f"sampling {a.tiles} tile(s) of {TILE}x{TILE} uniformly inside the specimen, "
          f"across {len(pool)} frame(s)\n")
    for m in pool:
        if n >= a.tiles:
            break
        img, spots = candidates(m, rng, min(per, a.tiles - n))
        for (y, x) in spots:
            sub = img[y:y + TILE, x:x + TILE]
            u16 = (np.clip(sub, 0, 1) * 65535).astype(np.uint16)
            name = f"tile_{n:03d}.png"
            Image.fromarray(u16).save(os.path.join(OUT, name), optimize=True)
            man.append(dict(tile=name, image_id=m["id"], filename=m.get("filename"),
                            group=group_of(m.get("filename")), y=int(y), x=int(x),
                            size=TILE))
            n += 1
            print(f"  {name}  {group_of(m.get('filename')):<8} "
                  f"({y:>5},{x:>5})  {(m.get('filename') or '')[22:52]}")
        del img
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(dict(tile=TILE, uniform_random=True, seed=17, tiles=man), f, indent=1)
    px = n * TILE * TILE
    print(f"\n  {n} tiles = {px/1e6:.1f} M pixels to judge "
          f"({px/(8e6):.1f}x one 8 MP frame's worth, spread across frames)")
    print(f"  -> {os.path.relpath(OUT, PROJECT)}/  plus manifest.json")
    print("\nAnnotate each tile densely -- every pixel crack or not-crack -- in whatever")
    print("you find fastest. Uniform sampling means IoU measured on these is an unbiased")
    print("estimate of the frame IoU, with a confidence interval you can quote.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
