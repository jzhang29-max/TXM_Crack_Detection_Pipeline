#!/usr/bin/env python3
"""Rebuild the per-image export archive that ~/Desktop/txm_crack_export used to hold.

WHY THIS EXISTS. That folder was deleted on 2026-09-24 in an unrelated Desktop
reorganisation and is in no backup. It held 71 per-image directories of the form
<stem>/<stem>_crack_mask.png, and 170 references across the sibling crack-depth-3d repo read
it -- every one of that repo's 42 remaining test failures is a missing file from it.

It was the app's own export_all archive, unzipped: the ~1.4 GB figure in api_export_all's
docstring matches the deleted folder's size exactly, and the layout matches byte for byte.

WHAT THIS IS NOT. A restoration. The masks are recomputed from the model that is deployed
NOW (see the fingerprint in PROVENANCE.txt), from cached probabilities and the surviving
per-image correction.npy files. If the original archive was produced by an earlier model, the
bytes will differ, and any published number computed against the original cannot be
reproduced from this. That is why this writes a provenance file rather than quietly filling
the directory back in.

FAITHFULNESS. Every parameter is taken from api_export_all rather than chosen here, because
the point is to reproduce that archive and not to make a better one:

    threshold   = pipeline.DEFAULT_THRESHOLD  (0.60)
    postprocess = False        request.args.get("postprocess", "0")
    corrections = "gate"       _corrections_mode(default="gate")
    tight       = True         _tight(default=True)
    mask PNG    = crack BLACK on white, via the app's own _mask_png_bytes convention

"gate" matters and is worth stating: inside a crack stroke the threshold drops to
CORRECTION_FLOOR (0.20) rather than being forced True, so a stroke means "believe weaker
evidence here" and the boundary is still drawn by the image. The brush's disc geometry does
not land in the mask. That is why this export is usable as a detection result at all.

Usage:  python3 code/rebuild_export.py [--out DIR] [--limit N] [--dry-run]
"""
import argparse
import csv
import hashlib
import io
import os
import sys
import time

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, PROJECT)
Image.MAX_IMAGE_PIXELS = None

from app.core import pipeline as P            # noqa: E402
from app.core import store as S               # noqa: E402

# Sibling of the repo group, which is where crack-depth-3d's PATHS["txm_export"] now points.
DEFAULT_OUT = os.path.join(os.path.dirname(PROJECT), "txm_crack_export")


def mask_png_bytes(mask):
    """Crack = BLACK on white. Identical to app/server.py:_mask_png_bytes."""
    buf = io.BytesIO()
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def model_fingerprint():
    """Identify the model that produced this rebuild, so a later reader can tell."""
    try:
        import json
        reg = os.path.join(PROJECT, "app_data", "models", "registry.json")
        cur = json.load(open(reg))["current"]
        h = hashlib.sha256()
        for k in ("path_17", "path_hybrid"):
            p = cur.get(k) or ""
            h.update(os.path.basename(p).encode())
            if os.path.exists(p):
                h.update(str(os.path.getsize(p)).encode())
        return cur.get("label"), cur.get("created"), h.hexdigest()[:12]
    except Exception as e:
        return f"unknown ({type(e).__name__})", None, "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    imgs = [m for m in S.list_images() if m.get("has_prob")]
    if a.limit:
        imgs = imgs[:a.limit]
    if not imgs:
        sys.exit("no image has a cached prediction -- nothing to export")

    label, created, fp = model_fingerprint()
    print(f"  model: {label}  created {created}  fingerprint {fp}")
    print(f"  images with a prediction: {len(imgs)}")
    print(f"  threshold {P.DEFAULT_THRESHOLD}  corrections=gate  tight=True  postprocess=False")
    print(f"  out: {a.out}")
    if a.dry_run:
        for m in imgs[:5]:
            print(f"    would write {os.path.splitext(m.get('filename') or m['id'])[0]}/")
        return

    os.makedirs(a.out, exist_ok=True)
    summary = io.StringIO()
    sw = csv.writer(summary)
    sw.writerow(["image", "id", "height_px", "width_px", "megapixels",
                 "crack_px", "crack_area_fraction", "n_regions"])
    from skimage.measure import label as cc

    t0, ok, failed = time.time(), 0, []
    for i, m in enumerate(imgs, 1):
        iid = m["id"]
        stem = os.path.splitext(m.get("filename") or iid)[0]
        try:
            mask = P.effective_mask(iid, threshold=P.DEFAULT_THRESHOLD, postprocess=False,
                                    corrections="gate", tight=True)
            if mask is None:
                failed.append((stem, "effective_mask returned None"))
                continue
            d = os.path.join(a.out, stem)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{stem}_crack_mask.png"), "wb") as fh:
                fh.write(mask_png_bytes(mask))
            n_regions = int(cc(mask, connectivity=2).max())
            sw.writerow([m.get("filename"), iid, mask.shape[0], mask.shape[1],
                         round(mask.size / 1e6, 3), int(mask.sum()),
                         round(float(mask.mean()), 6), n_regions])
            ok += 1
            print(f"  [{i}/{len(imgs)}] {stem[:56]:<56} {100*float(mask.mean()):6.2f}% crack",
                  flush=True)
            del mask
        except Exception as e:
            failed.append((stem, f"{type(e).__name__}: {e}"))
            print(f"  [{i}/{len(imgs)}] {stem[:56]:<56} FAILED {type(e).__name__}", flush=True)

    with open(os.path.join(a.out, "summary.csv"), "w") as fh:
        fh.write(summary.getvalue())
    with open(os.path.join(a.out, "PROVENANCE.txt"), "w") as fh:
        fh.write(
            "REBUILT, NOT RESTORED\n"
            "=====================\n\n"
            "The original ~/Desktop/txm_crack_export was deleted on 2026-09-24 during an\n"
            "unrelated Desktop reorganisation, with no backup. These masks were recomputed on\n"
            f"{time.strftime('%Y-%m-%d %H:%M %Z')} by code/rebuild_export.py in\n"
            "TXM_Crack_Detection_Pipeline, reproducing the parameters of the app's own\n"
            "api_export_all rather than choosing new ones.\n\n"
            f"  model label       {label}\n"
            f"  model created     {created}\n"
            f"  model fingerprint {fp}\n"
            f"  threshold         {P.DEFAULT_THRESHOLD}\n"
            f"  corrections       gate (CORRECTION_FLOOR {P.CORRECTION_FLOOR})\n"
            "  postprocess       False\n"
            "  tight             True\n"
            "  convention        crack = BLACK (0) on white (255)\n\n"
            f"  images written    {ok}\n"
            f"  failed            {len(failed)}\n\n"
            "WHAT THIS MEANS FOR PUBLISHED NUMBERS. If the original archive was produced by an\n"
            "earlier model, these bytes differ from it. Any figure or statistic computed against\n"
            "the original export is NOT reproducible from this directory, and must either be\n"
            "recomputed against it or reported against the original with the discrepancy stated.\n"
            "The sibling crack-depth-3d repo records the original as holding 64 readable masks of\n"
            "71 frames; a different readable count here is evidence that the two differ.\n")
        if failed:
            fh.write("\nFAILED\n")
            for s, why in failed:
                fh.write(f"  {s}: {why}\n")

    print(f"\n  wrote {ok}/{len(imgs)} in {time.time()-t0:.0f}s   failed {len(failed)}")
    for s, why in failed[:5]:
        print(f"    {s}: {why}")
    print(f"  provenance: {os.path.join(a.out, 'PROVENANCE.txt')}")


if __name__ == "__main__":
    main()
