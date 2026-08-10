"""
Write paint-tool-format correction masks for the new images, from
(a) the flatfielded model's prediction and (b) an automatic off-specimen
detection, informed by the visual review of all 71 images.

Produces exactly the artifact the browser paint tool produces --
paint/corrections/<name>_correction.npy, uint8, 0=untouched,
1=force-crack, 2=force-not-crack -- so downstream training code needs no
changes and a human can still open any image in the tool afterwards to
adjust by hand.

WHAT IS CORRECTED, and why only this:

  Off-specimen false positives. In a FLATFIELDED image the specimen
  material is normalized to a near-uniform bright level while genuinely
  empty field / out-of-specimen area stays dark, so the boundary is
  unambiguous -- unlike the raw images, where a broad brightness gradient
  made "dark" mean both "crack" and "off-specimen" and is exactly why the
  raw-trained model flooded whole frames. Any predicted-crack pixel lying
  in a large dark region connected to the frame border is forced to
  not-crack.

  This is deliberately the ONLY automatic correction. It encodes a fact
  about the imaging geometry (there is no material there, so there cannot
  be a crack there), not a judgment about crack morphology. Subtler calls
  -- faint thinning vs. hairline crack, pore vs. crack tip -- are NOT
  auto-labelled, because getting those wrong would put confidently-wrong
  labels into training, which is worse than leaving them untouched: an
  untouched pixel simply contributes no training signal.

  Frames the review judged unusable (essentially no discernible specimen)
  are skipped entirely rather than labelled.

Usage:
    python3 write_corrections_from_review.py [--review review.json] [--dry-run]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc

PREDCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache")

# Off-specimen detection thresholds. Tuned against the flatfielded
# intensity distribution: specimen normalizes to ~0.5-1.0, empty field
# sits far below that.
OFFSPEC_INTENSITY_MAX = 0.25    # flatfielded intensity below this is candidate empty field
OFFSPEC_MIN_AREA_FRAC = 0.005   # ignore small dark specks; only large regions count
OFFSPEC_SMOOTH_SIGMA = 8.0      # smooth before thresholding so texture doesn't fragment it


def detect_offspecimen(img01):
    """Large dark regions touching the frame border = empty field / outside
    the specimen. Border-connectivity matters: a dark blob in the middle of
    otherwise-bright material is far more likely to be a real void or crack
    than to be off-specimen, so it is deliberately NOT flagged."""
    sm = ndi.gaussian_filter(img01.astype(np.float32), sigma=OFFSPEC_SMOOTH_SIGMA)
    dark = sm < OFFSPEC_INTENSITY_MAX
    if not dark.any():
        return np.zeros_like(dark)

    lab = label(dark, connectivity=2)
    h, w = dark.shape
    border_labels = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    border_labels.discard(0)

    out = np.zeros_like(dark)
    min_area = OFFSPEC_MIN_AREA_FRAC * dark.size
    for lb in border_labels:
        region = lab == lb
        if region.sum() >= min_area:
            out |= region
    # Grow slightly: the specimen edge has a soft transition ring that also
    # isn't real material.
    if out.any():
        out = ndi.binary_dilation(out, structure=np.ones((5, 5)), iterations=3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default=None, help="JSON from the visual review (findings list)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    unusable = set()
    if args.review and os.path.exists(args.review):
        with open(args.review) as f:
            rev = json.load(f)
        findings = rev.get("findings", rev if isinstance(rev, list) else [])
        verification = rev.get("verification", {})
        for f_ in findings:
            # Only skip when the independent second look AGREED it's unusable.
            if not f_.get("usable", True):
                v = verification.get(f_.get("file", ""), {})
                if not verification or v.get("agree_unusable"):
                    unusable.add(f_["name"])
        print(f"Review marks {len(unusable)} frame(s) unusable (confirmed by second pass); skipping those.\n")

    existing = {os.path.basename(p)[: -len("_correction.npy")]
                for p in glob.glob(os.path.join(pc.CORRECTIONS_DIR, "*_correction.npy"))}

    written, skipped, rows = 0, 0, []
    for info in pc.list_images():
        name = info["name"]
        if name in unusable:
            print(f"  [unusable] {name[:62]}")
            skipped += 1
            continue

        mask_p = os.path.join(PREDCACHE, f"{name}_mask.npy")
        img_p = os.path.join(PREDCACHE, f"{name}_img.npy")
        if not (os.path.exists(mask_p) and os.path.exists(img_p)):
            print(f"  [no prediction] {name[:62]}")
            skipped += 1
            continue

        pred = np.load(mask_p)
        img01 = np.load(img_p)
        offspec = detect_offspecimen(img01)

        corr_path = os.path.join(pc.CORRECTIONS_DIR, f"{name}_correction.npy")
        # Preserve any human corrections already made for this image -- never
        # overwrite real hand-labelled work with an automatic pass.
        if name in existing:
            corr = np.load(corr_path)
            preexisting = int((corr != 0).sum())
        else:
            corr = np.zeros(pred.shape, dtype=np.uint8)
            preexisting = 0

        if corr.shape != pred.shape:
            print(f"  [SHAPE MISMATCH] {name[:56]}: correction {corr.shape} vs pred {pred.shape}")
            skipped += 1
            continue

        # Force not-crack where the model predicted crack inside off-specimen
        # area, but never clobber an existing human label.
        target = pred & offspec & (corr == 0)
        n_new = int(target.sum())
        corr[target] = 2

        rows.append(dict(name=name, group=info.get("group", "?"),
                         pred_area=float(pred.mean()), offspec_area=float(offspec.mean()),
                         forced_not_crack=n_new, preexisting_corrected=preexisting))
        if n_new:
            print(f"  {n_new:9,d} px -> not-crack ({n_new/pred.size*100:5.2f}% of frame)  {name[:52]}")
        if not args.dry_run:
            np.save(corr_path, corr)
        written += 1
        del pred, img01, offspec, corr

    out = os.path.join(pc.PROJECT_DIR, "results", "correction_write_summary.json")
    if not args.dry_run:
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
    tot = sum(r["forced_not_crack"] for r in rows)
    print(f"\n{'DRY RUN -- nothing written' if args.dry_run else f'Wrote {written} correction files'}"
          f"; {skipped} skipped; {tot:,} px total forced to not-crack")
    if not args.dry_run:
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
