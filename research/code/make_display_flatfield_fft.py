"""
Build the markup DISPLAY set: the flatfielded images with FFT notch filtering
applied on top, so both corrections are in effect at once.

Why on top of the flatfielded TIFFs rather than from raw: the flatfielded set in
'TXM DATA processed/flatfielded' was produced outside this repository, so
reimplementing that step here would risk a subtly different result. Starting
from those files keeps the owner's own flat-fielding exactly as it is and adds
only the FFT stage.

What the notch removes, and what it deliberately does not. Mosaic tile seams are
straight horizontal and vertical periodic structure, which in the 2D Fourier
transform is energy concentrated ON the u=0 and v=0 axes away from DC. The
filter suppresses narrow peaks there and leaves everything off-axis untouched,
so isotropic and diagonal structure -- which is what a crack is -- passes
through. A broad low-pass or a full periodic-component subtraction would also
attenuate cracks; this does not.

Guardrails, because a filter that quietly eats cracks is the exact failure this
project has already made four times:
  - DC and a low-frequency neighbourhood are excluded, so overall brightness and
    the broad illumination trend are preserved.
  - Peaks are notched only where they exceed the local axis background by
    PEAK_K sigma, so an image with no seams is left essentially unchanged.
  - On the 4 ground-truth images it reports crack-region contrast BEFORE and
    AFTER. If crack contrast drops, the filter is hurting visibility and the
    numbers say so instead of hiding it.

Usage:
    python3 make_display_flatfield_fft.py --only b2_336_25 --preview
    python3 make_display_flatfield_fft.py
"""

import argparse
import json
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
from txm_features import robust_normalize

OUT_DIR = os.path.join(pc.PROJECT_DIR, "paint", "display_ff_fft")
REPORT = os.path.join(pc.PROJECT_DIR, "results", "improve", "display_ff_fft.json")

DC_EXCLUDE = 6        # keep the lowest frequencies: brightness + broad trend
PEAK_K = 3.0          # notch an axis bin only if it exceeds local background by this many sigma
NOTCH_HALFWIDTH = 1   # bins either side of a detected peak to suppress
AXIS_BAND = 2         # how many rows/cols around each axis count as "on axis"


def _axis_peaks(mag_axis, dc_exclude=DC_EXCLUDE, k=PEAK_K):
    """Indices along one Fourier axis whose magnitude is anomalously high.

    Compared against a running median/MAD background rather than a global mean,
    so a broad 1/f falloff does not get mistaken for peaks.
    """
    n = len(mag_axis)
    x = mag_axis.copy()
    x[:dc_exclude] = 0.0
    if n > dc_exclude:
        x[-dc_exclude:] = 0.0
    # background from a wide median filter
    from scipy.ndimage import median_filter
    bg = median_filter(x, size=max(9, n // 64) | 1)
    resid = x - bg
    nz = resid[resid > 0]
    if nz.size == 0:
        return np.zeros(0, np.intp)
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    if mad <= 0:
        mad = nz.std() or 1.0
    return np.flatnonzero(resid > k * mad)


def destitch_fft(img):
    """Suppress axis-aligned periodic structure. Returns (filtered, stats)."""
    f = np.fft.fft2(img.astype(np.float32))
    fs = np.fft.fftshift(f)
    H, W = fs.shape
    cy, cx = H // 2, W // 2
    mag = np.abs(fs)

    # magnitude profiles along the two central bands
    row_prof = mag[max(cy - AXIS_BAND, 0):cy + AXIS_BAND + 1, :].mean(axis=0)
    col_prof = mag[:, max(cx - AXIS_BAND, 0):cx + AXIS_BAND + 1].mean(axis=1)

    # peak indices are relative to the shifted centre
    px = _axis_peaks(np.roll(row_prof, -cx))
    py = _axis_peaks(np.roll(col_prof, -cy))

    keep = np.ones((H, W), np.float32)
    n_notched = 0
    for i in px:
        u = (i + cx) % W
        for d in range(-NOTCH_HALFWIDTH, NOTCH_HALFWIDTH + 1):
            uu = np.clip(u + d, 0, W - 1)
            if abs(uu - cx) <= DC_EXCLUDE:
                continue
            keep[max(cy - AXIS_BAND, 0):cy + AXIS_BAND + 1, uu] = 0.0
            n_notched += 1
    for j in py:
        v = (j + cy) % H
        for d in range(-NOTCH_HALFWIDTH, NOTCH_HALFWIDTH + 1):
            vv = np.clip(v + d, 0, H - 1)
            if abs(vv - cy) <= DC_EXCLUDE:
                continue
            keep[vv, max(cx - AXIS_BAND, 0):cx + AXIS_BAND + 1] = 0.0
            n_notched += 1

    out = np.real(np.fft.ifft2(np.fft.ifftshift(fs * keep))).astype(np.float32)
    removed = float(1.0 - (np.abs(fs * keep) ** 2).sum() / max((mag ** 2).sum(), 1e-12))
    stats = dict(n_axis_peaks_x=int(len(px)), n_axis_peaks_y=int(len(py)),
                 n_notched_bins=int(n_notched), energy_removed_frac=removed,
                 corr_with_input=float(np.corrcoef(img.ravel(), out.ravel())[0, 1]))
    return out, stats


def crack_contrast(img, gt):
    """Mean |crack - surroundings| contrast, the thing that must NOT drop."""
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(gt, iterations=6) & ~gt
    if not gt.any() or not ring.any():
        return float("nan")
    return float(abs(img[gt].mean() - img[ring].mean()))


def gt_for(name):
    """Ground truth for the 4 B2 images, matched by dataset_cache stem."""
    import glob as _g
    for p in _g.glob(os.path.join(pc.PROJECT_DIR, "dataset_cache", "*_gt.npy")):
        stem = os.path.basename(p)[:-7]
        key = stem.replace("LARGE_343_75", "343_75_LARGE")
        if key in name:
            return np.load(p).astype(bool)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--preview", action="store_true", help="write a before/after PNG")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    infos = [i for i in pc.list_images() if not args.only or args.only in i["name"]]
    print(f"flatfielded + FFT display set -> {OUT_DIR}\n{len(infos)} images\n")

    rows = []
    for k, info in enumerate(infos, 1):
        name = info["name"]
        ffp = pc.display_path_for(pc._find_path(name), "flatfielded")
        if ffp is None:
            print(f"  [{k:2d}] no flatfielded counterpart: {name[:44]}")
            continue
        ff = robust_normalize(tifffile.imread(ffp).astype(np.float64), 1.0, 99.0).astype(np.float32)
        out, st = destitch_fft(ff)
        out01 = robust_normalize(out.astype(np.float64), 1.0, 99.0).astype(np.float32)

        gt = gt_for(name)
        if gt is not None and gt.shape == ff.shape:
            st["crack_contrast_before"] = crack_contrast(ff, gt)
            st["crack_contrast_after"] = crack_contrast(out01, gt)
            st["crack_contrast_ratio"] = (st["crack_contrast_after"]
                                          / max(st["crack_contrast_before"], 1e-9))

        np.save(os.path.join(OUT_DIR, f"{name}_img.npy"), out01)
        st.update(name=name, group=info.get("group"))
        rows.append(st)
        extra = ""
        if "crack_contrast_ratio" in st:
            extra = f"  crackContrast x{st['crack_contrast_ratio']:.3f}"
        print(f"  [{k:2d}/{len(infos)}] {info.get('group','?')[:18]:18s} "
              f"peaks {st['n_axis_peaks_x']:3d}x/{st['n_axis_peaks_y']:3d}y  "
              f"energy -{st['energy_removed_frac']*100:5.2f}%  "
              f"corr {st['corr_with_input']:.4f}{extra}  {name[:30]}")

        if args.preview:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
            ax[0].imshow(ff, cmap="gray", vmin=0, vmax=1); ax[0].set_title("flatfielded")
            ax[1].imshow(out01, cmap="gray", vmin=0, vmax=1); ax[1].set_title("flatfielded + FFT notch")
            for a in ax: a.set_xticks([]); a.set_yticks([])
            fig.tight_layout()
            pp = os.path.join(pc.PROJECT_DIR, "results", "improve", f"ff_fft_preview_{name[:36]}.png")
            fig.savefig(pp, dpi=110, bbox_inches="tight"); plt.close(fig)
            print(f"      preview -> {pp}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(dict(out_dir=OUT_DIR, dc_exclude=DC_EXCLUDE, peak_k=PEAK_K,
                       notch_halfwidth=NOTCH_HALFWIDTH, axis_band=AXIS_BAND,
                       rows=rows), f, indent=2)
    if rows:
        cr = [r["crack_contrast_ratio"] for r in rows if "crack_contrast_ratio" in r]
        print(f"\n{len(rows)} written. mean energy removed "
              f"{np.mean([r['energy_removed_frac'] for r in rows])*100:.2f}%, "
              f"mean corr {np.mean([r['corr_with_input'] for r in rows]):.4f}")
        if cr:
            print(f"crack contrast ratio on the {len(cr)} ground-truth images: "
                  f"{min(cr):.3f}-{max(cr):.3f} (want >= ~1.0; below 1 means the "
                  f"filter is REDUCING crack visibility)")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
