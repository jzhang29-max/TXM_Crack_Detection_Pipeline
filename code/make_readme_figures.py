"""Regenerate the README's figures from the app's own data.

    python3 code/make_readme_figures.py                  # all of them
    python3 code/make_readme_figures.py --only detection
    python3 code/make_readme_figures.py --list           # candidate images, ranked

WHY THIS EXISTS. The first set of figures was made by hand and went stale invisibly:
they were rendered from whichever model happened to be current that evening, which
turned out to be one of the two retrains that mark ~22% of crack-free specimen as
crack. The README's hero image therefore showed thick red bands smeared across
off-specimen background, and the detection example put a small crack in the corner of a
mostly empty frame. Nothing flagged it, because nothing recorded which model drew them.

So: figures are generated, not drawn, and from an EXPLICIT model.

DEFAULTS TO THE SHIPPED BASELINE, not to whatever is deployed. A README shows a reader
what they get when they clone the repo, and a fresh clone has no retrained model -- it
has models/pixel_sam_hybrid.joblib. Rendering the figures from a locally-retrained model
would document a model that exists on exactly one laptop. Pass --model to override.

CROPS, not whole frames. These are 3-32 MP images and a README renders them ~700 px
wide. A hairline crack three pixels across survives that downsample as nothing at all,
which is why the old detection figure looked like an empty grey square with a smudge.
Every figure here is a native-resolution window chosen by find_crop(), so what a reader
sees is the actual pixel detail the model is working on.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import store as S            # noqa: E402
import pipeline as P         # noqa: E402

OUT = os.path.join(PROJECT, "docs", "img")
GT_CACHE = os.path.join(PROJECT, "dataset_cache")

CRACK_RGB = np.array([230, 70, 70])
NOT_RGB = np.array([40, 190, 210])
BG = (16, 16, 18)
FG = (228, 228, 233)
DIM = (150, 150, 158)

# The images each figure is built from. Chosen by --list, which ranks every loaded image
# on how much of a well-formed crack sits inside a croppable window; these were the top
# of that ranking, checked by eye afterwards.
DETECTION_IMAGE = "HC_316L_fatigue_1200_cycles"
PREPROCESS_IMAGE = "HC_316L_fatigue_1200_cycles"
GT_IMAGE = ("336_25", "b2_336_25")          # (ground-truth stem, app filename fragment)
# All four, for the correction figure to search: it needs a false positive that ground
# truth can vouch for, and only one image turns out to have one big enough.
GT_CANDIDATES = [("LARGE_343_75", "b2_343_75_LARGE"), ("336_25", "b2_336_25"),
                 ("338_13", "b2_338_13"), ("333_75_um_zoom", "B2_333_75_um_zoom")]

# Crops are PINNED, not searched, for the images above. find_crop() picks the window from
# a model's mask, so the same figure moved when regenerated under a different model --
# and one of those moves pushed the crack half out of frame. A README figure is a curated
# object; --list plus find_crop() is how these were found, and pinning them is how they
# stay put. (y, x, h, w) at native resolution.
CROPS = {
    # detection: a stretch of the 1200-cycle crack DELIBERATELY not the window the
    # preprocessing figure zooms into -- the auto-chosen one contained that zoom entirely,
    # so the README showed the same few hundred pixels twice.
    "detection": (500, 1590, 760, 1013),
    "HC_316L_fatigue_1200_cycles": (797, 1014, 519, 692),   # preprocessing panel 3
    "b2_336_25": (885, 422, 759, 1012),
}


def font(size, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
              else "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial Unicode.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def find_image(fragment):
    for m in S.list_images():
        if fragment.lower() in (m.get("filename") or "").lower():
            return m
    raise SystemExit(f"no loaded image matches {fragment!r} -- run code/load_all_images.py")


def model_key(name):
    """Resolve a model name to its prediction-cache key."""
    if name in (None, "baseline", "shipped"):
        for e in S.available_models():
            if e.get("created") is None:
                return e["id"], e.get("label", "shipped baseline")
        raise SystemExit("the shipped baseline is not in the registry")
    if name == "current":
        cur = S.registry()["current"]
        return S.model_key(cur), cur.get("label", "current")
    e = next((e for e in S.available_models() if e["id"] == name), None)
    if e is None:
        raise SystemExit(f"unknown model {name!r}; have: "
                         f"{[m['id'] for m in S.available_models()]}")
    return e["id"], e.get("label", name)


def prob_of(iid, key):
    p = S.load_npy_at(S.prob_cache_path(iid, key), mmap=True)
    if p is None:
        raise SystemExit(f"no cached prediction for this image under model {key}. "
                         f"Select that model in the app once, or pass --model current.")
    return np.asarray(p)


def specimen_support(raw, ds=4):
    """Delegate to the app's implementation so a figure cannot disagree with the app."""
    return P.specimen_support(raw, ds=ds)


def find_crop(mask, spec, ar=4 / 3, frac=0.45):
    """The window holding the densest run of crack that stays on the specimen."""
    H, W = mask.shape
    ch = int(min(H, max(240, H * frac)))
    cw = int(min(W, ch * ar))
    ds = 16
    kh, kw = max(1, ch // ds), max(1, cw // ds)
    dens = ndi.uniform_filter(mask[::ds, ::ds].astype(np.float32), (kh, kw), mode="constant")
    onsp = ndi.uniform_filter(spec[::ds, ::ds].astype(np.float32), (kh, kw), mode="constant")
    score = dens * (onsp > 0.75)
    if not score.any():
        score = dens
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    return (int(np.clip(cy * ds - ch // 2, 0, H - ch)),
            int(np.clip(cx * ds - cw // 2, 0, W - cw)), ch, cw)


def crop_for(fragment, mask, raw, frac=0.45):
    """The pinned window for a figure image, or a searched one for anything else."""
    c = CROPS.get(fragment)
    return c if c else find_crop(mask, specimen_support(raw), frac=frac)


def grey(a, lo_hi=(0.5, 99.5)):
    """8-bit render with a percentile stretch measured on THIS array."""
    lo, hi = np.percentile(a, lo_hi)
    g = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)
    return np.repeat((g * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def as_app_shows(iid, a):
    """8-bit render using the app's own display limits for this image.

    Figures of the processed view go through this rather than grey(), so the README and
    the canvas cannot drift apart -- a per-crop percentile stretch would give the figure
    a contrast the user never actually sees.
    """
    lo, hi = P.display_limits(iid)
    g = np.clip(np.asarray(a, np.float32), 0, 1)
    if hi - lo > 1e-4:
        g = np.clip((g - lo) / (hi - lo), 0, 1)
    return np.repeat((g * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def tint(rgb, mask, colour, alpha=0.55):
    out = rgb.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * colour).astype(np.uint8)
    return out


def _wrap(draw, text, fnt, max_w):
    """Greedy word wrap to a pixel width. PIL will not do this and will not complain."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def strip(panels, height=380, gap=10, caption=None, title_h=26, max_w=1420):
    """Lay panels out in a row: [(label, HxWx3 uint8), ...].

    Scaled to a common HEIGHT, not a common width. These crops have different aspect
    ratios, and equalising width left the tall panel hanging below the others with a
    band of dead background beside the short ones.
    """
    scale = 1.0
    total = sum(height * a.shape[1] / a.shape[0] for _, a in panels) + gap * (len(panels) + 1)
    if total > max_w:                       # keep the sheet inside a readable page width
        scale = (max_w - gap * (len(panels) + 1)) / (total - gap * (len(panels) + 1))
    h = int(height * scale)

    ims = []
    for label, arr in panels:
        w = max(1, int(round(h * arr.shape[1] / arr.shape[0])))
        im = Image.fromarray(arr).resize((w, h), Image.LANCZOS)
        p = Image.new("RGB", (w, h + title_h), BG)
        p.paste(im, (0, title_h))
        ImageDraw.Draw(p).text((3, 5), label, font=font(15, bold=True), fill=FG)
        ims.append(p)

    W = sum(i.width for i in ims) + gap * (len(ims) + 1)
    cap_font = font(14)
    cap_lines = []
    if caption:
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        cap_lines = _wrap(probe, caption, cap_font, W - 2 * gap - 6)
    cap_h = (len(cap_lines) * 19 + 10) if cap_lines else 0
    H = max(i.height for i in ims) + gap * 2 + cap_h
    sheet = Image.new("RGB", (W, H), (10, 10, 12))
    x = gap
    for i in ims:
        sheet.paste(i, (x, gap))
        x += i.width + gap
    d = ImageDraw.Draw(sheet)
    for k, line in enumerate(cap_lines):
        d.text((gap + 3, H - cap_h + 4 + k * 19), line, font=cap_font, fill=DIM)
    return sheet


def save(sheet, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    sheet.save(p, optimize=True)
    print(f"  wrote {os.path.relpath(p, PROJECT)}  {sheet.size[0]}x{sheet.size[1]}  "
          f"{os.path.getsize(p)/1e3:.0f} kB")


# ---------------------------------------------------------------- figures

def fig_detection(key, label):
    m = find_image(DETECTION_IMAGE)
    disp = np.asarray(S.load_npy(m["id"], "display.npy", mmap=True))
    raw = np.asarray(S.load_npy(m["id"], "img.npy", mmap=True))
    prob = prob_of(m["id"], key)
    mask = prob > 0.5
    y, x, ch, cw = crop_for("detection", mask, raw)
    d, mk = disp[y:y + ch, x:x + cw], mask[y:y + ch, x:x + cw]
    g = as_app_shows(m['id'], d)
    save(strip([("what you see", g),
                (f"{label}  -  red = crack", tint(g, mk, CRACK_RGB))],
               caption=f"{m['filename'][:58]}  -  native resolution, "
                       f"{cw}x{ch} px window of a {raw.shape[1]}x{raw.shape[0]} frame  -  "
                       f"AM/HC specimen: no ground truth exists for this group"),
         "example_detection.png")


def fig_ground_truth(key, label):
    stem, frag = GT_IMAGE
    m = find_image(frag)
    gt_p = os.path.join(GT_CACHE, f"{stem}_gt.npy")
    if not os.path.exists(gt_p):
        print(f"  skipping ground_truth.png -- {gt_p} missing (run unpack_package.py)")
        return
    gt = np.asarray(np.load(gt_p, mmap_mode="r")).astype(bool)
    disp = np.asarray(S.load_npy(m["id"], "display.npy", mmap=True))
    raw = np.asarray(S.load_npy(m["id"], "img.npy", mmap=True))
    prob = prob_of(m["id"], key)
    if gt.shape != prob.shape:
        print(f"  skipping ground_truth.png -- gt {gt.shape} != image {prob.shape}")
        return
    mask = prob > 0.5
    y, x, ch, cw = crop_for(GT_IMAGE[1], mask, raw, frac=0.62)
    g = as_app_shows(m['id'], disp[y:y + ch, x:x + cw])
    mk, gk = mask[y:y + ch, x:x + cw], gt[y:y + ch, x:x + cw]
    inter, union = int((mk & gk).sum()), int((mk | gk).sum())
    save(strip([("what you see", g),
                (f"{label}", tint(g, mk, CRACK_RGB)),
                ("hand-labelled ground truth", tint(g, gk, np.array([120, 200, 120])))],
               caption=f"{stem}  -  IoU {inter/max(union,1):.3f} inside this window. "
                       f"Read that as agreement, not accuracy: every model here is "
                       f"fitted on these four ground-truth images, so a score measured "
                       f"on them is in-sample. The held-out number, leave-one-image-out "
                       f"across all four, is IoU 0.821 / recall 0.914."),
         "ground_truth.png")


def fig_preprocessing(key, label):
    m = find_image(PREPROCESS_IMAGE)
    raw = np.asarray(S.load_npy(m["id"], "img.npy", mmap=True))
    disp = np.asarray(S.load_npy(m["id"], "display.npy", mmap=True))
    prob = prob_of(m["id"], key)
    # zoom on the crack so the third panel shows what the flat-field actually revealed
    y, x, ch, cw = crop_for(PREPROCESS_IMAGE, prob > 0.5, raw, frac=0.30)
    save(strip([("as uploaded (raw)", grey(raw, (0.2, 99.8))),
                ("what you mark on: FFT destitch + flat-field", as_app_shows(m["id"], disp)),
                ("the same crack, native resolution",
                 as_app_shows(m["id"], disp[y:y + ch, x:x + cw]))],
               caption=f"{m['filename'][:52]}  -  applied automatically on upload. The "
                       f"raw frame's 5x3 tile grid and bright blob are gone from the "
                       f"middle panel; measured, the notch removes 91-99% of the "
                       f"tile-pitch amplitude, and the faint grid you can still make out "
                       f"is the remainder, visible only because the specimen's contrast "
                       f"is stretched ~4x for display. The model is fed the raw image."),
         "preprocessing.png")


def fig_correction(key, label):
    """Before and after one Flip region click, on a false positive GROUND TRUTH confirms.

    Three earlier versions of this figure were rejected, and the reasons are worth
    keeping because they constrain what an honest one can be:

    1. Painting the not-crack LABEL directly is unreadable. Most not-crack pixels in this
       dataset are imported research negatives covering whole crack-free areas, so every
       candidate window came out 56-86% flat cyan with nothing else to look at.
    2. Featuring a real flip from the label data is not defensible. On the four images
       where truth exists, EVERY flip larger than 8 k px that overlaps predicted crack
       removes real crack -- 72-94% of each such region is crack in ground truth. Those
       negatives came in with the research import and are wrong; captioning one as a
       correct correction would be showing a mislabelling as an exemplar.
    3. The crack-free specimens, where any prediction is a false positive by definition,
       only carry 4-23 k px of them and mostly at the frame edge. Good news about the
       model, no figure in it.

    What is left, and what this renders: the largest coherent region the model calls
    crack that ground truth says is 0.00% crack, in a window where real crack survives
    the click. One candidate in the whole dataset meets that bar. It happens to show the
    model's documented main failure mode -- running wide of a crack rather than inventing
    one somewhere else -- which is the honest thing for this figure to show.
    """
    from skimage import measure, morphology
    best = None
    for stem, frag in ((s2, f2) for s2, f2 in GT_CANDIDATES):
        gt_p = os.path.join(GT_CACHE, f"{stem}_gt.npy")
        if not os.path.exists(gt_p):
            continue
        try:
            m = find_image(frag)
        except SystemExit:
            continue
        gt = np.asarray(np.load(gt_p, mmap_mode="r")).astype(bool)
        prob = S.load_npy_at(S.prob_cache_path(m["id"], key), mmap=True)
        if prob is None:
            continue
        mask = np.asarray(prob) > 0.5
        if mask.shape != gt.shape:
            continue
        lab = measure.label(morphology.binary_opening(mask & ~gt, morphology.disk(4)))
        for pr in sorted(measure.regionprops(lab), key=lambda q: -q.area)[:4]:
            if pr.area < 15000:
                continue
            reg = lab == pr.label
            if float(gt[reg].mean()) > 0.001:          # must be verifiably not crack
                continue
            y0, x0, y1, x1 = pr.bbox
            pad = int(0.85 * max(y1 - y0, x1 - x0))
            wy0, wx0 = max(0, y0 - pad), max(0, x0 - pad)
            wy1, wx1 = min(gt.shape[0], y1 + pad), min(gt.shape[1], x1 + pad)
            kept = float((mask & gt)[wy0:wy1, wx0:wx1].mean())
            if kept < 0.05:                            # real crack must stay in frame
                continue
            score = pr.area * min(kept, 0.20)
            if best is None or score > best[0]:
                best = (score, stem, m, mask, reg, (wy0, wx0, wy1 - wy0, wx1 - wx0), int(pr.area))
    if best is None:
        print("  skipping example_correction.png -- no ground-truth-verified false "
              "positive is large enough to feature")
        return
    _, stem, m, mask, blob, (y, x, ch, cw), area = best

    disp = np.asarray(S.load_npy(m["id"], "display.npy", mmap=True))
    g = as_app_shows(m["id"], disp[y:y + ch, x:x + cw])
    mk, bl = mask[y:y + ch, x:x + cw], blob[y:y + ch, x:x + cw]
    save(strip([(f"{label}: red = crack", tint(g, mk, CRACK_RGB)),
                ("after one Flip region click", tint(tint(g, mk & ~bl, CRACK_RGB),
                                                     bl, NOT_RGB, alpha=0.45))],
               caption=f"{stem}  -  the model ran {area:,} px past the end of this crack. "
                       f"Ground truth says 0.00% of the cyan region is crack, so the click "
                       f"is verifiably right, and the real crack in the same window stays "
                       f"red. One click on the connected blob; Erase would brush the same "
                       f"area in about a minute. Both write the same not-crack label, "
                       f"which Retrain then learns from."),
         "example_correction.png")


FIGURES = dict(detection=fig_detection, ground_truth=fig_ground_truth,
               preprocessing=fig_preprocessing, correction=fig_correction)


def rank_candidates():
    """Print every loaded image ranked as a figure candidate."""
    key, label = model_key("current")
    print(f"ranking against {label}\n")
    rows = []
    for m in S.list_images():
        p = S.load_npy_at(S.prob_cache_path(m["id"], key), mmap=True)
        if p is None:
            continue
        mask = np.asarray(p) > 0.5
        if mask.sum() < 500:
            continue
        raw = np.asarray(S.load_npy(m["id"], "img.npy", mmap=True))
        spec = specimen_support(raw)
        y, x, ch, cw = find_crop(mask, spec)
        cm, cs = mask[y:y + ch, x:x + cw], spec[y:y + ch, x:x + cw]
        off = float(cm[~cs].sum()) / max(cm.sum(), 1)
        skel = morphology.skeletonize(morphology.remove_small_objects(cm, 32))
        rows.append((float(skel.sum()) / max(np.sqrt(cm.sum()), 1e-6),
                     float(cm.mean()) * 100, off * 100, m.get("filename", "")))
    rows.sort(key=lambda r: -(r[0] * min(r[1], 25)))
    print(f"{'branchiness':>12} {'crop crack':>11} {'off-spec':>9}  image")
    for b, f, o, n in rows[:20]:
        print(f"{b:>12.1f} {f:>10.1f}% {o:>8.1f}%  {n[:58]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline",
                    help="'baseline' (default), 'current', or a model id from /api/models")
    ap.add_argument("--only", choices=sorted(FIGURES), action="append")
    ap.add_argument("--list", action="store_true", help="rank figure candidates and exit")
    a = ap.parse_args()
    if a.list:
        return rank_candidates()
    key, label = model_key(a.model)
    print(f"rendering from: {label}  (cache key {key})")
    for name in (a.only or sorted(FIGURES)):
        FIGURES[name](key, label)
    print("\napp.png is a screenshot, not generated here -- retake it from the running app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
