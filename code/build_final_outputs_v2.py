"""
Best-effort final overlays + B&W masks for all 71 images.

Applies, as DETERMINISTIC POST-PROCESSING on top of the model's prediction,
the two suppressions the 46-agent audit measured as highest-leverage. Doing
them as post-processing rather than only as training labels is deliberate:
these are geometric/periodic facts about the imaging, so enforcing them
directly guarantees the improvement instead of hoping the model infers it.

  A. GEOMETRIC EXCLUSION (audit: removes ~75% of false-positive red in one
     step, with no risk to interior cracks). Drop predicted crack that lies
     outside the specimen, within BOUNDARY_INSET px inside the segmented
     specimen boundary, or within FRAME_BORDER px of the frame edge. The
     audit found the two dominant false-positive classes across 34 of 46
     images were the dark wedge/notch rim and the specimen boundary rim --
     both live in exactly this band. A previous attempt used a 45px wedge
     margin and 130px edge ring and left both classes still dominant, so
     this is stricter and boundary-distance-based rather than dilation-based.

  B. TILE-PHASE REJECTION (audit: ~57% of red in affected frames). Two
     agents independently proved the scattered "reference artifact" specks
     are mosaic-tile-locked, not microstructural, by autocorrelating the
     prediction mask: periods ~112px and ~84px, with 90.5% of interior red
     falling in the top 25% of intra-tile phase cells versus 25% expected by
     chance. Here the tile pitch is measured per image by autocorrelation of
     the IMAGE (not the mask, to avoid circularity), then predicted
     components whose centroid sits in an over-represented phase cell are
     dropped. Far stronger than the round-shape heuristic it replaces.

  C. Keep elongated interior components. A thin crack is elongated and
     central; nothing elongated and clear of the boundary band is removed.

Writes results/final_71_v2/ : <name>_crack_mask.png (crack=BLACK),
<name>_overlay.png, <name>_stats.csv, summary.csv/json, per-group montages,
and a before/after comparison against results/final_71.

Usage:
    python3 build_final_outputs_v2.py [--model models/pixel_flatfield_clean.joblib]
"""
import argparse, csv, json, os, sys, time
import joblib, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
from apply_pixel_model import postprocess_mask, predict_probability_map

PREDCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache")
OUT = os.path.join(pc.PROJECT_DIR, "results", "final_71_v4")
OLD = os.path.join(pc.PROJECT_DIR, "results", "final_71_v2")

BRIGHT_MIN     = 0.30   # flatfielded specimen is bright; below this is empty field / deep feature
BOUNDARY_INSET = 30     # audit-specified: drop this far INSIDE the specimen boundary
FRAME_BORDER   = 20     # audit-specified: drop this close to the frame edge
MIN_KEEP_AREA  = 150    # below this a component is noise
MIN_SKEL_LEN   = 45     # px of skeleton. A crack is a connected elongated PATH; surface
                        # texture is a blob. The v2 re-audit found texture/microstructure
                        # became the dominant false positive on 22 of 49 images once the
                        # wedge and rim classes were suppressed, and several auditors
                        # independently proposed exactly this gate. One measured that 84%
                        # of predicted regions have aspect ratio < 2 and only 2% exceed 4,
                        # median region 34px -- the model emits blobs, not traces.
MIN_ELONGATION = 2.0    # major/minor axis, applied as AND with the skeleton gate.
                        # First attempt used OR at 2.6 and removed essentially NOTHING
                        # (logged "curv -0.0%" on all 71, areas unchanged) -- the OR let
                        # any modestly elongated texture blob through. A crack must satisfy
                        # BOTH: long enough to be a path, and thin enough to be a trace.
KEEP_ECC       = 0.90   # elongated components are protected from phase rejection
PHASE_BINS     = 8      # intra-tile phase grid resolution
PHASE_EXCESS   = 2.5    # a phase cell holding >2.5x its expected share is artifact-locked


def specimen_mask(img01):
    """Largest bright connected region = the specimen body; filled to recover
    the envelope (a wedge biting into it is a hole in the bright region)."""
    bright = ndi.gaussian_filter(img01.astype(np.float32), 2.0) > BRIGHT_MIN
    lab = label(bright, connectivity=2)
    if lab.max() == 0:
        return np.ones_like(bright), np.ones_like(bright)
    sizes = ndi.sum(np.ones_like(lab), lab, index=range(1, lab.max() + 1))
    body = lab == int(np.argmax(sizes)) + 1
    env = ndi.binary_fill_holes(ndi.binary_closing(body, np.ones((3, 3)), iterations=20))
    return body, env


WEDGE_MIN_AREA_FRAC = 0.002   # a dark intrusion bigger than this is the WEDGE, not a crack


def allowed_region(img01):
    """Where a crack may legitimately be: inside the specimen envelope, clear
    of the outer boundary band, the frame border, AND clear of a band around
    any LARGE dark intrusion.

    That last clause is the fix for the dominant false positive. The audit
    found the wedge/notch rim and the specimen boundary rim dominate 34 of 46
    images. A first attempt filled holes to build the envelope, which put the
    wedge INSIDE the allowed region, so wedge-rim predictions survived and
    "geo removed" came out at 0.0-0.4% -- the mask barely fired.

    The wedge cannot simply be excluded as "dark", because a crack is a dark
    intrusion too. It is separated by SIZE: per the owner the real cracks are
    THIN, while the wedge is a large blunt feature. So dark regions above
    WEDGE_MIN_AREA_FRAC of the frame get a BOUNDARY_INSET margin excluded
    around them; smaller dark features (candidate cracks) are left alone.
    """
    _, env = specimen_mask(img01)
    inner = ndi.binary_erosion(env, np.ones((3, 3)), iterations=BOUNDARY_INSET)
    h, w = img01.shape
    frame = np.zeros_like(inner)
    frame[FRAME_BORDER:h - FRAME_BORDER, FRAME_BORDER:w - FRAME_BORDER] = True
    allowed = inner & frame

    dark = ndi.gaussian_filter(img01.astype(np.float32), 2.0) <= BRIGHT_MIN
    lab = label(dark, connectivity=2)
    min_area = WEDGE_MIN_AREA_FRAC * dark.size
    big = np.zeros_like(dark)
    for r in regionprops(lab):
        if r.area >= min_area:
            big[lab == r.label] = True
    if big.any():
        allowed &= ~ndi.binary_dilation(big, np.ones((3, 3)), iterations=BOUNDARY_INSET)
    return allowed


def tile_pitch(img01, lo=60, hi=400):
    """Dominant mosaic tile pitch per axis, by autocorrelation of the IMAGE."""
    a = img01.astype(np.float32)
    a = a - a.mean()
    out = []
    for prof in (a.mean(axis=1), a.mean(axis=0)):        # row profile, col profile
        p = prof - prof.mean()
        if p.size < 2 * hi or not np.isfinite(p).all() or p.std() == 0:
            out.append(None); continue
        ac = np.correlate(p, p, mode="full")[p.size - 1:]
        ac = ac / (ac[0] + 1e-12)
        seg = ac[lo:min(hi, ac.size)]
        out.append(int(lo + np.argmax(seg)) if seg.size and seg.max() > 0.05 else None)
    return out  # (pitch_y, pitch_x)


def reject_tile_phase(pred, img01, allowed):
    """Drop compact components whose intra-tile phase cell is over-represented
    -- the signature of a periodic instrument artifact rather than material."""
    py, px = tile_pitch(img01)
    if not py or not px:
        return pred, 0
    lab = label(pred & allowed, connectivity=2)
    regs = [r for r in regionprops(lab) if r.area >= MIN_KEEP_AREA]
    if len(regs) < 12:                       # too few to establish a phase pattern
        return pred, 0
    cells = {}
    for r in regs:
        cy, cx = r.centroid
        key = (int((cy % py) / py * PHASE_BINS), int((cx % px) / px * PHASE_BINS))
        cells.setdefault(key, []).append(r)
    expected = len(regs) / float(PHASE_BINS * PHASE_BINS)
    out = pred.copy(); removed = 0
    for key, rs in cells.items():
        if len(rs) <= expected * PHASE_EXCESS:
            continue
        for r in rs:
            if r.eccentricity >= KEEP_ECC:   # never drop an elongated (crack-like) component
                continue
            out[lab == r.label] = False
            removed += r.area
    return out, int(removed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(pc.PROJECT_DIR, "models", "pixel_flatfield_hgb.joblib"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    model = joblib.load(args.model)
    old = {}
    if os.path.exists(os.path.join(OLD, "summary.json")):
        old = {r["name"]: r for r in json.load(open(os.path.join(OLD, "summary.json")))}

    rows = []
    for info in pc.list_images():
        nm, grp = info["name"], info.get("group", "?")
        ip = os.path.join(PREDCACHE, f"{nm}_img.npy")
        if not os.path.exists(ip):
            print(f"  [skip] {nm[:56]}"); continue
        t0 = time.time()
        img01 = np.load(ip).astype(np.float64)
        raw = postprocess_mask(predict_probability_map(model, img01))
        allowed = allowed_region(img01)
        geo = raw & allowed
        final, n_phase = reject_tile_phase(geo, img01, allowed)
        final = ndi.binary_opening(final, np.ones((3, 3)))

        # Curvilinearity gate: keep a component only if it looks like a PATH.
        # Skeleton length is the primary test because it is what actually
        # separates a crack (long thin connected trace) from a texture blob of
        # the same area. Elongation is accepted as an alternative so a short but
        # clearly linear segment is not discarded.
        lab = label(final, connectivity=2)
        keep = np.zeros_like(final)
        n_texture_dropped = 0
        for r in regionprops(lab):
            if r.area < MIN_KEEP_AREA:
                continue
            skel_len = int(skeletonize(r.image).sum())
            minor = max(r.minor_axis_length, 1e-6)
            elong = r.major_axis_length / minor
            if skel_len >= MIN_SKEL_LEN and elong >= MIN_ELONGATION:
                keep[lab == r.label] = True
            else:
                n_texture_dropped += r.area
        final = keep
        lab = label(final, connectivity=2)

        Image.fromarray(np.where(final, 0, 255).astype(np.uint8), "L").save(os.path.join(OUT, f"{nm}_crack_mask.png"))
        g = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
        ov = np.stack([g] * 3, -1); ov[final] = [255, 0, 0]
        Image.fromarray(ov, "RGB").save(os.path.join(OUT, f"{nm}_overlay.png"))
        with open(os.path.join(OUT, f"{nm}_stats.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id","area_px","solidity","eccentricity","centroid_row","centroid_col"])
            for r in regionprops(lab):
                w.writerow([r.label, r.area, f"{r.solidity:.4f}", f"{r.eccentricity:.4f}",
                            f"{r.centroid[0]:.1f}", f"{r.centroid[1]:.1f}"])

        rows.append(dict(name=nm, group=grp, area_fraction=float(final.mean()), n_regions=int(lab.max()),
                         raw_area=float(raw.mean()), after_geo=float(geo.mean()),
                         removed_by_geometry=float((raw & ~allowed).mean()),
                         removed_by_tile_phase_px=n_phase,
                         removed_by_curvilinearity_px=int(n_texture_dropped),
                         prev_area=old.get(nm, {}).get("area_fraction")))
        pv = old.get(nm, {}).get("area_fraction")
        print(f"  [{time.time()-t0:5.1f}s] raw {raw.mean()*100:5.1f}% -> geo {geo.mean()*100:5.1f}% -> curv -{n_texture_dropped/final.size*100:4.1f}% -> final "
              f"{final.mean()*100:5.1f}% ({lab.max():4d}rg)" + (f"  was {pv*100:5.1f}%" if pv else "") +
              f"  [{grp[:18]:18s}] {nm[:34]}")
        del img01, raw, allowed, geo, final, ov, lab

    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name","group","area_fraction","n_regions","prev_area_fraction"])
        for r in rows: w.writerow([r["name"], r["group"], f"{r['area_fraction']:.5f}", r["n_regions"],
                                    f"{r['prev_area']:.5f}" if r["prev_area"] else ""])
    json.dump(rows, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    print(f"\n{len(rows)} images -> {OUT}")
    by = {}
    for r in rows: by.setdefault(r["group"], []).append(r)
    print(f"\n{'group':26s} {'n':>3s} {'prev':>8s} {'NEW':>8s} {'geo removed':>12s}")
    for g, rs in sorted(by.items()):
        pv = [r["prev_area"] for r in rs if r["prev_area"]]
        print(f"{g:26s} {len(rs):3d} {np.median(pv)*100 if pv else 0:7.1f}% "
              f"{np.median([r['area_fraction'] for r in rs])*100:7.1f}% "
              f"{np.median([r['removed_by_geometry'] for r in rs])*100:11.1f}%")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for g, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["name"]); cols = 5; rws = (len(rs)+cols-1)//cols
        fig, axes = plt.subplots(rws, cols, figsize=(3.0*cols, 2.1*rws)); axes = np.atleast_2d(axes)
        for k, r in enumerate(rs):
            ax = axes[k//cols, k%cols]
            p = os.path.join(OUT, f"{r['name']}_overlay.png")
            if os.path.exists(p):
                im = Image.open(p); im.thumbnail((420,420)); ax.imshow(np.array(im), aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{r['name'].split('_idx')[0][-26:]}\n{r['area_fraction']*100:.1f}%, {r['n_regions']}rg", fontsize=6)
        for k in range(len(rs), rws*cols): axes[k//cols, k%cols].axis("off")
        fig.suptitle(f"{g} -- FINAL v4 HGB ({len(rs)} images)", fontsize=12)
        fig.tight_layout(rect=[0,0,1,0.97])
        fig.savefig(os.path.join(OUT, f"_montage_{g.replace(' ','_')}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("montages written")

if __name__ == "__main__":
    main()
