"""
Label THICK crack interiors -- the gap write_positive_crack_labels.py could
not fill and explicitly skipped.

Why a separate method is needed: that script defines crack as "dark
relative to LOCAL surroundings", which is a band-pass test. Inside a wide
crack the local background is itself dark, so the contrast vanishes and
only the crack's PERIMETER survives. Training on a perimeter would teach
"crack = ring around a dark area" and produce hollow predictions, so those
images were skipped rather than mislabelled.

A thick crack is better characterised geometrically: it is a large
ABSOLUTELY-dark region that intrudes INTO the bright specimen body. That
distinguishes it from the off-specimen empty field, which is equally dark
but lies OUTSIDE the specimen. So:

  1. specimen body   = bright material, morphologically closed and
                       hole-filled to get the specimen ENVELOPE (a crack
                       biting into the specimen is a hole in the bright
                       region, so filling recovers the envelope).
  2. thick crack     = dark AND inside that envelope AND large.
  3. reject          = anything touching the frame border that is not
                       enclosed by the envelope (that is off-specimen), and
                       round blobs (pores, not cracks).

Preview first, write second -- the earlier local-contrast attempt looked
plausible in aggregate and was wrong on inspection, so nothing here is
written without a rendered check.

Usage:
    python3 label_thick_cracks.py [--dry-run] [--preview N]
"""
import argparse, os, sys, json
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc

PREDCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache")
BRIGHT_MIN = 0.35        # flatfielded specimen normalizes bright; below this is dark
CLOSE_ITERS = 30         # close the bright region so a crack becomes an enclosed hole
MIN_AREA = 3000          # thick cracks are big; below this leave to the thin-crack pass
MIN_ECC = 0.75           # still reject round voids/pores
MAX_FRAC = 0.30          # sanity ceiling: >30% of frame "thick crack" means it failed
DOMINANCE_MIN = 0.60     # the largest detected component must be at least this share of all
                         # detected thick-crack area. A real thick crack is ONE coherent wedge;
                         # scattered dark blobs are rough surface texture near the specimen
                         # edge, which this filter rejects. Measured on Wrought: 1300_cycles is
                         # a single wedge and passes (correctly filled), whereas 1000_cycles is
                         # many small blobs where the detector was labelling texture, not the
                         # thin crack actually present in that frame.

def thick_cracks(img01):
    bright = img01 > BRIGHT_MIN
    # Largest bright component = the specimen body.
    lab = label(bright, connectivity=2)
    if lab.max() == 0:
        return np.zeros_like(bright), np.zeros_like(bright)
    sizes = ndi.sum(np.ones_like(lab), lab, index=range(1, lab.max() + 1))
    body = lab == (int(np.argmax(sizes)) + 1)
    # Close + fill holes -> specimen envelope. A crack biting into the
    # specimen is a hole in `body`, so this recovers the uncracked outline.
    env = ndi.binary_closing(body, np.ones((3, 3)), iterations=CLOSE_ITERS)
    env = ndi.binary_fill_holes(env)

    cand = env & ~bright                      # dark, but inside the specimen envelope
    cand = ndi.binary_opening(cand, np.ones((3, 3)), iterations=2)

    keep = np.zeros_like(cand)
    cl = label(cand, connectivity=2)
    areas = []
    for r in regionprops(cl):
        if r.area < MIN_AREA or r.eccentricity < MIN_ECC:
            continue
        keep[cl == r.label] = True
        areas.append(r.area)
    # Coherence check: reject scattered-blob detections (surface texture).
    if areas and max(areas) / float(sum(areas)) < DOMINANCE_MIN:
        return np.zeros_like(cand), env
    return keep, env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()
    pd = os.path.join(pc.PROJECT_DIR, "results", "thick_crack_preview")
    if args.preview:
        os.makedirs(pd, exist_ok=True)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    # Never touch the images confirmed crack-free.
    import mark_zero_crack_images as mz
    ZERO = set(mz.ZERO_CRACK)

    rows, np_ = [], 0
    for info in pc.list_images():
        nm = info["name"]
        if nm in ZERO:
            continue
        ip = os.path.join(PREDCACHE, f"{nm}_img.npy")
        if not os.path.exists(ip):
            continue
        img = np.load(ip)
        tc, env = thick_cracks(img)
        frac = tc.mean()
        if frac > MAX_FRAC or frac == 0:
            if frac > MAX_FRAC:
                print(f"  [reject {frac*100:5.1f}%] {nm[:52]}")
            del img, tc, env
            continue
        cp = os.path.join(pc.CORRECTIONS_DIR, f"{nm}_correction.npy")
        corr = np.load(cp) if os.path.exists(cp) else np.zeros(img.shape, np.uint8)
        if corr.shape != img.shape:
            del img, tc, env; continue
        tgt = tc & (corr == 0)
        n = int(tgt.sum()); corr[tgt] = 1
        if not args.dry_run:
            np.save(cp, corr)
        rows.append(dict(name=nm, group=info.get("group","?"), forced_crack=n, frac=float(frac)))
        if n:
            print(f"  {n:9,d} px -> THICK CRACK ({frac*100:5.2f}%)  [{info.get('group','?')[:20]:20s}] {nm[:42]}")
        if args.preview and np_ < args.preview and n:
            g=(np.clip(img,0,1)*255).astype(np.uint8); rgb=np.stack([g]*3,-1); rgb[tc]=[255,0,255]
            fig,ax=plt.subplots(1,2,figsize=(11,4.4))
            ax[0].imshow(g,cmap='gray',aspect='auto'); ax[0].set_title('flatfielded',fontsize=9)
            ax[1].imshow(rgb,aspect='auto'); ax[1].set_title(f'thick-crack fill ({frac*100:.2f}%)',fontsize=9)
            for a in ax: a.set_xticks([]); a.set_yticks([])
            fig.suptitle(nm[:66],fontsize=8); fig.tight_layout()
            fig.savefig(os.path.join(pd,f'{np_:02d}.png'),dpi=95,bbox_inches='tight'); plt.close(fig)
            np_+=1
        del img, tc, env, corr
    tot=sum(r['forced_crack'] for r in rows)
    print(f"\n{'DRY RUN' if args.dry_run else 'Wrote'}: {tot:,} px thick-crack across {len(rows)} images")
    by={}
    for r in rows: by.setdefault(r['group'],[]).append(r)
    for g,rs in sorted(by.items()):
        print(f"  {g:26s} {sum(x['forced_crack'] for x in rs):11,d} px / {len(rs)} images")
    if not args.dry_run:
        json.dump(rows, open(os.path.join(pc.PROJECT_DIR,'results','thick_crack_summary.json'),'w'), indent=2)

if __name__ == "__main__":
    main()
