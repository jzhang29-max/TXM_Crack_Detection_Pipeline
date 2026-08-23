"""Is the blend's extra area a seam fix that invents crack, or just an operating-point shift?

Lookup-time blending removes the tile seams but raises predicted area everywhere, including
on crack-free specimens. Two explanations fit that: (a) the extrapolated edge cells invent
crack, in which case blending is unusable, or (b) blending smooths the embedding field and
lifts probabilities globally, in which case a higher threshold buys the seam fix back for
free. They are distinguishable: the seam ratio is measured on the probability map and so is
threshold-independent, while area is not. If a threshold exists where blended false
positives match today's and recall on the owner's strokes holds, (b) is true and blending is
a strict win. This measures that threshold.

Recall is computed on correction==1 pixels and false positives on correction==2 pixels: the
owner's strokes are thicker than the cracks they mark, so recall here is a loose proxy, but
it is the same proxy for both methods, which is what the comparison needs.
"""
import os, sys, time
import numpy as np
PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, os.path.join(PROJECT, "code"))
sys.path.insert(0, os.path.join(PROJECT, "research", "code"))
import model as M, store as S, pipeline as P
from pilot_seams import blended_rows, lastwins_rows, seam_ratio
from pilot_seam_margin import predict

THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
MARGIN = 64

def labelled(n):
    out = []
    for m in S.list_images():
        if "SELFTEST" in (m.get("filename") or ""):
            continue
        c = S.load_npy(m["id"], "correction.npy")
        if c is None:
            continue
        pos = int((c == 1).sum()); neg = int((c == 2).sum())
        if pos > 20000 and neg > 20000:
            out.append((m, pos + neg))
    out.sort(key=lambda t: -t[1])
    return [m for m, _ in out[:n]]

def main():
    mdl = P.get_model()
    lab = labelled(3)
    clean = [m for m in S.list_images()
             if any(k.lower() in (m.get("filename") or "").lower() for k in P.CLEAN_SPECIMENS)]
    clean.sort(key=lambda x: (x.get("megapixels") or 0))
    clean = clean[:3]
    seamy = [m for m in S.list_images()
             if m.get("has_prob") and (m.get("width") or 0) > 2*M.TILE
             and (m.get("height") or 0) > 2*M.TILE
             and "SELFTEST" not in (m.get("filename") or "")]
    seamy.sort(key=lambda x: (x.get("megapixels") or 0))
    seamy = seamy[:1]
    print(f"  labelled: {len(lab)}   crack-free: {len(clean)}   seam frame: "
          f"{(seamy[0].get('filename') or '')[22:52]}\n", flush=True)

    fns = {"last-wins": lastwins_rows,
           f"blend {MARGIN}": lambda c, e, r, x: blended_rows(c, e, r, x, MARGIN)}
    probs = {}
    for name, fn in fns.items():
        t0 = time.time()
        for m in lab + clean + seamy:
            probs[(name, m["id"])] = predict(m["id"], fn, mdl)[0]
        print(f"  predicted {len(lab)+len(clean)+len(seamy)} frames with {name} "
              f"({time.time()-t0:.0f}s)", flush=True)

    sf = seamy[0]
    shp = (sf.get("height"), sf.get("width"))
    print(f"\n  seam ratios on {(sf.get('filename') or '')[22:46]} "
          f"(threshold-independent):")
    for name in fns:
        p = probs[(name, sf["id"])]
        def ratio(axis):
            a, b, n = seam_ratio(p, axis, shp)
            return (a / b) if b else float("nan")
        print(f"    {name:<10} vertical {ratio(1):>5.1f}x   "
              f"horizontal {ratio(0):>5.1f}x")

    print(f"\n  {'method':<11} {'thr':>5} {'recall':>8} {'FP on notcrack':>15} "
          f"{'FP% crack-free':>15}")
    rows = []
    for name in fns:
        for thr in (THRESHOLDS if name != "last-wins" else (0.50,)):
            tp = fn_ = pos = neg = 0
            for m in lab:
                c = S.load_npy(m["id"], "correction.npy")
                mask = P.prune_specks(probs[(name, m["id"])] > thr)
                i1 = c == 1; i2 = c == 2
                tp += int(mask[i1].sum()); pos += int(i1.sum())
                fn_ += int(mask[i2].sum()); neg += int(i2.sum())
            fps = [P.prune_specks(probs[(name, m["id"])] > thr).mean()*100 for m in clean]
            rows.append((name, thr, tp/max(pos,1)*100, fn_/max(neg,1)*100, float(np.mean(fps))))
            print(f"  {name:<11} {thr:>5.2f} {rows[-1][2]:>7.2f}% {rows[-1][3]:>14.3f}% "
                  f"{rows[-1][4]:>14.3f}%", flush=True)

    base = rows[0]
    ok = [r for r in rows[1:] if r[4] <= base[4] + 1e-9]
    print()
    if ok:
        best = max(ok, key=lambda r: r[2])
        print(f"  at threshold {best[1]:.2f} the blend matches today's false positives "
              f"({best[4]:.3f}% vs {base[4]:.3f}%)")
        print(f"  and recall is {best[2]:.2f}% vs today's {base[2]:.2f}% "
              f"({best[2]-base[2]:+.2f} pp) -- the seam fix is free")
    else:
        print(f"  no threshold up to {max(THRESHOLDS):.2f} brings the blend's false positives "
              f"back to {base[4]:.3f}%; the extra area is not a pure operating-point shift")
    print("THR_DONE")

if __name__ == "__main__":
    main()
