"""Compare three embedding lookups on continuity, area and false positives.

  last-wins  what ships today: the last tile covering a pixel wins outright, so the
             embedding steps at every switch line.
  blend 64   weight 1 for every tile containing the pixel, ramping to 0 up to 64 px past
             its edge. Removes the switch-line step but averages two real embeddings flatly
             across the whole overlap band and then steps back to one tile at the band edge.
  hann       raised Hann window over each tile's own extent, so weight falls to ~0 at that
             tile's edge: continuous wherever tiles overlap, interiors dominated by their own
             real embedding, nothing ever read from outside a tile.

Continuity is scored without assuming where the seams are. Rather than checking known switch
lines, this ranks every interior column (and row) by mean |dp| and reports the worst few
against the median column, which catches discontinuities the switch-line metric would miss --
notably the band-edge steps that `blend` is suspected of introducing.
"""
import os, sys, time
import numpy as np
PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, os.path.join(PROJECT, "code"))
sys.path.insert(0, os.path.join(PROJECT, "research", "code"))
import model as M, store as S, pipeline as P
from pilot_seams import blended_rows, lastwins_rows, hann_rows
from pilot_seam_margin import predict

def worst_lines(prob, axis, k=3):
    """Top-k most discontinuous lines and their ratio to the median line.

    axis=1 scans columns (vertical seams), axis=0 scans rows.
    """
    d = np.abs(np.diff(prob, axis=axis)).mean(axis=1 - axis)
    med = float(np.median(d))
    idx = np.argsort(d)[::-1][:k]
    return [(int(i), float(d[i]), float(d[i]) / med if med else float("inf")) for i in idx], med

def tile_lines(shape):
    """tiles() yields (y0, y1, x0, x1) -- index 1 is a tile END, not an x start."""
    tl = M.tiles(shape)
    xs = sorted({int(t[2]) for t in tl})
    ys = sorted({int(t[0]) for t in tl})
    return xs, ys

def main():
    mdl = P.get_model()
    seamy = [m for m in S.list_images()
             if m.get("has_prob") and (m.get("width") or 0) > 2*M.TILE
             and (m.get("height") or 0) > 2*M.TILE
             and "SELFTEST" not in (m.get("filename") or "")]
    seamy.sort(key=lambda x: (x.get("megapixels") or 0))
    sf = seamy[0]
    clean = [m for m in S.list_images()
             if any(k.lower() in (m.get("filename") or "").lower() for k in P.CLEAN_SPECIMENS)]
    clean.sort(key=lambda x: (x.get("megapixels") or 0))
    clean = clean[:3]
    shp = (sf.get("height"), sf.get("width"))
    xs, ys = tile_lines(shp)
    print(f"  seam frame {(sf.get('filename') or '')[22:50]}  {shp[1]}x{shp[0]}")
    print(f"  tile starts x={xs} y={ys}  (overlap x={M.TILE-(xs[1]-xs[0])} px)\n", flush=True)

    fns = [("last-wins", lastwins_rows),
           ("blend 64", lambda c, e, r, x: blended_rows(c, e, r, x, 64)),
           ("hann", hann_rows)]
    for name, fn in fns:
        t0 = time.time()
        prob, _ = predict(sf["id"], fn, mdl)
        cols, mcol = worst_lines(prob, 1)
        rows, mrow = worst_lines(prob, 0)
        crack = P.prune_specks(prob > 0.5).mean() * 100
        fps = [P.prune_specks(predict(m["id"], fn, mdl)[0] > 0.5).mean()*100 for m in clean]
        print(f"  {name}   ({time.time()-t0:.0f}s)")
        print(f"    worst columns " + "  ".join(f"x={i} {r:.1f}x" for i, _, r in cols))
        print(f"    worst rows    " + "  ".join(f"y={i} {r:.1f}x" for i, _, r in rows))
        print(f"    crack {crack:.2f}%   false positives on 3 crack-free "
              f"{np.mean(fps):.3f}%  {['%.3f' % f for f in fps]}\n", flush=True)
    print("HANN_DONE")

if __name__ == "__main__":
    main()
