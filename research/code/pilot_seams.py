"""Can the SAM tile seams be removed WITHOUT re-embedding?

THE DEFECT. embed_image tiles the frame at 1024 px with no overlap, and the lookup takes
"last tile wins". Adjacent tiles are embedded independently, so the feature field steps at
every boundary. Measured on b2_343_75_LARGE: the mean |dp/dx| at 1024-px boundaries is 8.9x
the normal column-to-column value, and 12.6x for |dp/dy|. Where a step crosses the 0.5
threshold it draws a straight edge with square corners -- geometry no crack has.

TWO FIXES.

  (a) Overlapping tiles with blended embeddings. Correct, and costs a full re-embed: tiles
      every 768 px is 1.78x the passes, about 3.2 hours at SAM 1 ViT-H's 6.19 s/tile.

  (b) PARTITION-OF-UNITY BLENDING AT LOOKUP TIME, which is free. interp_tile already clamps
      out-of-range coordinates, so every tile can be evaluated slightly outside its own
      domain -- its edge cells extended. Give each tile a weight that is 1 in its interior
      and ramps to 0 a margin beyond its edge, then take the weighted average over the tiles
      that reach a pixel. The field becomes continuous by construction.

      The honest caveat: outside its domain a tile contributes extrapolated edge values, not
      a real embedding of that region. So (b) buys continuity, not information. Whether that
      is enough is what this measures.

Run: python3 research/code/pilot_seams.py
"""
import os, sys, time
import numpy as np
PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, os.path.join(PROJECT, "code"))
import model as M, store as S, pipeline as P
from txm_features import compute_feature_stack

MARGIN = 192          # how far past its own edge a tile is allowed to reach


def blended_rows(coords, embs, rr, cc, margin=MARGIN):
    """Weighted average over every tile whose domain, extended by `margin`, covers the pixel.

    Weight ramps linearly from 1 at the tile edge to 0 at `margin` beyond it, in each axis
    independently, so the result is continuous across a boundary instead of stepping.
    """
    C = embs.shape[1]
    acc = np.zeros((len(rr), C), np.float32)
    wsum = np.zeros(len(rr), np.float32)
    for t in range(len(coords)):
        y0, x0 = int(coords[t][0]), int(coords[t][1])
        dy = np.maximum(np.maximum(y0 - rr, rr - (y0 + M.TILE - 1)), 0).astype(np.float32)
        dx = np.maximum(np.maximum(x0 - cc, cc - (x0 + M.TILE - 1)), 0).astype(np.float32)
        w = np.clip(1.0 - dy / margin, 0, 1) * np.clip(1.0 - dx / margin, 0, 1)
        sel = w > 0
        if not sel.any():
            continue
        vals = M.interp_tile(embs[t], rr[sel] - y0, cc[sel] - x0)
        acc[sel] += vals * w[sel, None]
        wsum[sel] += w[sel]
    # a pixel is always inside at least one tile, so wsum > 0 everywhere
    return acc / np.maximum(wsum, 1e-6)[:, None]


def hann_rows(coords, embs, rr, cc, eps=1e-3):
    """Partition-of-unity average with a window that vanishes at each tile's own edge.

    `blended_rows` gives every tile containing the pixel a weight of exactly 1, so across a
    197 px overlap band it returns a flat average of two real embeddings and then steps back
    to a single tile at the band edge -- it moves the discontinuity instead of removing it,
    and it feeds the hybrid member an averaged vector it never saw in training. A raised
    Hann window over each tile's own extent instead falls smoothly to ~0 at that tile's edge,
    so the field is continuous wherever tiles overlap, a tile's interior is dominated by its
    own real embedding, and no value is ever read from outside a tile.
    """
    C = embs.shape[1]
    acc = np.zeros((len(rr), C), np.float32)
    wsum = np.zeros(len(rr), np.float32)
    for t in range(len(coords)):
        y0, x0 = int(coords[t][0]), int(coords[t][1])
        ly = rr - y0; lx = cc - x0
        sel = (ly >= 0) & (ly < M.TILE) & (lx >= 0) & (lx < M.TILE)
        if not sel.any():
            continue
        wy = 0.5 * (1.0 - np.cos(2 * np.pi * (ly[sel] + 0.5) / M.TILE))
        wx = 0.5 * (1.0 - np.cos(2 * np.pi * (lx[sel] + 0.5) / M.TILE))
        w = ((wy * wx) * (1.0 - eps) + eps).astype(np.float32)
        vals = M.interp_tile(embs[t], ly[sel], lx[sel])
        acc[sel] += vals * w[:, None]
        wsum[sel] += w
    return acc / np.maximum(wsum, 1e-12)[:, None]


def lastwins_rows(coords, embs, rr, cc):
    """What the app does today."""
    out = np.zeros((len(rr), embs.shape[1]), np.float32)
    todo = np.ones(len(rr), bool)
    for t in range(len(coords) - 1, -1, -1):
        y0, x0 = int(coords[t][0]), int(coords[t][1])
        sel = (todo & (rr >= y0) & (rr < y0 + M.TILE) & (cc >= x0) & (cc < x0 + M.TILE))
        if sel.any():
            out[sel] = M.interp_tile(embs[t], rr[sel] - y0, cc[sel] - x0)
            todo &= ~sel
    return out


def switch_columns(shape, axis):
    """Where last-tile-wins actually changes tile, derived from tiles() rather than assumed.

    THE FIRST VERSION OF THIS ASSUMED MULTIPLES OF 1024 AND WAS WRONG. tiles() clamps the
    final tile inward so every tile is exactly TILE wide, which means on a frame narrower
    than 2*TILE the tiles OVERLAP -- for width 1706 they sit at x=0 and x=682, overlap by
    342 px, and the only switch is at 682. Measuring at x=1024 there samples the middle of a
    tile and finds no seam, which is exactly the false negative that made the first run of
    this pilot look like there was nothing to fix.
    """
    tl = M.tiles(shape)
    # last tile wins, so the switch is at each tile's START, scanning in reverse order
    starts = sorted({(t[0] if axis == 0 else t[2]) for t in tl})
    n = shape[axis]
    return [k - 1 for k in starts if 0 < k - 1 < n - 1]


def seam_ratio(prob, axis, shape):
    d = np.abs(np.diff(prob, axis=axis))
    per = d.mean(axis=1 - axis)
    b = [k for k in switch_columns(shape, axis) if k < len(per)]
    if not b:
        return float("nan"), float("nan"), 0
    mask = np.ones(len(per), bool); mask[b] = False
    return per[b].mean(), per[mask].mean(), len(b)


def main():
    # a frame small enough to predict twice, big enough to have interior seams
    # need a frame wider than 2*TILE in both axes, or the tiles all overlap and there is no
    # interior seam to measure. Smallest such frame, to keep two full predictions affordable.
    cands = [m for m in S.list_images()
             if m.get("has_prob") and (m.get("width") or 0) > 2 * M.TILE
             and (m.get("height") or 0) > 2 * M.TILE
             and "SELFTEST" not in (m.get("filename") or "")]
    cands.sort(key=lambda x: (x.get("megapixels") or 0))
    m = cands[0]
    iid = m["id"]
    print(f"  {(m.get('filename') or '')[22:56]}  {m['width']}x{m['height']}")
    img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
    H, W = img.shape
    print(f"  tiles: {len(M.tiles((H, W)))}   real switch columns x={switch_columns((H,W),1)} "
          f"y={switch_columns((H,W),0)}")
    z = np.load(S.path(iid, "emb.npz"))
    coords, embs = z["coords"], z["emb"]
    mdl = P.get_model()
    feats = np.asarray(compute_feature_stack(img), np.float32)
    n17 = feats.shape[2]

    out = {}
    for name, fn in (("last-tile-wins (today)", lastwins_rows), ("blended", blended_rows)):
        t0 = time.time()
        prob = np.zeros((H, W), np.float32)
        for r0 in range(0, H, 96):
            r1 = min(r0 + 96, H)
            rr = np.repeat(np.arange(r0, r1), W); cc = np.tile(np.arange(W), r1 - r0)
            blk = feats[r0:r1].reshape(-1, n17)
            emb = fn(coords, embs, rr, cc)
            p17 = mdl.m17.predict_proba(blk)[:, 1]
            ph = mdl.hybrid.predict_proba(np.concatenate([blk, emb], axis=1))[:, 1]
            prob[r0:r1] = ((p17 + ph) / 2).reshape(r1 - r0, W)
        out[name] = prob
        v_b, v_e, nv = seam_ratio(prob, 1, (H, W))
        h_b, h_e, nh = seam_ratio(prob, 0, (H, W))
        print(f"\n  {name}   ({time.time()-t0:.0f}s)")
        print(f"    vertical seams   {v_b:.4f} vs {v_e:.4f} elsewhere = {v_b/v_e:5.1f}x  (n={nv})")
        print(f"    horizontal seams {h_b:.4f} vs {h_e:.4f} elsewhere = {h_b/h_e:5.1f}x  (n={nh})")
        print(f"    crack at 0.50: {(prob>0.5).mean()*100:.2f}%")
    a, b = out["last-tile-wins (today)"], out["blended"]
    agree = ((a > 0.5) == (b > 0.5)).mean()
    print(f"\n  the two masks agree on {agree*100:.3f}% of pixels")
    print(f"  mean |dp| between them: {np.abs(a-b).mean():.5f}")
    print("SEAMS_DONE")



if __name__ == "__main__":
    main()
