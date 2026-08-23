"""Validate the only sound seam fix on two frames before paying for all 71.

tiles() steps by exactly TILE and clamps only the final tile inward, so almost every interior
tile boundary has ZERO overlap: a 3914 px frame sits at x = 0, 1024, 2048, 2890. With no
overlap there is no real data spanning a boundary, which is why every lookup-time trick fails
-- a window that vanishes at each tile's edge reduces to last-wins where only one tile covers,
and a window that reaches past the edge must extrapolate, which invents crack (false
positives rose 6.2x on three crack-free specimens).

Re-embedding at a stride below TILE creates real overlap, and a window over each tile's own
extent then blends real embeddings into a continuous field. This embeds two frames at a
chosen stride and measures whether the seam actually goes away and whether the crack-free
false-positive rate holds -- roughly ten minutes, against three and a half hours for the full
corpus plus a retrain.
"""
import os, sys, time
import numpy as np
PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, os.path.join(PROJECT, "code"))
sys.path.insert(0, os.path.join(PROJECT, "research", "code"))
import model as M, store as S, pipeline as P
from txm_features import compute_feature_stack
from pilot_seams import lastwins_rows
from pilot_seam_hann import worst_lines

STRIDE = int(os.environ.get("STRIDE", "896"))
CACHE = os.environ.get("EMB_CACHE",
    "/private/tmp/claude-501/-Users-jiamingzhang-Desktop-APP/6c65cf52-47c9-4de6-a996-f3251ee258ff/scratchpad")

def tiles_stride(shape, stride, size=M.TILE):
    H, W = shape[:2]
    ys = sorted({max(min(y0+size, H)-size, 0) for y0 in range(0, max(H-1, 1), stride)})
    xs = sorted({max(min(x0+size, W)-size, 0) for x0 in range(0, max(W-1, 1), stride)})
    return [(y, x) for y in ys for x in xs]

def embed_stride(img01, stride, progress=None):
    """embed_image() with a caller-chosen stride so tiles genuinely overlap."""
    proc, model, dev, torch = M._get_sam()
    tl = tiles_stride(img01.shape, stride)
    coords, embs = [], []
    for k, (y0, x0) in enumerate(tl):
        crop = img01[y0:y0+M.TILE, x0:x0+M.TILE]
        if crop.shape != (M.TILE, M.TILE):
            crop = np.pad(crop, ((0, M.TILE-crop.shape[0]), (0, M.TILE-crop.shape[1])),
                          mode="reflect")
        u8 = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
        inp = proc(np.stack([u8]*3, -1), return_tensors="pt")
        px = inp["pixel_values"]
        px = (px.float() if px.dtype == torch.float64 else px).to(dev)
        with torch.no_grad():
            e = model.get_image_embeddings(px)
        embs.append(e.float().cpu().numpy()[0].astype(np.float16))
        coords.append((y0, x0))
        if dev == "mps":
            torch.mps.empty_cache()
        if progress:
            progress(k+1, len(tl))
    return np.asarray(coords, np.int32), np.stack(embs)

def cached_emb(iid, stride):
    p = os.path.join(CACHE, f"emb_{iid}_s{stride}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["coords"], z["emb"]
    img01 = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
    t0 = time.time()
    coords, embs = embed_stride(img01, stride,
        progress=lambda k, n: (print(f"      tile {k}/{n}", flush=True) if k % 8 == 0 else None))
    print(f"      embedded {len(coords)} tiles in {time.time()-t0:.0f}s", flush=True)
    np.savez(p[:-4], coords=coords, emb=embs)
    return coords, embs

def hann_own_extent(coords, embs, rr, cc, eps=1e-3):
    """Weighted average over tiles containing the pixel, weight ~0 at each tile's own edge.

    With real overlap this is a partition of unity over real data: continuous across every
    boundary, each tile's interior dominated by its own embedding, nothing read from outside
    a tile.
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
        wy = 0.5*(1.0-np.cos(2*np.pi*(ly[sel]+0.5)/M.TILE))
        wx = 0.5*(1.0-np.cos(2*np.pi*(lx[sel]+0.5)/M.TILE))
        w = ((wy*wx)*(1.0-eps)+eps).astype(np.float32)
        acc[sel] += M.interp_tile(embs[t], ly[sel], lx[sel]) * w[:, None]
        wsum[sel] += w
    return acc / np.maximum(wsum, 1e-12)[:, None]

def predict_with(iid, coords, embs, fn, mdl):
    img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
    H, W = img.shape
    feats = np.asarray(compute_feature_stack(img), np.float32)
    n17 = feats.shape[2]
    prob = np.zeros((H, W), np.float32)
    for r0 in range(0, H, 96):
        r1 = min(r0+96, H)
        rr = np.repeat(np.arange(r0, r1), W); cc = np.tile(np.arange(W), r1-r0)
        blk = feats[r0:r1].reshape(-1, n17)
        emb = fn(coords, embs, rr, cc)
        p17 = mdl.m17.predict_proba(blk)[:, 1]
        ph = mdl.hybrid.predict_proba(np.concatenate([blk, emb], axis=1))[:, 1]
        prob[r0:r1] = ((p17+ph)/2).reshape(r1-r0, W)
    return prob

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
    cf = clean[0]
    print(f"  stride {STRIDE} (overlap {M.TILE-STRIDE} px)")
    print(f"  seam frame  {(sf.get('filename') or '')[22:50]}")
    print(f"  crack-free  {(cf.get('filename') or '')[22:50]}\n", flush=True)

    for m, kind in ((sf, "seam frame"), (cf, "crack-free")):
        H, W = m.get("height"), m.get("width")
        print(f"  {kind}: {W}x{H}", flush=True)
        z = np.load(S.path(m["id"], "emb.npz"))
        c0, e0 = z["coords"], z["emb"]
        print(f"    stride 1024: {len(c0)} tiles", flush=True)
        print(f"    stride {STRIDE}: embedding...", flush=True)
        c1, e1 = cached_emb(m["id"], STRIDE)
        p_old = predict_with(m["id"], c0, e0, lastwins_rows, mdl)
        p_new = predict_with(m["id"], c1, e1, hann_own_extent, mdl)
        for lbl, p in (("last-wins @1024", p_old), (f"hann @{STRIDE}", p_new)):
            cols, _ = worst_lines(p, 1); rows, _ = worst_lines(p, 0)
            area = P.prune_specks(p > 0.5).mean()*100
            print(f"    {lbl:<16} worst col {cols[0][0]} {cols[0][2]:.1f}x   "
                  f"worst row {rows[0][0]} {rows[0][2]:.1f}x   area {area:.3f}%", flush=True)
        print(f"    masks agree on {(( p_old>0.5)==(p_new>0.5)).mean()*100:.3f}% of pixels\n",
              flush=True)
    print("OVERLAP_DONE")

if __name__ == "__main__":
    main()
