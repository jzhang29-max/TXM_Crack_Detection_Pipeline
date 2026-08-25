"""Per-image probability histograms, split by label class and by specimen support.

WHY HISTOGRAMS. Every question in this study is "how many pixels exceed t", for many t,
many subsets, and many fold splits. Re-reading 71 frames (up to 32 MP each) per question is
hours; a per-image histogram answers all of them exactly by summing bins, in milliseconds,
and lets a threshold be CHOSEN on one set of images and SCORED on another.

EXACTNESS. The serving path tests `prob > threshold`. Bin edges are shifted by +EPS so that
summing bins from the edge at t gives count(prob >= t+EPS) == count(prob > t) for float32
data. Bins are 0.0005 wide, so every sweep threshold at 0.01 (and the 0.20 CORRECTION_FLOOR)
lands exactly on an edge.

Read-only on app_data. Writes research/oppoint/cache/<iid>.npz only.
"""
import sys, os, json, glob, time
import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
import store as S, pipeline as P

OUT = os.path.join(P0, "research", "oppoint")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)

EPS = 1e-7
NB = 2000                                  # 0.0005-wide bins over [0,1]
EDGES = np.linspace(0.0, 1.0, NB + 1) + EPS
EDGES[0] = -1.0                            # catch exact zeros
EDGES[-1] = 2.0                            # catch exact ones


def h(vals):
    if vals.size == 0:
        return np.zeros(NB, np.int64)
    return np.histogram(vals, bins=EDGES)[0].astype(np.int64)


def gt_lookup():
    """dense GT mask keyed by unique shape match (verified 1:1 in circularity.py)."""
    out = {}
    for g in sorted(glob.glob(os.path.join(P0, "dataset_cache", "*_gt.npy"))):
        m = np.load(g, mmap_mode="r")
        out[tuple(m.shape)] = (os.path.basename(g)[:-7], g)
    return out


def main():
    inv = json.load(open(os.path.join(OUT, "inventory.json")))
    gts = gt_lookup()
    for i, r in enumerate(inv["rows"], 1):
        iid = r["iid"]
        dst = os.path.join(CACHE, iid + ".npz")
        if os.path.exists(dst):
            print(f"[{i}/71] cached {r['filename'][:50]}")
            continue
        t0 = time.time()
        prob = S.load_npy(iid, "prob.npy")
        if prob is None:
            continue
        prob = np.asarray(prob, np.float32)
        corr = S.load_npy(iid, "correction.npy")
        corr = None if corr is None or corr.shape != prob.shape else np.asarray(corr)

        raw = S.load_npy(iid, "img.npy", mmap=True)
        spec = P.specimen_support(np.asarray(raw)) if raw is not None else None
        del raw
        if spec is not None and spec.shape != prob.shape:
            spec = None

        d = dict(h_all=h(prob.ravel()), npix=np.int64(prob.size))
        if spec is not None:
            d["h_spec"] = h(prob[spec])
            d["h_off"] = h(prob[~spec])
            d["n_spec"] = np.int64(int(spec.sum()))
        if corr is not None:
            m1, m2 = corr == 1, corr == 2
            d["h_pos"], d["h_neg"] = h(prob[m1]), h(prob[m2])
            d["n_pos"], d["n_neg"] = np.int64(int(m1.sum())), np.int64(int(m2.sum()))
            if spec is not None:
                d["h_pos_spec"], d["h_neg_spec"] = h(prob[m1 & spec]), h(prob[m2 & spec])
                d["n_pos_spec"] = np.int64(int((m1 & spec).sum()))
                d["n_neg_spec"] = np.int64(int((m2 & spec).sum()))
            del m1, m2
        if tuple(prob.shape) in gts:
            stem, gp = gts[tuple(prob.shape)]
            g = np.asarray(np.load(gp)) > 0
            d["h_gt1"], d["h_gt0"] = h(prob[g]), h(prob[~g])
            d["n_gt1"] = np.int64(int(g.sum()))
            d["gt_stem"] = np.array(stem)
            if spec is not None:
                d["h_gt0_spec"] = h(prob[(~g) & spec])
                d["n_gt0_spec"] = np.int64(int(((~g) & spec).sum()))
            del g
        np.savez_compressed(dst, **d)
        print(f"[{i}/71] {time.time()-t0:5.1f}s {'GT ' if 'h_gt1' in d else '   '}"
              f"{'spec' if spec is not None else 'NOSPEC'} {r['filename'][:46]}")
        del prob, corr, spec


if __name__ == "__main__":
    main()
