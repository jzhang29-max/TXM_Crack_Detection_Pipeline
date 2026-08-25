"""Extract, ONCE, every training/eval row any arm of the thin-label experiment needs.

The arms differ only in which pixels are eligible for which class, so the 273 features
(17 from compute_feature_stack + 256 SAM channels) are computed a single time per image
and each row is tagged with the disjoint pool it came from. Four pools per image:

    S1  corr==1 AND inside the dark core      (tighten_to_image on the painted stroke)
    S2  corr==1, outside the core, within 3px of it   ("inner ring")
    S3  corr==1, outside the core, further than 3px   ("outer ring")
    S4  corr==2                                        (painted not-crack)

Every arm's crack/not-crack pool is a union of these four, drawn in the true area
proportions of the sets they stand for, so one extraction serves all arms AND a single
fixed evaluation set. Nothing is written outside research/thinlabels/; correction.npy is
read only.
"""
import json
import os
import sys
import time
import warnings

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
import store as S          # noqa: E402
import pipeline as P       # noqa: E402
import model as M          # noqa: E402
from txm_features import compute_feature_stack   # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

OUT = os.path.join(P0, "research", "thinlabels")
PER_POOL = 8000            # max rows sampled from each of the four pools, per image
RING_DILATE = 3


def main():
    metas = [m for m in S.list_images()
             if (m.get("corrected_crack_px") or 0) or (m.get("corrected_not_px") or 0)]
    from skimage.morphology import binary_dilation, disk
    sel = disk(RING_DILATE)

    blocks, pools, grps = [], [], []
    info = []
    t00 = time.time()
    for gi, m in enumerate(metas):
        iid = m["id"]
        t0 = time.time()
        corr = S.load_npy(iid, "correction.npy")
        img = S.load_npy(iid, "img.npy")
        if corr is None or img is None:
            print("skip (missing arrays)", iid, flush=True)
            continue
        img = np.asarray(img, np.float32)
        crack = corr == 1
        notc = corr == 2

        if crack.any():
            core = P.tighten_to_image(iid, crack)
            declined = bool(core.sum() == crack.sum() and np.array_equal(core, crack))
        else:
            core = np.zeros_like(crack)
            declined = False
        ring = crack & ~core
        inner = ring & binary_dilation(core, sel) if ring.any() and core.any() else \
            np.zeros_like(crack)
        outer = ring & ~inner

        sets = [core & crack, inner, outer, notc]
        counts = [int(s.sum()) for s in sets]
        rng = np.random.RandomState(0)
        idx_by_pool, rr_all, cc_all, pool_all = [], [], [], []
        for pi, s in enumerate(sets, 1):
            n = counts[pi - 1]
            if n == 0:
                idx_by_pool.append(np.empty(0, np.int64))
                continue
            flat = np.flatnonzero(s.reshape(-1))
            take = min(PER_POOL, n)
            pick = rng.choice(flat, take, replace=False) if take < n else flat
            idx_by_pool.append(pick)
            r, c = np.unravel_index(pick, corr.shape)
            rr_all.append(r); cc_all.append(c)
            pool_all.append(np.full(take, pi, np.int8))
        del sets, core, ring, inner, outer, crack, notc, corr
        if not rr_all:
            print("skip (no labelled pixels)", iid, flush=True)
            continue
        rr = np.concatenate(rr_all); cc = np.concatenate(cc_all)
        pool = np.concatenate(pool_all)

        f17 = compute_feature_stack(img)
        a = np.asarray(f17[rr, cc, :], np.float32)
        del f17, img
        got = M.read_emb(S.path(iid, "emb.npz"))
        if got is None:
            print("skip (no embedding)", iid, flush=True)
            continue
        coords, embs = got
        b = M.emb_rows(coords, embs, rr, cc)
        del coords, embs
        blocks.append(np.concatenate([a, b], axis=1))
        pools.append(pool)
        grps.append(np.full(len(pool), gi, np.int32))
        del a, b
        info.append(dict(gi=gi, id=iid, filename=m.get("filename"),
                         n_core=counts[0], n_inner=counts[1], n_outer=counts[2],
                         n_notcrack=counts[3],
                         n_crack_painted=counts[0] + counts[1] + counts[2],
                         tighten_declined=declined,
                         sampled=[int(len(x)) for x in idx_by_pool]))
        print(f"[{gi+1}/{len(metas)}] {iid[:52]:52s} core={counts[0]:8d} "
              f"inner={counts[1]:7d} outer={counts[2]:8d} neg={counts[3]:9d} "
              f"decl={int(declined)} {time.time()-t0:5.1f}s", flush=True)

    n_rows = sum(b.shape[0] for b in blocks)
    X = np.empty((n_rows, blocks[0].shape[1]), np.float32)
    pool = np.empty(n_rows, np.int8)
    grp = np.empty(n_rows, np.int32)
    at = 0
    for i in range(len(blocks)):
        n = blocks[i].shape[0]
        X[at:at + n] = blocks[i]; pool[at:at + n] = pools[i]; grp[at:at + n] = grps[i]
        blocks[i] = None; pools[i] = None; grps[i] = None
        at += n
    assert at == n_rows
    np.save(os.path.join(OUT, "X.npy"), X)
    np.save(os.path.join(OUT, "pool.npy"), pool)
    np.save(os.path.join(OUT, "grp.npy"), grp)
    with open(os.path.join(OUT, "rows_meta.json"), "w") as f:
        json.dump(dict(per_pool=PER_POOL, ring_dilate=RING_DILATE, n_rows=int(n_rows),
                       n_features=int(X.shape[1]), images=info,
                       seconds=round(time.time() - t00, 1)), f, indent=1)
    print("rows", n_rows, "features", X.shape[1], "in",
          round(time.time() - t00, 1), "s", flush=True)


if __name__ == "__main__":
    main()
