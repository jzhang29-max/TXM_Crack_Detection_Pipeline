#!/usr/bin/env python3
"""Blind crop set for the small isolated indications -- matched on SIZE *and* SPECIMEN.

WHY A SECOND VERSION. Run 1 stratified the positive control on size only (median 2418 vs
2403 px, p = 0.94) and left specimen mix unmatched, which turned out to be the whole result.
POSS came out 62% AM; TEST is 41% Wrought. AM is the group this project documents as
intensity-invisible (Cohen d +0.09), i.e. the HARDEST group to see, so loading the control
with it and the test class with Wrought makes the control look... and here the direction is
the opposite of naive intuition: direct standardisation of POSS to the TEST group mix drops
the control yes-rate from 0.562 to 0.313, essentially equal to TEST's 0.344.

So the reported "sensitivity 56% at this size against 92% at full size" is not a property of
size at all, and the Rogan-Gladen prevalence built on it is void. This run matches on both
axes so the comparison means something.

Emits crops + out/txm_small_component_cropkey_v2.json.

Usage:  python3 code/audit/build_small_component_crops.py <outdir> <keyfile> <smallsel.json> <smallcomps.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
os.chdir(HERE)

from app.core import pipeline as P                                    # noqa: E402
from app.core import store as S                                       # noqa: E402
from scipy.ndimage import label as cclabel, find_objects              # noqa: E402
from PIL import Image                                                 # noqa: E402

WIN = 400
CRACKFREE = ("B2_2_1_lbf", "B2_2_9_lbf", "B2_amb_mosaic_2", "b3_amb",
             "HC_316L_fatigue_200_cycles", "HC_316L_fatigue_400_cycles",
             "b3_3_0lbf_268_13um", "b3_3_18lbf_348_13um",
             "wrought_316L_fatigue_0_cycles", "wrought_316L_fatigue_300_cycles")


def is_crackfree(iid):
    return any(k in iid for k in CRACKFREE)


def group_of(iid):
    low = iid.lower()
    if "_b2_" in low:
        return "B2"
    if "b3" in low:
        return "B3"
    if "hc_316l" in low:
        return "AM"
    if "wrought" in low:
        return "WR"
    return "other"


def main(outdir, keyfile, selfile, compfile):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(20260921)
    sel = json.load(open(selfile))
    small = json.load(open(compfile))
    items = [dict(cls="TEST", iid=c["iid"], y=c["y"], x=c["x"], px=c["px"],
                  grp=group_of(c["iid"]), sig=c.get("sigma_n"), el=c.get("elong"))
             for c in sel]
    items += [dict(cls="CFREE", iid=c["iid"], y=c["y"], x=c["x"], px=c["px"],
                   grp=group_of(c["iid"]), sig=c.get("sigma_n"), el=c.get("elong"))
              for c in small if is_crackfree(c["iid"]) and c["px"] >= 200]

    pool, matrix = [], []
    for m in S.list_images():
        iid = m["id"]
        if is_crackfree(iid):
            continue
        try:
            mask = P.effective_mask(iid, corrections="none", tight=True)
        except Exception:
            continue
        corr = S.load_npy(iid, "correction.npy")
        if corr is None or corr.shape != mask.shape or not mask.any():
            continue
        lab, n = cclabel(mask & (corr == 1))
        if n:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            cand = np.flatnonzero((sizes >= 200) & (sizes <= 4000))
            objs = find_objects(lab)
            for c in cand:
                sl = objs[c - 1]
                ys, xs = np.nonzero(lab[sl] == c)
                pool.append(dict(cls="POSS", iid=iid, y=int(sl[0].start + ys.mean()),
                                 x=int(sl[1].start + xs.mean()), px=int(sizes[c]),
                                 grp=group_of(iid), sig=None, el=None))
        plain = (~mask) & (corr == 2)
        if plain.any():
            ys, xs = np.nonzero(plain[::41, ::41])
            for k in rng.choice(ys.size, min(2, ys.size), replace=False) if ys.size else []:
                matrix.append(dict(cls="NEGM", iid=iid, y=int(ys[k] * 41), x=int(xs[k] * 41),
                                   px=-1, grp=group_of(iid), sig=None, el=None))
        del mask, corr

    # MATCH WITHIN GROUP, THEN ON SIZE. For each TEST component take the nearest-size unused
    # positive-control component FROM THE SAME SPECIMEN GROUP. Where a group cannot supply
    # one, record the shortfall rather than silently substituting another group -- an
    # unmatched substitution is exactly what invalidated run 1.
    used, matched, short = set(), [], []
    for t in sorted(sel, key=lambda c: -c["px"]):
        g = group_of(t["iid"])
        cands = [i for i, p in enumerate(pool) if i not in used and p["grp"] == g]
        if not cands:
            short.append(dict(group=g, px=t["px"], iid=t["iid"]))
            continue
        best = min(cands, key=lambda i: abs(pool[i]["px"] - t["px"]))
        used.add(best)
        matched.append(pool[best])
    items += matched
    items += [matrix[i] for i in rng.choice(len(matrix), min(26, len(matrix)), replace=False)]

    from collections import Counter
    tg = Counter(group_of(c["iid"]) for c in sel)
    pg = Counter(p["grp"] for p in matched)
    print("group mix  TEST:", dict(tg))
    print("group mix  POSS:", dict(pg), " matched" if tg == pg else " MISMATCH")
    if short:
        print(f"SHORTFALL: {len(short)} TEST components had no same-group control:", short)

    order = rng.permutation(len(items))
    key, cache = [], {}
    for newid, i in enumerate(order, 1):
        it = items[i]
        iid = it["iid"]
        if iid not in cache:
            d = S.load_npy(iid, "display.npy")
            if d is None:
                d = S.load_npy(iid, "img.npy")
            d = np.asarray(d, np.float32)
            lo, hi = np.percentile(d, (0.5, 99.5))
            cache.clear()
            cache[iid] = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            del d
        g = cache[iid]
        h = WIN // 2
        y0 = max(0, min(it["y"] - h, g.shape[0] - WIN))
        x0 = max(0, min(it["x"] - h, g.shape[1] - WIN))
        crop = g[y0:y0 + WIN, x0:x0 + WIN]
        if crop.shape != (WIN, WIN):
            pad = np.ones((WIN, WIN), np.float32)
            pad[:crop.shape[0], :crop.shape[1]] = crop
            crop = pad
        name = f"site_{newid:03d}.png"
        Image.fromarray((crop * 255).astype(np.uint8)).save(os.path.join(outdir, name))
        key.append(dict(field=name, **it))
    json.dump(dict(group_mix_test=dict(tg), group_mix_control=dict(pg),
                   shortfall=short, crops=key), open(keyfile, "w"), indent=1)
    print(Counter(k["cls"] for k in key), "total", len(key))


if __name__ == "__main__":
    main(*sys.argv[1:5])
