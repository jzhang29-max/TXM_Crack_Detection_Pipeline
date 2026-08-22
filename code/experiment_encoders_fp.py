"""False positives on crack-free material: the axis a labelled-pixel score cannot see.

    python3 code/experiment_encoders_fp.py --arms sam1,sam2

WHY THIS DECIDES IT. experiment_encoders.py scores everything on LABELLED pixels, and on
this data HistGradientBoosting won every such metric and then marked 7.9x more crack-free
specimen as crack (docs/REFERENCE_FRAMES_AND_HGB.md). Crack-free specimen is exactly what
nobody labels. An encoder swap is held to the same standard.

TRAINED ON THE FULL CORPUS, which the first version of this script got wrong. Training only
on the four reference frames -- crops that are 18-30% crack -- produced models that marked
26-33% of crack-free specimen as crack, about a hundred times the deployed model's 0.25%.
That is a training-composition artifact, not an encoder property, and it was predictable:
the `gt-only` arm of research/fp_attribution.json hits 42% FPR on held-out groups for the
same reason. The arms were still comparable to each other, but the absolute numbers were
meaningless, so the comparison now trains on the owner's corrections across all 66 labelled
images -- the composition the deployed recipe actually uses.

MEASURED BY UNIFORM SAMPLING, not full-frame prediction. On a specimen confirmed to contain
no crack every pixel is background, so the fraction of a uniform sample predicted crack is an
unbiased estimate of the predicted area fraction -- at roughly a thousandth of the compute.
Full-frame prediction over these six mosaics is ~200 M MLP evaluations per arm and took over
three hours for two specimens; sampling 250k px each takes seconds and its standard error at
these rates is under 0.1 pp.

ONE CAVEAT, STATED BECAUSE IT MOVES THE NUMBERS. Speck pruning cannot be applied to scattered
sampled pixels, so these are UNPRUNED rates and run higher than the deployed figure, which is
measured after pruning. Both arms are unpruned identically, so the paired difference -- the
thing under test -- is unaffected.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import model as M            # noqa: E402
import pipeline as P         # noqa: E402
import store as S            # noqa: E402
import encoders as E         # noqa: E402
from experiment_encoders import mlp            # noqa: E402
from txm_features import compute_feature_stack  # noqa: E402

CRACK_PER_IMAGE = 8000
NEG_PER_IMAGE = 8000
CLEAN_SAMPLE = 250000


def enc_rows(arm, iid, img, rr, cc):
    """The [n, C] block for one arm at pixel coords rr, cc."""
    if arm == "sam1":
        p = S.path(iid, "emb.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p)
        return E.rows_at(z["coords"], z["emb"], float(M.EMB_STRIDE), rr, cc)
    c, e, stride = E.cached_embedding(arm, iid, img, progress=lambda k, n: None)
    return E.rows_at(c, e, stride, rr, cc)


def training_rows(arms, rng):
    """Paired rows from the owner's corrections across every labelled non-reference image."""
    items = [m for m in S.list_images()
             if (m.get("corrected_crack_px") or m.get("corrected_not_px"))
             and not P.is_reference_image(m.get("filename"))]
    print(f"  {len(items)} labelled images (the deployed recipe's set)")
    Xs = {a: [] for a in arms}
    ys, used, n17 = [], 0, None
    for k, m in enumerate(items, 1):
        iid = m["id"]
        corr = S.load_npy(iid, "correction.npy")
        img = S.load_npy(iid, "img.npy")
        if corr is None or img is None:
            continue
        corr = np.asarray(corr); img = np.asarray(img, np.float32)
        ci = np.flatnonzero(corr.reshape(-1) == 1)
        bi = np.flatnonzero(corr.reshape(-1) == 2)
        nc = min(CRACK_PER_IMAGE, len(ci)); nb = min(NEG_PER_IMAGE, len(bi))
        if nc + nb == 0:
            continue
        idx = np.concatenate([rng.choice(ci, nc, replace=False) if nc else ci[:0],
                              rng.choice(bi, nb, replace=False) if nb else bi[:0]])
        rr, cc = np.unravel_index(idx, corr.shape)
        feats = np.asarray(compute_feature_stack(img), np.float32)
        x17 = np.asarray(feats[rr, cc, :], np.float32)
        n17 = x17.shape[1]
        del feats
        ok = True
        blocks = {}
        for arm in arms:
            b = enc_rows(arm, iid, img, rr, cc)
            if b is None:
                ok = False; break
            blocks[arm] = np.concatenate([x17, b], axis=1)
        if ok:
            for arm in arms:
                Xs[arm].append(blocks[arm])
            ys.append(np.concatenate([np.ones(nc, bool), np.zeros(nb, bool)]))
            used += 1
        del img, corr, x17, blocks
        if k % 10 == 0 or k == len(items):
            print(f"    {k}/{len(items)} images, {sum(len(y) for y in ys):,} rows", flush=True)
    y = np.concatenate(ys)
    return {a: np.concatenate(Xs[a]) for a in arms}, y, n17, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="sam1,sam2")
    ap.add_argument("--json", default=os.path.join(PROJECT, "research", "encoder_fp.json"))
    a = ap.parse_args()
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    for arm in arms:
        if arm != "sam1":
            ok, why = E.availability(arm)
            if not ok:
                raise SystemExit(f"{arm} unavailable: {why}")

    t0 = time.time()
    print("building paired training rows from the FULL labelled corpus")
    X, y, n17, used = training_rows(arms, np.random.RandomState(0))
    print(f"  {len(y):,} rows from {used} images, {y.mean()*100:.1f}% crack "
          f"({time.time()-t0:.0f}s)")

    # the 17-feature member sees no embedding: identical in both arms, so train it once
    m17 = mlp(0).fit(X[arms[0]][:, :n17], y)
    hyb = {}
    for arm in arms:
        t1 = time.time()
        hyb[arm] = mlp(0).fit(X[arm], y)
        print(f"  {arm} hybrid trained ({time.time()-t1:.0f}s)")
        X[arm] = None
    del X, y

    imgs = sorted([m for m in S.list_images()
                   if any(k.lower() in (m.get("filename") or "").lower()
                          for k in P.CLEAN_SPECIMENS)],
                  key=lambda m: m.get("filename") or "")
    print(f"\n{len(imgs)} confirmed crack-free specimens, {CLEAN_SAMPLE:,} uniform px each")
    print(f"\n{'specimen':<34} " + " ".join(f"{arm:>10}" for arm in arms))
    out = []
    rng = np.random.RandomState(7)
    for m in imgs:
        iid = m["id"]
        img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
        H, W = img.shape
        idx = np.sort(rng.choice(H * W, min(CLEAN_SAMPLE, H * W), replace=False))
        rr, cc = np.unravel_index(idx, (H, W))
        feats = np.asarray(compute_feature_stack(img), np.float32)
        x17 = np.asarray(feats[rr, cc, :], np.float32)
        del feats
        p17 = m17.predict_proba(x17)[:, 1]
        rec = dict(image=m.get("filename"), n=int(len(idx)))
        cells = []
        for arm in arms:
            b = enc_rows(arm, iid, img, rr, cc)
            ph = hyb[arm].predict_proba(np.concatenate([x17, b], axis=1))[:, 1]
            frac = float((((p17 + ph) / 2.0) > 0.5).mean())
            rec[arm] = round(frac, 6)
            cells.append(f"{frac*100:8.3f}%")
            del b, ph
        out.append(rec)
        print(f"{(m.get('filename') or '')[22:56]:<34} " +
              " ".join(f"{c:>10}" for c in cells), flush=True)
        del img, x17, p17

    print(f"\n{'arm':<8} {'mean FP area (unpruned)':>26} {'worst':>10}")
    for arm in arms:
        v = [r[arm] for r in out]
        print(f"{arm:<8} {np.mean(v)*100:>25.3f}% {np.max(v)*100:>9.3f}%")
    if len(arms) == 2:
        d = np.array([r[arms[1]] - r[arms[0]] for r in out])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"\npaired per-specimen difference {arms[1]} - {arms[0]}:")
        print(f"  mean {d.mean()*100:+.4f} pp   sd {d.std(ddof=1)*100:.4f} pp   "
              f"SE {se*100:.4f} pp   n={len(d)}")
        print(f"  t = {d.mean()/se if se else float('nan'):.2f}")
        print(f"  the retrain gate's tolerance is {P.FP_TOL*100:.1f} pp -- this difference is "
              f"{'INSIDE' if abs(d.mean()) <= P.FP_TOL else 'OUTSIDE'} it")
    json.dump(dict(per_specimen=out, rows=int(len(y)) if False else None,
                   note="unpruned, uniform-sampled; paired"), open(a.json, "w"), indent=1)
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
