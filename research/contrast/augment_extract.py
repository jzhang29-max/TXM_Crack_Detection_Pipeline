"""Pass 1: cache the 37-column feature matrix for the labelled rows and the guardrail rows.

Every arm in augment_eval.py is a COLUMN SUBSET of what this writes, which is the only way
baseline_17 and the augmented arms are guaranteed self-comparable: identical rows,
identical sampling seed, identical filter code.

Writes only to the scratchpad cache directory given by --cache (default under /private/tmp).
Reads app_data through app/core/store.py. Touches nothing under app/, code/, or models/.

Usage:
    .venv/bin/python research/contrast/augment_extract.py --workers 5
"""

import argparse
import os
import sys
import time

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
sys.path.insert(0, os.path.join(P0, "research", "contrast"))

N_PER_CLASS = 8000
N_CLEAN_SAMPLE = 200_000
THIN_MAX_HALF_WIDTH = 3.0


def labelled_ids():
    import store as S
    out = []
    for m in S.list_images():
        c = S.load_npy(m["id"], "correction.npy", mmap=True)
        if c is None:
            continue
        a = np.asarray(c)
        if (a == 1).any() and (a == 2).any():
            out.append((m["id"], m.get("filename") or ""))
        del a, c
    return out


def clean_ids():
    import store as S
    import pipeline as P
    out = []
    for m in S.list_images():
        fn = m.get("filename") or ""
        if any(k.lower() in fn.lower() for k in P.CLEAN_SPECIMENS):
            out.append((m["id"], fn))
    return out


def do_labelled(args):
    iid, fn, cache = args
    out_p = os.path.join(cache, "lab__" + iid + ".npz")
    if os.path.exists(out_p):
        return iid, "cached", 0.0
    import store as S
    import augment_features as AF

    t0 = time.time()
    img = np.asarray(S.load_npy(iid, "img.npy")).astype(np.float32)
    corr = np.asarray(S.load_npy(iid, "correction.npy"))

    # Protocol: one RandomState(0) per image, crack pixels drawn first then not-crack.
    rng = np.random.RandomState(0)
    f1 = np.flatnonzero(corr.ravel() == 1)
    f2 = np.flatnonzero(corr.ravel() == 2)
    s1 = rng.choice(f1, min(N_PER_CLASS, len(f1)), replace=False)
    s2 = rng.choice(f2, min(N_PER_CLASS, len(f2)), replace=False)
    idx = np.concatenate([s1, s2])
    y = np.concatenate([np.ones(len(s1), np.uint8), np.zeros(len(s2), np.uint8)])

    X = AF.sample_rows(img, idx)
    hw, n_dark = AF.thin_half_width(img, corr)

    np.savez_compressed(out_p, X=X, y=y, idx=idx,
                        shape=np.array(img.shape), filename=np.array(fn),
                        half_width=np.array(-1.0 if hw is None else hw),
                        n_dark=np.array(n_dark),
                        n_crack_px=np.array(len(f1)), n_not_px=np.array(len(f2)))
    return iid, "ok hw=%s" % (None if hw is None else round(hw, 2)), time.time() - t0


def do_clean(args):
    iid, fn, cache = args
    out_p = os.path.join(cache, "clean__" + iid + ".npz")
    if os.path.exists(out_p):
        return iid, "cached", 0.0
    import store as S
    import pipeline as P
    import augment_features as AF

    t0 = time.time()
    img = np.asarray(S.load_npy(iid, "img.npy")).astype(np.float32)
    rng = np.random.RandomState(0)
    idx = rng.choice(img.size, min(N_CLEAN_SAMPLE, img.size), replace=False)
    X = AF.sample_rows(img, idx)
    # Secondary guardrail axis: 20-40% of these frames is off-specimen background sitting
    # near zero. A false positive there and one on the metal are not the same claim, so
    # the on-specimen subset is recorded alongside the whole-frame sample.
    sup = P.specimen_support(img)
    on_spec = sup.ravel()[idx]
    np.savez_compressed(out_p, X=X, idx=idx, on_spec=on_spec,
                        shape=np.array(img.shape), filename=np.array(fn))
    return iid, "ok on_spec=%.3f" % float(on_spec.mean()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        "/private/tmp/claude-501/-Users-jiamingzhang-Desktop-APP",
        "6c65cf52-47c9-4de6-a996-f3251ee258ff", "scratchpad", "featcache"))
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)

    lab = labelled_ids()
    cln = clean_ids()
    print("labelled frames (both classes): %d | crack-free specimens: %d" % (len(lab), len(cln)),
          flush=True)

    jobs = [(do_clean, (i, f, a.cache)) for i, f in cln] + \
           [(do_labelled, (i, f, a.cache)) for i, f in lab]

    t0 = time.time()
    if a.workers <= 1:
        for fn, arg in jobs:
            iid, msg, dt = fn(arg)
            print("  %-6.1fs %s %s" % (dt, msg, iid[:60]), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(fn, arg) for fn, arg in jobs]
            for k, fu in enumerate(futs, 1):
                iid, msg, dt = fu.result()
                print("  [%d/%d] %-6.1fs %s %s" % (k, len(futs), dt, msg, iid[:60]), flush=True)
    print("extract wall clock %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
