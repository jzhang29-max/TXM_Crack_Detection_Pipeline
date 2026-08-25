"""Pass 1: cache the 34-column feature matrix for the labelled rows and the guardrail rows.

Every arm in ridge_eval.py is a COLUMN SUBSET of what this writes. That is the only way
baseline_17 and the ridge arms are guaranteed self-comparable: identical rows, identical
sampling seed, identical filter code, one fit protocol.

Writes only to the scratchpad cache directory given by --cache. Reads app_data through
app/core/store.py. Touches nothing under app/, code/, models/ or app_data/.

Usage:
    .venv/bin/python research/ridge/ridge_extract.py --workers 5
"""

import argparse
import os
import sys
import time

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
sys.path.insert(0, os.path.join(P0, "research", "ridge"))

# Protocol constants, held fixed from docs/CONTRAST.md so the two sweeps are comparable.
N_PER_CLASS = 8000
N_CLEAN_SAMPLE = 200_000


def selftest():
    """The cached matrix must equal a straight full-frame computation at the sampled rows.

    Worth the 3 seconds: sample_rows() indexes 17 separately-computed filter planes into
    one output array by hand, and an off-by-one in that column cursor would put frangi's
    values in sato's column and silently produce a plausible-looking, wrong arm table.
    """
    import ridge_features as RF
    from txm_features import compute_feature_stack
    from skimage.filters import frangi, sato, meijering

    rng = np.random.RandomState(0)
    img = rng.rand(200, 210).astype(np.float32)
    img[80:83, 30:170] = 0.05                       # a dark line, so ridges are non-trivial
    idx = rng.choice(img.size, 500, replace=False)
    rr, cc = np.unravel_index(idx, img.shape)

    X = RF.sample_rows(img, idx)
    assert X.shape == (500, RF.N_COLS), X.shape

    ok = True
    ref = compute_feature_stack(img)[rr, cc, :]
    if not np.array_equal(X[:, RF.COL_BASE17], ref):
        ok = False
        print("  FAIL base17 columns differ", flush=True)

    checks = [
        (RF.COL_FRANGI[0], frangi(img, sigmas=RF.SIGMAS_FINE, black_ridges=True)),
        (RF.COL_FRANGI[1], frangi(img, sigmas=RF.SIGMAS_COARSE, black_ridges=True)),
        (RF.COL_SATO[0], sato(img, sigmas=RF.SIGMAS_FINE, black_ridges=True)),
        (RF.COL_SATO[1], sato(img, sigmas=RF.SIGMAS_COARSE, black_ridges=True)),
        (RF.COL_MEIJ[0], meijering(img, sigmas=RF.SIGMAS_FINE, black_ridges=True)),
        (RF.COL_MEIJ[1], meijering(img, sigmas=RF.SIGMAS_COARSE, black_ridges=True)),
        (RF.COL_FRANGI_FIX[0], frangi(img, sigmas=RF.SIGMAS_FINE, black_ridges=True,
                                      gamma=RF.GAMMA_FIXED)),
        (RF.COL_FRANGI_FIX[1], frangi(img, sigmas=RF.SIGMAS_COARSE, black_ridges=True,
                                      gamma=RF.GAMMA_FIXED)),
    ]
    for col, plane in checks:
        if not np.allclose(X[:, col], np.asarray(plane, np.float32)[rr, cc], atol=0):
            ok = False
            print("  FAIL column %d (%s)" % (col, RF.ALL_NAMES[col]), flush=True)
    for j, s in enumerate(RF.HESS_SIGMAS):
        lmaj, lmin, ratio = RF.hessian_eigs(img, s)
        for k, plane in enumerate((lmaj, lmin, ratio)):
            col = RF.COL_HESS[3 * j + k]
            if not np.array_equal(X[:, col], plane[rr, cc]):
                ok = False
                print("  FAIL column %d (%s)" % (col, RF.ALL_NAMES[col]), flush=True)

    # The fixed-gamma arm must actually DIFFER from the adaptive one, or it measures nothing.
    d = float(np.abs(X[:, RF.COL_FRANGI[0]] - X[:, RF.COL_FRANGI_FIX[0]]).max())
    print("  selftest: %s | frangi auto vs fixed-gamma max|diff| = %.5f%s"
          % ("34 columns match full-frame computation" if ok else "COLUMN MISMATCH", d,
             "  (WARNING: identical, fixed-gamma arm is a no-op)" if d == 0 else ""),
          flush=True)
    # Sanity on the sign convention the docstring claims.
    dark = RF.hessian_eigs(img, 2)[0][81, 100]
    print("  selftest: on-crack hess_lmaj_s2 = %+.5f (should be > 0 for a DARK ridge)"
          % dark, flush=True)
    if not ok:
        raise SystemExit("selftest failed -- refusing to build a cache that is wrong")
    return True


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
        return iid, "cached", 0.0, 0
    import store as S
    import ridge_features as RF

    t0 = time.time()
    img = np.asarray(S.load_npy(iid, "img.npy")).astype(np.float32)
    corr = np.asarray(S.load_npy(iid, "correction.npy"))

    # Protocol, held identical to the contrast sweep: one RandomState(0) per image, crack
    # pixels drawn first then not-crack. Same seed and same order => same rows, so a
    # difference between the two experiments is the features and nothing else.
    rng = np.random.RandomState(0)
    f1 = np.flatnonzero(corr.ravel() == 1)
    f2 = np.flatnonzero(corr.ravel() == 2)
    s1 = rng.choice(f1, min(N_PER_CLASS, len(f1)), replace=False)
    s2 = rng.choice(f2, min(N_PER_CLASS, len(f2)), replace=False)
    idx = np.concatenate([s1, s2])
    y = np.concatenate([np.ones(len(s1), np.uint8), np.zeros(len(s2), np.uint8)])

    X = RF.sample_rows(img, idx)
    hw, n_dark = RF.thin_half_width(img, corr)

    np.savez_compressed(out_p, X=X, y=y, idx=idx,
                        shape=np.array(img.shape), filename=np.array(fn),
                        half_width=np.array(-1.0 if hw is None else hw),
                        n_dark=np.array(n_dark),
                        n_crack_px=np.array(len(f1)), n_not_px=np.array(len(f2)))
    return iid, "ok hw=%s" % (None if hw is None else round(hw, 2)), time.time() - t0, img.size


def do_clean(args):
    iid, fn, cache = args
    out_p = os.path.join(cache, "clean__" + iid + ".npz")
    if os.path.exists(out_p):
        return iid, "cached", 0.0, 0
    import store as S
    import pipeline as P
    import ridge_features as RF

    t0 = time.time()
    img = np.asarray(S.load_npy(iid, "img.npy")).astype(np.float32)
    rng = np.random.RandomState(0)
    idx = rng.choice(img.size, min(N_CLEAN_SAMPLE, img.size), replace=False)
    X = RF.sample_rows(img, idx)
    # THE DECISIVE COLUMN. 20-40% of these frames is off-specimen background sitting near
    # zero, and the model calls dark background crack, so whole-frame FP is dominated by
    # area that is not the specimen -- 21.9% against 2.2% on-specimen for the baseline. An
    # experiment run earlier today INVERTED its conclusion on exactly this: one arm looked
    # like a 4x win whole-frame and was 2.9x worse on the metal. Both are recorded; only
    # the on-specimen one is read as the answer.
    sup = P.specimen_support(img)
    np.savez_compressed(out_p, X=X, idx=idx, on_spec=sup.ravel()[idx],
                        shape=np.array(img.shape), filename=np.array(fn))
    return iid, "ok on_spec=%.3f" % float(sup.ravel()[idx].mean()), time.time() - t0, img.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        "/private/tmp/claude-501/-Users-jiamingzhang-Desktop-APP",
        "6c65cf52-47c9-4de6-a996-f3251ee258ff", "scratchpad", "ridgecache"))
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)

    import ridge_features as RF
    print("columns %d: %s" % (RF.N_COLS, ", ".join(RF.ALL_NAMES[RF.N_COLS - 17:])), flush=True)
    selftest()

    lab = labelled_ids()
    cln = clean_ids()
    print("labelled frames (both classes): %d | crack-free specimens: %d"
          % (len(lab), len(cln)), flush=True)

    # Biggest frames first: with a fixed worker pool the long tail is what sets wall clock.
    import store as S
    def px(iid):
        im = S.load_npy(iid, "img.npy", mmap=True)
        n = 0 if im is None else int(np.asarray(im).size)
        del im
        return n
    jobs = [(do_clean, (i, f, a.cache), px(i)) for i, f in cln] + \
           [(do_labelled, (i, f, a.cache), px(i)) for i, f in lab]
    jobs.sort(key=lambda j: -j[2])
    print("total %.0f Mpx to filter" % (sum(j[2] for j in jobs) / 1e6), flush=True)

    t0 = time.time()
    done_px = 0
    if a.workers <= 1:
        for fn, arg, _ in jobs:
            iid, msg, dt, n = fn(arg)
            print("  %-7.1fs %s %s" % (dt, msg, iid[:55]), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [(ex.submit(fn, arg), n) for fn, arg, n in jobs]
            for k, (fu, n) in enumerate(futs, 1):
                iid, msg, dt, npx = fu.result()
                done_px += n
                print("  [%2d/%d] %5.1fMpx %-7.1fs %-22s %s"
                      % (k, len(futs), n / 1e6, dt, msg, iid[:52]), flush=True)
    print("extract wall clock %.1f s (%.1f Mpx)" % (time.time() - t0, done_px / 1e6),
          flush=True)


if __name__ == "__main__":
    main()
