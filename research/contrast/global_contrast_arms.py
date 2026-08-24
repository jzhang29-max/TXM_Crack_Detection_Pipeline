"""GLOBAL contrast-adjustment arms for the TXM crack pixel classifier.

Question: does adjusting the CONTRAST of the model input (img.npy) help the
17-feature pixel classifier find THIN and FAINT cracks?

This file is the GLOBAL (whole-image, one transfer function per frame) arm.

Two facts about this dataset that frame the whole experiment:

1. img.npy is ALREADY a 1st-99th percentile stretch of the raw frame
   (pipeline.py: `img01 = robust_normalize(raw, 1.0, 99.0)`), so exactly 1% of
   every frame sits at 0.0 and 1% at 1.0. Re-applying a 1-99 stretch is
   therefore a numerical no-op, and it is kept here as a CONTROL: if it does
   not reproduce `identity` to the digit, the harness is broken.

2. Any AFFINE map x -> a*x + b (a > 0) is absorbed exactly by StandardScaler.
   Every one of the 17 features is itself affine in the corresponding feature
   of the untransformed image -- intensity and the 6 Gaussian smooths pick up
   (a, b), the 4 gradient magnitudes / 4 Laplacians / 2 local-stds pick up
   (a, 0) because they annihilate constants -- and StandardScaler standardises
   each feature column independently. So for an affine arm an UNCHANGED score
   is the correct answer, not a bug. Only the non-linear part of a transform
   (gamma, equalisation, sigmoid, and the saturation from clipping) can move
   the model at all.

Prior result this must be read against (docs/MARKUP_GUIDE.md): flat-fielding
the model input cost 0.169 IoU, because large-radius intensity features carry
~41% of the model's importance and flat-fielding destroys them. Transforms
that flatten or saturate absolute intensity are expected to hurt for the same
reason.

Protocol (fixed across all arms of this study, do not edit without re-running
every arm): every frame with a correction.npy holding both classes; up to 8000
crack + 8000 not-crack pixels per frame under RandomState(0); transform is
applied to img01 FIRST, then the 17-feature stack; StandardScaler + MLP
(64,32); GroupKFold(5) grouped by frame; IoU/precision/recall at p>0.5;
repeated on held-out rows from THIN frames only; plus a false-positive
guardrail on the 6 confirmed crack-free specimens.

Usage:  .venv/bin/python research/contrast/global_contrast_arms.py [--workers N]
Outputs: research/contrast/global_results.json
         research/contrast/global_SUMMARY.md
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))

import store as S  # noqa: E402
import pipeline as P  # noqa: E402
from txm_features import (  # noqa: E402
    FEATURE_NAMES, N_FEATURES, SMOOTH_SIGMAS, GRADIENT_SIGMAS,
    LAPLACIAN_SIGMAS, TEXTURE_SIGMAS, compute_feature_stack, local_std,
    robust_normalize,
)
from scipy import ndimage as ndi  # noqa: E402

OUT_DIR = os.path.join(P0, "research", "contrast")
CACHE = os.environ.get(
    "CONTRAST_CACHE",
    "/private/tmp/claude-501/-Users-jiamingzhang-Desktop-APP/"
    "6c65cf52-47c9-4de6-a996-f3251ee258ff/scratchpad/global_cache",
)
N_PER_CLASS = 8000
N_FP_SAMPLE = 200_000
THIN_MAX_HALFWIDTH = 3.0


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------
def t_identity(img):
    """BASELINE: img.npy exactly as the production model sees it."""
    return img


def t_stretch_1_99(img):
    """CONTROL. Affine, and on this data a numerical no-op (see module docstring)."""
    return robust_normalize(img, 1.0, 99.0)


def _gamma(g):
    def f(img):
        return np.clip(robust_normalize(img, 1.0, 99.0) ** g, 0.0, 1.0).astype(np.float32)
    return f


def t_equalize_hist(img):
    """Global histogram equalisation: flattens the intensity histogram, so it
    deliberately destroys the absolute intensity scale that the large-radius
    smooth features depend on."""
    from skimage import exposure
    return np.clip(exposure.equalize_hist(img), 0.0, 1.0).astype(np.float32)


def t_dark_stretch(img):
    """OWN ARM 1 -- 'spend the dynamic range where the cracks are'.

    Cracks are dark; a faint crack is only marginally darker than the bulk.
    Stretching the window [p0.5, p40] onto [0,1] multiplies that margin by
    ~2.5x while staying monotone across the entire crack-relevant range. The
    cost is that everything brighter than p40 saturates to 1.0, which erases
    large-radius intensity variation over most of the specimen -- i.e. it
    attacks exactly the feature group that flat-fielding attacked. That makes
    it a sharp test of the mechanism, in the direction contrast enhancement is
    supposed to help."""
    lo, hi = (float(v) for v in np.percentile(img, [0.5, 40.0]))
    return np.clip((img.astype(np.float64) - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)


def t_sigmoid_med_g10(img):
    """OWN ARM 2 -- the non-destructive version of the same idea.

    An S-curve centred on the per-frame median puts maximum slope at the bulk
    intensity (amplifying faint-crack-vs-bulk contrast, like dark_stretch) but
    compresses rather than clips the tails, so intensity ordering survives
    everywhere. If contrast per se is what limits the model, this arm should
    help; if the limit is the absolute intensity scale, this one should be the
    least harmful of the non-linear arms."""
    from skimage import exposure
    c = float(np.median(img))
    return np.clip(exposure.adjust_sigmoid(img, cutoff=c, gain=10), 0.0, 1.0).astype(np.float32)


ARMS = {
    "identity": t_identity,
    "stretch_1_99": t_stretch_1_99,
    "gamma_0.5": _gamma(0.5),
    "gamma_2.0": _gamma(2.0),
    "equalize_hist": t_equalize_hist,
    "dark_stretch_p0.5_p40": t_dark_stretch,
    "sigmoid_med_g10": t_sigmoid_med_g10,
}
ARM_ORDER = list(ARMS)


# --------------------------------------------------------------------------
# featurisation
# --------------------------------------------------------------------------
def featurize_at(img01, flat_idx):
    """The 17 features of compute_feature_stack, evaluated only at flat_idx.

    Computed one plane at a time: the full (H, W, 17) float32 stack is 2.2 GB
    on the largest frame here (5046x6349), which does not survive being held
    once per worker process. Bit-identity with compute_feature_stack is
    asserted by check_featurizer_equivalence() before any arm is run.
    """
    img01 = np.ascontiguousarray(img01, dtype=np.float32)
    out = np.empty((flat_idx.size, N_FEATURES), dtype=np.float32)
    k = 0

    def take(plane):
        nonlocal k
        out[:, k] = np.ravel(plane)[flat_idx]
        k += 1

    take(img01)
    for s in SMOOTH_SIGMAS:
        take(ndi.gaussian_filter(img01, sigma=s))
    for s in GRADIENT_SIGMAS:
        take(ndi.gaussian_gradient_magnitude(img01, sigma=s))
    for s in LAPLACIAN_SIGMAS:
        take(ndi.gaussian_laplace(img01, sigma=s))
    for s in TEXTURE_SIGMAS:
        take(local_std(img01, sigma=s))
    assert k == N_FEATURES
    return out


def check_featurizer_equivalence(img01, n=5000):
    rng = np.random.RandomState(123)
    idx = rng.choice(img01.size, n, replace=False)
    ref = compute_feature_stack(np.ascontiguousarray(img01, np.float32))
    ref = ref.reshape(-1, N_FEATURES)[idx]
    got = featurize_at(img01, idx)
    same = bool(np.array_equal(ref, got))
    maxdiff = float(np.max(np.abs(ref.astype(np.float64) - got.astype(np.float64))))
    del ref
    return same, maxdiff


# --------------------------------------------------------------------------
# stage A: sample indices + thin-frame geometry (arm independent)
# --------------------------------------------------------------------------
def sample_indices(corr):
    """Up to 8000 crack + 8000 not-crack flat pixel indices, RandomState(0)."""
    rng = np.random.RandomState(0)
    pos = np.flatnonzero(np.ravel(corr) == 1)
    neg = np.flatnonzero(np.ravel(corr) == 2)
    if pos.size > N_PER_CLASS:
        pos = rng.choice(pos, N_PER_CLASS, replace=False)
    if neg.size > N_PER_CLASS:
        neg = rng.choice(neg, N_PER_CLASS, replace=False)
    idx = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)])
    return idx.astype(np.int64), y


def frame_halfwidth(img01, corr):
    """Median half-width of the dark core of the painted crack, in px.

    Inside correction==1 take the pixels darker than the 20th percentile of
    img01 within those strokes, drop components under 64 px, then read the
    Euclidean distance transform of that set along its skeleton. This is a
    property of the FRAME, computed on the untransformed img01, so the thin/
    thick split is identical for every arm.
    """
    from skimage import morphology
    mask = np.asarray(corr) == 1
    vals = img01[mask]
    if vals.size < 64:
        return None, 0
    thr = float(np.percentile(vals, 20))
    core = mask & (img01 <= thr)
    core = morphology.remove_small_objects(core, min_size=64)
    if not core.any():
        return None, 0
    sk = morphology.skeletonize(core)
    if not sk.any():
        return None, int(core.sum())
    dt = ndi.distance_transform_edt(core)
    return float(np.median(dt[sk])), int(core.sum())


def stage_a_one(iid):
    img01 = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
    corr = np.asarray(S.load_npy(iid, "correction.npy"))
    idx, y = sample_indices(corr)
    hw, core_px = frame_halfwidth(img01, corr)
    np.savez_compressed(os.path.join(CACHE, f"idx_{iid}.npz"), idx=idx, y=y)
    return dict(id=iid, shape=list(img01.shape), n_rows=int(idx.size),
                n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
                halfwidth_px=hw, core_px=core_px,
                thin=(hw is not None and hw <= THIN_MAX_HALFWIDTH))


# --------------------------------------------------------------------------
# stage B: per-image, per-arm feature rows
# --------------------------------------------------------------------------
def stage_b_one(iid):
    """All arms for one frame: load img.npy once, transform + featurise 7x."""
    t0 = time.time()
    out_path = os.path.join(CACHE, f"feat_{iid}.npz")
    if os.path.exists(out_path):
        return dict(id=iid, cached=True, sec=0.0)
    idx = np.load(os.path.join(CACHE, f"idx_{iid}.npz"))["idx"]
    img01 = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
    blobs = {}
    diag = {}
    for name in ARM_ORDER:
        x = np.ascontiguousarray(ARMS[name](img01), np.float32)
        assert x.shape == img01.shape and x.min() >= 0.0 and x.max() <= 1.0, name
        diag[name] = affinity_diag(img01, x)
        blobs[name] = featurize_at(x, idx)
        del x
    np.savez_compressed(out_path, **blobs)
    return dict(id=iid, cached=False, sec=round(time.time() - t0, 1), diag=diag)


def affinity_diag(src, dst, n=200_000):
    """Is this transform affine on this frame, and how much does it saturate?

    Fits dst ~ a*src + b by least squares on a random sample and reports the
    worst residual. ~0 means affine, hence invisible to StandardScaler.
    """
    rng = np.random.RandomState(7)
    i = rng.choice(src.size, min(n, src.size), replace=False)
    a_, b_ = np.ravel(src)[i].astype(np.float64), np.ravel(dst)[i].astype(np.float64)
    A = np.vstack([a_, np.ones_like(a_)]).T
    coef, *_ = np.linalg.lstsq(A, b_, rcond=None)
    resid = float(np.max(np.abs(A @ coef - b_)))
    return dict(slope=float(coef[0]), intercept=float(coef[1]),
                max_affine_residual=resid,
                frac_at_0=float((b_ <= 0.0).mean()), frac_at_1=float((b_ >= 1.0).mean()),
                identical_to_input=bool(np.array_equal(np.ravel(src)[i], np.ravel(dst)[i])))


# --------------------------------------------------------------------------
# stage C/D: scoring
# --------------------------------------------------------------------------
def make_clf():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def iou_pr(y, p):
    yhat = p > 0.5
    yt = y == 1
    tp = int(np.sum(yhat & yt)); fp = int(np.sum(yhat & ~yt)); fn = int(np.sum(~yhat & yt))
    den = tp + fp + fn
    return (dict(iou=(tp / den) if den else float("nan"),
                 precision=(tp / (tp + fp)) if (tp + fp) else float("nan"),
                 recall=(tp / (tp + fn)) if (tp + fn) else float("nan"),
                 n=int(y.size), n_pos=int(yt.sum())))


def _load_arm(arm):
    """Rows for one arm, from the on-disk matrix dumped by main()."""
    X = np.load(os.path.join(CACHE, f"X_{arm}.npy"), mmap_mode="r")
    y = np.load(os.path.join(CACHE, "y.npy"))
    g = np.load(os.path.join(CACHE, "g.npy"), allow_pickle=True)
    return X, y, g


def _fold(args):
    """One (arm, fold). Reads its own rows off disk so the 7 x 960k x 17
    matrices are never pickled to workers."""
    arm, tr, te, thin_ids = args
    X, y, g = _load_arm(arm)
    clf = make_clf()
    clf.fit(np.asarray(X[tr]), y[tr])
    p = clf.predict_proba(np.asarray(X[te]))[:, 1]
    all_m = iou_pr(y[te], p)
    sel = np.isin(g[te], list(thin_ids))
    thin_m = iou_pr(y[te][sel], p[sel]) if sel.any() else None
    return dict(arm=arm, all=all_m, thin=thin_m,
                test_groups=sorted(set(g[te].tolist())),
                thin_test_groups=sorted(set(g[te][sel].tolist())) if sel.any() else [])


def _train_full(arm):
    """Model trained on ALL rows for one arm -- used by the FP guardrail."""
    import joblib
    X, y, _ = _load_arm(arm)
    c = make_clf()
    c.fit(np.asarray(X), y)
    p = os.path.join(CACHE, f"model_{arm}.pkl")
    joblib.dump(c, p)
    return arm, p


def _fp_one(iid):
    """False-positive fraction on one crack-free specimen, for every arm."""
    import joblib
    img01 = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
    rng = np.random.RandomState(0)
    idx = rng.choice(img01.size, N_FP_SAMPLE, replace=False).astype(np.int64)
    try:
        sup = np.ravel(P.specimen_support(img01))[idx]
        sup_err = None
    except Exception as e:  # noqa: BLE001
        sup, sup_err = np.ones(idx.size, bool), repr(e)
    out = {}
    for arm in ARM_ORDER:
        mdl = joblib.load(os.path.join(CACHE, f"model_{arm}.pkl"))
        x = np.ascontiguousarray(ARMS[arm](img01), np.float32)
        F = featurize_at(x, idx)
        del x
        pred = mdl.predict_proba(F)[:, 1] > 0.5
        out[arm] = dict(fp_frac=float(pred.mean()),
                        fp_frac_on_specimen=float(pred[sup].mean()) if sup.any() else None,
                        support_frac=float(sup.mean()), support_error=sup_err)
        del F
    return iid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: use only the N smallest frames / 2 clean specimens")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    T0 = time.time()
    timing = {}

    # ---- inventory -------------------------------------------------------
    ims = S.list_images()
    labelled = []
    for m in ims:
        c = S.load_npy(m["id"], "correction.npy", mmap=True)
        if c is None:
            continue
        c = np.asarray(c)
        if (c == 1).any() and (c == 2).any():
            labelled.append(m)
        del c
    clean = [m for m in ims
             if any(k.lower() in (m.get("filename") or "").lower() for k in P.CLEAN_SPECIMENS)]
    if args.limit:
        labelled = sorted(labelled, key=lambda m: m["width"] * m["height"])[:args.limit]
        clean = clean[:2]
    print(f"[inv] {len(labelled)} labelled frames, {len(clean)} crack-free specimens", flush=True)
    name_of = {m["id"]: m["filename"] for m in ims}

    # ---- featuriser equivalence -----------------------------------------
    t = time.time()
    small = min(labelled, key=lambda m: m["width"] * m["height"])
    same, maxdiff = check_featurizer_equivalence(
        np.asarray(S.load_npy(small["id"], "img.npy"), np.float32))
    print(f"[chk] featurize_at == compute_feature_stack: {same} (maxdiff {maxdiff:g})", flush=True)
    assert same, "plane-by-plane featuriser diverged from compute_feature_stack"
    timing["featurizer_check_sec"] = round(time.time() - t, 1)

    # ---- stage A ---------------------------------------------------------
    t = time.time()
    meta_path = os.path.join(CACHE, "stage_a.json")
    if os.path.exists(meta_path):
        frames = json.load(open(meta_path))
        print("[A] cached", flush=True)
    else:
        frames = []
        with ProcessPoolExecutor(args.workers) as ex:
            for i, r in enumerate(ex.map(stage_a_one, [m["id"] for m in labelled]), 1):
                frames.append(r)
                print(f"[A] {i}/{len(labelled)} hw={r['halfwidth_px']} thin={r['thin']}", flush=True)
        json.dump(frames, open(meta_path, "w"))
    timing["stage_a_sec"] = round(time.time() - t, 1)
    thin_ids = {f["id"] for f in frames if f["thin"]}
    print(f"[A] {len(thin_ids)}/{len(frames)} THIN frames", flush=True)

    # ---- stage B ---------------------------------------------------------
    t = time.time()
    diags = {}
    order = sorted(labelled, key=lambda m: -m["width"] * m["height"])
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(stage_b_one, [m["id"] for m in order]), 1):
            if r.get("diag"):
                diags[r["id"]] = r["diag"]
            print(f"[B] {i}/{len(order)} {r['sec']}s cached={r['cached']}", flush=True)
    timing["stage_b_sec"] = round(time.time() - t, 1)

    # ---- assemble matrices ----------------------------------------------
    t = time.time()
    ys, gs = [], []
    for f in frames:
        d = np.load(os.path.join(CACHE, f"idx_{f['id']}.npz"))
        ys.append(d["y"])
        gs.append(np.array([f["id"]] * d["y"].size))
    y = np.concatenate(ys).astype(np.int8)
    g = np.concatenate(gs)
    np.save(os.path.join(CACHE, "y.npy"), y)
    np.save(os.path.join(CACHE, "g.npy"), g)
    for arm in ARM_ORDER:
        Xa = np.concatenate(
            [np.load(os.path.join(CACHE, f"feat_{f['id']}.npz"))[arm] for f in frames]
        ).astype(np.float32)
        assert Xa.shape == (y.size, N_FEATURES)
        np.save(os.path.join(CACHE, f"X_{arm}.npy"), Xa)
        del Xa
    print(f"[X] {y.size} rows, {int((y==1).sum())} crack / {int((y==0).sum())} not", flush=True)
    timing["assemble_sec"] = round(time.time() - t, 1)

    # affine-equivalence evidence: does StandardScaler erase this transform?
    from sklearn.preprocessing import StandardScaler
    std_eq = {}
    Xid = np.load(os.path.join(CACHE, "X_identity.npy"), mmap_mode="r")
    a = StandardScaler().fit_transform(np.asarray(Xid[:50000], np.float64))
    for arm in ARM_ORDER:
        Xa = np.load(os.path.join(CACHE, f"X_{arm}.npy"), mmap_mode="r")
        b = StandardScaler().fit_transform(np.asarray(Xa[:50000], np.float64))
        std_eq[arm] = dict(max_abs_diff_standardised=float(np.max(np.abs(a - b))),
                           raw_identical=bool(np.array_equal(np.asarray(Xid), np.asarray(Xa))))
        print(f"[std] {arm}: max|z-z_id| = {std_eq[arm]['max_abs_diff_standardised']:.3g} "
              f"raw_identical={std_eq[arm]['raw_identical']}", flush=True)
        del Xa, b

    # ---- stage C: GroupKFold(5) -----------------------------------------
    from sklearn.model_selection import GroupKFold
    from joblib import Parallel, delayed
    t = time.time()
    splits = list(GroupKFold(5).split(np.zeros((y.size, 1), np.float32), y, groups=g))
    jobs = [(arm, tr, te, thin_ids) for arm in ARM_ORDER for (tr, te) in splits]
    print(f"[C] {len(jobs)} fold-fits on {args.workers} workers", flush=True)
    res = Parallel(n_jobs=args.workers, verbose=10)(delayed(_fold)(j) for j in jobs)
    timing["stage_c_sec"] = round(time.time() - t, 1)

    per_arm = {}
    for k, arm in enumerate(ARM_ORDER):
        folds = res[k * 5:(k + 1) * 5]
        assert all(f["arm"] == arm for f in folds)
        def agg(key, metric):
            v = [f[key][metric] for f in folds if f[key] is not None]
            return dict(mean=float(np.mean(v)), std=float(np.std(v)), folds=[float(x) for x in v])
        per_arm[arm] = dict(
            all={m: agg("all", m) for m in ("iou", "precision", "recall")},
            thin={m: agg("thin", m) for m in ("iou", "precision", "recall")},
            n_folds_with_thin=sum(1 for f in folds if f["thin"] is not None),
            fold_test_groups=[f["test_groups"] for f in folds],
        )
        print(f"[C] {arm:24s} IoU {per_arm[arm]['all']['iou']['mean']:.4f}"
              f" +-{per_arm[arm]['all']['iou']['std']:.4f}"
              f"  thinIoU {per_arm[arm]['thin']['iou']['mean']:.4f}", flush=True)

    # ---- stage D: crack-free false-positive guardrail --------------------
    t = time.time()
    print(f"[D] training {len(ARM_ORDER)} full-data models", flush=True)
    Parallel(n_jobs=min(args.workers, len(ARM_ORDER)), verbose=10)(
        delayed(_train_full)(arm) for arm in ARM_ORDER)
    fp = {arm: {} for arm in ARM_ORDER}
    fpres = Parallel(n_jobs=min(args.workers, len(clean)), verbose=10)(
        delayed(_fp_one)(m["id"]) for m in clean)
    for iid, per in fpres:
        for arm in ARM_ORDER:
            fp[arm][iid] = dict(filename=name_of.get(iid, ""), **per[arm])
        print(f"[D] {name_of.get(iid,'')[:44]} "
              + " ".join(f"{a}={fp[a][iid]['fp_frac']:.4f}" for a in ARM_ORDER), flush=True)
    timing["stage_d_sec"] = round(time.time() - t, 1)
    timing["total_sec"] = round(time.time() - T0, 1)

    # ---- write out -------------------------------------------------------
    out = dict(
        arm_group="GLOBAL (whole-image contrast transforms)",
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        protocol=dict(n_per_class=N_PER_CLASS, rng="RandomState(0)",
                      cv="GroupKFold(5) grouped by frame",
                      clf="StandardScaler + MLPClassifier((64,32), max_iter=300, "
                          "random_state=0, early_stopping=True, n_iter_no_change=8)",
                      threshold=0.5, features=FEATURE_NAMES,
                      thin_max_halfwidth_px=THIN_MAX_HALFWIDTH,
                      fp_sample_px=N_FP_SAMPLE),
        n_frames=len(frames), n_rows=int(y.size),
        n_crack_rows=int((y == 1).sum()), n_not_crack_rows=int((y == 0).sum()),
        featurizer_bit_identical=same,
        frames=[dict(f, filename=name_of.get(f["id"], "")) for f in frames],
        thin_frames=sorted(name_of.get(i, i) for i in thin_ids),
        arms=per_arm,
        standardised_vs_identity=std_eq,
        transform_diagnostics=diags,
        false_positive_guardrail=fp,
        fp_mean={arm: float(np.mean([v["fp_frac"] for v in fp[arm].values()])) for arm in ARM_ORDER},
        fp_mean_on_specimen={arm: float(np.mean(
            [v["fp_frac_on_specimen"] for v in fp[arm].values()
             if v["fp_frac_on_specimen"] is not None])) for arm in ARM_ORDER},
        timing_sec=timing,
    )
    with open(os.path.join(OUT_DIR, "global_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote research/contrast/global_results.json", flush=True)
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
