"""Pass 2: score every arm on the cached matrix, then the crack-free guardrail.

Arms (all are COLUMN SUBSETS of the one cached 37-column matrix, so the only thing that
differs between them is which columns the classifier sees):
    baseline_17             17   the features as they ship
    17_plus_contrast3       20   + lcn_w51, lcn_w151, dog_g8
    17_plus_clahe_int       18   + CLAHE intensity
    17_plus_clahe_int_grad  19   + CLAHE intensity, CLAHE gradmag_s2
    34_dup_clahe_stack      34   + the whole 17-feature stack recomputed on the CLAHE image

Protocol: GroupKFold(5) grouped by image; IoU/precision/recall at 0.5 on held-out rows;
the same numbers restricted to held-out rows from thin-crack frames; then a model trained
on all rows applied to 200k random pixels from each of the 6 crack-free specimens, where
every positive is a false positive by construction.

Usage:
    .venv/bin/python research/contrast/augment_eval.py --workers 6
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
sys.path.insert(0, os.path.join(P0, "research", "contrast"))

import augment_features as AF          # noqa: E402

THIN_MAX_HALF_WIDTH = 3.0

# FAINT is an addition to the brief, not a substitute for it. The brief defines THIN
# (median half-width <= 3 px) but the question is about thin AND faint cracks, and
# thinness alone does not measure amplitude: a 1 px crack can be pitch black. Faintness
# here is median(smooth_s64 - intensity) over a frame's crack rows -- how much darker a
# crack pixel is than its own broad neighbourhood -- and costs nothing extra, since both
# columns are already in the cached matrix. FAINT = lowest tertile of frames.
FAINT_TERTILE = 1.0 / 3.0


def clf():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def load_cache(cache):
    i_int = AF.FEATURE_NAMES.index("intensity")
    i_s64 = AF.FEATURE_NAMES.index("smooth_s64")
    Xs, ys, gs, frames = [], [], [], []
    for p in sorted(glob.glob(os.path.join(cache, "lab__*.npz"))):
        z = np.load(p, allow_pickle=False)
        iid = os.path.basename(p)[len("lab__"):-len(".npz")]
        X, y = z["X"], z["y"].astype(bool)
        Xs.append(X)
        ys.append(z["y"])
        gs.append(np.full(len(y), len(frames), np.int32))
        hw = float(z["half_width"])
        frames.append(dict(image_id=iid, filename=str(z["filename"]),
                           shape=[int(v) for v in z["shape"]],
                           half_width=None if hw < 0 else hw,
                           thin=bool(0 <= hw <= THIN_MAX_HALF_WIDTH),
                           contrast=float(np.median(X[y, i_s64] - X[y, i_int])),
                           n_dark=int(z["n_dark"]),
                           n_rows=int(len(y)),
                           n_crack_rows=int(y.sum()),
                           n_crack_px_total=int(z["n_crack_px"]),
                           n_not_px_total=int(z["n_not_px"])))
    cut = float(np.quantile([f["contrast"] for f in frames], FAINT_TERTILE))
    for f in frames:
        f["faint"] = bool(f["contrast"] <= cut)
        f["thin_and_faint"] = bool(f["thin"] and f["faint"])
        f["faint_cut"] = cut
    return (np.concatenate(Xs), np.concatenate(ys).astype(bool),
            np.concatenate(gs), frames)


def load_clean(cache):
    out = []
    for p in sorted(glob.glob(os.path.join(cache, "clean__*.npz"))):
        z = np.load(p, allow_pickle=False)
        out.append(dict(image_id=os.path.basename(p)[len("clean__"):-len(".npz")],
                        filename=str(z["filename"]), X=z["X"], on_spec=z["on_spec"]))
    return out


def score(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return dict(iou=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), tp=tp, fp=fp, fn=fn)


def iou_of(model, X, t):
    return score(model.predict_proba(X)[:, 1] > 0.5, t)["iou"]


def run_fold(payload):
    """One (arm, fold) fit. Returns all-rows and thin-rows scores for that fold."""
    arm, cols, fold, tr, te, cache, do_perm = payload
    X, y, g, frames = load_cache(cache)
    t0 = time.time()
    Xc = X[:, cols]
    m = clf().fit(Xc[tr], y[tr])
    p = m.predict_proba(Xc[te])[:, 1] > 0.5
    res = dict(arm=arm, fold=fold, n_train=len(tr), n_test=len(te),
               all=score(p, y[te]))
    for axis in ("thin", "faint", "thin_and_faint"):
        gsel = [i for i, f in enumerate(frames) if f[axis]]
        sel = np.isin(g[te], gsel)
        res[axis] = score(p[sel], y[te][sel]) if sel.any() else None
        res["n_test_" + axis] = int(sel.sum())
    res["n_iter"] = int(m.named_steps["m"].n_iter_)
    # Persist the held-out predictions so any frame subset can be re-scored later without
    # refitting: the fits are the whole cost of this script.
    np.savez_compressed(os.path.join(cache, "pred__%s__f%d.npz" % (arm, fold)),
                        te=te.astype(np.int32), pred=p)

    if do_perm:
        # Permutation importance on this one fold, scored with the headline metric (IoU at
        # 0.5) rather than accuracy, on a 60k subsample of the held-out rows. This is the
        # result that says whether the ADDED columns carry anything the 17 do not.
        rng = np.random.RandomState(0)
        sub = te if len(te) <= 60000 else rng.choice(te, 60000, replace=False)
        Xs, ts = Xc[sub], y[sub]
        base = iou_of(m, Xs, ts)
        drops = []
        for j in range(Xs.shape[1]):
            d = []
            for r in range(3):
                Xp = Xs.copy()
                Xp[:, j] = Xp[np.random.RandomState(100 + r).permutation(len(Xp)), j]
                d.append(base - iou_of(m, Xp, ts))
            drops.append(float(np.mean(d)))
        res["perm"] = dict(baseline_iou=float(base),
                           drop={AF.ALL_NAMES[c]: round(drops[j], 5)
                                 for j, c in enumerate(cols)})
    res["seconds"] = time.time() - t0
    return res


def run_guardrail(payload):
    """Fit on ALL labelled rows, then measure the crack-free false-positive rate."""
    arm, cols, cache = payload
    X, y, g, frames = load_cache(cache)
    t0 = time.time()
    m = clf().fit(X[:, cols], y)
    per = []
    for c in load_clean(cache):
        p = m.predict_proba(c["X"][:, cols])[:, 1] > 0.5
        on = c["on_spec"].astype(bool)
        per.append(dict(image=c["filename"], n=int(len(p)),
                        fp_frac=float(p.mean()),
                        fp_frac_on_specimen=float(p[on].mean()) if on.any() else None,
                        on_spec_frac=float(on.mean())))
    return dict(arm=arm, per_specimen=per,
                mean_fp_frac=float(np.mean([q["fp_frac"] for q in per])),
                max_fp_frac=float(max(q["fp_frac"] for q in per)),
                mean_fp_frac_on_specimen=float(np.mean(
                    [q["fp_frac_on_specimen"] for q in per
                     if q["fp_frac_on_specimen"] is not None])),
                seconds=time.time() - t0)


def run_rf_importance(payload):
    """RandomForest impurity importance on a subsample -- a cheap global ranking that
    includes every added channel at once, as a cross-check on the per-arm permutation."""
    arm, cols, cache, n_sub = payload
    from sklearn.ensemble import RandomForestClassifier
    X, y, g, frames = load_cache(cache)
    rng = np.random.RandomState(0)
    sub = rng.choice(len(y), min(n_sub, len(y)), replace=False)
    t0 = time.time()
    rf = RandomForestClassifier(200, max_depth=None, min_samples_leaf=5, n_jobs=4,
                                random_state=0).fit(X[sub][:, cols], y[sub])
    imp = rf.feature_importances_
    return dict(arm=arm, n_sub=len(sub), seconds=time.time() - t0,
                importance={AF.ALL_NAMES[c]: round(float(imp[j]), 5)
                            for j, c in enumerate(cols)})


def agg(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None
    out = {}
    for k in ("iou", "precision", "recall"):
        v = np.array([x[k] for x in vals], float)
        out[k] = round(float(v.mean()), 4)
        out[k + "_sd"] = round(float(v.std(ddof=0)), 4)
        out[k + "_folds"] = [round(float(x), 4) for x in v]
    out["n_folds"] = len(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        "/private/tmp/claude-501/-Users-jiamingzhang-Desktop-APP",
        "6c65cf52-47c9-4de6-a996-f3251ee258ff", "scratchpad", "featcache"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(P0, "research", "contrast",
                                                  "augment_results.json"))
    ap.add_argument("--rf-sub", type=int, default=250000)
    ap.add_argument("--arms", default="")
    a = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    T0 = time.time()
    X, y, g, frames = load_cache(a.cache)
    print("rows %d  cols %d  frames %d  positives %.3f" %
          (len(y), X.shape[1], len(frames), y.mean()), flush=True)
    for axis in ("thin", "faint", "thin_and_faint"):
        print("%-15s frames %d/%d" % (axis, sum(f[axis] for f in frames), len(frames)),
              flush=True)
    print("faint cut (median smooth_s64-intensity over crack rows) = %.5f"
          % frames[0]["faint_cut"], flush=True)

    arms = AF.ARMS if not a.arms else {k: AF.ARMS[k] for k in a.arms.split(",")}
    splits = list(GroupKFold(5).split(X, y, groups=g))
    del X

    fold_jobs, gr_jobs, rf_jobs = [], [], []
    for arm, cols in arms.items():
        for f, (tr, te) in enumerate(splits, 1):
            fold_jobs.append((arm, cols, f, tr, te, a.cache, f == 1))
        gr_jobs.append((arm, cols, a.cache))
    rf_jobs.append(("34_dup_clahe_stack", AF.ARMS["34_dup_clahe_stack"], a.cache, a.rf_sub))
    rf_jobs.append(("17_plus_contrast3", AF.ARMS["17_plus_contrast3"], a.cache, a.rf_sub))
    rf_jobs.append(("all_37", list(range(37)), a.cache, a.rf_sub))

    from concurrent.futures import ProcessPoolExecutor
    results = dict(n_rows=int(len(y)), n_frames=len(frames), positives_frac=float(y.mean()),
                   frames=frames, arm_dims={k: len(v) for k, v in arms.items()},
                   clahe=dict(kernel=AF.CLAHE_KERNEL, clip_limit=AF.CLAHE_CLIP,
                              nbins=AF.CLAHE_NBINS),
                   added_channels=dict(contrast3=AF.CONTRAST_NAMES,
                                       clahe_int=[AF.ALL_NAMES[c] for c in AF.COL_CLAHE_INT],
                                       clahe_int_grad=[AF.ALL_NAMES[c] for c in
                                                       AF.COL_CLAHE_INT + AF.COL_CLAHE_GRAD],
                                       clahe17=[AF.ALL_NAMES[c] for c in AF.COL_CLAHE17]))
    del y, g

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        ffuts = [ex.submit(run_fold, j) for j in fold_jobs]
        gfuts = [ex.submit(run_guardrail, j) for j in gr_jobs]
        rfuts = [ex.submit(run_rf_importance, j) for j in rf_jobs]

        folds = []
        for k, fu in enumerate(ffuts, 1):
            r = fu.result(); folds.append(r)
            print("  fold %-24s f%d  %6.1fs iou=%.4f thin_iou=%s iters=%d" %
                  (r["arm"], r["fold"], r["seconds"], r["all"]["iou"],
                   None if r["thin"] is None else round(r["thin"]["iou"], 4),
                   r["n_iter"]), flush=True)
        guards = [fu.result() for fu in gfuts]
        for q in guards:
            print("  guardrail %-24s mean_fp=%.5f max=%.5f on_spec=%.5f (%.0fs)" %
                  (q["arm"], q["mean_fp_frac"], q["max_fp_frac"],
                   q["mean_fp_frac_on_specimen"], q["seconds"]), flush=True)
        rfs = [fu.result() for fu in rfuts]

    results["folds"] = folds
    results["guardrail"] = {q["arm"]: q for q in guards}
    results["rf_importance"] = {q["arm"]: q for q in rfs}
    results["perm_importance"] = {r["arm"]: r["perm"] for r in folds if "perm" in r}
    summary = {}
    for arm in arms:
        rs = [r for r in folds if r["arm"] == arm]
        gq = results["guardrail"].get(arm, {})
        summary[arm] = dict(dim=len(arms[arm]), all_rows=agg(rs, "all"),
                            thin_frames=agg(rs, "thin"),
                            faint_frames=agg(rs, "faint"),
                            thin_and_faint_frames=agg(rs, "thin_and_faint"),
                            crackfree_fp=round(gq.get("mean_fp_frac", float("nan")), 5),
                            crackfree_fp_on_specimen=round(
                                gq.get("mean_fp_frac_on_specimen", float("nan")), 5),
                            fit_seconds=round(sum(r["seconds"] for r in rs), 1))
    results["summary"] = summary
    results["wall_clock_seconds"] = round(time.time() - T0, 1)

    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("\nwrote %s  (%.1f s)" % (a.out, results["wall_clock_seconds"]), flush=True)
    def _i(k):
        return "None" if summary[arm][k] is None else "%.4f" % summary[arm][k]["iou"]
    for arm, s in summary.items():
        print("%-24s dim=%-3d iou=%.4f+-%.4f p=%.4f r=%.4f | thin=%s faint=%s t&f=%s "
              "| fp=%.5f fp_spec=%.5f" % (
                  arm, s["dim"], s["all_rows"]["iou"], s["all_rows"]["iou_sd"],
                  s["all_rows"]["precision"], s["all_rows"]["recall"],
                  _i("thin_frames"), _i("faint_frames"), _i("thin_and_faint_frames"),
                  s["crackfree_fp"], s["crackfree_fp_on_specimen"]), flush=True)


if __name__ == "__main__":
    main()
