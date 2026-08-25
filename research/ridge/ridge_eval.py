"""Pass 2: score every arm on the cached matrix, then the crack-free guardrail.

Protocol, held fixed across arms and identical to docs/CONTRAST.md so the two sweeps can
be read side by side:
  rows        up to 8000 crack + 8000 not-crack per image, np.random.RandomState(0)
  classifier  Pipeline([StandardScaler, MLPClassifier((64,32), max_iter=300,
              random_state=0, early_stopping=True, n_iter_no_change=8)])
  scoring     GroupKFold(5) grouped by IMAGE -- train and test never share an image
  reported    mean IoU / precision / recall at 0.5 with the fold spread

Three extra columns, each answering a question the headline IoU cannot:

THIN FRAMES. The subset the owner actually cares about, by the brief's definition (median
half-width <= 3 px of the darkest-fifth core inside correction==1). Scored on held-out
rows from those frames only.

ON-SPECIMEN FALSE POSITIVES. The decisive column. Whole-frame FP on the crack-free
specimens is dominated by off-specimen background and inverted the conclusion of an
experiment run earlier today. Both are recorded, only the on-specimen one is read.

PAIRED PER-FRAME DELTA. Fold means hide small consistent effects behind between-frame
difficulty variance: the folds differ by ~0.03 IoU mostly because they contain different
frames. Every arm sees the SAME frames in the SAME folds, so per-frame IoU can be paired
against baseline_17, which is a far more sensitive test of "did this move anything" than
comparing two noisy fold means. It is a tie-breaker for direction, NOT a licence to call a
winner: an arm whose paired delta is real but smaller than the fold spread is still
indistinguishable in deployment terms.

Usage:
    .venv/bin/python research/ridge/ridge_eval.py --workers 6
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
sys.path.insert(0, os.path.join(P0, "research", "ridge"))

import ridge_features as RF          # noqa: E402

THIN_MAX_HALF_WIDTH = 3.0

# DIMENSIONALITY CONTROL. 17_plus_hessian_eigs has 26 inputs and 17_plus_all_ridge has 32,
# against the baseline's 17. Widening an MLP's input layer is not free even when the new
# columns are worthless: it adds parameters to fit from the same rows and shifts where
# early stopping lands. So "arm is 0.004 below baseline" is uninterpretable until we know
# what 9 columns of GUARANTEED-worthless input cost. This arm measures exactly that, and
# it is the yardstick every negative delta below should be read against.
#
# Not cached with the real features: generated deterministically at fit time from the row
# index, so it costs no disk and is identical across folds and workers.
NOISE_ARMS = {"17_plus_noise9": 9}


def add_noise_cols(Xc, n_noise, n_rows_total):
    if not n_noise:
        return Xc
    z = np.random.RandomState(12345).standard_normal((n_rows_total, n_noise)).astype(np.float32)
    return np.hstack([Xc, z])


def clf():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def load_cache(cache):
    Xs, ys, gs, frames = [], [], [], []
    for p in sorted(glob.glob(os.path.join(cache, "lab__*.npz"))):
        z = np.load(p, allow_pickle=False)
        iid = os.path.basename(p)[len("lab__"):-len(".npz")]
        X, y = z["X"], z["y"]
        Xs.append(X)
        ys.append(y)
        gs.append(np.full(len(y), len(frames), np.int32))
        hw = float(z["half_width"])
        frames.append(dict(image_id=iid, filename=str(z["filename"]),
                           shape=[int(v) for v in z["shape"]],
                           half_width=None if hw < 0 else hw,
                           thin=bool(0 <= hw <= THIN_MAX_HALF_WIDTH),
                           n_dark=int(z["n_dark"]), n_rows=int(len(y)),
                           n_crack_rows=int(y.sum()),
                           n_crack_px_total=int(z["n_crack_px"]),
                           n_not_px_total=int(z["n_not_px"])))
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
    """One (arm, fold) fit. Returns all-rows, thin-rows, and per-frame scores."""
    arm, cols, fold, tr, te, cache, do_perm = payload
    X, y, g, frames = load_cache(cache)
    t0 = time.time()
    Xc = X[:, cols]
    del X
    # Noise columns are appended over the FULL row set before the fold split, so a given
    # row carries the same noise in training and in testing -- as a real feature would.
    Xc = add_noise_cols(Xc, NOISE_ARMS.get(arm, 0), len(Xc))
    names = [RF.ALL_NAMES[c] for c in cols] + \
            ["noise_%d" % i for i in range(NOISE_ARMS.get(arm, 0))]
    m = clf().fit(Xc[tr], y[tr])
    p = m.predict_proba(Xc[te])[:, 1] > 0.5
    res = dict(arm=arm, fold=fold, n_train=int(len(tr)), n_test=int(len(te)),
               all=score(p, y[te]), n_iter=int(m.named_steps["m"].n_iter_))
    gsel = [i for i, f in enumerate(frames) if f["thin"]]
    sel = np.isin(g[te], gsel)
    res["thin"] = score(p[sel], y[te][sel]) if sel.any() else None
    res["n_test_thin"] = int(sel.sum())
    # Per-frame IoU on held-out rows, for the paired comparison against baseline.
    gt = g[te]
    res["per_frame"] = {}
    for i in np.unique(gt):
        s = gt == i
        res["per_frame"][int(i)] = round(score(p[s], y[te][s])["iou"], 6)

    if do_perm:
        # Permutation importance on fold 1, scored with the HEADLINE metric (IoU at 0.5)
        # rather than accuracy, on a 60k subsample of the held-out rows. This is the result
        # that says whether an added column carries anything the 17 do not -- and if a
        # ridge channel ranks high while arm IoU does not move, the channel is a
        # re-encoding of something already present rather than new information.
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
                           drop={names[j]: round(drops[j], 5)
                                 for j in range(len(names))})
    res["seconds"] = time.time() - t0
    return res


def run_guardrail(payload):
    """Fit on ALL labelled rows, then measure the crack-free false-positive rate.

    Every positive here is a false positive by construction: the owner confirmed these six
    specimens contain no crack (pipeline.CLEAN_SPECIMENS). Reported whole-frame AND
    restricted to P.specimen_support, because those two disagree by 10x and only the
    second is a claim about the metal.
    """
    arm, cols, cache = payload
    X, y, g, frames = load_cache(cache)
    t0 = time.time()
    n_noise = NOISE_ARMS.get(arm, 0)
    m = clf().fit(add_noise_cols(X[:, cols], n_noise, len(X)), y)
    del X
    per = []
    for c in load_clean(cache):
        Xg = add_noise_cols(c["X"][:, cols], n_noise, len(c["X"]))
        p = m.predict_proba(Xg)[:, 1] > 0.5
        on = c["on_spec"].astype(bool)
        per.append(dict(image=c["filename"], n=int(len(p)), fp_frac=float(p.mean()),
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
    """RandomForest impurity importance over ALL 34 columns at once.

    A cross-check on the per-arm permutation with a completely different model family: if
    a ridge channel is genuinely informative it should surface in both, and if it surfaces
    in neither then the arm's flat IoU is not an MLP artefact.
    """
    arm, cols, cache, n_sub = payload
    from sklearn.ensemble import RandomForestClassifier
    X, y, g, frames = load_cache(cache)
    rng = np.random.RandomState(0)
    sub = rng.choice(len(y), min(n_sub, len(y)), replace=False)
    t0 = time.time()
    rf = RandomForestClassifier(200, min_samples_leaf=5, n_jobs=4,
                                random_state=0).fit(X[sub][:, cols], y[sub])
    imp = rf.feature_importances_
    return dict(arm=arm, n_sub=int(len(sub)), seconds=time.time() - t0,
                importance={RF.ALL_NAMES[c]: round(float(imp[j]), 5)
                            for j, c in enumerate(cols)})


def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
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
        "6c65cf52-47c9-4de6-a996-f3251ee258ff", "scratchpad", "ridgecache"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(P0, "research", "ridge",
                                                  "ridge_results.json"))
    ap.add_argument("--rf-sub", type=int, default=250000)
    ap.add_argument("--arms", default="")
    a = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    T0 = time.time()
    X, y, g, frames = load_cache(a.cache)
    n_thin = sum(f["thin"] for f in frames)
    print("rows %d  cols %d  frames %d (%d thin)  positives %.3f"
          % (len(y), X.shape[1], len(frames), n_thin, y.mean()), flush=True)
    print("\nTHIN FRAMES (median half-width <= %.1f px):" % THIN_MAX_HALF_WIDTH, flush=True)
    for f in sorted((q for q in frames if q["thin"]), key=lambda q: q["half_width"]):
        print("   hw=%.2f px  %s" % (f["half_width"], f["filename"][:78]), flush=True)

    # The noise control rides along as an ordinary arm: same 17 real columns, plus 9
    # columns that cannot possibly help. Its delta IS the dimensionality tax.
    all_arms = dict(RF.ARMS)
    all_arms["17_plus_noise9"] = RF.COL_BASE17
    arms = all_arms if not a.arms else {k: all_arms[k] for k in a.arms.split(",")}

    def dim(arm):
        return len(arms[arm]) + NOISE_ARMS.get(arm, 0)

    splits = [(tr, te) for tr, te in GroupKFold(5).split(X, y, groups=g)]
    del X

    fold_jobs, gr_jobs, rf_jobs = [], [], []
    for arm, cols in arms.items():
        for f, (tr, te) in enumerate(splits, 1):
            fold_jobs.append((arm, cols, f, tr, te, a.cache, f == 1))
        gr_jobs.append((arm, cols, a.cache))
    rf_jobs.append(("all_34", list(range(RF.N_COLS)), a.cache, a.rf_sub))
    rf_jobs.append(("17_plus_all_ridge", RF.ARMS["17_plus_all_ridge"], a.cache, a.rf_sub))

    from concurrent.futures import ProcessPoolExecutor
    results = dict(n_rows=int(len(y)), n_frames=len(frames), n_thin=n_thin,
                   positives_frac=float(y.mean()), frames=frames,
                   arm_dims={k: dim(k) for k in arms},
                   arm_cols={k: ([RF.ALL_NAMES[c] for c in v]
                                 + ["noise_%d" % i for i in range(NOISE_ARMS.get(k, 0))])
                             for k, v in arms.items()},
                   config=dict(sigmas_fine=list(RF.SIGMAS_FINE),
                               sigmas_coarse=list(RF.SIGMAS_COARSE),
                               hess_sigmas=list(RF.HESS_SIGMAS),
                               gamma_fixed=RF.GAMMA_FIXED,
                               black_ridges=RF.BLACK_RIDGES))
    del y, g

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        ffuts = [ex.submit(run_fold, j) for j in fold_jobs]
        gfuts = [ex.submit(run_guardrail, j) for j in gr_jobs]
        rfuts = [ex.submit(run_rf_importance, j) for j in rf_jobs]

        folds = []
        for fu in ffuts:
            r = fu.result(); folds.append(r)
            print("  fold %-22s f%d %6.1fs iou=%.4f thin=%s iters=%d"
                  % (r["arm"], r["fold"], r["seconds"], r["all"]["iou"],
                     None if r["thin"] is None else round(r["thin"]["iou"], 4),
                     r["n_iter"]), flush=True)
        guards = [fu.result() for fu in gfuts]
        for q in guards:
            print("  guardrail %-22s whole=%.5f  ON-SPEC=%.5f (%.0fs)"
                  % (q["arm"], q["mean_fp_frac"], q["mean_fp_frac_on_specimen"],
                     q["seconds"]), flush=True)
        rfs = [fu.result() for fu in rfuts]

    results["folds"] = folds
    results["guardrail"] = {q["arm"]: q for q in guards}
    results["rf_importance"] = {q["arm"]: q for q in rfs}
    results["perm_importance"] = {r["arm"]: r["perm"] for r in folds if "perm" in r}

    # Paired per-frame delta against baseline_17.
    base_pf = {}
    for r in folds:
        if r["arm"] == "baseline_17":
            base_pf.update(r["per_frame"])
    summary = {}
    for arm in arms:
        rs = [r for r in folds if r["arm"] == arm]
        pf = {}
        for r in rs:
            pf.update(r["per_frame"])
        d = np.array([pf[k] - base_pf[k] for k in sorted(pf) if k in base_pf], float)
        thin_ids = {i for i, f in enumerate(frames) if f["thin"]}
        dt = np.array([pf[k] - base_pf[k] for k in sorted(pf)
                       if k in base_pf and k in thin_ids], float)
        gq = results["guardrail"].get(arm, {})
        summary[arm] = dict(
            dim=dim(arm), all_rows=agg(rs, "all"), thin_frames=agg(rs, "thin"),
            crackfree_fp_wholeframe=round(gq.get("mean_fp_frac", float("nan")), 5),
            crackfree_fp_on_specimen=round(gq.get("mean_fp_frac_on_specimen",
                                                  float("nan")), 5),
            paired_frame_delta=dict(
                n=int(len(d)), mean=round(float(d.mean()), 5) if len(d) else None,
                median=round(float(np.median(d)), 5) if len(d) else None,
                sd=round(float(d.std(ddof=1)), 5) if len(d) > 1 else None,
                n_better=int((d > 0).sum()), n_worse=int((d < 0).sum())),
            paired_frame_delta_thin=dict(
                n=int(len(dt)), mean=round(float(dt.mean()), 5) if len(dt) else None,
                n_better=int((dt > 0).sum()), n_worse=int((dt < 0).sum())),
            fit_seconds=round(sum(r["seconds"] for r in rs), 1))
    results["summary"] = summary
    results["wall_clock_seconds"] = round(time.time() - T0, 1)

    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=1)

    print("\n%-22s %4s %-17s %-17s %-9s %s" %
          ("arm", "dim", "IoU all", "IoU thin", "FP spec", "paired dIoU"), flush=True)
    for arm, s in summary.items():
        ar, th = s["all_rows"], s["thin_frames"]
        print("%-22s %4d %.4f+-%.4f  %.4f+-%.4f  %7.3f%%  %+.4f (%d up/%d dn)" % (
            arm, s["dim"], ar["iou"], ar["iou_sd"],
            th["iou"] if th else float("nan"), th["iou_sd"] if th else float("nan"),
            100 * s["crackfree_fp_on_specimen"],
            s["paired_frame_delta"]["mean"], s["paired_frame_delta"]["n_better"],
            s["paired_frame_delta"]["n_worse"]), flush=True)
    print("\nwrote %s  (%.1f s)" % (a.out, results["wall_clock_seconds"]), flush=True)


if __name__ == "__main__":
    main()
