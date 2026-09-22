#!/usr/bin/env python3
"""Does training on the crack-free specimens' labels flatter their false-positive score?

v1 of this failed and is kept at heldout_crackfree_fp.py with its failure recorded. It fitted
a 17-feature-only MLP, which is not the deployed detector: its arms scored 8.35% and 5.43%
mean predicted area on the crack-free specimens against 0.174% for the shipped ensemble, so
neither arm was the model whose contamination was in question. It also ran one fit per arm,
which cannot separate an effect from fit-to-fit variance.

v2 fixes both.

  ARCHITECTURE. Both branches are fitted and written to joblib, then loaded through
  app.core.model.CrackModel and scored with pipeline._score_clean -- the SHIPPED prediction
  path, not a copy of it. A reimplementation is what made v1 meaningless.

  REPEATS. N_SEEDS fits per arm, reported as a distribution with a paired test, not a pair.

  BUDGET. A fixed row budget per arm, subsampled with the arm's own seed. research/
  fp_attribution.json already compares arms this way, and it keeps 10 fits tractable.

  WHAT IS HELD OUT. Every row from a CLEAN_SPECIMENS frame, from BOTH branches. Those frames
  contribute 0 crack px and 95,351,647 not-crack px -- 25.4% of raw background label pixels,
  but only 4.3% of training ROWS after gather_training_data's per-image cap. The row figure
  is the one that describes the model that actually gets fitted.

Emits: out/txm_heldout_crackfree_fp_v2.json

Usage:  python3 code/audit/heldout_crackfree_fp_v2.py [out.json] [n_seeds] [row_budget]
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "app", "core"))
os.chdir(HERE)

from app.core import pipeline as P                                    # noqa: E402
from app.core import store as S                                       # noqa: E402
from app.core import model as M                                       # noqa: E402

N_SEEDS = 5
ROW_BUDGET = 400_000


def is_clean(meta):
    fn = (meta.get("filename") or "") + " " + meta.get("id", "")
    return any(k.lower() in fn.lower() for k in P.CLEAN_SPECIMENS)


def fit_arm(X, y, seed, budget, tmpdir, tag):
    """Fit both branches on a budgeted subsample and return a CrackModel over them."""
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    idx = (np.arange(X.shape[0]) if X.shape[0] <= budget
           else rng.choice(X.shape[0], budget, replace=False))
    Xs, ys = X[idx], y[idx]
    made = {}
    for name, sl in (("17", slice(0, 17)), ("hybrid", slice(0, X.shape[1]))):
        clf = Pipeline([("scaler", StandardScaler(copy=False)),
                        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                              random_state=seed))])
        clf.fit(Xs[:, sl], ys)
        path = os.path.join(tmpdir, f"{tag}_{name}_{seed}.joblib")
        joblib.dump(clf if name == "17"
                    else dict(model=clf, n_features=int(X.shape[1])), path)
        made[name] = path
    return M.CrackModel(path_17=made["17"], path_hybrid=made["hybrid"], ensemble=True), len(idx)


def main(outp="out_heldout_v2.json", n_seeds=N_SEEDS, budget=ROW_BUDGET):
    n_seeds, budget = int(n_seeds), int(budget)
    t0 = time.time()
    print("gathering training rows ...", flush=True)
    X, y, info, groups = P.gather_training_data()
    if X is None:
        raise SystemExit("no labelled data")
    labelled = [m for m in S.list_images()
                if m.get("corrected_crack_px") or m.get("corrected_not_px")]
    clean_idx = {i for i, m in enumerate(labelled) if is_clean(m)}
    keep = ~np.isin(groups, list(clean_idx))
    print(f"rows {X.shape[0]:,} x {X.shape[1]}   clean frames {len(clean_idx)}   "
          f"rows dropped {int((~keep).sum()):,} ({100*(~keep).mean():.1f}%)", flush=True)

    arms = {"with_clean": (X, y), "without_clean": (X[keep], y[keep])}
    out = {k: [] for k in arms}
    with tempfile.TemporaryDirectory() as td:
        for seed in range(n_seeds):
            for tag, (Xa, ya) in arms.items():
                t = time.time()
                mdl, nrows = fit_arm(Xa, ya, seed, budget, td, tag)
                fp, n, detail = P._score_clean(mdl)
                out[tag].append(dict(seed=seed, rows=nrows, mean_fp=fp, n=n, per_frame=detail))
                print(f"  seed {seed}  {tag:14s}  rows {nrows:,}  mean FP "
                      f"{100*fp:.4f}%   ({time.time()-t:.0f}s)", flush=True)
                del mdl

    a = np.array([r["mean_fp"] for r in out["with_clean"]])
    b = np.array([r["mean_fp"] for r in out["without_clean"]])
    from scipy.stats import wilcoxon
    try:
        wp = float(wilcoxon(a, b).pvalue)
    except ValueError:
        wp = float("nan")
    deployed = P.clean_fp_measured(S.model_key(S.registry().get("current")))
    res = dict(
        what="Does training on the crack-free specimens' own labels flatter their "
             "false-positive score? Deployed architecture, repeated fits. Run 2026-09-22.",
        generator="code/audit/heldout_crackfree_fp_v2.py",
        supersedes="code/audit/heldout_crackfree_fp.py (wrong architecture, n=1 per arm)",
        architecture="both branches fitted, dumped to joblib, loaded through "
                     "app.core.model.CrackModel(ensemble=True) and scored with "
                     "pipeline._score_clean -- the shipped prediction path",
        n_seeds=n_seeds, row_budget=budget,
        rows_total=int(X.shape[0]), rows_dropped=int((~keep).sum()),
        rows_dropped_pct=float(100 * (~keep).mean()),
        deployed_reference_mean_fp=(deployed[0] if deployed else None),
        with_clean=dict(mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else None,
                        values=[float(v) for v in a], runs=out["with_clean"]),
        without_clean=dict(mean=float(b.mean()), sd=float(b.std(ddof=1)) if len(b) > 1 else None,
                           values=[float(v) for v in b], runs=out["without_clean"]),
        paired_delta_mean=float((b - a).mean()), wilcoxon_p=wp,
        interpretation=("contamination would make with_clean LOWER than without_clean; a "
                        "higher or indistinguishable with_clean does not support it"),
        seconds=round(time.time() - t0, 1))
    json.dump(res, open(outp, "w"), indent=1)
    print(f"\nwith_clean    {100*a.mean():.4f}% +/- {100*a.std(ddof=1):.4f}")
    print(f"without_clean {100*b.mean():.4f}% +/- {100*b.std(ddof=1):.4f}")
    print(f"paired delta  {100*(b-a).mean():+.4f} pp   Wilcoxon p = {wp:.4f}")
    print(f"deployed reference {100*(deployed[0] if deployed else float('nan')):.4f}%")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main(*(sys.argv[1:4] or ["out_heldout_v2.json"]))
