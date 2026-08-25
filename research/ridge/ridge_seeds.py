"""Is the +0.02 IoU on the best-looking arms real, or one lucky MLP seed?

The main sweep fixes MLPClassifier(random_state=0), which is right for making arms
comparable but means every arm's IoU carries an unmeasured seed component. Two arms came
out above baseline by +0.015 to +0.021 -- smaller than the baseline's own fold spread
(+-0.023), so already "indistinguishable" by the brief's rule, but with a lopsided paired
per-frame count (43 frames up / 17 down) that a pure fluke would not usually produce.

Those two facts point in opposite directions, and the tie-breaker is the one number the
main sweep does not have: how much does the whole 5-fold estimate move if you change
nothing but the seed? If re-seeding moves an arm by as much as the gap between arms, the
gap is not a result.

This re-runs GroupKFold(5) for a few arms across MLP seeds 1..4 (seed 0 is already in
ridge_results.json) and reports mean +- sd ACROSS SEEDS of the 5-fold mean IoU. Same rows,
same folds, same columns -- only the seed moves.

Deliberately includes 17_plus_noise9 so the seed spread can be compared against the known
cost of 9 worthless columns.

Usage:
    .venv/bin/python research/ridge/ridge_seeds.py --workers 6
"""

import argparse
import json
import os
import sys
import time

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
sys.path.insert(0, os.path.join(P0, "research", "ridge"))

import ridge_features as RF                                  # noqa: E402
from ridge_eval import load_cache, score, add_noise_cols, NOISE_ARMS   # noqa: E402

ARMS = ["baseline_17", "17_plus_meijering", "17_plus_all_ridge", "17_plus_noise9"]
SEEDS = [1, 2, 3, 4]


def clf(seed):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=seed,
                                         early_stopping=True, n_iter_no_change=8))])


def run(payload):
    arm, cols, seed, fold, tr, te, cache = payload
    X, y, g, frames = load_cache(cache)
    t0 = time.time()
    Xc = add_noise_cols(X[:, cols], NOISE_ARMS.get(arm, 0), len(X))
    del X
    m = clf(seed).fit(Xc[tr], y[tr])
    p = m.predict_proba(Xc[te])[:, 1] > 0.5
    thin = np.array([f["thin"] for f in frames])
    sel = thin[g[te]]
    return dict(arm=arm, seed=seed, fold=fold, all=score(p, y[te]),
                thin=score(p[sel], y[te][sel]) if sel.any() else None,
                seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        "/private/tmp/claude-501/-Users-jiamingzhang-Desktop-APP",
        "6c65cf52-47c9-4de6-a996-f3251ee258ff", "scratchpad", "ridgecache"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(P0, "research", "ridge",
                                                  "ridge_seeds.json"))
    a = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    T0 = time.time()
    X, y, g, frames = load_cache(a.cache)
    all_arms = dict(RF.ARMS)
    all_arms["17_plus_noise9"] = RF.COL_BASE17
    splits = list(GroupKFold(5).split(X, y, groups=g))
    del X

    jobs = [(arm, all_arms[arm], s, f, tr, te, a.cache)
            for arm in ARMS for s in SEEDS
            for f, (tr, te) in enumerate(splits, 1)]
    print("%d fits (%d arms x %d seeds x 5 folds)" % (len(jobs), len(ARMS), len(SEEDS)),
          flush=True)
    del y, g

    from concurrent.futures import ProcessPoolExecutor
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run, jobs):
            rows.append(r)
            print("  %-22s seed=%d f%d %6.1fs iou=%.4f" %
                  (r["arm"], r["seed"], r["fold"], r["seconds"], r["all"]["iou"]),
                  flush=True)

    # Bring seed 0 in from the main sweep so the comparison spans seeds 0..4.
    prev = json.load(open(os.path.join(P0, "research", "ridge", "ridge_results.json")))
    for fr in prev["folds"]:
        if fr["arm"] in ARMS:
            rows.append(dict(arm=fr["arm"], seed=0, fold=fr["fold"], all=fr["all"],
                             thin=fr["thin"], seconds=fr["seconds"]))

    out = {}
    for arm in ARMS:
        per_seed, per_seed_thin = {}, {}
        for s in [0] + SEEDS:
            rs = [r for r in rows if r["arm"] == arm and r["seed"] == s]
            if len(rs) != 5:
                continue
            per_seed[s] = float(np.mean([r["all"]["iou"] for r in rs]))
            tv = [r["thin"]["iou"] for r in rs if r["thin"]]
            per_seed_thin[s] = float(np.mean(tv)) if tv else None
        v = np.array(list(per_seed.values()))
        tv = np.array([x for x in per_seed_thin.values() if x is not None])
        out[arm] = dict(
            per_seed_iou={k: round(x, 4) for k, x in per_seed.items()},
            mean_iou=round(float(v.mean()), 4),
            sd_across_seeds=round(float(v.std(ddof=1)), 4),
            min_iou=round(float(v.min()), 4), max_iou=round(float(v.max()), 4),
            per_seed_iou_thin={k: (round(x, 4) if x is not None else None)
                               for k, x in per_seed_thin.items()},
            mean_iou_thin=round(float(tv.mean()), 4),
            sd_across_seeds_thin=round(float(tv.std(ddof=1)), 4))

    with open(a.out, "w") as fh:
        json.dump(dict(arms=ARMS, seeds=[0] + SEEDS, summary=out,
                       wall_clock_seconds=round(time.time() - T0, 1)), fh, indent=1)

    b = out["baseline_17"]
    print("\n%-22s %-34s %16s %16s" %
          ("arm", "IoU per seed 0..4", "mean+-sd(seed)", "d vs baseline"), flush=True)
    for arm in ARMS:
        d = out[arm]
        seq = " ".join("%.4f" % d["per_seed_iou"][s] for s in sorted(d["per_seed_iou"]))
        print("%-22s %-34s %.4f+-%.4f   %+.4f" %
              (arm, seq, d["mean_iou"], d["sd_across_seeds"],
               d["mean_iou"] - b["mean_iou"]), flush=True)
    print("\nthin-frame IoU, same layout:", flush=True)
    for arm in ARMS:
        d = out[arm]
        print("%-22s %.4f+-%.4f   %+.4f" %
              (arm, d["mean_iou_thin"], d["sd_across_seeds_thin"],
               d["mean_iou_thin"] - b["mean_iou_thin"]), flush=True)
    print("\nwrote %s (%.1f s)" % (a.out, time.time() - T0), flush=True)


if __name__ == "__main__":
    main()
