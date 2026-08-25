"""Why are the ridge channels near chance? Test the label-geometry explanation.

ridge_diag.py found every ridge channel sits at pooled AUC 0.50-0.62, below `intensity`
at 0.64. There are two very different reasons that could happen, and they have opposite
implications:

  (a) The filters do not work on this imagery at all -- too noisy, wrong scale, bug.
  (b) The filters work fine on the thin dark CORE of a crack, but the owner's label is not
      the core. code/txm_features.py's own docstring is explicit that the pixel classifier
      exists because "the real crack extent is 18-31%" while an Otsu darkness mask covers
      only ~1.1-1.3% -- i.e. a label here is the broad graded DAMAGED ZONE, not a
      curvilinear line. A vesselness filter is a thin-tube detector. If the target is a
      wide region, a thin-tube detector is answering a different question, and no amount
      of it being correct about tubes will show up as IoU.

The test: split each frame's crack rows into the darkest quintile (the "core", where a
ridge filter should fire) and the rest (the outer damaged zone), then score each channel
CORE-vs-not-crack and OUTER-vs-not-crack separately. Under (a) both stay at chance. Under
(b) the ridge channels jump on the core and collapse on the outer zone, while `intensity`
degrades gently across both.

This distinguishes "ridge filters are useless" from "ridge filters answer a question this
label set is not asking" -- which is the difference between a dead end and a note about
what would have to change first.

Usage:
    .venv/bin/python research/ridge/ridge_core_vs_zone.py
"""

import argparse
import json
import os
import sys

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
sys.path.insert(0, os.path.join(P0, "research", "ridge"))

import ridge_features as RF          # noqa: E402
from ridge_eval import load_cache    # noqa: E402
from ridge_diag import auc           # noqa: E402

CORE_PCT = 20.0     # darkest fifth of each frame's crack rows, matching the thin-frame rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        "/private/tmp/claude-501/-Users-jiamingzhang-Desktop-APP",
        "6c65cf52-47c9-4de6-a996-f3251ee258ff", "scratchpad", "ridgecache"))
    ap.add_argument("--out", default=os.path.join(P0, "research", "ridge",
                                                  "ridge_core_vs_zone.json"))
    a = ap.parse_args()

    X, y, g, frames = load_cache(a.cache)
    i_int = RF.FEATURE_NAMES.index("intensity")

    # Per FRAME, split that frame's crack rows at its own 20th intensity percentile. Done
    # per frame rather than globally because absolute darkness differs between frames and
    # a global cut would just re-select the darkest frames.
    core = np.zeros(len(y), bool)
    for i in range(len(frames)):
        s = (g == i) & y
        if not s.any():
            continue
        v = X[s, i_int]
        cut = np.percentile(v, CORE_PCT)
        idx = np.flatnonzero(s)
        core[idx[v <= cut]] = True
    outer = y & ~core
    neg = ~y
    print("crack rows %d -> core %d (%.1f%%) outer %d | not-crack rows %d"
          % (int(y.sum()), int(core.sum()), 100 * core.sum() / max(y.sum(), 1),
             int(outer.sum()), int(neg.sum())), flush=True)

    cols = RF.COL_BASE17 + list(range(RF.N_FEATURES, RF.N_COLS))
    out = {}
    for c in cols:
        name = RF.ALL_NAMES[c]
        x = X[:, c].astype(np.float64)
        sc = np.concatenate([x[core], x[neg]])
        yc = np.concatenate([np.ones(core.sum(), bool), np.zeros(neg.sum(), bool)])
        so = np.concatenate([x[outer], x[neg]])
        yo = np.concatenate([np.ones(outer.sum(), bool), np.zeros(neg.sum(), bool)])
        out[name] = dict(auc_core=round(float(auc(sc, yc)), 4),
                         auc_outer=round(float(auc(so, yo)), 4),
                         auc_all=round(float(auc(x, y)), 4))
        out[name]["core_minus_outer"] = round(out[name]["auc_core"]
                                             - out[name]["auc_outer"], 4)

    with open(a.out, "w") as fh:
        json.dump(dict(core_pct=CORE_PCT, n_core=int(core.sum()),
                       n_outer=int(outer.sum()), n_neg=int(neg.sum()),
                       channels=out), fh, indent=1)

    print("\n%-18s %8s %8s %8s %9s" %
          ("channel", "CORE", "OUTER", "all", "core-out"), flush=True)
    print("-" * 58, flush=True)
    for c in cols:
        n = RF.ALL_NAMES[c]
        d = out[n]
        if c == RF.N_FEATURES:
            print("-" * 14 + " ADDED RIDGE CHANNELS " + "-" * 22, flush=True)
        print("%-18s %8.4f %8.4f %8.4f %+9.4f"
              % (n, d["auc_core"], d["auc_outer"], d["auc_all"],
                 d["core_minus_outer"]), flush=True)
    print("\nwrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
