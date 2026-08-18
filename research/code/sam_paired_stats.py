"""
Paired per-image statistics for the SAM comparison, and an honest statement of
what n=4 can and cannot support.

This exists because the headline claim -- "SAM features + our 17 beats our 17
alone by +0.05 mean IoU" -- is a mean over FOUR images. Four. Reporting a
difference that size from four paired observations without saying what the
power is would be the same class of mistake this project has already made
twice (accepting flatfielding and the curvilinearity gate on evidence that did
not support them).

Reports, for each SAM condition against the 17-feature baseline at the matched
sampling budget:
  - per-image paired deltas and their spread
  - exact paired sign test (the only test with any resolution at n=4)
  - Wilcoxon signed-rank, WITH its floor stated: at n=4 the smallest attainable
    two-sided p is 0.125, so it cannot reach 0.05 no matter how large the effect
  - a PIXEL-WEIGHTED mean as well as the unweighted one, because LARGE_343_75 is
    23.5 MP against 2.9 MP for the other three and the unweighted mean
    under-weights it 8x

Usage:
    python3 sam_paired_stats.py
"""

import json
import os
import sys
from itertools import combinations

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM_DIR = os.path.join(PROJECT_DIR, "results", "sam")
OUT = os.path.join(SAM_DIR, "paired_stats.json")

MEGAPIXELS = {"333_75_um_zoom": 1671 * 1706 / 1e6, "336_25": 1688 * 1693 / 1e6,
              "338_13": 1706 * 1693 / 1e6, "LARGE_343_75": 3691 * 6367 / 1e6}


def load_runs():
    merged = {}
    for fn in ["huge_gray.json", "huge_gray_fixed.json", "huge_gray_oracle32.json"]:
        p = os.path.join(SAM_DIR, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        for cond, r in d.get("results", {}).items():
            if r.get("mean_iou") is None and cond in merged:
                continue
            merged[cond] = r
    return merged


def iou_map(rows):
    return {r["image"]: r["iou"] for r in rows
            if isinstance(r.get("iou"), (int, float)) and np.isfinite(r["iou"])}


def sign_test_exact(deltas):
    """Exact two-sided binomial sign test. At n=4 the best attainable p is 0.125."""
    d = [x for x in deltas if x != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    k = sum(1 for x in d if x > 0)
    # two-sided: P(X <= min(k, n-k)) + P(X >= max(k, n-k)) under p=0.5
    from math import comb
    lo = min(k, n - k)
    p = 2.0 * sum(comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(p, 1.0), k, n


def main():
    runs = load_runs()
    bp = os.path.join(SAM_DIR, "baseline_pixel17_loio_n20000.json")
    if not os.path.exists(bp):
        sys.exit("missing matched-budget baseline; run baseline_loio_for_sam.py")
    with open(bp) as f:
        base = json.load(f)
    b = iou_map(base["rows"])
    budget = base["n_per_class"]

    print(f"Baseline: 17 features + MLP(64,32), LOIO, n_per_class={budget}")
    print(f"  per image: " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(b.items())))
    print(f"  unweighted mean {np.mean(list(b.values())):.4f}")
    wsum = sum(MEGAPIXELS[k] for k in b)
    print(f"  PIXEL-WEIGHTED mean {sum(b[k]*MEGAPIXELS[k] for k in b)/wsum:.4f}")
    print()

    out = {}
    order = ["embed_plus17", "embed_mlp", "embed_lr", "amg_relaxed",
             "amg_relaxed_oracle", "amg_tiled", "amg_whole", "amg_oracle",
             "pts_oracle", "pts_oracle_group", "box_oracle", "grid_points"]
    for cond in order:
        r = runs.get(cond)
        if not r or not r.get("rows"):
            continue
        s = iou_map(r["rows"])
        shared = sorted(set(s) & set(b))
        if len(shared) < 2:
            continue
        d = np.array([s[k] - b[k] for k in shared])
        p_sign, k, n = sign_test_exact(d)

        wl = None
        try:
            from scipy.stats import wilcoxon
            if len(d) >= 3 and np.any(d != 0):
                wl = float(wilcoxon(d).pvalue)
        except Exception:
            pass

        w = sum(MEGAPIXELS[x] for x in shared)
        uw = float(np.mean([s[x] for x in shared]))
        pw = float(sum(s[x] * MEGAPIXELS[x] for x in shared) / w)
        buw = float(np.mean([b[x] for x in shared]))
        bpw = float(sum(b[x] * MEGAPIXELS[x] for x in shared) / w)

        out[cond] = dict(images=shared, deltas={x: float(s[x] - b[x]) for x in shared},
                         mean_delta=float(d.mean()), sd_delta=float(d.std(ddof=1)) if len(d) > 1 else None,
                         wins=int((d > 0).sum()), n=len(d),
                         p_sign_exact=p_sign, p_wilcoxon=wl,
                         cond_mean_unweighted=uw, cond_mean_pixelweighted=pw,
                         base_mean_unweighted=buw, base_mean_pixelweighted=bpw,
                         flips_under_pixel_weighting=bool((uw > buw) != (pw > bpw)))

        print(f"[{cond}]  n={len(d)}")
        print("   deltas: " + "  ".join(f"{x}={s[x]-b[x]:+.3f}" for x in shared))
        print(f"   mean delta {d.mean():+.4f}"
              + (f"  sd {d.std(ddof=1):.4f}" if len(d) > 1 else ""))
        print(f"   wins {int((d>0).sum())}/{len(d)}   exact sign-test p={p_sign:.4f}"
              + (f"   wilcoxon p={wl:.4f}" if wl is not None else ""))
        print(f"   unweighted: {uw:.4f} vs base {buw:.4f}   "
              f"PIXEL-WEIGHTED: {pw:.4f} vs base {bpw:.4f}"
              + ("   ** CONCLUSION FLIPS **" if out[cond]["flips_under_pixel_weighting"] else ""))
        print()

    print("=" * 72)
    print("POWER CEILING. With 4 paired images the exact sign test can only")
    print("reach p=0.125 (a 4-0 sweep), and Wilcoxon's floor at n=4 is also")
    print("0.125 two-sided. NO result here can be significant at 0.05. A 4-0")
    print("sweep is the strongest evidence this dataset can produce, and it is")
    print("suggestive, not conclusive. Any claim of significance would be false.")
    print("=" * 72)

    with open(OUT, "w") as f:
        json.dump(dict(baseline=dict(rows=base["rows"], n_per_class=budget),
                       megapixels=MEGAPIXELS, comparisons=out,
                       power_note="n=4 paired; exact sign test and Wilcoxon both "
                                  "floor at p=0.125 two-sided, so significance at "
                                  "0.05 is unreachable by construction."), f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
