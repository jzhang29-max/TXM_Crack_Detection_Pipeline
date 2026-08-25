"""Is ONE GLOBAL threshold the right shape? Global vs per-group vs per-image.

A per-image rule is only deployable if it needs no labels to pick the threshold for a new
frame. So each candidate here derives its threshold from the probability map alone, and its
ONE free parameter is chosen on training images and scored on held-out images.

  global        one constant t, chosen on train images
  per-group     one constant per specimen group, chosen on that group's train images
  percentile    t_i = the q-th percentile of image i's on-specimen probabilities; q on train
  otsu          t_i = Otsu split of image i's on-specimen probability histogram (no parameter)
  otsu+shift    Otsu plus a constant offset chosen on train images
  oracle        t_i maximising image i's own IoU -- NOT DEPLOYABLE, upper bound only
"""
import sys, os, json
import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "research", "oppoint"))
os.chdir(P0)
from analyze import (load, GRID, NB, CENTERS, above, idx, corr_metrics, gt_metrics,
                     best_t, folds)

OUT = os.path.join(P0, "research", "oppoint")
QGRID = np.round(np.arange(90.0, 99.96, 0.25), 3)


def pct_thresh(d, q, key="h_spec"):
    """q-th percentile of image d's probabilities, from its histogram."""
    h = d.get(key, d["h_all"]).astype(np.float64)
    c = np.cumsum(h) / max(h.sum(), 1)
    return float(CENTERS[int(np.searchsorted(c, q / 100.0))])


def otsu_thresh(d, key="h_spec"):
    """Otsu on the probability histogram -- exact from bins."""
    h = d.get(key, d["h_all"]).astype(np.float64)
    tot = h.sum()
    if tot <= 0:
        return 0.5
    w0 = np.cumsum(h) / tot
    m = np.cumsum(h * CENTERS) / tot
    mt = m[-1]
    denom = w0 * (1 - w0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m) ** 2 / denom
    sb[~np.isfinite(sb)] = -1
    return float(CENTERS[int(np.argmax(sb))])


def score_per_image(sel, tmap, scorer=corr_metrics):
    """pooled metric where each image uses its own threshold."""
    tp = fp = fn = 0
    for d in sel:
        m = scorer([d], tmap[d["iid"]])
        tp += m["tp"]; fp += m["fp"]; fn += m["fn"]
    return tp / max(tp + fp + fn, 1)


def clip(t):
    return float(min(0.80, max(0.20, round(t, 2))))


def main():
    imgs = load()
    lab = [d for d in imgs if "h_pos" in d and int(d["n_pos"]) > 0]

    # precompute per-image rule thresholds
    otsu = {d["iid"]: clip(otsu_thresh(d)) for d in imgs}
    pcts = {q: {d["iid"]: clip(pct_thresh(d, q)) for d in imgs} for q in QGRID}
    print("Otsu-on-probability per-image threshold: mean %.3f sd %.3f min %.2f max %.2f"
          % (np.mean(list(otsu.values())), np.std(list(otsu.values())),
             min(otsu.values()), max(otsu.values())))
    ex = sorted(otsu.items(), key=lambda kv: kv[1])[:3]

    print("\n" + "=" * 78)
    print("HELD OUT BY IMAGE, 5 folds x 5 seeds -- corrections IoU on test images")
    print("=" * 78)
    rows = {k: [] for k in ("global", "per-group", "percentile", "otsu", "otsu+shift",
                            "fixed0.50", "oracle")}
    for seed in range(5):
        for te in folds(lab, k=5, seed=seed):
            tr = [d for d in lab if d["iid"] not in te]
            tes = [d for d in lab if d["iid"] in te]
            if not tr or not tes:
                continue
            # fixed shipped
            rows["fixed0.50"].append(corr_metrics(tes, 0.50)["iou"])
            # global tuned on train
            gt_, _ = best_t(tr, "iou", corr_metrics)
            rows["global"].append(corr_metrics(tes, gt_)["iou"])
            # per specimen group, tuned on train within group (fall back to global)
            gmap = {}
            for g in set(d["grp"] for d in tes):
                sub = [d for d in tr if d["grp"] == g]
                gmap[g] = best_t(sub, "iou", corr_metrics)[0] if sub else gt_
            rows["per-group"].append(score_per_image(tes, {d["iid"]: gmap[d["grp"]]
                                                           for d in tes}))
            # percentile rule: q chosen on train
            bq, bv = None, -1
            for q in QGRID:
                v = score_per_image(tr, pcts[q])
                if v > bv:
                    bq, bv = q, v
            rows["percentile"].append(score_per_image(tes, pcts[bq]))
            # otsu, no parameter
            rows["otsu"].append(score_per_image(tes, otsu))
            # otsu + constant shift chosen on train
            bs, bv = 0.0, -1
            for s in np.arange(-0.20, 0.301, 0.02):
                v = score_per_image(tr, {k: clip(x + s) for k, x in otsu.items()})
                if v > bv:
                    bs, bv = s, v
            rows["otsu+shift"].append(
                score_per_image(tes, {k: clip(x + bs) for k, x in otsu.items()}))
            # oracle per image (uses test labels -- upper bound)
            rows["oracle"].append(score_per_image(
                tes, {d["iid"]: best_t([d], "iou", corr_metrics)[0] for d in tes}))

    base = np.array(rows["fixed0.50"])
    print(f"{'rule':>12} {'mean IoU':>9} {'sd':>7} {'vs 0.50':>9} {'wins':>7}  deployable")
    order = ["fixed0.50", "global", "per-group", "percentile", "otsu", "otsu+shift", "oracle"]
    summary = {}
    for k in order:
        v = np.array(rows[k])
        d = v - base
        dep = "NO (labels)" if k == "oracle" else "yes"
        print(f"{k:>12} {v.mean():9.4f} {v.std():7.4f} {d.mean():+9.4f} "
              f"{int((d>1e-9).sum()):4d}/{len(d):<3}  {dep}")
        summary[k] = dict(mean=float(v.mean()), sd=float(v.std()),
                          delta=float(d.mean()), wins=int((d > 1e-9).sum()), n=len(d))

    # ---- the same three rules on DENSE GT, leave-one-image-out ------------
    print("\n" + "=" * 78)
    print("SAME QUESTION ON DENSE GT (4 frames, leave-one-out) -- true IoU")
    print("=" * 78)
    gtim = [d for d in imgs if "h_gt1" in d]
    print(f"{'frame':>16} {'t_otsu':>7} {'t_p99':>7} {'IoU@.50':>8} {'IoU@otsu':>9} "
          f"{'IoU@.65':>8} {'oracle t':>9} {'oracle':>7}")
    for d in gtim:
        o = otsu[d["iid"]]
        p99 = clip(pct_thresh(d, 99.0))
        bt, bv = best_t([d], "iou", gt_metrics)
        print(f"{str(d.get('gt_stem','?')):>16} {o:7.2f} {p99:7.2f} "
              f"{gt_metrics([d],0.50)['iou']:8.4f} {gt_metrics([d],o)['iou']:9.4f} "
              f"{gt_metrics([d],0.65)['iou']:8.4f} {bt:9.2f} {bv:7.4f}")

    json.dump(summary, open(os.path.join(OUT, "shape.json"), "w"), indent=2)
    print("\nwrote shape.json")


if __name__ == "__main__":
    main()
