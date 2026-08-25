"""The operating-point study. Reads only the histogram cache.

Metrics, and what each one can and cannot say:

  ON THE OWNER'S CORRECTIONS (h_pos / h_neg). Sparse labels: corr==0 is "no opinion", so
  every rate here is conditional on pixels the owner judged. 99.64% of corr==1 already sits
  above 0.50, because "flip region" mints a positive label by accepting a blob the model
  found at 0.50 -- so recall here is near-saturated BY CONSTRUCTION and this axis cannot
  reward a lower threshold. Reported anyway, because it is what the earlier 0.43 note used.

  ON DENSE GT (h_gt1 / h_gt0). Four B2 frames, every pixel labelled, drawn in another tool
  and NOT used to train or score v4 (pipeline.py header). True IoU is computable. The four
  IMAGES are still ordinary training images via their corrections, so a threshold picked
  here is optimistic -- hence leave-one-image-out.

  ON CRACK-FREE SPECIMEN (h_spec). No labels needed: the owner asserts no crack, so anything
  on-specimen above t is a false positive. The only axis that reads material the labelled
  distribution does not cover, and the one that caught two bad models before.
"""
import sys, os, json, glob
import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
OUT = os.path.join(P0, "research", "oppoint")
CACHE = os.path.join(OUT, "cache")
NB = 2000
GRID = np.round(np.arange(0.20, 0.8001, 0.01), 4)
CENTERS = (np.arange(NB) + 0.5) / NB


def idx(t):
    return int(round(float(t) * NB))


def above(h, t):
    return int(h[idx(t):].sum())


def group_of(fn):
    n = fn.lower()
    if "wrought" in n:
        return "Wrought"
    if "hc_316l" in n:
        return "AM/HC"
    if "_b3_" in n or "b3_" in n:
        return "B3"
    return "B2"


def load():
    inv = json.load(open(os.path.join(OUT, "inventory.json")))
    imgs = []
    for r in inv["rows"]:
        p = os.path.join(CACHE, r["iid"] + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["iid"], d["fn"] = r["iid"], r["filename"]
        d["clean"], d["grp"] = r["clean"], group_of(r["filename"])
        imgs.append(d)
    return imgs


# ---------------------------------------------------------------- metric cores
def corr_metrics(sel, t):
    """pooled recall / fpr / IoU on the owner's labelled pixels."""
    tp = sum(above(d["h_pos"], t) for d in sel if "h_pos" in d)
    fp = sum(above(d["h_neg"], t) for d in sel if "h_neg" in d)
    np_ = sum(int(d["n_pos"]) for d in sel if "n_pos" in d)
    nn = sum(int(d["n_neg"]) for d in sel if "n_neg" in d)
    fn = np_ - tp
    return dict(recall=tp / max(np_, 1), fpr=fp / max(nn, 1),
                iou=tp / max(tp + fp + fn, 1), prec=tp / max(tp + fp, 1),
                tp=tp, fp=fp, fn=fn, n_pos=np_, n_neg=nn)


def gt_metrics(sel, t):
    """true IoU on densely-labelled frames."""
    g = [d for d in sel if "h_gt1" in d]
    tp = sum(above(d["h_gt1"], t) for d in g)
    fp = sum(above(d["h_gt0"], t) for d in g)
    n1 = sum(int(d["n_gt1"]) for d in g)
    fn = n1 - tp
    return dict(recall=tp / max(n1, 1), iou=tp / max(tp + fp + fn, 1),
                prec=tp / max(tp + fp, 1), tp=tp, fp=fp, fn=fn, n=len(g))


def clean_fp(sel, t):
    """on-specimen and whole-frame FP fraction over crack-free specimens."""
    c = [d for d in sel if d["clean"] and "h_spec" in d]
    on = sum(above(d["h_spec"], t) for d in c)
    ns = sum(int(d["n_spec"]) for d in c)
    allp = sum(above(d["h_all"], t) for d in c)
    na = sum(int(d["npix"]) for d in c)
    off = sum(above(d["h_off"], t) for d in c)
    no = na - ns
    return dict(on_spec=on / max(ns, 1), whole=allp / max(na, 1),
                off_spec=off / max(no, 1), n=len(c))


def folds(imgs, k=5, seed=0):
    """k folds GROUPED BY IMAGE: an image is wholly in train or wholly in test."""
    ids = sorted(d["iid"] for d in imgs)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(ids))
    return [[ids[j] for j in order[i::k]] for i in range(k)]


def best_t(sel, metric="iou", scorer=corr_metrics):
    best, bt = -1, 0.5
    for t in GRID:
        v = scorer(sel, t)[metric]
        if v > best:
            best, bt = v, t
    return bt, best


def sec(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def main():
    imgs = load()
    lab = [d for d in imgs if "h_pos" in d and int(d["n_pos"]) > 0]
    gtim = [d for d in imgs if "h_gt1" in d]
    print(f"images={len(imgs)} with_pos_labels={len(lab)} dense_gt={len(gtim)} "
          f"clean={sum(d['clean'] for d in imgs)}")

    res = {}

    # ---- 1. global sweep, in-sample ---------------------------------------
    sec("1. GLOBAL SWEEP (pooled, in-sample) -- corrections | dense GT | crack-free FP")
    print(f"{'t':>5} {'recall':>7} {'fpr':>8} {'IoU':>7} {'prec':>7} | "
          f"{'gtIoU':>7} {'gtRec':>7} {'gtPrec':>7} | {'onSpecFP':>9} {'wholeFP':>8}")
    curve = []
    for t in GRID:
        c, g, f = corr_metrics(lab, t), gt_metrics(gtim, t), clean_fp(imgs, t)
        curve.append(dict(t=float(t), corr=c, gt=g, clean=f))
        star = " <<< SHIPPED" if abs(t - 0.50) < 1e-9 else ""
        print(f"{t:5.2f} {c['recall']:7.4f} {c['fpr']:8.5f} {c['iou']:7.4f} "
              f"{c['prec']:7.4f} | {g['iou']:7.4f} {g['recall']:7.4f} {g['prec']:7.4f} | "
              f"{f['on_spec']*100:8.3f}% {f['whole']*100:7.3f}%{star}")
    res["curve"] = curve

    for nm, sc in (("corrections IoU", lambda t: corr_metrics(lab, t)["iou"]),
                   ("dense-GT IoU", lambda t: gt_metrics(gtim, t)["iou"])):
        vals = [sc(t) for t in GRID]
        b = int(np.argmax(vals))
        at50 = sc(0.50)
        print(f"\n{nm}: argmax t={GRID[b]:.2f} -> {vals[b]:.4f};  "
              f"at 0.50 -> {at50:.4f};  gain={vals[b]-at50:+.4f}")

    # ---- 2. held-out threshold selection ----------------------------------
    sec("2. HELD OUT, GROUPED BY IMAGE -- t chosen on train images, scored on test")
    for scorer, gname, pool in (("corr", "corrections IoU", lab),
                                ("gt", "dense-GT IoU", gtim)):
        fn = corr_metrics if scorer == "corr" else gt_metrics
        print(f"\n--- {gname} ---")
        rows = []
        for seed in range(5):
            fl = folds(pool, k=min(5, len(pool)), seed=seed)
            for i, te in enumerate(fl):
                tr = [d for d in pool if d["iid"] not in te]
                tes = [d for d in pool if d["iid"] in te]
                if not tr or not tes:
                    continue
                bt, _ = best_t(tr, "iou", fn)
                s_sel = fn(tes, bt)["iou"]
                s_50 = fn(tes, 0.50)["iou"]
                rows.append((seed, i, bt, s_sel, s_50, s_sel - s_50))
        print(f"{'seed':>4} {'fold':>4} {'t*':>5} {'IoU@t*':>8} {'IoU@0.50':>9} {'delta':>8}")
        for r in rows:
            print(f"{r[0]:4d} {r[1]:4d} {r[2]:5.2f} {r[3]:8.4f} {r[4]:9.4f} {r[5]:+8.4f}")
        ts = np.array([r[2] for r in rows]); dl = np.array([r[5] for r in rows])
        print(f"  t* : mean {ts.mean():.3f}  sd {ts.std():.3f}  min {ts.min():.2f} "
              f"max {ts.max():.2f}")
        print(f"  held-out delta vs 0.50: mean {dl.mean():+.5f}  sd {dl.std():.5f}  "
              f"min {dl.min():+.4f}  max {dl.max():+.4f}  wins {int((dl>0).sum())}/{len(dl)}")
        res[f"heldout_{scorer}"] = [dict(seed=int(r[0]), fold=int(r[1]), t=float(r[2]),
                                         iou_t=r[3], iou_50=r[4], delta=r[5]) for r in rows]

    # ---- 3. per-image spread of the oracle threshold ----------------------
    sec("3. IS ONE GLOBAL THRESHOLD THE RIGHT SHAPE? per-image oracle (NEEDS LABELS)")
    print(f"{'t_oracle':>8} {'IoU@t*':>8} {'IoU@.50':>8} {'gain':>7} {'crack%':>7}  image")
    per = []
    for d in sorted(lab, key=lambda d: -int(d["n_pos"])):
        bt, bv = best_t([d], "iou", corr_metrics)
        v50 = corr_metrics([d], 0.50)["iou"]
        frac = int(d["n_pos"]) / max(int(d["npix"]), 1) * 100
        per.append(dict(iid=d["iid"], fn=d["fn"], t=float(bt), iou_t=bv, iou_50=v50,
                        gain=bv - v50, crackpct=frac, grp=d["grp"]))
        print(f"{bt:8.2f} {bv:8.4f} {v50:8.4f} {bv-v50:+7.4f} {frac:6.2f}%  "
              f"{d['fn'][:44]}")
    res["per_image"] = per
    to = np.array([p["t"] for p in per])
    print(f"\nper-image oracle t: mean {to.mean():.3f} sd {to.std():.3f} "
          f"median {np.median(to):.2f} min {to.min():.2f} max {to.max():.2f}")
    print(f"oracle mean per-image IoU gain over 0.50: "
          f"{np.mean([p['gain'] for p in per]):+.4f}  (UPPER BOUND, uses labels)")

    # ---- 4. per-specimen-group -------------------------------------------
    sec("4. PER-SPECIMEN-GROUP threshold, held out by image within group")
    for g in sorted(set(d["grp"] for d in lab)):
        sub = [d for d in lab if d["grp"] == g]
        bt, bv = best_t(sub, "iou", corr_metrics)
        v50 = corr_metrics(sub, 0.50)["iou"]
        print(f"{g:>8}: n={len(sub):>2}  in-sample t*={bt:.2f} IoU {bv:.4f} "
              f"vs 0.50 {v50:.4f} ({bv-v50:+.4f})")
    print("\nheld out (leave-one-image-out inside each group, t from the group's others):")
    tot_sel = tot_50 = 0.0
    n = 0
    for g in sorted(set(d["grp"] for d in lab)):
        sub = [d for d in lab if d["grp"] == g]
        if len(sub) < 2:
            continue
        ds = []
        for d in sub:
            tr = [x for x in sub if x["iid"] != d["iid"]]
            bt, _ = best_t(tr, "iou", corr_metrics)
            a = corr_metrics([d], bt)["iou"]; b = corr_metrics([d], 0.50)["iou"]
            ds.append(a - b); tot_sel += a; tot_50 += b; n += 1
        print(f"{g:>8}: mean per-image delta {np.mean(ds):+.5f} "
              f"(sd {np.std(ds):.5f}, wins {int((np.array(ds)>0).sum())}/{len(ds)})")
    print(f"overall held-out per-specimen-group: mean IoU {tot_sel/max(n,1):.4f} "
          f"vs 0.50 {tot_50/max(n,1):.4f} ({(tot_sel-tot_50)/max(n,1):+.5f})")

    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
