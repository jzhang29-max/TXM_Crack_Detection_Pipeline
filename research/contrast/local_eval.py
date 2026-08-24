"""Phase 2/3 of the LOCAL / ADAPTIVE contrast-enhancement arm.

Reads the per-image feature samples cached by local_extract.py and, for every
arm:
  * GroupKFold(5) grouped by image -> IoU@0.5 / precision / recall on held-out
    rows, both as a per-fold mean+-std and pooled over all out-of-fold rows;
  * the same numbers restricted to held-out rows from THIN-crack frames;
  * trains on ALL rows and reports the fraction of pixels predicted crack on
    each of the 6 crack-free specimens (every positive there is a false
    positive by construction).

Writes research/contrast/local_results.json only.
"""

import json
import os
import sys
import time

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = os.path.join(P0, "research", "contrast")
CACHE = os.path.join(OUT, "local_cache")
CLEAN_CACHE = os.path.join(OUT, "local_cache_clean")


def make_clf():
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def load_arm(arm, ids):
    X, y, g, thin_row = [], [], [], []
    thin_ids = set()
    for gi, (iid, is_thin) in enumerate(ids):
        with np.load(os.path.join(CACHE, iid + ".npz")) as z:
            x = z["X_" + arm]
            yy = z["y"]
        X.append(x)
        y.append(yy)
        g.append(np.full(len(yy), gi, np.int32))
        thin_row.append(np.full(len(yy), is_thin, bool))
        if is_thin:
            thin_ids.add(iid)
    return (np.concatenate(X), np.concatenate(y).astype(np.int8),
            np.concatenate(g), np.concatenate(thin_row))


def scores(y, p):
    y = y.astype(bool)
    p = p.astype(bool)
    tp = int((y & p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return dict(iou=iou, precision=prec, recall=rec, tp=tp, fp=fp, fn=fn,
                n=int(y.size))


def ms(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return dict(mean=float("nan"), std=float("nan"), n_folds=0)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=0)), n_folds=int(v.size))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(OUT, "local_manifest.json"))
    ap.add_argument("--out", default=os.path.join(OUT, "local_results.json"))
    ap.add_argument("--arms", default="all")
    ap.add_argument("--limit-images", type=int, default=0)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    arms = man["arms"] if args.arms == "all" else args.arms.split(",")
    ids = [(m["id"], bool(m["thin"])) for m in man["images"]]
    if args.limit_images:
        ids = ids[:args.limit_images]
        man = dict(man, images=man["images"][:args.limit_images])
    thin_list = [m["id"] for m in man["images"] if m["thin"]]
    print(f"{len(ids)} images, {len(thin_list)} thin, {len(arms)} arms")

    clean = [(c["specimen"], c["id"]) for c in man["clean"]]

    results = {}
    for arm in arms:
        t0 = time.time()
        X, y, g, thin_row = load_arm(arm, ids)
        print(f"\n=== {arm}  X={X.shape} pos={int(y.sum())} neg={int((y==0).sum())}")

        oof = np.zeros(len(y), np.int8)
        oof_seen = np.zeros(len(y), bool)
        fold_all, fold_thin = [], []
        for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=g)):
            clf = make_clf()
            clf.fit(X[tr], y[tr])
            pred = (clf.predict_proba(X[te])[:, 1] >= 0.5).astype(np.int8)
            oof[te] = pred
            oof_seen[te] = True
            sa = scores(y[te], pred)
            fold_all.append(sa)
            m = thin_row[te]
            st = scores(y[te][m], pred[m]) if m.any() else None
            fold_thin.append(st)
            print(f"  fold{f} all IoU={sa['iou']:.4f} P={sa['precision']:.4f} "
                  f"R={sa['recall']:.4f} | thin " +
                  (f"IoU={st['iou']:.4f} P={st['precision']:.4f} R={st['recall']:.4f} "
                   f"n={st['n']}" if st else "no thin frames held out"))

        assert oof_seen.all()
        pooled_all = scores(y, oof)
        pooled_thin = scores(y[thin_row], oof[thin_row]) if thin_row.any() else None

        # Guardrail: one model on every row, then the crack-free specimens.
        tfit = time.time()
        full = make_clf()
        full.fit(X, y)
        fit_s = time.time() - tfit
        del X, y, g, thin_row

        fp_rows = []
        for spec, iid in clean:
            with np.load(os.path.join(CLEAN_CACHE, iid + ".npz")) as z:
                Xc = z["X_" + arm]
                on_spec = z["on_specimen"]
            pc = (full.predict_proba(Xc)[:, 1] >= 0.5)
            frac = float(pc.mean())
            frac_spec = float(pc[on_spec].mean()) if on_spec.any() else float("nan")
            fp_rows.append(dict(specimen=spec, id=iid, n=int(len(pc)),
                                frac_pred_crack=frac,
                                frac_pred_crack_on_specimen=frac_spec,
                                on_specimen_frac=float(on_spec.mean())))
            print(f"  clean {spec:32s} FP={frac*100:6.3f}%  "
                  f"(on-specimen {frac_spec*100:6.3f}%)")
            del Xc

        fps = [r["frac_pred_crack"] for r in fp_rows]
        fps_spec = [r["frac_pred_crack_on_specimen"] for r in fp_rows]
        results[arm] = dict(
            params=man["arm_params"].get(arm, {}), rescale=man["rescale"].get(arm),
            all_rows=dict(
                per_fold=fold_all,
                iou=ms([s["iou"] for s in fold_all]),
                precision=ms([s["precision"] for s in fold_all]),
                recall=ms([s["recall"] for s in fold_all]),
                pooled=pooled_all),
            thin_rows=dict(
                per_fold=fold_thin,
                iou=ms([s["iou"] for s in fold_thin if s]),
                precision=ms([s["precision"] for s in fold_thin if s]),
                recall=ms([s["recall"] for s in fold_thin if s]),
                pooled=pooled_thin),
            crack_free=dict(per_specimen=fp_rows,
                            mean_frac_pred_crack=float(np.mean(fps)),
                            max_frac_pred_crack=float(np.max(fps)),
                            mean_frac_pred_crack_on_specimen=float(np.nanmean(fps_spec)),
                            max_frac_pred_crack_on_specimen=float(np.nanmax(fps_spec))),
            seconds=dict(eval_total=round(time.time() - t0, 1),
                         full_fit=round(fit_s, 1)))
        a = results[arm]
        print(f"  MEAN all IoU={a['all_rows']['iou']['mean']:.4f}"
              f"+-{a['all_rows']['iou']['std']:.4f}  "
              f"thin pooled IoU={pooled_thin['iou'] if pooled_thin else float('nan'):.4f}  "
              f"cleanFP={a['crack_free']['mean_frac_pred_crack']*100:.3f}%  "
              f"({results[arm]['seconds']['eval_total']}s)")

        json.dump(dict(meta=dict(
            n_images=len(ids), thin_frames=thin_list,
            max_per_class=man["max_per_class"], clean_sample=man["clean_sample"],
            thin_max_halfwidth=man["thin_max_halfwidth"], lcn_eps=man["lcn_eps"],
            extract_seconds=man.get("extract_seconds", 0),
            halfwidth_px={m["id"]: m["median_halfwidth_px"] for m in man["images"]}),
            arms=results), open(args.out, "w"), indent=1)

    print("\ndone")


if __name__ == "__main__":
    main()
