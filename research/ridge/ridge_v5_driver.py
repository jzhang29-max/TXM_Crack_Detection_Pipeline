"""Do ridge features still help once the labels are the thin core, and is the gain one specimen?

Two questions an earlier run could not finish. It measured +0.0204 IoU from ridge channels
against the WIDE-label baseline, with a 9-random-channel control at -0.0040 -- so the gain was
information rather than model capacity. It never checked whether that gain was carried by a
single specimen group, and the deployed model has since been retrained on labels narrowed to
their dark core (RECIPE thincore_v5), which could make ridge filters redundant OR more useful.

Run detached (`nohup ... &`). Every stage writes its JSON before the next begins, so a sleep or
an interrupt costs one stage, not the run.
"""
import json, os, sys, time
import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(P0, "app", "core"), os.path.join(P0, "code"), HERE):
    sys.path.insert(0, p)
import store as S, pipeline as P                                    # noqa: E402
import ridge_features as RF                                         # noqa: E402
from sklearn.pipeline import Pipeline                               # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402
from sklearn.neural_network import MLPClassifier                    # noqa: E402
from sklearn.model_selection import GroupKFold                      # noqa: E402

PER_CLASS = 8000
N_NOISE = 9
ROWS = os.path.join(HERE, "rows_thincore_cache.npz")     # multi-GB, delete when done
T0 = time.time()


def log(msg):
    print(f"[{(time.time()-T0)/60:6.1f}m] {msg}", flush=True)


def specimen_group(fn):
    f = fn.lower()
    if "hc_" in f or "_hc" in f: return "AM/HC"
    if "wrought" in f: return "wrought"
    if "b2" in f: return "B2"
    if "b3" in f: return "B3"
    return "other"


def stage_rows():
    """Production sampling: crack rows from the narrowed core, negatives from erase + ring."""
    if os.path.exists(ROWS):
        log(f"reusing {os.path.basename(ROWS)}")
        return
    rng = np.random.RandomState(0)
    X, y, g, grp, thin = [], [], [], [], {}
    items = [m for m in S.list_images() if "SELFTEST" not in (m.get("filename") or "")]
    for k, m in enumerate(items, 1):
        iid = m["id"]
        corr = S.load_npy(iid, "correction.npy")
        img = S.load_npy(iid, "img.npy")
        if corr is None or img is None or not (corr == 1).any() or not (corr == 2).any():
            continue
        img01 = np.asarray(img, np.float32)
        core = corr == 1
        narrowed = P.tighten_to_image(iid, core, prune=False)
        if narrowed is not None and narrowed.any():
            core = narrowed
        ring = (corr == 1) & ~core
        ci = np.flatnonzero(core.reshape(-1))
        bi = np.flatnonzero(((corr == 2) | ring).reshape(-1))
        if len(ci) == 0 or len(bi) == 0:
            continue
        nc = min(PER_CLASS, len(ci)); nb = min(PER_CLASS, len(bi))
        idx = np.concatenate([rng.choice(ci, nc, replace=False),
                              rng.choice(bi, nb, replace=False)])
        X.append(RF.sample_rows(img01, idx))
        y.append(np.concatenate([np.ones(nc, bool), np.zeros(nb, bool)]))
        g.append(np.full(nc + nb, iid))
        grp.append(np.full(nc + nb, specimen_group(m.get("filename") or "")))
        _hw, _ndark = RF.thin_half_width(img01, corr)   # returns (half_width|None, n_px)
        thin[iid] = float(_hw) if _hw is not None else -1.0
        log(f"rows {k}/{len(items)}  {(m.get('filename') or '')[22:52]}  "
            f"core {nc} neg {nb}")
        del img01, corr
    np.savez(ROWS[:-4], X=np.concatenate(X), y=np.concatenate(y),
             img=np.concatenate(g), grp=np.concatenate(grp))
    json.dump(thin, open(os.path.join(HERE, "ridge_v5_thin.json"), "w"), indent=1)
    log(f"rows cached: {sum(len(v) for v in y):,}")


def clf():
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64, 32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])


def iou(pred, t):
    tp = int((pred & t).sum()); fp = int((pred & ~t).sum()); fn = int((~pred & t).sum())
    return tp / max(tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def stage_score():
    z = np.load(ROWS)
    X, y, img, grp = z["X"], z["y"], z["img"], z["grp"]
    thin = json.load(open(os.path.join(HERE, "ridge_v5_thin.json")))
    thin_img = {k for k, v in thin.items() if 0 < v <= 3.0}
    rs = np.random.RandomState(7)
    noise = rs.standard_normal((len(y), N_NOISE)).astype(np.float32)
    arms = {"baseline_17": RF.COL_BASE17,
            "17_plus_meijering": RF.COL_BASE17 + RF.COL_MEIJ,
            "17_plus_all_ridge": RF.COL_BASE17 + RF.COL_PACKAGED + RF.COL_HESS,
            "17_plus_noise9": None}
    out = {"n_rows": int(len(y)), "n_images": int(len(set(img.tolist()))),
           "thin_images": len(thin_img),
           "group_counts": {k: int((grp == k).sum()) for k in sorted(set(grp.tolist()))},
           "logo": {}, "kfold": {}}
    def mat(cols):
        # The control is 17 features PLUS 9 random columns -- appending noise to the real
        # features, which is the only way it tests "does extra width help by itself". An
        # earlier version returned `noise` alone, i.e. noise-only, and scored 0.28: a
        # mislabelled arm, not a finding.
        return np.hstack([X[:, RF.COL_BASE17], noise]) if cols is None else X[:, cols]
    def build(cols, tr):
        M = mat(cols); c = clf(); c.fit(M[tr], y[tr]); return c, M
    # leave-one-specimen-group-out
    for gname in sorted(set(grp.tolist())):
        te = grp == gname; tr = ~te
        for arm, cols in arms.items():
            c, M = build(cols, tr)
            p = c.predict_proba(M[te])[:, 1] > 0.5
            i, pr, rc = iou(p, y[te])
            tmask = te & np.isin(img, list(thin_img))
            it = iou(c.predict_proba(M[tmask])[:, 1] > 0.5, y[tmask])[0] if tmask.any() else None
            out["logo"].setdefault(arm, {})[gname] = dict(
                iou=round(i, 4), precision=round(pr, 4), recall=round(rc, 4),
                iou_thin=(round(it, 4) if it is not None else None), n=int(te.sum()))
            log(f"LOGO hold out {gname:<8} {arm:<20} IoU {i:.4f}")
        json.dump(out, open(os.path.join(HERE, "ridge_v5_scores.json"), "w"), indent=1)
    # grouped-by-image, for comparability with the earlier run
    for arm, cols in arms.items():
        ious, thins = [], []
        for tr, te in GroupKFold(5).split(X, groups=img):
            c, M = build(cols, tr)
            ious.append(iou(c.predict_proba(M[te])[:, 1] > 0.5, y[te])[0])
            tmask = np.zeros(len(y), bool); tmask[te] = True
            tmask &= np.isin(img, list(thin_img))
            if tmask.any():
                thins.append(iou(c.predict_proba(M[tmask])[:, 1] > 0.5, y[tmask])[0])
        out["kfold"][arm] = dict(iou=round(float(np.mean(ious)), 4),
                                 iou_sd=round(float(np.std(ious, ddof=1)), 4),
                                 iou_thin=round(float(np.mean(thins)), 4) if thins else None,
                                 folds=[round(v, 4) for v in ious])
        log(f"KFOLD {arm:<20} IoU {out['kfold'][arm]['iou']:.4f} "
            f"+-{out['kfold'][arm]['iou_sd']:.4f}")
        json.dump(out, open(os.path.join(HERE, "ridge_v5_scores.json"), "w"), indent=1)
    log("scores written")


if __name__ == "__main__":
    stage_rows()
    stage_score()
    log("RIDGE_V5_DONE")
