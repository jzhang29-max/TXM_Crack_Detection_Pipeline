"""Full-frame predictions for every arm, on frames each arm's model did NOT train on.

Two things IoU cannot show and this measures:

  WIDTH   median half-width along the centreline of the predicted mask
          (distance_transform_edt sampled on skeletonize), against the same measure on
          the painted stroke and on the dark core of the same frame.
  FALSE POSITIVES  positive rate on the six confirmed crack-free specimens, restricted
          to ON-SPECIMEN pixels via pipeline.specimen_support. Whole-frame rate is
          dominated by off-specimen background and inverts the conclusion.

One feature pass per frame serves all four arms: the 273 columns are assembled once per
block and every arm's two members predict on them. The fold model used for a frame is
always the one whose training side excluded that frame (folds.json).
"""
import json
import os
import sys
import time
import warnings

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
import store as S          # noqa: E402
import pipeline as P       # noqa: E402
import model as M          # noqa: E402
from txm_features import compute_feature_stack   # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = os.path.join(P0, "research", "thinlabels")
ARMS = ["baseline", "thin_ring_neg", "thin_plus_margin", "core_dilate3",
        "ring_neg_forced"]
BAND = 128
N17 = 17


def load_fits(fold):
    import joblib
    d = os.path.join(OUT, "models")
    return {a: (joblib.load(os.path.join(d, f"{a}_f{fold}_17.joblib")),
                joblib.load(os.path.join(d, f"{a}_f{fold}_273.joblib"))) for a in ARMS}


def predict_full(iid, fits, progress_every=20):
    """{arm: prob map} for the whole frame."""
    img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
    H, W = img.shape
    f17 = compute_feature_stack(img)
    coords, embs = M.read_emb(S.path(iid, "emb.npz"))
    out = {a: np.zeros((H, W), np.float32) for a in ARMS}
    nb = 0
    t0 = time.time()
    for b0 in range(0, H, BAND):
        b1 = min(b0 + BAND, H)
        for c0 in range(0, W, M.TILE):
            c1 = min(c0 + M.TILE, W)
            rr = np.repeat(np.arange(b0, b1), c1 - c0)
            cc = np.tile(np.arange(c0, c1), b1 - b0)
            X = np.concatenate([np.asarray(f17[rr, cc, :], np.float32),
                                M.emb_rows(coords, embs, rr, cc)], axis=1)
            for a in ARMS:
                m17, m273 = fits[a]
                p = 0.5 * (m17.predict_proba(X[:, :N17])[:, 1]
                           + m273.predict_proba(X)[:, 1])
                out[a][b0:b1, c0:c1] = p.astype(np.float32).reshape(b1 - b0, c1 - c0)
            del X
        nb += 1
        if nb % progress_every == 0:
            print(f"    {b1}/{H} rows  {time.time()-t0:.0f}s", flush=True)
    del f17
    return out, img


def predict_at(iid, fits, rr, cc):
    """{arm: prob vector} at scattered pixels -- for sampled false-positive rates."""
    img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
    f17 = compute_feature_stack(img)
    coords, embs = M.read_emb(S.path(iid, "emb.npz"))
    res = {a: np.empty(len(rr), np.float32) for a in ARMS}
    step = 250000
    for i in range(0, len(rr), step):
        j = min(i + step, len(rr))
        X = np.concatenate([np.asarray(f17[rr[i:j], cc[i:j], :], np.float32),
                            M.emb_rows(coords, embs, rr[i:j], cc[i:j])], axis=1)
        for a in ARMS:
            m17, m273 = fits[a]
            res[a][i:j] = 0.5 * (m17.predict_proba(X[:, :N17])[:, 1]
                                 + m273.predict_proba(X)[:, 1])
        del X
    del f17
    return res


def half_width(mask):
    """Median distance-to-edge along the mask's centreline, in pixels."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    if not mask.any():
        return None, 0
    skel = skeletonize(mask)
    if not skel.any():
        return None, 0
    edt = distance_transform_edt(mask)
    return round(float(np.median(edt[skel])), 2), int(skel.sum())


def width_report(iid, probs, img):
    corr = S.load_npy(iid, "correction.npy")
    crack = corr == 1
    core = P.tighten_to_image(iid, crack) if crack.any() else np.zeros_like(crack)
    n = crack.size
    rep = dict(id=iid, shape=list(img.shape),
               painted=dict(area_pct=round(100 * crack.mean(), 4),
                            half_width=half_width(crack)[0]),
               dark_core=dict(area_pct=round(100 * core.mean(), 4),
                              half_width=half_width(core)[0]),
               arms={})
    for a, pr in probs.items():
        raw = pr > 0.5
        pruned = P.prune_specks(raw)
        hw_raw = half_width(raw)
        hw_pr = half_width(pruned)
        rep["arms"][a] = dict(
            area_pct=round(100 * raw.mean(), 4),
            area_pct_pruned=round(100 * pruned.mean(), 4),
            half_width=hw_raw[0], half_width_pruned=hw_pr[0],
            skel_px=hw_raw[1],
            recall_core=round(float((pruned & core).sum() / max(core.sum(), 1)), 4),
            recall_painted=round(float((pruned & crack).sum() / max(crack.sum(), 1)), 4),
            precision_vs_core=round(float((pruned & core).sum() / max(pruned.sum(), 1)), 4),
            iou_core=round(float((pruned & core).sum()
                                 / max((pruned | core).sum(), 1)), 4),
        )
        del raw, pruned
    rep["n_px"] = int(n)
    return rep


def main():
    fold_of = {int(k): v for k, v in json.load(open(os.path.join(OUT, "folds.json"))).items()}
    meta = json.load(open(os.path.join(OUT, "rows_meta.json")))
    gi_of_id = {d["id"]: d["gi"] for d in meta["images"]}
    fname = {d["id"]: (d["filename"] or "") for d in meta["images"]}
    n_crack = {d["id"]: d["n_crack_painted"] for d in meta["images"]}
    declined = {d["id"]: d["tighten_declined"] for d in meta["images"]}

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    outp = os.path.join(OUT, f"frames_{which}.json")

    if which in ("width", "all"):
        # Frames chosen for width: the frame the hypothesis quotes (wrought 1200 cycles)
        # plus the most heavily painted frame of each other specimen group, so the number
        # is not one specimen's story.
        want = ["wrought_316L_fatigue_1200_cycles_crack",
                "b2_343_75_LARGE", "b3_388_13um_LARGE_2",
                "HC_316L_fatigue_1750_cycles"]
        picks = []
        for w in want:
            hit = [i for i in gi_of_id if w.lower() in (i + fname[i]).lower()
                   and n_crack[i] > 0]
            if hit:
                picks.append(sorted(hit, key=lambda i: -n_crack[i])[0])
        res["width"] = []
        for iid in picks:
            f = fold_of[gi_of_id[iid]]
            print(f"[width] {iid[:60]} fold={f} declined={declined[iid]}", flush=True)
            fits = load_fits(f)
            probs, img = predict_full(iid, fits)
            r = width_report(iid, probs, img)
            r.update(fold=f, tighten_declined=declined[iid])
            res["width"].append(r)
            for a in ARMS:
                d = r["arms"][a]
                print(f"    {a:17s} hw={d['half_width_pruned']}  area={d['area_pct_pruned']}%"
                      f"  recall_core={d['recall_core']}  IoU_core={d['iou_core']}", flush=True)
            print(f"    reference: painted hw={r['painted']['half_width']} "
                  f"({r['painted']['area_pct']}%)  core hw={r['dark_core']['half_width']} "
                  f"({r['dark_core']['area_pct']}%)", flush=True)
            del probs, img
            json.dump(res, open(outp, "w"), indent=1)

    if which in ("fp", "all"):
        clean = [i for i in gi_of_id
                 if any(k.lower() in (i + fname[i]).lower() for k in P.CLEAN_SPECIMENS)]
        print("crack-free specimens matched:", len(clean), flush=True)
        res["fp"] = []
        for iid in sorted(clean):
            f = fold_of[gi_of_id[iid]]
            img = np.asarray(S.load_npy(iid, "img.npy"), np.float32)
            sup = P.specimen_support(img)
            H, W = img.shape
            del img
            on = np.flatnonzero(sup.reshape(-1))
            off = np.flatnonzero((~sup).reshape(-1))
            rng = np.random.RandomState(0)
            take_on = rng.choice(on, min(400000, len(on)), replace=False)
            take_off = (rng.choice(off, min(200000, len(off)), replace=False)
                        if len(off) else np.empty(0, np.int64))
            idx = np.sort(np.concatenate([take_on, take_off]))
            is_on = np.isin(idx, take_on, assume_unique=False)
            rr, cc = np.unravel_index(idx, (H, W))
            fits = load_fits(f)
            pr = predict_at(iid, fits, rr, cc)
            row = dict(id=iid, fold=f, support_pct=round(100 * float(sup.mean()), 2),
                       n_on=int(is_on.sum()), n_off=int((~is_on).sum()), arms={})
            for a in ARMS:
                p = pr[a] > 0.5
                row["arms"][a] = dict(
                    on_specimen_fp_pct=round(100 * float(p[is_on].mean()), 4),
                    off_specimen_fp_pct=(round(100 * float(p[~is_on].mean()), 4)
                                         if (~is_on).any() else None),
                    whole_frame_fp_pct=round(100 * float(p.mean()), 4))
            res["fp"].append(row)
            print(f"[fp] {iid[:52]:52s} sup={row['support_pct']}% " + "  ".join(
                f"{a}={row['arms'][a]['on_specimen_fp_pct']}%" for a in ARMS), flush=True)
            del sup, pr
            json.dump(res, open(outp, "w"), indent=1)

    json.dump(res, open(outp, "w"), indent=1)
    print("wrote", outp)


if __name__ == "__main__":
    main()
