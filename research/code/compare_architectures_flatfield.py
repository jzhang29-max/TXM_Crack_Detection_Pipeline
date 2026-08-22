"""
Compare model architectures on the CURRENT flatfielded pipeline and the
71-image label set.

Why re-run this: the earlier architecture comparison (final_figures/fig1-8)
was done on RAW input with only the 4 B2 ground-truth images. Everything
that matters has changed since -- flatfielded input, 71 labelled images,
four specimen types -- so those results no longer describe this problem.

Evaluated on the three things that actually go wrong here, not IoU alone.
IoU only exists for the 4 B2 images, and a model can score well there while
still flooding a Wrought frame or hallucinating crack in an undamaged
specimen, which is precisely what happened repeatedly in this project:

  1. ACCURACY   mean IoU over the 4 external reference images (B2 only --
                the only pixel-level truth that exists).
  2. FALSE POSITIVE RATE  mean predicted area over the 6 user-confirmed
                CRACK-FREE specimens. Should be ~0. This is the metric the
                original raw-trained model failed catastrophically (41% on
                an undamaged specimen).
  3. ARTIFACT SENSITIVITY  region count on two AM frames dense with
                reference/calibration artifacts. Hundreds of scattered
                specks inflate region count far more than area, so this
                catches artifact confusion that area alone hides.

A model must do well on ALL THREE to be worth using. Ranking by any single
one of them is how you end up shipping a model that is accurate on B2 and
useless on everything else.

Usage:
    python3 compare_architectures_flatfield.py
"""
import glob, json, os, sys, time
import numpy as np
import tifffile
from skimage.measure import label
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
import retrain_with_corrections as rc
import build_flatfield_dataset as bf
from apply_pixel_model import postprocess_mask, predict_probability_map
import train_flatfield_model as tf
import mark_zero_crack_images as mz

PREDCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache")
OUT = os.path.join(pc.PROJECT_DIR, "results", "arch_compare_flatfield.json")

ARCHS = {
    "RandomForest":        (lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0), False),
    "ExtraTrees":          (lambda: ExtraTreesClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=0), False),
    "DecisionTree":        (lambda: DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=0), False),
    "HistGradientBoosting":(lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.1, class_weight="balanced", random_state=0), False),
    "LogisticRegression":  (lambda: Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]), True),
    "MLP (neural net)":    (lambda: Pipeline([("s", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(64,32), alpha=1e-4, max_iter=300, early_stopping=True, random_state=0))]), True),
}

ARTIFACT_IMGS = [
    "Average_mosaic_260619_HC_316L_fatigue_1000_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
    "Average_mosaic_260619_HC_316L_fatigue_1450_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim",
]

def build_training(crack_cap=30000, neg_cap=6000):
    rng = np.random.RandomState(0)
    Xs, ys, ws = [], [], []
    with open(os.path.join(bf.CACHE_DIR, "manifest.json")) as f:
        man = json.load(f)
    for img in man["images"]:
        feats = np.load(img["feat_path"], mmap_mode="r"); gt = np.load(img["gt_path"])
        flat = np.asarray(feats).reshape(-1, feats.shape[-1]); fg = gt.reshape(-1)
        ci, bi = np.flatnonzero(fg), np.flatnonzero(~fg)
        nc, nb = min(100000, len(ci)), min(100000, len(bi))
        idx = np.concatenate([rng.choice(ci, nc, False), rng.choice(bi, nb, False)])
        Xs.append(flat[idx]); ys.append(np.concatenate([np.ones(nc,bool), np.zeros(nb,bool)])); ws.append(np.ones(len(idx)))
        del feats, flat, gt
    for cp in sorted(glob.glob(os.path.join(pc.CORRECTIONS_DIR, "*_correction.npy"))):
        nm = os.path.basename(cp)[:-len("_correction.npy")]
        corr = np.load(cp)
        if not (corr != 0).any(): continue
        feats = tf.flat_features_for(nm)
        if feats is None or feats.shape[:2] != corr.shape: continue
        flat = np.asarray(feats).reshape(-1, feats.shape[-1]); fc = corr.reshape(-1)
        ci, bi = np.flatnonzero(fc==1), np.flatnonzero(fc==2)
        nc, nb = min(crack_cap,len(ci)), min(neg_cap,len(bi))
        parts=[]
        if nc: parts.append(rng.choice(ci,nc,False))
        if nb: parts.append(rng.choice(bi,nb,False))
        if not parts: continue
        idx=np.concatenate(parts)
        Xs.append(flat[idx]); ys.append(np.concatenate([np.ones(nc,bool),np.zeros(nb,bool)])); ws.append(np.ones(len(idx)))
        del feats, flat
    X=np.concatenate(Xs).astype(np.float32); y=np.concatenate(ys); w=np.concatenate(ws)
    return X, y, compute_sample_weight("balanced", y)*w

def iou(a,b):
    u=np.logical_or(a,b).sum()
    return float(np.logical_and(a,b).sum()/u) if u else float("nan")

def main():
    print("Building training set (flatfielded, 71 labelled images)...")
    X, y, sw = build_training()
    print(f"  {len(y):,} px, {y.mean()*100:.1f}% crack\n")

    with open(os.path.join(bf.CACHE_DIR, "manifest.json")) as f:
        gt_imgs = json.load(f)["images"]
    zero = mz.ZERO_CRACK

    results = {}
    for name, (mk, _) in ARCHS.items():
        print(f"=== {name} ===")
        clf = mk(); t0=time.time()
        rc.fit_with_sample_weight(clf, X, y, sw)
        fit_s = time.time()-t0
        print(f"  fit {fit_s:.1f}s")

        ious=[]
        for g in gt_imgs:
            feats=np.load(g["feat_path"], mmap_mode="r")
            flat=np.asarray(feats).reshape(-1,feats.shape[-1])
            pred=(clf.predict_proba(flat)[:,1]>=0.5).reshape(np.load(g["gt_path"]).shape)
            ious.append(iou(pred, np.load(g["gt_path"])))
            del feats, flat, pred
        mean_iou=float(np.mean(ious))

        fps=[]
        for nm in zero:
            ip=os.path.join(PREDCACHE,f"{nm}_img.npy")
            if not os.path.exists(ip): continue
            img=np.load(ip).astype(np.float64)
            fps.append(float(postprocess_mask(predict_probability_map(clf,img)).mean()))
            del img
        mean_fp=float(np.mean(fps)) if fps else float("nan")

        rgs=[]
        for nm in ARTIFACT_IMGS:
            ip=os.path.join(PREDCACHE,f"{nm}_img.npy")
            if not os.path.exists(ip): continue
            img=np.load(ip).astype(np.float64)
            m=postprocess_mask(predict_probability_map(clf,img))
            rgs.append(int(label(m,connectivity=2).max()))
            del img,m
        mean_rg=float(np.mean(rgs)) if rgs else float("nan")

        results[name]=dict(mean_iou_gt=mean_iou, per_image_iou=ious,
                           mean_area_on_crackfree=mean_fp, per_image_crackfree=fps,
                           mean_regions_on_artifact_imgs=mean_rg, artifact_regions=rgs,
                           fit_seconds=fit_s)
        print(f"  IoU(GT)={mean_iou:.4f}  area on CRACK-FREE={mean_fp*100:.2f}%  "
              f"regions on artifact imgs={mean_rg:.0f}\n")

    json.dump(results, open(OUT,"w"), indent=2)
    print("="*84)
    print(f"{'architecture':22s} {'IoU(GT)':>9s} {'crack-free area':>16s} {'artifact rg':>12s} {'fit s':>8s}")
    print("="*84)
    for n,r in sorted(results.items(), key=lambda kv:-kv[1]['mean_iou_gt']):
        print(f"{n:22s} {r['mean_iou_gt']:9.4f} {r['mean_area_on_crackfree']*100:15.2f}% "
              f"{r['mean_regions_on_artifact_imgs']:12.0f} {r['fit_seconds']:8.1f}")
    print("\nIoU higher=better; crack-free area LOWER=better (should be ~0);")
    print("artifact regions LOWER=better. A model must win on all three.")
    print(f"\nSaved {OUT}")

if __name__=="__main__":
    main()
