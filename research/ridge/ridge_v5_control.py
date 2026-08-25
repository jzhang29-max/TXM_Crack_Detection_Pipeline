"""Re-run only the control arm, on the cached rows, to confirm the harness is sound."""
import json, os, sys, time
import numpy as np
P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(P0, "app", "core"), os.path.join(P0, "code"), HERE):
    sys.path.insert(0, p)
import ridge_features as RF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
z = np.load(os.path.join(HERE, "rows_thincore_cache.npz"))
X, y, img, grp = z["X"], z["y"], z["img"], z["grp"]
noise = np.random.RandomState(7).standard_normal((len(y), 9)).astype(np.float32)
M = np.hstack([X[:, RF.COL_BASE17], noise])
B = X[:, RF.COL_BASE17]
def clf():
    return Pipeline([("s", StandardScaler()),
                     ("m", MLPClassifier((64,32), max_iter=300, random_state=0,
                                         early_stopping=True, n_iter_no_change=8))])
def iou(p, t):
    tp=int((p&t).sum()); fp=int((p&~t).sum()); fn=int((~p&t).sum())
    return tp/max(tp+fp+fn,1)
out = {"kfold": {}, "logo": {}}
for name, D in (("baseline_17", B), ("17_plus_noise9", M)):
    ious = []
    for tr, te in GroupKFold(5).split(D, groups=img):
        c = clf(); c.fit(D[tr], y[tr])
        ious.append(iou(c.predict_proba(D[te])[:,1] > 0.5, y[te]))
    out["kfold"][name] = dict(iou=round(float(np.mean(ious)),4),
                              sd=round(float(np.std(ious, ddof=1)),4))
    print(f"  kfold {name:<18} {out['kfold'][name]['iou']:.4f}", flush=True)
    for g in sorted(set(grp.tolist())):
        te = grp == g
        c = clf(); c.fit(D[~te], y[~te])
        out["logo"].setdefault(name, {})[g] = round(iou(c.predict_proba(D[te])[:,1] > 0.5, y[te]),4)
    print(f"  logo  {name:<18} {out['logo'][name]}", flush=True)
json.dump(out, open(os.path.join(HERE, "ridge_v5_control.json"), "w"), indent=1)
print("CONTROL_DONE")
