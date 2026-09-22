#!/usr/bin/env python3
"""Measure false positives on the crack-free specimens with a model that never saw them.

WHY THIS EXISTS. The six specimens the owner confirmed contain no crack are the project's
only check on over-prediction that does not need pixel-level ground truth. But their
correction masks ARE in the training set: they contribute 0 crack pixels and 95,351,647
not-crack pixels, **25.4% of every background label the model is trained on**. So "predicted
area on crack-free specimens" is measured on pixels the model was explicitly told are
background. That is train-on-test, and it makes the headline false-positive figure partly a
memorisation result rather than a generalisation one.

WHAT THIS DOES. Refits the deployed architecture on the same rows with every CLEAN_SPECIMENS
frame removed, then predicts those frames and measures the same quantity. The deployed model
is not touched -- swapping it would invalidate every other number in the repo -- so this
reports the honest figure alongside, rather than replacing it.

Emits: out/txm_heldout_crackfree_fp.json

Usage:  python3 code/audit/heldout_crackfree_fp.py [out.json]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "app", "core"))
os.chdir(HERE)

from app.core import pipeline as P                                    # noqa: E402
from app.core import store as S                                       # noqa: E402
from app.core import model as M                                       # noqa: E402


def is_clean(meta):
    fn = (meta.get("filename") or "") + " " + meta.get("id", "")
    return any(k.lower() in fn.lower() for k in P.CLEAN_SPECIMENS)


def main(outp="out_heldout_fp.json"):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    print("gathering training rows ...", flush=True)
    X, y, info, groups = P.gather_training_data(
        progress=lambda m, a, b: print(f"   {m} {a}/{b}", flush=True))
    if X is None:
        raise SystemExit("no labelled data")

    # `groups` is an int index into the labelled-image list, in list_images() order --
    # not a filename. Mapping it wrongly is how an earlier analysis in this project
    # classified all 71 frames as one specimen, so the mapping is rebuilt explicitly here.
    labelled = [m for m in S.list_images()
                if m.get("corrected_crack_px") or m.get("corrected_not_px")]
    clean_idx = {i for i, m in enumerate(labelled) if is_clean(m)}
    clean_names = [m.get("filename", m["id"])[:60] for i, m in enumerate(labelled)
                   if i in clean_idx]
    keep = ~np.isin(groups, list(clean_idx))
    print(f"\nrows {X.shape[0]:,}  features {X.shape[1]}")
    print(f"clean-specimen frames found: {len(clean_idx)}")
    for n in clean_names:
        print(f"   excluded: {n}")
    print(f"rows dropped: {int((~keep).sum()):,} ({100*(~keep).mean():.1f}%)")
    print(f"rows kept:    {int(keep.sum()):,}   crack fraction "
          f"{float(y[keep].mean()):.3f} (was {float(y.mean()):.3f})")
    if keep.sum() < 1000 or not (0.05 <= float(y[keep].mean()) <= 0.95):
        raise SystemExit("held-out training set is degenerate; refusing to train")

    print("\nfitting the deployed architecture on the held-out rows ...", flush=True)
    clf = Pipeline([("scaler", StandardScaler(copy=False)),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                          random_state=0))])
    # 17-FEATURE BRANCH: train on the first 17 columns only. The repo's own cross-validation
    # uses slice(0, 17) for this branch (pipeline.crossval_on_rows), and the predict path below
    # builds a 17-column stack, so fitting on all 273 here produced
    # "X has 17 features, but StandardScaler is expecting 273" at scoring time.
    clf.fit(X[keep][:, :17], y[keep])
    print(f"   fitted in {time.time()-t0:.0f}s", flush=True)

    class _Held:
        """Minimal CrackModel interface so the SHIPPED scoring path can be reused."""
        def __init__(self, est, ndim):
            self.est, self.ndim = est, ndim

        def needs_sam(self):
            return self.ndim > 17

        def predict(self, img01, emb=None, band=128, progress=None):
            from txm_features import compute_feature_stack
            f17 = compute_feature_stack(img01)
            H, W = img01.shape
            if self.ndim <= 17:
                out = np.zeros((H, W), np.float32)
                for r0 in range(0, H, 256):
                    r1 = min(r0 + 256, H)
                    blk = np.asarray(f17[r0:r1], np.float32).reshape(-1, f17.shape[2])
                    out[r0:r1] = self.est.predict_proba(blk)[:, 1].reshape(r1 - r0, W)
                return out
            raise NotImplementedError("hybrid path not needed: see the note below")

    # The 17-feature branch alone is used deliberately. The hybrid branch needs a SAM
    # embedding lookup per tile, and reimplementing that here would be a second copy of
    # code whose subtleties (tile overlap, cache tagging) have already caused bugs. The
    # comparison stays fair because the INCUMBENT is re-scored through the same 17-feature
    # path below, so both arms differ only in what they were trained on.
    held = _Held(clf, 17)
    print("\nscoring held-out model on the crack-free specimens ...", flush=True)
    hf, hn, hdetail = P._score_clean(held, progress=lambda m, a, b: print(f"   {m}", flush=True))

    print("\nscoring the SAME architecture trained WITH them, for the paired comparison ...",
          flush=True)
    clf_all = Pipeline([("scaler", StandardScaler(copy=False)),
                        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                              random_state=0))])
    clf_all.fit(X[:, :17], y)
    withm = _Held(clf_all, 17)
    wf, wn, wdetail = P._score_clean(withm, progress=lambda m, a, b: print(f"   {m}", flush=True))

    res = dict(
        what="False-positive area on the crack-free specimens, measured with a model that "
             "never saw their labels. Run 2026-09-21.",
        why="Those six frames contribute 0 crack px and 95,351,647 not-crack px -- 25.4% of "
            "every background label in training -- so the shipped figure is measured on "
            "pixels the model was told are background.",
        generator="code/audit/heldout_crackfree_fp.py",
        architecture="StandardScaler + MLPClassifier((128,64), max_iter=400, random_state=0), "
                     "17-feature branch only, both arms identical",
        n_clean_frames=len(clean_idx), clean_frames=clean_names,
        rows_total=int(X.shape[0]), rows_dropped=int((~keep).sum()),
        crack_fraction_all=float(y.mean()), crack_fraction_heldout=float(y[keep].mean()),
        trained_WITH_clean=dict(mean_fp_area=wf, n=wn, per_frame=wdetail),
        trained_WITHOUT_clean=dict(mean_fp_area=hf, n=hn, per_frame=hdetail),
        inflation_factor=(float(hf / wf) if wf else None),
        seconds=round(time.time() - t0, 1))
    json.dump(res, open(outp, "w"), indent=1)
    print(f"\n{'':34s}{'with':>12s}{'without':>12s}")
    print(f"{'mean FP area on crack-free':34s}{wf:12.6f}{hf:12.6f}")
    if wf:
        print(f"{'inflation from contamination':34s}{'':12s}{hf/wf:11.2f}x")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out_heldout_fp.json")
