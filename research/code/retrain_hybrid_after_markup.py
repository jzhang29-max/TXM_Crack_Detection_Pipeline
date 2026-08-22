"""
ONE COMMAND for the hybrid after a round of markup. Sweeps class balance,
scores every candidate on BOTH axes, and only deploys if it wins on both.

    python3 code/retrain_hybrid_after_markup.py [--deploy]

The same discipline as retrain_after_markup.py, for the SAM+17 model:

  1. Sweep --neg-cap around the balance point computed from the CURRENT label
     inventory. Class balance is the knob that caused four separate regressions
     in this project, and a fixed sweep goes stale -- the balance point moved
     8,028 -> 12,971 when B3 and Wrought crack labels were added.
  2. Score every candidate on the 4 external reference images (IoU, recall)
     AND on the 6 owner-confirmed crack-free specimens (predicted area, i.e.
     pure false positives).
  3. Deploy only if IoU holds within tolerance AND crack-free false positives
     do not grow. Every regression this project has had passed a single-metric
     check: flat-fielding looked good on false positives and cost 0.169 IoU;
     the curvilinearity gate cut area 8x while destroying 98% of true crack on
     one image. An over-aggressive filter and a good one both reduce area, so
     recall against ground truth is the only thing that separates them.

Scoring note, stated because it matters: the 4 ground-truth images contribute
their own corrections to training, so these IoU numbers are LEAKY in absolute
terms -- they are not held-out. Every candidate and the incumbent are scored the
same leaky way, so the RANKING is valid and that is what this script uses them
for. For an honest absolute number use baseline_loio_for_sam.py.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sam_embeddings as EC
import paint_common as pc
from apply_pixel_model import postprocess_mask
from apply_sam_hybrid import predict_image
from retrain_after_markup import balancing_neg_cap
from txm_features import robust_normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY_PATH = os.path.join(pc.PROJECT_DIR, "models", "pixel_sam_hybrid.joblib")
REPORT = os.path.join(pc.PROJECT_DIR, "results", "improve", "hybrid_retrain_rounds.json")

IOU_TOL = 0.01          # allow a hair of IoU loss if crack-free improves a lot
CRACKFREE_TOL = 0.005   # ...but never let crack-free false positives grow >0.5pp

STEMS = ["333_75_um_zoom", "336_25", "338_13", "LARGE_343_75"]


def _full_name(stem):
    key = stem.replace("LARGE_343_75", "343_75_LARGE")
    for i in pc.list_images():
        if key in i["name"]:
            return i["name"]
    return None


def score(model_path):
    """(mean IoU, mean recall) on ground truth, mean predicted area on crack-free."""
    b = joblib.load(model_path)
    clf, nf = b["model"], b["n_features"]
    ious, recs = [], []
    for stem in STEMS:
        name = _full_name(stem)
        gt = np.load(os.path.join(pc.PROJECT_DIR, "dataset_cache", f"{stem}_gt.npy")).astype(bool)
        prob, _ = predict_image(name, clf, nf)
        pm = postprocess_mask(prob)
        if pm.shape != gt.shape:
            print(f"    [warn] {stem}: shape {pm.shape} vs gt {gt.shape}, skipped")
            continue
        inter = int(np.logical_and(pm, gt).sum())
        union = int(np.logical_or(pm, gt).sum())
        ious.append(inter / union if union else float("nan"))
        recs.append(inter / int(gt.sum()) if gt.sum() else float("nan"))
        del gt, prob, pm

    import mark_zero_crack_images as mz
    cf = []
    for frag in mz.ZERO_CRACK:
        hits = [i["name"] for i in pc.list_images() if frag in i["name"]]
        if not hits or not os.path.exists(EC.cache_path(hits[0])):
            continue
        prob, _ = predict_image(hits[0], clf, nf)
        cf.append(float(postprocess_mask(prob).mean()))
        del prob
    return float(np.mean(ious)), float(np.mean(recs)), float(np.mean(cf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, nargs="+", default=None)
    ap.add_argument("--deploy", action="store_true")
    args = ap.parse_args()

    print("=" * 74); print("CLASS BALANCE (from the CURRENT label inventory)"); print("=" * 74)
    c = balancing_neg_cap()
    if args.sweep is None:
        args.sweep = sorted({max(500, int(c * f)) for f in (0.6, 0.85, 1.0, 1.3)})
    print(f"  sweep: {args.sweep}")

    print("\n" + "=" * 74); print("INCUMBENT"); print("=" * 74)
    inc = None
    if os.path.exists(DEPLOY_PATH):
        t0 = time.time()
        i0, r0, c0 = score(DEPLOY_PATH)
        inc = dict(iou=i0, recall=r0, crackfree=c0)
        print(f"  {os.path.basename(DEPLOY_PATH)}:  IoU {i0:.3f}  recall {r0:.3f}  "
              f"crack-free {c0*100:.2f}%   ({time.time()-t0:.0f}s)")
    else:
        print("  (no incumbent hybrid yet)")

    results = []
    for cap in args.sweep:
        out = os.path.join(pc.PROJECT_DIR, "models", f"pixel_sam_hybrid_neg{cap}.joblib")
        print(f"\n--- training candidate, --neg-cap {cap} ---")
        r = subprocess.run([sys.executable, os.path.join(HERE, "train_sam_hybrid.py"),
                            "--neg-cap", str(cap), "--out", out],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "Training set:" in line or "WARNING" in line:
                print("  " + line.strip())
        if not os.path.exists(out):
            print(f"  FAILED: {r.stderr.strip()[-400:]}")
            continue
        i, rc, cf = score(out)
        results.append(dict(cap=cap, path=out, iou=i, recall=rc, crackfree=cf))
        print(f"  IoU {i:.3f}  recall {rc:.3f}  crack-free {cf*100:.2f}%")

    if not results:
        print("\nno candidates trained")
        return

    print("\n" + "=" * 74); print("RESULTS"); print("=" * 74)
    print(f"{'candidate':24s} {'IoU':>7s} {'recall':>8s} {'crack-free':>11s}  verdict")
    if inc:
        print(f"{'INCUMBENT':24s} {inc['iou']:7.3f} {inc['recall']:8.3f} "
              f"{inc['crackfree']*100:10.2f}%  (baseline)")
    for r in sorted(results, key=lambda x: -x["iou"]):
        if inc:
            ok = (r["iou"] >= inc["iou"] - IOU_TOL
                  and r["crackfree"] <= inc["crackfree"] + CRACKFREE_TOL)
            why = "PASSES both gates" if ok else (
                "IoU regressed" if r["iou"] < inc["iou"] - IOU_TOL
                else "crack-free false positives grew")
        else:
            ok, why = True, "no incumbent to beat"
        print(f"{'neg-cap ' + str(r['cap']):24s} {r['iou']:7.3f} {r['recall']:8.3f} "
              f"{r['crackfree']*100:10.2f}%  {why}")
        r["passes"] = bool(ok)

    winners = [r for r in results if r.get("passes")]
    best = max(winners, key=lambda x: x["iou"]) if winners else None

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    hist = []
    if os.path.exists(REPORT):
        try:
            hist = json.load(open(REPORT)).get("rounds", [])
        except Exception:
            hist = []
    hist.append(dict(balance_point=c, sweep=args.sweep, incumbent=inc,
                     candidates=results, best_cap=(best or {}).get("cap"),
                     deployed=bool(args.deploy and best)))
    with open(REPORT, "w") as f:
        json.dump(dict(rounds=hist), f, indent=2)

    if not best:
        print("\nNo candidate passed both gates. Keeping the incumbent.")
        return
    print(f"\nBEST: neg-cap {best['cap']}  IoU {best['iou']:.3f}"
          + (f"  (incumbent {inc['iou']:.3f})" if inc else ""))
    if args.deploy:
        import shutil
        if os.path.exists(DEPLOY_PATH):
            bak = DEPLOY_PATH.replace(".joblib", f".backup-{int(os.path.getmtime(DEPLOY_PATH))}.joblib")
            shutil.copy2(DEPLOY_PATH, bak)
            print(f"  backed up incumbent -> {os.path.basename(bak)}")
        shutil.copy2(best["path"], DEPLOY_PATH)
        print(f"  DEPLOYED {os.path.basename(best['path'])} -> {os.path.basename(DEPLOY_PATH)}")
        print("  Next: python3 code/apply_sam_hybrid.py && "
              "python3 code/populate_hybrid_paint_cache.py")
    else:
        print("  (not deployed -- rerun with --deploy to swap it in)")


if __name__ == "__main__":
    main()
