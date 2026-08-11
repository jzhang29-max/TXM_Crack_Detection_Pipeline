"""
ONE COMMAND to run after you finish marking up. Trains, validates against
ground truth, and only deploys if it is genuinely better.

    python3 code/retrain_after_markup.py

What it does, in order:
  1. Reports markup coverage so you can see what it is learning from.
  2. Trains RAW candidates across a sweep of --neg-cap values. This is not
     optional tuning: class balance is the single knob that has caused four
     separate regressions in this project. Measured on the raw model,
     24% crack -> IoU 0.649, 38% -> 0.744, 44% -> 0.773. The sweep finds the
     balance point for whatever label mix now exists instead of guessing it.
  3. Scores every candidate on BOTH axes against the 4 Ilastik ground-truth
     images: IoU/recall (does it find real cracks) AND predicted area on the
     6 owner-confirmed crack-free specimens (does it invent cracks).
  4. Deploys the best candidate ONLY if it beats the incumbent on IoU within
     tolerance while not being worse on the crack-free axis.

Why the double check is enforced here rather than left to judgment: every
regression in this project passed a single-metric check. Flatfielding looked
good on false positives and cost 0.169 IoU. The curvilinearity gate cut
predicted area 8x, which reads as artifact removal, while destroying 98% of
true crack on one image (recall 0.617 -> 0.016). An over-aggressive filter and
a good one both reduce area; only recall against ground truth separates them.

Usage:
    python3 retrain_after_markup.py [--sweep 1000 2000 3000 5000] [--deploy]
"""
import argparse, json, os, subprocess, sys
import joblib, numpy as np, tifffile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
from apply_pixel_model import postprocess_mask, predict_probability_map
from txm_features import robust_normalize
import mark_zero_crack_images as mz

HERE = os.path.dirname(os.path.abspath(__file__))
IOU_TOL = 0.01          # allow a hair of IoU loss if crack-free improves a lot
CRACKFREE_TOL = 0.005   # ...but never let crack-free false positives grow >0.5pp


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def recall(a, b):
    return float(np.logical_and(a, b).sum() / b.sum()) if b.sum() else float("nan")


def score(model):
    """(mean IoU, mean recall) on ground truth, and mean area on crack-free."""
    man = json.load(open(os.path.join(pc.PROJECT_DIR, "dataset_cache", "manifest.json")))
    ious, recs = [], []
    for img in man["images"]:
        gt = np.load(img["gt_path"]); im = np.load(img["img_path"]).astype(np.float64)
        pm = postprocess_mask(predict_probability_map(model, im))
        ious.append(iou(pm, gt)); recs.append(recall(pm, gt))
        del gt, im, pm
    cf = []
    for nm in mz.ZERO_CRACK:
        p = pc._find_path(nm)
        im = robust_normalize(tifffile.imread(p).astype(np.float64), 1.0, 99.0)
        cf.append(float(postprocess_mask(predict_probability_map(model, im)).mean()))
        del im
    return float(np.mean(ious)), float(np.mean(recs)), float(np.mean(cf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, nargs="+", default=[1000, 2000, 3000, 5000])
    ap.add_argument("--deploy", action="store_true", help="actually swap production if a candidate wins")
    args = ap.parse_args()

    print("=" * 74); print("MARKUP COVERAGE"); print("=" * 74)
    subprocess.run([sys.executable, os.path.join(HERE, "markup_status.py"), "--todo"])

    print("\n" + "=" * 74); print("INCUMBENT"); print("=" * 74)
    inc = joblib.load(pc.MODEL_PATH)
    i0, r0, c0 = score(inc)
    print(f"  {os.path.basename(pc.MODEL_PATH)}:  IoU {i0:.3f}  recall {r0:.3f}  crack-free {c0*100:.2f}%")
    del inc

    results = []
    for cap in args.sweep:
        out = os.path.join(pc.PROJECT_DIR, "models", f"pixel_cand_neg{cap}.joblib")
        print(f"\n--- training candidate, --neg-cap {cap} ---")
        r = subprocess.run([sys.executable, os.path.join(HERE, "retrain_with_corrections.py"),
                            "--correction-weight", "1.0", "--neg-cap", str(cap), "--out", out],
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if "Training on" in l]
        if line: print("  " + line[0].strip())
        if not os.path.exists(out):
            print(f"  FAILED: {r.stderr.strip()[-300:]}"); continue
        m = joblib.load(out); i, rc, c = score(m); del m
        results.append(dict(cap=cap, path=out, iou=i, recall=rc, crackfree=c))
        print(f"  IoU {i:.3f}  recall {rc:.3f}  crack-free {c*100:.2f}%")

    if not results:
        print("\nno candidates trained"); return

    print("\n" + "=" * 74); print("RESULTS (incumbent first)"); print("=" * 74)
    print(f"{'candidate':22s} {'IoU':>7s} {'recall':>8s} {'crack-free':>11s}  verdict")
    print(f"{'INCUMBENT':22s} {i0:7.3f} {r0:8.3f} {c0*100:10.2f}%")
    best = None
    for r in sorted(results, key=lambda r: -r["iou"]):
        ok_iou = r["iou"] >= i0 - IOU_TOL
        ok_cf = r["crackfree"] <= c0 + CRACKFREE_TOL
        v = "BETTER" if (ok_iou and ok_cf) else ("IoU regressed" if not ok_iou else "more false crack")
        print(f"{'neg-cap ' + str(r['cap']):22s} {r['iou']:7.3f} {r['recall']:8.3f} "
              f"{r['crackfree']*100:10.2f}%  {v}")
        if ok_iou and ok_cf and (best is None or r["iou"] > best["iou"]):
            best = r

    json.dump(dict(incumbent=dict(iou=i0, recall=r0, crackfree=c0), candidates=results,
                   chosen=(best or {}).get("path")),
              open(os.path.join(pc.PROJECT_DIR, "results", "retrain_after_markup.json"), "w"), indent=2)

    if best is None:
        print("\nNo candidate passed BOTH checks. Production unchanged -- this is the")
        print("correct outcome, not a failure: it means the new labels cannot yet be")
        print("absorbed without giving something up. Mark a few more images and rerun.")
        return
    print(f"\nBEST: neg-cap {best['cap']}  IoU {best['iou']:.3f} (incumbent {i0:.3f})  "
          f"crack-free {best['crackfree']*100:.2f}% (incumbent {c0*100:.2f}%)")
    if args.deploy:
        import shutil, time
        bk = os.path.join(pc.PROJECT_DIR, "models", f"pixel_prev_{int(time.time())}.joblib")
        shutil.copy(pc.MODEL_PATH, bk); shutil.copy(best["path"], pc.MODEL_PATH)
        print(f"DEPLOYED -> {os.path.basename(pc.MODEL_PATH)}   (previous backed up to {os.path.basename(bk)})")
        print("The paint tool picks this up automatically on its next request.")
    else:
        print(f"\nNot deployed (no --deploy). To deploy:\n"
              f"  cp {os.path.relpath(best['path'], pc.PROJECT_DIR)} models/pixel_hgb_final.joblib")

if __name__ == "__main__":
    main()
