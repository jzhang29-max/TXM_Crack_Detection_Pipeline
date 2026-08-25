"""Collect every measurement into one table. Reads only research/thinlabels/*.json."""
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
ARMS = ["baseline", "thin_ring_neg", "thin_plus_margin", "core_dilate3", "ring_neg_forced"]

cv = json.load(open(os.path.join(OUT, "cv_results.json")))
pp = json.load(open(os.path.join(OUT, "per_pool_rates.json")))
fr = json.load(open(os.path.join(OUT, "frames_all.json")))
lw = json.load(open(os.path.join(OUT, "label_widths.json")))

print("PREMISE, over all 61 frames with painted crack")
s = lw["summary"]
print(f"  median painted-stroke half-width   {s['median_hw_painted']} px")
print(f"  median dark-core half-width        {s['median_hw_core']} px")
print(f"  median DEPLOYED model half-width   {s['median_hw_deployed_pred']} px")
print(f"  median painted/core width ratio    {s['median_ratio']}x")
print(f"  median core area / painted area    {s['median_core_frac']}")
print(f"  frames where tightening declined   {s['n_declined']}/{s['n_frames']}")
hw_p = np.array([f["hw_painted"] for f in lw["frames"] if f["hw_painted"] and f["hw_deployed_pred"]])
hw_d = np.array([f["hw_deployed_pred"] for f in lw["frames"] if f["hw_painted"] and f["hw_deployed_pred"]])
hw_c = np.array([f["hw_core"] for f in lw["frames"] if f["hw_core"] and f["hw_deployed_pred"]])
print(f"  corr(pred half-width, painted half-width) = {np.corrcoef(hw_p, hw_d)[0,1]:.3f}"
      f"   corr(pred, core) = {np.corrcoef(hw_c, hw_d)[0,1]:.3f}   (n={len(hw_p)})")

print("\nHELD-OUT ROWS, GroupKFold(5) by image, SAME fixed target (the dark core)")
print(f"{'arm':18s} {'IoU':>15s} {'prec':>7s} {'rec':>7s} {'d(IoU)':>8s} {'folds won':>10s}"
      f" {'own IoU':>8s}")
b = np.array([pf["same"]["iou"] for pf in cv["arms"]["baseline"]["per_fold"]])
for a in ARMS:
    m = cv["arms"][a]["mean"]
    x = np.array([pf["same"]["iou"] for pf in cv["arms"][a]["per_fold"]])
    d = x - b
    print(f"{a:18s} {m['same_iou']:.4f}+-{m['same_iou_sd']:.4f} {m['same_precision']:7.4f}"
          f" {m['same_recall']:7.4f} {d.mean():+8.4f} {int((d>0).sum()):7d}/5"
          f" {m['own_iou']:8.4f}")

print("\nPOSITIVE RATE PER LABEL POOL on the same held-out rows")
print(f"{'arm':18s} {'dark core':>10s} {'inner ring':>11s} {'outer ring':>11s} {'corr==2':>9s}")
for a in ARMS:
    r = pp[a]
    print(f"{a:18s} {r['pool1']['mean']:10.4f} {r['pool2']['mean']:11.4f}"
          f" {r['pool3']['mean']:11.4f} {r['pool4']['mean']:9.4f}")

print("\nPREDICTED WIDTH, full frame, each frame held out of its arm's training fold")
for f in fr["width"]:
    print(f"  {f['id'][:58]}")
    print(f"    reference: painted hw={f['painted']['half_width']:6} "
          f"({f['painted']['area_pct']}%)   dark core hw={f['dark_core']['half_width']:6} "
          f"({f['dark_core']['area_pct']}%)")
    for a in ARMS:
        d = f["arms"][a]
        mw = (d["area_pct"] / 100 * f["n_px"]) / max(d["skel_px"], 1)
        print(f"    {a:18s} hw={d['half_width_pruned']:6}  area={d['area_pct_pruned']:7.3f}%"
              f"  mean_width={mw:6.1f}px  recall_core={d['recall_core']:.4f}"
              f"  IoU_core={d['iou_core']:.4f}")

print("\nmedian over the four width frames")
print(f"{'arm':18s} {'half-width':>11s} {'area %':>8s} {'mean width':>11s} {'recall core':>12s}")
for a in ARMS:
    hw = np.median([f["arms"][a]["half_width_pruned"] for f in fr["width"]])
    ar = np.median([f["arms"][a]["area_pct_pruned"] for f in fr["width"]])
    mw = np.median([(f["arms"][a]["area_pct"] / 100 * f["n_px"])
                    / max(f["arms"][a]["skel_px"], 1) for f in fr["width"]])
    rc = np.median([f["arms"][a]["recall_core"] for f in fr["width"]])
    print(f"{a:18s} {hw:11.2f} {ar:8.3f} {mw:11.1f} {rc:12.4f}")
print(f"{'painted stroke':18s} "
      f"{np.median([f['painted']['half_width'] for f in fr['width']]):11.2f} "
      f"{np.median([f['painted']['area_pct'] for f in fr['width']]):8.3f}")
print(f"{'dark core':18s} "
      f"{np.median([f['dark_core']['half_width'] for f in fr['width']]):11.2f} "
      f"{np.median([f['dark_core']['area_pct'] for f in fr['width']]):8.3f}")

print("\nON-SPECIMEN FALSE POSITIVES, six confirmed crack-free specimens "
      "(400k sampled on-specimen px each)")
print(f"{'arm':18s} {'mean on-spec %':>15s} {'worst %':>9s} {'vs baseline':>12s}"
      f" {'specimens better':>17s}")
base = np.array([r["arms"]["baseline"]["on_specimen_fp_pct"] for r in fr["fp"]])
for a in ARMS:
    v = np.array([r["arms"][a]["on_specimen_fp_pct"] for r in fr["fp"]])
    print(f"{a:18s} {v.mean():15.4f} {v.max():9.4f} {v.mean()-base.mean():+12.4f}"
          f" {int((v < base).sum()):14d}/6")
print("\nsame, whole-frame (the number that inverted an earlier conclusion -- not the axis"
      " to use)")
for a in ARMS:
    v = np.array([r["arms"][a]["whole_frame_fp_pct"] for r in fr["fp"]])
    o = np.array([r["arms"][a]["off_specimen_fp_pct"] for r in fr["fp"]])
    print(f"{a:18s} whole-frame {v.mean():7.4f}%   off-specimen {o.mean():7.4f}%")
