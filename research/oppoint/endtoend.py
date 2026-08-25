"""End-to-end through the REAL serving path: does a raw-threshold change survive it?

The serving path is threshold -> prune specks (2000 px) -> fill holes (1024 px) ->
corrections -> tighten (local mean, w=301, declined per frame). A raw threshold that looks
better on pixels can be undone or amplified by those, and one failure mode is specific and
serious: raising the threshold thins a crack, thinning fragments it, and the speck pruner
DELETES the fragments. Pixel IoU can rise while the crack stops being one object.

So this measures area, COMPONENT COUNT, and recall end-to-end, plus true IoU on the densely
labelled frames and on-specimen false positives on the crack-free ones.

NOTE: corrections="none" returns before the tighten block in effective_mask, so tight is
inert in that mode; this uses "gate" (the export/deliverable path) and "paste" (the canvas).

Read-only on app_data. Writes research/oppoint/endtoend.json.
"""
import sys, os, json, time, glob
import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
import store as S, pipeline as P

OUT = os.path.join(P0, "research", "oppoint")
THRESHOLDS = [0.43, 0.50, 0.65]
FRAGS = ["b2_336_25", "b2_338_13", "B2_333_75_um_zoom",          # dense GT
         "HC_316L_fatigue_1400_cycles_idx", "HC_316L_fatigue_800_cycles",
         "HC_316L_fatigue_1200_cycles",                           # thin crack, no GT
         "wrought_316L_fatigue_1200_cycles_crack",
         "b3_amb", "wrought_316L_fatigue_0_cycles"]                # crack-free


def ncomp(m):
    from skimage.measure import label
    return int(label(m, connectivity=2).max())


def main():
    inv = json.load(open(os.path.join(OUT, "inventory.json")))
    gtmap = {}
    for g in sorted(glob.glob(os.path.join(P0, "dataset_cache", "*_gt.npy"))):
        a = np.load(g, mmap_mode="r")
        gtmap[tuple(a.shape)] = g

    res = []
    for frag in FRAGS:
        hit = [r for r in inv["rows"] if frag in r["filename"]]
        if not hit:
            print("MISSING", frag); continue
        r = hit[0]
        iid = r["iid"]
        corr = S.load_npy(iid, "correction.npy")
        gtp = gtmap.get(tuple(r["prob_shape"]))
        gt = np.asarray(np.load(gtp)) > 0 if gtp else None
        raw = S.load_npy(iid, "img.npy", mmap=True)
        spec = P.specimen_support(np.asarray(raw)) if raw is not None else None
        del raw
        print(f"\n=== {r['filename'][:64]}  clean={r['clean']}  "
              f"dense_gt={gtp is not None} ===")
        print(f"{'mode':>6} {'t':>5} {'area%':>8} {'ncomp':>7} {'recall_corr':>12} "
              f"{'onSpecFP%':>10} {'gtIoU':>7} {'gtRec':>7} {'gtPrec':>7}")
        for mode in ("gate", "paste"):
            for t in THRESHOLDS:
                t0 = time.time()
                m = P.effective_mask(iid, threshold=t, corrections=mode, tight=True)
                if m is None:
                    continue
                row = dict(frag=frag, mode=mode, t=t, area=float(m.mean()),
                           ncomp=ncomp(m), secs=round(time.time() - t0, 1),
                           clean=r["clean"])
                if corr is not None and corr.shape == m.shape and (corr == 1).any():
                    row["recall_corr"] = float(m[corr == 1].mean())
                if spec is not None and spec.shape == m.shape:
                    row["on_spec_fp"] = float(m[spec].mean())
                if gt is not None:
                    tp = int((m & gt).sum()); fp = int((m & ~gt).sum())
                    fn = int((~m & gt).sum())
                    row.update(gt_iou=tp / max(tp + fp + fn, 1),
                               gt_recall=tp / max(tp + fn, 1),
                               gt_prec=tp / max(tp + fp, 1))
                res.append(row)
                print(f"{mode:>6} {t:5.2f} {row['area']*100:8.4f} {row['ncomp']:7d} "
                      f"{row.get('recall_corr', float('nan')):12.4f} "
                      f"{row.get('on_spec_fp', float('nan'))*100:10.4f} "
                      f"{row.get('gt_iou', float('nan')):7.4f} "
                      f"{row.get('gt_recall', float('nan')):7.4f} "
                      f"{row.get('gt_prec', float('nan')):7.4f}")
                del m
        del corr, gt, spec

    json.dump(res, open(os.path.join(OUT, "endtoend.json"), "w"), indent=2, default=float)

    # ---- summary: what raising 0.50 -> 0.65 does end-to-end ---------------
    print("\n" + "=" * 78)
    print("SUMMARY (mode=gate): 0.50 -> 0.65 end-to-end")
    print("=" * 78)
    print(f"{'frame':>40} {'dArea%':>8} {'dNcomp':>8} {'dRecall':>9} {'dGtIoU':>8}")
    for frag in FRAGS:
        g = {row["t"]: row for row in res if row["frag"] == frag and row["mode"] == "gate"}
        if 0.50 not in g or 0.65 not in g:
            continue
        a, b = g[0.50], g[0.65]
        dr = (b.get("recall_corr", np.nan) - a.get("recall_corr", np.nan))
        di = (b.get("gt_iou", np.nan) - a.get("gt_iou", np.nan))
        print(f"{frag[:40]:>40} {(b['area']-a['area'])*100:+8.4f} "
              f"{b['ncomp']-a['ncomp']:+8d} {dr:+9.4f} {di:+8.4f}")


if __name__ == "__main__":
    main()
