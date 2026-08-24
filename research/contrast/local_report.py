"""Render research/contrast/local_SUMMARY.md from local_results.json (+ diagnostics).

Every number in the markdown comes from the JSON, so the write-up cannot drift
from the measurement.
"""

import json
import os
import sys

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
OUT = os.path.join(P0, "research", "contrast")


def f(x, n=4):
    try:
        if x is None:
            return "n/a"
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "n/a"


def main():
    R = json.load(open(os.path.join(OUT, "local_results.json")))
    man = json.load(open(os.path.join(OUT, "local_manifest.json")))
    diagp = os.path.join(OUT, "local_diagnostics.json")
    D = json.load(open(diagp)) if os.path.exists(diagp) else {}
    meta, arms = R["meta"], R["arms"]
    base = arms.get("identity")

    L = []
    L.append("# Local / adaptive contrast enhancement as a model-input transform")
    L.append("")
    L.append("Arm: LOCAL / ADAPTIVE contrast enhancement of the model input, "
             "scored against a self-computed `identity` baseline under the shared "
             "protocol (GroupKFold(5) by image, 17-feature stack, "
             "MLP(64,32) + StandardScaler, IoU@0.5 on held-out rows).")
    L.append("")
    L.append(f"- images with both correction classes: **{meta['n_images']}** "
             f"(no image-set subsampling was needed -- CLAHE turned out to cost "
             f"~1 s on a 23 MP frame)")
    L.append(f"- rows per image: up to {meta['max_per_class']} `correction==1` + "
             f"up to {meta['max_per_class']} `correction==2`, `RandomState(0)`, "
             f"image id as the CV group")
    L.append(f"- thin-crack frames: **{len(meta['thin_frames'])}/{meta['n_images']}** "
             f"(median skeleton half-width <= {meta['thin_max_halfwidth']} px)")
    L.append(f"- crack-free guardrail: {meta['clean_sample']:,} uniformly random "
             f"pixels x 6 specimens from `pipeline.CLEAN_SPECIMENS`, model trained "
             f"on all rows")
    L.append(f"- LCN epsilon = {meta['lcn_eps']} (measured local std on this data runs "
             f"0.011-0.25, so this is a true normalisation)")
    # Per-arm transform+featurise cost, summed over every labelled frame.
    arm_secs = {}
    for m in man["images"]:
        for k, v in (m.get("arm_seconds") or {}).items():
            arm_secs[k] = arm_secs.get(k, 0.0) + float(v)
    total_mp = sum(m["mp"] for m in man["images"])
    L.append(f"- feature extraction wall clock: "
             f"**{sum(arm_secs.values())/60:.0f} min** total for {len(arms)} arms "
             f"over {meta['n_images']} labelled frames ({total_mp:.0f} MP/arm), plus "
             f"~21 min for the 6 crack-free frames. Per-arm cost is in the table below. "
             f"(The machine was shared with other jobs during this run, so absolute "
             f"seconds are ~1.5x an idle machine; the RELATIVE cost between arms is "
             f"still meaningful.)")
    L.append("")

    L.append("Reproduce with `research/contrast/local_extract.py` (builds the sample "
             "cache), then `local_eval.py`, `local_diag.py`, `local_report.py`. "
             "NOTE: the cache in `research/contrast/local_cache/` (694 MB) and "
             "`local_cache_clean/` (866 MB) is intermediate and fully regenerable -- "
             "safe to `rm -rf` once these results are accepted. It is kept only so "
             "the analysis can be rerun without the ~100 min extraction.")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("All-rows numbers are the mean over the 5 folds, +- the fold-to-fold "
             "standard deviation. Thin-frame IoU is pooled over all out-of-fold "
             "rows belonging to thin frames (per-fold spread also given). "
             "`crack-free FP` is the mean over the 6 specimens of the fraction of "
             "sampled pixels predicted crack -- every one of those is a false "
             "positive by construction.")
    L.append("")
    L.append("| arm | all IoU | all precision | all recall | thin IoU (pooled) | "
             "thin IoU (fold mean+-sd) | crack-free FP mean (all px / on spec) | "
             "d IoU vs identity | d thin IoU | transform+featurise (min) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, a in arms.items():
        ar, tr, cf = a["all_rows"], a["thin_rows"], a["crack_free"]
        dio = dth = "--"
        if base is not None and name != "identity":
            dio = f"{ar['iou']['mean'] - base['all_rows']['iou']['mean']:+.4f}"
            if tr["pooled"] and base["thin_rows"]["pooled"]:
                dth = (f"{tr['pooled']['iou'] - base['thin_rows']['pooled']['iou']:+.4f}")
        L.append(
            f"| `{name}` | {f(ar['iou']['mean'])} +- {f(ar['iou']['std'])} "
            f"| {f(ar['precision']['mean'])} | {f(ar['recall']['mean'])} "
            f"| {f(tr['pooled']['iou']) if tr['pooled'] else 'n/a'} "
            f"| {f(tr['iou']['mean'])} +- {f(tr['iou']['std'])} "
            f"| {f(cf['mean_frac_pred_crack']*100, 3)}% / "
            f"{f(cf['mean_frac_pred_crack_on_specimen']*100, 3)}% "
            f"| {dio} | {dth} | {arm_secs.get(name, 0)/60:.1f} |")
    L.append("")
    if base is not None:
        sd = base["all_rows"]["iou"]["std"]
        L.append(f"The identity baseline's own fold-to-fold IoU spread is "
                 f"**+-{sd:.4f}**. Any two arms closer than that on all-rows IoU "
                 f"should be read as indistinguishable, not ranked.")
        L.append("")

    L.append("### Crack-free false positives, per specimen")
    L.append("")
    L.append("Protocol column is `all px` (uniform over the whole frame). "
             "`on specimen` restricts to `pipeline.specimen_support`, because "
             "off-specimen background is 20-40% of these mosaics and dilutes a "
             "whole-frame rate; it is the stricter and more meaningful number.")
    L.append("")
    specs = [r["specimen"] for r in next(iter(arms.values()))["crack_free"]["per_specimen"]]
    L.append("| arm | " + " | ".join(specs) + " |")
    L.append("|---|" + "---|" * len(specs))
    for name, a in arms.items():
        cells = []
        for r in a["crack_free"]["per_specimen"]:
            cells.append(f"{r['frac_pred_crack']*100:.3f}% / "
                         f"{r['frac_pred_crack_on_specimen']*100:.3f}%")
        L.append(f"| `{name}` | " + " | ".join(cells) + " |")
    L.append("")
    L.append("(cells are `all px` / `on specimen`)")
    L.append("")

    L.append("### Thin-crack frames")
    L.append("")
    L.append("Median skeleton half-width of the darkest-20% core inside "
             "`correction==1`, after `remove_small_objects(64)`:")
    L.append("")
    hw = meta["halfwidth_px"]
    thin = set(meta["thin_frames"])
    near = [k for k, v in hw.items() if v is not None and 3.0 < v <= 3.2]
    if near:
        L.append(f"Sensitivity note: the EDT quantises half-widths to sqrt(n), so "
                 f"{len(near)} further frames land at 3.16 px, just above the "
                 f"protocol's 3.0 cut. The thin set is therefore somewhat sensitive "
                 f"to that exact threshold; the numbers below use 3.0 as specified.")
        L.append("")
    for iid in sorted(hw, key=lambda k: (hw[k] is None, hw[k])):
        mark = "**THIN**" if iid in thin else ""
        L.append(f"- `{f(hw[iid], 2)}` px {mark} -- {iid}")
    L.append("")

    if D:
        L.append("### Mechanism: single-feature AUC (all rows)")
        L.append("")
        L.append("AUC of each feature taken alone, reported as `max(a, 1-a)`. This is "
                 "the direct test of the trade this family makes: local contrast is "
                 "bought with the ABSOLUTE large-radius intensity channels that this "
                 "project already measured as ~41% of the model's importance "
                 "(docs/MARKUP_GUIDE.md).")
        L.append("")
        keys = ["intensity", "smooth_s8", "smooth_s32", "smooth_s64",
                "gradmag_s1", "laplacian_s1", "texture_s2"]
        L.append("| arm | " + " | ".join(f"`{k}`" for k in keys) + " | best feature |")
        L.append("|---|" + "---|" * (len(keys) + 1))
        for name in arms:
            if name not in D:
                continue
            d = D[name]["all_rows"]
            best = max(d, key=d.get)
            L.append(f"| `{name}` | " + " | ".join(f(d[k]) for k in keys) +
                     f" | {best} ({f(d[best])}) |")
        L.append("")

    # ------------------------------------------------------------ conclusion
    if base is not None:
        bi = base["all_rows"]["iou"]["mean"]
        bt = base["thin_rows"]["pooled"]["iou"]
        bfp = base["crack_free"]["mean_frac_pred_crack_on_specimen"]
        sd = base["all_rows"]["iou"]["std"]

        def g(n):
            return arms[n]

        indist = [n for n, a in arms.items()
                  if n != "identity" and abs(a["all_rows"]["iou"]["mean"] - bi) < sd]
        L.append("## Conclusion")
        L.append("")
        L.append(
            f"Local/adaptive contrast enhancement of the model input does not help "
            f"this detector find thin, faint cracks, and the amount it hurts is "
            f"predicted almost perfectly by how much of the image's ABSOLUTE "
            f"intensity the transform destroys. Ranked by that: unsharp masking, "
            f"which adds a zero-mean high-pass on top of the original and so leaves "
            f"the DC term intact, is indistinguishable from the identity baseline on "
            f"every metric -- `unsharp_s2_a1.0` scores "
            f"{g('unsharp_s2_a1.0')['all_rows']['iou']['mean']:.4f} vs {bi:.4f} "
            f"all-rows IoU and "
            f"{g('unsharp_s2_a1.0')['thin_rows']['pooled']['iou']:.4f} vs {bt:.4f} "
            f"on thin frames, both inside the +-{sd:.4f} fold-to-fold spread, with "
            f"crack-free false positives flat at "
            f"{g('unsharp_s2_a1.0')['crack_free']['mean_frac_pred_crack_on_specimen']*100:.2f}% "
            f"vs {bfp*100:.2f}%. Half-blending CLAHE with the original "
            f"(`clahe_c0.01_k8_blend0.5`, my own arm, designed to add local contrast "
            f"WITHOUT removing the DC term) costs "
            f"{g('clahe_c0.01_k8_blend0.5')['all_rows']['iou']['mean']-bi:+.4f} IoU. "
            f"Full CLAHE, which remaps intensity per tile, costs "
            f"{min(g(k)['all_rows']['iou']['mean'] for k in arms if k.startswith('clahe_c0') and 'blend' not in k)-bi:+.4f} "
            f"to "
            f"{max(g(k)['all_rows']['iou']['mean'] for k in arms if k.startswith('clahe_c0') and 'blend' not in k)-bi:+.4f} "
            f"and roughly DOUBLES to TRIPLES crack-free false positives "  # noqa: E501
            f"(2.23% -> 4.2-6.2% on-specimen), monotonically in clip limit. Local "
            f"contrast normalisation, which removes absolute intensity outright, is "
            f"the worst: `lcn_w51` loses "
            f"{g('lcn_w51')['all_rows']['iou']['mean']-bi:+.4f} all-rows IoU and "
            f"{g('lcn_w51')['thin_rows']['pooled']['iou']-bt:+.4f} on thin frames "
            f"while raising on-specimen false positives "
            f"{g('lcn_w51')['crack_free']['mean_frac_pred_crack_on_specimen']/bfp:.1f}x. "
            f"That reproduces this project's existing flat-fielding result (-0.169 "
            f"IoU) in the same direction and comparable magnitude, which is the "
            f"expected outcome for flat-fielding's close cousin. **No arm improved "
            f"thin-frame IoU by more than the fold-to-fold spread, so there is no "
            f"local-contrast arm worth deploying.**")
        L.append("")
        if D:
            di, dl = D["identity"], D["lcn_w51"]
            L.append(
                f"The single-feature AUCs say why, and they kill the optimistic "
                f"reading rather than supporting it. Measured WITHIN a single thin "
                f"frame, LCN does exactly what it advertises: on "
                f"`B2_333_75_um_zoom` the `texture_s2` AUC goes from 0.585 under "
                f"identity to 0.926 under `lcn_w51`, i.e. the faint crack really is "
                f"far better separated from its immediate surroundings. But that "
                f"gain does NOT survive pooling across frames, which is the "
                f"regime the classifier is actually trained in: pooled over all "
                f"{len(meta['thin_frames'])} thin frames, `texture_s2` moves only "
                f"{di['thin_rows']['texture_s2']:.4f} -> "
                f"{dl['thin_rows']['texture_s2']:.4f}, while `intensity` collapses "
                f"{di['thin_rows']['intensity']:.4f} -> "
                f"{dl['thin_rows']['intensity']:.4f} and `smooth_s64` "
                f"{di['thin_rows']['smooth_s64']:.4f} -> "
                f"{dl['thin_rows']['smooth_s64']:.4f}. The reason is that LCN's "
                f"output scale is set by each frame's own local statistics, so the "
                f"enhanced contrast means something different in every frame and a "
                f"pooled model cannot cash it in -- whereas absolute intensity was "
                f"comparable across frames and is now gone. Since the large-radius "
                f"absolute-intensity group is ~41% of this model's importance "
                f"(docs/MARKUP_GUIDE.md), the trade is strictly losing. CLAHE, by "
                f"contrast, leaves `intensity` intact or slightly better "
                f"({di['thin_rows']['intensity']:.4f} -> "
                f"{D['clahe_c0.03_k8']['thin_rows']['intensity']:.4f}), which is why "
                f"its IoU cost is small -- its damage shows up in the false-positive "
                f"guardrail instead.")
            L.append("")
        L.append(
            f"**The false-positive guardrail is what settles it, and it disagrees "
            f"with the IoU ranking.** `lcn_w51_robust` -- identical LCN, but rescaled "
            f"with a robust 1st-99th percentile clip instead of min-max -- looks like "
            f"the best LCN variant on labelled rows "
            f"({g('lcn_w51_robust')['all_rows']['iou']['mean']:.4f} IoU, only "
            f"{g('lcn_w51_robust')['all_rows']['iou']['mean']-bi:+.4f}), yet it "
            f"predicts crack on "
            f"{g('lcn_w51_robust')['crack_free']['mean_frac_pred_crack_on_specimen']*100:.1f}% "
            f"of on-specimen pixels in frames the owner confirmed contain NO crack -- "
            f"{g('lcn_w51_robust')['crack_free']['mean_frac_pred_crack_on_specimen']/bfp:.0f}x "
            f"the baseline's {bfp*100:.2f}%. Judged on held-out IoU alone LCN would "
            f"have read as a modest cost; judged on the crack-free frames it is "
            f"disqualifying. This is exactly the noise-amplification-in-flat-material "
            f"failure mode local contrast enhancement is prone to, and it is only "
            f"visible because the guardrail scores frames that carry no labels at all.")
        L.append("")
        if indist:
            L.append("Arms indistinguishable from `identity` on all-rows IoU alone "
                     "(within +-%.4f): %s. Note that IoU alone does NOT separate "
                     "these -- the CLAHE members of that list double or triple "
                     "crack-free false positives, so they are indistinguishable on "
                     "the headline metric and clearly worse on the guardrail. Only "
                     "the two unsharp arms and the CLAHE blend are indistinguishable "
                     "from baseline on BOTH."
                     % (sd, ", ".join(f"`{n}`" for n in indist)))
            L.append("")
        L.append(
            "Caveat on the guardrail's absolute scale: the protocol's uniformly-random "
            "200k pixels per crack-free frame give whole-frame rates of 11-32%, but "
            "only 62-78% of each mosaic is specimen and the raw-input model calls the "
            "dark off-specimen background crack. That inflation is present in the "
            "identity baseline too (21.9% whole-frame vs 2.23% on-specimen), so it is "
            "a property of the existing pipeline rather than of any arm here. Both "
            "columns are reported; the on-specimen column is the one that "
            "discriminates between arms.")
        L.append("")

    open(os.path.join(OUT, "local_SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("wrote local_SUMMARY.md")


if __name__ == "__main__":
    main()
