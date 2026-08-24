"""Render research/contrast/global_results.json into global_SUMMARY.md.

Kept separate from global_contrast_arms.py so the write-up can be regenerated
without re-running the 60-frame experiment.
"""
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "global_results.json")))
A = R["arms"]
ARMS = list(A)
TREAT = [a for a in ARMS if a not in ("identity", "stretch_1_99")]
FP = R["false_positive_guardrail"]
diag = R["transform_diagnostics"]
std = R["standardised_vs_identity"]
try:
    AC = json.load(open(os.path.join(OUT, "global_affine_control.json")))
    floor = AC["float32_noise_floor_max_abs_z_diff"]
except FileNotFoundError:
    AC, floor = None, None


def med(arm, key):
    return float(np.median([diag[i][arm][key] for i in diag]))


def f(x, n=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


# paired fold-wise deltas: every arm saw the identical GroupKFold(5) splits, so
# differencing fold by fold cancels the fold-difficulty variance that dominates
# the raw +- numbers.
base_f = np.array(A["identity"]["all"]["iou"]["folds"])
baset_f = np.array(A["identity"]["thin"]["iou"]["folds"])
D = {}
for a in ARMS:
    d = np.array(A[a]["all"]["iou"]["folds"]) - base_f
    dt = np.array(A[a]["thin"]["iou"]["folds"]) - baset_f
    sd = d.std(ddof=1)
    D[a] = dict(mean=d.mean(), sd=sd, wins=int((d > 0).sum()),
                t=(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else float("nan"),
                thin_mean=dt.mean(), thin_sd=dt.std(ddof=1))

base = A["identity"]["all"]["iou"]["mean"]
spread = max(A[a]["all"]["iou"]["std"] for a in ARMS)
fp_base, fps_base = R["fp_mean"]["identity"], R["fp_mean_on_specimen"]["identity"]
best = max(TREAT, key=lambda a: A[a]["all"]["iou"]["mean"])
best_thin = max(TREAT, key=lambda a: A[a]["thin"]["iou"]["mean"])

L = []
L.append("# GLOBAL contrast arms — does whole-image contrast adjustment help "
         "thin/faint TXM crack detection?")
L.append("")
L.append(f"Generated {R['generated']}. {R['n_frames']} frames, {R['n_rows']:,} rows "
         f"({R['n_crack_rows']:,} crack / {R['n_not_crack_rows']:,} not-crack), "
         f"GroupKFold(5) by frame, {len(ARMS)} arms.")
L.append("")

# ---- verdict, computed not asserted -------------------------------------
if D[best]["mean"] > spread and R["fp_mean_on_specimen"][best] <= fps_base:
    verdict = (f"**Answer: yes, for one arm.** `{best}` gains {D[best]['mean']:+.4f} IoU, "
               f"more than the {spread:.4f} fold-to-fold spread, without raising "
               f"on-specimen false positives.")
else:
    verdict = (
        f"**Answer: no.** No global contrast transform beats the identity baseline. "
        f"The best of them, `{best}`, is {D[best]['mean']:+.4f} IoU "
        f"(paired across folds: sd {D[best]['sd']:.4f}, t={D[best]['t']:.2f}, "
        f"better on {D[best]['wins']}/5 folds) — indistinguishable from doing nothing. "
        f"On THIN frames specifically the best arm is `{best_thin}` at "
        f"{D[best_thin]['thin_mean']:+.4f} IoU, also indistinguishable. Every arm that "
        f"is *decisively* different from baseline is decisively **worse**, and the "
        f"ordering tracks how much of the absolute intensity scale the transform "
        f"destroys.")
L.append(verdict)
L.append("")
L.append("Two structural facts have to be stated before the table, because they change "
         "what counts as a null result here:")
L.append("")
L.append("1. **`img.npy` is already a 1–99 percentile stretch.** "
         "`pipeline.py` builds the model input as `robust_normalize(raw, 1.0, 99.0)`, so "
         "exactly 1% of every frame sits at 0.0 and 1% at 1.0. Re-applying a 1–99 stretch "
         "is a numerical no-op, and `stretch_1_99` is therefore a **harness control**, not "
         "a treatment. It must reproduce `identity` — it does, to "
         f"{abs(D['stretch_1_99']['mean']):.4f} IoU.")
L.append("2. **An affine transform is invisible to this model by construction.** For "
         "`x -> a*x + b` (a>0) every one of the 17 features is affine in its untransformed "
         "counterpart — intensity and the 6 Gaussian smooths pick up `(a, b)`; the 4 "
         "gradient magnitudes, 4 Laplacians and 2 local-stds pick up `(a, 0)` because they "
         "annihilate constants — and StandardScaler standardises each column "
         "independently. So for an affine arm an unchanged score is the **correct** answer, "
         "not a failed measurement. Only the non-linear part of a transform (gamma, "
         "equalisation, sigmoid, and saturation from clipping) can move this model at all.")
L.append("")

# ---- main table ----------------------------------------------------------
L.append("## Arm vs score")
L.append("")
L.append("| arm | all-rows IoU | prec | recall | THIN-frame IoU | thin prec | thin rec | "
         "crack-free FP, whole frame | crack-free FP, **on specimen** |")
L.append("|---|---|---|---|---|---|---|---|---|")
for a in ARMS:
    al, th = A[a]["all"], A[a]["thin"]
    tag = {"identity": " ← baseline", "stretch_1_99": " ← control"}.get(a, "")
    L.append(f"| `{a}`{tag} | {f(al['iou']['mean'])} ±{f(al['iou']['std'])} | "
             f"{f(al['precision']['mean'])} | {f(al['recall']['mean'])} | "
             f"{f(th['iou']['mean'])} ±{f(th['iou']['std'])} | "
             f"{f(th['precision']['mean'])} | {f(th['recall']['mean'])} | "
             f"{f(R['fp_mean'][a])} | {f(R['fp_mean_on_specimen'][a])} |")
L.append("")
L.append(f"± is the std across the 5 folds; the largest for any arm is **{spread:.4f} IoU**. "
         "Fold difficulty dominates that number, so the table below differences each arm "
         "against `identity` *within* each fold instead — the splits are identical across "
         "arms, which makes the paired comparison far more sensitive.")
L.append("")

# ---- paired table -------------------------------------------------------
L.append("## Paired against identity, fold by fold")
L.append("")
L.append("| arm | Δ IoU (paired) | sd | t | folds better | Δ THIN IoU | reading |")
L.append("|---|---|---|---|---|---|---|")
for a in ARMS:
    if a == "identity":
        continue
    d = D[a]
    if abs(d["t"]) < 2:
        rd = "indistinguishable"
    elif d["mean"] < 0:
        rd = "**worse**"
    else:
        rd = "better"
    L.append(f"| `{a}` | {d['mean']:+.4f} | {d['sd']:.4f} | {d['t']:+.2f} | "
             f"{d['wins']}/5 | {d['thin_mean']:+.4f} | {rd} |")
L.append("")
L.append("`dark_stretch_p0.5_p40` is the informative failure. It saturates ~60% of every "
         "frame at 1.0 to spend the dynamic range on the dark end where cracks live, and it "
         f"costs **{abs(D['dark_stretch_p0.5_p40']['mean']):.3f} IoU** — on 0/5 folds is it "
         "better, and the loss is almost entirely recall "
         f"({A['dark_stretch_p0.5_p40']['all']['recall']['mean']:.4f} vs "
         f"{A['identity']['all']['recall']['mean']:.4f}) at unchanged precision. That is "
         "the same mechanism, and very nearly the same magnitude, as the 0.169 IoU that "
         "flat-fielding cost this project (docs/MARKUP_GUIDE.md): destroy the large-radius "
         "absolute-intensity features and the model stops finding crack. Enhancing local "
         "contrast does not compensate for it.")
L.append("")

# ---- the guardrail trap -------------------------------------------------
L.append("## The false-positive guardrail reverses when you restrict to the specimen")
L.append("")
L.append("This is the part worth not skimming. Read whole-frame FP alone and "
         "`equalize_hist` looks like a big win — a 4x drop. But these frames are only "
         f"~{np.mean([FP['identity'][i]['support_frac'] for i in FP['identity']]):.0%} "
         "specimen; the rest is off-specimen background, which is dark and therefore easy "
         "to call crack. Restricting to `pipeline.specimen_support`, the ranking inverts:")
L.append("")
L.append("| arm | FP whole frame | vs identity | FP on specimen | vs identity |")
L.append("|---|---|---|---|---|")
for a in ARMS:
    w, s = R["fp_mean"][a], R["fp_mean_on_specimen"][a]
    L.append(f"| `{a}` | {w:.4f} | {w/fp_base:5.2f}x | {s:.4f} | {s/fps_base:5.2f}x |")
L.append("")
L.append("So `equalize_hist` and `sigmoid_med_g10` cut false positives on the *background* "
         "while making them **2.4–3x worse on the specimen itself**, which is the only "
         "place a false positive matters. `gamma_0.5`'s whole-frame improvement "
         f"({fp_base:.4f} → {R['fp_mean']['gamma_0.5']:.4f}) is likewise entirely "
         f"off-specimen — on the specimen it is {R['fp_mean_on_specimen']['gamma_0.5']:.4f} "
         f"vs {fps_base:.4f}, i.e. unchanged. And `dark_stretch`'s low on-specimen FP is "
         "not virtue but sedation: it predicts less of everything, hence the recall "
         "collapse.")
L.append("")
L.append("Per specimen, `whole-frame / on-specimen`:")
L.append("")
ids = list(FP[ARMS[0]])
L.append("| specimen | specimen frac | " + " | ".join(f"`{a}`" for a in ARMS) + " |")
L.append("|---" * (len(ARMS) + 2) + "|")
for i in ids:
    L.append(f"| {FP[ARMS[0]][i]['filename'][:46]} | "
             f"{FP[ARMS[0]][i]['support_frac']:.3f} | "
             + " | ".join(f"{FP[a][i]['fp_frac']:.3f} / {FP[a][i]['fp_frac_on_specimen']:.3f}"
                          for a in ARMS) + " |")
L.append("")
L.append("Absolute FP levels here are **not** comparable to the production detector's "
         "~0.02% on these specimens: this model is trained on a 50/50-balanced 8k+8k pixel "
         "sample with no post-processing, so it sits at a far more permissive operating "
         "point by design. Only the arm-vs-arm comparison is meaningful.")
L.append("")

# ---- affinity -----------------------------------------------------------
L.append("## Which arms can the model even see?")
L.append("")
L.append(f"Per-frame least-squares fit of `transformed ~ a*img01 + b`, median over the "
         f"{len(diag)} frames, plus the largest standardised-feature difference against "
         "`identity`.")
L.append("")
L.append("| arm | slope a | max affine residual | saturated at 0 | at 1 | "
         "max&#124;z − z_identity&#124; | verdict |")
L.append("|---|---|---|---|---|---|---|")
for a in ARMS:
    r = med(a, "max_affine_residual")
    z = std[a]["max_abs_diff_standardised"]
    v = ("**affine → invisible to the model**" if floor and z <= floor
         else "non-linear → the model can see it")
    L.append(f"| `{a}` | {med(a,'slope'):.4g} | {r:.3g} | {med(a,'frac_at_0'):.3f} | "
             f"{med(a,'frac_at_1'):.3f} | {z:.3g} | {v} |")
L.append("")
if AC:
    L.append(f"The threshold for that verdict is measured, not assumed "
             f"(`global_affine_control.py` → `global_affine_control.json`): a *provably* "
             f"affine map still shifts standardised features by up to "
             f"**{floor:.3g}**, all of it float32 cancellation noise in `local_std`, which "
             f"evaluates `sqrt(E[x²] − E[x]²)`. `stretch_1_99` comes in at "
             f"{std['stretch_1_99']['max_abs_diff_standardised']:.3g} — *below* that floor, "
             f"so it is a genuine no-op — while `gamma_2.0` "
             f"({std['gamma_2.0']['max_abs_diff_standardised']:.3g}) and `equalize_hist` "
             f"({std['equalize_hist']['max_abs_diff_standardised']:.3g}) are three orders "
             f"of magnitude above it and are real changes.")
    L.append("")

# ---- thin frames --------------------------------------------------------
thin = [fr for fr in R["frames"] if fr["thin"]]
thick = [fr for fr in R["frames"] if not fr["thin"]]
hw = sorted(fr["halfwidth_px"] for fr in R["frames"] if fr["halfwidth_px"] is not None)
L.append(f"## THIN frames ({len(thin)} of {R['n_frames']})")
L.append("")
L.append("Median half-width of the dark core inside the painted strokes: pixels darker than "
         "the 20th percentile of `img01` within `correction==1`, components under 64 px "
         "dropped, `distance_transform_edt` read along the `skeletonize` of that set. "
         f"THIN means ≤ {R['protocol']['thin_max_halfwidth_px']} px. Computed on the "
         "untransformed `img01`, so the split is identical for every arm. All 5 folds "
         "contain thin frames.")
L.append("")
L.append(f"Range {hw[0]:.2f}–{hw[-1]:.2f} px. Caveat: "
         f"{sum(1 for x in hw if x == 1.0)} frames sit exactly at the 1.0 px floor, so the "
         f"thin class is partly saturated — taking the darkest fifth of a painted stroke "
         f"tends to yield a 1-px thread whatever the crack's true width. The thin/thick "
         f"contrast is still real (the thick end reaches {hw[-1]:.1f} px), but 'THIN' here "
         f"means 'thin dark core', which is not quite the same as 'thin crack'.")
L.append("")
L.append("| frame | half-width px |")
L.append("|---|---|")
for fr in sorted(thin, key=lambda x: x["halfwidth_px"]):
    L.append(f"| {fr['filename']} | {fr['halfwidth_px']:.2f} |")
L.append("")
L.append(f"<details><summary>the {len(thick)} non-THIN frames</summary>")
L.append("")
L.append("| frame | half-width px |")
L.append("|---|---|")
for fr in sorted(thick, key=lambda x: -(x["halfwidth_px"] or 0)):
    L.append(f"| {fr['filename']} | {f(fr['halfwidth_px'], 2)} |")
L.append("")
L.append("</details>")
L.append("")

# ---- cost ---------------------------------------------------------------
t = R["timing_sec"]
L.append("## Cost")
L.append("")
L.append("| stage | sec |")
L.append("|---|---|")
for k, v in t.items():
    L.append(f"| {k} | {v:,.1f} |")
L.append("")
L.append(f"**{t['total_sec']/60:.1f} min** total on 6 worker processes: "
         f"{R['n_frames']} labelled frames + 6 crack-free specimens x {len(ARMS)} arms of "
         f"17-feature extraction ({(R['n_frames']+6)*len(ARMS)} full-frame stacks, "
         f"740 MP per arm), {len(ARMS)*5} MLP fold-fits, {len(ARMS)} full-data fits. "
         f"Caveat: five other CPU-heavy jobs from parallel arms of this study were running "
         f"on the same 14-core machine throughout, so this is an upper bound rather than a "
         f"clean cost measurement.")
L.append("")

# ---- conclusion ---------------------------------------------------------
L.append("## Conclusion")
L.append("")
L.append(
    "Global contrast adjustment does not help this detector find thin or faint cracks. "
    "The one arm that is nominally above baseline, `gamma_0.5`, is "
    f"{D['gamma_0.5']['mean']:+.4f} IoU with a paired t of {D['gamma_0.5']['t']:.2f} over 5 "
    f"folds and {D['gamma_0.5']['thin_mean']:+.4f} on thin frames — a coin flip, not an "
    "effect; and its apparent 1.7x false-positive improvement lives entirely on the "
    "off-specimen background, vanishing once you restrict to the specimen. Everything that "
    "moves the score significantly moves it down, in an order that tracks how much "
    "absolute intensity the transform destroys: sigmoid and gamma 2.0 (mild, compressive) "
    "cost ~0.01–0.02, histogram equalisation (flattens the histogram globally) costs "
    f"{abs(D['equalize_hist']['mean']):.3f} on 0/5 folds, and my own `dark_stretch` arm — "
    "the most aggressive contrast enhancement of the set, saturating 60% of the frame to "
    f"buy 2.5x contrast in the dark tail — costs {abs(D['dark_stretch_p0.5_p40']['mean']):.3f}, "
    "reproducing the flat-fielding failure (0.169) almost exactly and by the same "
    "mechanism. The 1–99 stretch arm is a no-op because the model input is already 1–99 "
    "stretched, and its null result is a *verification* of the harness rather than a "
    "finding. The mechanistic reading is consistent throughout: this model's signal is "
    "largely *absolute, large-radius* darkness, and a whole-image transfer function can "
    "only redistribute that — it adds no information, while any non-linearity it applies "
    "degrades the monotone relationship between a pixel's brightness and how deeply it "
    "sits inside a dark region. Contrast is not the bottleneck on thin cracks. If "
    "anything here is worth following up it is the guardrail artefact, not a transform: "
    "whole-frame false-positive rates on these specimens are dominated by off-specimen "
    "background and rank arms in the opposite order to on-specimen rates.")
L.append("")

with open(os.path.join(OUT, "global_SUMMARY.md"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print("wrote global_SUMMARY.md")
for a in ARMS:
    print(f"{a:24s} IoU {A[a]['all']['iou']['mean']:.4f} d={D[a]['mean']:+.4f} "
          f"t={D[a]['t']:+6.2f} thin {A[a]['thin']['iou']['mean']:.4f} "
          f"fp {R['fp_mean'][a]:.4f} fpspec {R['fp_mean_on_specimen'][a]:.4f}")
