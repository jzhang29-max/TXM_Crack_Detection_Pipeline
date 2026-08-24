# GLOBAL contrast arms — does whole-image contrast adjustment help thin/faint TXM crack detection?

Generated 2026-08-24 14:07:30. 60 frames, 960,000 rows (480,000 crack / 480,000 not-crack), GroupKFold(5) by frame, 7 arms.

**Answer: no.** No global contrast transform beats the identity baseline. The best of them, `gamma_0.5`, is +0.0026 IoU (paired across folds: sd 0.0102, t=0.57, better on 3/5 folds) — indistinguishable from doing nothing. On THIN frames specifically the best arm is `gamma_0.5` at -0.0001 IoU, also indistinguishable. Every arm that is *decisively* different from baseline is decisively **worse**, and the ordering tracks how much of the absolute intensity scale the transform destroys.

Two structural facts have to be stated before the table, because they change what counts as a null result here:

1. **`img.npy` is already a 1–99 percentile stretch.** `pipeline.py` builds the model input as `robust_normalize(raw, 1.0, 99.0)`, so exactly 1% of every frame sits at 0.0 and 1% at 1.0. Re-applying a 1–99 stretch is a numerical no-op, and `stretch_1_99` is therefore a **harness control**, not a treatment. It must reproduce `identity` — it does, to 0.0016 IoU.
2. **An affine transform is invisible to this model by construction.** For `x -> a*x + b` (a>0) every one of the 17 features is affine in its untransformed counterpart — intensity and the 6 Gaussian smooths pick up `(a, b)`; the 4 gradient magnitudes, 4 Laplacians and 2 local-stds pick up `(a, 0)` because they annihilate constants — and StandardScaler standardises each column independently. So for an affine arm an unchanged score is the **correct** answer, not a failed measurement. Only the non-linear part of a transform (gamma, equalisation, sigmoid, and saturation from clipping) can move this model at all.

## Arm vs score

| arm | all-rows IoU | prec | recall | THIN-frame IoU | thin prec | thin rec | crack-free FP, whole frame | crack-free FP, **on specimen** |
|---|---|---|---|---|---|---|---|---|
| `identity` ← baseline | 0.6667 ±0.0275 | 0.8104 | 0.7898 | 0.6239 ±0.0292 | 0.7888 | 0.7491 | 0.2191 | 0.0223 |
| `stretch_1_99` ← control | 0.6651 ±0.0248 | 0.8080 | 0.7899 | 0.6213 ±0.0247 | 0.7853 | 0.7489 | 0.2105 | 0.0222 |
| `gamma_0.5` | 0.6693 ±0.0308 | 0.8174 | 0.7866 | 0.6238 ±0.0344 | 0.7975 | 0.7410 | 0.1286 | 0.0216 |
| `gamma_2.0` | 0.6516 ±0.0436 | 0.7914 | 0.7859 | 0.6066 ±0.0520 | 0.7608 | 0.7488 | 0.2140 | 0.0241 |
| `equalize_hist` | 0.6441 ±0.0288 | 0.8086 | 0.7604 | 0.5866 ±0.0460 | 0.7746 | 0.7076 | 0.0567 | 0.0649 |
| `dark_stretch_p0.5_p40` | 0.4829 ±0.0182 | 0.8046 | 0.5477 | 0.4854 ±0.0640 | 0.7866 | 0.5563 | 0.1821 | 0.0098 |
| `sigmoid_med_g10` | 0.6577 ±0.0287 | 0.8147 | 0.7735 | 0.6048 ±0.0317 | 0.7807 | 0.7292 | 0.1717 | 0.0543 |

± is the std across the 5 folds; the largest for any arm is **0.0436 IoU**. Fold difficulty dominates that number, so the table below differences each arm against `identity` *within* each fold instead — the splits are identical across arms, which makes the paired comparison far more sensitive.

## Paired against identity, fold by fold

| arm | Δ IoU (paired) | sd | t | folds better | Δ THIN IoU | reading |
|---|---|---|---|---|---|---|
| `stretch_1_99` | -0.0016 | 0.0035 | -1.00 | 2/5 | -0.0026 | indistinguishable |
| `gamma_0.5` | +0.0026 | 0.0102 | +0.57 | 3/5 | -0.0001 | indistinguishable |
| `gamma_2.0` | -0.0151 | 0.0215 | -1.57 | 1/5 | -0.0173 | indistinguishable |
| `equalize_hist` | -0.0226 | 0.0189 | -2.67 | 0/5 | -0.0373 | **worse** |
| `dark_stretch_p0.5_p40` | -0.1838 | 0.0394 | -10.43 | 0/5 | -0.1384 | **worse** |
| `sigmoid_med_g10` | -0.0090 | 0.0366 | -0.55 | 3/5 | -0.0191 | indistinguishable |

`dark_stretch_p0.5_p40` is the informative failure. It saturates ~60% of every frame at 1.0 to spend the dynamic range on the dark end where cracks live, and it costs **0.184 IoU** — on 0/5 folds is it better, and the loss is almost entirely recall (0.5477 vs 0.7898) at unchanged precision. That is the same mechanism, and very nearly the same magnitude, as the 0.169 IoU that flat-fielding cost this project (docs/MARKUP_GUIDE.md): destroy the large-radius absolute-intensity features and the model stops finding crack. Enhancing local contrast does not compensate for it.

## The false-positive guardrail reverses when you restrict to the specimen

This is the part worth not skimming. Read whole-frame FP alone and `equalize_hist` looks like a big win — a 4x drop. But these frames are only ~69% specimen; the rest is off-specimen background, which is dark and therefore easy to call crack. Restricting to `pipeline.specimen_support`, the ranking inverts:

| arm | FP whole frame | vs identity | FP on specimen | vs identity |
|---|---|---|---|---|
| `identity` | 0.2191 |  1.00x | 0.0223 |  1.00x |
| `stretch_1_99` | 0.2105 |  0.96x | 0.0222 |  0.99x |
| `gamma_0.5` | 0.1286 |  0.59x | 0.0216 |  0.97x |
| `gamma_2.0` | 0.2140 |  0.98x | 0.0241 |  1.08x |
| `equalize_hist` | 0.0567 |  0.26x | 0.0649 |  2.91x |
| `dark_stretch_p0.5_p40` | 0.1821 |  0.83x | 0.0098 |  0.44x |
| `sigmoid_med_g10` | 0.1717 |  0.78x | 0.0543 |  2.44x |

So `equalize_hist` and `sigmoid_med_g10` cut false positives on the *background* while making them **2.4–3x worse on the specimen itself**, which is the only place a false positive matters. `gamma_0.5`'s whole-frame improvement (0.2191 → 0.1286) is likewise entirely off-specimen — on the specimen it is 0.0216 vs 0.0223, i.e. unchanged. And `dark_stretch`'s low on-specimen FP is not virtue but sedation: it predicts less of everything, hence the recall collapse.

Per specimen, `whole-frame / on-specimen`:

| specimen | specimen frac | `identity` | `stretch_1_99` | `gamma_0.5` | `gamma_2.0` | `equalize_hist` | `dark_stretch_p0.5_p40` | `sigmoid_med_g10` |
|---|---|---|---|---|---|---|---|---|
| Average_mosaic_260618_B2_2_1_lbf_idx00000_mosa | 0.627 | 0.150 / 0.019 | 0.137 / 0.017 | 0.093 / 0.017 | 0.147 / 0.018 | 0.041 / 0.031 | 0.074 / 0.006 | 0.047 / 0.039 |
| Average_mosaic_260618_B2_2_9_lbf_idx00000_mosa | 0.617 | 0.181 / 0.021 | 0.172 / 0.021 | 0.110 / 0.020 | 0.188 / 0.020 | 0.029 / 0.034 | 0.137 / 0.004 | 0.043 / 0.042 |
| Average_mosaic_260618_B2_amb_mosaic_2_idx00000 | 0.737 | 0.210 / 0.021 | 0.198 / 0.021 | 0.114 / 0.021 | 0.221 / 0.023 | 0.026 / 0.024 | 0.164 / 0.012 | 0.264 / 0.031 |
| Average_mosaic_260618_b3_amb_idx00000_mosaicti | 0.716 | 0.234 / 0.006 | 0.226 / 0.006 | 0.182 / 0.005 | 0.204 / 0.006 | 0.026 / 0.026 | 0.240 / 0.006 | 0.295 / 0.021 |
| Average_mosaic_260620_b3_3_18lbf_348_13um_idx0 | 0.777 | 0.183 / 0.008 | 0.180 / 0.010 | 0.131 / 0.009 | 0.176 / 0.012 | 0.040 / 0.038 | 0.188 / 0.006 | 0.239 / 0.034 |
| Average_mosaic_260620_wrought_316L_fatigue_0_c | 0.642 | 0.357 / 0.059 | 0.350 / 0.059 | 0.142 / 0.057 | 0.348 / 0.065 | 0.178 / 0.236 | 0.290 / 0.025 | 0.142 / 0.159 |

Absolute FP levels here are **not** comparable to the production detector's ~0.02% on these specimens: this model is trained on a 50/50-balanced 8k+8k pixel sample with no post-processing, so it sits at a far more permissive operating point by design. Only the arm-vs-arm comparison is meaningful.

## Which arms can the model even see?

Per-frame least-squares fit of `transformed ~ a*img01 + b`, median over the 60 frames, plus the largest standardised-feature difference against `identity`.

| arm | slope a | max affine residual | saturated at 0 | at 1 | max&#124;z − z_identity&#124; | verdict |
|---|---|---|---|---|---|---|
| `identity` | 1 | 4.95e-14 | 0.010 | 0.010 | 0 | **affine → invisible to the model** |
| `stretch_1_99` | 1 | 9.07e-08 | 0.010 | 0.010 | 0.00356 | **affine → invisible to the model** |
| `gamma_0.5` | 0.8338 | 0.251 | 0.010 | 0.010 | 8.45 | non-linear → the model can see it |
| `gamma_2.0` | 0.9486 | 0.256 | 0.010 | 0.010 | 15.6 | non-linear → the model can see it |
| `equalize_hist` | 1.158 | 0.262 | 0.000 | 0.011 | 25.2 | non-linear → the model can see it |
| `dark_stretch_p0.5_p40` | 0.9554 | 0.422 | 0.010 | 0.600 | 8.15 | non-linear → the model can see it |
| `sigmoid_med_g10` | 1.296 | 0.377 | 0.000 | 0.000 | 26.6 | non-linear → the model can see it |

The threshold for that verdict is measured, not assumed (`global_affine_control.py` → `global_affine_control.json`): a *provably* affine map still shifts standardised features by up to **0.0317**, all of it float32 cancellation noise in `local_std`, which evaluates `sqrt(E[x²] − E[x]²)`. `stretch_1_99` comes in at 0.00356 — *below* that floor, so it is a genuine no-op — while `gamma_2.0` (15.6) and `equalize_hist` (25.2) are three orders of magnitude above it and are real changes.

## THIN frames (34 of 60)

Median half-width of the dark core inside the painted strokes: pixels darker than the 20th percentile of `img01` within `correction==1`, components under 64 px dropped, `distance_transform_edt` read along the `skeletonize` of that set. THIN means ≤ 3.0 px. Computed on the untransformed `img01`, so the split is identical for every arm. All 5 folds contain thin frames.

Range 1.00–10.44 px. Caveat: 15 frames sit exactly at the 1.0 px floor, so the thin class is partly saturated — taking the darkest fifth of a painted stroke tends to yield a 1-px thread whatever the crack's true width. The thin/thick contrast is still real (the thick end reaches 10.4 px), but 'THIN' here means 'thin dark core', which is not quite the same as 'thin crack'.

| frame | half-width px |
|---|---|
| Average_mosaic_260618_b2_343_75_LARGE_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_b3_362_50um_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_b3_380_00um_4_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_b3_380_00um_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_b3_383_75um_ZOOM_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_b3_385_63um_ZOOM_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_b3_388_13um_LARGE_2_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_b3_388_13um_ZOOM_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_wrought_316L_fatigue_1200_cycles_crack_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_wrought_316L_fatigue_1250_cycles_crack_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_wrought_316L_fatigue_1260_cycles_crack_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_wrought_316L_fatigue_1270_cycles_crack_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_wrought_316L_fatigue_1280_cycles_crack_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_wrought_316L_fatigue_1290_cycles_crack_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260620_wrought_316L_fatigue_1300_cycles_crack_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.00 |
| Average_mosaic_260618_b2_335_31um_FRFR_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.41 |
| Average_mosaic_260619_HC_316L_fatigue_1760_tip_zoom_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.41 |
| Average_mosaic_260619_HC_316L_fatigue_1770_tip_zoom_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.41 |
| Average_mosaic_260619_HC_316L_fatigue_700_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.41 |
| Average_mosaic_260619_HC_316L_fatigue_800_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.41 |
| Average_mosaic_260620_wrought_316L_fatigue_1100_cycles_crack_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.41 |
| Average_mosaic_260620_wrought_316L_fatigue_800_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 1.41 |
| Average_mosaic_260619_HC_316L_fatigue_1250_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.00 |
| Average_mosaic_260619_HC_316L_fatigue_1400_cycles_tip_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.00 |
| Average_mosaic_260619_HC_316L_fatigue_1790_tip_zoom_2_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.00 |
| Average_mosaic_260619_HC_316L_fatigue_1750_tip_zoom_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.24 |
| Average_mosaic_260619_HC_316L_fatigue_1780_tip_zoom_2_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.24 |
| Average_mosaic_260619_HC_316L_fatigue_1790_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.24 |
| Average_mosaic_260619_HC_316L_fatigue_600_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.24 |
| Average_mosaic_260620_wrought_316L_fatigue_900_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.24 |
| Average_mosaic_260618_B2_333_75_um_zoom_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.83 |
| Average_mosaic_260619_HC_316L_fatigue_1100_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.83 |
| Average_mosaic_260619_HC_316L_fatigue_1400_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 2.83 |
| Average_mosaic_260619_HC_316L_fatigue_1200_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 3.00 |

<details><summary>the 26 non-THIN frames</summary>

| frame | half-width px |
|---|---|
| Average_mosaic_260620_b3_381_88um_ZOOM_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 10.44 |
| Average_mosaic_260620_b3_380_00um_ZOOM_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 10.20 |
| Average_mosaic_260620_b3_380_94um_ZOOM_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 10.00 |
| Average_mosaic_260618_b2_338_13_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 9.00 |
| Average_mosaic_260619_HC_316L_fatigue_1350_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 8.25 |
| Average_mosaic_260618_b2_343_75_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 6.40 |
| Average_mosaic_260619_HC_316L_fatigue_1650_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 6.08 |
| Average_mosaic_260619_HC_316L_fatigue_1450_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 6.04 |
| Average_mosaic_260619_HC_316L_fatigue_1500_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 5.39 |
| Average_mosaic_260618_b2_340_94_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 5.10 |
| Average_mosaic_260619_HC_316L_fatigue_1000_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 5.00 |
| Average_mosaic_260619_HC_316L_fatigue_1600_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 5.00 |
| Average_mosaic_260619_HC_316L_fatigue_1750_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 5.00 |
| Average_mosaic_260620_wrought_316L_fatigue_1100_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 5.00 |
| Average_mosaic_260618_b2_336_25_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 4.47 |
| Average_mosaic_260618_b2_337_19_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 4.47 |
| Average_mosaic_260618_b2_341_88_take2_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 4.47 |
| Average_mosaic_260618_b2_342_81_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 4.47 |
| Average_mosaic_260618_B2_3_2_lbf_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 4.12 |
| Average_mosaic_260619_HC_316L_fatigue_1300_cycles_2_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 3.61 |
| Average_mosaic_260619_HC_316L_fatigue_1450_cycles_tip_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 3.61 |
| Average_mosaic_260620_wrought_316L_fatigue_1000_cycles_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 3.61 |
| Average_mosaic_260618_b2_339_06_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 3.16 |
| Average_mosaic_260618_b2_340_00_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 3.16 |
| Average_mosaic_260619_HC_316L_fatigue_1650_cycles_tip_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 3.16 |
| Average_mosaic_260619_HC_316L_fatigue_900_cycles_FRFR_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif | 3.16 |

</details>

## Cost

| stage | sec |
|---|---|
| featurizer_check_sec | 3.4 |
| stage_a_sec | 10.4 |
| stage_b_sec | 672.6 |
| assemble_sec | 2.4 |
| stage_c_sec | 502.3 |
| stage_d_sec | 232.2 |
| total_sec | 1,425.4 |

**23.8 min** total on 6 worker processes: 60 labelled frames + 6 crack-free specimens x 7 arms of 17-feature extraction (462 full-frame stacks, 740 MP per arm), 35 MLP fold-fits, 7 full-data fits. Caveat: five other CPU-heavy jobs from parallel arms of this study were running on the same 14-core machine throughout, so this is an upper bound rather than a clean cost measurement.

## Conclusion

Global contrast adjustment does not help this detector find thin or faint cracks. The one arm that is nominally above baseline, `gamma_0.5`, is +0.0026 IoU with a paired t of 0.57 over 5 folds and -0.0001 on thin frames — a coin flip, not an effect; and its apparent 1.7x false-positive improvement lives entirely on the off-specimen background, vanishing once you restrict to the specimen. Everything that moves the score significantly moves it down, in an order that tracks how much absolute intensity the transform destroys: sigmoid and gamma 2.0 (mild, compressive) cost ~0.01–0.02, histogram equalisation (flattens the histogram globally) costs 0.023 on 0/5 folds, and my own `dark_stretch` arm — the most aggressive contrast enhancement of the set, saturating 60% of the frame to buy 2.5x contrast in the dark tail — costs 0.184, reproducing the flat-fielding failure (0.169) almost exactly and by the same mechanism. The 1–99 stretch arm is a no-op because the model input is already 1–99 stretched, and its null result is a *verification* of the harness rather than a finding. The mechanistic reading is consistent throughout: this model's signal is largely *absolute, large-radius* darkness, and a whole-image transfer function can only redistribute that — it adds no information, while any non-linearity it applies degrades the monotone relationship between a pixel's brightness and how deeply it sits inside a dark region. Contrast is not the bottleneck on thin cracks. If anything here is worth following up it is the guardrail artefact, not a transform: whole-frame false-positive rates on these specimens are dominated by off-specimen background and rank arms in the opposite order to on-specimen rates.

