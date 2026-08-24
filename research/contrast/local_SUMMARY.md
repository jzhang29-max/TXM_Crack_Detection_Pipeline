# Local / adaptive contrast enhancement as a model-input transform

Arm: LOCAL / ADAPTIVE contrast enhancement of the model input, scored against a self-computed `identity` baseline under the shared protocol (GroupKFold(5) by image, 17-feature stack, MLP(64,32) + StandardScaler, IoU@0.5 on held-out rows).

- images with both correction classes: **60** (no image-set subsampling was needed -- CLAHE turned out to cost ~1 s on a 23 MP frame)
- rows per image: up to 8000 `correction==1` + up to 8000 `correction==2`, `RandomState(0)`, image id as the CV group
- thin-crack frames: **33/60** (median skeleton half-width <= 3.0 px)
- crack-free guardrail: 200,000 uniformly random pixels x 6 specimens from `pipeline.CLEAN_SPECIMENS`, model trained on all rows
- LCN epsilon = 0.001 (measured local std on this data runs 0.011-0.25, so this is a true normalisation)
- feature extraction wall clock: **77 min** total for 11 arms over 60 labelled frames (642 MP/arm), plus ~21 min for the 6 crack-free frames. Per-arm cost is in the table below. (The machine was shared with other jobs during this run, so absolute seconds are ~1.5x an idle machine; the RELATIVE cost between arms is still meaningful.)

Reproduce with `research/contrast/local_extract.py` (builds the sample cache), then `local_eval.py`, `local_diag.py`, `local_report.py`. NOTE: the cache in `research/contrast/local_cache/` (694 MB) and `local_cache_clean/` (866 MB) is intermediate and fully regenerable -- safe to `rm -rf` once these results are accepted. It is kept only so the analysis can be rerun without the ~100 min extraction.

## Results

All-rows numbers are the mean over the 5 folds, +- the fold-to-fold standard deviation. Thin-frame IoU is pooled over all out-of-fold rows belonging to thin frames (per-fold spread also given). `crack-free FP` is the mean over the 6 specimens of the fraction of sampled pixels predicted crack -- every one of those is a false positive by construction.

| arm | all IoU | all precision | all recall | thin IoU (pooled) | thin IoU (fold mean+-sd) | crack-free FP mean (all px / on spec) | d IoU vs identity | d thin IoU | transform+featurise (min) |
|---|---|---|---|---|---|---|---|---|---|
| `identity` | 0.6667 +- 0.0275 | 0.8104 | 0.7898 | 0.6187 | 0.6208 +- 0.0259 | 21.907% / 2.229% | -- | -- | 6.7 |
| `clahe_c0.01_k8` | 0.6501 +- 0.0297 | 0.8103 | 0.7666 | 0.5897 | 0.5933 +- 0.0364 | 20.251% / 4.749% | -0.0166 | -0.0290 | 7.1 |
| `clahe_c0.01_k16` | 0.6558 +- 0.0304 | 0.8008 | 0.7842 | 0.5941 | 0.5959 +- 0.0295 | 11.124% / 4.233% | -0.0109 | -0.0246 | 7.0 |
| `clahe_c0.03_k8` | 0.6314 +- 0.0203 | 0.7907 | 0.7584 | 0.5779 | 0.5802 +- 0.0241 | 15.944% / 6.198% | -0.0353 | -0.0408 | 7.1 |
| `clahe_c0.03_k16` | 0.6438 +- 0.0106 | 0.7943 | 0.7730 | 0.6054 | 0.6044 +- 0.0139 | 13.863% / 5.691% | -0.0229 | -0.0133 | 7.1 |
| `lcn_w51` | 0.5476 +- 0.0339 | 0.7399 | 0.6822 | 0.4981 | 0.4866 +- 0.0793 | 15.331% / 13.614% | -0.1191 | -0.1206 | 6.9 |
| `lcn_w151` | 0.5828 +- 0.0268 | 0.7746 | 0.7024 | 0.5205 | 0.5186 +- 0.0540 | 15.494% / 9.918% | -0.0840 | -0.0982 | 6.9 |
| `lcn_w51_robust` | 0.6251 +- 0.0060 | 0.7874 | 0.7525 | 0.5952 | 0.5941 +- 0.0116 | 31.955% / 21.870% | -0.0416 | -0.0235 | 7.0 |
| `unsharp_s2_a1.0` | 0.6665 +- 0.0157 | 0.8276 | 0.7755 | 0.6273 | 0.6273 +- 0.0377 | 17.007% / 2.382% | -0.0002 | +0.0086 | 6.8 |
| `unsharp_s8_a1.5` | 0.6510 +- 0.0379 | 0.8227 | 0.7603 | 0.6066 | 0.6081 +- 0.0579 | 14.384% / 2.422% | -0.0157 | -0.0121 | 6.9 |
| `clahe_c0.01_k8_blend0.5` | 0.6606 +- 0.0340 | 0.8034 | 0.7877 | 0.6060 | 0.6106 +- 0.0362 | 13.580% / 3.116% | -0.0061 | -0.0127 | 7.1 |

The identity baseline's own fold-to-fold IoU spread is **+-0.0275**. Any two arms closer than that on all-rows IoU should be read as indistinguishable, not ranked.

### Crack-free false positives, per specimen

Protocol column is `all px` (uniform over the whole frame). `on specimen` restricts to `pipeline.specimen_support`, because off-specimen background is 20-40% of these mosaics and dilutes a whole-frame rate; it is the stricter and more meaningful number.

| arm | B2_2_1_lbf | B2_2_9_lbf | B2_amb_mosaic_2 | b3_3_18lbf | b3_amb | wrought_316L_fatigue_0_cycles |
|---|---|---|---|---|---|---|
| `identity` | 15.018% / 1.863% | 18.083% / 2.089% | 21.039% / 2.113% | 18.254% / 0.840% | 23.389% / 0.581% | 35.660% / 5.887% |
| `clahe_c0.01_k8` | 14.569% / 4.604% | 11.057% / 4.735% | 16.082% / 4.751% | 17.783% / 1.581% | 23.020% / 0.621% | 38.995% / 12.205% |
| `clahe_c0.01_k16` | 10.190% / 2.817% | 6.995% / 3.553% | 10.418% / 3.585% | 10.522% / 1.828% | 12.821% / 1.100% | 15.797% / 12.516% |
| `clahe_c0.03_k8` | 11.316% / 3.583% | 9.527% / 4.357% | 12.087% / 5.467% | 20.190% / 5.428% | 24.120% / 3.093% | 18.425% / 15.259% |
| `clahe_c0.03_k16` | 10.939% / 3.563% | 10.000% / 4.816% | 12.919% / 3.768% | 11.917% / 3.846% | 14.171% / 2.069% | 23.232% / 16.085% |
| `lcn_w51` | 0.699% / 0.266% | 8.369% / 3.866% | 57.137% / 50.291% | 3.633% / 3.069% | 4.477% / 3.596% | 17.672% / 20.596% |
| `lcn_w151` | 18.713% / 8.583% | 20.455% / 12.671% | 14.906% / 4.805% | 2.121% / 2.177% | 21.818% / 17.628% | 14.953% / 13.641% |
| `lcn_w51_robust` | 31.928% / 23.097% | 32.932% / 21.100% | 43.163% / 32.946% | 35.871% / 24.517% | 28.907% / 19.568% | 18.926% / 9.991% |
| `unsharp_s2_a1.0` | 6.082% / 1.553% | 19.774% / 2.352% | 23.303% / 2.809% | 20.540% / 1.030% | 24.869% / 0.402% | 7.472% / 6.146% |
| `unsharp_s8_a1.5` | 9.065% / 1.193% | 22.009% / 5.149% | 4.173% / 2.113% | 3.188% / 1.893% | 11.889% / 0.552% | 35.979% / 3.633% |
| `clahe_c0.01_k8_blend0.5` | 12.618% / 1.945% | 10.233% / 2.593% | 14.947% / 3.442% | 12.715% / 1.297% | 15.216% / 0.610% | 15.749% / 8.806% |

(cells are `all px` / `on specimen`)

### Thin-crack frames

Median skeleton half-width of the darkest-20% core inside `correction==1`, after `remove_small_objects(64)`:

Sensitivity note: the EDT quantises half-widths to sqrt(n), so 4 further frames land at 3.16 px, just above the protocol's 3.0 cut. The thin set is therefore somewhat sensitive to that exact threshold; the numbers below use 3.0 as specified.

- `1.00` px **THIN** -- Average_mosaic_260618_b2_343_75_LARGE_idx00000_mosaictileAA_img001of010_xrm_bim___56d03fa7
- `1.00` px **THIN** -- Average_mosaic_260620_b3_362_50um_idx00000_mosaictileAA_img001of010_xrm_bim_bim__2deda36a
- `1.00` px **THIN** -- Average_mosaic_260620_b3_380_00um_idx00000_mosaictileAA_img001of010_xrm_bim_bim__7e5607f2
- `1.00` px **THIN** -- Average_mosaic_260620_b3_383_75um_ZOOM_idx00000_mosaictileAA_img001of010_xrm_bim__74e56e11
- `1.00` px **THIN** -- Average_mosaic_260620_b3_385_63um_ZOOM_idx00000_mosaictileAA_img001of010_xrm_bim__d4fafaac
- `1.00` px **THIN** -- Average_mosaic_260620_b3_388_13um_LARGE_2_idx00000_mosaictileAA_img001of010_xrm___5397bc81
- `1.00` px **THIN** -- Average_mosaic_260620_b3_388_13um_ZOOM_idx00000_mosaictileAA_img001of010_xrm_bim__2a8e0a9d
- `1.00` px **THIN** -- Average_mosaic_260620_wrought_316L_fatigue_1200_cycles_crack_idx00000_mosaictile__dc137639
- `1.00` px **THIN** -- Average_mosaic_260620_wrought_316L_fatigue_1250_cycles_crack_idx00000_mosaictile__9e764cf5
- `1.00` px **THIN** -- Average_mosaic_260620_wrought_316L_fatigue_1260_cycles_crack_idx00000_mosaictile__aee1a727
- `1.00` px **THIN** -- Average_mosaic_260620_wrought_316L_fatigue_1270_cycles_crack_idx00000_mosaictile__ac4f59b7
- `1.00` px **THIN** -- Average_mosaic_260620_wrought_316L_fatigue_1280_cycles_crack_idx00000_mosaictile__f2896a3b
- `1.00` px **THIN** -- Average_mosaic_260620_wrought_316L_fatigue_1290_cycles_crack_idx00000_mosaictile__e8b7c17d
- `1.00` px **THIN** -- Average_mosaic_260620_wrought_316L_fatigue_1300_cycles_crack_idx00000_mosaictile__161c4589
- `1.41` px **THIN** -- Average_mosaic_260618_b2_335_31um_FRFR_idx00000_mosaictileAA_img001of010_xrm_bim__37d5f720
- `1.41` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1760_tip_zoom_idx00000_mosaictileAA_img001__a365650e
- `1.41` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1770_tip_zoom_idx00000_mosaictileAA_img001__30787019
- `1.41` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_700_cycles_idx00000_mosaictileAA_img001of0__9379d383
- `1.41` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_800_cycles_idx00000_mosaictileAA_img001of0__96ccb451
- `1.41` px **THIN** -- Average_mosaic_260620_wrought_316L_fatigue_1100_cycles_crack_idx00000_mosaictile__b604e4b1
- `1.41` px **THIN** -- Average_mosaic_260620_wrought_316L_fatigue_800_cycles_idx00000_mosaictileAA_img0__7fbb94ed
- `2.00` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1250_cycles_idx00000_mosaictileAA_img001of__081042c6
- `2.00` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1400_cycles_tip_idx00000_mosaictileAA_img0__d71eab68
- `2.00` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1790_tip_zoom_2_idx00000_mosaictileAA_img0__5b53f617
- `2.24` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1750_tip_zoom_idx00000_mosaictileAA_img001__2695b632
- `2.24` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1780_tip_zoom_2_idx00000_mosaictileAA_img0__9e44d44b
- `2.24` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1790_cycles_idx00000_mosaictileAA_img001of__061e601b
- `2.24` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_600_cycles_idx00000_mosaictileAA_img001of0__6b5a8408
- `2.24` px **THIN** -- Average_mosaic_260620_wrought_316L_fatigue_900_cycles_idx00000_mosaictileAA_img0__33a185d3
- `2.83` px **THIN** -- Average_mosaic_260618_B2_333_75_um_zoom_idx00000_mosaictileAA_img001of010_xrm_bi__250c5bf6
- `2.83` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1100_cycles_idx00000_mosaictileAA_img001of__9d0bf16b
- `2.83` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1400_cycles_idx00000_mosaictileAA_img001of__95dd4588
- `3.00` px **THIN** -- Average_mosaic_260619_HC_316L_fatigue_1200_cycles_idx00000_mosaictileAA_img001of__4296df23
- `3.16` px  -- Average_mosaic_260618_b2_339_06_idx00000_mosaictileAA_img001of010_xrm_bim_bim__c0d2117b
- `3.16` px  -- Average_mosaic_260618_b2_340_00_idx00000_mosaictileAA_img001of010_xrm_bim_bim__9866c456
- `3.16` px  -- Average_mosaic_260619_HC_316L_fatigue_1650_cycles_tip_idx00000_mosaictileAA_img0__6610c8eb
- `3.16` px  -- Average_mosaic_260619_HC_316L_fatigue_900_cycles_FRFR_idx00000_mosaictileAA_img0__56a1f4c1
- `3.61` px  -- Average_mosaic_260619_HC_316L_fatigue_1300_cycles_2_idx00000_mosaictileAA_img001__51b19943
- `3.61` px  -- Average_mosaic_260619_HC_316L_fatigue_1450_cycles_tip_idx00000_mosaictileAA_img0__b6661578
- `3.61` px  -- Average_mosaic_260620_wrought_316L_fatigue_1000_cycles_idx00000_mosaictileAA_img__e4928d15
- `4.12` px  -- Average_mosaic_260618_B2_3_2_lbf_idx00000_mosaictileAA_img001of010_xrm_bim_bim__b0112039
- `4.24` px  -- Average_mosaic_260618_b2_337_19_idx00000_mosaictileAA_img001of010_xrm_bim_bim__6261f4a6
- `4.47` px  -- Average_mosaic_260618_b2_336_25_idx00000_mosaictileAA_img001of010_xrm_bim_bim__ab7335a5
- `4.47` px  -- Average_mosaic_260618_b2_341_88_take2_idx00000_mosaictileAA_img001of010_xrm_bim___a66b1ab0
- `4.47` px  -- Average_mosaic_260618_b2_342_81_idx00000_mosaictileAA_img001of010_xrm_bim_bim__ea67efab
- `5.00` px  -- Average_mosaic_260619_HC_316L_fatigue_1000_cycles_idx00000_mosaictileAA_img001of__09ed91ba
- `5.00` px  -- Average_mosaic_260619_HC_316L_fatigue_1600_cycles_idx00000_mosaictileAA_img001of__4cb30dc6
- `5.00` px  -- Average_mosaic_260619_HC_316L_fatigue_1750_cycles_idx00000_mosaictileAA_img001of__6a0a53d0
- `5.00` px  -- Average_mosaic_260620_wrought_316L_fatigue_1100_cycles_idx00000_mosaictileAA_img__fd88ea0a
- `5.10` px  -- Average_mosaic_260618_b2_340_94_idx00000_mosaictileAA_img001of010_xrm_bim_bim__3000138d
- `5.39` px  -- Average_mosaic_260619_HC_316L_fatigue_1500_cycles_idx00000_mosaictileAA_img001of__d30ef8fa
- `6.04` px  -- Average_mosaic_260619_HC_316L_fatigue_1450_cycles_idx00000_mosaictileAA_img001of__df1e43e7
- `6.08` px  -- Average_mosaic_260619_HC_316L_fatigue_1650_cycles_idx00000_mosaictileAA_img001of__f73d8dda
- `6.40` px  -- Average_mosaic_260618_b2_343_75_idx00000_mosaictileAA_img001of010_xrm_bim_bim__f36e4ce8
- `8.25` px  -- Average_mosaic_260619_HC_316L_fatigue_1350_cycles_idx00000_mosaictileAA_img001of__83c0413c
- `9.00` px  -- Average_mosaic_260618_b2_338_13_idx00000_mosaictileAA_img001of010_xrm_bim_bim__cea4b1d0
- `n/a` px  -- Average_mosaic_260620_b3_380_00um_4_idx00000_mosaictileAA_img001of010_xrm_bim_bi__4b09d210
- `n/a` px  -- Average_mosaic_260620_b3_380_00um_ZOOM_idx00000_mosaictileAA_img001of010_xrm_bim__42d31c46
- `n/a` px  -- Average_mosaic_260620_b3_380_94um_ZOOM_idx00000_mosaictileAA_img001of010_xrm_bim__9f79af8d
- `n/a` px  -- Average_mosaic_260620_b3_381_88um_ZOOM_idx00000_mosaictileAA_img001of010_xrm_bim__4c28fcd7

### Mechanism: single-feature AUC (all rows)

AUC of each feature taken alone, reported as `max(a, 1-a)`. This is the direct test of the trade this family makes: local contrast is bought with the ABSOLUTE large-radius intensity channels that this project already measured as ~41% of the model's importance (docs/MARKUP_GUIDE.md).

| arm | `intensity` | `smooth_s8` | `smooth_s32` | `smooth_s64` | `gradmag_s1` | `laplacian_s1` | `texture_s2` | best feature |
|---|---|---|---|---|---|---|---|---|
| `identity` | 0.6399 | 0.6389 | 0.6290 | 0.6152 | 0.5582 | 0.5079 | 0.5419 | smooth_s2 (0.6400) |
| `clahe_c0.01_k8` | 0.6552 | 0.6540 | 0.6387 | 0.6178 | 0.5398 | 0.5072 | 0.5154 | smooth_s2 (0.6557) |
| `clahe_c0.01_k16` | 0.6547 | 0.6538 | 0.6388 | 0.6173 | 0.5558 | 0.5069 | 0.5371 | smooth_s2 (0.6553) |
| `clahe_c0.03_k8` | 0.6565 | 0.6552 | 0.6363 | 0.6103 | 0.5018 | 0.5067 | 0.5405 | smooth_s2 (0.6571) |
| `clahe_c0.03_k16` | 0.6621 | 0.6635 | 0.6454 | 0.6150 | 0.5098 | 0.5066 | 0.5270 | smooth_s4 (0.6647) |
| `lcn_w51` | 0.5315 | 0.5328 | 0.5220 | 0.5142 | 0.5407 | 0.5027 | 0.6099 | texture_s2 (0.6099) |
| `lcn_w151` | 0.5676 | 0.5675 | 0.5520 | 0.5341 | 0.5201 | 0.5036 | 0.5692 | gradmag_s8 (0.6175) |
| `lcn_w51_robust` | 0.5619 | 0.6286 | 0.6566 | 0.6160 | 0.5427 | 0.5030 | 0.6196 | smooth_s16 (0.6586) |
| `unsharp_s2_a1.0` | 0.6121 | 0.6114 | 0.6043 | 0.5929 | 0.5412 | 0.5047 | 0.5168 | smooth_s2 (0.6122) |
| `unsharp_s8_a1.5` | 0.6072 | 0.6064 | 0.5993 | 0.5885 | 0.5554 | 0.5040 | 0.5366 | gradmag_s8 (0.6101) |
| `clahe_c0.01_k8_blend0.5` | 0.6490 | 0.6478 | 0.6353 | 0.6180 | 0.5482 | 0.5074 | 0.5272 | smooth_s2 (0.6493) |

## Conclusion

Local/adaptive contrast enhancement of the model input does not help this detector find thin, faint cracks, and the amount it hurts is predicted almost perfectly by how much of the image's ABSOLUTE intensity the transform destroys. Ranked by that: unsharp masking, which adds a zero-mean high-pass on top of the original and so leaves the DC term intact, is indistinguishable from the identity baseline on every metric -- `unsharp_s2_a1.0` scores 0.6665 vs 0.6667 all-rows IoU and 0.6273 vs 0.6187 on thin frames, both inside the +-0.0275 fold-to-fold spread, with crack-free false positives flat at 2.38% vs 2.23%. Half-blending CLAHE with the original (`clahe_c0.01_k8_blend0.5`, my own arm, designed to add local contrast WITHOUT removing the DC term) costs -0.0061 IoU. Full CLAHE, which remaps intensity per tile, costs -0.0353 to -0.0109 and roughly DOUBLES to TRIPLES crack-free false positives (2.23% -> 4.2-6.2% on-specimen), monotonically in clip limit. Local contrast normalisation, which removes absolute intensity outright, is the worst: `lcn_w51` loses -0.1191 all-rows IoU and -0.1206 on thin frames while raising on-specimen false positives 6.1x. That reproduces this project's existing flat-fielding result (-0.169 IoU) in the same direction and comparable magnitude, which is the expected outcome for flat-fielding's close cousin. **No arm improved thin-frame IoU by more than the fold-to-fold spread, so there is no local-contrast arm worth deploying.**

The single-feature AUCs say why, and they kill the optimistic reading rather than supporting it. Measured WITHIN a single thin frame, LCN does exactly what it advertises: on `B2_333_75_um_zoom` the `texture_s2` AUC goes from 0.585 under identity to 0.926 under `lcn_w51`, i.e. the faint crack really is far better separated from its immediate surroundings. But that gain does NOT survive pooling across frames, which is the regime the classifier is actually trained in: pooled over all 33 thin frames, `texture_s2` moves only 0.5116 -> 0.5219, while `intensity` collapses 0.6417 -> 0.5293 and `smooth_s64` 0.6218 -> 0.5115. The reason is that LCN's output scale is set by each frame's own local statistics, so the enhanced contrast means something different in every frame and a pooled model cannot cash it in -- whereas absolute intensity was comparable across frames and is now gone. Since the large-radius absolute-intensity group is ~41% of this model's importance (docs/MARKUP_GUIDE.md), the trade is strictly losing. CLAHE, by contrast, leaves `intensity` intact or slightly better (0.6417 -> 0.6634), which is why its IoU cost is small -- its damage shows up in the false-positive guardrail instead.

**The false-positive guardrail is what settles it, and it disagrees with the IoU ranking.** `lcn_w51_robust` -- identical LCN, but rescaled with a robust 1st-99th percentile clip instead of min-max -- looks like the best LCN variant on labelled rows (0.6251 IoU, only -0.0416), yet it predicts crack on 21.9% of on-specimen pixels in frames the owner confirmed contain NO crack -- 10x the baseline's 2.23%. Judged on held-out IoU alone LCN would have read as a modest cost; judged on the crack-free frames it is disqualifying. This is exactly the noise-amplification-in-flat-material failure mode local contrast enhancement is prone to, and it is only visible because the guardrail scores frames that carry no labels at all.

Arms indistinguishable from `identity` on all-rows IoU alone (within +-0.0275): `clahe_c0.01_k8`, `clahe_c0.01_k16`, `clahe_c0.03_k16`, `unsharp_s2_a1.0`, `unsharp_s8_a1.5`, `clahe_c0.01_k8_blend0.5`. Note that IoU alone does NOT separate these -- the CLAHE members of that list double or triple crack-free false positives, so they are indistinguishable on the headline metric and clearly worse on the guardrail. Only the two unsharp arms and the CLAHE blend are indistinguishable from baseline on BOTH.

Caveat on the guardrail's absolute scale: the protocol's uniformly-random 200k pixels per crack-free frame give whole-frame rates of 11-32%, but only 62-78% of each mosaic is specimen and the raw-input model calls the dark off-specimen background crack. That inflation is present in the identity baseline too (21.9% whole-frame vs 2.23% on-specimen), so it is a property of the existing pipeline rather than of any arm here. Both columns are reported; the on-specimen column is the one that discriminates between arms.

