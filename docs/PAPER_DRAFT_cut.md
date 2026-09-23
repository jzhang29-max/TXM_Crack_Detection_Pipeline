<!--
PAPER DRAFT, 11,981 words, assembled 2026-09-23. All five sections through the final cut.
  full 23,909-word version: docs/PAPER_DRAFT_full_2026-09-21.md
  2026-08-26 original:         docs/PAPER_DRAFT_2026-08-26_superseded.md
  editors' notes:              crack-evolution-5d/out/paper_final_cut_notes.md

LENGTH: 11,981 against a 6,000-8,000 target. Three sections hit their budgets --
front 905 (target 850), Results 3 2,048 (1,900), Results 4 1,458 (1,350). Two did not:
Methods 4,493 against 1,900, and the back matter 3,077 against 2,000. Those two editors
reported that further cuts would have cost evidence they were instructed to keep, and the
instruction was absolute, so they stopped. Roughly 4,000 words still have to come out of
Methods and the back matter, and doing it means deciding what evidence to move to
supplementary -- an author's call, not an editor's. NOT SUBMITTABLE at this length.

EVIDENCE CHECKED MECHANICALLY after assembly: 18 load-bearing figures all present, every
section heading appears exactly once, 34 [TODO-AUTHOR] markers intact. All required
self-corrections survive -- the specimen confound that reversed the small-component
conclusion; the spliced width table; the IoU ceiling ratio measured against a superseded
mask (13.6x, with 10.1x retired by name); the prior-art finding that the width method is
Orthogonal Profile Extraction + ISO50 and not novel; the failed v1 held-out experiment;
the 5.5x crack-free false-positive inflation that puts the held-out detector above the 1%
MIL-HDBK-1823A line the deployed figure passes; the one-frame grouping bug; and that the
adjudication panels are language-model agents, not human experts.

TWO AUTHOR DECISIONS THE EDITORS RAISED RATHER THAN TAKE:
  - a [TODO-AUTHOR] asking for the held-out false-positive figure is now satisfied by the
    Addendum; retire it.
  - "25.4% of every background label" (Intro) and "4.3% of training rows after per-image
    capping" (Addendum) answer different questions. Both sourced, neither edited.

NOT CLOSABLE FROM THIS REPOSITORY:
  - acquisition metadata: beam energy, reconstruction, voxel-size provenance, loading
    protocol, specimen history. Absent everywhere; must be supplied, never inferred.
    docs/AUTHOR_TODO.md groups the markers into four fill-in blocks.
  - human expert adjudication. Protocol, crops and seed preserved; two blinded readers are
    the step that separates a Q2 submission from a Q1 one.
-->

# A quarter of an operator-trained crack detector's output has never been adjudicated, and it reads as crack: labelled-pixel metrics on broad-brush TXM annotations of 316L steel

## Abstract

We audit an interactive crack detector on 71 transmission X-ray microscopy (TXM) mosaic frames of broad-brush-annotated fatigue-cracked 316L steel across four specimen groups. Area-weighted over the 65 frames the census covers, 24.7% of accepted mask area was never labelled either way (`txm_unlabelled_area_per_frame.json`) — invisible to IoU, clDice, precision and recall, which are computed only where an operator painted something. Blinded three-panel adjudication calls its large components crack: 25 of 26, indistinguishable from painted crack (p = 0.31), far from plain matrix (p = 4.4e-08); the one rejection is a genuine false positive at a specimen free surface (`txm_unlabelled_adjudication.json`). Size- and specimen-matched small indications tie with the painted-crack control at 13/32 and both exceed matrix at 3/26 (`txm_small_component_adjudication_v2.json`). Against a median label width of 73 px, a physically correct 3 px crack scores IoU 0.0498 (`txm_iou_ceiling.json`): the metric ranks imitation of a paintbrush. Detector and annotator widths, measured against the image, are indistinguishable (0.506 against 0.529 of the transverse feature, paired Wilcoxon p = 0.061) and both under-mark (`txm_width_ratio_unified.json`). Our adjudication panels are language-model agents, not human experts — this audit's principal limitation; we release the protocol so human readers can repeat it.

**Keywords:** crack segmentation, transmission X-ray microscopy, 316L stainless steel, annotation quality, segmentation metrics, blinded adjudication, audit

## 1. Introduction

A predicted pixel where the operator painted nothing is neither a true positive nor a false positive: it is dropped from every term.

The corpus is 71 TXM mosaic frames of fatigue-cracked 316L steel — additively manufactured (AM / HC_316L), bending series B2 and B3, wrought — annotated by one operator with a broad brush; 61 carry labels. On the shipped mask the detector scores clDice 0.823, topological precision 0.970, topological sensitivity 0.761 and IoU 0.675 over those 61 frames (`txm_width_clip_final.json`). The contribution is not an architecture but a measurement of how much of that score the annotation decides.

**A quarter of the mask has never been adjudicated.** Area-weighted over the 65 frames the census covers, 24.7% of accepted area — 8,016,109 px of 32,504,070 — was painted neither crack nor not-crack (`txm_unlabelled_area_per_frame.json`); six of the 71 frames are skipped for want of a mask or correction map (`code/audit/measure_unlabelled_area.py`). It is uneven — 51.1% on AM (n = 24) against 4.5% on B3 (n = 13), 97.3% on the worst frame — and not dust: 87.9% is in components of 4,000 px or more, though 32.4% lies within 25 px of a painted stroke and is to that extent stroke rim.

**Adjudicated, it reads as crack.** Against four control classes (§2.11), blinded three-panel adjudication called 25 of 26 large unlabelled regions crack — indistinguishable from painted crack (p = 0.31), separated from plain matrix (p = 4.4e-08); the one rejection, a specimen free surface, is a genuine false positive (§3.4) (`txm_unlabelled_adjudication.json`). Size- and specimen-matched small indications scored 13/32, as did the painted-crack control (two-sided Fisher p = 1.000), against matrix's 3/26 (p = 0.013, computed one-sided; `code/audit/score_blind_panel.py`, `txm_small_component_adjudication_v2.json`). Two limits: the large-region sample is components of 4,000 px or more, 21.7% of accepted area, not the whole 24.7%; and the small-component class is 0.23% of accepted area, its entire exposure even if every member were a false positive (`txm_small_component_adjudication.json`).

**Why the default metric cannot see it.** The operator's labels scored against themselves — skeletonised, re-dilated to a physical width — set a ceiling: against a median label width of 73 px, a correct 3 px crack reaches a median IoU of 0.0498 over 61 frames (`txm_iou_ceiling.json`; further rungs in §3.2). The detector's IoU 0.6748, scored under one convention in one pass, is **13.6x that ceiling** (`txm_iou_ceiling_v2.json`); the older artifact's "10.1x" paired the same ceiling with a mask measured before `clip_to_measured_width` shipped, and its three model keys are retired by name in the v2 `.supersedes` block.

**A width instrument pointed at the annotation.** The measurement is standard metrology, not claimed as new (§2.9.1); only its target is ours. On one estimator and shared sample points, the shipped mask covers a median 0.505 of the transverse feature over all 64 frames, 0.506 against the brush's 0.529 on the 61 labelled frames (paired Wilcoxon p = 0.061); it exceeds the feature on 0 of 64 frames (`txm_width_ratio_unified.json`, §4).

**What we do not claim, and what is weak.** We claim no corpus-wide over-marking (the detector under-marks) and no crack growth; that test lost to doing nothing. Our adjudication panels are language-model agents, not human experts: their validation shows only agreement with the same annotator's strokes, which is not competence at 316L TXM, and their votes are not regenerable from the released code (`code/audit/README.md`). Panel sensitivity on known crack of that size is 13/32, so that arm shows indistinguishability from known crack, not crack. The crack-free false-positive rate is contaminated: those six specimens contribute 0 crack pixels and 95,351,647 not-crack pixels, 25.4% of every background label the model trains on (header of `code/audit/heldout_crackfree_fp.py`, not an `out/` artifact), so it is partly memorisation. Held out, the refit reads 1.581% against 0.289% trained with those labels — above the 1% MIL-HDBK-1823A yardstick the deployed 0.174% passes (`txm_heldout_crackfree_fp_v2.json`, Addendum). [TODO-AUTHOR: insert the held-out crack-free false-positive rate, and the deployed figure beside it, when the refit completes.] Beam energy, reconstruction, loading protocol and specimen provenance are in neither repository, and are marked [TODO-AUTHOR] throughout.

## 2. Methods

**Conventions.** `txm_*.json` / `cldice_*.json` = `out/` artifacts; `app/…`, `code/…`, `docs/…`, `models/…` = pipeline paths. *Derived* = recomputed here from per-frame records. A number with no artifact behind it is named, not used; disagreeing artifacts are named, never averaged. All measurement is on the shipped mask: `effective_mask(corrections='none', tight=True)`, `WIDTH_CLIP = True`, `drop_straight_lines` active (`txm_unlabelled_area_per_frame.json .mask_version`).

### 2.1 Specimens, loading and acquisition — [TODO-AUTHOR]

Unrecorded in both repositories, not reconstructible, to be supplied rather than inferred: microscope, beam energy, detector, objective/zone plate, exposure, projections, reconstruction, ring/beam-hardening correction; rig, load levels, cycle counting (`_1_lbf`, `_3_18lbf`, `_1250_cycles` imply pound-force steps and a fatigue count; no protocol is recorded, so the units must not be stated unconfirmed); 316L composition and heat treatment; AM build process and parameters (AM = filenames beginning `HC_316L_`; `HC` unrecorded); specimen shape and through-thickness (the 22 µm value used in a separate 3D arm is unverified here and unused); annotator identity, training, materials background, time per frame, whether any frame was annotated twice (one annotator produced all labels); ethics, data availability, a DOI deposit for the 71 correction masks. [TODO-AUTHOR]

### 2.2 Spatial scale

**29.24 nm/px** is geometric, not an instrument header: 9 × 5 tiles of a 30 µm window at 0.35 overlap = 186.0 × 108.0 µm over 6367 × 3691 px, isotropic to 0.2% (`crack-depth-3d/README.md:69–70`). Window and overlap are operator-supplied, to be confirmed [TODO-AUTHOR]; lengths scale with them, ratios do not. The width artifact rounds to **29.0 nm/px** (`measure_transverse_fwhm.py:37`): median 45.0 px transverse FWHM = 1,305 nm (`txm_width_reference.json`) against 1,316 nm at 29.24, 0.8% apart, below the derivation's own uncertainty — unify the constants [TODO-AUTHOR].

### 2.3 Corpus and preprocessing

Seventy-one mosaic frames, four 316L groups, float32, 2.85–32.12 MP (median 10.39, total 849.59; *derived* from `app_data/images/*/meta.json`).

| group | frames | with painted crack | crack px | not-crack px | MP |
|---|---|---|---|---|---|
| AM (`HC_316L_*`) | 27 | 25 | 4,628,821 | 165,191,178 | 5.15–32.12 |
| B2 | 17 | 14 | 9,321,938 | 67,038,533 | 2.85–23.50 |
| B3 | 13 | 10 | 7,616,710 | 48,700,962 | 3.86–23.46 |
| Wrought | 14 | 12 | 9,750,198 | 94,701,499 | 8.83–22.20 |
| **total** | **71** | **61** | **31,317,667** | **375,632,172** | |

`correction_crack_px` = 31,317,667 recurs in `app_data/models/retrain_history.json` (stamp `20260824_225236`) and `models/hybrid_v5_20260824.joblib`.

**Grouping is by filename, and this is a correction.** Stored-ID tokens `_b2_`, `b3`, `hc_316l`, `wrought` mis-file `HC_316L_fatigue_1600_cycles` (`__4cb30dc6`) as B3 in `txm_unlabelled_area_per_frame.json`, `txm_width_ratio_unified.json`, `txm_iou_ceiling_v2.json`; `txm_iou_ceiling_per_frame.json` and `cldice_centreline_per_frame.json` group by filename (AM *n* = 25). Regrouped here (*derived* from `.per_frame`); totals, B2, Wrought unchanged:

| artifact | AM | B3 |
|---|---|---|
| `txm_unlabelled_area_per_frame.json`, *n* | 24 → **25** | 13 → **12** |
| — unlabelled share | 51.06% → **50.16%** | 4.49% → **2.83%** |
| `txm_iou_ceiling_v2.json`, *n* | 24 → **25** | 11 → **10** |
| — shipped IoU | 0.5310 → **0.5468** | 0.8933 → **0.9063** |
| `txm_width_ratio_unified.json`, *n* | 24 → **25** | 13 → **12** |

**Human and model see different images.** De-stitched (`code/destitch.py`), pseudo-flat-fielded by normalised-convolution division (σ_y = 16, σ_x = 22; `code/flatfield.py:57`), display only: model reads raw `img.npy`, annotator `display.npy` stretched 1st–99th percentile of specimen support (`app/core/pipeline.py:131`, `:106`). The `display_limits` docstring's 0.169 IoU cost for flat-fielded model input has no `out/` artifact; not used as evidence [TODO-AUTHOR: re-measure or drop].

### 2.4 Annotation

Three-valued labels: 1 = crack, 2 = not crack, 0 = never labelled either way — an absent assertion, not background (§2.7); a stroke marks *that there is crack here*, not *where its edges are*. **Two artifacts measure painted-label width and disagree by 1.8×**; both are reported:

| instrument | median, 61 labelled frames | AM | B2 | B3 | Wrought |
|---|---|---|---|---|---|
| **primary — `txm_iou_ceiling_v2.json`** (2026-09-21), 2 × median EDT on the label skeleton, definition in `measure_iou_ceiling.py:86` | **41.2 px** (`.median_label_width_px` = 41.1825) | 18.97 | 54.58 | 77.24 | 73.83 |
| secondary — `txm_iou_ceiling_per_frame.json` / `cldice_centreline_per_frame.json` (2026-09-19; byte-identical `label_width_px`, no generator, definition unrecorded) | 73.4 px (*derived*; 73.0 in `txm_iou_ceiling.json`) | 36 | 88 | 126 | 97 |

Primary = its definition survives in a generator. At 29.24 nm/px the medians are **1.20 µm** and 2.15 µm; primary per-group columns *derived* by regrouping (§2.3), secondary row from `txm_iou_ceiling.json .per_specimen` [TODO-AUTHOR: re-emit the 2026-09-19 label-width column with its definition, or retire it]. The stroke is one to two orders of magnitude wider than a crack: every pixel-overlap metric here is agreement with a brush of that width (§2.8).

**Six specimens are asserted crack-free** — `b3_amb`, `B2_amb_mosaic_2`, `B2_2_1_lbf`, `B2_2_9_lbf`, `b3_3_18lbf`, `wrought_316L_fatigue_0_cycles` (`CLEAN_SPECIMENS`, `app/core/pipeline.py:102`): **0 painted crack px, 95,351,647 painted not-crack px, 25.38%** of labelled not-crack (25.4% in `code/audit/heldout_crackfree_fp.py`'s docstring; verified, 95,351,647 / 375,632,172, *derived*). A rival set reaches §2.5.4: `build_small_component_crops.py:36` treats **ten** frames as crack-free (`CRACKFREE`), four more than the gate's six; CFREE field `site_033` = `b3_3_0lbf_268_13um`, which the gate excludes (`txm_small_component_cropkey_v2.json`); CFREE *n* = 4, underpowered.

### 2.5 Detector

**2.5.1 Features (273/px).** Seventeen isotropic hand-crafted features, σ = 1–64 px (intensity, Gaussian smoothing, gradient magnitude, signed Laplacian of Gaussian, local standard deviation; `code/txm_features.py`); 256 channels from a frozen Segment Anything ViT-H encoder, 1024 px tiles at stride **896 px**, 64 × 64 × 256 per tile, bilinear read, Hann-windowed overlap (`app/core/model.py:58–92`; `docs/TILE_SEAMS.md`).

**2.5.2 Classifier, training set.** Two scaler + MLP models, probability-averaged: 17 features, (64, 32), 204 iterations; 273 features, (128, 64), 64 iterations; `max_iter=400`, `random_state=0` (`models/f17_v5_20260824.joblib`, `models/hybrid_v5_20260824.joblib`). Rows are the annotator's corrections only; no external label enters training or the gate. Deployed `thincore_v5` (stamp `20260824_225236`): **3,530,484 rows at 273 features, crack fraction 0.4915, all 71 labelled images**, ≤ 30,000 crack rows per image, `neg_cap` = 25,536 background rows (`app/core/pipeline.py:1518`). **Labels are narrowed before sampling:** `tighten_to_image` cuts each painted region to its dark core, the ring joining the *negative* pool at true area weight (`:1496`). Its motivating numbers (26.93 px median painted half-width vs 3.16 px dark core; predicted width correlating 0.810 with label width, 0.304 with crack width) are source comments only (`:1537–1552`, `docs/THIN_LABELS.md`), no artifact, not used as evidence [TODO-AUTHOR: re-measure into an artifact or drop].

**2.5.3 Operating point; three narrowing steps, fixed load-bearing order.** **p > 0.60**; components under **2,000 px** removed (`MIN_BLOB_PX`); not-crack islands under **1,024 px** inside crack filled (`FILL_HOLES_MAX_PX`; `app/core/pipeline.py:90`, `:470`, `:527`).

1. **`tighten_to_image`** (`:1021`): keep pixels darker than the 301 px local mean; decline entirely if under 60% of the darkest fifth survives (`TIGHTEN_MIN_CORE` = 0.60, `:541`, `:550`). Its docstring carries a 2026-09-19 correction to the table above it: over all 71 frames at the deployed point the step drops about **6% of area, not the ~21% that table's first two rows imply**, moving the corpus median half-width 49.0 → 44.4 px, **a factor of 1.10, not 12×**.
2. **`clip_to_measured_width`** (`:675`): clip to measured width, never widen; ≤ 4,000 skeleton probes per frame, §2.9.2 estimator, radius by inverse-distance weighting over the 12 nearest (`WIDTH_CLIP_*`, `:595–612`). **Below 8 measurable profiles it declines, mask unchanged.** Its safety is monotonicity, not inertness: of 53 frames with pre-clip ratio ≤ 1.0, 50 lose area (median 4.6%, max 14.1%), 48 end further from 1.0 (`docs/WIDTH_REFERENCE.md:150–157`).
3. **`drop_straight_lines`** (`:809`): the §2.10 guard (`STRAIGHT_*`, `:786–788`).

The empty-mask fallback covers steps 1–2 only; step 3 is final (`:850`). That fallback existed twice — one of three checks that reported success while measuring nothing (§5.4). Declining matters most on AM, where painted crack has almost no contrast against painted not-crack: median Cohen *d* = **−0.31** against +1.76 (B2), +3.74 (B3), +1.38 (Wrought), painted crack *brighter* on 17 of 25 AM frames (`am_label_separability.json`). Five documents give an unsupported alternative set (+0.09 / +1.10 / +2.91 / +1.57: `docs/START_HERE.md:103`, `docs/WIDTH_REFERENCE.md:155`, `docs/UNLABELLED_AREA.md:156`, `app/core/pipeline.py:680`, `build_small_component_crops.py:7`); the artifact is used. Its prose says AM is "27 of 71 frames = 38%" while its table has *n* = 25; with 27 AM frames, 25 painted, both 38% and 35.2% are defensible and the artifact does not say which [TODO-AUTHOR: reconcile].

**2.5.4 Gate, negative control, contamination.** Both axes: image-grouped 5-fold cross-validation against a 0.60 IoU floor (`MIN_ABS_IOU`, `:91`); crack-free predicted area within 0.5 pp (`FP_TOL` = 0.005, `:96`). Deployed: mean IoU **0.8113**, sd 0.0229, worst fold 0.7777, precision 0.9355, recall 0.8597, crack-free area **0.1744%** (`app_data/models/retrain_history.json`, stamp `20260824_225236`, `.candidate_clean_fp` = 0.001744, `_score_clean` over `CLEAN_SPECIMENS`, `:173`).

**0.1744% is not the figure this paper reports.** Four incompatible crack-free false-positive figures, about 7× apart, at different mask stages: 0.1744% at the gate; 0.0230% → 0.0209% (`app/core/pipeline.py:1072`); 0.0230% → 0.0197% and 0.0230% → 0.0192% (`docs/OVERMARKING.md:181`, `:256`); 0.025278% → **0.023588%** on the shipped mask (`txm_small_component_adjudication.json .guard.verified`). Only the last is on this paper's mask, so the paper carries **0.0236%**.

**Its denominator is ten frames, not six, and this is a correction.** `.guard.verified` has no generator in either repository (`grep` for `crackfree_fp` returns only the artifact); its crack-free set is recoverable: the guard fired on four frames, exactly one crack-free under any definition (`b3_amb`, 0.00016907 → 0.0), and 0.025278% − 0.023588% = 0.0016907 pp = 100 × 0.00016907 / **10** exactly — a mean over the ten-frame `CRACKFREE` list (§2.4), not the gate's six. Used with that denominator stated [TODO-AUTHOR: re-emit the guard verification with its crack-free set and generator recorded].

**Partly memorisation; treated as pending.** The six gate frames are 25.38% of all labelled not-crack pixels — the model was told those pixels are background; in *sampled rows*, at `neg_cap` = 25,536, at most 153,216 of roughly 1,795,000 background rows, about 8.5% (*derived* from `n_px` 3,530,484, `crack_fraction` 0.4915). Either way the negative control is measured on frames the model was trained to call background. `code/audit/heldout_crackfree_fp.py` refits with every `CLEAN_SPECIMENS` frame's rows removed and emits `inflation_factor` to `out/txm_heldout_crackfree_fp.json`, but **it has not been run and the artifact does not exist**, so the figure is a contaminated measurement, not a bound in either direction, until it does [TODO-AUTHOR: run `code/audit/heldout_crackfree_fp.py`]. It refits only the 17-feature branch, both arms, deliberately (`:78–111`) — bounding that branch, not the deployed ensemble.

### 2.6 Metrics and their domain

Computed **inside the labelled domain**, painted crack ∪ painted not-crack; outside it there is no truth to agree with (`cldice_centreline.json .definition.domain_restriction`) — the blind spot §2.7 measures. Tprec, Tsens, clDice and IoU are used as defined there.

**Shipped values, one artifact and one mask.** `txm_width_clip_final.json` (n = 71; 61 labelled and measurable), before → after clip: clDice 0.8462 → **0.8234**, Tsens 0.7978 → **0.7605**, Tprec 0.9687 → **0.9703**, IoU 0.7334 → **0.6748**, area fraction 0.03872 → **0.03589**; paired deltas −0.0168 and −0.0260. The 0.863 / 0.955 / 0.814 values in project documents are superseded pre-clip figures from another run.

**A disagreement we do not paper over.** `cldice_centreline.json` and `txm_width_clip_final.json` both claim the deployed *pre-clip* point: clDice 0.8635 vs 0.8462, Tsens 0.8145 vs 0.7978, Tprec 0.9546 vs 0.9687, IoU 0.5047 vs 0.7334 — a 45% IoU gap the clip cannot explain, both being pre-clip; an undocumented difference in domain restriction or estimator, still unidentified. The shipped IoU's family is settled: independent 2026-09-21 run `txm_iou_ceiling_v2.json` gives shipped-mask median 0.6748000339306119, byte-identical to `.IoU_after`. clDice and Tsens *derived* from `txm_width_clip_final.json .per_frame`, IoU from `txm_iou_ceiling_v2.json .per_frame`, filename-grouped (§2.3):

| group | n | clDice | Tsens | IoU |
|---|---|---|---|---|
| AM | 25 | 0.7140 | 0.5897 | 0.5468 |
| B2 | 14 | 0.8809 | 0.7918 | 0.7836 |
| B3 | 10 | 0.9573 | 0.9757 | 0.9063 |
| Wrought | 12 | 0.8388 | 0.7874 | 0.7720 |

clDice reproduces `docs/WIDTH_REFERENCE.md:171–175` to three decimals. **Per-frame Tprec on the shipped mask is the one quantity no `out/` artifact carries**; the only per-specimen Tprec (`cldice_centreline.json`) is pre-clip, non-comparable, and labelled as such wherever used [TODO-AUTHOR: re-run per-frame Tprec on the shipped mask].

### 2.7 The unlabelled-area census

Accepted mask ∩ correction map per frame, area on label 0 counted (`code/audit/measure_unlabelled_area.py` → `txm_unlabelled_area_per_frame.json`, one record per frame with its mask version); 4,000 px component floor, matching the §2.11 sampling frame; stroke distance banded at 25 px (`.frac_unlab_within_25px_of_stroke` = 0.3239). Coverage **65 of 71 frames**; the six absent carry zero painted crack and an empty shipped mask, five before the straight-line guard and `b3_amb` only after it (§3.1). Totals: 32,504,070 accepted px, 8,016,109 unlabelled px, **24.66%**. Censuses drift — 12,303 sub-4,000 px components on 2026-09-20 (`txm_small_component_adjudication.json .population`) vs 12,583 minus 284 at ≥4,000 px on 2026-09-21 — `drop_straight_lines` shipped between runs and the mask moved underneath the measurement; the later census is used, the drift stated.

### 2.8 The IoU ceiling

Painted crack skeletonised, dilated to uniform width *w* ∈ {3, 5, 11, 21} px, scored in IoU **against the label itself** (`code/audit/measure_iou_ceiling.py`, n = 61) — what a *physically correct* answer can score, a property of the annotation alone. **Two runs exist; the later is used throughout.** `txm_iou_ceiling_v2.json` (2026-09-21) computes ceiling and shipped detector in one pass under one IoU convention; its `.supersedes` block retires three keys of `txm_iou_ceiling.json` (2026-09-19) — `headline.deployed_model_IoU_same_frames`, `headline.model_over_ceiling`, `per_specimen.*.model_iou` — for a verifiable reason: `txm_iou_ceiling_per_frame[0].iou_model` = 0.10710188107052317 is byte-identical to `cldice_centreline_per_frame[0].iou`, from a run at 02:05 on the day the clip shipped at 14:14, so the older file's "10.1× the ceiling" scored a current ceiling against a superseded mask.

| quantity | v2 (used) | v1 (retired) |
|---|---|---|
| perfect 3 px trace | **0.04977** | 0.0498 |
| **shipped detector, same convention** | **0.6748** | — |
| ratio, shipped / 3 px ceiling | **13.6x** | — |
| perfect 5 px | **0.08219** | 0.0822 |
| perfect 11 px | **0.19571** | 0.1754 |
| perfect 21 px | **0.34307** | 0.3140 |
| detector, same frames | **0.6748** (shipped) | 0.5047 (pre-clip) |
| multiple of the 3 px ceiling | **13.56** (`.shipped_over_3px_ceiling`) | 10.1 |

13.56 does not mean the detector beats a correct answer thirteenfold: agreement with these labels is maximised by being about an order of magnitude wider than a crack.

### 2.9 The width instrument

**2.9.1 Prior art, stated plainly.** The width measurement is **not novel and no part of it is claimed as a contribution.** Perpendicular profile sampling along a centreline is an established automated crack-width algorithm, *Orthogonal Profile Extraction*; half-peak edge-taking is the **ISO50** surface-determination rule, the X-ray CT metrology default, codified in **VDI/VDE 2630 Blatt 1.1**; oblique profiles overestimate by 1/cos θ, hence the minimum over 12 directions. Ours differs only in taking a *local* half-maximum per profile against an estimated background instead of ISO50's *global* air/material midpoint — a routine variation, not a new instrument (`docs/WIDTH_REFERENCE.md:28–55`; same notice at `measure_transverse_fwhm.py:9–11`). The contribution is the use: a standard metrology rule pointed at the *annotations*, not the specimen.

**2.9.2 Estimator.** Background = local mean of **non-mask** pixels by normalised convolution, windows 301, 901, 2401 px until over 2% of the window is non-mask; contrast = background − image; noise σ = 84.1st minus 50th percentile of contrast outside the mask; 12 directions, half-length 200 px, ±3 px recentring, peak must clear 2σ, **minimum** FWHM of the 12 kept (`measure_width_ratio.py:30`; `txm_width_reference.json .method`, own run 160–200 px); mask width = 2 × Euclidean distance transform. Mask ÷ image FWHM is dimensionless, 1.00 = exactly as wide as the feature.

**2.9.3 Which instrument, and the two it replaces.** Three sampling conventions give three "shipped width ratios"; one is named primary.

| instrument | artifact | sampling | n | shipped ratio | frames wider than the feature |
|---|---|---|---|---|---|
| **primary — shared skeleton** | `txm_width_ratio_unified.json` | 300 points/frame on the **widest** mask's skeleton, identical points for every variant | 64 | **0.5047** | **0 / 64** |
| absolute — self skeleton | `txm_width_reference.json` | each mask on its own skeleton, 15,397 profiles | 61 | 0.6794 | 7 / 61 |
| clip run's internal | `txm_width_clip_final.json` | the clip operator's own probes | 63 of 71 measurable | 0.5580 | 1 |

Shared-skeleton is primary **for comparisons across masks**: it removes the splice that produced the earlier published table. Cost, stated by the artifact: scoring a narrow mask at a wide mask's points necessarily lowers it, so **these are comparative, not absolute, width estimates** — 0.5047 must not be quoted as "the detector is half as wide as the crack"; self-skeleton is the absolute estimate. Over-marking counts carry their instrument (0/64, 7/61, 1/63). The paired detector-against-annotator comparison uses the 61 frames carrying both, same points and estimator, Wilcoxon signed-rank. The primary run covers 64 of 71 frames; **all seven absent frames carry zero painted crack**, but no exclusion reason is recorded and the obvious one fails: `B2_2_1_lbf` and `b3_3_0lbf_268_13um` have non-zero accepted area (0.000137, 0.000262) while `b3_amb`, whose area the guard removes entirely, *is* in the run. The generator raises on an empty wide-mask (`tight=0`) skeleton (`measure_width_ratio.py:114`), consistent but unrecorded; `docs/WIDTH_REFERENCE.md:129` states five frames were excluded, not matching the 64 records [TODO-AUTHOR: reconcile the exclusion count and record the rule in the artifact].

**2.9.4 Pre-registration.** Decided before the corpus was read: crack width is instrument-limited only if pooled transverse FWHM has a coefficient of variation below 0.35 *and* |Spearman ρ| against peak contrast below 0.3 (`txm_width_reference.json .is_it_the_instrument`); both were missed by a wide margin (§4.1). Three further rules were pre-registered in `txm_width_prereg_and_failed_variants.md`, two of which failed:

- **Per-component FWHM variant**, rejected: allowance a paired median Tsens drop of 0.02, measured drop 0.462, 23× the allowance (same file, "RESULT, 61 labelled frames, FRAC = 0.5: FAILS").
- **Adjudication of the material a narrowing step deletes**, verdict (A): the deleted material is dark, 1.63σ above the matrix reference at 42% of core contrast. The pre-registered sign test failed at p = 0.145 (33/57 frames; rule required p < 0.05) even though the paired Wilcoxon gives p = 0.016 — the pre-registered test is the one that counts, and the operating point did not change.
- **Four acceptance criteria (a)–(d)** for `clip_to_measured_width`, same file, which ends at the criteria; outcomes at `docs/WIDTH_REFERENCE.md:163–167`: (a) Tsens −0.026 against an allowance of −0.020, **FAIL** (clDice −0.017, pass); (b) median |ratio − 1| 0.399 → 0.442, **FAIL**; (c) PASS; (d) PASS. **Two of the four failed and the step ships enabled by default.** Criterion (c) also changed denominator between pre-registration ("all 7 over-marking frames") and result ("10/10 down … 0 of 53 rose"), from different width instruments [TODO-AUTHOR: state which instrument (c) was adjudicated on].

### 2.10 Straight-line artefact census and guard

Centreline wander = sd of a component's centreline about a straight-line fit along its long axis, from per-row (or per-column) centroid means, needing ≥8 distinct rows or columns (`app/core/pipeline.py:791`). Census: standalone components ≥200 px, bounding-box aspect ≥6, all 71 frames (`txm_small_component_adjudication.json .straight_line_artefacts`); the guard removes them at wander < 1.0 px, exempting any component containing a painted pixel. The 1.0 px cut replaced 2.0 px, at which the rule removed 1,360 px of painted crack on `b2_341_88_take2` against a "must not remove painted crack" condition fixed before the run (`.guard.cut_history`). Three decisions follow the artifact, not its summary keys:

- `.census.artefacts` = 6, `.artefact_px` = 17,674 and `.pct_of_accepted` = 0.0543 **include a genuine painted crack** — the wander-1.73 px, 2,433 px component on `b2_341_88_take2` the 2.0 px cut took 1,360 painted pixels from. Artefact-only totals used here: **5 components, 15,241 px, 0.0469% of accepted area** (*derived*: 17,674 − 2,433; 15,241 / 32,504,070).
- Two different sets of five components appear in the same artifact — the census five (wander 0.13–0.71 px) and the five in the 32-component tractable subset (0.13–0.93 px over 171–788 px of length). Both are true; the ranges are not merged.
- Separation is the two medians the artifact contains, **0.4736 px (straight) against 5.2511 px (curved)**. The "genuine long crack, 2.9–12.9 px" range in `docs/UNLABELLED_AREA.md` is **in no artifact** — the six curved components are not enumerated — and is not used [TODO-AUTHOR: re-emit the census with the curved components listed, or the claim rests on medians alone].

Guard verification over all 71 frames is in §3.6. It is **component-level and therefore a lower bound**: two further artefacts fused into 440k px and 228k px crack systems ride through attached to real crack, and no within-component straightness test exists (`.guard.lower_bound`).

### 2.11 Blind adjudication protocol

**The panels are language-model agents, not human experts.** There is no human expert arm. The instrument validation (§3.3) shows the agents agree with the annotator's own strokes, not that they can independently identify crack in 316L TXM. This limits the paper's central evidence: a materials venue will require at least two blinded human readers through the same protocol before the claim can stand, as the project's own generator documentation says unprompted (`code/audit/README.md`). Crops, crop key, shuffle seed, leak-guard statistics and a deterministic scorer are preserved, so the human run needs only readers; the **votes themselves are not regenerable** and ship as data (`txm_small_component_adjudication_v2.json .generators.note`).

**Design.** Fixed square crops, **no overlay drawn**, fixed window not fit-to-region (region size differs by class, so zoom would leak it); IDs shuffled on a fixed seed; no class in any filename; voters told nothing about how many fields contain crack and told not to balance answers. Panels **A** morphology, **B** conservative, told it would be heavily penalised for false alarms, **C** "would you paint here"; different batch groupings, three votes per field; yes / unsure / no = 1 / 0.5 / 0; majority "crack" at two of three yes (`code/audit/score_blind_panel.py`).

**Run 1, large unlabelled regions** (`txm_unlabelled_adjudication.json`): 88 fields of **700 × 700 px** native (≈20.5 µm), random points *inside* regions, app display transform (§2.3); 33 agents, 0 errors; sampling frame unlabelled components ≥4,000 px, holding 87.88% of unlabelled area (`txm_unlabelled_area_per_frame.json`; the adjudication artifact rounds to 87.8%). **Run 2, small isolated indications, matched on size *and* specimen** (`txm_small_component_adjudication_v2.json`, superseding `txm_small_component_adjudication.json`): 94 fields of **400 × 400 px** (≈11.7 µm), `display.npy` clipped between its 0.5th and 99.5th percentile over the whole frame (`build_small_component_crops.py:138`) — a different rendering from run 1, recorded rather than retroactively harmonised.

| run | class | definition | n |
|---|---|---|---|
| 1 | UNLAB | accepted, never labelled — **the question** | 26 |
| 1 | POS | accepted, painted crack — positive control | 26 |
| 1 | NEGP | accepted, painted not-crack — known-disagreement control | 8 |
| 1 | NEGF | accepted on a specimen asserted crack-free | 2 (underpowered; excluded by design) |
| 1 | NEGM | not accepted, painted not-crack — plain matrix, the floor | 26 |
| 2 | TEST | tractable subset: ≥200 px **and** ≥200 px from any stroke, 266–3,987 px (`txm_small_component_cropkey_v2.json`) | 32 |
| 2 | POSS | painted-crack components 200–4,000 px inside the accepted mask (observed 272–3,952 px) | 32 |
| 2 | NEGM | unaccepted painted not-crack | 26 |
| 2 | CFREE | fields on crack-free-asserted specimens | 4 |

**Leak guard, before the panels:** crop mean brightness, sd and dark fraction by class. Run 1: POS vs UNLAB p = 0.96 / 0.96 / 0.98, POS vs NEGM p = 0.92 / 0.25 / 0.52, source frames spread 20 / 23 / 24. Instrument validation is POS 24/26 against NEGM 4/26 (§3.3); the panel's false-negative rate on painted crack, 2/26, is the error bar on everything run 1 reports.

**Control matching** is run 2's methodological point: each TEST component matched to the nearest-size unused control **from the same specimen group**, shortfalls recorded, never substituted across groups. Shortfall zero; mix identical (B2 11, Wrought 13, AM 6, B3 2 in both arms); sizes match overall at p = 0.468, within every group at p = 0.333–0.937; leak-guard minimum p = 0.057. Identical mixes make the standardised control rate equal the raw one, 0.40625. **Why it exists, reported as a failure rather than deleted:** run 1 of this design matched size only; its control came out 62% AM against the test class's 41% Wrought, and standardising to the test mix moved the yes-rate 0.562 → 0.313 against a test rate of 0.344 — the entire reported gap was the confound, and the prevalence estimate built on that sensitivity was void (`.why`, `.supersedes`). §3.5 reports it as a methods failure; the scorer now prints group mix and standardised control rate every run.

**Prevalence.** Rogan–Gladen, sensitivity from the matched control arm, false-positive rate from the matrix arm, bootstrap 95% interval (`.sens` 0.40625, `.fpr` 0.11538, `.prevalence` 1.0, `.ci` [0.3209, 1.0]). The 1.0 is an arithmetic artefact of the observed rate landing exactly on the estimated sensitivity (0.40625 = 0.40625) and is never quoted without both the interval and that sentence. Panel sensitivity is low in absolute terms — 13/32 on **known** crack — capping what this arm can settle.

### 2.12 Statistics and reporting conventions

- **Tests.** Fisher's exact on majority-vote counts; Mann–Whitney U on mean vote score; Wilcoxon signed-rank for paired within-frame mask comparisons; Spearman ρ for width against contrast.
- **Sidedness is stated for every p-value, and two-sided values are reported.** The scorer calls Fisher `alternative='greater'` and Mann–Whitney two-sided; the artifacts store both families unlabelled, so each stored value was recomputed. Instrument validation 1.1554522808284791 × 10⁻⁸ (stored, one-sided) → **2.3109 × 10⁻⁸**; small-component test vs matrix 0.013402893334705752 (stored, one-sided, 13/32 against 3/26) → **0.018454**; Panel-B UNLAB vs NEGM (16/26 against 2/26) 4.2396 × 10⁻⁵ one-sided, absent from the artifact → **8.4792 × 10⁻⁵**. Already-two-sided values are labelled: `.result.vs_POS_p` = 0.3135 and `.result.vs_NEGM_p` = 4.44 × 10⁻⁸ are two-sided Mann–Whitney on vote score, **not** Fisher; the corresponding Fisher tests on majority counts are UNLAB vs POS **p = 1.000** and UNLAB vs NEGM **p = 2.2178 × 10⁻⁹** (computed here from 25/26, 24/26 and 4/26).
- **Rounding.** Figures are never rounded in the direction that flatters the method. Two corrections: `docs/UNLABELLED_AREA.md:32` calls the unlabelled share on `b2_336_25` "95% of everything the detector marked" where the per-frame record gives **94.65%** (0.9464765648728102), and `b2_336_25` is not the corpus maximum — `HC_316L_fatigue_1770_tip_zoom` at **97.27%** is, then `HC_316L_fatigue_1790_tip_zoom_2` at 95.01% (*derived*, same field); ≥4,000 px coverage is **87.88%** (0.8787714837709917), not the 87.8% stored in the adjudication artifact.
- **Numbers with no artifact field are not used.** Three are named: the "0.197% of painted not-crack" false-positive figure (`docs/UNLABELLED_AREA.md:119–121`), surviving elsewhere only as artifact prose (`txm_unlabelled_adjudication.json .unexpected.caveat`) with no numeric field behind it, nearest numeric match 0.0197% a different quantity with a different denominator; the 2.9–12.9 px curved-crack wander range (§2.10); and the "0.09%" and "0.13%" area shares of the 1 px and 2–9 px components, whose counts (6,921 and 3,141) are in `txm_small_component_adjudication.json .population` but whose area shares are not.

### 2.13 Software, artifacts, and what is not reproducible

Python: scikit-learn, scikit-image, SciPy, NumPy; encoder frozen Segment Anything ViT-H. Audit generators are in `code/audit/`, mapped to artifacts in §7 with the gaps recorded there: `measure_transverse_fwhm.py` is listed as emitting `out/txm_transverse_fwhm.json`, which does not exist (§2.9.4's numbers come from `txm_width_reference.json`, which has no listed generator); `measure_iou_ceiling.py` and `heldout_crackfree_fp.py` are not listed at all, though the first produced the artifact §2.8 relies on; the guard-verification generator is in neither repository, which is why its crack-free denominator had to be recovered arithmetically; and the panel votes are not regenerable (§2.11). Three checks in this project reported success while measuring nothing; §5.4 reports them rather than omitting them.

## 3. Results: metric blindness and the unlabelled quarter

Outside the operator's strokes there is no reference; labels cap IoU below 0.05.

### 3.1 A quarter of the accepted mask was never labelled either way

Of 32,504,070 px accepted across 65 frames, 8,016,109 (**24.7%**, area-weighted) carry
no judgement in the correction map (`txm_unlabelled_area_per_frame.json`,
`.corpus_frac_unlabelled` = 0.2466). By set (`.per_group.<G>.*`):

| set | n | accepted px | unlabelled share |
|---|---|---|---|
| AM / HC 316L | 24 | 5,681,798 | **51.06%** |
| B2 | 16 | 10,348,884 | **36.84%** |
| Wrought | 12 | 9,098,031 | **10.67%** |
| B3 | 13 | 7,375,357 | **4.49%** |
| **corpus** | **65** | **32,504,070** | **24.66%** |

Nineteen of the 65 are over half unlabelled (*derived*, `.per_frame[].frac_unlabelled`);
the extreme, `HC_316L_fatigue_1770_tip_zoom`, 97.27% of accepted area; `b2_336_25`,
**94.65%** of what the detector marked and **17.34%** of the frame
(`.unlabelled_frac_of_frame` = 0.17338).

The area resolves into 12,583 components; 284 of ≥4,000 px hold **87.88%**
(`.frac_unlab_in_components_ge_4000px` = 0.87877); only **32.39%** lies within 25 px of
a painted stroke (`.frac_unlab_within_25px_of_stroke`). The other 12,299 fall below
4,000 px (§3.5).

**Six frames are missing.** The census covers 65 of 71. `measure_unlabelled_area.py`
skips frames with an empty accepted mask or no correction map, tags both causes with one
string, records neither; only `b3_amb` is explained — accepted area zeroed by the
straight-line guard (§3.6). **[TODO-AUTHOR]** Re-run with a distinct skip reason per
frame and skipped records kept, so all six are accounted for.

### 3.2 The annotation caps IoU at 0.0498 for a physically correct crack

Painted crack skeletonised, re-dilated to width *w*, scored against its own label (§2.8;
`txm_iou_ceiling.json`, `.method`; per-frame `txm_iou_ceiling_per_frame.json`, n = 61).

| trace width | 3 px | 5 px | 11 px | 21 px |
|---|---|---|---|---|
| IoU against the operator's label | **0.0498** | 0.0822 | 0.1754 | 0.3140 |

(`.headline.perfect_3px_trace_IoU` / `.perfect_5px` / `.perfect_11px` / `.perfect_21px`,
each the verified median over 61 per-frame records.) A 21 px trace scores below 0.5 on
44/61 frames (*derived*, `iou_w21`). The driver is label width: median stroke **73.4
px** (*derived* median of `label_width_px`; `.headline.median_label_width_px` rounds to
73.0), 10th–90th percentile **19.7–165.3 px**, maximum 337.5 px (*derived*, linear
interpolation). By set:

| set | AM | B2 | Wrought | B3 |
|---|---|---|---|---|
| n | 25 | 14 | 12 | 10 |
| 3 px ceiling | 0.0732 | 0.0339 | 0.0343 | 0.0291 |
| median label width | 36.4 px | 88.1 px | 96.6 px | 126.2 px |

(`.per_specimen.<G>.{n, ceiling_3px}`; widths *derived* as per-group medians of
`label_width_px`, artifact-rounded to 36 / 88 / 97 / 126 px.) The same construction on
an SEM corpus (59 px median brush) gives 0.1662 at 3 px (`.why`, `.verdict`), three times
the TXM ceiling. IoU here ranks imitation of a ~73 px paintbrush; it is not comparable
with published IoU from tighter-annotated corpora.

**We report no "× the ceiling" multiple.** `txm_iou_ceiling.json` gives
`.headline.deployed_model_IoU_same_frames` = 0.5047, `.model_over_ceiling` = 10.1 and
`.per_specimen.<G>.model_iou`, but all 61 records are byte-identical to the
pre-`WIDTH_CLIP` run in `cldice_centreline_per_frame.json` (field `iou`, not
`iou_model`; record 0 = 0.10710188107052317 in both) — a mask no longer shipped, whose
superseded clDice / Tprec / Tsens medians are 0.8635 / 0.9546 / 0.8145 against the
shipped 0.8234 / 0.970 / 0.761. The ceiling is label-only, unaffected. **[TODO-AUTHOR]**
Re-run the ceiling script against the shipped mask if a multiple is wanted.

### 3.3 Blind adjudication: instrument validation

Blind adjudication (`txm_unlabelled_adjudication.json`; design, blinding, leak guard:
§2.11). The panels are language-model agents, not human experts (`.design.votes`: 3
votes per field, 33 agents). UNLAB fields: components ≥4,000 px, median 9,706.5 px,
range 4,001–64,509 px (*derived*, `.per_field[].comp_px`). All p-values two-sided;
stored one-sided values flagged. On the two known classes (POS 24/26, NEGM 4/26 in §3.4;
`.instrument_validation`): Fisher exact p = 2.31 × 10⁻⁸, Mann–Whitney on vote score p =
4.50 × 10⁻⁸ (*recomputed; stored `.fisher_p` = 1.16 × 10⁻⁸, `.mannwhitney_p` = 2.25 ×
10⁻⁸ are one-sided*). Panel false-negative rate on known crack: 2/26
(`.panel_false_negative_rate`) — the error bar on §3.4.

### 3.4 The large unlabelled regions read as crack

| class | majority called crack | mean vote score |
|---|---|---|
| POS (painted crack) | 24/26 (92.3%) | 0.8910 |
| **UNLAB (never labelled)** | **25/26 (96.15%)** | **0.8590** |
| NEGP (painted not-crack) | 7/8 (87.5%) | 0.8333 |
| NEGM (plain matrix) | 4/26 (15.4%) | 0.2115 |
| NEGF (crack-free specimen) | 0/2 | 0.0000 |

(`.result.UNLAB_majority_yes` / `.UNLAB_pct` = 96.154, `.instrument_validation`; scores
*derived*, `.per_field[].s`.) Area-weighted, **93.42%** of sampled unlabelled area reads
as crack (`.result.area_weighted_pct`).

UNLAB vs POS: **Mann–Whitney p = 0.3135** (`.vs_POS_p`, the artifact's test), Fisher on
25/26 vs 24/26 p = 1.000 (*recomputed*) — indistinguishable from painted crack. UNLAB vs
NEGM: **Mann–Whitney p = 4.44 × 10⁻⁸** (`.vs_NEGM_p`), Fisher p = 2.22 × 10⁻⁹
(*recomputed*).

**Strictest reader.** Panel B (false-alarm penalised, default "no"): yes-rate 49% vs A
75%, C 69% (`.panel_agreement.yes_rate`). B alone: POS 20/26, UNLAB 16/26, NEGM 2/26
(`.result.conservative_panel_only`); UNLAB vs POS Fisher p = 0.3678 (`.UNLAB_vs_POS_p`,
two-sided), vs NEGM p = 8.48 × 10⁻⁵ (*recomputed; `UNLABELLED_AREA.md` prints one-sided
4.2 × 10⁻⁵*). Pairwise agreement A–B 73%, A–C 90%, B–C 76% (`.panel_agreement`).

**Coverage.** The sampled class holds 87.9% of unlabelled area (§3.1; `.result.coverage`
prints 87.8%, rounding the wrong way), so the claim covers **21.7%** of accepted area
(0.8788 × 0.2466 = 0.2167).

**The one genuine false positive.** `field_085`, 27,179 px on
`wrought_316L_fatigue_1280_cycles`: A and B read a specimen free surface with parallel
Fresnel fringes, **C voted crack** (`.per_field[]`: 1 yes / 2 no, `s` = 0.333;
`.the_one_rejection.why`) — one rejection in 26.

On NEGP — accepted regions the operator painted not-crack — the panel sided with the
detector 7/8, mean 0.8333 against 0.8910 (`.unexpected.what`). Caveat: n = 8, the
*largest* detector/operator disagreements, not a random sample — selection, not
accuracy. **This is not a measurement of annotation accuracy and must not be quoted as
one**; it supports only that a false-positive rate measured against painted not-crack
area is an upper bound, part of that area being the annotator. **[TODO-AUTHOR]**
`UNLABELLED_AREA.md` and `.unexpected.caveat` cite "the 0.197%-of-painted-not-crack
figure", in no artifact in either repository; the nearest match, 0.0197%, is predicted
area on crack-free specimens — different quantity and denominator, off by ten. Source or
drop it.

### 3.5 The small isolated indications: a confound, and the reversal it forced

**Most small components are not detections.** No accepted component falls below
`MIN_BLOB_PX = 2000`: the 12,299 unlabelled components under 4,000 px are unpainted
remainder where accepted components meet the label map — stroke rim, not findings.
982,920 px, 12.2% of unlabelled area; 6,921 single-pixel, 3,141 of 2–9 px; 80.4% of
their area within 25 px of a painted stroke, median distance 11 px
(`txm_small_component_adjudication.json`, `.population.*`). *Those are the 2026-09-20
run (12,303 components); the 2026-09-21 census on the current mask gives 12,299
(`txm_unlabelled_area_per_frame.json`, 12,583 minus 284): the mask moved under the
measurement when `drop_straight_lines` shipped. We use 12,299; area, distance and
tractable-subset figures are the earlier run, not recomputed.*

Components ≥200 px **and** ≥200 px from any painted stroke number **32, 74,404 px,
0.229% of accepted area** (`.tractable_subset.n`, `.px`, `.share_of_all_accepted_area`)
— the class's whole exposure even if all were false positives.

**First run, confounded.** A second blind panel scored test 11/32 against size-matched
control 18/32, one-sided p = 0.0656, concluding the instrument weak at this size
(`.adjudication`).

**The entire gap was a confound.** Size was matched; specimen was not. The control was
62% AM, the test class 41% Wrought; standardising the control to the test specimen mix
moves its yes-rate 0.562 → 0.313 against a test rate of 0.344
(`txm_small_component_adjudication_v2.json`, `.why`). No gap; and the sensitivity the
prevalence estimate rested on was void.

**Second run, matched on size and specimen.** Identical group mix in both arms (B2 11,
Wrought 13, AM 6, B3 2); size matched overall p = 0.468, within every group p =
0.333–0.937; leak guard clear at minimum p = 0.057 (`.matching.*`).

| class | majority called crack | mean vote score |
|---|---|---|
| painted crack, size- and specimen-matched | 13/32 (40.6%) | 0.4010 |
| **isolated small indications** | **13/32 (40.6%)** | **0.4583** |
| plain matrix | 3/26 (11.5%) | 0.1154 |
| crack-free specimens, false positive by assertion | 0/4 | 0.0833 |

(`.by_class.{POSS, TEST, NEGM, CFREE}.{majority_yes, score}`.) Instrument validation:
control vs matrix, Fisher p = 0.0185 (*stored `.instrument.fisher_p` = 0.0134
one-sided*). Test vs matched control: **13/32 vs 13/32, Fisher p = 1.000**
(`.test_vs_control_fisher_p`), Mann–Whitney p = 0.5319 (`.test.vs_control_p`). Test vs
matrix: Fisher p = 0.0185 (*stored `.test_vs_matrix_fisher_p` = 0.0134 one-sided*),
Mann–Whitney p = 2.06 × 10⁻⁴ (`.test.vs_matrix_p`).

**The conclusion reverses.** The small isolated indications are indistinguishable from
known crack of the same size in the same specimens and clearly separated from matrix, as
in §3.4. A clean-looking negative was killed by a matching criterion we had not
enforced, found by standardising a control we had already reported.
`code/audit/score_blind_panel.py` now prints group mix and standardised control rate.

**Residuals.** Rogan–Gladen prevalence with the matched sensitivity: 1.0, bootstrap 95%
CI **[0.3209, 1.0]** (`.prevalence`, `.ci`). No artifact says so: the 100% point
estimate is arithmetic, the observed rate landing exactly on the sensitivity (`.obs` =
`.sens` = 0.40625) — not a claim that every component is crack. The 40.6% control rate
is low: the panel misses most known crack at this size, capping what the arm can settle.
Panel yes-rates test/control: A 69/59%, B 25/22%, C 41/38% (`.per_panel`); the
conservative lens is near silent. The crack-free arm, n = 4, proves nothing alone,
though Panel A called one of its four fields crack (majority 0/4; score 0.0833).

### 3.6 One class in the unlabelled area is not crack: straight-line artefacts

Five of the 32 tractable components read as obvious cracks on every summary statistic —
elongation 24–197, 12–49 sigma darker than their surroundings — yet run near-straight
(`.straight_line_artefacts.finding`). **The panel did not settle them**: three
majority-rejected, two majority-called crack (*derived*, `.per_field[]` with `cls ==
"TEST"` and `el >= 6`: scores 0.000, 0.000, 0.333, 0.667, 0.667).

Over all 71 frames, standalone components ≥200 px with aspect ratio ≥6
(`.census.scope`): median centreline wander **0.4736 px** straight against **5.2511 px**
curved (`.median_wander_straight`, `.median_wander_curved`). Five of the six enumerated
components wander 0.13–0.71 px over 415–877 px at elongations 37.7–197.0
(`.census.components[]`); one is on `b3_amb`, asserted crack-free throughout. **The
artefact class is therefore 5 components, 15,241 px, 0.0469% of accepted area**; the
top-level `.census.artefacts` = 6, `.artefact_px` = 17,674, `.pct_of_accepted` = 0.0543
include a sixth, 55.9% painted crack (*derived*: `cut_history` records 1,360 px
removed), not to be quoted; it is why the guard threshold sits at 1.0 px, not 2.0 px
(§2.10).

Guard-on vs guard-off, all 71 frames: fires on 4, removes those 15,241 px and **0 px of
painted crack**, against a pre-registered criterion of zero. clDice unchanged to sixteen
decimal places (0.8234071528920338 before and after); false-positive area on crack-free
material falls from 0.025278 to 0.023588 in the artifact's units, a 6.7% relative
reduction (`.guard.verified.*`). On `b3_amb`, accepted area goes to exactly zero
(`.guard.verified.detail[1]`).

- **Partly a memorisation result.** The six crack-free specimens sit inside the training
  background labels (§6.2): 0.025278 → 0.023588 is measured on pixels the model was told
  are background — an optimistic bound, the favourable case. The held-out
  refit specified in `code/audit/heldout_crackfree_fp.py`, emitting
  `out/txm_heldout_crackfree_fp.json`, **does not yet exist**. **[TODO-AUTHOR]** Report
  the held-out figure beside the deployed one when it does; report both, do not replace.
- **[TODO-AUTHOR]** The two crack-free figures are in inconsistent units — corpus value
  apparently a percentage, per-frame `area_before` a fraction — to reconcile before
  either is printed with a unit.

**The guard is a lower bound.** Component-level, it catches standalone artefacts only;
two further artefacts, fused into 440k px and 228k px crack systems, ride through
attached to real crack (`.guard.lower_bound`). A within-component straightness test
would find them; none exists.

### 3.7 Figures

`results/blind_adjudication.png` (1250 × 2048), the §3.3–3.4 panel: five classes with
vote tallies and reasons, including `field_085`; cited by `UNLABELLED_AREA.md`, present.
`results/small_component_adjudication.png` (1085 × 2164), the §3.5 panel, renders the
**superseded run 1**, 18/32 control against 11/32 test: not to be printed.
**[TODO-AUTHOR]** (i) Regenerate it from `txm_small_component_adjudication_v2.json` at
13/32 against 13/32, specimen-matched. No figure exists for (ii) the §3.1 per-group
census, per-frame points on group bars (`txm_unlabelled_area_per_frame.json`), or (iii)
the §3.2 IoU-ceiling curve, IoU vs trace width, 61 per-frame traces
(`txm_iou_ceiling_per_frame.json`).

## 4. Results: width, and what the annotations cannot adjudicate

The instrument is standard metrology, not claimed as new (§2.9.1).

### 4.1 An image-side width reference

11,426 profiles on 61 frames: transverse FWHM median 45.0 px, 10th–90th percentile 3–147 px (`txm_width_reference.json`, `.is_it_the_instrument`; estimator §2.9.2). Pre-registered: instrument-limited only at pooled CV < 0.35 **and** |ρ| (FWHM vs peak contrast) < 0.3. Measured CV = 0.9157, ρ = +0.6732, frame medians ρ = +0.7558, p = 1.914 × 10⁻¹² (same artifact) — **both thresholds missed by a wide margin**. No fixed point-spread produces a width ranging 50× that rises with peak depth: the width is the crack's. Median 1,305 nm (`.fwhm_median_nm`; §2.2); figures below in pixels.

### 4.2 One instrument, one frame set

Three estimators gave three "shipped width ratio" figures before the splice was caught (§2.9.3). Every comparison below is one run — same estimator, same frames, **same sample points**, the widest variant's skeleton — **comparative across masks, not absolute width estimates** (`txm_width_ratio_unified.json`, `.what`, `.rows.*`; `code/audit/measure_width_ratio.py`).

| mask (n frames) | median mask width / image FWHM | IQR | frames wider than feature |
|---|---|---|---|
| wide, `tight=0` (64) | 0.672 | 0.576–0.935 | 14 |
| tighten only, no clip (64) | 0.590 | 0.512–0.739 | 7 |
| **shipped** (64) | **0.505** | 0.444–0.587 | **0** |
| human brush label (61) | 0.529 | 0.431–0.770 | 11 |

### 4.3 Detector and annotator widths

On the 61 frames carrying both: shipped 0.5061, annotator 0.5289, paired median difference **−0.0113**, paired Wilcoxon **p = 0.0613** (`.shipped_vs_label`, verdict "indistinguishable"), an order of magnitude below either mask's spread (IQRs 0.143 shipped, 0.338 label; *derived* from `.rows.*.iqr`). Per group (*derived* medians of `.per_frame[].ratio_shipped`/`.ratio_label`; three frames carry a mask and no label, one B2, two B3):

| group (n shipped / label) | shipped | label |
|---|---|---|
| B2 (15 / 14) | 0.4915 | 0.5462 |
| B3 (13 / 11) | 0.4548 | 0.4526 |
| AM / HC_316L (24 / 24) | 0.5634 | 0.5819 |
| Wrought (12 / 12) | 0.5064 | 0.7032 |

**The two-annotator reading is an analogy, not a measurement**: no second annotator, no frame known to have been labelled twice **[TODO-AUTHOR: state whether any frame was annotated independently more than once; if so, measure the between-annotator width difference and replace this analogy]**. **p = 0.0613 is a failure to reject, not equivalence**; no artifact in `out/` holds an equivalence test or CI on the paired difference **[TODO-AUTHOR: add a TOST or bootstrap CI on the −0.0113 paired difference to state what differences the data exclude]**.

### 4.4 Both masks under-mark; the detector never over-marks

Against 1.0 both fail in the same direction: shipped p = 3.53 × 10⁻¹², label p = 1.06 × 10⁻⁶ (`.shipped_vs_one_p`, `.label_vs_one_p`). Shipped is wider than the feature on **0 of 64 frames**, largest anywhere 0.966 (*derived*, max of `.per_frame[].ratio_shipped`); the brush on 11 of 61. Seven frames over-mark under the self-skeleton estimator (§2.9.3), all hairlines — `HC_316L_fatigue_600`, FWHM 7 px against a 36 px mask: `MIN_BLOB_PX = 2000` filters components thinner than roughly 2000/length px, so only hairlines the model drew fat are exported (`txm_width_prereg_and_failed_variants.md`).

### 4.5 IoU charges the detector for exactly this agreement

For a **correctly centred** mask, IoU here is nearly a width ratio: the §3.2 ceiling rungs 0.0498, 0.0822, 0.1754, 0.3140 (3, 5, 11, 21 px against a 73.4 px median label width) are 0.0166, 0.0164, 0.0159, 0.0150 per pixel of marked width (*derived*) — flat to within 11% over a sevenfold change. Right width does not rescue a mis-centred mask, and on a 73 px brush the label skeleton can sit far from the crack.

Clipping to measured width moved the corpus ratio 0.610 → 0.558, over-marking frames 10 → 1 (all ten down; `HC_316L_fatigue_1250` marginally over at 1.06), costing IoU 0.7334 → **0.6748**, clDice 0.8462 → **0.8234**, Tsens 0.7978 → **0.7605**, only Tprec rising, 0.9687 → 0.9703 (`txm_width_clip_final.json`): 5.9 IoU points for width fidelity.

**Two of the four pre-registered clip criteria failed; the step ships by default**: (a) Tsens drop allowed 0.02, observed 0.026, missing by 0.006; (b) median |ratio − 1| must fall, went 0.399 → 0.442, and is mis-specified besides (`docs/WIDTH_REFERENCE.md`). Five of the six worst Tsens losses are over-marking frames the step exists to fix; the sixth, `b3_385_63um_ZOOM` (Tsens 1.000 → 0.820, ratio 0.40×), was already well under-marked — a genuine cost, not a metric artefact.

The ceiling artifact's "10.1×" rests on a superseded pre-clip mask and is quoted nowhere here (§3.2). The 73.4 px label width and the 0.529 label ratio come from **different estimators and must not be divided into one another**: the ceiling's per-frame label width is a median 1.57× the profile estimator's on the same 61 frames (*derived*, `txm_iou_ceiling_per_frame.json` `.label_width_px` vs `txm_width_reference.json` `.per_frame[].w_lab`); nothing in `out/` reconciles them.

### 4.6 What the annotations cannot adjudicate

Pre-registered (`txm_width_prereg_and_failed_variants.md`): over 57 frames with a real disagreement, contrast of label-centreline pixels the shipped mask covers and a FRAC 0.4 narrowing drops, against kept centreline and matrix (unlabelled, unmasked, same frame) — matrix −0.0016, **dropped +0.0379 (1.63σ above matrix)**, kept +0.0969 (4.83σ), pre-registered midpoint +0.0502. Dropped fell below the midpoint on **33/57** frames, sign test **p = 0.145** against a required p < 0.05; a paired Wilcoxon gives p = 0.016, but the pre-registered test counts: the deleted material is dark, and narrowing further is not licensed.

The dropped halo sits at 39% of kept-centreline contrast (*derived*, 0.0379/0.0969; the source note asserts 42%, unreproducible from its own medians), 1.63σ above background: not matrix, but faint crack, partly closed flank, out-of-plane crack and instrument skirt are not separable from these images and annotations. The brush asserts a region, not a boundary: 36 px mean stroke width on AM to 126 px on B3, median 73.4 px (`txm_iou_ceiling.json`, `.per_specimen.*.label_width_px`). Nor can a half-maximum cut adjudicate the halo it is asked about. **[TODO-AUTHOR: settle with a depth-axis modality or a higher-magnification repeat acquisition of one frame; state which if either is available.]**

A second pre-registered variant failed: per-component FWHM at FRAC 0.5 took Tsens 0.798 → 0.261 and clDice 0.846 → 0.412, 23× the allowed drop; only FRAC 0.2 clears the guard, a 6% width reduction.

### 4.7 A straight-line artefact class: the crack signature is also the artefact's

Elongation and contrast cannot separate them; centreline wander can (§3.6, §2.10).

Five artefacts: elongations 134, 197, 175, 94, 38 at 2,090–4,094 px, above the `MIN_BLOB_PX = 2000` floor, the smallest by 90 px, transverse-width sd 0.37–2.19 px against 6.39 px for the sixth, painted-crack component. Two in one frame at x = 6353 and 6359, a third in a sibling frame of the same specimen at x = 6366 — within 13 px of one column across two frames — a fourth at x = 0, the fifth interior at x = 1,689 (`txm_small_component_adjudication.json`, `.straight_line_artefacts.census.components[].x`).

Two of the five reached the blind 3-panel adjudication and split: elongation-134 crack 2 of 3, elongation-197 not-crack 0 of 3 (*derived*, `.per_field` `site_007` and `site_083` matched to `.census.components` by frame id, pixel count and elongation). Those panels are language-model agents (§2.11): the split illustrates the signature, not an adjudication of these components.

The separation claim rests on two census medians and five enumerated wander values; the "2.9–12.9 px" curved range in `docs/UNLABELLED_AREA.md` is in no artifact in `out/` and stays out (§2.10). Component-level, the guard is a **lower bound**: two artefacts fused into 440k and 228k px crack systems ride through attached to real crack, no within-component test (`.guard.lower_bound`).

### 4.8 Limitations

- **Estimator dependence is the dominant uncertainty.** Three estimators of the shipped width ratio give 0.505, 0.558 and 0.679, over-marking on 0 of 64, 1 of 63 and 7 of 61 frames (§2.9.3); every number above is the shared-skeleton one, and mixing them would change the qualitative conclusion about over-marking.
- **Conditioned on measurable features.** A profile enters only if its peak clears 2 background σ; on AM / HC_316L — 38% of the corpus (§2.5.3) — cracks are intensity-invisible, Cohen d = +0.09 against +1.10 to +2.91 on the other three groups, measured against each crack's own local ring (`am_label_separability.json`, `.CORRECTION_ring_control.result`; that file's top-level AM d = −0.31 is explicitly **withdrawn**, an artefact of using painted not-crack as reference). AM width figures therefore rest on AM locations bright enough to profile and do not represent AM crack as a whole; the two profile counts in `txm_width_reference.json` (11,426 instrument test, 15,397 ratio table, both over 61 frames) reflect different gating and are not interchangeable.
- **No absolute width claim is made.** All ratios are mask width against image FWHM at shared sample points, and FWHM is a convention (ISO50), not a physical crack-opening displacement.
- **No second annotator, and no repeat annotation** (§4.3).

## 5. Discussion

### 5.1 Report what fraction of the prediction the evaluation could see

Every accuracy number here is on pixels the annotator touched; **24.7 % of accepted mask area was never labelled either way** (0.2466, area-weighted, 65 frames; `txm_unlabelled_area_per_frame.json` `.corpus_frac_unlabelled`; §3.1). Three disclosures belong beside the headline metric.

- **E1, evaluated fraction:** predicted positive area inside the annotated domain (painted crack ∪ painted not-crack), area-weighted, per group, with denominators — 75.3 % visible, 95.5 % on B3, **48.9 % on AM** (*derived*).
- **E2, label-geometry ceiling:** what a physically correct instance scores — IoU 0.0498 against median label width 73.4 px (§3.2, §5.2); a property of the annotation, improved by no detector.
- **E3, disposition of the unevaluated fraction:** blind adjudication of components ≥ 4000 px (87.9 % of unlabelled, 21.7 % of accepted area) returns crack in **25/26 fields**, 93.4 % area-weighted (§3.4); the one rejection recorded, not absorbed — `field_085`, 27,179 px, a specimen free surface, a real false positive (`txm_unlabelled_adjudication.json` `.the_one_rejection`).

E1 alone is not a performance metric: low can mean unannotated crack found (this corpus's answer, per E3) or hallucination. Only E1 with E3 separates them, subject to §6.1.

### 5.2 What a ceiling does and does not license

E2 changes how an IoU is read, not whether the detector works: 0.5 against a 73 px median brush is agreement with brush geometry. In one pass under one convention the shipped mask scores IoU 0.6748 against the 0.0498 ceiling, **13.6x** (`txm_iou_ceiling_v2.json`, 2026-09-21; §2.8). The older multiple (`txm_iou_ceiling.json` `.headline.model_over_ceiling` = 10.1, with every `.per_specimen.<G>.model_iou` there) was computed on the **pre-`clip_to_measured_width`** mask — per-frame values byte-identical to `cldice_centreline_per_frame.json`, run 02:05 on 2026-09-19, before the clip shipped at 14:14 that day — and is retired by name in the v2 `.supersedes` block. The ceiling, from the labels alone, is unaffected.

Crack IoU is also not portable: Segment-Any-Crack 0.4413 IoU / 0.6122 F1 on OmniCrack30k, MixSegNet 0.848 / 0.915 on surface-crack photography (`model_vs_literature.json`), both against undisclosed brush geometry; two widely cited figures are not crack IoU at all — a nickel-superalloy U-Net's 0.995 with MCC 0.826 on the same runs (possible only if background-inclusive), a shale FRRN-B's 85 %, a three-phase mean whose crack-class value is unpublished.

### 5.3 Instruments that do not pass through the annotations

One instrument must not read the annotations that bound the agreement metrics. The transverse-profile width ratio scores a mask against the image — one estimator, one frame set, shared sample points (`txm_width_ratio_unified.json`, n = 64) — and finds detector and annotator indistinguishable: 0.506 against 0.529 on the 61 frames carrying both, difference −0.011, Wilcoxon p = 0.061, both under-marking the dark feature, the mask never wider than it (0 of 64; §4.2–4.4). It is not an absolute width estimate. The other such instrument is predicted area on crack-free specimens; §6.2 is why ours is not one.

### 5.4 Report the decision rules that failed

**Width-narrowing candidate.** Four tests fixed before the data were read (`txm_width_prereg_and_failed_variants.md`); **two failed and are reported as failures**. The per-component FWHM variant missed its Tsens allowance by 23×, rejected outright. The adjudication of what the narrowing deletes failed its sign test — dropped centreline below the midpoint on 33/57 frames, p = 0.145 against a required p < 0.05 — verdict (A): the dropped material is not matrix, and an aggressive cut deletes real signal (§4.6). The other two settled: local-FWHM replacement passes only at FRAC 0.2, a 6 % width reduction; the instrument test returned "width is real" (§6.4).

**Width clip.** Four criteria (a)–(d) fixed before the run (`docs/WIDTH_REFERENCE.md:157–165`). **(a) failed on Tsens by 0.006** (paired −0.026 against allowance −0.020; `txm_width_clip_final.json` `.Tsens_paired_delta`). **(b) failed outright** (median |ratio − 1| 0.399 → 0.442, a figure in `docs/WIDTH_REFERENCE.md:165` and **in no artifact**), and is mis-specified: no clip-only operator can improve a population statistic on a corpus whose median ratio is already 0.61× (`.width_ratio_before` = 0.6096). (c) and (d) passed. **The step nonetheless ships on by default** (§4.5).

**Three checks reported success while measuring nothing** (`txm_small_component_adjudication.json` `.checks_that_reported_success_while_measuring_nothing`): a NumPy-2.0 `ptp` removal crashed the straightness census on exactly the 10 frames with long components, and the summary printed 0.00 % over the survivors; the first guard verification compared `effective_mask` with `effective_mask`, so neither arm could fail; a duplicated empty-result fallback let crack-free `b3_amb` ship its false positive through a guard reporting PASS. The self-test covers the last two (`app/selftest.py:977–1006`) and that width clipping only removes and is not inert (`:963`); **no self-test asserts the census property**, and nothing exercises the `ptp` path.

---

## 6. Limitations

### 6.1 The adjudication panels are language-model agents, not human experts

The blind adjudication resolving the unlabelled area (§5.1, E3) was voted by language-model agents — 3 panels, 3 votes per field, 33 agents, 0 errors (`txm_unlabelled_adjudication.json` `.design`) — with **no human expert arm**. Validation: panels recover the annotator's own painted crack at 24/26 against 4/26 on plain matrix (`.instrument_validation`; stored `fisher_p` = 1.16 × 10⁻⁸ and `mannwhitney_p` = 2.25 × 10⁻⁸ are **one-sided**; two-sided 2.31 × 10⁻⁸ and 4.50 × 10⁻⁸, *derived*) — agreement with this annotator's strokes, not competence at identifying fatigue crack in 316L TXM, which is untested. **[TODO-AUTHOR]** At least two blinded human readers through the identical protocol, before this evidence is offered as settled.

Sensitivity is also low at small component size: 13/32 on known painted crack (`txm_small_component_adjudication_v2.json` `.by_class.POSS`), capping what that arm can settle. The test class matched it exactly, 13/32, Fisher p = 1.000 (`.test_vs_control_fisher_p`), separating from matrix at 3/26 (`.by_class.NEGM`; Fisher p = 0.0134 **one-sided**, two-sided 0.0185, *derived*; two-sided Mann–Whitney on the vote score p = 2.06 × 10⁻⁴). `prevalence` = 1.0 is an artefact of the observed rate equalling the sensitivity (0.40625 = 0.40625); never quote it without its bootstrap CI **[0.321, 1.000]** (`.ci`) and that sentence. Crack-free arms are underpowered by design: n = 2 (NEGF, excluded), n = 4 (CFREE, 0/4).

### 6.2 The crack-free specimens are in the training set

The six specimens confirmed crack-free (`CLEAN_SPECIMENS`, `app/core/pipeline.py:102`: `b3_amb`, `B2_amb_mosaic_2`, `B2_2_1_lbf`, `B2_2_9_lbf`, `b3_3_18lbf`, `wrought_316L_fatigue_0_cycles`) — the only over-prediction check needing no pixel ground truth — are in training: 0 crack px, **95,351,647 not-crack px, 25.4 % of every background label** (`code/audit/heldout_crackfree_fp.py` docstring; in that source file alone, no artifact). The shipped **0.0236 % of crack-free area** (`txm_small_component_adjudication.json` `.straight_line_artefacts.guard.verified.crackfree_fp_after` = 0.023588, stored without a unit, read as a percentage; 0.0253 % before the straight-line guard) is therefore scored on pixels the model was told are background: partly a memorisation result. Five older figures from earlier mask stages circulate in our documentation — 0.174 %, 0.0230 %, 0.0209 %, 0.0197 %, 0.0192 % (`README.md:296`, `docs/START_HERE.md:60`, `docs/OVERMARKING.md:181,256`, `docs/TILE_SEAMS.md:169`, `app/core/pipeline.py:1072`); only 0.0236 % is on the shipped mask, and only it is reported.

The held-out refit has since been run at the deployed architecture; it and the failed v1 attempt are in the Addendum. **[TODO-AUTHOR]** fold those figures into this section, beside the deployed figure wherever it appears.

Two weaknesses are structural. The six are not independent specimens — three B2, two B3, one Wrought, at ambient or zero load, from the groups supplying the cracked frames. And **AM / HC_316L contributes no crack-free control at all**: 27 of 71 frames, highest unlabelled fraction at 51.1 %, cracks intensity-invisible against their surroundings (Cohen's d = +0.09 against +1.10 / +2.91 / +1.57 for B2 / B3 / WR, `am_label_separability.json` `.CORRECTION_ring_control.result.*.d_vs_local_ring`; the earlier painted-not-crack reference is withdrawn in that artifact). The false-positive axis is silent on the hardest third of the corpus.

### 6.3 n = 71 frames from four specimen groups, which are not independent

71 frames — AM / HC_316L 27, B2 17, Wrought 14, B3 13 (`txm_width_clip_final.json` `.per_frame[].group`); 65 with an accepted mask, 61 with labels (`txm_unlabelled_area_per_frame.json` `.n_frames`, `.n_labelled`). Each group is one specimen imaged as a series (AM 200 to 1790 cycles, Wrought 0 to 1300, B2 and B3 load/displacement), sharing specimen, mounting, imaging and annotator session, none separable in the data we hold: effective sample size for any specimen-level statement is nearer **four than seventy-one**. At least **six fields of view recur, 13 of the 71 frames** (`_tip`, `_ZOOM`, `_LARGE`, `_4` suffixes; *derived*, `.per_frame[].id`); eight fields, 17 frames, counting two near-duplicate pairs.

Cross-validation groups by **image**, not specimen (`ensemble_vs_hybrid_by_specimen.json`), so held-out folds share specimens — non-AM 44 images, ensemble 0.7850 vs hybrid 0.7696, 5/5 folds, paired t p = 0.049; AM 27 images, 0.7447 vs 0.7483, 1/5 folds, p = 0.321. The panels inherit that structure; the small-component leak guard's minimum p is 0.057 (`txm_small_component_adjudication_v2.json` `.matching.leak_guard_min_p`) — a pass, not a comfortable one. No confidence interval here is a population interval over 316L specimens. One annotator, no frame known to have been labelled twice: no inter- or intra-operator uncertainty can be estimated **[TODO-AUTHOR]**. The labels are both the reference for every agreement metric and the audit's target.

### 6.4 The width instrument is standard metrology, not a new measurement

The transverse-FWHM measurement is not novel and we do not claim it: intensity sampled along normals to a skeletonised centreline is **Orthogonal Profile Extraction**, an established automated crack-width algorithm, and the half-maximum edge is **ISO50**, the default surface-determination rule in X-ray CT dimensional metrology, codified in VDI/VDE 2630; our local-background variant is a routine adaptation (§2.9.1). Ours is only the **use**: a standard rule pointed at the annotations, not the specimen.

Three "shipped width ratio" figures exist because three estimators were built — **0.505** (shared skeleton, 64 frames, `txm_width_ratio_unified.json` `.rows.shipped.median`), 0.558 (`txm_width_clip_final.json` `.width_ratio_after`, 71 frames of which 63 yield a ratio), 0.679 (per-mask self-skeleton, 61 frames, `txm_width_reference.json` `.what_it_says_about_overmarking.shipped_ratio_median`) — over-marking on 0/64, 1/63 and 7/61 frames. This paper uses `txm_width_ratio_unified.json` throughout, the one built so no variant is scored at sample points chosen by its own geometry; the other two appear only here. Three incompatible numbers for one quantity, inside a project about measurement hygiene, is itself a result.

The width is nevertheless physical, not a point-spread floor: both pre-registered thresholds — coefficient of variation < 0.35, Spearman FWHM vs peak contrast < 0.3 — were missed by a wide margin, 0.916 and +0.673 (`txm_width_reference.json` `.is_it_the_instrument`), in the direction that says the variation is real (§4.1).

### 6.5 Residual limitations

- **The straight-line artefact guard is a lower bound.** Its verified removal (§3.6) is component-level, so two further artefacts fused into 440 k and 228 k px crack systems ride through attached to real crack (`txm_small_component_adjudication.json` `.guard.lower_bound`). No within-component straightness test exists.
- **The small-component population is mostly rim, not detections.** Of 12,583 unlabelled components (`txm_unlabelled_area_per_frame.json` `.n_unlab_components`), 284 are ≥ 4000 px; the remaining 12,299 (*derived*) hold 12.2 % of unlabelled area, 6,921 are exactly 1 px, 80.4 % of their area within 25 px of a painted stroke (`txm_small_component_adjudication.json` `.population`, whose own count of 12,303 was measured on the 2026-09-20 mask, so these drift with mask version). At `MIN_BLOB_PX` = 2000 they are unpainted rim of large accepted components, not findings. Only 32 components, 74,404 px, **0.229 % of all accepted area**, are both ≥ 200 px and ≥ 200 px from any stroke (`.tractable_subset`) — the whole of what that arm can be about.
- **No instrument or loading metadata. [TODO-AUTHOR]** Neither repository records beam energy, microscope, detector, exposure, reconstruction, loading protocol, specimen provenance, heat treatment, AM build parameters, geometry or thickness (§2.1); none reconstructable, none invented here.
- **The pixel size is derived, not read from a header. [TODO-AUTHOR]** 29.24 nm/px is derived from operator-supplied mosaic geometry, not instrument metadata (§2.2); `circ2_scale_and_units.json` sustains the value while **retracting its advertised "three independent routes" provenance** and recording two frames that close only at 0.30 overlap, not 0.35. Every physical length here inherits that assumption.
- **Two-dimensional.** Areas on single mosaic frames, not crack volumes or crack-front shapes; deepening and widening both read as more area. No temporal or growth-rate claim is made anywhere: the monotonicity arm was tested on a replicate set, lost to doing nothing, and is withdrawn.

---

## 7. Data and code availability

**Repository.** `https://github.com/jzhang29-max/TXM_Crack_Detection_Pipeline`: tool, shipped models, analysis code, all 71 operator correction masks (`paint/corrections/corrections.npz`, 71 keyed arrays). Code MIT (`LICENSE`); data CC BY 4.0 (`LICENSE-DATA`): 71 raw TXM images (`images/`), four reference ground truths and masks, the labels, the derived result sets. CI (`.github/workflows/linux.yml`, `macos.yml`) runs install, self-tests and a real ensemble prediction each commit, but four of five jobs run the 17-feature model alone, which marks 26.9–83.7 % of a crack-free frame: plumbing, not detection quality. **[TODO-AUTHOR]** deposit the 71 masks and both models with a DOI before submission.

**Generators**, re-runnable from `code/audit/` in the pipeline repository into `out/` of the analysis repository:

| script | artifact | backs |
|---|---|---|
| `measure_unlabelled_area.py` | `txm_unlabelled_area_per_frame.json` | §5.1 census (E1) |
| `measure_width_ratio.py` | `txm_width_ratio_unified.json` | §5.3 width ratios |
| `measure_iou_ceiling.py` | `txm_iou_ceiling_v2.json` (run 2026-09-21, §5.2) | ceiling and model IoU in one pass, replacing the superseded model column of `txm_iou_ceiling.json` |
| `measure_transverse_fwhm.py` | `txm_transverse_fwhm.json` (**never emitted**, below) | §6.4, the pre-registered "is the width physical" test |
| `build_small_component_crops.py` | `txm_small_component_cropkey_v2.json` + crops | blind set for small indications, matched on size **and** specimen |
| `score_blind_panel.py` | `txm_small_component_adjudication_v2.json` | unblinding, instrument validation, group-confound check |
| `heldout_crackfree_fp.py` | `txm_heldout_crackfree_fp.json` (v1, superseded by `heldout_crackfree_fp_v2.py` → `txm_heldout_crackfree_fp_v2.json`, Addendum) | the uncontaminated false-positive figure |

**Mask.** `effective_mask` (`app/core/pipeline.py:885`); stages `clip_to_measured_width` (`:675`), `drop_straight_lines` (`:809`), `local_background`/`_transverse_fwhm` (`:615`, `:643`), `MIN_BLOB_PX` = 2000 (`:470`); shipped `effective_mask(corrections='none', tight=True)` with `WIDTH_CLIP=True` and `drop_straight_lines` active (`.mask_version`). Every number in this paper is on it unless stated.

**Panel votes are not reproducible here:** they need a blind multi-agent panel over the crops, are an artifact of one model and harness, and **ship as data, not regenerated** (`txm_small_component_adjudication_v2.json` `.generators.note`; `code/audit/README.md`). Preserved: crop keys with frame id and centre (`txm_unlabelled_adjudication_cropkey.json`, 88 fields; `txm_small_component_cropkey_v2.json`, 94 fields); crop windows **700 × 700 px** and **400 × 400 px**, the latter sized so a 200 px clearance puts no painted crack in frame (`build_small_component_crops.py:35`); the seed (`:61`, `default_rng(20260921)`); leak-guard statistics, group-matching record, deterministic scorer. Crops are byte-reproducible from `images/` and the keys; every vote, its panel and each panel's free-text evidence ship in the `per_field` arrays of both adjudication artifacts.

**Known gaps in the provenance chain.** Four load-bearing artifacts carry no `generator` key: `txm_iou_ceiling.json`, `txm_width_reference.json`, `txm_unlabelled_adjudication.json`, `txm_width_clip_final.json`. Three have no driver in `code/audit/`; the fourth's, `measure_iou_ceiling.py`, emits a new file rather than reproducing the old, so `txm_iou_ceiling.json` remains an unreproduced output with a superseded model column (§5.2). `code/audit/README.md` also lists `measure_transverse_fwhm.py` as emitting `out/txm_transverse_fwhm.json`; **that file does not exist**, and the §6.4 numbers are read from `txm_width_reference.json`. **[TODO-AUTHOR]** re-emit these under named generators, or state in the final version that they are recorded outputs of scripts lost to a scratch directory — which is what they are.

## Figures and tables

All figures regenerate from the artifacts by checked-in code.

| | content | source |
|---|---|---|
| Figure 1 | all 71 frames with identified crack, contact sheet | `results/all_overlays_contact_sheet.png`, `code/make_deployed_overlays.py` |
| Figure 2 | blind adjudication: the five classes and the panels' verdicts | `results/blind_adjudication.png` |
| Figure 3 | the four frames where over-marking was real, before and after width clipping | `results/width_clip_before_after.png` |
| Figure 4 | small isolated indications against size- and specimen-matched controls | `results/small_component_adjudication.png` |

**[TODO-AUTHOR]** Figures are referenced by name, not number, in the body text; renumber to the journal's convention at submission, and supply a micrograph of each specimen group if the venue expects one.

## References

Verified against the primary sources 2026-09-21/22.

1. **Shit, S., Paetzold, J. C., Sekuboyina, A., et al.** clDice — a novel topology-preserving loss function for tubular structure segmentation. *CVPR*, 2021, pp. 16560–16569. arXiv:2003.07311.
2. **Benz, C., Rodehorst, V.** OmniCrack30k: a benchmark for crack segmentation and the reasonable effectiveness of transfer learning. *CVPRW*, June 2024. — 30k samples, ~9 billion pixels, 20+ datasets; its best nnU-Net reaches mean clIoU4px 64%, the comparator our centreline figures are read against.
3. **Kirillov, A., Mintun, E., Ravi, N., et al.** Segment Anything. *ICCV*, 2023. arXiv:2304.02643. — the frozen ViT-H encoder supplying the 256 embedding channels.
4. **VDI/VDE 2630 Blatt 1.1.** Computed tomography in dimensional measurement: fundamentals and definitions. Verein Deutscher Ingenieure / Verband der Elektrotechnik, 2009. — codifies the framework the ISO50 rule sits in; see also Blatt 1.2.
5. **Lifton, J. J., Carmignato, S.** Simulating and estimating the measurement uncertainty due to the ISO50 surface determination method for dimensional computed tomography. *Precision Engineering*, 2019. doi:10.1016/j.precisioneng.2019.05.006.
6. **US Department of Defense.** MIL-HDBK-1823A: Nondestructive evaluation system reliability assessment. 7 April 2009. — the ≤1% false-call yardstick the crack-free gate is read against in §3 and the Addendum.

**[TODO-AUTHOR]** Two citation slots this repository cannot supply: a primary reference for Orthogonal Profile Extraction as a named crack-width algorithm (pick one from your own field's literature), and any prior TXM/XCT study of the same alloy or loading regime for the Introduction.

## Addendum: the crack-free gate is contaminated

**The crack-free false-positive figure is inflated by training on those specimens.** The six contribute 0 crack pixels and 95,351,647 not-crack pixels -- 4.3% of training rows after per-image capping -- and the gate measures false positives on those same frames. Held out properly (5 seeds per arm, both branches refitted at a 400,000-row budget, scored through the shipped `CrackModel` ensemble path):

| arm | mean predicted area on crack-free material |
|---|---|
| trained WITH their labels | 0.289% +/- 0.032 |
| **trained WITHOUT them** | **1.581% +/- 0.337** |
| inflation | **5.5x**, all 5/5 pairs same direction, Wilcoxon p = 0.0625 |

p = 0.0625 is the **smallest value a two-sided signed-rank test can return at five pairs** (2/2^5), floored by sample size, not by a weak effect: every pair separated, and the arms do not overlap.

The deployed model reads 0.174% on the same frames, so the gate's headline is a **lower bound**, and the corrected figure crosses a threshold the contaminated one did not: the repo cites MIL-HDBK-1823A's 1% yardstick, and 0.174% sits comfortably under it; held out, **1.58% is above it**. That moves the detector from passing that yardstick to failing it on the only material where a false positive is certain.

Two things it is NOT. Not the shipped weights' false-positive rate: these are retrained models at a reduced row budget, so the comparison bounds the contamination rather than replacing the gate's number. Not a verdict on the ensemble: both arms use it. Generator `code/audit/heldout_crackfree_fp_v2.py`, artifact `crack-evolution-5d/out/txm_heldout_crackfree_fp_v2.json`. An earlier attempt (`heldout_crackfree_fp.py`) fitted a 17-feature-only model, landed 48x off the deployed figure and got the sign backwards; it is kept with that failure recorded.