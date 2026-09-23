<!--
REWRITTEN 2026-09-21. The previous draft (2026-08-26, kept at
docs/PAPER_DRAFT_2026-08-26_superseded.md) predated the width, unlabelled-area, artefact-class
and confound work, and an audit found only two of its four Results subsections survived. Do not
mine it for numbers.

HOW THIS WAS PRODUCED, because it matters for how much to trust it. A claim ledger was built by
opening every txm_* artifact and recording each number with its JSON key path. Each section was
then drafted against that ledger and independently re-verified by a second pass that reopened the
artifacts and checked every figure. The verification pass found and fixed, among others: a p-value
of 0.018 that appears in no artifact (the real one is 0.013, and is one-sided); a paired test
reported with the wrong n; a contamination figure cited to an artifact that does not contain it;
an over-reaching "the unlabelled quarter is crack" that ignored the 21.7% scope and the one
genuine false positive; and an omitted adverse fact (32.4% of the unlabelled area is stroke rim
within 25 px of a painted stroke). The per-section verification log is preserved at
out/paper_verification_log.md in the analysis repo.

THIS IS NOT SUBMITTABLE AS IT STANDS, for three reasons, in order of how much work they are:

1. LENGTH. 23,559 words. A materials-characterisation research article is 6,000-8,000. This is a
   complete and checked draft, not a manuscript; it needs roughly a 3x editorial cut. Nothing
   here should be cut for being inconvenient -- the limitations and the reversals are the
   contribution.

2. [TODO-AUTHOR] MARKERS. Acquisition metadata -- beam energy, reconstruction method, voxel size
   provenance, loading protocol, specimen history -- is in neither repository. It cannot be
   inferred from the data and must not be invented.

3. HUMAN ADJUDICATION. The central claim rests on blind panels of language-model agents. Their
   validation shows they agree with the annotator's own strokes, which is not the same as
   competence at 316L TXM. Two blinded human readers through the preserved protocol
   (code/audit/build_small_component_crops.py, the crop key, the fixed seed) would settle it.
   Until then the limitation stands in the paper's own voice and should not be softened.
-->

# A quarter of an operator-trained crack detector's output has never been adjudicated, and it reads as crack: labelled-pixel metrics on broad-brush TXM annotations of 316L steel

*Artifact names in backticks refer to files in the analysis repository's `out/` directory; where a number exists only in pipeline-repository code, the file and its role are named instead. Every quantitative claim below names its source.*

## Abstract

Accuracy in crack segmentation is reported almost entirely inside the annotated domain: IoU, clDice, precision and recall are all computed where an operator painted something. We audit an interactive detector on 71 transmission X-ray microscopy mosaic frames of fatigue-cracked 316L steel — four specimen groups, broad-brush hand annotation — and measure what that convention cannot see. Area-weighted over the 65 frames the census covers, 24.7% of all accepted mask area was never labelled either way, 51.1% on the additively manufactured group and 4.5% on the best-annotated one (`txm_unlabelled_area_per_frame.json`); every metric named above is blind to this area by construction. Blind three-panel adjudication of the large-component part of it — components of 4,000 px or more, 21.7% of all accepted area — returns crack: 25 of 26 regions, indistinguishable from painted crack (Mann–Whitney on the vote score, p = 0.31) and far from plain matrix (p = 4.4e-08), with the single rejection a specimen free surface and so a genuine false positive (`txm_unlabelled_adjudication.json`). For small isolated indications with controls matched on size and specimen, the test arm and the painted-crack positive control are identical at 13/32 (two-sided Fisher p = 1.000) and both exceed matrix at 3/26 (Fisher p = 0.013, computed one-sided) (`txm_small_component_adjudication_v2.json`). The default metric cannot flag any of this: against a median label width of 73 px, a physically correct 3 px crack scores IoU 0.0498 (`txm_iou_ceiling.json`), so IoU here ranks imitation of a paintbrush. Measured against the image instead of the labels, detector and annotator widths are indistinguishable — 0.506 against 0.529 of the transverse feature on the 61 frames carrying labels, paired Wilcoxon p = 0.061 — and both under-mark (`txm_width_ratio_unified.json`). Our adjudication panels are language-model agents, not human experts: that is this audit's principal limitation, and we release the protocol so human readers can repeat it.

**Keywords:** crack segmentation, transmission X-ray microscopy, 316L stainless steel, annotation quality, segmentation metrics, blinded adjudication, audit

## 1. Introduction

A crack segmentation is scored against what a person painted. Intersection-over-union, Dice, clDice, precision and recall are all ratios whose terms are counted over annotated pixels, because outside the annotation there is nothing to agree or disagree with. Where the operator painted nothing, a predicted pixel is neither a true positive nor a false positive: it is dropped from every term and reported by nothing. This is usually treated as harmless, on the assumption that what is unannotated is empty. In sparse, thin, low-contrast tomographic imaging that assumption is not safe, and this paper measures how unsafe it is.

Our corpus is 71 transmission X-ray microscopy mosaic frames of fatigue-cracked 316L stainless steel across four specimen groups — additively manufactured (AM / HC_316L), two bending series (B2, B3) and wrought — annotated by one operator with a broad brush during interactive correction; 61 frames carry labels (`txm_width_clip_final.json`). The detector is not weak by the conventions of the field: on the shipped mask it scores clDice 0.823, topological precision 0.970, topological sensitivity 0.761 and IoU 0.675 over those 61 frames (`txm_width_clip_final.json`). Nothing here is a new architecture. The contribution is honest measurement of what a detector trained this way can be said to do, and of how much of that is decided by the annotation rather than by the specimen.

**A quarter of the mask has never been adjudicated.** Area-weighted over the 65 frames the census covers, 24.7% of everything the detector accepted falls where the operator painted neither crack nor not-crack — 8,016,109 px of 32,504,070 (`txm_unlabelled_area_per_frame.json`). The census omits six of the 71 frames, which its generator skips for having no mask or no correction map (`code/audit/measure_unlabelled_area.py`). The unlabelled share is not evenly spread: 51.1% on AM (n = 24), 36.8% on B2 (n = 16), 10.7% on wrought (n = 12), 4.5% on B3 (n = 13). On the worst frame 97.3% of the accepted area is unadjudicated; on the frame with the largest absolute hole, 17.3% of the entire field is mask nobody checked. Nor is it dust: 87.9% of it sits in components of 4,000 px or more. It is, however, not all far from the strokes — 32.4% of it lies within 25 px of a painted stroke and is to that extent stroke rim rather than independent indication (`txm_unlabelled_area_per_frame.json`). Every metric in the previous paragraph is blind to all of it, by construction.

**Adjudicated, it reads as crack.** We put that area through a blinded three-panel protocol against four control classes: painted crack (positive), plain matrix (floor), painted not-crack where the model disagreed with the operator, and accepted area on specimens asserted crack-free — the last at n = 2 and excluded as underpowered (`txm_unlabelled_adjudication.json`). Large regions: 25 of 26 called crack, indistinguishable from painted crack (Mann–Whitney on the vote score, p = 0.31) and separated from plain matrix (p = 4.4e-08), the instrument validated on its own controls at 24/26 against 4/26 (Fisher p = 1.2e-08). The one rejection was unanimous and reasoned: a specimen free surface with Fresnel fringes on `wrought_316L_fatigue_1280_cycles`, a real false positive, one in 26. Small isolated indications, with controls matched on both size and specimen: 13/32 in the test arm and 13/32 in the painted-crack control, two-sided Fisher p = 1.000, against 3/26 on matrix (Fisher p = 0.013; the artifact computes this test one-sided, `alternative="greater"` in `code/audit/score_blind_panel.py`, so it is the more permissive of the two conventions) (`txm_small_component_adjudication_v2.json`). Two limits bound what this supports. The sample was drawn from components of 4,000 px or more, so the large-region result covers 21.7% of accepted area, not the whole 24.7% (`txm_unlabelled_adjudication.json`); and the small-component class is 0.23% of accepted area, which is its entire exposure even if every member were a false positive (`txm_small_component_adjudication.json`). Within that scope, the adjudicated unlabelled area is crack the annotator did not reach rather than false positive.

**Why the default metric cannot see it.** Skeletonising the operator's own painted crack and re-dilating it to a physical width scores the labels against themselves: a correct 3 px crack reaches a median IoU of 0.0498 over 61 frames, 5 px reaches 0.0822 and 21 px reaches 0.3140, against a median label width of 73 px (`txm_iou_ceiling.json`, `txm_iou_ceiling_per_frame.json`). The 3 px ceiling runs from 0.0291 on B3 to 0.0732 on AM, tracking label width (126 px to 36 px). An IoU above that ceiling is not bought by better localisation; it is bought by being wider than a crack. The ceiling is a property of the annotations, not of any method, and no detector can raise it.[^1]

**A width instrument pointed at the annotation.** Sampling transverse intensity profiles and taking a half-maximum edge is standard — Orthogonal Profile Extraction with the ISO50 criterion of X-ray CT dimensional metrology (VDI/VDE 2630) — and we claim no novelty for the measurement. What is ours is aiming it at the labels. On one estimator, one frame set and shared sample points, the shipped mask covers a median 0.505 of the transverse feature over all 64 frames, and on the 61 frames that carry labels it covers 0.506 against the human brush's 0.529 (paired Wilcoxon p = 0.061, indistinguishable); both fall short of the dark feature (p = 3.5e-12 and p = 1.1e-06), and the shipped mask is wider than the feature on 0 of 64 frames against the brush's 11 of 61 (`txm_width_ratio_unified.json`).[^2] A separate census isolates a straight-line artefact class — median wander about a line fit 0.47 px for straight components against 5.25 px for curved ones, one of the six on a specimen asserted crack-free — and the guard built on it fires on 4 of the 71 frames, removes 15,241 px, none of them painted crack, and leaves clDice unchanged (`txm_small_component_adjudication.json`; that artifact's own adjudication arm is superseded by `txm_small_component_adjudication_v2.json` for a group-mix confound, but the straight-line census exists only in the earlier run and is unaffected by it).

**What we do not claim, and what is weak.** We make no corpus-wide over-marking claim; the detector under-marks, and is indistinguishable from the annotator. We make no crack-growth claim; that test lost to doing nothing. Our adjudication panels are language-model agents, not human experts: their validation shows they agree with the same annotator's strokes, which is not competence at 316L TXM, and their votes are not regenerable from the released code (`code/audit/README.md`). The small-component arm is further limited by a panel sensitivity of 13/32 on known crack of that size, so it establishes that the test indications are indistinguishable from known crack, not that they are crack. The false-positive rate on crack-free specimens is contaminated: the six confirmed crack-free specimens contribute 0 crack pixels and 95,351,647 not-crack pixels, 25.4% of every background label the model trains on, so that figure is partly memorisation — a number recorded in the header of the refit script `code/audit/heldout_crackfree_fp.py`, not in an `out/` artifact. A held-out refit with those frames removed is running; `out/txm_heldout_crackfree_fp.json` does not yet exist, no held-out false-positive figure is quoted here, and the deployed figure should be read as an upper bound until one lands. [TODO-AUTHOR: insert the held-out crack-free false-positive rate, and the deployed figure beside it, when the refit completes.] Beam energy, reconstruction, loading protocol and specimen provenance are recorded in neither repository and are marked [TODO-AUTHOR] throughout.

[^1]: We deliberately report no ratio of detector IoU to this ceiling. `txm_iou_ceiling.json` carries such a ratio — `deployed_model_IoU_same_frames` 0.5047 and `model_over_ceiling` 10.1 — and it must not be quoted: its generator records that the model column is byte-identical to a run made before `clip_to_measured_width` shipped, so it scores a superseded mask (`code/audit/measure_iou_ceiling.py`). The shipped 0.675 comes from a different run and a different aggregation (`txm_width_clip_final.json`), and the two are not interchangeable. The ceiling itself is label-only and is unaffected by either.

[^2]: Three width estimators exist in our artifacts and disagree: 0.505 on shared sample points over 64 frames (`txm_width_ratio_unified.json`), 0.558 over 71 frames (`txm_width_clip_final.json`) and 0.679 on per-mask self-skeletons over 61 frames (`txm_width_reference.json`), with 0, 1 and 7 over-marking frames respectively. This paper uses the first throughout, because it is the only one that scores every mask at the same points; it is a comparison between masks and not an absolute width estimate.

## 2. Methods

**Conventions used throughout.** Every quantitative statement below names the file it is read from. Names of the form `txm_*.json` / `cldice_*.json` are artifacts in the analysis repository's `out/` directory; paths of the form `app/…`, `code/…`, `docs/…`, `models/…` and `app_data/…` are in the pipeline repository. A value marked *derived* was recomputed for this paper from an artifact's per-frame records, and the derivation is given. Where a number appears in a repository document but in no artifact, we say so and do not use it. Where two artifacts disagree, we name one instrument, quote it, and record the disagreement rather than averaging or silently choosing.

**The mask this paper audits.** Unless stated otherwise, every measurement is made on the *shipped* mask: `effective_mask(corrections='none', tight=True)` with `WIDTH_CLIP = True` and `drop_straight_lines` active. That exact string is recorded in every record of `txm_unlabelled_area_per_frame.json` (`.mask_version`). `corrections='none'` means the model's own output with no operator strokes pasted back in; auditing the `paste` or `gate` variants would score the annotator's strokes as part of the detector's answer.

### 2.1 Specimens, loading and acquisition — [TODO-AUTHOR]

None of the acquisition or specimen metadata is recorded in either repository and none of it can be reconstructed from the data. The following must be supplied by the authors and must not be inferred:

- **Instrument and acquisition:** microscope, beam energy, detector, objective/zone plate, exposure, number of projections, tomographic reconstruction method and any ring/beam-hardening correction. [TODO-AUTHOR]
- **Loading protocol:** rig, load levels, and cycle counting. Filenames carry tokens such as `_1_lbf`, `_3_18lbf` and `_1250_cycles`, which imply pound-force load steps and a fatigue cycle count, but no protocol is recorded anywhere in the repositories. Do not state the implied units without confirming them. [TODO-AUTHOR]
- **Specimen provenance:** composition and heat treatment of the 316L stock; for the group referred to here as AM, the build process and parameters. The group key `AM` corresponds to filenames beginning `HC_316L_`; what `HC` denotes is not recorded. [TODO-AUTHOR]
- **Geometry:** specimen shape and through-thickness. A 22 µm through-thickness is used in a separate 3D arm of this project; it is unverified here and is not used. [TODO-AUTHOR]
- **Annotator:** identity, training, materials background, time per frame, and whether any frame was annotated more than once. A single annotator produced all labels; nothing else about the annotation session is recorded. [TODO-AUTHOR]
- **Ethics, data availability and a DOI deposit** for the 71 correction masks. [TODO-AUTHOR]

### 2.2 Spatial scale

The pixel size is **29.24 nm/px**, and it is a *geometric derivation, not an instrument header value*. It follows from operator-supplied mosaic geometry — 9 × 5 tiles of a 30 µm viewing window at 0.35 overlap, giving 186.0 × 108.0 µm over a 6367 × 3691 px mosaic, isotropic to 0.2% (`crack-depth-3d/README.md:69–70`, constant `core.TXM_UM_PER_PX`). The 30 µm window and the 0.35 overlap are operator-supplied inputs to that derivation and must be confirmed [TODO-AUTHOR]; if either is wrong, every length in nanometres in this paper scales with it, while every ratio (mask width against feature width, tortuosity, area fraction) is unaffected.

One inconsistency is carried openly: the width artifact converts pixels to nanometres with a rounded **29.0 nm/px** (`code/audit/measure_transverse_fwhm.py:37`, `NM_PER_PX = 29.0`), so the median transverse FWHM of 45.0 px is recorded as 1,305 nm (`txm_width_reference.json .is_it_the_instrument.fwhm_median_px`, `.fwhm_median_nm`). At 29.24 nm/px the same 45.0 px is 1,316 nm. The 0.8% difference is below the uncertainty of the geometric derivation itself, but the two constants should be unified before submission [TODO-AUTHOR].

### 2.3 Image corpus and preprocessing

Seventy-one mosaic frames of four 316L specimen groups, float32, 2.85–32.12 megapixels (median 10.39 MP, 849.59 MP in total; *derived* from `app_data/images/*/meta.json`, field `megapixels`).

| group | frames | frames with painted crack | painted crack px | painted not-crack px | frame size (MP) |
|---|---|---|---|---|---|
| AM (`HC_316L_*`) | 27 | 25 | 4,628,821 | 165,191,178 | 5.15–32.12 |
| B2 | 17 | 14 | 9,321,938 | 67,038,533 | 2.85–23.50 |
| B3 | 13 | 10 | 7,616,710 | 48,700,962 | 3.86–23.46 |
| Wrought | 14 | 12 | 9,750,198 | 94,701,499 | 8.83–22.20 |
| **total** | **71** | **61** | **31,317,667** | **375,632,172** | |

*Derived* from `app_data/images/*/meta.json` (`corr_counts`); the crack total is independently recorded as `correction_crack_px` = 31,317,667 in the deployment gate record `app_data/models/retrain_history.json` (stamp `20260824_225236`) and in `models/hybrid_v5_20260824.joblib`.

**Group assignment is by filename, and this is a correction.** Group is assigned by the tokens `_b2_`, `b3`, `hc_316l`, `wrought`. Several audit generators apply that rule to the *stored image ID*, which carries an eight-character hash suffix, and one frame — `HC_316L_fatigue_1600_cycles`, ID suffix `__4cb30dc6` — contains the substring `b3` in its hash and is therefore filed as B3 in `txm_unlabelled_area_per_frame.json`, `txm_width_ratio_unified.json` and `txm_iou_ceiling_v2.json`. The older `txm_iou_ceiling_per_frame.json` and `cldice_centreline_per_frame.json` group by filename and are unaffected (their AM *n* = 25 matches the table above). Per-group figures in this paper are recomputed from the per-frame records with filename grouping. The corpus totals are unchanged; the per-group figures that move are:

| artifact | AM | B3 |
|---|---|---|
| `txm_unlabelled_area_per_frame.json`, *n* | 24 → **25** | 13 → **12** |
| — unlabelled share | 51.06% → **50.16%** | 4.49% → **2.83%** |
| `txm_iou_ceiling_v2.json`, *n* | 24 → **25** | 11 → **10** |
| — shipped IoU | 0.5310 → **0.5468** | 0.8933 → **0.9063** |
| `txm_width_ratio_unified.json`, *n* | 24 → **25** | 13 → **12** |

(all *derived* by regrouping the named artifacts' `.per_frame` records.) B2 and Wrought are unaffected in all three.

**Preprocessing, and the fact that the human and the model see different images.** Frames are de-stitched (per-pixel correction of the mosaic border strip plus a validated 2D FFT notch at the detected tile pitch, `code/destitch.py`) and pseudo-flat-fielded by division with an anisotropic Gaussian blur of the frame itself, computed by normalised convolution so that no-data regions outside the ragged mosaic outline do not drag the background estimate (σ_y = 16, σ_x = 22; `code/flatfield.py:57`). **The flat-fielded image is the display path only.** The model reads the raw frame (`img.npy`); the annotator sees `display.npy` stretched between the 1st and 99th percentile of the specimen support (`app/core/pipeline.py: display_limits`, `:131`, with off-specimen background excluded by an Otsu-plus-topology support mask, `specimen_support`, `:106`). The docstring of `display_limits` records a cost of 0.169 IoU for using the flat-fielded image as model input; that figure has no supporting artifact in `out/` and is not used as evidence here [TODO-AUTHOR: re-measure or drop].

### 2.4 Annotation

Labels are a three-valued correction map per frame: 1 = crack, 2 = not crack, 0 = never labelled either way. The 0 class is not "background" — it is the absence of an assertion, and it is the subject of §2.7.

The annotation is deliberately broad-brush: a stroke marks *that there is crack here*, not *where its edges are*.

**Two artifacts measure the painted-label width and they disagree by a factor of 1.8.** They are reported side by side and are never averaged or cross-quoted:

| instrument | median over 61 labelled frames | AM | B2 | B3 | Wrought |
|---|---|---|---|---|---|
| **primary — `txm_iou_ceiling_v2.json`** (2026-09-21): 2 × median Euclidean distance transform on the label skeleton, definition in the preserved generator `code/audit/measure_iou_ceiling.py:86` | **41.2 px** (`.median_label_width_px` = 41.1825) | 18.97 | 54.58 | 77.24 | 73.83 |
| secondary — `txm_iou_ceiling_per_frame.json` / `cldice_centreline_per_frame.json` (2026-09-19, byte-identical `label_width_px` columns; no preserved generator, definition not recorded) | 73.4 px (*derived*: median of `label_width_px`; the summary rounds this to 73.0 at `txm_iou_ceiling.json .headline.median_label_width_px`) | 36 | 88 | 126 | 97 |

The primary instrument is named primary only because its definition is recorded in a generator that still exists; we do not claim it is the more accurate of the two. At 29.24 nm/px the two medians are **1.20 µm** and 2.15 µm. The per-group columns of the primary row are *derived* by regrouping `.per_frame` with filename grouping (§2.3); the secondary row is read from `txm_iou_ceiling.json .per_specimen.<G>.label_width_px`. [TODO-AUTHOR: re-emit the 2026-09-19 label-width column with its definition, or retire it.]

Either way the stroke is one to two orders of magnitude wider than a crack, and every pixel-overlap metric in this paper is therefore an agreement with a brush of that width, which is the point of §2.8.

**Six specimens are asserted crack-free by the annotator** and serve as the project's only over-prediction check that needs no pixel-level ground truth: `b3_amb`, `B2_amb_mosaic_2`, `B2_2_1_lbf`, `B2_2_9_lbf`, `b3_3_18lbf`, `wrought_316L_fatigue_0_cycles` (`CLEAN_SPECIMENS`, `app/core/pipeline.py:102`). Between them they carry **0 painted crack pixels and 95,351,647 painted not-crack pixels**, which is **25.38%** of all labelled not-crack pixels in the corpus (the 25.4% figure is stated in the docstring of `code/audit/heldout_crackfree_fp.py`; verified here as 95,351,647 / 375,632,172 = 25.384% *derived* from `app_data/images/*/meta.json`). The consequence for the false-positive figure is in §2.5.4.

One inconsistency in the crack-free bookkeeping is noted rather than harmonised, and it propagates into a reported number (§2.5.4): the blind-crop generator `code/audit/build_small_component_crops.py:36` treats **ten** frames as crack-free (`CRACKFREE`), four more than the deployment gate's six, and one of the four CFREE control fields in the small-component run (`site_033`) is drawn from `b3_3_0lbf_268_13um`, a frame the gate does not treat as crack-free (`txm_small_component_cropkey_v2.json`). The CFREE arm has *n* = 4 and is underpowered in any case (§2.11).

### 2.5 Detector

The detector is not a contribution of this paper; it is the object under audit. It is an instance of the established pattern — frozen foundation-model encoder, light classifier, operator correction, retrain — and is described here only in the detail needed to reproduce what was measured.

#### 2.5.1 Features (273 per pixel)

Seventeen isotropic hand-crafted features (`code/txm_features.py`): raw normalised intensity; Gaussian-smoothed intensity at σ = 2, 4, 8, 16, 32, 64 px; Gaussian gradient magnitude at σ = 1, 2, 4, 8 px; signed Laplacian of Gaussian at σ = 1, 2, 4, 8 px; local standard deviation at σ = 2, 8 px.

Two hundred and fifty-six channels from the Segment Anything ViT-H image encoder (`facebook/sam-vit-huge`), used frozen. Tiles are 1024 px at a stride of **896 px**, giving a 64 × 64 × 256 embedding grid per tile (`TILE`, `TILE_STRIDE`, `EMB_STRIDE` in `app/core/model.py:58–92`); per-pixel values are read by bilinear interpolation and overlapping tiles are combined with a Hann window, because abutting tiles produce a step in the embedding field at the boundary (`docs/TILE_SEAMS.md`).

#### 2.5.2 Classifier and training set

Two multilayer perceptrons behind standard scalers, averaged in probability:

| member | input | hidden layers | solver settings |
|---|---|---|---|
| 17-feature MLP | 17 | (64, 32) | `max_iter=400`, `random_state=0`, converged at 204 iterations |
| SAM + 17 hybrid | 273 | (128, 64) | `max_iter=400`, `random_state=0`, converged at 64 iterations |

Verified by loading `models/f17_v5_20260824.joblib` and `models/hybrid_v5_20260824.joblib`. The deployed prediction is the mean of the two probabilities (`CrackModel.predict`, `app/core/model.py`).

Training rows come from the annotator's corrections only; no external label is used in training or in the gate. The deployed fit (recipe `thincore_v5`, stamp `20260824_225236`) used **3,530,484 rows at 273 features, crack fraction 0.4915, over all 71 labelled images**, with a per-image cap of 30,000 crack rows (`per_img_cap`, `app/core/pipeline.py:1518`) and a per-image background cap `neg_cap` = 25,536 (model metadata in `models/hybrid_v5_20260824.joblib`: `n_px`, `crack_fraction`, `neg_cap`, `trained_on_images`).

**Training labels are narrowed before sampling.** Each painted crack region is narrowed to its dark core with `tighten_to_image`, and the discarded ring joins the *negative* pool at its true area weight rather than being dropped (`gather_training_data`, `app/core/pipeline.py:1496`). This is the `thincore_v5` recipe and it is the reason the detector's masks are narrower than the strokes that trained them. The stroke- and core-width measurements that motivated the change (a median painted half-width of 26.93 px against a 3.16 px dark core, predicted width correlating 0.810 with label width and 0.304 with crack width) exist only as a source comment at `app/core/pipeline.py:1537–1552` and in `docs/THIN_LABELS.md`; no artifact in `out/` supports them and they are not used as evidence here [TODO-AUTHOR: re-measure into an artifact or drop].

#### 2.5.3 Operating point and the three narrowing steps

Probability threshold **p > 0.60** (`DEFAULT_THRESHOLD`, `app/core/pipeline.py:90`), then connected components below **2,000 px** are removed (`MIN_BLOB_PX`, `:470`) and not-crack islands below 1,024 px inside crack are filled (`FILL_HOLES_MAX_PX`, `:527`). Three narrowing steps then run in a fixed order, and the order is load-bearing:

1. **`tighten_to_image`** (`:1021`) — keep pixels darker than the local mean over a 301 px window (`TIGHTEN_WINDOW`, `:541`), declining to narrow the frame at all if fewer than 60% of the darkest fifth of the accepted region survive (`TIGHTEN_MIN_CORE` = 0.60, `:550`). Its own docstring carries a 2026-09-19 correction to the table above it: re-measured over all 71 frames at the deployed operating point, the step drops about **6% of area, not the ~21% the table's first two rows imply**, and moves the corpus median half-width 49.0 → 44.4 px, **a factor of 1.10 rather than the 12× the third column suggests** — because `uniform_filter` averages over the mask as well as around it, so a 30–50 px band inside a 301 px window passes its own test.
2. **`clip_to_measured_width`** (`:675`) — clip the mask to the width the image shows at each location, never widening it. Background is estimated from non-mask pixels by normalised convolution over windows of 301/901/2401 px until more than 2% of the window is non-mask; contrast is background minus image; up to 4,000 skeleton probes per frame are measured along 12 directions at a half-length of 200 px, with ±3 px recentring and a peak that must clear 2σ; the radius field is carried to the rest of the skeleton by inverse-distance weighting over the 12 nearest probes (`WIDTH_CLIP_*` constants, `:595–612`). With fewer than 8 measurable profiles it **declines and returns the mask unchanged**. Its safety is monotonicity, not inertness: `docs/WIDTH_REFERENCE.md:150–157` records that of the 53 frames with a pre-clip ratio at or below 1.0, 50 lose area (median 4.6%, max 14.1%) and 48 end further from a ratio of 1.0 than they started.
3. **`drop_straight_lines`** (`:809`) — remove standalone components of ≥200 px with bounding-box aspect ≥6 whose centreline wanders less than 1.0 px about a straight-line fit; any component containing a painted pixel is exempt (`STRAIGHT_*`, `:786–788`; §2.10).

The "keep the mask if narrowing emptied it" fallback covers steps 1 and 2 only, because both rest on a measurement that can fail. Step 3 runs after that decision and is final: an empty mask on a crack-free specimen is a correct answer, not a failure to be caught (`_narrow_to_image`, `:850`). That empty-result fallback existed twice in the code, and the redundancy is one of the three checks that reported success while measuring nothing (§2.13).

Declining rather than guessing matters most on AM. Painted AM crack has essentially no intensity contrast against painted not-crack: median Cohen *d* = **−0.31** on AM against +1.76 (B2), +3.74 (B3) and +1.38 (Wrought), with the painted crack *brighter* than the painted not-crack on 17 of 25 AM frames (`am_label_separability.json .per_specimen.<G>.d_median`, `.frames_d_negative`). Five other places in the project quote a different set of separabilities (+0.09 / +1.10 / +2.91 / +1.57 — `docs/START_HERE.md:103`, `docs/WIDTH_REFERENCE.md:155`, `docs/UNLABELLED_AREA.md:156`, `app/core/pipeline.py:680`, `code/audit/build_small_component_crops.py:7`) that no artifact in `out/` supports; we use the artifact values. The same artifact's prose states AM is "27 of 71 frames = 38%" while its own per-specimen table has *n* = 25; the corpus has 27 AM frames, 25 of them with painted crack (§2.3), so 38% and 35.2% are both defensible for different quantities and the artifact does not say which it means [TODO-AUTHOR: reconcile].

#### 2.5.4 Deployment gate, the negative control, and its contamination

A retrained model becomes current only if it passes both axes (`app/core/pipeline.py`): held-out agreement under 5-fold cross-validation **grouped by whole image**, against an absolute floor of 0.60 IoU (`MIN_ABS_IOU`, `:91`); and predicted area on the six crack-free specimens, against a tolerance of 0.5 percentage points (`FP_TOL` = 0.005, `:96`). The deployed model's gate record is mean IoU **0.8113**, sd 0.0229, worst fold 0.7777, precision 0.9355, recall 0.8597, crack-free predicted area **0.1744%** (`app_data/models/retrain_history.json`, stamp `20260824_225236`: `.heldout.mean_iou`, `.std_iou`, `.min_iou`, `.mean_precision`, `.mean_recall`, `.candidate_clean_fp` = 0.001744, computed over `CLEAN_SPECIMENS` by `_score_clean`, `:173`).

Two things must be said about that 0.1744% and they are said here rather than in the discussion.

**It is not the figure this paper reports.** The repositories contain four mutually incompatible crack-free false-positive figures, differing by about 7× and measured at different stages of the mask: 0.1744% raw at the gate (`retrain_history.json`); 0.0230% → 0.0209% (`app/core/pipeline.py:1072`); 0.0230% → 0.0197% and 0.0230% → 0.0192% (`docs/OVERMARKING.md:181`, `:256`); and 0.025278% → **0.023588%** on the shipped mask (`txm_small_component_adjudication.json .guard.verified.crackfree_fp_before` / `_after`). Only the last is measured on the mask this paper describes, so the paper carries **one** figure, 0.0236% on the shipped mask, and names its artifact every time.

**Its denominator is ten frames, not six, and this is a correction.** The generator that produced `.guard.verified` is in neither repository — `grep` for `crackfree_fp` over both repos returns only the artifact itself — so the crack-free set it averages over is not recorded. It can be recovered arithmetically. The guard fired on four frames and exactly one of them is crack-free under any definition (`b3_amb`, predicted area fraction 0.00016907 → 0.0, `.guard.verified.detail[1]`), and the reported change is 0.025278% − 0.023588% = 0.0016907 pp = 100 × 0.00016907 / **10** exactly. The figure is therefore a mean in percent over the ten-frame `CRACKFREE` list of `code/audit/build_small_component_crops.py:36`, which includes four frames the deployment gate does not treat as crack-free (§2.4) — not over the gate's six. The figure is used with that denominator stated [TODO-AUTHOR: re-emit the guard verification with its crack-free set and generator recorded].

**It is partly memorisation and is treated as pending.** The six gate frames contribute 25.38% of all labelled not-crack pixels (§2.4) — the model was explicitly told those pixels are background. In *sampled rows* the share is smaller: at `neg_cap` = 25,536 rows per image, the six frames supply at most 153,216 of roughly 1,795,000 background rows, about 8.5% (*derived* from `models/hybrid_v5_20260824.joblib`: `n_px` 3,530,484, `crack_fraction` 0.4915, `neg_cap` 25,536). The two shares answer different questions and neither changes the direction: the negative control is measured on frames the model was trained to call background. A held-out refit is specified and written — `code/audit/heldout_crackfree_fp.py` refits the architecture with every `CLEAN_SPECIMENS` frame's rows removed and re-measures the same quantity, emitting `trained_WITH_clean.mean_fp_area`, `trained_WITHOUT_clean.mean_fp_area` and `inflation_factor` to `out/txm_heldout_crackfree_fp.json`. **That script has not been run and the artifact does not exist**; the figure is reported as a contaminated measurement, not an upper bound in any direction, until it does [TODO-AUTHOR: run `code/audit/heldout_crackfree_fp.py`]. The script's own limitation travels with its result: it refits the 17-feature branch only, in both arms, deliberately and with the reason stated in-code (`:78–111`), so it bounds the contamination for that branch and not for the deployed ensemble.

### 2.6 Metrics, and the domain in which they are computed

Because the labels are sparse strokes, all agreement metrics are computed **inside the labelled domain** — the union of painted crack and painted not-crack. Outside it there is no truth to agree with (`cldice_centreline.json .definition.domain_restriction`). That restriction is not a technicality; it is the blind spot §2.7 measures.

- **Tprec** = |skel(pred) ∩ label| / |skel(pred)| — is the predicted centreline inside the label.
- **Tsens** = |skel(label) ∩ pred| / |skel(label)| — is the label centreline inside the prediction.
- **clDice** = harmonic mean of the two.
- **IoU** = pixel intersection over union within the labelled domain.

Definitions verbatim from `cldice_centreline.json .definition`.

**The shipped values come from one artifact and one mask.** `txm_width_clip_final.json` (n = 71 frames, 61 labelled and measurable) carries the deployed operating point before and after the width clip: clDice 0.8462 → **0.8234**, Tsens 0.7978 → **0.7605**, Tprec 0.9687 → **0.9703**, IoU 0.7334 → **0.6748**, predicted area fraction 0.03872 → **0.03589**. The paired deltas are −0.0168 (clDice) and −0.0260 (Tsens). Values of 0.863 / 0.955 / 0.814 that appear in project documents are superseded: they are pre-clip and come from a different run.

**A disagreement we do not paper over, and how far it has been resolved.** `cldice_centreline.json` and `txm_width_clip_final.json` both claim to measure the deployed operating point *before* the clip, and they disagree: clDice 0.8635 vs 0.8462, Tsens 0.8145 vs 0.7978, Tprec 0.9546 vs 0.9687, and IoU 0.5047 vs 0.7334 — a 45% gap in IoU that the clip cannot explain, since both are pre-clip. The cause is an undocumented difference in domain restriction or estimator and is still unidentified. What has since been settled is which family the shipped IoU belongs to: the independent 2026-09-21 run `txm_iou_ceiling_v2.json` (generator `code/audit/measure_iou_ceiling.py`, one pass, one stated IoU convention) reports a shipped-mask median of 0.6748000339306119, byte-identical to `txm_width_clip_final.json .IoU_after`. Consequently: the two runs are never cross-quoted, and where a Tprec is reported the run is named.

**What is and is not derivable per specimen on the shipped mask.** `txm_width_clip_final.json .per_frame` stores `clDice_new` and `Tsens_new` for the 61 measurable frames, and `txm_iou_ceiling_v2.json .per_frame` stores `iou_shipped` for the 61 labelled frames, so per-specimen shipped clDice, Tsens and IoU are all derivable. Recomputed for this paper with filename grouping (§2.3):

| group | n | shipped clDice | shipped Tsens | shipped IoU |
|---|---|---|---|---|
| AM | 25 | 0.7140 | 0.5897 | 0.5468 |
| B2 | 14 | 0.8809 | 0.7918 | 0.7836 |
| B3 | 10 | 0.9573 | 0.9757 | 0.9063 |
| Wrought | 12 | 0.8388 | 0.7874 | 0.7720 |

(clDice and Tsens *derived* from `txm_width_clip_final.json .per_frame`; IoU *derived* from `txm_iou_ceiling_v2.json .per_frame`. The clDice column reproduces `docs/WIDTH_REFERENCE.md:171–175` to three decimals.) **Per-frame Tprec on the shipped mask is the one quantity no artifact in `out/` carries**; the only per-specimen Tprec that exists (`cldice_centreline.json .per_specimen`) is on the pre-clip mask, is from the run this section shows is not comparable, and is labelled as such wherever it is used [TODO-AUTHOR: re-run per-frame Tprec on the shipped mask].

### 2.7 The unlabelled-area census

For each frame, the accepted mask is intersected with the correction map and the area falling on label value 0 — never labelled either way — is counted (`code/audit/measure_unlabelled_area.py`, emitting `txm_unlabelled_area_per_frame.json`, one record per frame with the mask version in the record). Components of that unlabelled area are counted with a 4,000 px floor, matching the sampling frame of the blind adjudication (§2.11), and the distance from each unlabelled pixel to the nearest painted stroke is computed with a 25 px band, so that stroke rim can be separated from genuine unvisited area (`.frac_unlab_within_25px_of_stroke` = 0.3239).

The census covers **65 of 71 frames** (`.n_frames`): the six absent frames all carry zero painted crack, and all six have an empty shipped mask — five of them already before the straight-line guard, and `b3_amb` only after it (`txm_width_clip_final.json .per_frame.area_new` = 0.0 for five, 0.00016907 for `b3_amb`, which `drop_straight_lines` then empties, `txm_small_component_adjudication.json .guard.verified.detail[1]`; *derived* by set difference against `app_data/images/*/meta.json`). Corpus totals are 32,504,070 accepted px and 8,016,109 unlabelled px (`.accepted_px`, `.unlabelled_px`), a corpus share of **24.66%** (`.corpus_frac_unlabelled`).

Component counts drift between the two censuses — 12,303 sub-4,000 px components on 2026-09-20 (`txm_small_component_adjudication.json .population.n_components`) against 12,583 total minus 284 at ≥4,000 px on 2026-09-21 (`txm_unlabelled_area_per_frame.json .n_unlab_components`, `.n_unlab_components_ge_4000px`) — because `drop_straight_lines` shipped between the runs and the mask moved underneath the measurement. The later census is used and the drift is stated.

### 2.8 The IoU ceiling

To establish what a *physically correct* answer can score against these labels, the annotator's painted crack is skeletonised, dilated to a uniform width *w* ∈ {3, 5, 11, 21} px, and scored in IoU **against the label itself** (`code/audit/measure_iou_ceiling.py`, n = 61 labelled frames). The resulting ceiling is a property of the annotation alone: no detector, however good, can change it, and it is independent of model version.

**Two runs exist and the later one is used throughout.** `txm_iou_ceiling_v2.json` (2026-09-21) computes the ceiling and the shipped detector in one pass under one stated IoU convention, and its `.supersedes` block retires three keys of `txm_iou_ceiling.json` (2026-09-19) by name: `headline.deployed_model_IoU_same_frames`, `headline.model_over_ceiling` and `per_specimen.*.model_iou`. The reason is verifiable: `txm_iou_ceiling_per_frame[0].iou_model` = 0.10710188107052317 is byte-identical to `cldice_centreline_per_frame[0].iou`, from a run at 02:05 on the day the clip shipped at 14:14, so the older file's "10.1× the ceiling" compared a current ceiling against a superseded mask. We report:

| quantity | v2 (used) | v1 (retired) |
|---|---|---|
| perfect 3 px trace | **0.04977** | 0.0498 |
| perfect 5 px | **0.08219** | 0.0822 |
| perfect 11 px | **0.19571** | 0.1754 |
| perfect 21 px | **0.34307** | 0.3140 |
| detector, same frames | **0.6748** (shipped mask) | 0.5047 (pre-clip) |
| multiple of the 3 px ceiling | **13.56** (`.shipped_over_3px_ceiling`) | 10.1 |

The 3 px and 5 px ceilings agree between the runs to the digits v1 stored; the 11 px and 21 px ceilings do not, and the v2 values are used. The multiple 13.56 is quoted only with what it means attached: it is not a claim that the detector is thirteen times better than a correct answer, it is the statement that agreement with these labels is maximised by being about an order of magnitude wider than a crack. Per-specimen ceiling and shipped IoU are in §2.3 and §2.6, regrouped by filename.

### 2.9 The width instrument

#### 2.9.1 Prior art, stated plainly

The width measurement used here is **not novel, and no part of the measurement is claimed as a contribution.** Sampling intensity along lines perpendicular to a crack centreline is a named, established algorithm for automated crack width measurement — *Orthogonal Profile Extraction* — and its canonical steps are exactly the ones implemented here: skeletonise, take the local tangent, take the normal, sample the profile. Taking the edge at half the profile's peak is the **ISO50** surface-determination rule, the default in X-ray CT dimensional metrology, codified in **VDI/VDE 2630 Blatt 1.1** and with its own measurement-uncertainty literature. That width must be taken perpendicular or it overestimates by 1/cos θ is textbook, and is why the minimum over 12 directions is taken. One difference is stated precisely rather than inflated: ISO50 in XCT is a *global* threshold at the midpoint of the air/material histogram peaks, whereas this is a *local* half-maximum of each transverse profile against a locally estimated background — a routine variation, not a new instrument. The prior-art audit that established this is `docs/WIDTH_REFERENCE.md:28–55`; the method file itself carries the same notice (`code/audit/measure_transverse_fwhm.py:9–11`).

**What is ours is the use, not the instrument:** pointing a standard metrology rule at the *annotations* rather than at the specimen, and asking whether the mask — human or machine — is as wide as the feature it sits on. That question is not answerable by any metric computed against the strokes.

#### 2.9.2 The estimator

Background is the local mean of **non-mask** pixels by normalised convolution, with windows of 301, 901 and 2401 px tried in turn until more than 2% of the window is non-mask, so that the corridor being measured cannot lower its own reference. Contrast is background minus image. Noise σ is the 84.1st minus the 50th percentile of contrast outside the mask. At each sample point, profiles are taken along 12 directions at a half-length of 200 px with ±3 px recentring; the peak must clear 2σ; the **minimum** FWHM over the 12 directions is kept, because along the crack the profile never returns below half-maximum and across it the width is the feature's (`code/audit/measure_width_ratio.py:30`, `NDIR, RAD, NSAMP, RECENTRE, PEAK_SIGMA = 12, 200, 300, 3, 2.0`; `txm_width_reference.json .method`, whose own run used a half-length of 160–200 px). Mask width at a point is 2 × the Euclidean distance transform, so the reported quantity is a dimensionless ratio, mask width ÷ image FWHM, where 1.00 means the mask is exactly as wide as the feature it sits on.

#### 2.9.3 Which instrument, and the two it replaces

Three different sampling conventions exist in the artifacts and they produce three different "shipped width ratios". The paper names one primary instrument and footnotes the others rather than letting the reader collide with them:

| instrument | artifact | sampling | n | shipped ratio | frames wider than the feature |
|---|---|---|---|---|---|
| **primary — shared skeleton** | `txm_width_ratio_unified.json` | 300 points/frame on the skeleton of the **widest** mask, identical points for every variant | 64 | **0.5047** (`.rows.shipped.median`) | **0 / 64** (`.rows.shipped.over_marking`) |
| absolute — self skeleton | `txm_width_reference.json` | each mask scored on its own skeleton, 15,397 profiles | 61 | 0.6794 (`.what_it_says_about_overmarking.shipped_ratio_median`) | 7 / 61 |
| clip run's internal | `txm_width_clip_final.json` | the clip operator's own probes | 63 measurable of 71 | 0.5580 (`.width_ratio_after`) | 1 (`.overmarking_frames_after`) |

The shared-skeleton run is primary **for comparisons across masks**, because it was built to remove exactly the splice that produced the earlier published table: every variant is scored at points chosen by the widest mask's geometry, not its own. That property comes at a cost the artifact states and this paper repeats: scoring a narrow mask at points a wide mask reaches necessarily lowers it, so **these are a fair comparison across masks and are not absolute width estimates**; 0.5047 must not be quoted as "the detector is half as wide as the crack". The self-skeleton run is the better absolute estimate and is quoted whenever an absolute statement is made. Any over-marking count is given with the instrument attached (0/64, 7/61 or 1/63); they are not interchangeable.

Paired comparison of detector against human annotator uses the 61 frames carrying both, the same points and the same estimator, by Wilcoxon signed-rank (`.shipped_vs_label`). The primary run covers 64 of the 71 frames; **all seven absent frames carry zero painted crack**, but the artifact records no exclusion reason and the obvious one does not hold for all of them — `B2_2_1_lbf` and `b3_3_0lbf_268_13um` have non-zero accepted area (0.000137 and 0.000262, `txm_width_clip_final.json .per_frame.area_new`), while `b3_amb`, whose accepted area the straight-line guard removes entirely, *is* in the run. The generator raises on an empty skeleton of the wide (`tight=0`) mask (`measure_width_ratio.py:114`), which is consistent with the exclusions but is not recorded in the artifact. `docs/WIDTH_REFERENCE.md:129` states that five frames were excluded, which does not match the 64 records [TODO-AUTHOR: reconcile the exclusion count and record the rule in the artifact].

Per-specimen width medians are recomputed for this paper from `.per_frame` with filename grouping (§2.3), which moves one frame from B3 to AM: shipped AM 0.5656 (n = 25), B2 0.4915 (15), B3 0.4445 (12), Wrought 0.5064 (12); label AM 0.6154 (25), B2 0.5462 (14), B3 0.4419 (10), Wrought 0.7032 (12). The shipped and label *n* differ by group and both are given, because the published table (`docs/WIDTH_REFERENCE.md:133–138`) prints a single *n* column.

#### 2.9.4 Pre-registration

Whether the measured width is the crack's or the instrument's point spread was decided by a **pre-registered** test with the rule fixed before the corpus was read: call the width instrument-limited only if the pooled transverse FWHM has a coefficient of variation below 0.35 *and* |Spearman ρ| between FWHM and peak contrast is below 0.3 (`txm_width_reference.json .is_it_the_instrument.cv_threshold_for_instrument_limited`, `.spearman_threshold`). Measured over 11,426 profiles on 61 frames: median FWHM 45.0 px, 10th–90th percentile 3–147 px, CV 0.9157, Spearman +0.6732, frame-median FWHM against frame-median peak +0.7558 (p = 1.9 × 10⁻¹²). Both thresholds were missed by a wide margin.

Three further decision rules were pre-registered in `txm_width_prereg_and_failed_variants.md` and are reported with their outcomes, including the two that failed:

- A **per-component FWHM variant**, rejected: the rule allowed a paired median Tsens drop of 0.02 and the measured drop was 0.462, 23× the allowance (same file, "RESULT, 61 labelled frames, FRAC = 0.5: FAILS").
- An **adjudication of the material a narrowing step deletes**, verdict (A): the deleted material is dark, sitting 1.63σ above the matrix reference at 42% of the core contrast. The pre-registered sign test failed at p = 0.145 (33/57 frames, rule required p < 0.05) even though the paired Wilcoxon gives p = 0.016 — the pre-registered test is the one that counts, and the operating point did not change.
- The **four acceptance criteria (a)–(d)** for `clip_to_measured_width`, fixed before the run in the same file. Their outcomes are *not* in that file, which ends at the criteria; they are tabulated in `docs/WIDTH_REFERENCE.md:163–167`: (a) Tsens −0.026 against an allowance of −0.020, **FAIL** (clDice −0.017, pass); (b) median |ratio − 1| 0.399 → 0.442, **FAIL**; (c) PASS; (d) PASS. **Two of the four failed and the step ships enabled by default.** That is reported as an honesty item, not buried. Criterion (c) also changed denominator between pre-registration and result — it was written as "all 7 over-marking frames" and reported as "10/10 down … 0 of 53 rose" — and the two counts come from different width instruments (§2.9.3) [TODO-AUTHOR: state which instrument (c) was adjudicated on].

### 2.10 Straight-line artefact census and guard

Centreline wander is the standard deviation of a component's centreline about a straight-line fit along its own long axis, computed from per-row (or per-column) centroid means and requiring at least 8 distinct rows or columns (`_centreline_wander`, `app/core/pipeline.py:791`). The census covers standalone components of ≥200 px with bounding-box aspect ≥6 across all 71 frames (`txm_small_component_adjudication.json .straight_line_artefacts.census.scope`). The guard removes such components when wander < 1.0 px; any component containing a painted pixel is exempt, because a stroke is an assertion and shape does not overrule it.

The cut at 1.0 px rather than 2.0 px is recorded with its history: at 2.0 px the rule removed 1,360 px of painted crack on `b2_341_88_take2`, and "must not remove painted crack" had been fixed before the run (`.guard.cut_history`).

Three reporting decisions follow from reading the artifact rather than its summary keys:

- The artifact's top-level `.census.artefacts` = 6, `.artefact_px` = 17,674 and `.pct_of_accepted` = 0.0543 **include a genuine painted crack** — the wander-1.73 px, 2,433 px component on `b2_341_88_take2`, the same component the 2.0 px cut removed 1,360 painted pixels from. The artefact-only totals used here are **5 components, 15,241 px, 0.0469% of accepted area** (*derived*: 17,674 − 2,433 = 15,241; 15,241 / 32,504,070 from `txm_unlabelled_area_per_frame.json .accepted_px`).
- Two different sets of five components appear in the same artifact — the five in the 71-frame census (wander 0.13–0.71 px, *derived* from `.census.components`) and the five inside the 32-component tractable subset (wander 0.13–0.93 px over 171–788 px of length, `.straight_line_artefacts.finding`). Both statements are true; the ranges are not merged.
- The separation between artefact and crack is quoted as the two medians the artifact actually contains, **0.4736 px (straight) against 5.2511 px (curved)** (`.census.median_wander_straight`, `.median_wander_curved`). The "genuine long crack, 2.9–12.9 px" range printed in `docs/UNLABELLED_AREA.md` is **not in any artifact** — the six curved components are not enumerated — and is not used [TODO-AUTHOR: re-emit the census with the curved components listed, or the claim rests on medians alone].

Guard verification across all 71 frames, guard-on against guard-off: fires on 4 frames, removes 15,241 px (2,090 + 2,339 + 7,808 + 3,004, `.guard.verified.detail`), removes **0 px of painted crack**, leaves clDice unchanged to sixteen digits (0.8234071528920338 both sides), and moves crack-free false-positive area 0.025278% → 0.023588% over the ten-frame crack-free list (§2.5.4). On `b3_amb`, a specimen asserted crack-free throughout, predicted area goes to exactly zero (`.guard.verified.detail[1]`). The test is **component-level and therefore a lower bound**: two further artefacts fused into 440k px and 228k px crack systems ride through attached to real crack, and no within-component straightness test exists (`.guard.lower_bound`).

### 2.11 Blind adjudication protocol

Two blind runs were conducted, one on large unlabelled regions and one on small isolated indications. Both share the following design, and one property of both is stated first because it conditions everything that follows.

**The panels are language-model agents, not human experts.** There is no human expert arm. The instrument validation below shows that the agents agree with the annotator's own strokes; it does **not** show that they can independently identify crack in 316L TXM, and those are different competences. We state this as a limitation of this paper's central evidence, not as a caveat: a materials venue will require at least two blinded human readers through the same protocol before the claim can stand, and the project's own generator documentation says so unprompted (`code/audit/README.md`). The crops, the crop key, the shuffle seed, the leak-guard statistics and a deterministic scorer are all preserved, so the human run is a matter of recruiting readers rather than rebuilding anything. The **votes themselves are not regenerable** from the repository and ship as data (`txm_small_component_adjudication_v2.json .generators.note`).

**Common design.** Fixed-size square crops rendered with **no overlay drawn**; a fixed window rather than a fit-to-region crop, because region size differs systematically by class and a fit-to-region crop would hand the voter the class through the zoom level. Field IDs shuffled behind a fixed seed; no class encoded in any filename; voters told nothing about how many fields contain crack and instructed not to balance their answers. Three independent panels with different lenses — **A** morphology, **B** conservative and told it would be heavily penalised for false alarms, **C** "would you paint here" (the annotator's lens) — with different batch groupings. Three votes per field. Responses are yes / unsure / no, scored 1 / 0.5 / 0; a field is a majority "crack" when at least two of three votes are yes (`code/audit/score_blind_panel.py`).

**Run 1 — large unlabelled regions** (`txm_unlabelled_adjudication.json`). 88 fields of **700 × 700 px** native (≈20.5 µm at 29.24 nm/px), each centred on a random point *inside* a region, rendered through the app's own display transform (§2.3). 33 agents, 0 errors. Sampling frame: unlabelled components of ≥4,000 px, which hold 87.88% of all unlabelled area (`txm_unlabelled_area_per_frame.json .frac_unlab_in_components_ge_4000px`; the adjudication artifact's `.result.coverage` rounds this down to 87.8%). Five classes, four of them controls whose answer is already known:

| class | definition | n |
|---|---|---|
| UNLAB | model accepted, human never labelled — **the question** | 26 |
| POS | model accepted and human painted crack — positive control | 26 |
| NEGP | model accepted but human painted not-crack — known-disagreement control | 8 |
| NEGF | model accepted on a specimen asserted crack-free throughout | 2 (underpowered; excluded from the analysis by design) |
| NEGM | not accepted, painted not-crack — plain matrix, the floor | 26 |

`.design.classes`. **Leak guard, run before the panels:** crop mean brightness, standard deviation and dark fraction by class, so that a voter could not score a rendering statistic instead of morphology. POS vs UNLAB p = 0.96 / 0.96 / 0.98 and POS vs NEGM p = 0.92 / 0.25 / 0.52, with source frames spread 20 / 23 / 24, so no class carries one frame's texture (`.design.leak_guard`). **Instrument validation:** POS 24/26 against NEGM 4/26, two-sided Fisher **p = 2.311 × 10⁻⁸** (the artifact stores the one-sided 1.1554522808284791 × 10⁻⁸ at `.instrument_validation.fisher_p`); the panel's own false-negative rate on painted crack is 2/26, which is the error bar on everything the run reports.

**Run 2 — small isolated indications, matched on size *and* specimen** (`txm_small_component_adjudication_v2.json`; supersedes `txm_small_component_adjudication.json`). 94 fields of **400 × 400 px** (≈11.7 µm), rendered by clipping each frame's `display.npy` between its 0.5th and 99.5th percentile over the whole frame (`build_small_component_crops.py:138`) — a different rendering from run 1, which is recorded rather than retroactively harmonised. Classes: **TEST** = the 32 components of the tractable subset, defined as ≥200 px **and** ≥200 px from any painted stroke (`txm_small_component_adjudication.json .tractable_subset.rule`), spanning 266–3,987 px (`txm_small_component_cropkey_v2.json`); **POSS** = 32 painted-crack components of 200–4,000 px inside the accepted mask (observed 272–3,952 px); **NEGM** = 26 fields of unaccepted painted not-crack; **CFREE** = 4 fields on crack-free-asserted specimens.

Control matching is the methodological point of the second run. Each TEST component is matched to the nearest-size unused control component **from the same specimen group**, and where a group cannot supply one the generator records a shortfall rather than substituting across groups. Shortfall was zero; group mix is identical (B2 11, Wrought 13, AM 6, B3 2 in both arms); sizes match overall at p = 0.468 and within every group at p = 0.333–0.937; the leak guard's minimum p is 0.057 (`.matching`). Because the mixes are identical, the control standardised to the test mix equals the raw control rate, 0.40625 (`.control_standardised_to_test_mix`, `.control_raw`).

**Why that matching exists, reported as a failure rather than deleted.** Run 1 of this design matched on size only. The control came out 62% AM against the test class's 41% Wrought, and direct standardisation of the control to the test mix moved its yes-rate from 0.562 to 0.313 against a test rate of 0.344 — the entire reported gap was the confound, and the prevalence estimate built on that sensitivity was void (`txm_small_component_adjudication_v2.json .why`, `.supersedes`; `code/audit/README.md`). The superseded run is reported in §3 as a methods failure. The scorer now prints the group mix and the standardised control rate on every run, asked for or not.

**Prevalence.** Rogan–Gladen correction using panel sensitivity from the matched control arm and false-positive rate from the matrix arm, with a bootstrap 95% interval (`.sens` 0.40625, `.fpr` 0.11538, `.prevalence` 1.0, `.ci` [0.3209, 1.0]). The point estimate of 1.0 is an arithmetic artefact of the observed rate landing exactly on the estimated sensitivity (0.40625 = 0.40625) and is never quoted without both the interval and that sentence. Panel sensitivity at this size is low in absolute terms — 13/32 on **known** crack — which caps how much this arm can settle, and the interval says so.

### 2.12 Statistics and reporting conventions

- **Tests.** Fisher's exact test on majority-vote counts; Mann–Whitney U on the mean vote score; Wilcoxon signed-rank for paired within-frame mask comparisons; Spearman ρ for the width/contrast association.
- **Sidedness is stated for every p-value, and two-sided values are reported.** The scorer calls Fisher with `alternative='greater'` (one-sided) and Mann–Whitney two-sided, and the artifacts store both families without labelling them, so each stored value was recomputed. Conversions used here: instrument validation 1.1554522808284791 × 10⁻⁸ (stored, one-sided) → **2.3109 × 10⁻⁸** two-sided; small-component test vs matrix 0.013402893334705752 (stored, one-sided, 13/32 against 3/26) → **0.018454** two-sided; Panel-B UNLAB vs NEGM (16/26 against 2/26) 4.2396 × 10⁻⁵ one-sided, absent from the artifact → **8.4792 × 10⁻⁵** two-sided. Values already two-sided are labelled as such: `.result.vs_POS_p` = 0.3135 and `.result.vs_NEGM_p` = 4.44 × 10⁻⁸ are two-sided Mann–Whitney on the vote score, **not** Fisher; the corresponding Fisher tests on the majority counts are UNLAB vs POS **p = 1.000** and UNLAB vs NEGM **p = 2.2178 × 10⁻⁹** (computed here from 25/26, 24/26 and 4/26). Where an artifact mixes the two test families, the paper labels each.
- **Rounding.** Figures are never rounded in the direction that flatters the method. Two corrections carried from project documents: `docs/UNLABELLED_AREA.md:32` states that on `b2_336_25` the unlabelled share is "95% of everything the detector marked", and the per-frame record gives **94.65%** (`.per_frame` `frac_unlabelled` = 0.9464765648728102) — and `b2_336_25` is not the corpus maximum, which is `HC_316L_fatigue_1770_tip_zoom` at **97.27%**, followed by `HC_316L_fatigue_1790_tip_zoom_2` at 95.01% (*derived* from the same field); and the ≥4,000 px coverage is **87.88%** (`.frac_unlab_in_components_ge_4000px` = 0.8787714837709917), not the 87.8% stored in the adjudication artifact.
- **Numbers with no artifact field are not used.** Three are named so that a reader comparing the paper against the repositories can see they were dropped deliberately: the "0.197% of painted not-crack" false-positive figure in `docs/UNLABELLED_AREA.md:119–121`, which survives elsewhere only as prose inside an artifact (`txm_unlabelled_adjudication.json .unexpected.caveat`) with no numeric field behind it, and whose nearest numeric match, 0.0197%, is a different quantity with a different denominator; the 2.9–12.9 px curved-crack wander range (§2.10); and the "0.09%" and "0.13%" area shares of the 1 px and 2–9 px components, whose counts (6,921 and 3,141) are in `txm_small_component_adjudication.json .population` but whose area shares are not.

### 2.13 Software, artifacts, and what is not reproducible

Implementation is Python with scikit-learn (MLP, `StandardScaler`), scikit-image, SciPy and NumPy; the encoder is Segment Anything ViT-H (`facebook/sam-vit-huge`), used frozen. Audit generators are in `code/audit/`: `measure_width_ratio.py` → `txm_width_ratio_unified.json`; `measure_unlabelled_area.py` → `txm_unlabelled_area_per_frame.json`; `measure_iou_ceiling.py` → `txm_iou_ceiling_v2.json`; `measure_transverse_fwhm.py`; `build_small_component_crops.py` → crops plus `txm_small_component_cropkey_v2.json`; `score_blind_panel.py` → the scored adjudication JSONs; `heldout_crackfree_fp.py` (written, not yet run). Each is restartable and appends JSONL, because a full sweep takes about 20 minutes and was repeatedly killed by memory pressure.

Four reproducibility limits are stated rather than softened.

- **The panel votes are not regenerable** (§2.11); the scoring of them is deterministic and re-runnable, the votes are an artifact of a specific model and harness and ship as data.
- **`code/audit/README.md` lists `measure_transverse_fwhm.py` as emitting `out/txm_transverse_fwhm.json`, and that file does not exist**; the §2.9.4 numbers are read from `txm_width_reference.json`, which has no listed generator [TODO-AUTHOR: re-run the generator or correct the mapping].
- **The same README does not list `measure_iou_ceiling.py` or `heldout_crackfree_fp.py` at all**, although the first produced the artifact §2.8 relies on [TODO-AUTHOR: complete the mapping table].
- **The generator of the straight-line guard verification is in neither repository** (§2.5.4, §2.10), which is why its crack-free denominator had to be recovered arithmetically rather than read.

Finally, this work's own instrumentation failures are recorded and are reported in §3 rather than omitted, because a paper about measurement that hides its own broken measurements is not an audit. Three checks in this project reported success while measuring nothing (`txm_small_component_adjudication.json .checks_that_reported_success_while_measuring_nothing[0..2]`): a NumPy 2.0 `ptp` removal crashed the straightness census on exactly the 10 frames containing long components while the per-frame handler swallowed the error and the summary averaged the survivors; a guard verification compared `effective_mask` against `effective_mask`, so both arms were guarded and the test could not fail; and a duplicated empty-result fallback meant `b3_amb` kept shipping its false positive through a guard that reported PASS. A selftest probe now asserts all three properties, including the ordering of the fallback and the straightness guard described in §2.5.3.

All numbers checked against the JSON. Findings first, then the corrected markdown.

**Numbers that do not survive the artifacts**

| Where | Drafted | Artifact |
|---|---|---|
| §3.2 | label width 10th–90th pct "19.6–197.1 px" | 19.7–165.3 px on every standard percentile method (`linear`/`lower`/`higher`/`nearest`/`midpoint`/`inverted_cdf`). 197.089 appears only under the `weibull` plotting position. Inflates the tail in the flattering direction. |
| §3.2 | `cldice_centreline_per_frame.json` "record 0: `iou_model`" | the key there is `iou` (`iou_model` is the ceiling file's key). Value 0.10710188107052317 is byte-identical in both across all 61 records — the claim holds, the key name does not. |
| §5.6 | "elongation 18 to 197" | 18.0 is census component [1] — the 1.73 px-wander, 2,433 px component that is 55.9% painted crack and is *excluded*. The artifact's range for the five is 24–197 (`.straight_line_artefacts.finding`; derived 24.4–197.0). |
| §3.4 | UNLAB median comp 9,707 px | 9,706.5 |

**Claims stronger than their evidence**

1. §3.4, `field_085`: "rejected independently by all three panels for the same reason." It was **1 yes / 2 no** — Panel A no, Panel B no, **Panel C yes**, `s = 0.333`. The printed figure shows the same tally.
2. §5.6: "the panel rejected them." Of the five high-elongation tractable components, scores are 0.000, 0.000, 0.333, **0.667, 0.667** — three majority-rejected, **two majority-called crack**.
3. §5.6 conflates two different five-component sets. The adjudicated five (of 32) wander 0.13–0.93 px over 171–788 px; the geometric census five wander 0.13–0.71 px over 415–877 px. Census numbers were attributed to the adjudicated set.
4. §5.5: "the artifact's own `.supersedes` note says so." `.supersedes.run2_verdict` only restates "prevalence 100%, CI 32-100%" — it carries no disclaimer of the point estimate.
5. §5.5: "Per stratum the two arms were never far apart" — no per-stratum run-1 data in any artifact. Removed.
6. §5.7 / Figures: "No figure for this section currently exists in either repository; `UNLABELLED_AREA.md` cites `results/blind_adjudication.png`, which is not present." **Both are false.** `/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline/results/blind_adjudication.png` (1250×2048) and `results/small_component_adjudication.png` (1085×2164) both exist. The second shows the **superseded run 1** (18/32 control, 11/32 test) and must be regenerated.
7. §5.7: "Every crop … preserved." No `field_*.png` or `site_*.png` exists in either repository. Only crop keys plus the fixed seed `np.random.default_rng(20260921)` (`code/audit/build_small_component_crops.py:61`) — regenerable, not preserved.

**Limitation softened / omitted**

§5.6 quotes `crackfree_fp_before/after` with no mention that the six crack-free specimens contribute 95,351,647 px = **25.4% of every not-crack training label** (`code/audit/heldout_crackfree_fp.py` docstring), so that figure is partly memorisation. `out/txm_heldout_crackfree_fp.json` does not exist — the held-out run has not produced an artifact. Added as a drop-in slot.

**Dead claims:** none crept back. clDice 0.8234 is the shipped value (the superseded 0.8635/0.9546/0.8145 set is correctly used only to identify the stale run); no corpus-wide over-marking claim; no temporal claim; no FWHM-novelty claim. Everything else in the draft reproduced exactly, including all recomputed one-sided/two-sided corrections.

```markdown
## 3. Results: metric blindness and the unlabelled quarter

Every accuracy number reported for this detector — clDice, topological precision and
sensitivity, IoU — is computed over pixels the operator labelled, because outside the
strokes there is no reference to compare against. This section measures how large that
blind spot is, shows that the labels themselves cap IoU below 0.05 for a physically
correct crack, and then asks what the unmeasured area actually contains. The last
question is answered by blind adjudication, and the first attempt at it was confounded;
we report the confound and the reversal it forced, because the correction is the evidence
that the protocol has teeth.

### 5.1 A quarter of the accepted mask was never labelled either way

We classified every pixel the shipped detector accepts into three states against the
operator's correction map: painted crack, painted not-crack, and never painted. The mask
measured is the deployed one — `effective_mask(corrections='none', tight=True)` with
`WIDTH_CLIP` and `drop_straight_lines` active, and the mask version is recorded in every
per-frame record of `txm_unlabelled_area_per_frame.json`. Across 65 frames the detector
accepts 32,504,070 px, of which 8,016,109 px (**24.7%**, area-weighted) carry no operator
judgement of either kind (`txm_unlabelled_area_per_frame.json`,
`.accepted_px` / `.unlabelled_px` / `.corpus_frac_unlabelled` = 0.2466).

The corpus figure conceals a fourfold spread between specimen sets:

| set | frames | accepted px | never labelled px | share never labelled |
|---|---|---|---|---|
| AM / HC 316L | 24 | 5,681,798 | 2,900,998 | **51.06%** |
| B2 | 16 | 10,348,884 | 3,812,723 | **36.84%** |
| Wrought | 12 | 9,098,031 | 971,071 | **10.67%** |
| B3 | 13 | 7,375,357 | 331,317 | **4.49%** |
| **corpus** | **65** | **32,504,070** | **8,016,109** | **24.66%** |

All entries from `txm_unlabelled_area_per_frame.json`, `.per_group.<G>.{n, accepted_px,
unlabelled_px, frac_unlabelled}`.

The per-frame distribution is worse than the corpus mean suggests. Nineteen of the 65
frames have more than half their accepted area unlabelled (*derived* from
`.per_frame[].frac_unlabelled`). The extreme by that measure is an AM frame,
`HC_316L_fatigue_1770_tip_zoom`, at 97.27% of its accepted area; the extreme in absolute
terms is `b2_336_25`, where the unlabelled accepted area is **94.65%** of what the
detector marked and **17.34%** of the entire frame
(`.per_frame[].frac_unlabelled` = 0.94648, `.unlabelled_frac_of_frame` = 0.17338). On that
frame, essentially every accuracy statistic we can compute is computed on the remaining
5.35% of the mask.

The unlabelled area is not a scatter of stray pixels. It resolves into 12,583 connected
components, of which 284 are at least 4,000 px and hold **87.88%** of all unlabelled area
(`.n_unlab_components`, `.n_unlab_components_ge_4000px`,
`.frac_unlab_in_components_ge_4000px` = 0.87877). Only **32.39%** of the unlabelled area
lies within 25 px of a painted stroke (`.frac_unlab_within_25px_of_stroke`), so the bulk of
it is not stroke rim — it is territory the brush never approached. The remaining 12,299
components are smaller than 4,000 px and are treated separately in §5.5.

One ordering is worth recording and not over-reading. Ranked by median label width from
`txm_iou_ceiling_per_frame.json` — AM 36.4 px, B2 88.1 px, Wrought 96.6 px, B3 126.2 px —
the four sets fall in exactly the reverse order of their unlabelled share. With n = 4 sets
this is an observation, not a result (a perfect inverse rank correlation on four points
does not reach significance at any conventional threshold), but it is consistent with the
simplest reading: the sets where the annotator painted narrowly are the sets where the
detector found the most material the annotator never ruled on.

**Why six frames are missing.** The census covers 65 of the 71 frames. The generator
`measure_unlabelled_area.py` skips any frame with an empty accepted mask or no correction
map, tagging both causes with a single combined string (`"no mask or no correction map"`)
that does not distinguish them, and the aggregated artifact retains no record of the
skipped frames at all. At least one exclusion is explained inside the evidence: `b3_amb`
is driven to zero accepted area by the straight-line guard
(`txm_small_component_adjudication.json`, `.straight_line_artefacts.guard.verified.detail[1]`,
`area_after` = 0.0), and it is indeed absent from the census. **[TODO-AUTHOR]** Re-run the
census with a distinct skip reason retained per frame and the skipped records carried into
the artifact, so the six exclusions are individually accounted for rather than inferred.

### 5.2 The annotation caps IoU at 0.0498 for a physically correct crack

The operator's strokes are a broad brush. To quantify what that costs any pixel-overlap
metric, we skeletonised each painted crack, re-dilated the skeleton to a fixed width *w*,
and scored the result against the label it came from
(`txm_iou_ceiling.json`, `.method`; per-frame values in `txm_iou_ceiling_per_frame.json`,
n = 61 labelled frames). This quantity is a property of the annotation alone. No detector
enters it, and no detector can raise it.

| trace width | IoU against the operator's label |
|---|---|
| 3 px | **0.0498** |
| 5 px | 0.0822 |
| 11 px | 0.1754 |
| 21 px | 0.3140 |

(`txm_iou_ceiling.json`, `.headline.perfect_3px_trace_IoU` / `.perfect_5px` /
`.perfect_11px` / `.perfect_21px`; each verified as the median over the 61 records in
`txm_iou_ceiling_per_frame.json`.)

A perfectly placed 3 px crack — the correct answer, in the right place, at a physically
plausible width — scores IoU 0.0498 on this corpus. Even a 21 px trace, seven times wider
than the feature, scores below 0.5 on 44 of the 61 frames (*derived* from
`txm_iou_ceiling_per_frame.json`, `iou_w21`). The driver is label width: the median painted
stroke is **73.4 px** across (*derived* median of `label_width_px`; the artifact's
`.headline.median_label_width_px` rounds this to 73.0), with a 10th–90th percentile range
of **19.7–165.3 px** and a maximum of 337.5 px (*derived*, linear-interpolation
percentiles).

The ceiling tracks brush width by specimen set:

| set | n | 3 px ceiling | median label width |
|---|---|---|---|
| AM | 25 | 0.0732 | 36.4 px |
| B2 | 14 | 0.0339 | 88.1 px |
| Wrought | 12 | 0.0343 | 96.6 px |
| B3 | 10 | 0.0291 | 126.2 px |

(`txm_iou_ceiling.json`, `.per_specimen.<G>.{n, ceiling_3px}`; widths *derived* as
per-group medians of `label_width_px` in `txm_iou_ceiling_per_frame.json`, which the
artifact rounds to 36 / 88 / 97 / 126 px.)

For orientation, the same construction on an SEM corpus with a 59 px median brush gives a
3 px ceiling of 0.1662 (`txm_iou_ceiling.json`, `.why`, `.verdict`). The TXM ceiling is
roughly three times worse.

Two consequences follow, and a third claim must be withheld.

1. An IoU measured on this corpus ranks how closely a mask reproduces a ~73 px paintbrush.
   It does not rank crack-segmentation accuracy, and it must never be compared against
   published IoU from corpora with tighter annotation.
2. The ceiling cannot be improved by a better detector. It moves only if the annotation
   moves.
3. **We do not report a "× the ceiling" multiple.** `txm_iou_ceiling.json` carries
   `.headline.deployed_model_IoU_same_frames` = 0.5047 and `.model_over_ceiling` = 10.1,
   and its `.per_specimen.<G>.model_iou` column. Those model values are byte-identical,
   record for record across all 61 frames, to the pre-`WIDTH_CLIP` run in
   `cldice_centreline_per_frame.json` (whose corresponding field is named `iou`, not
   `iou_model`; record 0 is 0.10710188107052317 in both), i.e. they are scored on a mask
   version the project no longer ships — the same run whose superseded clDice / Tprec /
   Tsens medians are 0.8635 / 0.9546 / 0.8145 against the shipped 0.8234 / 0.970 / 0.761.
   The ceiling itself is unaffected, being label-only. **[TODO-AUTHOR]** Re-run the ceiling
   script against the shipped mask if a multiple is wanted; otherwise the ceiling stands
   alone, which is how we use it here.

### 5.3 Blind adjudication: design and instrument validation

Sections 5.1 and 5.2 establish that a quarter of the mask is unmeasured and that the
available overlap metric is uninformative about crack accuracy. Neither says whether the
unmeasured area is crack the annotator missed or material the detector invented. We
addressed this by blind adjudication (`txm_unlabelled_adjudication.json`).

Eighty-eight fields of 700 × 700 px at native resolution were rendered through the
application's own display transform with **no overlay drawn**, each centred on a random
point inside a region (`.design.crops`). A fixed window was used rather than a
fit-to-region crop, because region size differs systematically by class and a
fit-to-region crop would hand the reader the class through the zoom level. Field
identifiers were shuffled behind a fixed seed, no filename encoded a class, and readers
were told nothing about how many fields contained cracks and were instructed not to
balance their answers (`.design.blinding`). Three panels with different stated lenses —
A morphology lens, B conservative lens explicitly penalised for false alarms, C annotator
lens — voted independently, three votes per field, in different batch groupings; 33
agents, 0 errors (`.design.panels`, `.design.votes`).

Five classes were presented, four of them controls with a known answer (`.design.classes`):

| class | what it is | n |
|---|---|---|
| **UNLAB** | detector accepted, operator never labelled — **the question** | 26 |
| POS | detector accepted and operator painted crack — positive control | 26 |
| NEGP | detector accepted but operator painted not-crack — known disagreement | 8 |
| NEGF | detector accepted on a specimen asserted crack-free | 2 (excluded, underpowered) |
| NEGM | not accepted, painted not-crack — plain matrix, the floor | 26 |

The UNLAB fields were drawn from components of at least 4,000 px, median 9,706.5 px, range
4,001–64,509 px (*derived* from `.per_field[].comp_px` where `cls == "UNLAB"`).

**Leak guard, run before the panels.** If crop brightness, contrast or dark fraction
separated the classes, readers could have scored those instead of morphology. They do not:
POS vs UNLAB p = 0.96 / 0.96 / 0.98 and POS vs NEGM p = 0.92 / 0.25 / 0.52 on mean
brightness, standard deviation and dark fraction respectively, with source frames spread
20 / 23 / 24 across the three classes so no class carries a single frame's texture
(`.design.leak_guard`).

**Instrument validation.** On the two classes with a known answer, the panel called crack
on 24 of 26 painted-crack fields and 4 of 26 plain-matrix fields
(`.instrument_validation.POS_majority_yes`, `.NEGM_majority_yes`). Two-sided Fisher exact
p = 2.31 × 10⁻⁸; two-sided Mann–Whitney on the continuous vote score
p = 4.50 × 10⁻⁸ (*recomputed here from `.per_field[]`; note that the stored
`.instrument_validation.fisher_p` = 1.16 × 10⁻⁸ and `.mannwhitney_p` = 2.25 × 10⁻⁸ are
the one-sided values, exactly half*). The panel's own false-negative rate on known crack is
2/26 (`.panel_false_negative_rate`), and that is the error bar on everything in §3.4.

### 5.4 The large unlabelled regions read as crack

| class | majority called crack | mean vote score |
|---|---|---|
| POS (painted crack) | 24/26 (92.3%) | 0.8910 |
| **UNLAB (never labelled)** | **25/26 (96.15%)** | **0.8590** |
| NEGP (painted not-crack) | 7/8 (87.5%) | 0.8333 |
| NEGM (plain matrix) | 4/26 (15.4%) | 0.2115 |
| NEGF (crack-free specimen) | 0/2 | 0.0000 |

Majorities from `txm_unlabelled_adjudication.json`, `.result.UNLAB_majority_yes` /
`.UNLAB_pct` = 96.154 and `.instrument_validation`; mean scores *derived* from
`.per_field[].s` grouped by `.cls`.

Weighted by component area, **93.42%** of the sampled unlabelled area reads as crack
(`.result.area_weighted_pct`).

The two comparisons the artifact reports are Mann–Whitney tests on the continuous vote
score, not Fisher tests on the majority counts, and we label them accordingly because the
artifact does not:

- UNLAB vs POS: **two-sided Mann–Whitney p = 0.3135** (`.result.vs_POS_p`). Two-sided
  Fisher on the majority counts (25/26 vs 24/26) gives p = 1.000 (*recomputed*). By either
  test family, the unlabelled area is indistinguishable from crack the operator painted.
- UNLAB vs NEGM: **two-sided Mann–Whitney p = 4.44 × 10⁻⁸** (`.result.vs_NEGM_p`);
  two-sided Fisher p = 2.22 × 10⁻⁹ (*recomputed*).

**The result survives the strictest reader alone.** Panel B was instructed that false
alarms would be heavily penalised and to default to "no"; its overall yes-rate is 49%
against 75% for Panel A and 69% for Panel C (`.panel_agreement.yes_rate`). On Panel B
alone: POS 20/26, UNLAB 16/26, NEGM 2/26 (`.result.conservative_panel_only`). UNLAB vs POS
two-sided Fisher p = 0.3678 (`.UNLAB_vs_POS_p`, verified as two-sided); UNLAB vs NEGM
two-sided Fisher p = 8.48 × 10⁻⁵ (*recomputed; `UNLABELLED_AREA.md` prints 4.2 × 10⁻⁵,
which is the one-sided value*). Pairwise panel agreement is A–B 73%, A–C 90%, B–C 76%
(`.panel_agreement`).

**Coverage.** The sample was drawn from components of at least 4,000 px, which hold 87.9%
of all unlabelled area (`txm_unlabelled_area_per_frame.json`,
`.frac_unlab_in_components_ge_4000px` = 0.87877; the adjudication artifact's
`.result.coverage` prints 87.8%, which rounds the wrong way). The claim therefore covers
**21.7%** of all accepted area (0.8788 × 0.2466 = 0.2167) — not a residual fringe.

**The one genuine false positive, and it was not unanimous.** `field_085`, 27,179 px on
`wrought_316L_fatigue_1280_cycles`, was called not-crack by two of the three panels —
Panel A and Panel B rejected it as a specimen free surface, a large dark region with metal
on one side only bordered by parallel Fresnel fringes, while **Panel C voted crack**
(`.per_field[]` for `field_085`: 1 yes / 2 no, `s` = 0.333; the free-surface reading is
`.the_one_rejection.why`). It is one rejection in 26, carried on a 2–1 split, not a
unanimous one.

**An unexpected result, reported with the artifact's own caveat.** NEGP is the class where
the detector accepted a region the operator had painted as not-crack. The blind panel sided
with the detector on 7 of 8, mean score 0.8333 against 0.8910 for painted crack
(`.unexpected.what`). The artifact's caveat travels with the number and we repeat it in our
own voice: n = 8, and these are the *largest* detector/operator disagreements — components
of at least 4,000 px inside painted not-crack — not a random sample of that area. Selecting
the biggest disagreements and finding the detector looks right on them is what selection
does. **This is not a measurement of annotation accuracy and must not be quoted as one.**
What it supports, qualitatively, is that any false-positive rate computed as a fraction of
painted not-crack area is an upper bound, because some of that area is the annotator.
**[TODO-AUTHOR]** `UNLABELLED_AREA.md` and `.unexpected.caveat` both anchor this argument
to "the 0.197%-of-painted-not-crack figure". That figure appears in no artifact in either
repository and cannot be sourced; the nearest match, 0.0197%, is predicted area on
crack-free specimens, a different quantity with a different denominator and off by a factor
of ten. Source it or state the argument without a number.

### 5.5 The small isolated indications: a confound, and the reversal it forced

§3.4 covers components of at least 4,000 px. The remainder needs separate treatment, and it
is where our first attempt at this measurement went wrong.

**Most of the small components are not detections.** The accepted-component floor is
`MIN_BLOB_PX = 2000` (`app/core/pipeline.py:470`), so no accepted component is smaller than
that. The 12,299 unlabelled components below 4,000 px are the unpainted remainder of large
accepted components after intersection with the label map — the rim of a stroke that did
not quite cover the crack beneath it. They total 982,920 px, 12.2% of unlabelled area;
6,921 of them are exactly one pixel and 3,141 are 2–9 px; 80.4% of their area lies within
25 px of a painted stroke and the median distance to the nearest stroke is 11 px
(`txm_small_component_adjudication.json`, `.population.*`). *This whole population block is
the 2026-09-20 run, which counts 12,303 components; the 2026-09-21 census on the current
mask gives 12,299 (`txm_unlabelled_area_per_frame.json`, 12,583 total minus 284 at or above
4,000 px). The four-component difference is the mask moving under the measurement when
`drop_straight_lines` shipped. We use 12,299 for the count; the area and distance figures
beside it, and the tractable-subset figures below, have not been recomputed on the current
mask and are quoted from the earlier run.*

Restricting to components where "nobody checked this" is a real claim — at least 200 px in
size **and** at least 200 px from any painted stroke — leaves **32 components, 74,404 px,
0.229% of all accepted area** (`.tractable_subset.n`, `.px`,
`.share_of_all_accepted_area`). That is the entire exposure of this class even if every one
of them were a false positive.

**First run: size-matched, and confounded.** A second blind panel, 400 px fields (so a
200 px clearance guarantees no painted crack appears in frame), with positive controls
stratified to match the test sizes at median 2,418 vs 2,403 px, p = 0.94. It reported the
test class at 11/32 against a control of 18/32, one-sided p = 0.0656, concluded the
instrument was weak at this size, and quoted a Rogan–Gladen prevalence CI of 0–100%
(`txm_small_component_adjudication.json`, `.adjudication`).

**The entire gap was a confound.** Size was matched; specimen was not. The control class
came out 62% AM while the test class was 41% Wrought, and direct standardisation of the
control to the test specimen mix moves its yes-rate from 0.562 to 0.313, against a test
rate of 0.344 (`txm_small_component_adjudication_v2.json`, `.why`). There was no gap, the
"weak at small size" contrast was not a size contrast, and the sensitivity that the
prevalence estimate rested on was void.

**Second run: matched on size and specimen.** The crop builder
(`code/audit/build_small_component_crops.py`) was rewritten to match within specimen group
first and then on size, and to refuse cross-group substitution — recording a shortfall
instead. Zero shortfall was needed. Group mix is identical in both arms (B2 11, Wrought 13,
AM 6, B3 2), size is matched overall at p = 0.468 and within every group at p = 0.333–0.937,
and the leak guard is clear at minimum p = 0.057
(`txm_small_component_adjudication_v2.json`, `.matching.*`). Ninety-four fields, three
votes each (`.n_fields`, `.votes_per_field`).

| class | majority called crack | mean vote score |
|---|---|---|
| painted crack, size- and specimen-matched | 13/32 (40.6%) | 0.4010 |
| **isolated small indications** | **13/32 (40.6%)** | **0.4583** |
| plain matrix | 3/26 (11.5%) | 0.1154 |
| crack-free specimens, false positive by assertion | 0/4 | 0.0833 |

(`.by_class.{POSS, TEST, NEGM, CFREE}.{majority_yes, score}`. Test components: median
2,403 px, range 266–3,987 px; matched controls median 2,218.5 px, range 272–3,952 px,
*derived* from `.per_field[].px`.)

Instrument validation at this size: control 13/32 against matrix 3/26, two-sided Fisher
p = 0.0185 (*recomputed; the stored `.instrument.fisher_p` = 0.0134 is one-sided*).

- **Test against matched control: 13/32 vs 13/32, Fisher p = 1.000 — identical**
  (`.test_vs_control_fisher_p`). Two-sided Mann–Whitney on vote score p = 0.5319
  (`.test.vs_control_p`).
- Test against matrix: two-sided Fisher p = 0.0185 (*recomputed; stored
  `.test_vs_matrix_fisher_p` = 0.0134 is one-sided*); two-sided Mann–Whitney
  p = 2.06 × 10⁻⁴ (`.test.vs_matrix_p`).

**The conclusion reverses.** The small isolated indications are indistinguishable from
known crack of the same size in the same specimens, and clearly separated from matrix —
the same answer the large regions gave in §3.4. Both size classes agree: the unadjudicated
area is crack.

We report this reversal rather than only its outcome because it is the load-bearing
evidence that the protocol is capable of overturning its own result. The first run produced
a clean-looking negative, and the thing that killed it was a matching criterion we had not
enforced, found by standardising a control we had already reported. The scoring script now
prints the group mix and the standardised control rate on every run, asked for or not
(`code/audit/score_blind_panel.py`).

**The honest residuals of this arm.** Rogan–Gladen prevalence with the matched sensitivity
is 1.0 with a bootstrap 95% CI of **[0.3209, 1.0]** (`.prevalence`, `.ci`). We state in our
own voice, because no artifact states it, that the point estimate of 100% is an artefact of
the observed rate landing exactly on the sensitivity (`.obs` = `.sens` = 0.40625, so the
Rogan–Gladen numerator and denominator are the same number) and is not a claim that every
one of these components is a crack; the artifact's `.supersedes` note records only the
number and the CI. The controlling weakness is that a 40.6% control rate is low in absolute
terms: at this size the panel misses most known crack, which caps how much this arm can
ever settle, and the CI reflects it. Panel-level yes-rates show where the sensitivity goes
— Panel A 69% on test and 59% on control, Panel B 25% and 22%, Panel C 41% and 38%
(`.per_panel`) — so the conservative lens is close to silent at this component size. The
crack-free arm is n = 4 and proves nothing on its own, though Panel A did call one of its
four fields a crack (majority still 0/4; score 0.0833).

### 5.6 One class in the unlabelled area is not crack: straight-line artefacts

The adjudication is not unanimous in the detector's favour, and the exception is
systematic. Five of the 32 tractable components read as obvious cracks on every summary
statistic — elongation 24 to 197, 12 to 49 sigma darker than their surroundings — while
being near-straight, wandering 0.13–0.93 px about a line fit over lengths of 171–788 px
(`txm_small_component_adjudication.json`, `.straight_line_artefacts.finding`). **The panel
did not settle them.** Of those five, three were majority-rejected and two were
majority-called crack (*derived* from `.per_field[]` where `cls == "TEST"` and `el >= 6`:
vote scores 0.000, 0.000, 0.333, 0.667, 0.667). Geometry settles it instead.

Censused independently over all 71 frames on standalone components of at least 200 px with
bounding-box aspect ratio at least 6 (`.straight_line_artefacts.census.scope`), the
straight class has a median centreline wander of **0.4736 px** against **5.2511 px** for
the curved class (`.median_wander_straight`, `.median_wander_curved`). The census
enumerates six such components. Five of them wander **0.13, 0.18, 0.34, 0.61 and 0.71 px**
over lengths of 415–877 px, at elongations of 37.7–197.0 (`.census.components[]`). A crack
at that aspect ratio does not run straight. One of the five is on `b3_amb`, a specimen
asserted crack-free throughout. Note that the census set and the five adjudicated
components above are overlapping but not identical: the census runs over all 71 frames,
the adjudicated five are the high-aspect members of §5.5's tractable subset.

The sixth census component, at 1.73 px wander and 2,433 px, is 55.9% painted crack
(*derived*: the `cut_history` records 1,360 px of painted crack removed from it), and it is
why the guard threshold sits at 1.0 px rather than 2.0 px — a 2.0 px cut removed that
painted crack, and "must not remove painted crack" was fixed before the run
(`.guard.cut_history`). **The artefact class is therefore 5 components and 15,241 px, which
is 0.0469% of accepted area.** The artifact's top-level keys `.census.artefacts` = 6,
`.artefact_px` = 17,674 and `.pct_of_accepted` = 0.0543 include that painted crack and
should not be quoted.

The resulting guard — at least 200 px, aspect at least 6, wander below 1.0 px, painted
pixels exempt — was verified guard-on against guard-off across all 71 frames. It fires on
4 frames, removes 15,241 px, and removes **0 px of painted crack** against a
pre-registered criterion of zero. clDice is unchanged to sixteen decimal places
(0.8234071528920338 before and after), and false-positive area on crack-free material falls
from 0.025278 to 0.023588 in the artifact's units, a 6.7% relative reduction
(`.guard.verified.*`). On `b3_amb`, accepted area goes to exactly zero
(`.guard.verified.detail[1]`).

Two caveats attach to those crack-free figures and neither is optional.

- **They are partly a memorisation result.** The six crack-free specimens contribute 0
  crack pixels and 95,351,647 not-crack pixels to training — **25.4% of every background
  label the model sees** (`code/audit/heldout_crackfree_fp.py`, module docstring). Predicted
  area on those specimens is therefore measured on pixels the model was explicitly told are
  background, and the figure is not a generalisation result. A held-out refit with every
  crack-free frame removed is specified in that script and emits
  `out/txm_heldout_crackfree_fp.json`; **that artifact does not yet exist**. Until it does,
  0.025278 → 0.023588 is an optimistic bound and the guard's 6.7% relative reduction is
  measured on the favourable case. **[TODO-AUTHOR]** Drop the held-out figure in here beside
  the deployed one when the run completes; do not replace the deployed number, report both.
- **[TODO-AUTHOR]** The two crack-free figures in that block are stored in inconsistent
  units — the corpus value appears to be a percentage while the per-frame `area_before` is a
  fraction — which needs reconciling before either is printed with a unit.

**The guard is a lower bound and we state it as one.** The test is component-level, so it
catches standalone artefacts only. Two further artefacts are fused into 440k px and 228k px
crack systems and ride through attached to real crack (`.guard.lower_bound`). A
within-component straightness test would find them; none exists.

### 5.7 What this section does and does not establish

Established, and each traceable to a named artifact:

1. 24.7% of the accepted mask area, area-weighted over 65 frames, carries no operator
   judgement, ranging from 4.49% on B3 to 51.06% on AM
   (`txm_unlabelled_area_per_frame.json`). Every accuracy metric this project reports is
   blind to that area by construction.
2. A physically correct 3 px crack scores IoU 0.0498 against these annotations
   (`txm_iou_ceiling.json`). Pixel IoU on this corpus ranks brush imitation, and the
   ceiling is a property of the labels that no detector can raise.
3. Blind, controlled adjudication of the large unlabelled regions calls them crack 25/26,
   93.42% area-weighted, indistinguishable from painted crack and overwhelmingly separated
   from matrix (`txm_unlabelled_adjudication.json`); the claim covers 21.7% of all accepted
   area. Specimen- and size-matched adjudication of the small isolated indications agrees,
   at 13/32 against a matched control of 13/32
   (`txm_small_component_adjudication_v2.json`).
4. One false-positive class inside that area is identified and bounded: straight-line
   artefacts, 5 components and 15,241 px, 0.0469% of accepted area
   (`txm_small_component_adjudication.json`).

Not established, and the limits are structural rather than incidental:

- **The adjudication panels are language-model agents, not human experts.** What the
  instrument validation shows is that the panels agree with this annotator's own strokes —
  24/26 at full size, 13/32 at small size. That is not the same property as competence at
  identifying crack in 316L transmission X-ray microscopy, and we do not claim it is. The
  crop keys, the shuffling seed and the deterministic scorer are retained
  (`txm_unlabelled_adjudication_cropkey.json`, `txm_small_component_cropkey_v2.json`;
  `np.random.default_rng(20260921)` in `code/audit/build_small_component_crops.py`), so
  every field is regenerable from the source frames through the same display transform and
  the identical protocol can be re-run with blinded human readers. The rendered crop images
  themselves are **not** retained in either repository. **[TODO-AUTHOR]** At least two
  blinded human readers through the identical protocol, before this evidence is offered as
  settled.
- **The panel votes are not regenerable from the repository.** They ship as data, not as a
  reproducible computation (`txm_small_component_adjudication_v2.json`, `.generators.note`).
- **Panel sensitivity at small component size is 40.6%**, which caps what the small-component
  arm can resolve and is why its prevalence CI spans 0.32–1.00.
- **The crack-free control arms are n = 2 and n = 4** (NEGF in the large-region run, excluded
  from analysis by design; CFREE in the small-component run). They are underpowered and
  support nothing on their own. Separately, the crack-free specimens are inside the training
  background labels (§5.6), so no figure computed on them is yet a held-out figure.
- **The straight-line guard catches only standalone components**, as above.
- **The 700 px field is stated in pixels, not micrometres.** The ≈20 µm equivalent rests on
  a pixel size of 29.24 nm/px derived from operator-supplied mosaic geometry — a 30 µm
  viewing window, 0.35 overlap, 9 × 5 tiles over 6367 × 3691 px — not read from an
  instrument header. **[TODO-AUTHOR]** Confirm the 30 µm and 0.35 figures with the operator
  before any physical length in this section is presented as instrument metadata.
- **Instrument and acquisition metadata are absent from both repositories.**
  **[TODO-AUTHOR]** Beam energy, microscope, detector, exposure and reconstruction method;
  loading rig, load levels and cycle counting; specimen provenance, heat treatment, AM build
  parameters and through-thickness geometry; annotator identity, training, time per frame,
  and whether any frame was labelled twice.

**Figures.** Two exist and are present in the pipeline repository.
`results/blind_adjudication.png` (1250 × 2048) is the §3.3–5.4 panel: all five classes as
rendered fields with each field's vote tally and the panel's written reason, including the
`field_085` rejection at its true 1-yes / 2-no split; `UNLABELLED_AREA.md` cites it and the
file is there. `results/small_component_adjudication.png` (1085 × 2164) is the corresponding
panel for §5.5, but it renders the **superseded run 1** — its headline rows read 18/32
control against 11/32 test — and it must not be printed as it stands.
**[TODO-AUTHOR]** (i) Regenerate the small-component panel from
`txm_small_component_adjudication_v2.json` so it shows 13/32 against 13/32 and the
specimen-matched design. Two further slots have no figure at all: (ii) the per-group
unlabelled-area census of §3.1 with per-frame points overlaid on the group bars, from
`txm_unlabelled_area_per_frame.json`; (iii) the IoU-ceiling curve of §3.2, IoU against trace
width, with the 61 per-frame traces behind the median, from
`txm_iou_ceiling_per_frame.json`.
```

Verified every number against the artifacts. Findings and corrected section below.

**Errors found (12):**

1. **`.../x = 0` miscount — factual error.** Draft: "Two more sit at x = 0." Of the five artefacts only *one* is at x = 0 (`b3_amb`, wander 0.605); the other x = 0 component is the **sixth**, wander-1.73 painted one. The fifth artefact is interior at x = 1,689 (`txm_small_component_adjudication.json`, `.straight_line_artefacts.census.components`).
2. **"55.9% painted crack"** appears in no artifact. It is 1,360/2,433 from `.guard.cut_history` — derived, and marked as such now.
3. **"comfortably clear of the `MIN_BLOB_PX = 2000` floor"** — smallest is 2,090 px, 4.5% above the floor. Claim stronger than evidence.
4. **"flat to within 10%"** — 0.0166→0.01495 is an 11% spread. Rounded in the flattering direction (flatness is what the argument needs).
5. **"42% of the core contrast"** is asserted in `txm_width_prereg_and_failed_variants.md` but is not reproducible from its own medians: 0.0379/0.0969 = 39.1% (40.1% baseline-corrected). Replaced with the derived 39%, with the discrepancy noted.
6. **"a 73–126 px stroke"** conflates the corpus median (73.4) with the largest per-specimen value (B3, 126). Per-specimen range is 36 (AM) to 126 (B3) — `txm_iou_ceiling.json`, `.per_specimen`.
7. **"median label width of 73.4 px (`.headline`)"** — `.headline.median_label_width_px` is 73.0; 73.43 is the per-frame median. Attribution split.
8. **Mosaic geometry uncited** — 30 µm / 0.35 / 9×5 / 6367×3691 / 29.24 nm/px is `crack-depth-3d/configs/default.yaml:6` (and `README.md:70`). Now cited.
9. **"a further group-dependent subset"** is muddled; exactly 3 frames lack a label (1 B2, 2 B3, derived from `.per_frame`).
10. **"IoU … is very nearly a width ratio and almost nothing else"** — true only for a correctly centred mask; the ceiling artifact itself notes the label skeleton can sit far from the crack. Conditioned.
11. **Denominator mismatch** — 15,241 px is a 71-frame guard count, 32,504,070 px is a 65-frame accepted total. Same to 4 dp; now stated.
12. **`am_label_separability.json` cited bare** — the same file's top-level `d = −0.31` for AM is explicitly **WITHDRAWN**. Citation narrowed to `.CORRECTION_ring_control.result`.

**Dead claims:** none crept back. No 0.863/0.955/0.814 (correctly confirmed as the pre-clip `cldice_centreline.json` run, byte-identical per-frame to the ceiling's comparator — I verified 61/61 identical), no corpus-wide over-marking, no temporal claim, width-method novelty correctly disclaimed. **Softened limitations:** none; one addition made (blind panel called one artefact crack 2/3, with the LM-panel caveat).

```markdown
## 4. Results: width, and what the annotations cannot adjudicate

Every accuracy figure reported so far is scored against the operator's brush, and the preceding subsections have shown what that brush does and does not assert. Width is the one mask property that can be scored against the image instead, with no annotation in the loop. This subsection reports that measurement, shows that the detector and the annotator are indistinguishable on it, shows that the field's standard metric charges the detector for precisely that agreement, and closes with one class of indication that the intuitive crack signature cannot separate from a detector artefact.

The measurement itself is not new and is not claimed as new. Sampling intensity along lines normal to a skeletonised centreline is *Orthogonal Profile Extraction*, one of the two established algorithms for automated crack-width measurement; taking the edge at half the profile's peak is the ISO50 surface-determination rule, codified for X-ray CT dimensional metrology in VDI/VDE 2630 Blatt 1.1; and that width must be taken perpendicular or it overestimates by 1/cos θ is textbook, which is why the minimum over 12 directions is used here (prior-art review in `docs/WIDTH_REFERENCE.md`). The one difference worth stating precisely is that ISO50 in XCT is a global threshold at the midpoint of the air/material histogram peaks, whereas this is a local half-maximum of each transverse profile against a locally estimated background — a routine variation, not a new instrument. What is ours is the *use*: pointing a standard metrology rule at the annotations rather than at the specimen.

### The image-side width reference, and a pre-registered test that it is physical

At each skeleton point the contrast field (local background minus image, background estimated by normalised convolution over non-mask pixels only, window widened through 301/901/2401 px until >2% of it is non-mask) is profiled along 12 directions over ±160–200 px and the **minimum** full-width-at-half-maximum is kept; along the crack the profile never returns below half maximum and that direction is discarded. A profile is used only if its peak clears 2 background σ (`txm_width_reference.json`, `.method`).

Over 11,426 profiles on 61 frames the transverse FWHM has median 45.0 px, 10th–90th percentile 3–147 px (`txm_width_reference.json`, `.is_it_the_instrument`). Whether that width is the crack or the instrument's point-spread was settled by a test whose decision rule was fixed before the corpus was read: call the width instrument-limited only if the pooled coefficient of variation is below 0.35 **and** |Spearman ρ| between FWHM and peak contrast is below 0.3. Measured CV = 0.9157 and ρ = +0.6732; on frame medians, ρ = +0.7558, p = 1.914 × 10⁻¹² (same artifact). Both thresholds were missed by a wide margin. A fixed point-spread cannot produce a width that ranges over 50× and rises with peak depth, so width in these images is a property of each crack, and "is the mask as wide as the feature it sits on" is a question the image can answer.

In physical units the median is 1,305 nm (`txm_width_reference.json`, `.fwhm_median_nm`), but that conversion rests on a pixel size of 29.24 nm/px that is **derived from operator-supplied mosaic geometry** (30 µm viewing window, 0.35 overlap, 9 × 5 tiles over 6367 × 3691 px; `configs/default.yaml:6` in the analysis repo, where the value is carried with the comment `DERIVED`), not read from an instrument header. **[TODO-AUTHOR: confirm the 30 µm window and 0.35 overlap with the operator, or supply the pixel size from the acquisition record. All width figures below are reported in pixels and are unaffected; only the nanometre conversion depends on this.]**

### One instrument, one frame set, one set of sample points

Width ratios are trivially manipulable by the choice of where to sample. A mask scored at skeleton points of its own geometry is scored where it is thickest. This project produced three different "shipped width ratio" figures from three estimators before this was noticed. All comparisons in this subsection therefore come from a single run in which every mask variant is scored with the same estimator, on the same frames, **at the same sample points** — the skeleton of the widest variant — so that no mask is measured at points chosen by its own shape (`txm_width_ratio_unified.json`, `.what`; generator `code/audit/measure_width_ratio.py`).

| mask | n frames | median ratio (mask width / image FWHM) | IQR | frames wider than the feature |
|---|---|---|---|---|
| wide (`tight=0`) | 64 | 0.672 | 0.576–0.935 | 14 |
| tighten only, no clip | 64 | 0.590 | 0.512–0.739 | 7 |
| **shipped** | 64 | **0.505** | 0.444–0.587 | **0** |
| human brush label | 61 | 0.529 | 0.431–0.770 | 11 |

All values from `txm_width_ratio_unified.json`, `.rows.*`. The artifact's own qualifier travels with the table: because every variant is scored at one shared skeleton, **these are a fair comparison across masks and not absolute width estimates.** They do not license a statement of the form "the detector is half as wide as the crack".

### The detector and the annotator disagree about width no more than two annotators would

On the 61 frames carrying both a shipped mask and a brush label, the shipped detector's median ratio is 0.5061 and the annotator's is 0.5289; the paired median difference is **−0.0113** and a paired Wilcoxon test gives **p = 0.0613** (`txm_width_ratio_unified.json`, `.shipped_vs_label`, whose own verdict field reads "indistinguishable"). That difference is an order of magnitude smaller than the frame-to-frame spread of either mask — the interquartile ranges are 0.143 wide for the shipped mask and 0.338 wide for the label (*derived* from `.rows.shipped.iqr`, `.rows.label.iqr`). The direction of the difference is that the detector is the *narrower* of the two.

Per specimen group (*derived* as medians of `.per_frame[].ratio_shipped` and `.ratio_label`; the two arms have different n because exactly three frames carry a mask and no label — one B2 and two B3, *derived* from `.per_frame`):

| group | n (shipped) | shipped | n (label) | label |
|---|---|---|---|---|
| B2 | 15 | 0.4915 | 14 | 0.5462 |
| B3 | 13 | 0.4548 | 11 | 0.4526 |
| AM / HC_316L | 24 | 0.5634 | 24 | 0.5819 |
| Wrought | 12 | 0.5064 | 12 | 0.7032 |

The natural reading is that the detector and the annotation disagree about width no more than two annotators would. **That reading is an analogy, not a measurement, and this corpus cannot upgrade it**: there is no second annotator and no frame is known to have been labelled twice **[TODO-AUTHOR: state whether any frame was annotated independently more than once; if any were, the between-annotator width difference is directly measurable and should replace this analogy]**. Two further statistical cautions belong with the number. First, p = 0.0613 is a failure to reject, not a demonstration of equivalence; no artifact in `out/` contains an equivalence test or a confidence interval on the paired difference, so **[TODO-AUTHOR: add a TOST or a bootstrap CI on the −0.0113 paired difference to state what size of difference the data actually exclude]**. Second, at n = 61 the test sits close enough to the conventional threshold that a modest change in frame set could move it either way; the honest claim is "no difference detectable at this n", and it is reported here in that form.

### Both masks under-mark the feature, and the detector never over-marks

Against a ratio of 1.0 — a mask exactly as wide as the dark feature — both arms fail in the same direction: shipped p = 3.53 × 10⁻¹², label p = 1.06 × 10⁻⁶ (`txm_width_ratio_unified.json`, `.shipped_vs_one_p`, `.label_vs_one_p`). The shipped mask is wider than the feature on **0 of 64 frames**, its largest ratio anywhere being 0.966 (*derived*, max of `.per_frame[].ratio_shipped`); the brush is wider than the feature on 11 of 61.

This retires the over-marking framing that motivated the width work in the first place. The detector does not over-mark this corpus. It under-marks it, by very nearly the factor the human annotator does.

> **Footnote on the other two estimators.** Two earlier measurements of the same quantity survive in the artifact set and disagree, because they sample differently: `txm_width_clip_final.json` (`.width_ratio_after`) gives 0.558 over the 63 frames that carry a ratio, with 1 frame still over-marking, and `txm_width_reference.json` (`.what_it_says_about_overmarking`) gives 0.679 over 61 frames with 7 over-marking, each mask there being scored on its own skeleton. The self-skeleton figure is the better *absolute* estimate of how wide the shipped mask is; the shared-skeleton figure above is the only one valid for comparing masks to each other, and is the one this paper uses throughout. The seven frames that over-mark under the self-skeleton estimator are all hairline-feature frames (e.g. `HC_316L_fatigue_600`, FWHM 7 px against a 36 px mask), and the mechanism is documented: `MIN_BLOB_PX = 2000` means a component thinner than roughly 2000/length px cannot survive the size filter, so the only hairlines that reach the export are ones the model drew fat (`txm_width_prereg_and_failed_variants.md`).

### IoU charges the detector for exactly this agreement

Against these annotations, IoU in the under-marking regime is, for a correctly centred mask, very nearly a width ratio and almost nothing else. Dilating the operator's own painted centreline to a fixed width and scoring it against the label itself gives IoU 0.0498 at 3 px, 0.0822 at 5 px, 0.1754 at 11 px and 0.3140 at 21 px (`txm_iou_ceiling.json`, `.headline`), against a median label width of 73.4 px (*derived*, median over the 61 records of `txm_iou_ceiling_per_frame.json`, `.label_width_px`; the headline field rounds this to 73.0). Per pixel of marked width that is 0.0166, 0.0164, 0.0159 and 0.0150 (*derived*) — flat to within 11% across a sevenfold change in width. A correctly centred mask therefore scores, to first order, its own width divided by the brush's, and a method that matches the feature rather than the brush is penalised in exact proportion to how much narrower the feature is. The qualifier "correctly centred" is load-bearing and is the artifact's own: a mask that is the right width in the wrong place is not rescued by this argument, and on a 73 px brush the label skeleton can itself sit far from the crack.

The clip step supplies a direct, paired demonstration on one run and one frame set. Clipping the mask to its measured width moved the corpus width ratio from 0.610 to 0.558 and reduced over-marking frames from 10 to 1 (all ten came down; `HC_316L_fatigue_1250` remains marginally over at 1.06). The metrics paid for it: IoU 0.7334 → **0.6748**, clDice 0.8462 → **0.8234** (paired Δ −0.0168), Tsens 0.7978 → **0.7605** (paired Δ −0.0260); only Tprec rose, 0.9687 → 0.9703 (`txm_width_clip_final.json`). Making the mask more nearly the width of the thing it sits on cost 5.9 IoU points.

Two honesty items attach to this step, both already recorded in the project's own notes and neither of which should be read past. **Two of the four criteria fixed before the clip run failed, and the step ships on by default**: criterion (a) allowed a Tsens drop of 0.02 and the observed drop was 0.026, missing by 0.006; criterion (b), median |ratio − 1| must fall, went 0.399 → 0.442 and is additionally mis-specified, since a clip-only operator can only move frames downward and therefore cannot improve a population statistic on a corpus whose median is already 0.61 (`docs/WIDTH_REFERENCE.md`). Five of the six worst Tsens losses are over-marking frames the step exists to fix, where the brush centreline ran through the excess width; the sixth, `b3_385_63um_ZOOM` (Tsens 1.000 → 0.820 at ratio 0.40×), was already well under-marked and should have been left alone. That one is a genuine cost, not a metric artefact.

Two cross-artifact cautions. First, `txm_iou_ceiling.json` compares its ceiling against a model IoU of 0.5047 and reports a "10.1× the ceiling" multiple; that comparator is a **pre-clip mask** (its per-frame values are byte-identical on all 61 frames to `cldice_centreline_per_frame.json`, the run whose headline clDice 0.863 / Tprec 0.955 / Tsens 0.814 the clip step superseded), so the multiple is not quoted here. The ceiling itself is a property of the labels alone and is unaffected by any model version. Second, the 73.4 px median label width above and the 0.529 label ratio above come from **different width estimators and must not be divided into one another**: on the same 61 frames, the ceiling artifact's per-frame label width is a median of 1.57× the on-skeleton label width used by the profile estimator (*derived* from `txm_iou_ceiling_per_frame.json`, `.label_width_px` against `txm_width_reference.json`, `.per_frame[].w_lab`). Both instruments are internally consistent; nothing in `out/` reconciles them, and this paper keeps each to its own comparison.

### What neither the brush nor a half-maximum cut can adjudicate

A narrower mask than the shipped one is defensible only if the material being removed is not crack. That question was pre-registered before it was run, with a fixed decision rule, because it has two opposite answers with opposite consequences: either narrowing deletes crack, or the brush midline runs over plain matrix and the narrowing correctly declines to follow it (`txm_width_prereg_and_failed_variants.md`).

The test takes the label-centreline pixels the shipped mask covers and an aggressive narrowing (FRAC 0.4) drops, and measures their image contrast against the kept centreline and against a matrix reference from unlabelled, unmasked pixels on the same frame. Over the 57 frames with a real disagreement: matrix −0.0016, **dropped +0.0379 (1.63σ above matrix)**, kept +0.0969 (4.83σ), pre-registered midpoint +0.0502. Dropped fell below the midpoint on 33/57 frames, sign test **p = 0.145** against a required p < 0.05. A paired Wilcoxon gives p = 0.016, but the sign test is what was pre-registered and it fails; the pre-registered verdict is therefore that the deleted material is dark and narrowing further is not licensed.

That verdict is the one this paper reports, and it leaves the substantive question open. The dropped halo sits at 39% of the kept-centreline contrast (*derived*, 0.0379/0.0969; the source note asserts 42%, which its own medians do not reproduce, so the derived figure is used) and 1.63σ above background: it is not matrix, but whether it is faint crack, a partly closed flank, out-of-plane crack, or the instrument's own skirt is not decidable from these images plus these annotations. The brush cannot adjudicate it — a stroke whose per-specimen mean width runs from 36 px on AM to 126 px on B3, corpus median 73.4 px (`txm_iou_ceiling.json`, `.per_specimen.*.label_width_px`), asserts a region, not a boundary — and a half-maximum cut cannot adjudicate it either, since the halo's existence is what the half-maximum is being asked about. **[TODO-AUTHOR: settle this with a modality that carries a depth axis, or with a repeat acquisition of one frame at higher magnification; state which if either is available.]**

One further pre-registered variant failed and is reported as a failure rather than dropped: a per-component FWHM at FRAC 0.5 took Tsens from 0.798 to 0.261 and clDice from 0.846 to 0.412, 23× the allowed drop. The two causes were both in the per-component statistic — one `peak` percentile taken over a whole component that runs from near-black open crack to faint hairline, and a background estimator that silently falls back to a box mean including the mask inside components wider than the 301 px window. Replacing it with a local half-maximum degrades smoothly, but the only setting that clears the pre-registered guard is FRAC 0.2, a 6% width reduction (`txm_width_prereg_and_failed_variants.md`).

### A straight-line artefact class: elongation and contrast are the crack signature, and also the artefact's

The intuitive signature of a crack in these frames is a long, thin, dark, high-contrast feature. That is also, exactly, the signature of a detector-column artefact. Elongation and contrast cannot tell them apart; deviation of the centreline from a straight-line fit can.

A census of all standalone components ≥200 px with bounding-box aspect ratio ≥6 across all 71 frames returns six components (`txm_small_component_adjudication.json`, `.straight_line_artefacts.census`). Five of them wander **0.13, 0.18, 0.34, 0.61 and 0.71 px** about a fitted line, at elongations of 134, 197, 175, 94 and 38 and sizes of 2,090–4,094 px — all of them above the `MIN_BLOB_PX = 2000` size floor, though the smallest clears it by only 90 px, so none was filtered out as a speck. The sixth wanders 1.73 px, has a transverse width standard deviation of 6.39 px against 0.37–2.19 for the five, and is at least 55.9% painted crack (*derived*, the 1,360 px of painted crack removed from it at the abandoned 2.0 px cut against its 2,433 px total, `.census.components[1]` and `.guard.cut_history`). Across the census the median wander is **0.4736 px for the straight population and 5.2511 px for the curved population** (`.census.median_wander_straight`, `.median_wander_curved`).

Three of the five artefacts are strong geometric evidence for their own origin: two sit in one frame at x = 6353 and x = 6359, and a third sits in a sibling frame of the same specimen at x = 6366 — three near-identical vertical features within 13 px of the same image column, across two frames. A fourth sits at x = 0; the fifth is interior, at x = 1,689 (`.census.components[].x`).

That elongation and contrast do not separate these from crack is not only an argument from statistics. Two of the five reached the blind 3-panel adjudication as test fields and were called opposite ways: the elongation-134 component was called crack 2 of 3, the elongation-197 component not-crack 0 of 3 (*derived*, matching `.per_field` entries `site_007` and `site_083` to `.census.components` on identical frame id, pixel count and elongation). Those panels are language-model agents, not human 316L microscopists, and their split here is reported as an illustration of how the intuitive signature behaves, not as an adjudication of these components.

**A gap in the evidence that must be closed before submission.** `docs/UNLABELLED_AREA.md` states the separation as "genuine long crack, 6 components, 2.9–12.9 px" of wander. That range is **not in any artifact in `out/`** — the census enumerates the straight components and the one painted crack, but not the six curved ones, and carries only the curved median. The claim as published here is therefore restricted to the two medians (0.4736 vs 5.2511) and the five enumerated artefact values (0.13–0.71 px). **[TODO-AUTHOR: re-emit the census with the curved components listed individually, or the 2.9–12.9 range must stay out of the paper.]** A separate five-component range of "0.13–0.93 px over 171–788 px" appears in the same artifact's `finding` field; it refers to a different five — those inside a 32-component tractable subset, not the 71-frame census — and the two ranges must not be merged.

The guard derived from this is deliberately narrow: drop a component only if it is ≥200 px, aspect ≥6, and wanders <1.0 px about a line fit; painted pixels are exempt. The cut was first set at 2.0 px, which removed 1,360 px of painted crack on `b2_341_88_take2` — the wander-1.73 component — and since "must not remove painted crack" was fixed before the run, the cut moved to where the population actually separates (`.guard.cut_history`). An earlier note claiming an 11× separation with nothing in between was read off group medians and was wrong.

Verified over all 71 frames, the guard fires on 4 frames and removes **15,241 px, 0.0469% of accepted area** (*derived*, 15,241 against the 32,504,070 px accepted total in `txm_unlabelled_area_per_frame.json`, `.accepted_px`; that total is over the 65 frames that artifact covers while the guard was verified over 71, and the census artifact's own `.pct_of_accepted` uses a marginally larger denominator — neither choice changes the figure at this precision), of which **0 px are painted crack**. clDice is unchanged to sixteen decimal places, 0.8234071528920338 before and after. Predicted area on the crack-free specimens falls from 0.0253% to 0.0236%, a 6.7% relative reduction, and the whole of that reduction comes from a single component: of the four frames the guard fires on, only `b3_amb` is among the six specimens the operator confirmed crack-free (`CLEAN_SPECIMENS`, `app/core/pipeline.py:102`), and its predicted area goes from 0.000169 to exactly zero (`.guard.verified`). One component on a specimen asserted to contain no crack was a 655 px-long vertical line wandering 0.61 px.

Three caveats. The artifact's own top-level keys `.census.artefacts` = 6, `.artefact_px` = 17,674 and `.pct_of_accepted` = 0.0543% **include the wander-1.73 painted crack** and are mislabelled; the artefact-only figures are 5, 15,241 px and 0.0469%, as used above. The guard is a **lower bound**, because it is component-level: two further artefacts fused into 440k and 228k px crack systems ride through attached to real crack, and no within-component test exists (`.guard.lower_bound`). And the census run itself produced three checks that reported success while measuring nothing — a NumPy 2.0 `ptp` failure swallowed per-frame, an `effective_mask` compared against itself, and a duplicated empty-result fallback (`.checks_that_reported_success_while_measuring_nothing`); these are listed in full in the Limitations section rather than buried, because an audit paper that does not print its own false passes is not an audit.

### Limitations of the width measurement

- **Estimator dependence is the dominant uncertainty.** Three estimators of "the shipped mask's width ratio" give 0.505, 0.558 and 0.679, and three counts of over-marking frames give 0 of 64, 1 of 63 and 7 of 61. Every number in this subsection is from the shared-skeleton estimator; mixing them would change the qualitative conclusion about over-marking.
- **The measurement is conditioned on measurable features.** A profile enters only if its peak clears 2 background σ. On AM / HC_316L — 38% of the corpus — cracks are intensity-invisible, with Cohen d = +0.09 against +1.10 to +2.91 on the other three groups, measured against each crack's own local ring (`am_label_separability.json`, `.CORRECTION_ring_control.result`; the same file's top-level d = −0.31 for AM is explicitly **withdrawn**, being an artefact of using the operator's painted not-crack as reference). The AM width figures therefore rest on the subset of AM locations bright enough to profile and are not representative of AM crack as a whole. The two profile counts in `txm_width_reference.json` (11,426 for the instrument test, 15,397 for the ratio table, both over 61 frames) reflect different gating and are not interchangeable.
- **No absolute width claim is made.** All ratios are mask width against image FWHM at shared sample points; FWHM is itself a convention (ISO50), not a physical crack-opening displacement.
- **No second annotator, and no repeat annotation** — see above.
- **Acquisition metadata is absent from both repositories.** **[TODO-AUTHOR: beam energy, microscope and detector, exposure, and reconstruction method; the loading protocol, rig and load levels behind the `_lbf` filenames; cycle counting for the fatigue sets; specimen provenance, heat treatment, AM build parameters, and through-thickness geometry.]** None of these can be inferred from the artifacts and none is stated here.
```

VERIFICATION REPORT — 11 defects found, all corrected below.

**Numbers attached to the wrong object**
1. §6.1 "On the worst single frame, `b2_336_25`, 94.65%" — FALSE. `b2_336_25` is third. The worst is `HC_316L_fatigue_1770_tip_zoom` at 97.27%, then `HC_316L_fatigue_1790_tip_zoom_2` at 95.01%. Corrected, and both AM frames named.
2. §7.3 "seven fields of view ... 15 of the 71 frames" — not reproducible. Under the draft's own suffix list the answer is six fields / 13 frames; the widest defensible reading (adding `_1790_cycles`/`_1790_tip_zoom_2` and `_1100_cycles`/`_1100_cycles_crack`) is eight / 17. Neither is 7/15. Rewritten with both counts.

**Wrong key / wrong file**
3. §6.3 cites `.rows.shipped.n_over`; the key is `.rows.shipped.over_marking`. Also 0.505 is `.rows.shipped.median` (64 frames), not the paired figure `.shipped_vs_label.shipped` = 0.5061 (61 frames) — both now named.
4. §6.1 cites `txm_iou_ceiling.json` for `label_width_px`; that key lives in `txm_iou_ceiling_per_frame.json` (median 73.430).

**Claim stronger than evidence**
5. §6.4 "A self-test now asserts all three properties including their ordering" — OVERCLAIM. `app/selftest.py:977-1006` asserts the ordering and the guard behaviour, and `:963` covers the both-arms-guarded case. Nothing asserts the NumPy-2.0 `ptp` census property (no `ptp` in `selftest.py`). Downgraded to two of three.
6. §6.2 states the nickel 0.995 "is background-inclusive" as fact; `model_vs_literature.json` states it as the only inference compatible with MCC 0.826. Softened to match.
7. §8 "Four load-bearing artifacts carry no `generator` key and their drivers are not in `code/audit/`" — half wrong. The no-generator part is confirmed for all four, but `code/audit/measure_iou_ceiling.py` DOES exist; it emits `txm_iou_ceiling_v2.json`, which has not been run. Corrected to three-without-drivers plus one written-but-unrun.
8. §8 "the fixed 700 × 700 px native window" applied to both crop keys — the small-component run used 400 px (`build_small_component_crops.py:35`, `WIN = 400`), and its own artifact says "400 px fields so a 200 px clearance means no painted crack can appear in frame". Split.

**Unsourced quantities**
9. §6.4 "median |ratio − 1| 0.399 → 0.442" exists in no artifact — only `docs/WIDTH_REFERENCE.md:165`. Now cited as such and flagged.
10. §7.5 "55.9 % painted crack" exists only at `docs/UNLABELLED_AREA.md:206`. Cited. §7.5's pixel-size derivation had no citation; `circ2_scale_and_units.json` supports it and is now named.
11. §6.4 conflated two different sets of four: the four pre-registered *tests* in `txm_width_prereg_and_failed_variants.md` and the four *criteria* (a)–(d) in `docs/WIDTH_REFERENCE.md`. Disentangled.

Verified correct and left alone: 24.7%/0.2466, all four group fractions, 32,504,070 / 8,016,109, IoU ceiling 0.0498 and the 5/11/21 px rungs, 25/26 and 93.4%, all four *derived* Fisher p-values (1.000, 2.22e-09), the one-sided/two-sided forensics in §7.1 (I recomputed Mann–Whitney from `per_field`: `.instrument_validation.mannwhitney_p` is indeed one-sided, two-sided 4.5017e-08; `.result.vs_POS_p` and `.vs_NEGM_p` are indeed two-sided), 13/32 vs 13/32 and CI [0.321, 1.000], 0.505/0.529/p=0.061/0 of 64, Tsens −0.026, clDice 0.8462→0.8234, 10→1, `b3_385_63um_ZOOM` 1.000→0.820 at 0.40×, 15,241 px / 5 components / 4 frames / 0 painted removed, 0.229%, 12,583 / 284 / 12.2% / 6,921 / 80.4%, 95,351,647 px and 25.4%, 0.023588 (I confirmed the unit: it is a percentage, cross-checked against `.github/workflows/linux.yml`'s "0.000% to 0.144%" per-frame range), Cohen's d +0.09 vs +1.10/+2.91/+1.57 (the corrected ring control, not the withdrawn figure), all §7.3 cross-validation numbers, all literature numbers, FWHM 45.0 px / 11,426 / CV 0.916 / rho +0.673, every `pipeline.py` line reference, and all 71 corrections are in fact tracked (in `paint/corrections/corrections.npz`).

No DEAD claim crept back in: no clDice 0.863 / Tprec 0.955 / Tsens 0.814, no corpus-wide over-marking, no temporal claim, no width-method novelty.

---

## 5. Discussion

### 6.1 Report what fraction of the prediction the evaluation could see

Every accuracy number on this corpus — IoU, clDice, topological precision and sensitivity — is computed on pixels the annotator touched. That is not a property of our metrics; it is a property of every metric in crack segmentation, because an unpainted pixel supplies no target to score against. The consequence is measurable and, on this corpus, large: **24.7 % of accepted mask area (0.2466, area-weighted over 65 frames; `txm_unlabelled_area_per_frame.json` `.corpus_frac_unlabelled`) was never labelled either way**, so 75.3 % of what the detector marked is all the evaluation could see (*derived*, 1 − `.corpus_frac_unlabelled`). Per specimen group the evaluated share runs from 95.5 % on B3 down to **48.9 % on AM** (*derived* from `.per_group.<G>.frac_unlabelled` = 0.0449 / 0.1067 / 0.3684 / 0.5106 for B3 / WR / B2 / AM, n = 13 / 12 / 16 / 24). On the three worst single frames — `HC_316L_fatigue_1770_tip_zoom` (97.27 %), `HC_316L_fatigue_1790_tip_zoom_2` (95.01 %) and `b2_336_25` (94.65 %), per `.per_frame[].frac_unlabelled` — a per-frame IoU is a statement about three to five per cent of what the model did there. Two of those three are AM crack-tip zooms.

We propose that papers reporting operator-trained crack segmentation carry a short, fixed disclosure alongside the headline metric. It requires no new annotation and no new instrument — only the prediction and the two label channels the project already has:

**E1 — Evaluated fraction.** The share of predicted positive area lying inside the annotated domain (painted crack ∪ painted not-crack), area-weighted, with the per-group breakdown and the denominators. Here: 32,504,070 accepted px, 8,016,109 of them unlabelled (`txm_unlabelled_area_per_frame.json` `.accepted_px`, `.unlabelled_px`).

**E2 — Label-geometry ceiling.** What a physically correct instance would score under the headline metric against these particular labels. Here a skeletonised-and-redilated 3 px trace of the operator's own stroke scores **IoU 0.0498** (`txm_iou_ceiling.json` `.headline.perfect_3px_trace_IoU`, n = 61 labelled frames), rising to 0.0822 / 0.1754 / 0.3140 at 5 / 11 / 21 px, against a median label width of 73.4 px (*derived* median of `label_width_px` over the 61 rows of `txm_iou_ceiling_per_frame.json` = 73.430; the summary artifact rounds to 73.0). E2 is a property of the annotation, not of any method, and cannot be improved by a better detector.

**E3 — Disposition of the unevaluated fraction.** What an independent read says the unseen area actually is. Here, blind three-panel adjudication of components ≥ 4000 px — which hold 87.9 % of unlabelled area (`txm_unlabelled_area_per_frame.json` `.frac_unlab_in_components_ge_4000px` = 0.8788) and therefore cover 21.7 % of all accepted area (*derived*) — called it crack in **25/26 fields** (`txm_unlabelled_adjudication.json` `.result.UNLAB_majority_yes`), 93.4 % area-weighted (`.result.area_weighted_pct`), indistinguishable from painted crack (two-sided Mann–Whitney on the vote score p = 0.313, `.result.vs_POS_p`; Fisher on the majority counts p = 1.000, *derived*) and separated from plain matrix (two-sided Mann–Whitney p = 4.44 × 10⁻⁸, `.result.vs_NEGM_p`; Fisher p = 2.22 × 10⁻⁹, *derived*). The single rejection is recorded rather than absorbed: field 085 on `wrought_316L_fatigue_1280_cycles`, 27,179 px, read as a specimen free surface with Fresnel fringes and metal on one side only — a real false positive, and the only one in 26 (`.the_one_rejection`).

E1 alone is not a performance metric and must not be read as one. A high evaluated fraction can mean the annotator was exhaustive or it can mean the detector only fires where the brush went; a low one can mean the detector is finding unannotated crack (this corpus's answer, per E3) or that it is hallucinating. The point of the disclosure is that at present the reader cannot tell which, because the quantity is not reported at all. E1 without E3 states the size of the unknown; E1 with E3 resolves it — subject, here, to §7.1.

### 6.2 What a ceiling does and does not license

E2 changes how an IoU should be read, not whether the detector works. An IoU of 0.5 on labels whose median width is 73 px is not five times better than a physically correct answer at 0.0498 in any sense a materials reader cares about — it is a measure of agreement with brush geometry. We therefore do not quote a "multiple of the ceiling" anywhere in this paper. The multiple recorded in `txm_iou_ceiling.json` (`.headline.model_over_ceiling` = 10.1, against `.headline.deployed_model_IoU_same_frames` = 0.5047) is computed against the **pre-`clip_to_measured_width`** mask and not the mask this paper describes: its per-frame values are byte-identical to `cldice_centreline_per_frame.json`, run at 02:05 on 2026-09-19, before the width clip shipped at 14:14 the same day (`code/audit/measure_iou_ceiling.py`, docstring). The same applies to every `.per_specimen.<G>.model_iou` in that file. The ceiling itself is unaffected, because it is derived from the labels alone. A recomputation that puts both sides through one IoU convention is written (`measure_iou_ceiling.py`, emitting `txm_iou_ceiling_v2.json`) and has **not been run**; until it is, no model-over-ceiling ratio should be quoted from this corpus at all.

The corollary for the field is that crack-segmentation IoU is not portable between papers. Genuinely crack-class comparables — Segment-Any-Crack at IoU 0.4413 / F1 0.6122 on OmniCrack30k, MixSegNet at 0.848 / 0.915 on surface-crack photography (`model_vs_literature.json` `.genuinely_comparable_crack_class_numbers.*`) — were scored against annotations with their own, undisclosed, brush geometry; and two figures widely cited as crack IoU are not crack IoU at all. A nickel-superalloy U-Net reports 0.995 alongside MCC 0.826 on the same runs, a pairing possible only if the IoU is background-inclusive; a shale FRRN-B's 85 % is a three-phase mean over background, matrix and crack, and the crack-class value is not published (both per `model_vs_literature.json` `.the_two_numbers_that_look_like_they_beat_us_and_why_they_do_not.*`). Reporting E2 makes such comparisons refusable on stated grounds instead of tacitly wrong.

### 6.3 Instruments that do not pass through the annotations

If the annotations bound what agreement metrics can show, at least one instrument in the evaluation should not read them. The transverse-profile width ratio is one such: it scores a mask against the image. Measured with one estimator, one frame set, and the same sample points for every variant (`txm_width_ratio_unified.json`, n = 64), the shipped detector's median width ratio is **0.505** (`.rows.shipped.median`) and the human brush label's is 0.529 (`.rows.label.median`, n = 61). On the 61 frames carrying both, the paired comparison is shipped 0.506 against label 0.529, median difference −0.011, Wilcoxon p = 0.061, verdict "indistinguishable" (`.shipped_vs_label.*`). Both under-mark the dark feature (shipped vs 1.0, p = 3.5 × 10⁻¹², `.shipped_vs_one_p`; label vs 1.0, p = 1.1 × 10⁻⁶, `.label_vs_one_p`), and the shipped mask is wider than the feature on 0 of 64 frames (`.rows.shipped.over_marking`). These are a fair comparison *across masks*, not absolute width estimates, because every variant is scored at the shared skeleton; we do not claim the detector is half as wide as the crack. The negative-control axis — predicted area on specimens confirmed to contain no crack — is the other such instrument, and §7.2 is why ours is currently not one.

### 6.4 Report the decision rules that failed

An audit paper that only prints its passing tests is the failure it is auditing. Two separate pre-registrations are involved here and we keep them apart.

First, four tests of the width-narrowing candidate were pre-registered with their decision rules fixed before the data were read (`txm_width_prereg_and_failed_variants.md`). Two returned failures and are reported as failures: the per-component FWHM variant missed its Tsens allowance by 23× (paired median −0.462 against −0.020; Tsens 0.798 → 0.261, clDice 0.846 → 0.412) and was rejected outright, and the adjudication test of what the narrowing deletes failed its pre-registered sign test (dropped centreline below the midpoint on 33/57 frames, p = 0.145 against a required p < 0.05), returning verdict (A) — the dropped material is not matrix and an aggressive cut deletes real signal. The other two settled: the local-FWHM replacement passes only at FRAC 0.2, a 6 % width reduction, and the instrument test returned "width is real" (§7.4).

Second, four acceptance criteria (a)–(d) were fixed before the width-clip run and are scored in `docs/WIDTH_REFERENCE.md:157–165`. **(a) failed on Tsens by 0.006** (paired −0.026 against an allowance of −0.020; `txm_width_clip_final.json` `.Tsens_paired_delta`; clDice passed at −0.017, `.clDice_paired_delta`) and **(b) failed outright** (median |ratio − 1| 0.399 → 0.442 — a figure that exists in `docs/WIDTH_REFERENCE.md:165` and **in no artifact**) and is additionally mis-specified, since no clip-only operator can improve a population statistic on a corpus whose median ratio is already 0.61× (`.width_ratio_before` = 0.6096). (c) and (d) passed. **The step nonetheless ships on by default**, trading clDice 0.8462 → 0.8234 (`.clDice_before`, `.clDice_after`) for the removal of every over-marking frame but one (10 → 1, `.overmarking_frames_before/after`). One frame, `b3_385_63um_ZOOM` (Tsens 1.000 → 0.820 at ratio 0.40× → 0.37×, `.per_frame[]`), was already under-marked and the step should have left it alone; that is a genuine cost, not a metric artefact.

Three checks in this work reported success while measuring nothing, and are printed rather than buried (`txm_small_component_adjudication.json` `.checks_that_reported_success_while_measuring_nothing`): a NumPy-2.0 `ptp` removal crashed the straightness census on exactly the 10 frames containing long components while the summary averaged the survivors and printed 0.00 %; the first guard verification compared `effective_mask` against `effective_mask`, so both arms were guarded and the test could not fail; and a duplicated empty-result fallback let `b3_amb`, a specimen asserted crack-free, ship its false positive through a guard reporting PASS. The self-test now covers two of the three — the fallback/guard ordering and the guard's behaviour on synthetic straight and wandering components (`app/selftest.py:977–1006`), and separately that width clipping only ever removes and is not inert (`:963`). **No self-test asserts the census property**; nothing in `app/selftest.py` exercises the `ptp` path, so that failure mode is recorded but not yet guarded. We print all three because the pattern recurred, and because a reader has no way to distinguish a check that passed from a check that ran.

---

## 6. Limitations

### 7.1 The adjudication panels are language-model agents, not human experts

The blind adjudication that resolves the unlabelled area (§6.1, E3) was voted by language-model agents — 3 panels, 3 votes per field, 33 agents, 0 errors (`txm_unlabelled_adjudication.json` `.design.panels`, `.design.votes`) — with **no human expert arm**. The instrument validation shows the panels recover the annotator's own painted crack at 24/26 against 4/26 on plain matrix (`.instrument_validation`; the stored `fisher_p` = 1.16 × 10⁻⁸ and `mannwhitney_p` = 2.25 × 10⁻⁸ are **one-sided**, two-sided 2.31 × 10⁻⁸ and 4.50 × 10⁻⁸, *derived* and recomputed from `.per_field[].s`). That establishes agreement with this annotator's strokes. It does not establish competence at identifying fatigue crack in 316L TXM, which is a different claim and one we have not tested. A materials venue should require at least two blinded human readers through the same protocol before the E3 result is treated as settled; the crops, crop key, fixed seed, leak guard and deterministic scorer are all preserved (§8) so that run requires readers, not rebuilding.

Panel sensitivity is additionally low at small component size: on known painted crack the panels returned only 13/32 (`txm_small_component_adjudication_v2.json` `.by_class.POSS.majority_yes`), which caps what the small-component arm can settle — the test class matched it exactly at 13/32, Fisher p = 1.000 (`.test_vs_control_fisher_p`), separating from matrix at 3/26 (`.by_class.NEGM.majority_yes`; Fisher p = 0.0134 **one-sided**, `.test_vs_matrix_fisher_p`, two-sided 0.0185, *derived*; two-sided Mann–Whitney on the vote score p = 2.06 × 10⁻⁴, `.test.vs_matrix_p`). The point estimate `prevalence` = 1.0 in that artifact is an arithmetic artefact of the observed rate landing exactly on the sensitivity (0.40625 = 0.40625) and must never be quoted without its bootstrap CI of **[0.321, 1.000]** (`.ci`) and that sentence. The crack-free arms are underpowered by design and prove little alone: n = 2 (NEGF, excluded from the analysis, `txm_unlabelled_adjudication.json` `.design.classes.NEGF`) and n = 4 (CFREE, 0/4, `txm_small_component_adjudication_v2.json` `.by_class.CFREE`).

### 7.2 The crack-free specimens are in the training set

The six specimens confirmed to contain no crack (`CLEAN_SPECIMENS`, `app/core/pipeline.py:102`: `b3_amb`, `B2_amb_mosaic_2`, `B2_2_1_lbf`, `B2_2_9_lbf`, `b3_3_18lbf`, `wrought_316L_fatigue_0_cycles`) are the project's only check on over-prediction that needs no pixel ground truth — and their correction masks are in training. They contribute 0 crack pixels and **95,351,647 not-crack pixels, 25.4 % of every background label the model is trained on** (`code/audit/heldout_crackfree_fp.py`, docstring; this figure exists in that source file alone and in no artifact). Predicted area on those frames is therefore measured on pixels the model was explicitly told are background, and the shipped figure — **0.0236 % of crack-free area** on the mask this paper describes (`txm_small_component_adjudication.json` `.straight_line_artefacts.guard.verified.crackfree_fp_after` = 0.023588, stored without a unit and read as a percentage; 0.0253 % before the straight-line guard) — is partly a memorisation result. Older figures of 0.174 %, 0.0230 %, 0.0209 %, 0.0197 % and 0.0192 % circulate in this project's own documentation (`README.md:296`, `docs/START_HERE.md:60`, `docs/OVERMARKING.md:181,256`, `docs/TILE_SEAMS.md:169`, `app/core/pipeline.py:1072`) on earlier mask stages; only the 0.0236 % figure is on the shipped mask, and it is the only one we report.

The held-out measurement is specified and pending: `code/audit/heldout_crackfree_fp.py` refits the deployed architecture on the same rows with every crack-free frame removed and re-scores, with the incumbent re-scored through the identical path so the arms differ only in training data. **[TODO-AUTHOR]** run it and insert: with the six frames held out, mean false-positive area on crack-free material is [`trained_WITHOUT_clean.mean_fp_area`] against [`trained_WITH_clean.mean_fp_area`] for the same architecture trained with them, an inflation factor of [`inflation_factor`] (`txm_heldout_crackfree_fp.json`, not yet emitted). That script refits the 17-feature branch only, in both arms and deliberately, rather than the deployed ensemble; the comparison is internally fair but is not a measurement of the deployed ensemble.

Two further weaknesses of this axis are structural rather than fixable by that run. The six crack-free frames are not independent specimens: they are three B2, two B3 and one Wrought frame at ambient or zero load, drawn from the same specimen groups that supply the cracked frames. And **AM / HC_316L contributes no crack-free control at all** — the group that is 27 of 71 frames, carries the highest unlabelled fraction at 51.1 %, and is the group whose cracks are intensity-invisible against their own immediate surroundings (Cohen's d = +0.09 against +1.10 / +2.91 / +1.57 for B2 / B3 / WR, `am_label_separability.json` `.CORRECTION_ring_control.result.*.d_vs_local_ring`; the earlier painted-not-crack reference is withdrawn in that same artifact). The false-positive axis is silent on precisely the hardest third of the corpus.

### 7.3 n = 71 frames from four specimen groups, which are not independent

The corpus is 71 frames distributed AM / HC_316L 27, B2 17, Wrought 14, B3 13 (`txm_width_clip_final.json` `.per_frame[].group`), of which 65 carry an accepted mask (`txm_unlabelled_area_per_frame.json` `.n_frames`) and 61 carry labels (`txm_width_clip_final.json` `.n_labelled`). Within a group the frames are a sequence on one specimen — AM is a single fatigue series from 200 to 1790 cycles, Wrought from 0 to 1300 cycles, B2 and B3 are load/displacement series — so the effective sample size for any specimen-level statement is closer to **four than to seventy-one**. Frames within a group share a specimen, a mounting, an imaging session and an annotator session, and none of those are separable in the data we hold.

The dependence is tighter still in places. At least **six fields of view appear more than once** under a `_tip`, `_ZOOM`, `_LARGE` or `_4` suffix, accounting for **13 of the 71 frames** (*derived* from `.per_frame[].id`: `HC_316L_fatigue_1400_cycles`, `_1450_cycles` and `_1650_cycles` each with a `_tip` partner; `b2_343_75` with `b2_343_75_LARGE`; `b3_380_00um` appearing three times as itself, `_4` and `_ZOOM`; `b3_388_13um` as `_ZOOM` and `_LARGE_2`). Counting two further near-duplicate pairs whose suffixes differ in form — `HC_316L_fatigue_1790_cycles` with `_1790_tip_zoom_2`, and `wrought_316L_fatigue_1100_cycles` with `_1100_cycles_crack` — takes it to eight fields of view and 17 frames. Cross-validation in this project groups by **image**, not by specimen (`ensemble_vs_hybrid_by_specimen.json` `.raw.*.grouped_by`), so folds described as held-out share specimens; its own verdict text is explicit about the size (non-AM 44 images, ensemble 0.7850 vs hybrid 0.7696, 5/5 folds, paired t p = 0.049; AM 27 images, 0.7447 vs 0.7483, 1/5 folds, p = 0.321). The blind panels inherit the same structure — the 88 large fields were drawn from 20 / 23 / 24 source frames by class (`txm_unlabelled_adjudication.json` `.design.leak_guard.source_frames`), and the small-component run's leak guard has a minimum p of 0.057 (`txm_small_component_adjudication_v2.json` `.matching.leak_guard_min_p`), which is a pass but not a comfortable one. No confidence interval in this paper should be read as a population interval over 316L specimens.

There is one annotator, and no frame is known to have been labelled twice, so no inter- or intra-operator uncertainty can be estimated **[TODO-AUTHOR]**. Because the labels are the reference for every agreement metric and the target of the audit, that is a limitation on both sides of the comparison at once.

### 7.4 The width instrument is standard metrology, not a new measurement

The transverse-FWHM width measurement is not novel and we do not claim it. Sampling intensity along normals to the crack centreline is **Orthogonal Profile Extraction**, an established automated crack-width algorithm; taking the edge at half the profile's maximum is **ISO50**, the default surface-determination rule in X-ray CT dimensional metrology codified in VDI/VDE 2630; and the requirement that width be taken perpendicular, which is why the minimum over 12 directions is kept, is textbook (prior art checked 2026-09-21, `docs/WIDTH_REFERENCE.md:28–60`). Our variant takes a local half-maximum against a locally estimated background rather than a global histogram threshold, which the same section records as a routine adaptation. What is ours is the **use**: pointing a standard metrology rule at the annotations rather than at the specimen, and asking whether the mask is as wide as the feature it sits on.

Even that use has a scoping caveat. Three different "shipped width ratio" figures exist in this project's artifacts because three estimators were built: **0.505** (shared skeleton, 64 frames, `txm_width_ratio_unified.json` `.rows.shipped.median`), 0.558 (`txm_width_clip_final.json` `.width_ratio_after`, 71 frames of which 63 yield a ratio) and 0.679 (per-mask self-skeleton, 61 frames, `txm_width_reference.json` `.what_it_says_about_overmarking.shipped_ratio_median`), with over-marking frame counts of 0/64, 1/63 and 7/61 respectively. This paper uses `txm_width_ratio_unified.json` throughout — it is the one constructed so that no variant is scored at sample points chosen by its own geometry — and the other two appear only here. That three incompatible numbers for one quantity arose inside a project whose subject is measurement hygiene is itself a result worth stating.

Separately, the width the instrument measures is physical rather than a point-spread floor: median transverse FWHM 45.0 px over 11,426 profiles on 61 frames, 10th–90th percentile 3–147 px, coefficient of variation 0.916 against a pre-registered instrument-limited threshold of < 0.35, Spearman FWHM vs peak contrast +0.673 against a threshold of < 0.3 (`txm_width_reference.json` `.is_it_the_instrument.*`). Both pre-registered thresholds were missed by a wide margin, in the direction that says the variation is real.

### 7.5 Residual limitations

- **The straight-line artefact guard is a lower bound.** It removes 5 standalone components totalling 15,241 px, 0.0469 % of accepted area (*derived*, 15,241 / 32,504,070). It fires on 4 of 71 frames, removes **0 px of painted crack**, leaves clDice unchanged to ten decimals, and takes `b3_amb` — a specimen asserted crack-free — to exactly zero predicted area (`txm_small_component_adjudication.json` `.straight_line_artefacts.guard.verified`). The wider census in the same artifact reports 6 artefacts and 17,674 px (`.census.artefacts`, `.census.artefact_px`, 0.0543 % of accepted); the difference is one 2,433 px component that is 55.9 % painted crack (`docs/UNLABELLED_AREA.md:206` — a figure that exists in that document and in no artifact) and is exempt under the guard's painted-pixel rule, so the guard's 5 / 15,241 is the figure that describes the shipped mask. But the test is component-level, so two further artefacts fused into 440 k and 228 k px crack systems ride through attached to real crack (`.guard.lower_bound`). No within-component straightness test exists.
- **The small-component population is mostly rim, not detections.** Of 12,583 unlabelled components (`txm_unlabelled_area_per_frame.json` `.n_unlab_components`), 284 are ≥ 4000 px, leaving 12,299 below that (*derived*); the sub-4000 px remainder is 12.2 % of unlabelled area, 6,921 of them exactly 1 px, and 80.4 % of their area lies within 25 px of a painted stroke (`txm_small_component_adjudication.json` `.population.*`, whose own count of 12,303 was measured on the 2026-09-20 mask, so these counts drift with mask version). Since `MIN_BLOB_PX` = 2000, these are the unpainted rim of large accepted components, not independent findings. Only 32 components, 74,404 px, **0.229 % of all accepted area**, are both ≥ 200 px and ≥ 200 px from any stroke (`.tractable_subset`) — that is the whole of what the small-component arm can ever be about.
- **No instrument or loading metadata. [TODO-AUTHOR]** Beam energy, microscope, detector, exposure and reconstruction method are not recorded in either repository, nor is the loading protocol (rig, load levels — the `_lbf` filenames imply pound-force but the protocol is not recorded — or cycle counting), nor specimen provenance, heat treatment, AM build parameters, geometry or thickness. None of it can be reconstructed from the data and none of it is invented here.
- **The pixel size is derived, not read from a header. [TODO-AUTHOR]** 29.24 nm/px (code constant `NM_PER_PX = 29.0` at `code/audit/measure_transverse_fwhm.py:37`) comes from operator-supplied mosaic geometry — a 30 µm viewing window, 0.35 overlap, 9 × 5 tiles over 6367 × 3691 px — as one geometric derivation with two consistency checks, not from instrument metadata (`circ2_scale_and_units.json`, which sustains the value while retracting its advertised "three independent routes" provenance, and which records that two frames close only at 0.30 overlap rather than 0.35). Every physical length in this paper (e.g. median FWHM 45.0 px = 1,305 nm, `txm_width_reference.json` `.is_it_the_instrument.fwhm_median_nm`) inherits that assumption. The 30 µm and 0.35 figures require the operator's confirmation before submission.
- **Two-dimensional.** These are areas on single mosaic frames, not crack volumes or crack-front shapes; a crack advancing in depth and one widening both read as more area. No temporal or growth-rate claim is made anywhere in this paper: the monotonicity arm was tested on a replicate set and lost to doing nothing, and is withdrawn.

---

## 7. Data and code availability

**Repository.** Tool, shipped models, analysis code and all 71 operator correction masks (`paint/corrections/corrections.npz`, 71 keyed arrays): `https://github.com/jzhang29-max/TXM_Crack_Detection_Pipeline`. Code is MIT (`LICENSE`); the experimental data — the 71 raw TXM images (`images/`), the four reference ground-truth images and masks, the human correction labels and the derived result sets — is CC BY 4.0 (`LICENSE-DATA`). Continuous integration runs the install, the self-test suite and a real ensemble prediction on Linux and macOS on every commit (`.github/workflows/linux.yml`, `macos.yml`); that file's own header states what the green tick does not cover — four of its five jobs run the 17-feature model alone, which marks 26.9–83.7 % of a crack-free frame, so they test plumbing and invariants rather than detection quality. **[TODO-AUTHOR]** deposit the 71 correction masks and the two models with a DOI before submission; the corrections are the irreplaceable artifact and the models are reproducible from them.

**Generators.** Each audit result in this paper is re-runnable from `code/audit/` in the pipeline repository, with the artifact it emits into `out/` of the analysis repository:

| script | artifact | backs |
|---|---|---|
| `code/audit/measure_unlabelled_area.py` | `txm_unlabelled_area_per_frame.json` | the evaluated-fraction census, §6.1 (E1) |
| `code/audit/measure_width_ratio.py` | `txm_width_ratio_unified.json` | the width-ratio table, §6.3 — every mask variant, one estimator, one frame set, same sample points |
| `code/audit/measure_iou_ceiling.py` | `txm_iou_ceiling_v2.json` (**not yet run**, §6.2) | ceiling and model IoU in one pass, replacing the superseded model column of `txm_iou_ceiling.json` |
| `code/audit/measure_transverse_fwhm.py` | `txm_transverse_fwhm.json` (**never emitted**, see below) | the pre-registered "is the width physical" test, §7.4 |
| `code/audit/build_small_component_crops.py` | `txm_small_component_cropkey_v2.json` + crops | the blind set for small isolated indications, matched on size **and** specimen |
| `code/audit/score_blind_panel.py` | `txm_small_component_adjudication_v2.json` | unblinding, instrument validation and the group-confound check for any blind run |
| `code/audit/heldout_crackfree_fp.py` | `txm_heldout_crackfree_fp.json` (**not yet run**, §7.2) | the uncontaminated false-positive figure |

The mask itself is `effective_mask` at `app/core/pipeline.py:885`, with its three stages at `clip_to_measured_width` (`:675`), `drop_straight_lines` (`:809`) and `local_background`/`_transverse_fwhm` (`:615`, `:643`); `MIN_BLOB_PX` = 2000 at `:470`. The shipped configuration is `effective_mask(corrections='none', tight=True)` with `WIDTH_CLIP=True` and `drop_straight_lines` active, and every number in this paper is on that configuration unless stated otherwise (`txm_unlabelled_area_per_frame.json` `.mask_version`).

**Not reproducible from this repository: the panel votes.** Producing them requires running a blind multi-agent panel over the crop images; they are an artifact of a specific model and harness and are **shipped as data, not regenerated** (`txm_small_component_adjudication_v2.json` `.generators.note`; `code/audit/README.md`). What *is* preserved and re-runnable: the crop keys with frame id and crop centre for every field (`txm_unlabelled_adjudication_cropkey.json`, 88 fields; `txm_small_component_cropkey_v2.json`, 94 fields), the fixed native crop window — **700 × 700 px for the large-region run** (`txm_unlabelled_adjudication.json` `.design.crops`) and **400 × 400 px for the small-component run**, chosen so that a 200 px clearance puts no painted crack in frame (`code/audit/build_small_component_crops.py:35` `WIN = 400`; `txm_small_component_adjudication.json` `.adjudication.design`) — the fixed shuffling seed (`build_small_component_crops.py:61`, `default_rng(20260921)`), the leak-guard statistics, the group-matching record, and the deterministic scorer. The crop PNGs themselves are not stored in either repository; they are byte-reproducible from `images/` and the crop keys via `build_small_component_crops.py`. Every individual vote, its panel, and the free-text evidence each panel gave ship inside the `per_field` arrays of both adjudication artifacts, so the reads can be inspected and disputed without re-running anything.

**Known gaps in the provenance chain, stated rather than papered over.** Four load-bearing artifacts carry no `generator` key: `txm_iou_ceiling.json`, `txm_width_reference.json`, `txm_unlabelled_adjudication.json` and `txm_width_clip_final.json`. For three of them no driver exists in `code/audit/` at all. For the fourth, `measure_iou_ceiling.py` was written afterwards and emits a new file (`txm_iou_ceiling_v2.json`) rather than reproducing the old one, and has not been run — so the shipped `txm_iou_ceiling.json` remains an unreproduced output with a superseded model column (§6.2). Separately, `code/audit/README.md` lists `measure_transverse_fwhm.py` as emitting `out/txm_transverse_fwhm.json`; **that file does not exist**, and the §7.4 numbers are read from `txm_width_reference.json` instead. **[TODO-AUTHOR]** re-emit these under named generators, or state in the final version that they are recorded outputs of scripts that were lost to a scratch directory — which is what they are.

**Pre-registration record.** The four pre-registered tests of §6.4, their decision rules fixed before the data were read, and the two that failed, are recorded in full at `out/txm_width_prereg_and_failed_variants.md`, including the rejected per-component FWHM variant (Tsens 0.798 → 0.261, clDice 0.846 → 0.412) and the named mechanisms for both failures. The four acceptance criteria (a)–(d) of the width-clip run are scored at `docs/WIDTH_REFERENCE.md:157–165`; that table is the only record of criterion (b), and it is not backed by an artifact.

## Addendum: the crack-free gate is contaminated

**The crack-free false-positive figure is inflated by training on those specimens.**
The six specimens asserted crack-free contribute 0 crack pixels and 95,351,647 not-crack
pixels to the labels -- 4.3% of training rows after per-image capping -- and the gate then
measures false positives on those same frames. Held out properly (5 seeds per arm, both
branches refitted at a 400,000-row budget and scored through the shipped
`CrackModel` ensemble path):

| arm | mean predicted area on crack-free material |
|---|---|
| trained WITH their labels | 0.289% +/- 0.032 |
| **trained WITHOUT them** | **1.581% +/- 0.337** |
| inflation | **5.5x**, all 5/5 pairs same direction, Wilcoxon p = 0.0625 |

p = 0.0625 is the **smallest value a two-sided signed-rank test can return at five pairs**
(2/2^5). It is floored by the sample size, not by a weak effect: every pair separated, and
the arms do not overlap (0.289 +/- 0.032 against 1.581 +/- 0.337). More seeds would
lower it; the direction is not in doubt.

The deployed model reads 0.174% on the same frames, so the gate's headline is a **lower
bound**: a model that has not seen those labels marks several times more of that material.
This does not condemn the detector -- 1.58% still clears the MIL-HDBK-1823A yardstick of
1%... -- but the figure as reported is partly memorisation.

Two things it is NOT. It is not the shipped weights' false-positive rate: these are
retrained models at a reduced row budget, so the comparison bounds the contamination rather
than replacing the gate's number. And it is not a verdict on the ensemble: both arms use it.
Generator `code/audit/heldout_crackfree_fp_v2.py`, artifact
`crack-evolution-5d/out/txm_heldout_crackfree_fp_v2.json`. An earlier attempt
(`heldout_crackfree_fp.py`) fitted a 17-feature-only model, landed 48x off the deployed
figure and got the sign backwards; it is kept with that failure recorded.
