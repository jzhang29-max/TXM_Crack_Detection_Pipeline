# What only you can supply

45 `[TODO-AUTHOR]` markers in `docs/PAPER_DRAFT.md`, grouped so this is one sitting rather than
a hunt. Each entry gives the line, what is needed, and the sentence it lands in.

Most of them are one fact repeated: the acquisition. Supply the block at the top and roughly
half of these close at once.

---

## A. The acquisition block — fill this in once

This is not in either repository and cannot be derived from the data. Everything in Methods
that is marked depends on it.

```
instrument (make/model):
beam energy / filtration:
source-detector and source-sample distance, or effective magnification:
detector, pixel pitch:
VOXEL SIZE at the specimen:            # the repo assumes 29 nm/px from tile geometry --
                                       # confirm or correct, it scales every physical number
projections / exposure / total scan time:
reconstruction algorithm and software (+ version):
ring-artefact and beam-hardening correction applied?:
mosaic stitching software (+ version):  # ASHLAR and Fiji are inferred, not recorded
flat/dark field protocol:
```

## B. Specimen and loading provenance

```
316L source, form, heat treatment:
AM group: process, parameters, build orientation, post-processing:
specimen geometry and dimensions per group (B2, B3, AM/HC_316L, wrought):
loading: rig, mode (bend/fatigue), R-ratio, frequency, peak load:
what the frame labels mean: "1200 cycles", "3_1_lbf", "338_13um" -- cycles, load, displacement?
were frames acquired in situ under load, or unloaded between steps?:
how were the six crack-free specimens confirmed crack-free?:
```

## C. Annotation provenance

```
how many annotators?:                  # the paper says one; confirm
their expertise level:
was any frame annotated independently more than once?:   # if yes, between-annotator
                                       # agreement is the natural comparator for the
                                       # detector-vs-annotator width result and would
                                       # strengthen it considerably
annotation software and brush settings:
roughly how long per frame?:
```

## D. Authorship and admin

```
author list, ORCIDs, affiliations:
corresponding author:
funding and grant numbers:
beamline/facility acknowledgement and proposal number, if applicable:
target journal:
data repository + DOI for deposit:
conflicts of interest:
```

---

## E. Every marker, by section


### (front matter)  (1)

- **L25**
  > 2. [TODO-AUTHOR] MARKERS. Acquisition metadata -- beam energy, reconstruction method, voxel size provenance, loading protocol, specimen history -- is in neither repository. It cannot be


### 1. Introduction  (2)

- **L60** — insert the held-out crack-free false-positive rate, and the deployed figure beside it, when the refit completes.
  > **What we do not claim, and what is weak.** We make no corpus-wide over-marking claim; the detector under-marks, and is indistinguishable from the annotator. We make no crack-growth claim; that test lost to doing nothing. Our adjudication panels are language-model agents, not human experts: their va

- **L60**
  > **What we do not claim, and what is weak.** We make no corpus-wide over-marking claim; the detector under-marks, and is indistinguishable from the annotator. We make no crack-growth claim; that test lost to doing nothing. Our adjudication panels are language-model agents, not human experts: their va


### 2. Methods  (21)

- **L72**
  > ### 2.1 Specimens, loading and acquisition — [TODO-AUTHOR]

- **L76**
  > - **Instrument and acquisition:** microscope, beam energy, detector, objective/zone plate, exposure, number of projections, tomographic reconstruction method and any ring/beam-hardening correction. [TODO-AUTHOR] - **Loading protocol:** rig, load levels, and cycle counting. Filenames carry tokens suc

- **L77**
  > - **Instrument and acquisition:** microscope, beam energy, detector, objective/zone plate, exposure, number of projections, tomographic reconstruction method and any ring/beam-hardening correction. [TODO-AUTHOR] - **Loading protocol:** rig, load levels, and cycle counting. Filenames carry tokens suc

- **L78**
  > - **Loading protocol:** rig, load levels, and cycle counting. Filenames carry tokens such as `_1_lbf`, `_3_18lbf` and `_1250_cycles`, which imply pound-force load steps and a fatigue cycle count, but no protocol is recorded anywhere in the repositories. Do not state the implied units without confirm

- **L79**
  > - **Specimen provenance:** composition and heat treatment of the 316L stock; for the group referred to here as AM, the build process and parameters. The group key `AM` corresponds to filenames beginning `HC_316L_`; what `HC` denotes is not recorded. [TODO-AUTHOR] - **Geometry:** specimen shape and t

- **L80**
  > - **Geometry:** specimen shape and through-thickness. A 22 µm through-thickness is used in a separate 3D arm of this project; it is unverified here and is not used. [TODO-AUTHOR] - **Annotator:** identity, training, materials background, time per frame, and whether any frame was annotated more than 

- **L81**
  > - **Annotator:** identity, training, materials background, time per frame, and whether any frame was annotated more than once. A single annotator produced all labels; nothing else about the annotation session is recorded. [TODO-AUTHOR] - **Ethics, data availability and a DOI deposit** for the 71 cor

- **L85**
  > The pixel size is **29.24 nm/px**, and it is a *geometric derivation, not an instrument header value*. It follows from operator-supplied mosaic geometry — 9 × 5 tiles of a 30 µm viewing window at 0.35 overlap, giving 186.0 × 108.0 µm over a 6367 × 3691 px mosaic, isotropic to 0.2% (`crack-depth-3d/R

- **L87**
  > One inconsistency is carried openly: the width artifact converts pixels to nanometres with a rounded **29.0 nm/px** (`code/audit/measure_transverse_fwhm.py:37`, `NM_PER_PX = 29.0`), so the median transverse FWHM of 45.0 px is recorded as 1,305 nm (`txm_width_reference.json .is_it_the_instrument.fwhm

- **L115** — re-measure or drop
  > **Preprocessing, and the fact that the human and the model see different images.** Frames are de-stitched (per-pixel correction of the mosaic border strip plus a validated 2D FFT notch at the detected tile pitch, `code/destitch.py`) and pseudo-flat-fielded by division with an anisotropic Gaussian bl

- **L130** — re-emit the 2026-09-19 label-width column with its definition, or retire it.
  > The primary instrument is named primary only because its definition is recorded in a generator that still exists; we do not claim it is the more accurate of the two. At 29.24 nm/px the two medians are **1.20 µm** and 2.15 µm. The per-group columns of the primary row are *derived* by regrouping `.per

- **L161** — re-measure into an artifact or drop
  > **Training labels are narrowed before sampling.** Each painted crack region is narrowed to its dark core with `tighten_to_image`, and the discarded ring joins the *negative* pool at its true area weight rather than being dropped (`gather_training_data`, `app/core/pipeline.py:1496`). This is the `thi

- **L173** — reconcile
  > Declining rather than guessing matters most on AM. Painted AM crack has essentially no intensity contrast against painted not-crack: median Cohen *d* = **−0.31** on AM against +1.76 (B2), +3.74 (B3) and +1.38 (Wrought), with the painted crack *brighter* than the painted not-crack on 17 of 25 AM fram

- **L183** — re-emit the guard verification with its crack-free set and generator recorded
  > **Its denominator is ten frames, not six, and this is a correction.** The generator that produced `.guard.verified` is in neither repository — `grep` for `crackfree_fp` over both repos returns only the artifact itself — so the crack-free set it averages over is not recorded. It can be recovered arit

- **L185** — run `code/audit/heldout_crackfree_fp.py`
  > **It is partly memorisation and is treated as pending.** The six gate frames contribute 25.38% of all labelled not-crack pixels (§2.4) — the model was explicitly told those pixels are background. In *sampled rows* the share is smaller: at `neg_cap` = 25,536 rows per image, the six frames supply at m

- **L211** — re-run per-frame Tprec on the shipped mask
  > (clDice and Tsens *derived* from `txm_width_clip_final.json .per_frame`; IoU *derived* from `txm_iou_ceiling_v2.json .per_frame`. The clDice column reproduces `docs/WIDTH_REFERENCE.md:171–175` to three decimals.) **Per-frame Tprec on the shipped mask is the one quantity no artifact in `out/` carries

- **L262** — reconcile the exclusion count and record the rule in the artifact
  > Paired comparison of detector against human annotator uses the 61 frames carrying both, the same points and the same estimator, by Wilcoxon signed-rank (`.shipped_vs_label`). The primary run covers 64 of the 71 frames; **all seven absent frames carry zero painted crack**, but the artifact records no

- **L274** — state which instrument (c) was adjudicated on
  > - An **adjudication of the material a narrowing step deletes**, verdict (A): the deleted material is dark, sitting 1.63σ above the matrix reference at 42% of the core contrast. The pre-registered sign test failed at p = 0.145 (33/57 frames, rule required p < 0.05) even though the paired Wilcoxon giv

- **L286** — re-emit the census with the curved components listed, or the claim rests on medians alone
  > - Two different sets of five components appear in the same artifact — the five in the 71-frame census (wander 0.13–0.71 px, *derived* from `.census.components`) and the five inside the 32-component tractable subset (wander 0.13–0.93 px over 171–788 px of length, `.straight_line_artefacts.finding`). 

- **L332** — re-run the generator or correct the mapping
  > - **The panel votes are not regenerable** (§2.11); the scoring of them is deterministic and re-runnable, the votes are an artifact of a specific model and harness and ship as data. - **`code/audit/README.md` lists `measure_transverse_fwhm.py` as emitting `out/txm_transverse_fwhm.json`, and that file

- **L333** — complete the mapping table
  > - **`code/audit/README.md` lists `measure_transverse_fwhm.py` as emitting `out/txm_transverse_fwhm.json`, and that file does not exist**; the §2.9.4 numbers are read from `txm_width_reference.json`, which has no listed generator [TODO-AUTHOR: re-run the generator or correct the mapping]. - **The sam


### 3. Results: metric blindness and the unlabelled quarter  (9)

- **L434**
  > (`txm_small_component_adjudication.json`, `.straight_line_artefacts.guard.verified.detail[1]`, `area_after` = 0.0), and it is indeed absent from the census. **[TODO-AUTHOR]** Re-run the census with a distinct skip reason retained per frame and the skipped records carried into

- **L499**
  > Tsens medians are 0.8635 / 0.9546 / 0.8145 against the shipped 0.8234 / 0.970 / 0.761. The ceiling itself is unaffected, being label-only. **[TODO-AUTHOR]** Re-run the ceiling script against the shipped mask if a multiple is wanted; otherwise the ceiling stands

- **L611**
  > painted not-crack area is an upper bound, because some of that area is the annotator. **[TODO-AUTHOR]** `UNLABELLED_AREA.md` and `.unexpected.caveat` both anchor this argument to "the 0.197%-of-painted-not-crack figure". That figure appears in no artifact in either

- **L763**
  > 0.025278 → 0.023588 is an optimistic bound and the guard's 6.7% relative reduction is measured on the favourable case. **[TODO-AUTHOR]** Drop the held-out figure in here beside the deployed one when the run completes; do not replace the deployed number, report both.

- **L765**
  > the deployed one when the run completes; do not replace the deployed number, report both. - **[TODO-AUTHOR]** The two crack-free figures in that block are stored in inconsistent units — the corpus value appears to be a percentage while the per-frame `area_before` is a

- **L806**
  > the identical protocol can be re-run with blinded human readers. The rendered crop images themselves are **not** retained in either repository. **[TODO-AUTHOR]** At least two blinded human readers through the identical protocol, before this evidence is offered as

- **L821**
  > viewing window, 0.35 overlap, 9 × 5 tiles over 6367 × 3691 px — not read from an instrument header. **[TODO-AUTHOR]** Confirm the 30 µm and 0.35 figures with the operator before any physical length in this section is presented as instrument metadata.

- **L824**
  > - **Instrument and acquisition metadata are absent from both repositories.** **[TODO-AUTHOR]** Beam energy, microscope, detector, exposure and reconstruction method; loading rig, load levels and cycle counting; specimen provenance, heat treatment, AM build

- **L836**
  > control against 11/32 test — and it must not be printed as it stands. **[TODO-AUTHOR]** (i) Regenerate the small-component panel from `txm_small_component_adjudication_v2.json` so it shows 13/32 against 13/32 and the


### 4. Results: width, and what the annotations cannot adjudicate  (6)

- **L877** — confirm the 30 µm window and 0.35 overlap with the operator, or supply the pixel size from the acquisition record. All width figures below are reported in pixels and are unaffected; only the nanometre conversion depends on this.
  > In physical units the median is 1,305 nm (`txm_width_reference.json`, `.fwhm_median_nm`), but that conversion rests on a pixel size of 29.24 nm/px that is **derived from operator-supplied mosaic geometry** (30 µm viewing window, 0.35 overlap, 9 × 5 tiles over 6367 × 3691 px; `configs/default.yaml:6`

- **L905** — state whether any frame was annotated independently more than once; if any were, the between-annotator width difference is directly measurable and should replace this analogy
  > The natural reading is that the detector and the annotation disagree about width no more than two annotators would. **That reading is an analogy, not a measurement, and this corpus cannot upgrade it**: there is no second annotator and no frame is known to have been labelled twice **[TODO-AUTHOR: sta

- **L905** — add a TOST or a bootstrap CI on the −0.0113 paired difference to state what size of difference the data actually exclude
  > The natural reading is that the detector and the annotation disagree about width no more than two annotators would. **That reading is an analogy, not a measurement, and this corpus cannot upgrade it**: there is no second annotator and no frame is known to have been labelled twice **[TODO-AUTHOR: sta

- **L931** — settle this with a modality that carries a depth axis, or with a repeat acquisition of one frame at higher magnification; state which if either is available.
  > That verdict is the one this paper reports, and it leaves the substantive question open. The dropped halo sits at 39% of the kept-centreline contrast (*derived*, 0.0379/0.0969; the source note asserts 42%, which its own medians do not reproduce, so the derived figure is used) and 1.63σ above backgro

- **L945** — re-emit the census with the curved components listed individually, or the 2.9–12.9 range must stay out of the paper.
  > **A gap in the evidence that must be closed before submission.** `docs/UNLABELLED_AREA.md` states the separation as "genuine long crack, 6 components, 2.9–12.9 px" of wander. That range is **not in any artifact in `out/`** — the census enumerates the straight components and the one painted crack, bu

- **L959** — beam energy, microscope and detector, exposure, and reconstruction method; the loading protocol, rig and load levels behind the `_lbf` filenames; cycle counting for the fatigue sets; specimen provenance, heat treatment, AM build parameters, and through-thickness geometry.
  > - **No second annotator, and no repeat annotation** — see above. - **Acquisition metadata is absent from both repositories.** **[TODO-AUTHOR: beam energy, microscope and detector, exposure, and reconstruction method; the loading protocol, rig and load levels behind the `_lbf` filenames; cycle counti


### 6. Limitations  (4)

- **L1039**
  > The held-out measurement is specified and pending: `code/audit/heldout_crackfree_fp.py` refits the deployed architecture on the same rows with every crack-free frame removed and re-scores, with the incumbent re-scored through the identical path so the arms differ only in training data. **[TODO-AUTHO

- **L1049**
  > There is one annotator, and no frame is known to have been labelled twice, so no inter- or intra-operator uncertainty can be estimated **[TODO-AUTHOR]**. Because the labels are the reference for every agreement metric and the target of the audit, that is a limitation on both sides of the comparison 

- **L1063**
  > - **The small-component population is mostly rim, not detections.** Of 12,583 unlabelled components (`txm_unlabelled_area_per_frame.json` `.n_unlab_components`), 284 are ≥ 4000 px, leaving 12,299 below that (*derived*); the sub-4000 px remainder is 12.2 % of unlabelled area, 6,921 of them exactly 1 

- **L1064**
  > - **No instrument or loading metadata. [TODO-AUTHOR]** Beam energy, microscope, detector, exposure and reconstruction method are not recorded in either repository, nor is the loading protocol (rig, load levels — the `_lbf` filenames imply pound-force but the protocol is not recorded — or cycle count


### 7. Data and code availability  (2)

- **L1071**
  > **Repository.** Tool, shipped models, analysis code and all 71 operator correction masks (`paint/corrections/corrections.npz`, 71 keyed arrays): `https://github.com/jzhang29-max/TXM_Crack_Detection_Pipeline`. Code is MIT (`LICENSE`); the experimental data — the 71 raw TXM images (`images/`), the fou

- **L1089**
  > **Known gaps in the provenance chain, stated rather than papered over.** Four load-bearing artifacts carry no `generator` key: `txm_iou_ceiling.json`, `txm_width_reference.json`, `txm_unlabelled_adjudication.json` and `txm_width_clip_final.json`. For three of them no driver exists in `code/audit/` a


---

## F. The one thing that is not a form to fill in

The paper's central claim — that the unadjudicated quarter of the detector's output is crack —
rests on blind panels of language-model agents. Their validation shows they agree with the
annotator's own strokes, which is not the same as competence at 316L TXM, and the paper says so
in its own voice.

**Two blinded human readers through the existing protocol would settle it.** Nothing needs
rebuilding: `code/audit/build_small_component_crops.py` regenerates the crops from a fixed seed,
the crop key records the true class of each field, and `code/audit/score_blind_panel.py` scores
any set of votes and prints the group-mix and standardised-control checks automatically.

A reader needs: the crop folder, a spreadsheet with one row per field and a yes/no/unsure
column, and no knowledge of which class any field belongs to. About 94 fields, a few seconds
each. Two readers and the claim is on human evidence.
