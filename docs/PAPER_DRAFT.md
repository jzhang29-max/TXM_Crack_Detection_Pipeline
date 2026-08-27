# Labelled-pixel metrics are blind to false positives in sparse crack segmentation: a negative-control criterion for operator-trained models in transmission X-ray microscopy

**DRAFT.** Every number below is read from a recorded artifact in this repository, and the
provenance is given inline. Items still outstanding before submission are collected in §8 and
marked **[TODO]** where they occur. Author list, affiliations and funding are placeholders.

---

## Abstract

Interactive segmentation — a frozen vision-foundation-model encoder, a light classifier, and a
human correcting the output — is now a standard way to build a task-specific segmenter from a
few dozen images. Model selection in this loop is almost always driven by agreement with the
operator's labels: intersection-over-union, AUC, recall at matched false-positive rate. We show
on transmission X-ray microscopy (TXM) of fatigue cracks in 316L stainless steel that this
practice has a structural blind spot. In a controlled four-arm comparison at a fixed training
budget, a gradient-boosted classifier outperformed a neural-network ensemble on
intersection-over-union over labelled pixels (0.748 against 0.714) while marking **4.7 times
more** area of specimens confirmed to contain no crack (2.141% against 0.451%). The failure is
invisible to every labelled-pixel metric by construction, because the material on which it
occurs is precisely the material nobody annotates. We propose a mandatory negative-control axis
— predicted area on confirmed-empty specimens — as a deployment criterion, and report that it
refused two candidate models in this project that labelled-pixel metrics accepted. We
additionally report (i) a label-side correction for annotation width bias, in which each
hand-drawn stroke is narrowed to the darker core inside it before sampling, reducing output mask
half-width from 22 px to 5 px against a measured 2.5–3 px core; and (ii) two measurement
pitfalls with consequences beyond this dataset — crack area fraction is not comparable across
fields of view, and operators systematically zoom in as cracks grow, so naive growth curves
partly measure the camera; and a frozen encoder produced predicted areas differing by a factor
of two across compute stacks on bit-identical input. Tool, models, corrections and analysis code
are released.

**Keywords:** interactive segmentation, human-in-the-loop, model selection, false positives,
transmission X-ray microscopy, fatigue cracks, 316L, reproducibility

---

## 1. Introduction

Quantifying a fatigue crack from tomographic imaging requires a segmentation of the crack, and
segmentation of thin, sparse, low-contrast features remains the bottleneck. Fully supervised
deep learning needs dense annotation that nobody has. The now-conventional alternative is
interactive: take a frozen foundation-model encoder, put a small classifier on top, let an
operator correct the output, and retrain from those corrections.

That loop has a model-selection problem which, to our knowledge, has not been stated plainly.
The operator's corrections are the only labels available, so every metric used to decide whether
a retrain is an improvement is computed on the pixels the operator touched. In a sparse task the
overwhelming majority of the specimen is untouched — not because it is unimportant, but because
there is nothing there to mark. A model can fit the annotated distribution better and behave far
worse on the unannotated remainder, and no metric computed over annotations can detect it.

We encountered this concretely. This paper reports:

1. **A controlled demonstration of the blind spot** (§5.1) and a negative-control criterion that
   catches it, together with an honest account of how often the two axes actually disagree.
2. **A label-side correction for annotation width bias** (§5.2). Brush strokes are far wider than
   the crack they mark; a model trained on them reproduces the brush. Existing remedies modify
   the loss or the evaluation metric; we modify the label.
3. **Two measurement pitfalls** (§5.3) that we believe affect published tomographic crack
   quantification generally.
4. **An application** (§5.4): 71 TXM frames across four 316L specimens, segmented from operator
   corrections alone.

---

## 2. Related work and what is new here

**Interactive segmentation with foundation models.** μSAM [1] adapts Segment Anything to
microscopy with interactive correction and fine-tuning. Cellpose 2.0 [2] provides a
human-in-the-loop retraining pipeline and reports that 100–200 corrected regions suffice for a
custom model. Both are developed for and evaluated on biological imaging.

**Foundation models in materials imaging.** MatSAM [3] extracts eleven classes of metallic
microstructure from SAM without training, using an automated point-prompt strategy. SAM-I-Am [4]
applies semantic boosting to SAM for atomic-scale electron micrographs. Both are
prompt-driven and training-free; neither learns from operator corrections.

**Machine learning on TXM.** CNN segmentation of TXM tomograms is established [5], as are
classical machine-learning segmentation pipelines for TXM [6]. These are automatic segmenters
trained on curated ground truth, not interactive tools.

**Human-in-the-loop for microstructure.** A unified microstructure segmentation approach using
weak supervision and active learning [7] is the closest materials-side analogue.

**Thin-structure annotation quality.** Label error in thin-crack detection has been
characterised [8]; topology-aware losses such as clDice and skeleton-based evaluation metrics
address thin-structure segmentation from the loss and metric side respectively.

**What is therefore not new here.** The architecture. A frozen encoder plus a light classifier
plus interactive correction is the established pattern, and we claim no novelty for it. Nor is
machine learning on TXM new [5,6], nor SAM applied to materials imaging [3,4].

**What we believe is new.** (i) The demonstration that labelled-pixel metrics are structurally
blind to false positives in this loop, with a controlled measurement of the effect size and a
deployment criterion that catches it. (ii) Correcting annotation width bias at the label rather
than the loss or the metric. (iii) The two measurement pitfalls of §5.3. (iv) As a secondary
point, the combination — corrections-only training with a negative-control gate, on TXM fatigue
cracks — appears not to have been reported; the materials-side SAM work is prompt-driven and the
interactive-correction work is biological.

---

## 3. Method

### 3.1 Features

Each pixel is described by 273 numbers: 17 isotropic hand-crafted measurements (local darkness,
darkness relative to neighbourhood, edge and ridge responses at several scales) and 256 channels
from the Segment Anything ViT-H image encoder, used frozen. The encoder runs over 1024 px tiles
at a stride of 896 px, producing a 64×64×256 grid per tile; per-pixel values are read from that
grid by bilinear interpolation, and overlapping tiles are combined with a Hann window so tile
boundaries do not appear in the output.

### 3.2 Classifier

Two multilayer perceptrons — 17→64→32→1 and 273→128→64→1, each behind a standard scaler — and
the mean of their two probabilities. The ensemble beats either member in all five
cross-validation folds (member means 0.726 and 0.786 against 0.811 for the average). No external
labels are used at any stage: both members are fitted on the operator's corrections only.

### 3.3 Label narrowing

A brush stroke marking a crack is far wider than the crack. Measured against the darkest fifth
of each stroke, strokes on these frames are 2.6× to 15× wider than the feature, with a median
half-width of 26.9 px against a 3.2 px dark core. Before sampling training rows, each stroke
labelled *crack* is narrowed to the pixels darker than the local mean of their own
neighbourhood, and the discarded ring is sampled as background. The narrowing is verified per
frame and **declined** where the assumption fails — on 2 of 61 frames the dark core does not
survive a 60% retention test, and those frames keep their original labels.

### 3.4 Deployment gate

A retrain is accepted only if it passes both of:

- **Held-out agreement.** Cross-validation grouped by whole image, so train and test never share
  a frame, against an absolute floor of 0.60 IoU.
- **Negative control.** Predicted crack area on six specimens the operator has confirmed contain
  no crack, against a tolerance of 0.5 percentage points.

A model failing either is written to disk and remains selectable for inspection, but does not
become current. Both axes are recorded for every retrain.

---

## 4. Data

Seventy-one TXM frames, 2.9–32.1 megapixels, float32, of four 316L stainless steel specimens:

| set | frames | ordered by | range |
|---|---|---|---|
| HC 316L, fatigue | 27 | cycles | 200 → 1790 |
| wrought 316L, fatigue | 14 | cycles | 0 → 1300 |
| B2 | 17 | depth | 333.75 → 343.75 µm |
| B3 | 13 | depth | 268.13 → 388.13 µm |

Six further specimens are confirmed by the operator to contain no crack and are reserved
entirely for the negative-control axis; they are never trained on. Total operator annotation:
approximately 264 million labelled pixels across the 71 frames, from a single annotator.

**[TODO]** Instrument, beam energy, voxel size, reconstruction method, loading conditions and
specimen geometry. The absence of a recorded µm/px scale is a substantive limitation (§7).

---

## 5. Results

### 5.1 Labelled-pixel metrics are blind to false positives

**Figure 1** (`docs/paper_figs/fig1_metric_blindness.png`). Four arms at a fixed 400,000-row
training budget, crossing {reference frames included, excluded} with {gradient boosting, MLP
ensemble}, each scored on both gate axes:

| arm | IoU on labelled pixels | crack-free area called crack |
|---|---|---|
| HGB, no reference frames | **0.748** | 2.141% |
| MLP ensemble, no reference frames | 0.714 | **0.451%** |
| HGB, with reference frames | 0.869 * | 2.540% |
| MLP ensemble, with reference frames | 0.921 * | 0.544% |

\* trains on the frames it is scored on; reported for completeness only.

On the uncontaminated pair, the classifier that wins on labelled pixels is **4.7× worse** on
material confirmed to contain no crack. The effect persists in both training compositions, so it
is a property of the classifier rather than of the data mixture; removing the reference frames
slightly *lowers* false positives (0.544% → 0.451%).

Every metric that favoured the losing choice — IoU, AUC, recall at matched false-positive rate,
cross-group AUC — is computed over labelled pixels. Crack-free specimen is exactly the material
nobody labels: there is nothing to find, so nothing is marked. A model that separates the
labelled distribution better while behaving worse off it is therefore undetectable from
annotations alone, in principle and not merely in practice.

**How often do the axes disagree?** **Figure 2** places the five recipe-tagged models actually
deployed in this project on both axes. Kendall τ between the two rankings is **+0.80**: nine of
ten pairs concordant. One inversion: `corrections_only_v4_overlap` raised IoU from 0.776 to
0.789 while worsening crack-free false positives from 0.188% to 0.209%.

We therefore make the weaker and defensible claim: labelled-pixel agreement is usually
informative, is **never sufficient**, and provides no signal about whether this failure mode has
occurred. In this project the negative-control axis refused two candidate models that
labelled-pixel metrics accepted — one at 2.141% and an earlier one at 22.4% predicted area on
crack-free material.

### 5.2 Annotation width bias, corrected at the label

**Figure 3.** Narrowing each stroke to its dark core before sampling, against training on full
strokes:

| | full-stroke labels | narrowed labels |
|---|---|---|
| held-out grouped IoU | 0.789 ± 0.040 | 0.811 ± 0.023 |
| precision / recall | 0.933 / 0.837 | 0.936 / 0.860 |
| crack-free area called crack | 0.209% | 0.174% |
| **output mask half-width** | **22 px** | **5 px** |

The mask stops reproducing the brush: half-width falls from 22 px to 5 px against a measured
2.5–3 px dark core, so roughly half the remaining gap is closed and the residual is honestly
reported rather than claimed away.

**A comparison we cannot make cleanly.** The two IoU values are measured against *different
targets* — the narrowed recipe is scored against narrowed labels — so that row is not
like-for-like and we do not rest any claim on it. The comparable axes are mask width and
crack-free false positives; both improve. **[TODO]** A matched-target ablation, training on both
label variants and evaluating both against a single held-out label set, is outstanding and
should be completed before submission.

Existing remedies for thin-structure annotation act on the loss (clDice and related
topology-aware objectives) or on the metric (skeleton-based evaluation). The correction here acts
on the label, costs one filter pass, and is independent of both model and loss.

### 5.3 Two measurement pitfalls

**(a) Crack area fraction is not comparable across fields of view.** Crack area fraction is
crack pixels divided by *frame* pixels, so it depends on how much surrounding material is in
shot. Three frame pairs in this dataset image the same physical state at two fields of view:

| state | wide field | close field |
|---|---|---|
| wrought, 1100 cycles | 2.03% (10.4 MP) | 6.25% (8.8 MP) |
| B2, 343.75 µm | 12.29% (23.5 MP) | 23.13% (4.0 MP) |
| B3, 388.13 µm | 9.04% (23.3 MP) | 10.61% (3.9 MP) |

Nothing about the specimen changed. Absolute pixel counts do not rescue the comparison: the
23.5 MP B2 frame contains 2,888,587 crack pixels against 924,887 in the 4.0 MP frame, because a
wider field simply contains more crack.

**And the operator zooms in as the crack grows.** Correlation between fatigue cycles and frame
megapixels is **−0.76** (HC) and **−0.68** (wrought). A growth curve plotted from area fraction
therefore rises partly because the crack grew and partly because the frame shrank.

**Shape descriptors are nearly immune.** Correlation with frame megapixels, over all 64 frames
containing crack: mean width **−0.05**, branch ratio **+0.10**, skeleton fractal dimension
**−0.10**, branch points per unit length **−0.06** — against **−0.49** for area fraction. The
recommendation follows: report field-of-view-independent descriptors, or hold the field constant.

**(b) A frozen encoder is not automatically reproducible across compute stacks.** **Figure 4.**
Identical code, bit-identical decoded input (verified by checksum), identical model files:
predicted area 0.1880 on four environments and **0.0925** on one, with the mask shattering from
968 connected components to 21,019 and the probability mean falling from 0.381 to 0.369. A
two-arm matrix on the outlying machine isolated the cause to the encoder's compute device; four
alternatives (image decode, BLAS backend, tiling geometry, image-processor backend) were tested
and refuted. Anyone publishing numbers from a frozen foundation-model encoder should pin and
report the device.

### 5.4 Application: crack development in four 316L specimens

Analysis is restricted to within-field-of-view comparisons throughout, per §5.3.

**Growth, unconfounded.** Six wrought-316L frames at a constant 21.9–22.2 MP field: no crack at
300 cycles, 39,313 crack pixels at 800, 158,310 at 1000, 447,473 at 1100. Growth rate rises
roughly **37×** between the 800→900 and 1000→1100 intervals.

**Initiation.** Two HC frames at a matched ~32 MP field bracket initiation: 0 crack pixels at 400
cycles, 234,343 (0.731% of frame) at 600. At first detection the crack is already a single
region holding 95.4% of the crack area.

**Depth.** Six B3 frames at a constant 3.9–4.0 MP field: 2.79% → 10.61% over 380.00 → 388.13 µm,
a 3.8× increase across 8 µm, remaining a single connected region throughout.

**Morphology develops in roughness and branching, not in meander.** Skeleton fractal dimension
rises with crack development in all four sets (r = +0.56, +0.66, +0.54, +0.59) and branch ratio
rises strongly in three (+0.49, +0.72, +0.72), while main-path tortuosity stays flat. Every one
of these descriptors is field-of-view independent per §5.3.

**One dominant crack, not a network.** Per-region analysis of 439 regions shows the largest
region holds a median 79% (HC), 80–99% (wrought) and 83–100% (B2) of each frame's crack area.
Region *counts* are not crack counts: the export retains elongated components down to a 200 px
floor, so a frame reporting 16 regions had 9 of them below 2,000 px holding 0.59% of the area.
Frame-level means over regions are therefore contaminated — per-region mean width and boundary
roughness correlate +0.77 and +0.72 with log region area — and morphology should be read off the
dominant region only.

**What we do not claim.** Nothing here distinguishes transgranular from intergranular cracking.
That is a statement about the crack path relative to grain boundaries, and no grain-boundary
information is registered to these frames. We report straight-segment lengths (median 53–60 px)
and turn angles (median 25–27°), strikingly consistent across three specimens, as a fingerprint
that becomes diagnostic only once a grain size and a µm/px scale are supplied.

---

## 6. Discussion

The negative-control axis is not a second opinion on the same evidence. It is the only axis that
reads material the training distribution does not cover, which is why it can contradict every
other metric at once. Its cost is low — a handful of specimens the operator confirms are empty,
never trained on — and in this project it twice prevented a regression that labelled-pixel
metrics endorsed.

The generalisation is not specific to cracks. Any task in which the positive class is sparse,
thin, and annotated by a human who marks only what they find will have the same structure:
vessels, fibres, road networks, weld defects, delamination. Wherever the negative class is
*unlabelled rather than labelled-negative*, agreement with annotation cannot bound false
positives.

Label narrowing addresses a different failure of the same asymmetry. The operator's stroke
encodes *where* a crack is far more reliably than *how wide* it is, and a model trained on the
stroke inherits the width. Correcting the label rather than the loss keeps the fix independent
of architecture.

---

## 7. Limitations

- **No physical scale.** No µm/px is recorded in the exported data, so all lengths and areas here
  are in pixels or as a fraction of frame. Crack length, opening and growth rate in physical
  units are unavailable, as is any comparison to ΔK. This is the single most consequential
  limitation and it is a recording gap, not an analysis one.
- **Two-dimensional.** These are areas on single slices, not crack volumes or crack-front shapes.
  A crack growing in depth and one growing in width both read as more area.
- **One annotator, no repeats.** No inter- or intra-operator uncertainty can be estimated, so no
  error bars appear on any measurement in §5.4.
- **The masks are heavily human-curated.** Across the four sets the operator painted 4.6, 9.8,
  9.3 and 7.6 million pixels as crack against final mask areas of 6.3, 9.4, 10.6 and 7.3
  million. These numbers reflect one person's judgement as much as the model's.
- **The width comparison in §5.2 is not matched-target** — see the **[TODO]** there.
- **Six negative controls** is a small sample for the axis on which the paper's main claim rests.
- **A data-integrity defect** remains in the B3 set: two frames labelled 380.00 µm in the same
  field-of-view tier disagree by 2× in area and 4.7× in skeleton length.

---

## 8. Outstanding before submission

1. Matched-target ablation for §5.2.
2. µm/px scale from the instrument headers; re-express §5.4 in physical units.
3. Second annotator on ≥10 frames, for uncertainty.
4. Instrument and loading metadata for §4.
5. Resolve the duplicated 380.00 µm frames.
6. Decide figure/supplementary split — 36 per-set charts exist and most belong in supplementary.

---

## Data and code availability

Tool, both shipped models, the analysis code and all 71 operator correction masks are in the
project repository. Continuous integration runs the install, the self-test suite and a real
ensemble prediction on Linux and macOS on every commit. **[TODO]** Deposit the corrections and
models with a DOI; they are the irreplaceable artifact and the models are reproducible from them.

## References

[1] Archit et al. Segment Anything for Microscopy. *Nature Methods*, 2024.
[2] Pachitariu & Stringer. Cellpose 2.0: how to train your own model. *Nature Methods*, 2022.
[3] MatSAM: a training-free approach to extracting material microstructures via a visual large
    model. *Acta Materialia*, 2025.
[4] SAM-I-Am: semantic boosting for zero-shot atomic-scale electron micrograph segmentation.
    *Computational Materials Science*, 2024.
[5] Automated correlative segmentation of large transmission X-ray microscopy tomograms using
    deep learning. *Materials Characterization*, 2018.
[6] Machine-learning-based algorithms for automated image segmentation of transmission X-ray
    microscopy. *JOM*, 2021.
[7] A unified microstructure segmentation approach via human-in-the-loop machine learning.
    *Acta Materialia*, 2023.
[8] Xu et al. How do label errors affect thin crack detection by DNNs. *CVPR Workshops*, 2023.

**[TODO]** Complete bibliographic details, volumes, pages and DOIs.
