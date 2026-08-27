# Paper outline — what is publishable here, and what is not

Written after checking this work against the 2024–2026 literature. The conclusion is that the
**system** is not novel and should not be the claim; **three specific results** are, and one of
them is strong enough to carry a paper on its own.

## The honest starting position

Frozen foundation-model features + a light classifier + interactive correction + retrain is an
established pattern by 2026:

- Segment Anything for Microscopy (μSAM), *Nature Methods* 2024 — SAM as a frozen encoder with
  interactive correction. Architecturally the same idea as this project.
- Cellpose 2.0, *Nature Methods* — human-in-the-loop retraining; 100–200 corrected ROIs suffice.
- "A unified microstructure segmentation approach via human-in-the-loop machine learning",
  *Acta Materialia* — the materials-side version.

A paper claiming "we used SAM features and a small MLP with human correction" is a
reimplementation of those. It should not be written.

---

## Claim 1 (the paper's spine) — labelled-pixel metrics are structurally blind to false positives

**The result.** Four arms at a fixed 400,000-row budget, crossing {reference frames in, out}
with {HGB, MLP ensemble}, each scored on both axes (`research/fp_attribution.json`):

| arm | IoU on labelled pixels | area of confirmed crack-free specimen called crack |
|---|---|---|
| HGB | **0.748** (better) | **2.141%** |
| MLP ensemble | 0.714 | **0.451%** (4.7× cleaner) |

The classifier that wins on labelled pixels is 4.7× worse on material confirmed to contain no
crack, and the effect survives in both training compositions (2.540% vs 0.544% with reference
frames in), so it is the classifier, not the data mix.

**Why it is not a fluke of one metric.** IoU, AUC, matched-FPR recall and cross-group AUC are
all computed over *labelled* pixels. Crack-free specimen is exactly the material nobody labels —
there is nothing to find, so nothing gets marked. A model can separate the labelled
distribution better and behave far worse off it, and no amount of AUC over labels can see that.

**The honest limit, which must be stated in the paper.** Across the five comparably measured
models actually deployed in this project, the two axes mostly *agree*: Kendall τ = +0.80, with
one inversion (v4 raised IoU 0.776 → 0.789 while worsening crack-free FP 0.188% → 0.209%). So
the claim is not "IoU is useless". It is: **IoU cannot detect this failure mode when it occurs,
and nothing in the labelled-pixel evidence tells you whether it has occurred.** That is
sufficient to justify a mandatory negative-control axis, and it is a weaker, defensible claim.

**What this generalises to.** Any sparse thin-feature segmentation task where the negative class
dominates and is unlabelled: cracks, vessels, fibres, road networks, defects.

---

## Claim 2 — annotation width bias, corrected at the label rather than the loss

**The problem is known.** Label error in thin-crack detection (Xu et al., CVPR-W 2023),
topology-aware losses such as clDice, skeleton-based evaluation metrics.

**What is different here.** Those remedies change the **loss function** or the **evaluation
metric**. This one changes the **label**: before sampling training rows, each hand-drawn crack
stroke is narrowed to the darker core inside it using the image itself, and the discarded ring
is treated as background.

**The result** (`docs/THIN_LABELS.md`, recipe `thincore_v5`):

| | full-stroke labels (v4) | narrowed labels (v5) |
|---|---|---|
| held-out grouped IoU | 0.789 ±0.040 | 0.811 ±0.023 |
| precision / recall | 0.933 / 0.837 | 0.936 / 0.860 |
| crack-free area marked | 0.209% | 0.174% |
| **mask half-width** | **22 px** | **5 px** |

Against a measured 2.5–3 px dark core, so the remaining gap is real and should be reported as
such. The technique is model-agnostic and costs one filter pass.

**Caveat to state plainly.** The two IoU figures are measured against different targets (v5's
target is the narrowed label), so they are not directly comparable. The comparable axes are mask
width and crack-free false positives, and both improve. A reviewer will find this if the paper
does not say it first.

---

## Claim 3 — two measurement pitfalls with consequences beyond this dataset

**(a) Field-of-view contamination of area fractions.** Crack area fraction is crack pixels ÷
frame pixels, so it depends on how much material is in shot. Three frame pairs in this dataset
image the *same physical state* at two fields of view: 2.03% vs 6.25% (wrought, 1100 cycles),
12.29% vs 23.13% (B2, 343.75 µm), 9.04% vs 10.61% (B3, 388.13 µm). And operators zoom in as
cracks grow — correlation between cycles and frame megapixels is −0.76 and −0.68 — so a naive
growth curve *manufactures* growth. Shape descriptors are nearly immune (correlation with
megapixels: mean width −0.05, branch ratio +0.10, fractal dimension −0.10) against −0.49 for
area fraction, which is a concrete recommendation: **report field-of-view-independent
descriptors, or hold the field constant.**

**(b) Cross-platform irreproducibility of a foundation-model feature stack.** Identical code,
bit-identical input (same sha1), identical model files: predicted area 0.1880 on four
environments and 0.0925 on a `macos-26-arm64` runner, traced by controlled matrix to the SAM
encoder running on that runner's virtualised Metal stack. Probability mean 0.381 → 0.369, mask
shattered from 968 to 21,019 components. Four alternative causes were tested and refuted
(decode, BLAS, tiling geometry, processor backend). Relevant to anyone publishing numbers from
a frozen foundation-model encoder without pinning the device.

---

## What this cannot be

Not a fatigue-mechanics paper. Comparable tomography studies report crack extent in physical
units and relate growth to ΔK. This dataset has **no µm/px scale** and no load or geometry data,
so crack length, opening and growth rate in physical units are unavailable. That is a recording
gap, not an analysis gap.

Ranked list of what unlocks it:
1. **µm/px from the `.xrm` headers.** Converts every measurement to physical units and makes
   fields of view comparable. Cheapest, highest value, probably already on disk.
2. **Load and specimen geometry**, for ΔK and therefore da/dN.
3. **A second annotator on ~10 frames**, for inter-operator uncertainty — currently there are no
   error bars anywhere and no way to compute them.
4. **Fix the duplicated 380.00 µm labels in B3**, where two frames in the same tier disagree 2×
   in area and 4.7× in skeleton length.

---

## Target venue

**Primary: *Measurement Science and Technology* (IOP).** It publishes measurement methodology
and negative results, its readership is the people who need Claim 1, and Claims 1–3 together
form one coherent argument about how to measure sparse features credibly. Claim 1 is the title.

Title shape: *"Labelled-pixel metrics are blind to false positives in sparse crack segmentation:
a negative-control criterion for human-in-the-loop model selection."*

**Companion: *SoftwareX* or JOSS**, for the tool. JOSS reviews software quality rather than
novelty, and this repo already has CI on two platforms, a 70-check self test, and honest
scorecards — unusually strong for that venue. Cite the methods paper from it.

**Do not submit to** *Acta Materialia*, *International Journal of Fatigue*, or *FFEMS* until
items 1 and 2 above are closed. Descriptive pixel statistics submitted to a mechanics audience
will be rejected on the grounds that nothing is in physical units.

---

## Figures the paper already has

| figure | source | supports |
|---|---|---|
| the 4.7× inversion, both axes, four arms | `research/fp_attribution.json` | Claim 1 |
| Kendall τ = +0.80 across the deployed lineage, with the one inversion marked | `app_data/models/retrain_history.json` | Claim 1's honest limit |
| stroke vs narrowed label, and the resulting mask width | `docs/THIN_LABELS.md` | Claim 2 |
| same crack at two fields of view | `analysis/charts/06_same_crack_two_views.png` | Claim 3a |
| area-% vs cycles beside field-of-view vs cycles | `analysis/charts/01_confound_zoom.png` | Claim 3a |
| the one constant-field growth curve | `analysis/charts/02_wrought_clean_growth.png` | Claim 3a |
| cpu vs mps arm on one runner image | `.github/workflows/macos.yml` header | Claim 3b |

## What still needs doing before submission

- A proper ablation for Claim 2 on *matched* targets, so the IoU comparison is fair: train on
  full strokes and on narrowed strokes, evaluate both against the same held-out labels.
- Inter-annotator agreement on a subset, for any error bar at all.
- Decide whether the 36 per-set charts are figures or supplementary. Most are supplementary.
