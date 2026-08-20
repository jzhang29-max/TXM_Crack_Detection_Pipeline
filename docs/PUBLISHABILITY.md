# What is publishable here, and what is not

Written after an attempt to build a Q1 methods paper's central experiment, which did not
come off. Recorded because the negative result changes what can honestly be claimed.

## Not publishable: the architecture

Prior art twice over. "Engineered multiscale filters plus a shallow pixel classifier" is
ilastik and ZEISS ZEN Intellesis, which has shipped VGG19 activations into a 25-tree
RandomForest since ~2017 — from the vendor that builds the Xradia TXM this data comes from.
"Frozen foundation-model embeddings plus a shallow classifier in an interactive app" is
FeatureForest (npj Imaging 2025, MobileSAM at 320 features/px against this project's 273),
SAMBA (arXiv 2312.04197) and napari-convpaint (Cell Reports Methods 2026, which benchmarks
108,642 feature combinations across VGG16/DINOv2/Cellpose/ilastik). There is no architectural
novelty claim available, and a reviewer will know these tools.

## Not publishable: the accuracy

Held-out IoU 0.815–0.839 rests on four dense ground-truth images, one specimen group, all
wide-open cracks at ~65 px median width, with Ilastik-derived labels and no second annotator.
n = 4 floors an exact two-sided sign test at p = 0.125, so nothing here can be significant.
And in this project's own easy width regime the best published figure is Dice 0.964 (Riesz
Networks) against this project's Dice 0.898–0.912 — roughly what published methods score at
3 px width, not 65 px.

## Not demonstrated: self-confirmation drift

The most promising claim, and the experiment failed to support it.

**The observation.** Three consecutive real retrains moved the false-call rate on confirmed
crack-free specimens 0.137% → 0.238% → 0.264% while ground-truth IoU sat flat at 0.936–0.940.
**The mechanism proposed.** Correcting a model is cheap when you agree and expensive when you
do not — one click accepts a whole connected region, disagreeing means drawing — and in this
project's label set **98.3% of force-crack pixels lie where the model had already predicted
crack**. Training on confirmations is self-training, which amplifies existing bias.

`code/drift_experiment.py` tests this directly: replace the human with a perfectly agreeable
one, accept the model's own confident predictions as labels each generation, and watch the
watched and unwatched metrics separately. Two attempts were discarded for measurement faults
before the third produced usable numbers, and both faults are worth recording:

1. Generation 0 trained on the four ground-truth images alone. Those are 25% crack and carry
   no negatives from other specimens, so the baseline already marked **25.1%** of crack-free
   material as crack — the same regime as the two models this project measured at 22% and
   rejected. The false-call rate was saturated before the loop began.
2. The false-call rate was measured on the *labelled* pixels of those specimens. Those labels
   are largely imported research negatives which HANDOFF.md records as "false-positive
   cleanup (wedge margin, edge ring, round speckle)" — i.e. exactly the pixels a model is
   most likely to fire on. That sample read **4.97%** where the real model measures 0.106%,
   47× too high. Any conclusion from it, drift or no drift, would have been a sampling
   artifact.

Fixed — generation 0 matching the deployed condition, false calls measured on 300,000 pixels
sampled uniformly inside six crack-free specimens — the six-generation result is:

| | generation 0 | generation 5 | change |
|---|---|---|---|
| ground-truth IoU, in-sample | 0.9502 | 0.9169 | **−0.033** |
| ground-truth IoU, held out | 0.7991 | 0.8342 | **+0.035** |
| false calls on clean material | 1.008% | 1.378% | 1.4× |

**This is not the predicted dissociation.** The false-call rate did rise, but
non-monotonically (0.727% at generation 2), the in-sample metric *also* fell — by more than
the deployment gate's 0.01 tolerance, so the gate would have caught it — and held-out IoU
*improved*. One seed, six generations, sampled pixels: too weak to claim a mechanism.

So the honest position is: **the production drift is an observation whose cause is not
established.** It may be self-confirmation at a scale this simulation does not reach, or
something else entirely — the three real retrains also differed in label volume, in the
imported-negative cleanup, and in specimen coverage. Claiming a demonstrated drift mechanism
would not survive review.

## What IS publishable: a validation protocol, with the traps quantified

Not a model, and not a metric — MIL-HDBK-1823A has specified false-call analysis since 2009.
What appears to be genuinely new is automating that analysis *inside the interactive
annotation loop*, tying it to a deployment gate, and quantifying the blind spots of the
metrics the field currently uses. Each of the following is measured, reproducible from this
repo, and to our knowledge unreported for interactive microscopy segmentation:

1. **Spatial leakage in cross-validation, quantified.** Random pixel-level 4-fold gives IoU
   0.930; grouping by image gives 0.824. **+0.106 inflation**, on identical rows and models.
   And a reusable diagnostic: the leaked protocol's fold sd is **0.003** against 0.050 for
   the honest one — four different specimens cannot agree to 0.003, and that implausible
   tightness is the tell. `code/crossval.py --demo-leak` reproduces both columns.
2. **Confirmation bias in interactive correction, quantified.** 98.3% of the human's
   force-crack pixels lie on pixels the model already called crack, because one-click region
   acceptance makes agreement far cheaper than disagreement. Only 1.7% of the label effort
   carries new information. This is a measurable property of the interaction design, not of
   the annotator.
3. **Pixel overlap is structurally blind to small-flaw detection.** Held out, object-level
   detection is **100.0%** for flaws over 20 k px and **25.3%** under 500 px — and small
   flaws are 0.3% of crack area, so missing three quarters of them barely moves an IoU. A
   tool can report 0.83 while failing at the task an NDT user cares about.
4. **A negative-control false-call rate belongs in the loop.** 4.0 spurious indications per
   frame (worst 11, one of six specimens clean) on material confirmed to contain no crack.
   Two models that scored normally on ground truth marked 22% of crack-free specimen as
   crack; only this check caught them.
5. **A learned segmentation network does not rescue n=4.** A 1.93 M-parameter U-Net under the
   identical leave-one-image-out protocol scores 0.6124 against 0.8320, losing 4/4 folds, and
   its worst fold is the out-of-distribution image (0.32 vs 0.78). Scoped honestly: one
   standard U-Net at a matched budget, not nnU-Net's full recipe.
6. **Cross-specimen generalisation fails, and is invisible within one group.** Leave-one-
   specimen-group-out on the human's own labels: crack recall 0.836 (B2), 0.795 (B3), 0.763
   (wrought), **0.397 (AM/HC)** — with 27 random images held out instead giving 0.755–0.816,
   so it is the specimen and not the training-set size.
7. **Annotation acceleration has a measurable ceiling on fine structures.** Superpixel
   tessellation, the standard trick, tops out at **IoU 0.700** even if every superpixel were
   labelled perfectly (8216 SLIC segments) — below the model's own held-out score. Ridge-
   filter region proposals capture 63% of known crack at 15% frame coverage.

Framed as *"the metrics used to govern interactive segmentation tools cannot see three of
their most consequential failure modes, and here is what each costs, measured"*, this is a
methods contribution with seven quantified results and reproducible code. That is a defensible
Q1 methods or tools paper.

## What it still needs

- **Dense ground truth in a second specimen group.** Every number above except #6 comes from
  four B2 images. `code/export_annotation_tiles.py` writes the uniform tile sample; 27 tiles
  are exported and awaiting annotation.
- **A second annotator on a subset**, to establish the label ceiling. ~2 px of boundary
  disagreement on a 65 px crack costs ~6 IoU points, so the gap to published Dice 0.964 may
  be labels rather than model — currently unknown.
- **More seeds and generations on the drift experiment**, or an explicit statement that the
  cause is unestablished. Do not publish it as a demonstrated mechanism on this evidence.
- **Claims to avoid** are listed in this repo's history and worth re-reading before writing:
  no "novel architecture", no "new metric", no "statistically significant" at n=4, no
  "validated on 71 images" when pixel truth exists for four.
