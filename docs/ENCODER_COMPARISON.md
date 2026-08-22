# Is a newer SAM encoder a better feature source than SAM 1 ViT-H?

Reproduce with `python3 code/experiment_encoders.py --arms sam1,sam2`. Raw numbers in
`research/encoder_compare.json`.

## What was actually asked

`app/core/model.py` uses SAM as a **frozen image encoder** and never calls its prompt
encoder or mask decoder — zero-shot SAM prompted the way SAM is designed to be prompted
measures 0.23–0.36 IoU on this data against 0.82 for the trained hybrid. So SAM 3's headline
capability, open-vocabulary concept segmentation from a text prompt, is the part of it this
project has least use for: there is no noun phrase for a hairline fatigue crack in a
grayscale X-ray of 316L that a web-trained model holds a prior on.

The only question worth measuring is whether a newer encoder produces more discriminative
**dense features**.

## Why SAM 2 and not SAM 3

`facebook/sam3` is `gated=manual` — Meta reviews each access request individually, and until
one is granted the API answers `403 ... you are not in the authorized list`. The
`facebookresearch/sam3` GitHub repository does not route around this: it ships code, not
checkpoints, and there is nothing to fine-tune from. A randomly-initialised Perception
Encoder would produce features far worse than SAM 1's trained ones and would answer nothing.

SAM 2.1 Hiera-Large is ungated, and its Hiera backbone is a genuinely different architecture
from SAM 1's plain ViT — so it is a real test of the same hypothesis rather than a
consolation prize. The `sam3` arm stays wired in and reports itself skipped; when access is
granted the same command produces a three-way comparison.

## The comparison is like-for-like by measurement, not by assumption

All three encoders emit **256 channels** (`fpn_hidden_size`), so the 273-d feature vector is
unchanged and nothing downstream is touched. Only SAM 1's stride is knowable in advance —
SAM 2 and SAM 3 report `backbone_feature_sizes`/`scale_factors` as `None` in transformers
because those arrive with the checkpoint — so `code/encoders.py` reads the shapes off the
tensors it receives and selects the pyramid level nearest SAM 1's stride 16. Measured: SAM 2
gives **256 channels at stride 16**, identical granularity to SAM 1.

Rows are **paired on identical pixels**: the same uniformly-sampled indices are looked up in
both encoders, so the 17 hand-crafted columns are byte-identical between arms and only the
256 embedding channels differ. Leave-one-frame-out over the four dense reference frames,
120,000 px per frame, two seeds.

## Results

| encoder | model | IoU | AUC | recall @5% FPR |
|---|---|---|---|---|
| SAM 1 ViT-H | ensemble | 0.8192 | 0.9812 | 0.9227 |
| SAM 2 Hiera-L | ensemble | **0.8203** | **0.9826** | **0.9315** |
| SAM 1 ViT-H | hybrid alone | 0.7845 | 0.9746 | 0.8902 |
| SAM 2 Hiera-L | hybrid alone | **0.8058** | **0.9799** | **0.9118** |

Paired t-test, SAM 2 − SAM 1, n = 8 (4 frames × 2 seeds):

| model | metric | mean diff | t | p |
|---|---|---|---|---|
| ensemble | IoU | +0.0011 | 0.16 | 0.87 |
| ensemble | AUC | +0.0013 | 1.41 | 0.20 |
| hybrid alone | IoU | +0.0213 | 2.23 | 0.061 |
| hybrid alone | AUC | +0.0052 | 2.74 | **0.029** |

## Reading it

**For the model that would actually ship — the ensemble — SAM 2 buys nothing.** +0.0011 IoU
at p = 0.87.

**For the hybrid member alone, SAM 2 is genuinely better**: +0.021 IoU, +0.005 AUC at
p = 0.029. SAM 2's features are more discriminative; the ensemble conceals it, because it
averages in a 17-feature MLP that is identical across arms and therefore halves whatever the
encoder contributes.

**Where it improves is the informative part.** Per-frame, hybrid alone:

| frame | SAM 1 | SAM 2 | diff |
|---|---|---|---|
| LARGE_343_75 (23.5 MP mosaic) | 0.6140 | 0.6642 | **+0.0502** |
| 333_75_um_zoom | 0.7847 | 0.8155 | +0.0309 |
| 336_25 | 0.8731 | 0.8771 | +0.0040 |
| 338_13 | 0.8663 | 0.8664 | +0.0001 |

Nothing on the two easy frames, and the gain concentrated on the large mosaic — which is
precisely the weakness that motivated the ensemble in the first place. `model.py`'s own
docstring records that the hybrid alone "loses badly on the one 23.5 MP mosaic," which is why
a weaker 17-feature model is averaged in at all. SAM 2 partially repairs that specific
failure.

## The deciding axis: false positives on crack-free material

Run with `python3 code/experiment_encoders_fp.py --arms sam1,sam2`. Raw numbers in
`research/encoder_fp.json`.

Trained on the owner's corrections across all 66 labelled images — the composition the
deployed recipe uses — 962,096 paired rows, then measured on the six specimens confirmed to
contain no crack, where every positive is by definition a false positive. Measured by uniform
pixel sampling (250,000 px per specimen) rather than full-frame prediction: on crack-free
material the sampled positive rate is an unbiased estimate of the predicted area fraction at
~1/1000th the compute. Unpruned, because speck pruning cannot apply to scattered pixels —
both arms identically so, leaving the paired difference unaffected.

| specimen | SAM 1 | SAM 2 |
|---|---|---|
| B2_2_1_lbf | 0.384% | 0.316% |
| B2_2_9_lbf | 0.445% | 0.560% |
| B2_amb_mosaic_2 | 0.129% | 0.236% |
| b3_amb | 0.186% | 0.531% |
| b3_3_18lbf | 0.345% | 0.796% |
| wrought_316L_fatigue_0_cycles | 0.269% | 0.191% |
| **mean** | **0.293%** | **0.439%** |
| worst | 0.445% | 0.796% |

Paired difference **+0.146 pp** (sd 0.215, SE 0.088, n=6, t=1.66, p≈0.16). Worse on four of
six specimens, better on two. **Inside** the gate's 0.5 pp tolerance, and not statistically
significant.

**Sanity check on the harness.** SAM 1 reads 0.293% unpruned here against the deployed
model's 0.250% measured after pruning. Those should be close, with pruning accounting for the
gap, and they are — evidence this measures realistic behaviour.

### A false start worth recording

The first version of this test trained only on the four reference frames and reported 26–33%
false-positive area, roughly a hundred times the deployed model's 0.25%. That was a
training-composition artifact, not an encoder property, and it was predictable from this
project's own data: the `gt-only` arm in `research/fp_attribution.json` hits 42% FPR on
held-out groups for the same reason — crops that are 18–30% crack teach a model to over-call.
It also reversed the apparent magnitude: SAM 2 looked +4.5 pp worse under that composition
and is +0.15 pp worse under the real one. A narrow training set amplified a small encoder
difference roughly thirtyfold.

It also cost three hours to discover, because that version predicted every pixel of six
mosaics — ~200 M MLP evaluations per arm — and had finished two specimens in that time.
Sampling replaced it.

## Decision: do not switch

Three axes, one conclusion:

| axis | SAM 2 vs SAM 1 | verdict |
|---|---|---|
| ensemble IoU (what ships) | +0.001, p=0.87 | nothing |
| hybrid member alone | +0.021 IoU, +0.005 AUC, p=0.029 | real |
| crack-free false positives | +0.146 pp, p=0.16 | nominally worse, within tolerance |

SAM 2's features are genuinely more discriminative in isolation. That advantage disappears in
the model actually shipped, and it arrives alongside a nominal false-positive cost pointing
the wrong way. Capturing the gain would mean dropping the 17-only member — forfeiting the
ensemble's advantage elsewhere, and on the false-positive axis that is untested.

Paying for a second 856 MB encoder download in every tester's install to gain +0.001 IoU is
not a trade worth making.

## What this predicts about SAM 3

A full architecture generation (ViT-H → Hiera) bought +0.005 AUC on the isolated member and
nothing on the ensemble. SAM 3 is therefore unlikely to be transformative here, which agrees
with the rest of the evidence: the binding limitation is AM/HC label coverage — precision
0.36, still undecided between model error and annotator disagreement — not the encoder.

## Note on dependencies

`Sam2ImageProcessor` requires `torchvision`, which is deliberately **not** in
`requirements.txt`: the app does not need it and no tester should install it for a comparison
they will never run. `pip install torchvision` pins `torch==2.13.0`, the version already
present, so the app environment is unaffected.
