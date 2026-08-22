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

## Decision: do not switch

The gain lives entirely in a configuration that is not shipped. Capturing it would mean
dropping the 17-only member, which forfeits the ensemble's advantage elsewhere and is
untested on the axis that actually decides deployment.

Because every number here is scored on **labelled pixels** — and that is the exact blind spot
that let HistGradientBoosting win every such metric and then mark 7.9× more crack-free
specimen as crack (`REFERENCE_FRAMES_AND_HGB.md`). Against that risk, +0.001 IoU on the
shipping model does not justify adding a second 856 MB encoder to the install.

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
