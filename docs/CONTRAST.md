# Does contrast adjustment help the model find thin, faint cracks?

No. 19 arms across three families, none beat its own baseline by more than that baseline's
fold-to-fold spread. Everything with a measurable effect made it worse, ordered by how much
absolute intensity the transform destroys.

Protocol, identical for every arm: every labelled frame, up to 8000 crack + 8000 not-crack
pixels per image (seeded), the 17 hand-crafted features computed from the *transformed* image,
`MLPClassifier((64,32))`, 5-fold **grouped by image**. A thin frame is one whose darkest-fifth
core inside the owner's strokes has median half-width ≤ 3 px.

| arm | family | Δ IoU | Δ IoU thin | on-specimen FP |
|---|---|---|---|---|
| **identity** | — | — | — | **2.23%** |
| unsharp σ2 a1.0 | local | −0.0002 | **+0.0065** | 2.38% |
| 17 + CLAHE int + gradmag | augment | +0.0070 | +0.0044 | 2.00% |
| gamma 0.5 | global | +0.0026 | −0.0001 | — |
| 1–99 stretch *(control)* | global | −0.0016 | −0.0026 | — |
| CLAHE, 4 settings | local | −0.011 … −0.035 | −0.013 … −0.041 | 4.2 – 6.2% |
| histogram equalisation | global | −0.0226 | −0.0373 | — |
| LCN w151 | local | −0.0840 | −0.1022 | 9.92% |
| LCN w51 | local | −0.1191 | −0.1206 | 13.61% |
| LCN w51 robust | local | −0.0416 | −0.0235 | **21.87%** |
| dark-end stretch p0.5–p40 | global | **−0.1838** | −0.1384 | — |

## Four things worth keeping

**A brightness/contrast slider is a mathematical no-op here.** Any affine map is absorbed by
`StandardScaler`. Measured rather than asserted: a provably affine map shifts standardised
features by 0.032, all of it float32 cancellation in `local_std`'s `sqrt(E[x²] − E[x]²)`, and
the 1–99 stretch comes in *below* that floor at 0.0036. Only nonlinear transforms are visible
to the model at all (gamma 2.0: 15.6; equalisation: 25.2).

**Contrast does make cracks more separable — within one frame.** Inside a single thin frame,
local contrast normalisation lifts `texture_s2` from AUC 0.585 to **0.926**. It still fails,
because training pools frames: over the 33 thin frames that same feature moves only 0.5116 →
0.5219, while `intensity` collapses 0.6417 → 0.5293 and `smooth_s64` 0.6218 → 0.5115. LCN
scales by each frame's own statistics, so "dark for its neighbourhood" means something
different per frame and cannot be cashed in — and the absolute intensity that *was* comparable
across frames is gone. This is the mechanism behind the −0.169 flat-fielding result in
docs/MARKUP_GUIDE.md.

**An independent replication of that flat-fielding cost.** The `dark_stretch` arm saturates
60% of each frame to buy 2.5× contrast in the dark tail — what a person does by hand to see a
faint crack. It costs **−0.184 IoU**, against flat-fielding's −0.169, arrived at by a different
route, with the loss *entirely* in recall (0.548 vs 0.790) at unchanged precision.

**Whole-frame false positives are a trap; use on-specimen.** These mosaics are only 62–78%
specimen, and the raw-input model calls dark background crack, so whole-frame FP is dominated
by off-specimen area — 21.9% against 2.23% on-specimen, in the *baseline*. Read whole-frame
alone and `equalize_hist` looks like a 4× win when it is 2.9× worse on the specimen, and
`lcn_w51_robust` looks like the mildest LCN variant at −0.042 IoU while predicting crack on
**21.9% of on-specimen pixels in frames confirmed to contain none** — ~10× baseline. Adding
contrast channels beside the 17 also looked like a 2.4× FP win; on-specimen it is flat.

## Limits

These are 17-feature models with **no SAM embedding**, so absolute IoU (~0.67) and FP levels
are far from the deployed ensemble and only arm-vs-arm within a family is meaningful. The
untested gap is whether contrast helps the **SAM** path — that needs re-embedding per arm
(~40 min each) and was not run. The thin/thick split uses the darkest-fifth core, and 15 of
the qualifying frames sit exactly at a 1 px floor, so it separates thin from thick but is not
a physical width.

Scripts and raw results: `research/contrast/`.

## What this means in practice

Adjust the display all you like — the app already flat-fields and percentile-stretches the
*display* view for exactly that reason, and it helps a person see a faint crack. Do not feed
it to the model. The lever that would actually improve thin-crack detection is thinner labels
and a retrain, not preprocessing.
