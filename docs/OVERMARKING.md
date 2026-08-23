# The predicted crack is ~1.7× wider than the label, and four levers do not fix it

Reproduce with `research/code/pilot_stride8.py`, `pilot_threshold.py`, `pilot_features.py`.

## The observation

The owner reported that exported masks over-mark: the crack is drawn wider than it is. That
is measurable and real. Mean local thickness from the medial axis, on thin-crack frames:

| frame | owner's strokes | model @0.5 |
|---|---|---|
| HC_316L_fatigue_600 | 15.6 px | 56.4 px |
| HC_316L_fatigue_800 | 16.4 px | 53.9 px |
| wrought_800_cycles | 18.4 px | 37.5 px |
| wrought_900_cycles | 43.6 px | 48.5 px |

And 37–48% of every mask is brighter than its frame's median intensity — for a feature that
is defined by being dark.

Note the direction: **the owner's brush strokes are the tighter boundary, by up to 3.6×.** The
over-marking belongs to the model, not to the painting. This is the same defect as the AM/HC
precision of 0.355 ("marks 3–12× too much material") seen from the other end.

## Four levers, measured

All four use the same protocol: 8 crops of 1024×1024 centred on painted crack,
leave-one-crop-out, the deployed architecture, IoU against the owner's strokes.

| lever | mean IoU | mean thickness | cost |
|---|---|---|---|
| baseline (stride 16, all 17 features, @0.50) | 0.1707 | 36.2 px | — |
| raise threshold to 0.90 | 0.1635 | 35.3 px | free |
| drop `smooth_s32` | 0.1601 | 35.1 px | free |
| drop `smooth_s32`+`s64` | 0.1629 | 35.9 px | free |
| drop `s16`+`s32`+`s64` | 0.1657 | **37.3 px** | free |
| **halve the embedding stride to 8** | 0.1555 | **33.2 px** | **7.2 h + 8.4 GB** |

Plus one post-processing route measured on the full frames: keeping only the darkest 70%
inside the mask, seeded from the interior so a thin crack cannot be erased wholesale, moved
area 22.78% → 22.53% and the bright fraction 37.7% → 36.6%. Negligible.

**Every lever trades accuracy for thinness and none improves localisation.** The stride-8
result is the most efficient trade — it reaches a thinness thresholding cannot, and it was the
one worth 40 SAM passes to check — but IoU fell in 8 of 8 folds, and at 33.2 px it is still
1.5× the label. Dropping the large smooths, the intuitive fix given that σ=64 cannot represent
a sharp edge, is *worse*: less accurate, and thicker when three are dropped.

## Why the 7.2-hour re-embed was not done

Stride 8 buys 15% thinner for 11% less accurate. That is a different operating point on the
same curve, not a better model, and the Sensitivity slider already offers points on that
curve for free. Spending 7.2 hours of embedding, 8.4 GB of cache, a retrain and a
re-measurement of every published number to move along a curve is not a good trade. It would
also have to be re-justified afterwards, because every number in this repo would change.

The one thing stride 8 *would* fix independently is the tile-seam artifact: probability jumps
at 1024-px SAM tile boundaries measured at 8.9× (vertical) and 12.6× (horizontal) the normal
column-to-column gradient, worst at x=4096. That produces the straight edges and square
corners visible in some masks. It is a real defect and it remains open.

## What the measurements actually point at

There is a ceiling on this question that none of the four levers can cross: **every IoU above
is scored against the owner's brush strokes**, and the owner's own report is that those
strokes over-mark. If both the label and the prediction are too wide, IoU against the label
cannot measure over-marking, and optimising it cannot reduce it.

So the blocker is not resolution or feature scale. It is that no tight reference exists. The
way forward is a small amount of deliberately tight ground truth — a few hundred-pixel windows
annotated at pixel precision rather than with a brush — after which over-marking becomes
measurable, and only then optimisable. That is the same missing piece as the unresolved AM/HC
precision question and the absent second annotator.
