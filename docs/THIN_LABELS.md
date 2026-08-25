# Training on the crack, not on the brush

The exported masks looked like brush strokes because they were. This is the measurement and
the fix.

## The premise, over all 61 painted frames

| | median half-width |
|---|---|
| the painted stroke | **26.93 px** |
| the dark crack inside it | **3.16 px** |
| what the model trained on those strokes predicted | **28.46 px** |

Correlation of predicted width with **label** width: **0.810**. With **crack** width: **0.304**.
The model was reproducing the brush. No threshold or morphology recovers a width that was never
in the training signal — pushing the probability cut to 0.95 still left a 12 px half-width while
discarding two thirds of the area, deleting crack rather than narrowing it.

## The fix

Before sampling crack rows, narrow `correction == 1` to its dark core (the same local-threshold
rule as `tighten_to_image`), and let the discarded ring fall into the **negative** pool at its
true area weight. `correction.npy` is never modified; the narrowing happens at training time.

**The ring has to be negative, not merely dropped.** Leaving it unlabelled scores well and does
not narrow the output at all:

| training labels | IoU (fixed target) | predicted half-width |
|---|---|---|
| as painted | 0.6475 ±0.023 | 13.18 px |
| core, ring **unlabelled** | 0.7111 | 14.18 px — no better |
| core, ring **negative** | **0.7382**, wins 5/5 folds | **8.03 px** |

Removing those pixels from training takes away nothing that pushes the boundary inward. Calling
them background does. Per-pool positive rates make it explicit — the outer ring goes from 0.697
to 0.206 while the core holds at 0.851.

## What shipped

`thincore_v5`, at a served threshold of **0.60** rather than 0.50. The threshold is part of the
change: a thinner mask sits lower in probability, so at a matched 0.50 v5 marks 2–3× more
crack-free area than v4. Matched on false alarms instead, it wins outright.

| | v4 @ 0.50 | v5 @ 0.60 |
|---|---|---|
| held-out grouped IoU | 0.789 ±0.040 | **0.811 ±0.023** (own recipe baseline) |
| precision / recall | 0.933 / 0.837 | 0.936 / 0.860 |
| crack-free area, unpruned | 0.209% | **0.174%** |
| crack-free area, pruned | 0.035% | 0.046% |
| false indications / frame | 1.83 | 2.0 |
| mask half-width | 22 px | **5 px** |

The IoU figures are not directly comparable — v5's target is the narrowed label, which is why
it carries its own `RECIPE` tag and was gated against the absolute floor rather than against
v4's number.

## The narrowing left the centres unfilled

Reported from a black-and-white export: the middle of a wide crack came out speckled rather
than solid. The cause is ordering. `effective_mask` fills small holes at
`FILL_HOLES_MAX_PX = 1024`, then hands the corridor to `tighten_to_image`, which re-cuts the
boundary from the image — so every void the local threshold opens is created *after* the only
step that would have closed it. Nothing was wrong with either step; the fill just ran too early
to see the holes.

| enclosed voids | before narrowing | after narrowing | after the second fill |
|---|---|---|---|
| b2_343_75_LARGE | 177 | 35,572 | 38 |
| b2_338_13 | 25 | 2,724 | — |
| wrought_316L_fatigue_1200_cycles | 80 | 3,732 | — |
| **all 71 frames, total** | | **342,963** | **894** |

The fix is the same `remove_small_holes` at the same 1024 px, applied a second time after the
narrowing, intersected back with the corridor so it stays a subset of what the detector
accepted. Swept over all 71 frames, 1024 is the best of 64 / 256 / 1024 on every axis at once:

| post-narrowing cap | none | ≤64 px | ≤256 px | ≤1024 px |
|---|---|---|---|---|
| enclosed voids, all frames | 342,963 | 2,471 | 1,145 | **894** |
| recall on painted crack | 74.74% | 76.02% | 76.32% | **76.58%** |
| on-specimen FP, 6 crack-free frames | 0.0040% | 0.0040% | 0.0040% | 0.0040% |
| leak into painted not-crack | 0.000% | 0.000% | 0.000% | 0.000% |
| mean predicted area | 5.979% | 6.083% | 6.107% | 6.125% |

Recall rose on 58 of 61 painted frames and fell on none. The false-positive axis does not move
at any cap because filling cannot add area on a frame with no enclosed voids, and a crack-free
frame has none — the operation is inert exactly where a false alarm would be expensive.

### Two treatments that measured better and looked worse

- **Radius-1 closing** removes fewer pixels for a comparable drop in void count, which is why
  it was tried first. skimage's `disk(1)` is a 3×3 cross, and closing stamps that shape onto
  every void it fails to remove: the export picks up a lattice of diamonds, and `square(3)`
  gives boxes. Filling imposes no shape of its own — it removes a void or leaves it.
- **Shape-aware filling** — fill only the roundish voids, by the same aspect test
  `prune_specks_keeping` applies to specks — sounded more principled and keeps the speckle:
  5,956 voids left on b2_343_75_LARGE against 38 for a plain cap, because the dust is mostly
  1–2 px slots with no meaningful aspect ratio. A speck's shape says whether it is crack; a
  void's does not.

### The measurement that argued against the fix

Median half-width jumps 18 px → 36 px under filling, and on that number the whole approach was
rejected once. It is a distance transform over a mask whose *topology* changed: removing
interior voids moves the medial axis outward without moving the outline, which filling cannot
move by construction. Area is the honest number and it moves +0.15 pp. Rendering the two masks
at native resolution settles it in a way neither number did.

Filling is off for thin-core label sampling (`fill_voids=False`). Widening the training core
would change what `thincore_v5` means while the model already on disk stayed fitted the old
way, and a recipe tag that no longer describes its own data is worse than no tag.

## Limits

- **~5 px against a 2.5–3 px core.** Label refinement closes roughly half the width gap, not
  all of it. The remainder needs labels drawn at crack width in the first place; the brush now
  defaults to radius 8 instead of 24.
- **The dark core is a proxy**, not physical ground truth, and the IoU target inherits its bias
  (it counts the ring as negative, which favours arms trained to reject the ring). The width
  and crack-free axes are the label-free ones, and they agree.
- **Tightening declines on 2 of 61 frames**, where "crack is darker than its surroundings" does
  not hold; those keep their original labels.
- False indications rose slightly, 1.83 → 2.0 per frame. Thinner masks fragment more easily.
  Area fell 4.6× at the same time, so the trade is favourable, but it is a trade.

Scripts and raw results: `research/thinlabels/`.
