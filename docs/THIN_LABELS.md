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
