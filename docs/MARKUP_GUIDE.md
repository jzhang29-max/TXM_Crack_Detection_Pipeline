# Markup guide — mark all 71, then one command retrains

## 1. Start the tool

```bash
python3 code/paint_server.py
```
Open **http://127.0.0.1:8766**

Each specimen group is automatically served the input and model that works
best for it, so you are always correcting the cleanest available starting mask:

| group | input shown | why |
|---|---|---|
| B2, B3 | raw | flatfielding costs 0.169 IoU here (large-radius intensity features are ~41% of importance and flatfielding removes exactly that) |
| AM, Wrought | flatfielded | the mosaic tile grid is their dominant failure; flatfielding suppresses it ~10x. On raw, Wrought predicts 23.6% area / 204 regions vs 2.2% / 51 flatfielded |

First open of an image takes 10–30s while it predicts, then it caches.

## 2. What to mark

**The single most valuable thing: mark real cracks on AM and Wrought frames.**

Every crack pixel the model has ever learned comes from the 12 original B2
images. AM and Wrought have **zero** positive examples, which is exactly why
they mark the wedge rim instead of the thin centre cracks. Even 2–3 marked
frames per group changes what the model knows.

- **Add crack** brush — mark cracks the model missed. This is the high-value action.
- **Eraser** — mark false positives as background.
- **Click-to-remove** — clears an entire connected false-positive region in one click. Fastest way to kill the wedge rim and scattered speckle.
- **Save corrections** — persists to `paint/corrections/`, and regenerates that image's outputs immediately.

Corrections accumulate across sessions. You can stop and resume any time.

## 3. Check progress

```bash
python3 code/markup_status.py           # full table
python3 code/markup_status.py --todo    # just the summary + what to mark next
```

States: **HAND-MARKED** (has your force-crack labels), **NEG-ONLY** (only
automatic not-crack labels — contributes no positive signal), **UNTOUCHED**.

Current: 12 of 71 hand-marked, all B2. AM 0/27, Wrought 0/14, B3 0/13.

## 4. Retrain when done

```bash
python3 code/retrain_after_markup.py            # train + validate, report only
python3 code/retrain_after_markup.py --deploy   # ...and deploy if it wins
```

This sweeps class balance (`--neg-cap`), scores every candidate against ground
truth on **both** axes, and deploys only if a candidate beats the incumbent on
IoU without increasing false cracks on the crack-free specimens.

The double check is not ceremony. Every regression in this project passed a
single-metric check: flatfielding looked good on false positives and cost
0.169 IoU; a curvilinearity gate cut predicted area 8x — which reads as
artifact removal — while destroying 98% of true crack on one image
(recall 0.617 → 0.016). An over-aggressive filter and a good one both reduce
area; only recall against ground truth tells them apart.

If nothing passes, production stays put and it says so. That is the correct
outcome, not a failure.

## 5. Regenerate all outputs

```bash
python3 code/build_outputs_per_group.py    # masks + overlays + stats + montages
```

Writes `results/final_71_pergroup/`. Crack = **black** in the `_crack_mask.png`
files, matching the original convention.
