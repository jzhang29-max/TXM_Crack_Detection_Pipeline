# Tile seams

The SAM embedding is computed per 1024 px tile. Until this change the tiles **abutted**, and
the embedding stepped across every interior boundary. The step survived into the output as a
visible seam.

## What the seam measured

Scored as mean |Δp| along a line, against the median line of the same frame. Ranking *every*
line rather than checking the boundaries I expected mattered — the first version of this
metric assumed boundaries at multiples of 1024 and found nothing, because `tiles()` clamps
the last tile inward.

| frame | worst line | ratio |
|---|---|---|
| HC_316L_fatigue_1650_cycles (3044×2354) | y=1023 | **32.8×** |
| b3_amb crack-free (3702×3737) | y=1023 | 9.0× |

Averaging over boundaries hid most of it: the same frame scored 12.3× when y=1023 was
averaged with y=1329, a boundary that has 718 px of overlap and barely seams.

## Why the geometry makes lookup-time fixes impossible

`tiles()` steps by exactly `TILE` and clamps only the **final** tile inward. A 3914 px frame
sits at x = 0, 1024, 2048, 2890 — two abutting boundaries with **zero** shared data, and one
182 px overlap. Zero overlap means no real embedding spans the boundary, which kills both
cheap fixes:

| lookup | worst row | crack area | FP on 3 crack-free |
|---|---|---|---|
| last-tile-wins (was shipped) | y=1023 **32.8×** | 8.25% | 0.038% |
| reach 64 px past each tile edge | y=672 5.5× *(fixed)* | 8.34% | **0.231% (6.1×)** |
| window vanishing at tile edge | y=1023 **32.8×** *(unchanged)* | 8.24% | 0.113% (3.0×) |

- Reaching past an edge **has to invent data** — it reads a tile's clamped edge cells as if
  they were a measurement. It removes the seam and costs 6.1× the false positives.
- A window that stops at the edge **cannot fix anything** where only one tile covers the
  pixel: it reduces to last-tile-wins exactly, 32.8× to three digits. It still shifts false
  positives 3.0×, because on the *last* boundary — the clamped one, with real overlap — it
  averages two real embeddings flatly across a band the model never trained on. Worst of both.

## The fix

Embed at `TILE_STRIDE = 896` so adjacent tiles overlap by 128 px, then blend with a Hann
window over each tile's own extent (`model.emb_rows`). Costs 13% more tiles — 1190 vs 1050
over the corpus, ~37 min to rebuild, 2.5 GB instead of 2.2 GB.

Validated on two frames before paying for all 71:

| frame | lookup | worst row | area |
|---|---|---|---|
| seam frame | last-wins @1024 | y=1023 32.8× | 8.255% |
| | hann @896 | y=677 **7.8×** | 7.746% |
| crack-free | last-wins @1024 | y=1023 9.0× | 0.019% |
| | hann @896 | y=2 **5.6×** | 0.080% |

y=677 is not a tile start (at stride 896 they are y = 0, 896, 1330) and was already 5.8× in
the baseline — it is crack, not an artifact. Area on the cracked frame went **down**, so
nothing was invented.

Two details that cost real time to find:

- **The window must vanish at each tile's own edge, with no floor.** A 1e-3 weight floor put
  a **0.4999 step** back in: along the frame's top row every window is ~0, so the floor
  dominated and made it a flat unweighted average, stepping wherever a tile joined the blend.
  Without the floor that row is still correct — the shared row factor cancels in the
  normalisation. Guarded by a selftest that checks the top, middle and bottom rows.
- **Training and inference must share the lookup.** Fitting on last-tile-wins and serving
  blended embeddings took crack-free false positives from 0.019% to 0.080% on its own, with
  the weights untouched. `emb_rows` is the single definition; `read_emb` rejects any cache
  built at a stride that leaves tiles abutting, so a stale cache cannot silently reintroduce
  the mismatch.

## Not fixed here: the frame border, which is now the dominant artifact

Ranking every line rather than the boundaries I expected also surfaced a **frame-border**
artifact, unrelated to tiling and larger than the seam ever was. Measured on the deployed v4
output:

| frame | worst border line | ratio |
|---|---|---|
| b3_amb | x=16 | **62.9×** |
| B2_2_9_lbf | x=16 | 40.2× |
| HC_316L_600_cycles | y=6346 (bottom edge) | 12.8× |
| b2_343_75_LARGE | x=6365 (right edge) | 13.6× |

It is invariant to the embedding path — 134.8× before the seam fix and 136.9× with the
lookup-time blend — so it is not a tiling effect. **And it marks pixels.** In the final
pruned mask on the six crack-free specimens, the outer 24 px band is marked far more than the
interior:

| specimen | border <24 px | interior | ratio |
|---|---|---|---|
| wrought_316L_0_cycles | 2.362% | 0.021% | **113.6×** |
| b3_amb | 1.623% | 0.041% | 39.3× |
| b3_3_18lbf_348_13um | 2.674% | 0.199% | 13.5× |
| B2_2_1_lbf, B2_amb, B2_2_9 | 0.000% | ≤0.049% | — |
| **mean** | **1.110%** | **0.052%** | **21.5×** |

The band is ~1.2% of frame area, so it contributes roughly a fifth of the false positives
that survive pruning, on three of six specimens.

**It is not a padding bug.** The borders contain no constant columns; they are genuinely
darker than the interior (mean 0.31 against 0.43, and similar on every specimen) — the usual
TXM mosaic illumination rolloff. A model whose strongest features are multi-scale intensity
reads dark as crack, so it is behaving as trained. Flat-fielding the model input is not the
answer either: it was tried and cost 0.169 IoU, because large-radius intensity features carry
~41% of the model's importance.

The fix is training signal, not code: a few eraser strokes along the frame edges of two or
three specimens and a retrain. That is exactly what the correction workflow is for, and it is
deliberately left to the owner rather than solved by synthesising border labels nobody drew —
this project's whole gate rests on no label existing that the owner did not paint.

---

# Speck debris from erasing (fixed separately)

Exported masks carried **tiny black dots that look nothing like crack**. They were not a
downscaling artefact — that was checked first and ruled out: resampling the full-resolution
mask to 812 px by nearest, bilinear, box and Lanczos all produced 2–3 specks, not the dozens
visible.

`prune_specks` promises no crack blob under `MIN_BLOB_PX` = 2000. Corrections were applied
**after** that prune, which quietly undid it. An eraser stroke does not only remove area: it
cuts *through* blobs and leaves the offcuts behind as separate sub-floor components.

Measured on b2_343_75_LARGE (6367×3691, 19.9% crack), the worst frame of 71:

| corrections | components | under 200 px | crack area |
|---|---|---|---|
| none | 19 | **0** | 20.005% |
| gate / paste (before) | 70 | **51** | 19.865% |
| gate (after) | 19 | **0** | 19.864% |

Crack area moves by 0.001 pp, because the specks are tiny by definition — this is a
cosmetic-looking defect with a real cause, not a trade-off.

The fix re-prunes after corrections, with one asymmetry that matters:

- **paste** (the canvas) spares any component containing a painted pixel, at any size. There a
  stroke is an assertion, and silently deleting a deliberate dab smaller than the floor is the
  one thing painting must never do.
- **gate** (the export) spares nothing. A gated pixel is not an assertion that this is crack,
  it is "believe weaker evidence here" with the boundary still drawn by the image — so a
  stroke over a region the model barely likes shatters into slivers at the floor, and those
  slivers *are* the specks. Sparing them would preserve the artefact on the one path that
  produces the deliverable.

Verified across all 71 frames in export mode: **0 components under 2000 px**. Guarded by a
selftest that samples 8 frames, confirmed to fail when the second prune is removed.
