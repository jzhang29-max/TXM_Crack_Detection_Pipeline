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

## Not fixed here

Ranking every line surfaced a **frame-border** artifact, unrelated to tiling and larger than
the seam: x=17 measured **134.8×** on the crack-free specimen and **136.9×** after the seam
fix — invariant to the embedding path, so it comes from the 17-feature stack's padding in the
first ~16 px, not from SAM. Untouched by this work.
