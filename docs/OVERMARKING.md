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

---

# Why exported masks look like brush strokes (measured, 2026-08-24)

Reported as "dotted or brush-like structure I don't like". It is neither speckle nor a
rendering artefact: **the exported mask is the brush.**

On wrought_316L_fatigue_1200_cycles_crack:

| | % of frame |
|---|---|
| model alone, no corrections | 9.280% |
| the export | 8.178% |
| the owner's crack strokes | 5.919% |
| the owner's eraser strokes | 20.540% |
| in the export *only because* it was painted | **0.015%** (0.2% of the mask) |

The eraser covers 20.5% of the frame, so almost everything the model found outside a crack
stroke is removed, and inside the strokes the model already says crack at p>0.5 — so the gate
has nothing left to narrow. The mask that survives has the brush's shape.

**Width is the tell.** Median half-width along the centreline: model 28.4 px, export 16.3 px,
the strokes 15.0 px. The export's width *is* the brush's width.

**And the strokes are far wider than the cracks.** Isolating the darkest fifth inside each
stroke — the crack itself — gives:

| frame | stroke half-width | dark core | over-marked by |
|---|---|---|---|
| wrought_316L_fatigue_1200_cycles | 15.0 px | **1.0 px** | **15×** |
| HC_316L_fatigue_1200_cycles | 20.0 px | 3.0 px | 6.7× |
| HC_316L_fatigue_1650_cycles | 31.6 px | 6.1 px | 5.2× |
| b2_338_13 | 23.7 px | 9.0 px | 2.6× |

The brush shipped at **radius 24 — a 48 px wide stroke** — against cracks 2–18 px across.
Every label over-marked by construction, and the model learned it faithfully. Raising the
probability cut does not undo this: at p>0.95 the median half-width is still 12 px while area
collapses from 8.18% to 2.63%, so it deletes crack instead of narrowing it. **The width was
never in the training signal.**

## Two changes

**The brush default is now radius 8**, grounded in the dark-core measurement above. The
slider still spans 2–120 for erasing large regions.

**"Tight crack boundary"** (Advanced, off by default) narrows the accepted region using the
image rather than the label: Otsu on the pixels the mask already accepted, which is
parameter-free and per-frame.

| frame | as exported | tight | median half-width |
|---|---|---|---|
| wrought_1200 | 8.178% | 4.683% | 16.3 → 12.1 px |
| HC_1200 | 2.024% | 1.270% | 19.8 → 9.2 px |
| HC_1650 | 6.882% | 3.143% | 43.0 → 14.1 px |
| b2_338_13 | 27.329% | 13.867% | 57.2 → 11.7 px |

100% of the dark core is kept in all four.

**It is ON by default since 2026-08-24**, at the owner's instruction and with the cost
measured first:

| | tight off | tight on |
|---|---|---|
| of the crack area they painted, still marked | 99.97% | **61.91%** (43.6% worst frame) |
| predicted area, 6 crack-free specimens | 0.0230% | **0.0197%** |

Most of that 38-point difference is the over-marking this whole note is about — the strokes are
2.6–15× wider than the dark core. But faint crack that is not among the darkest pixels goes
with it, and the two cannot be separated without pixel-accurate reference annotation, which
this project does not have. `tight=0`, or unticking the box in Advanced, restores the wider
boundary.

**Painted pixels are exempt on the canvas.** Under `paste` an explicitly painted pixel is
never narrowed, because a stroke over a crack that is not among the darkest pixels would
otherwise produce no visible change at all — the same failure that disqualified `gate` as the
canvas default. Measured: painted crack stays at 100.00% on the canvas and narrows to 59.51%
in the export, on the same frame.

**Sub-floor components are judged by shape, not size.** Tightening splits a wide band into
thin threads, so the 2000 px speck floor would delete real crack; dropping it to 200 px kept
the threads but let 32 roundish blobs through across the corpus, and a round 200 px blob is
exactly what reads as a black dot. Below the floor, elongated survives and roundish does not:
of 107 sub-2000 px components, 75 elongated (aspect ≥ 3) are kept and all 32 roundish are
removed.

Neither change fixes existing labels. Thin cracks painted with a 48 px brush stay 48 px wide
until they are repainted and the model retrained on them.

**A single click must survive.** `spare` is exempt from the size floor *and* the shape rule. A
click is a round blob of about 1250 px, so the shape rule deleted it and clicking on the canvas
did nothing at all — caught by the selftest for "painting invalidates the cached overlay",
which is the only reason it was not shipped. An assertion the user drew is not judged on its
shape.

**Every path that renders or counts the mask had to be wired**, not just the export: the canvas
status bar was still reporting the wide area beside a narrowed picture, the ZIP export — the
main deliverable — was still writing wide masks, and thumbnails disagreed with the image they
link to. The overlay cache key gained the flag too, or toggling the box changes the URL and not
the picture. `_apply_flip_region` is the one deliberate exception: its result is written into
`correction.npy` as a label, and tightening there would bake Otsu's boundary into the owner's
own labels where turning the switch off could never undo it, then feed it to the next retrain
as if it had been drawn.

---

## The tightening rule, corrected (2026-08-24, later)

Reported after the default was flipped: *"a lot of the initial correct overlay disappeared."*
True, and the cause was the rule, not the idea. Otsu over the whole corridor is one threshold
for the entire frame, so faint crack in a brighter region falls out wholesale.

A **local** comparison — a pixel stays if it is darker than the mean of its own 301 px
neighbourhood — is better on both axes at once:

| rule | % of frame | median half-width | painted crack kept |
|---|---|---|---|
| wide, no tightening | 8.178% | 16.3 px | 100.0% |
| global Otsu | 4.698% | 10.0 px | 59.7% |
| **local mean, w=301** | 6.464% | **1.4 px** | **83.7%** |

1.4 px against the 1.0 px dark core actually present, while keeping 83.7% instead of 59.7%.
Across three frames retention went 59.7/43.8/62.4% → 83.2/62.8/78.3%, and predicted area on
the crack-free specimens improved (0.0230% → 0.0192%).

**The probability ridge was tried and rejected.** Using the model's own probability instead of
the image keeps ~100% of everything — and barely narrows, 12–28 px half-width. The model's
probability is flat-topped across the whole brush-wide band because that is what it was
trained on. It carries no width information; only the image does.

**And the rule is now verified per frame.** "Crack is darker than its surroundings" is an
assumption, and on some frames it is false. Scored against the darkest fifth of the corridor
the rule keeps a median of 97.0%, but 8 of 66 frames fall below 70% and 4 below 50% — worst
34.4% on B2_3_1_lbf, where narrowing deletes the crack rather than trimming it. So each frame
is checked and the tightening is **declined** where the assumption fails: 60 frames narrow, 6
are left alone. `TIGHTEN_MIN_CORE = 0.60` sits between the 49.8% and 65.1% frames.

The selftest asserts that contract, not a fixed retention number: either the core survives at
`TIGHTEN_MIN_CORE`, or the frame was left untouched. Silently deleting the crack is the one
outcome ruled out.
