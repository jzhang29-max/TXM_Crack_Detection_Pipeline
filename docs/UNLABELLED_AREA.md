# The quarter of the mask nobody ever checked, checked

Run 2026-09-19. Artifacts: `crack-evolution-5d/out/txm_unlabelled_adjudication.json`,
figure `results/blind_adjudication.png`.

## The gap

Every accuracy number on this corpus — clDice, Tprec, Tsens, IoU — is computed inside the
labelled domain, because outside it there is no truth to compare against. So none of them
can see how much of the mask falls where the annotator never painted anything. Measured,
area-weighted over 64 frames:

| | share of ACCEPTED area never labelled either way |
|---|---|
| corpus | **24.7%** |
| AM / HC_316L | 51.1% |
| B2 | 36.9% |
| Wrought | 10.7% |
| B3 | 4.7% |

On `b2_336_25` it is 17.3% of the whole frame and 95% of everything the detector marked
there. That area is not wrong — it is **unchecked**, and it is the largest single hole in
the evidence behind every headline number.

## Design

88 fields of 700×700 px native (~20 µm), each centred on a random point INSIDE a region,
rendered through the app's own display transform with **no overlay drawn**. A fixed window
rather than a fit-to-region crop, because region size differs systematically by class and a
fit-to-region crop hands the voter the class through the zoom level. IDs shuffled behind a
fixed seed; nothing in a filename encodes its class; voters were told nothing about how many
fields contain cracks and were instructed not to balance their answers.

Five classes, four of them controls whose answer is already known — without those, a verdict
on the test class is just an opinion about pictures:

| class | what it is | n |
|---|---|---|
| **UNLAB** | model accepted, human never labelled it — **the question** | 26 |
| POS | model accepted AND human painted crack — positive control | 26 |
| NEGP | model accepted BUT human painted not-crack — known disagreement | 8 |
| NEGF | model accepted on a specimen asserted crack-free throughout | 2 |
| NEGM | not accepted, painted not-crack — plain matrix, the floor | 26 |

Three independent panels with different lenses (morphology / conservative, penalised for
false alarms / "would you paint here"), different batch groupings, 3 votes per field,
33 agents, 0 errors.

**Leak guard, run before the panels.** If crop mean brightness, contrast or dark-fraction
separated the classes, voters could score those instead of crack morphology and the whole
exercise would be measuring the renderer. They do not: POS vs UNLAB p = 0.96 / 0.96 / 0.98,
POS vs NEGM p = 0.92 / 0.25 / 0.52. Source frames are spread 20 / 23 / 24, so no class is
carrying one frame's texture.

## Does the instrument work

| | majority called CRACK |
|---|---|
| POS (painted crack) | **24 / 26** |
| NEGM (plain matrix) | **4 / 26** |

Fisher p = 1.2e-08, Mann-Whitney on vote score p = 2.3e-08. The panel separates crack from
matrix. Its own false-negative rate is 2/26 on painted crack — that is the error bar on
everything below.

## The answer

| class | majority CRACK | mean vote score |
|---|---|---|
| POS | 24/26 (92%) | 0.891 |
| **UNLAB** | **25/26 (96%)** | **0.859** |
| NEGP | 7/8 (88%) | 0.833 |
| NEGM | 4/26 (15%) | 0.212 |
| NEGF | 0/2 | 0.000 |

UNLAB vs POS: **p = 0.31** — statistically indistinguishable from crack the human painted.
UNLAB vs NEGM: **p = 4.4e-08**.

It survives the strictest lens alone. Panel B was told it would be heavily penalised for
false alarms and to default to "no"; its overall yes-rate is 49% against 75% for Panel A.
On Panel B alone: POS 20/26, UNLAB 16/26, NEGM 2/26 — UNLAB vs POS p = 0.37, UNLAB vs NEGM
p = 4.2e-05. The conclusion does not depend on a permissive voter.

Area-weighted by component size: **93.4%** of the unlabelled area sampled reads as crack.
The sample was drawn from components ≥ 4000 px, which hold 87.8% of all unlabelled area, so
the claim covers **21.7% of all accepted area** — not the last 3%.

**The unlabelled quarter of the mask is crack the annotator did not reach.** The detector
is finding crack outside the strokes, and every metric this project reports is blind to it
by construction.

## The one real false positive

`field_085`, 27,179 px on `wrought_316L_fatigue_1280_cycles`. All three panels rejected it
independently and gave the same reason: a large black region with metal on one side only,
bordered by parallel Fresnel fringes — a specimen **free surface**, not a crack. One in 26.

## An unexpected result, reported with its caveat

NEGP is the class where the model accepted a region the human had painted as **not-crack**.
The blind panel sided with the **model** on 7 of 8, mean score 0.833 against 0.891 for
painted crack — five of them unanimous. The one it rejected (`field_072`) is 94% off-specimen
black on a crack-free control.

**This is not a measurement of annotation accuracy and must not be quoted as one.** Those 8
are the LARGEST model/human disagreements — components ≥ 4000 px inside painted not-crack —
not a random sample of that area. Selecting the biggest disagreements and finding the model
looks right on them is what selection does. What it does support: the 0.197%-of-painted-not-
crack figure quoted as the detector's false-positive rate is an **upper bound**, because some
of that 0.197% is the annotator, not the model.

## The sub-4000 px components, checked (added 2026-09-20)

They are not 12,303 detections. `MIN_BLOB_PX = 2000` means no ACCEPTED component is that
small -- these are the unpainted REMAINDER of large accepted components after intersecting
with the label map, i.e. the rim of a stroke that did not quite cover the crack under it.

| | |
|---|---|
| exactly 1 px | 6,921 comps (56.3%), 0.09% of unlabelled area |
| 2-9 px | 3,141 comps, 0.13% |
| within 25 px of a painted stroke | **80.4% of their area** |
| median distance to nearest stroke | **11 px** |

Filtering to where "nobody checked this" is a real claim -- >= 200 px AND >= 200 px from any
stroke -- leaves **32 components = 0.229% of all accepted area**. That is the entire
exposure even if every one were wrong.

Second blind run, same design, 400 px fields so a test component's 200 px clearance means no
painted crack can appear in frame. Positive controls stratified to match the test sizes
(median 2418 vs 2403 px, p = 0.94; a first attempt matched at p = 0.003 and was rebuilt,
since larger controls would have biased the comparison toward calling the test class crack).
Leak guard clear at p >= 0.089.

| class | majority CRACK | score |
|---|---|---|
| painted crack, size-matched | 18/32 (56%) | 0.594 |
| **isolated small indications** | **11/32 (34%)** | **0.417** |
| crack-free specimens, FP by assertion | **0/4** | 0.000 |
| plain matrix | 4/26 (15%) | 0.192 |

Above matrix (p = 0.020), not separable from real crack (p = 0.066). **The instrument is
weak at this size**: it recognises only 56% of KNOWN crack here against 92% on the large
regions. A Rogan-Gladen correction for that gives a bootstrap 95% CI of **0-100%**, so no
prevalence is quoted. This class is unresolved, and it is 0.23% of the mask.

A limitation of the method at this scale: each crop shows a 400 px FIELD, not the component
in isolation, so a voter answers "is there a crack here", not "is this component a crack".
Two of the five artefacts below were called crack, and the panel's written evidence shows it
was describing a different feature in the same field.

## A new artefact class: cracks do not run straight

Five of the 32 read as obvious cracks on every summary statistic -- elongation 24 to 197,
12 to 49 sigma darker than their surroundings. The panel rejected them. Geometry settles it:
they are **vertical lines wandering 0.13 to 0.93 px over 171 to 788 px of length**, width
sd 0.37 to 2.19 px. A crack at that aspect wanders tens of pixels.

Censused over all 71 frames, standalone components of >= 200 px with bbox aspect >= 6:

| | n | wander | area |
|---|---|---|---|
| straight-line artefacts | 5 | 0.13 - 0.71 px | 15,241 px = 0.047% of accepted |
| one painted crack | 1 | 1.73 px | 2,433 px, 55.9% painted |
| genuine long crack | 6 | 2.9 - 12.9 px | |

`drop_straight_lines` removes components of >= 200 px, aspect >= 6 and wander < 1.0 px;
painted pixels are exempt. Verified guard-ON against guard-OFF across all 71 frames:

| | |
|---|---|
| fires on | 4 frames, 15,241 px |
| painted crack removed | **0 px** (criterion was 0) |
| clDice / Tsens / Tprec / IoU | **unchanged to six decimals** |
| `b3_amb`, a crack-free specimen | 0.0169% -> **0.0000%** |
| false-positive area on crack-free material | 0.02528% -> 0.02359%, a 6.7% reduction |

**It is a lower bound.** The test is component-level, so it catches the three standalone
artefacts. Two more are fused into 440k and 228k px crack systems and ride in attached to
real cracks; those need a within-component test that does not exist yet.

## Three checks in this work reported success while measuring nothing

Recorded because the pattern repeated, not for confession's sake.

1. The corpus straightness census printed **0.00%**. It crashed with `ptp was removed in
   NumPy 2.0` on exactly the 10 frames that contained long components -- the crash only
   fires once one is found -- and the per-frame error handler caught it while the summary
   filtered those frames out and averaged the survivors.
2. The first guard verification compared `effective_mask` against `effective_mask`. The
   guard runs inside it, so both arms were guarded: `0 px removed, clDice delta 0.000000,
   PASS`. A test that could not fail.
3. The empty-result fallback existed **twice**, in `_narrow_to_image` and inside
   `drop_straight_lines`. Removing the inner one changed nothing because the outer one
   caught the fall, and `b3_amb` -- the single confirmed false positive in the corpus --
   kept shipping through a guard that reported PASS. Fixed by ordering: the fallback covers
   tighten and clip, which rest on measurements that can fail; the straightness guard runs
   after it and is final, because an empty mask on a crack-free specimen is a correct answer.

A selftest probe now asserts all three properties, including the ordering.

## What is still not checked

- The two artefacts fused into large crack systems. A within-component straightness test
  would find them; none exists.
- NEGF/CFREE have n = 2 and n = 4. The crack-free specimens barely produce components large
  enough to sample, which is itself the good news, but those arms prove little on their own.
- The panel is a careful reader applying stated physics, not a metallurgist. It agrees with
  the annotator on 24/26 of their own strokes at full size, which is the strongest thing
  that can be said for it -- and only 18/32 at small size, which is the honest caveat.
