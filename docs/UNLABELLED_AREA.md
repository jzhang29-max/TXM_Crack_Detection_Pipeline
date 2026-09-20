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

## What is still not checked

- The 12.2% of unlabelled area in components under 4000 px. 12,588 components, 285 of them
  over the floor. Small components are where speck-level false positives would live.
- NEGF has n = 2. The crack-free specimens barely produce components large enough to sample,
  which is itself good news, but it means that arm proves nothing.
- The panel is a careful reader applying stated physics, not a metallurgist. It agrees with
  the annotator on 24/26 of their own strokes, which is the strongest thing that can be said
  for it.
