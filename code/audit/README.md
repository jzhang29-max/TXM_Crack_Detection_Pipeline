# Audit generators

Every quantitative claim in `docs/WIDTH_REFERENCE.md` and `docs/UNLABELLED_AREA.md` should
be re-runnable from this directory. Until 2026-09-21 none of it was: the docs said "every
number below is reproducible from `app/core/pipeline.py: local_background`,
`_transverse_fwhm` and `clip_to_measured_width`", but those are primitives, not drivers, and
`grep -rl` for the four artifact names over both repos returned exactly one hit — a doc.
The sweeps, the blind designs, the leak guards and the censuses lived in a scratch directory
and were lost to the reader. That is the gap these files close.

| script | emits | backs |
|---|---|---|
| `measure_width_ratio.py` | `out/txm_width_ratio_unified.json` | the width-ratio table — every mask variant, one estimator, one frame set, same sample points |
| `measure_unlabelled_area.py` | `out/txm_unlabelled_area_per_frame.json` | the 24.7% / per-group unlabelled-area table |
| `measure_transverse_fwhm.py` | `out/txm_transverse_fwhm.json` | the "width is physical, not point spread" pre-registered test |
| `build_small_component_crops.py` | crops + `out/txm_small_component_cropkey_v2.json` | the blind set for the small isolated indications, matched on size **and** specimen |
| `score_blind_panel.py` | scored JSON | unblinding, instrument validation, and the group-confound check for any blind run |

Run them from the repo root. Each takes an optional output path and is restartable — they
append JSONL and skip frames already present, because a full sweep is ~20 minutes and was
repeatedly killed by memory pressure when it held everything in RAM.

## What is NOT reproducible from here, stated plainly

**The panel votes.** `score_blind_panel.py` scores them deterministically, but producing
them requires running a blind multi-agent panel over the crop images. The crops, the crop
key, the leak-guard statistics and the scoring are all here and re-runnable; the votes are
an artifact of a specific model and harness and are shipped as data, not regenerated.

This matters for the paper and should not be softened. The load-bearing evidence for the
unlabelled-area result is a vote by language-model agents with no human expert arm, and the
instrument's validation shows the agents agree with the annotator's own strokes — not that
they can identify crack in 316L TXM independently. `docs/UNLABELLED_AREA.md` records this;
a materials venue will require at least two blinded human readers through the same protocol
before the claim can stand. The protocol, the crops and the seed are all preserved here so
that run is a matter of finding readers, not of rebuilding anything.

## The confound that made this directory necessary

The first small-component run stratified its positive control on size only. Specimen mix
came out 62% AM in the control against 41% Wrought in the test class, and AM is the group
this project documents as intensity-invisible. Standardising the control to the test mix
moved its yes-rate from 0.562 to 0.313 — against a test rate of 0.344 — which erased the
entire reported gap and voided the prevalence estimate built on it.

`build_small_component_crops.py` now matches within group and then on size, and refuses to
substitute across groups: it records a shortfall instead. `score_blind_panel.py` prints the
group mix and the standardised control rate on every run, whether or not anyone asked.
