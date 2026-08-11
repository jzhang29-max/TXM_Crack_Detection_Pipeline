# HANDOFF — state of play

Single source of truth for picking this up in a new session. Everything below
is committed to git; nothing important lives only in a chat transcript.

---

## 1. Where things actually stand

**Dataset:** 71 TXM images across 4 specimen groups (AM 316LH Fatigue 27,
B2 316L H Tension 17, B3 316L Amb Tension 13, Wrought 316L H Fatigue 14).
Raw images live outside the repo under `~/Desktop/TXM DATA/<group>/`, with
flatfielded counterparts at `~/Desktop/TXM DATA processed/flatfielded/<group>/`.

**Pipeline runs on FLATFIELDED input.** This was measured, not assumed:
raw median brightness varies 2.6x across specimen groups (Wrought 0.575 vs
B2 1.518), and the raw-trained model's dominant rule was "broad dark region
= crack", so it flooded darker specimens. Moving to flatfielded input cut
over-prediction by 40.0pp on Wrought and 24.0pp on AM with no regression on
B2. Flatfielding also removes the mosaic tile-grid pattern and the broad
illumination gradient.

**Current best model:** `models/pixel_flatfield_clean.joblib` (MLP + StandardScaler).

**Current best outputs:** `results/final_71_v2/` — all 71 B&W masks, stats
CSVs, summaries and montages are COMMITTED (4MB). Overlays are gitignored
(558MB) but are a pure rendering of mask+image, regenerated in seconds (§5).

Per-group median predicted crack area, `final_71` -> `final_71_v2`:

| group | n | final_71 | final_71_v2 |
|---|---|---|---|
| AM 316LH Fatigue | 27 | 18.9% | **6.7%** |
| B2 316L H Tension | 17 | 31.4% | **24.9%** |
| B3 316L Amb Tension | 13 | 1.0% | **0.4%** |
| Wrought 316L H Fatigue | 14 | 1.9% | **1.0%** |

`final_71_v2` has NOT been re-audited — the 87% figure in §2 was measured on
`final_71`. Whether these suppressions moved it is unmeasured (see §8 step 2).

---

## 2. THE HONEST ASSESSMENT — read this before trusting any output

A 46-agent audit (one per image, each shown a local-contrast-enhanced view so
faint cracks were visible, and briefed with the owner's domain corrections)
found on the previous `final_71` outputs:

- **median 87% of predicted crack area is FALSE POSITIVE** (mean 85%, range
  40-100%)
- verdicts: 19 mostly_false_positive, 20 severely_over_predicts,
  7 over_predicts, **0 mostly_correct**
- dominant false-positive cause per image: dark wedge/notch 17, specimen edge
  rim 17, surface texture 6, reference artifacts 2, off-specimen 2,
  tile grid 1, pores 1
- of 37 images where a real crack IS visible: 10 fully marked, 21 partially,
  6 not at all

So the model usually *finds* the crack but buries it in ~6x more false
positive. **This is a precision problem, not primarily a recall problem.**
Raw audit data: `results/qc50_audit_cleanmodel.json`.

`final_71_v2` applies two audit-specified suppressions and is better, but has
NOT been re-audited. Do not quote any crack area fraction as a measurement
without spot-checking the overlay.

---

## 3. What the owner told us about crack morphology — TRUST THIS OVER INTUITION

These are domain corrections from the specimen owner. Two automated labelling
attempts were reverted because they violated them:

1. **The large dark WEDGE is NOT a crack.** Models repeatedly trace its rim.
   An attempt to label the wedge as a "thick crack" was built on this
   misreading and reverted.
2. **The real cracks are THIN, VERY FAINT, and in the CENTRE of the frame.**
   They are often visible only under local-contrast enhancement.
3. **Elongated INCLUSIONS are not cracks.** An attempt to label them
   (12 of 13 B3 images) was reverted — 4,901,522 px of bogus force-crack
   labels zeroed.
4. Several ambient / low-load specimens have **zero cracks**. Six are
   confirmed crack-free and labelled as such (see §4).

Practical consequence: **do not generate positive crack labels
algorithmically.** Two attempts produced confidently-wrong labels. Positive
labelling needs the owner's strokes in the paint tool.

---

## 4. Label inventory — what is labelled, by whom, and how much

`paint/corrections/<name>_correction.npy`, uint8: 0=untouched, 1=force-crack,
2=force-not-crack. All 71 images have a file. **All committed to git.**

| source | kind | volume | trust |
|---|---|---|---|
| owner's hand-drawn strokes, 12 B2 images | crack + not-crack | — | HIGH, restored verbatim from git commit `df83a35` after the revert |
| 4 Ilastik-derived ground-truth masks (B2) | crack | `dataset_cache_flatfield/` | HIGH — the only pixel-level truth that exists |
| off-specimen geometric exclusion, all 71 | not-crack | 80.3M px | HIGH — imaging geometry, not a morphology judgment |
| 6 owner-confirmed crack-free specimens | not-crack | 91.6M px | HIGH — whole specimen interior |
| false-positive cleanup (wedge margin, edge ring, round speckle) | not-crack | 43.3M px | MEDIUM — audit later showed wedge+rim still dominant, so it was too weak |
| ~~automated positive crack labels~~ | ~~crack~~ | ~~4.9M px~~ | **REVERTED — was wrong** |

The 6 confirmed crack-free specimens (`code/mark_zero_crack_images.py`):
b3_amb, B2_amb_mosaic_2, B2_2_1_lbf, B2_2_9_lbf, b3_3_18lbf,
wrought_316L_fatigue_0_cycles.

Deliberately left UNLABELLED (untouched contributes no training signal; a
wrong label actively teaches the wrong thing):
- `B2_3_1_lbf`, `B2_3_2_lbf` — one short dark line / elongated specks;
  inclusion vs early initiation is unclear
- `b3_3_0lbf_268_13um` — degenerate frame, black bands are stitching artifacts
- all faint centre hairlines — the thing that most needs labelling

---

## 4b. NOTHING FROM AN AGENT RUN IS EVER LOST — how to recover it

Two independent mechanisms. A killed run does NOT need re-running.

**1. Harvest the journal** (works mid-run, on a dead run, repeatedly):

```bash
python3 code/harvest_workflow_results.py --list          # every run, started vs completed
python3 code/harvest_workflow_results.py                 # newest run -> results/harvested/
python3 code/harvest_workflow_results.py --run wf_xxxxx  # a specific run
```

Every agent that finishes has its return value appended to that run's
`journal.jsonl` immediately, so completed work is on disk the moment it lands.
One run here lost 51 of 92 agents to session limits and all 41 completed
results were recovered this way. Harvested output is committed under
`results/harvested/`.

Gotcha if reading the journal by hand: the payload key is **`result`**, not
`value`. Using the wrong key silently yields zero findings.

**2. Resume the workflow** — replays completed agents from cache instantly and
only re-runs the failed/new ones:

```
Workflow({scriptPath: "<path printed when the workflow launched>",
          resumeFromRunId: "wf_xxxxx"})
```

Run IDs and script paths for the audit runs so far:

| run | purpose | started/completed |
|---|---|---|
| `wf_b87f351e-d77` | re-audit of `final_71_v2` (the open question in §2) | in flight |
| `wf_d6f09f17-513` | audit of `final_71` -> the 87% figure | 61/47 |
| `wf_d18a107b-66e` | audit of v1 flatfielded predictions | 94/41 |
| `wf_a8077edb-b3a` | first 71-image review (raw predictions) | 64/24 |

Scripts live in
`~/.claude/projects/-Users-jiamingzhang-Desktop-APP/ca4727e5-.../workflows/scripts/`.

---

## 5. How to reproduce or continue

```bash
# regenerate the best outputs (masks, overlays, stats, montages) for all 71
python3 code/build_final_outputs_v2.py --model models/pixel_flatfield_clean.joblib

# open the paint tool to add corrections (flatfielded input + flatfielded model)
python3 code/paint_server.py     # then http://127.0.0.1:8766
#   TXM_PAINT_RAW=1 reverts it to raw input + the old raw model

# retrain after adding corrections. BOTH caps must be tuned together (see §6)
python3 code/train_flatfield_model.py --crack-cap 30000 --neg-cap 6000 \
    --out models/my_candidate.joblib

# architecture comparison on the current label set
python3 code/compare_architectures_flatfield.py

# rebuild the flatfielded feature cache from scratch if needed (~2GB, gitignored)
python3 code/build_flatfield_dataset.py
```

The paint tool auto-detects a swapped model file and invalidates its cache —
no restart needed. Verified live twice.

---

## 6. Traps that already cost time — do not rediscover these

**Class balance must be tuned as a PAIR.** `--crack-cap` and `--neg-cap`
interact. Measured: v1 50% crack (baseline), v2 27.5% (negatives-only labels
added → `class_weight="balanced"` upweighted crack ~2.6x → two B2 images
REGRESSED, one from 26.9% to 40.1%), v3 57.5% (good), v4 72.6% (positives
added → over-swung the other way), v5/final ~50% (good). Always check the
crack fraction printed at training time.

**Do not copy a flatfielded model to `models/pixel_hgb_final.joblib`.** That
path is the RAW pipeline's production slot and `apply_pixel_model.py` feeds it
raw images. Mixing input domains is worse than either alone.

**A retracted claim: there is NO flatfielding problem.** `qc_flatfield_quality.py`
reported 21 of 71 images failing, including 3 of 4 ground-truth images. That
was WRONG — it measured IQR *after* percentile normalization, which divides by
the p1-p99 span, so it was really measuring how deep an image's darkest
features are relative to its texture. Checked against `flatfield.py`'s own
spec, every image centres on 1.000 with raw IQR 0.008-0.023, and the images
flagged as "failed" are the CLOSEST to spec. The script is kept only as a
cautionary example; **its verdicts are not valid.**

**Local-background subtraction is a band-pass filter.** Inside a wide dark
feature the local background is also dark, so contrast vanishes and only the
PERIMETER survives. This is why an early crack detector outlined the wedge
instead of filling it, and why training on such a label would teach
"crack = ring around a dark area".

---

## 7. Architecture comparison (`results/arch_compare_flatfield.json`)

Six architectures on the flatfielded 71-image label set. Scored on three axes,
because IoU alone is misleading here — it only exists for the 4 B2
ground-truth images, and models score well there while flooding everything else.

| architecture | IoU (GT) | area on crack-free ↓ | artifact regions ↓ | fit |
|---|---|---|---|---|
| RandomForest | 0.702 | 1.24% | 605 | 310s |
| ExtraTrees | 0.672 | 1.36% | 414 | 57s |
| MLP (deployed) | 0.565 | 2.18% | 272 | 82s |
| HistGradientBoosting | 0.554 | 1.40% | **234** | **12s** |
| DecisionTree | 0.530 | 1.77% | 682 | 31s |
| LogisticRegression | 0.417 | 2.42% | 14 | 2s |

RandomForest is best at not hallucinating crack in undamaged material but has
the worst artifact behaviour. HistGradientBoosting is the best balance and 26x
faster than RF. LogisticRegression's 14 artifact regions come from barely
detecting anything. **The currently deployed MLP is beaten by HGB on both
non-IoU axes** — switching is worth testing but was not done.

---

## 8. Highest-value next steps, in order

1. **Owner marks the thin centre cracks** on ~3-4 AM and ~3-4 Wrought frames
   in the paint tool. This is the only way to close the real gap: every crack
   training pixel currently comes from the 12 B2 images, so the model has no
   example of AM or Wrought crack morphology. No amount of false-positive
   removal fixes this.
2. **Re-audit `final_71_v2`** the same way `final_71` was audited, to measure
   whether the geometric + tile-phase suppressions actually moved the 87%
   false-positive figure. Sheet builder and workflow pattern are in git history.
3. **Test HistGradientBoosting** as a replacement for the deployed MLP (§7).
4. **Tile-phase rejection is under-exploited.** Two agents proved the
   reference artifacts are mosaic-tile-locked by autocorrelating the
   prediction mask — periods ~112px and ~84px, 90.5% of interior red in the
   top 25% of intra-tile phase cells vs 25% expected by chance. That is a much
   stronger test than the round-shape heuristic, and
   `build_final_outputs_v2.py` implements only a first cut of it.
