# START HERE — manual correction

Everything is running. Three commands, nothing to set up.

## 1. Mark up

The paint tool is **already running**: open **http://127.0.0.1:8766**

All 71 predictions are pre-computed, so every image opens instantly (no 30s wait).

If it is ever not running:
```bash
python3 code/paint_server.py
```

**Controls**
| control | does |
|---|---|
| **Add crack** brush | mark crack the model missed — the high-value action |
| **Eraser** brush | mark a false positive as background |
| **Click-to-remove** | clears one whole connected false-positive region in a single click — fastest way to kill a wedge rim or speckle |
| **Save corrections** | writes to `paint/corrections/`, regenerates that image's outputs |

Work is saved per image. Stop and resume whenever.

## 2. Working through all 71

`WORKLIST.md` is a tickable checklist of every image, ordered by value rather
than alphabetically — Wrought first, then AM (neither has any crack example
yet), then B3, then B2. Within each group the worst-predicted frames come
first, since that is where a correction teaches the model the most.

```bash
python3 code/make_worklist.py     # refresh the ticks after a session
```

Current: **12 / 71 marked** — Wrought 0/14, AM 0/27, B3 0/13, B2 12/17.

You can retrain at any point, not just at the end. Marking one group and
retraining tells you what that group bought before you invest in the next.

## 3. What to mark, in priority order

**AM and Wrought have ZERO crack examples.** All 12 hand-marked images are B2.
That is the entire reason those groups mark wedge rims instead of the thin
centre cracks — the model has never been shown a crack in that material and
falls back on B2 morphology.

1. **2–3 Wrought frames** — `1250`–`1300_cycles_crack` are cleanest (1.8–2.7% predicted).
   Mark the thin centre crack. **Ignore the dark wedge — you have confirmed it is not a crack.**
2. **2–3 AM frames** — include one `_tip_zoom`, which is where output is worst.
3. Anything else that looks wrong.

You do NOT need to do all 71. The model generalises from a few examples per
material; the gap is variety, not volume.

## 4. Retrain

```bash
python3 code/markup_status.py --todo      # what is marked so far
python3 code/retrain_after_markup.py --deploy
```

It sweeps class balance, scores each candidate against ground truth on BOTH
axes — IoU/recall (does it find real cracks) and false crack on the six
confirmed crack-free specimens — and deploys only if a candidate beats the
current model on both. If nothing passes it says so and changes nothing.

Then regenerate outputs:
```bash
python3 code/build_outputs_per_group.py
```

## Current state


| | |
|---|---|
| deployed model | `models/pixel_hgb_final.joblib` (raw_v4) — IoU **0.773**, recall 0.881 |
| tuning | exhausted; a 4-point balance sweep confirmed the current setting is best |
| outputs | `results/final_71_pergroup/` — all 71 masks + overlays + stats |
| trustworthy groups | **B2, B3** (verified against ground truth) |
| unreliable groups | **AM, Wrought** — still trace wedge rims; this is the labelling gap above |
| rollback | `models/pixel_ORIG_raw_backup.joblib` |

Full history, including four things that were tried and failed, is in `HANDOFF.md`.
