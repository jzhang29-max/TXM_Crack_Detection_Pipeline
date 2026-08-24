# Adding contrast-derived features ALONGSIDE the 17 — result: no measurable gain

**Verdict: negative, and cleanly so.** Every augmented arm sits inside the fold-to-fold
spread of the baseline on every accuracy axis, including the thin-crack and faint-crack
subsets the change was supposed to help. The added channels are *not* ignored — a
RandomForest gives 55% of its total importance to the 17 CLAHE columns, and `lcn_w151` is
the single highest-ranked feature of 20 in the contrast arm — so the finding is not "the
classifier never looked at them". It is the more informative version: **the classifier
looks at them, spends importance on them, and does not get more accurate. Local contrast
is a re-encoding of what the shipped 17 already measure, not new information.**

That is the distinction the brief asked for. Flat-fielding the model INPUT cost 0.169 IoU
because it *removed* the large-radius intensity signal. Adding a local-contrast signal in
extra columns costs nothing — and buys nothing either.

Reproduce: `.venv/bin/python research/contrast/augment_extract.py --workers 5` then
`.venv/bin/python research/contrast/augment_eval.py --workers 6`. Numbers below are from
`augment_results.json`. Both eval runs I did reproduced identical per-fold IoUs, so the
figures are deterministic, not a lucky seed.

## Setup

60 frames carry a `correction.npy` with both classes. Per frame, `RandomState(0)` draws
8000 `correction==1` and 8000 `correction==2` pixels — every frame had at least 8000 of
each, so the matrix is exactly 960,000 rows at 50.0% positive, grouped by image.
`GroupKFold(5)`, `StandardScaler` + `MLPClassifier((64,32), max_iter=300, random_state=0,
early_stopping=True, n_iter_no_change=8)`, IoU/precision/recall at 0.5 on held-out rows.

Every arm is a **column subset of one cached 37-column matrix**, so the arms differ only
in which columns the classifier sees — identical rows, identical seed, identical filter
code. The baseline was recomputed here rather than quoted, as asked.

| arm | dim | added columns |
|---|---|---|
| `baseline_17` | 17 | — (the shipped `compute_feature_stack`) |
| `17_plus_contrast3` | 20 | `lcn_w51`, `lcn_w151`, `dog_g8` |
| `17_plus_clahe_int` | 18 | `clahe_intensity` |
| `17_plus_clahe_int_grad` | 19 | `clahe_intensity`, `clahe_gradmag_s2` |
| `34_dup_clahe_stack` | 34 | the whole 17-feature stack recomputed on the CLAHE image |

Exactly what was added:
- `lcn_w{51,151}` = `(img - uniform_filter(img,w)) / sqrt(uniform_filter(img^2,w) - uniform_filter(img,w)^2)`, float64 throughout (in float32 that cancellation goes negative often enough to matter).
- `dog_g8` = `img - gaussian_filter(img, 8)`.
- CLAHE = `skimage.exposure.equalize_adapthist(kernel_size=256, clip_limit=0.01, nbins=256)`. The kernel is fixed in **pixels**, not left at skimage's `shape//8` default, so a 1693 px frame and a 6367 px frame get the same physical amount of local adaptation — and 256 px matches the reach of the shipped features' largest Gaussian (sigma 64).

The 34-dim arm **did** run; nothing in the brief was skipped for cost.

## Results

| arm | dim | IoU (all rows) | precision | recall | thin-frame IoU | faint-frame IoU | thin&faint IoU | crack-free FP (whole frame) | crack-free FP (on specimen) |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_17` | 17 | **0.6667** ± 0.0275 | 0.8104 | 0.7898 | **0.6239** ± 0.0292 | 0.5615 ± 0.0654 | **0.5362** ± 0.0563 | 0.2191 | 0.0223 |
| `17_plus_contrast3` | 20 | 0.6718 ± 0.0254 | 0.8214 | 0.7865 | 0.6214 ± 0.0273 | 0.5577 ± 0.0681 | 0.5234 ± 0.0674 | 0.1532 | 0.0204 |
| `17_plus_clahe_int` | 18 | 0.6650 ± 0.0404 | 0.8182 | 0.7794 | 0.6143 ± 0.0459 | 0.5520 ± 0.0610 | 0.5167 ± 0.0535 | 0.1053 | 0.0223 |
| `17_plus_clahe_int_grad` | 19 | 0.6737 ± 0.0241 | 0.8262 | 0.7845 | 0.6283 ± 0.0285 | 0.5614 ± 0.0971 | 0.5344 ± 0.0961 | 0.1730 | 0.0200 |
| `34_dup_clahe_stack` | 34 | 0.6753 ± 0.0372 | 0.8363 | 0.7778 | 0.6221 ± 0.0518 | 0.5574 ± 0.0511 | 0.5162 ± 0.0365 | 0.0932 | 0.0193 |

`±` is the standard deviation across the 5 folds. **The whole spread of arm means (0.6650
to 0.6753, a range of 0.0103) is smaller than the baseline's own fold-to-fold spread
(0.0275).** Paired per-fold deltas against the baseline, which is the tighter test:

| arm | Δ IoU all rows | Δ IoU thin&faint rows |
|---|---|---|
| `17_plus_contrast3` | +0.0051 ± 0.0107 | −0.0127 ± 0.0316 |
| `17_plus_clahe_int` | −0.0017 ± 0.0176 | −0.0194 ± 0.0177 |
| `17_plus_clahe_int_grad` | +0.0070 ± 0.0091 | −0.0018 ± 0.0476 |
| `34_dup_clahe_stack` | +0.0086 ± 0.0175 | −0.0200 ± 0.0450 |

No arm reaches significance on 4 degrees of freedom (largest |t| = 1.7). **All five arms
are indistinguishable on accuracy and I am not ranking them.** The one consistent
direction is a small trade of recall for precision as dimensionality grows (0.8104 →
0.8363 precision, 0.7898 → 0.7778 recall) — an operating-point shift, not a capability
gain, and it does not move IoU.

On the axis where contrast adjustment had the strongest prior — thin AND faint frames —
**all four augmented arms come out at or below the baseline.**

### Which added channels actually earned importance

They did, heavily. This is the most informative result here.

RandomForest impurity importance (200 trees, `min_samples_leaf=5`, 250k-row subsample):

| ranking run | share of total importance on ADDED channels | top added channel |
|---|---|---|
| `17_plus_contrast3` (20 cols) | 16.6% | `lcn_w151` at 0.1143 — **rank 1 of 20** |
| `34_dup_clahe_stack` (34 cols) | 54.7% | `clahe_texture_s8` at 0.0745 — rank 1 of 34 |
| all 37 cols together | 58.1% | `clahe_texture_s8` 0.0712, `lcn_w151` 0.0666 |

Permutation importance on fold 1 (IoU drop at 0.5, 3 repeats, 60k held-out rows) agrees:
`clahe_intensity` is the #1 or #2 feature in both CLAHE arms (IoU drop 0.362–0.371, versus
0.442 for `smooth_s64`); in the 34-dim arm the CLAHE columns take 42.3% of all positive
drop, led by `clahe_texture_s8` (0.284) and `clahe_smooth_s32` (0.272). Within the contrast
arm the ordering is `lcn_w151` (0.106) >> `lcn_w51` (0.042) >> `dog_g8` (0.011) — so the
**large** window is the one that carries anything, and `dog_g8` is close to dead weight.

**Harness check against the project's prior finding.** RF importance on `baseline_17` alone
puts `intensity` + `smooth_s{2..64}` at **42.2%** of total, reproducing the ~41% figure in
`docs/MARKUP_GUIDE.md` from an independent implementation. So this harness is measuring the
same thing the flat-fielding experiment measured.

The two results together are the whole story: the added columns rank at the very top of the
feature list, and accuracy does not move. A feature can only do that if it is a re-encoding
of information already present. `lcn_w151` is literally `(intensity − local mean) /
local std`, and `intensity`, `smooth_s32`, `smooth_s64`, `texture_s2` and `texture_s8` are
all already columns — the MLP can and evidently does form that quotient itself.

### The crack-free guardrail — read the second column, not the first

The whole-frame false-positive fraction looks like a large win (0.219 → 0.093, a 2.4x
reduction). **It is not a win on the specimen.** Split by `pipeline.specimen_support`:

| arm | FP whole frame | FP on specimen | FP off specimen |
|---|---|---|---|
| `baseline_17` | 0.2191 | 0.0223 | 0.6732 |
| `17_plus_contrast3` | 0.1532 | 0.0204 | 0.4701 |
| `17_plus_clahe_int` | 0.1053 | 0.0223 | 0.3052 |
| `17_plus_clahe_int_grad` | 0.1730 | 0.0200 | 0.5285 |
| `34_dup_clahe_stack` | 0.0932 | 0.0194 | 0.2685 |

Off-specimen background is 22–38% of these frames and sits near zero intensity, so a
darkness-driven model calls most of it crack; local contrast normalisation flattens it and
makes it uninformative. That is real, but it is background rejection, which the app already
does structurally with `specimen_support` and `prune_specks`. **On the metal, where a false
positive is an actual false indication, the arms span 0.0193–0.0223 — flat.** One arm is
even slightly *worse* on the specimen on one frame while its whole-frame number looks 2x
better: `17_plus_contrast3` on `wrought_316L_fatigue_0_cycles` reads 0.0606 on-specimen vs
the baseline's 0.0589, while its whole-frame number reads 0.170 vs 0.357. Ranking arms on
the whole-frame column would have credited contrast for fixing a problem it did not fix.

## Thin-crack frames — 34 of 60

By the brief's definition (inside `correction==1`, keep pixels darker than the 20th
percentile of `img01` within the strokes, `remove_small_objects(64)`, median of
`distance_transform_edt` on the `skeletonize`; THIN if median half-width ≤ 3.0 px). Frame
half-widths ranged 1.0–10.4 px; 34 frames qualified. Sorted by half-width, with the
faintness measure in the next section:

| half-width (px) | frame | also faint |
|---|---|---|
| 1.00 | `b2_343_75_LARGE` | yes |
| 1.00 | `b3_362_50um` | yes |
| 1.00 | `b3_380_00um_4` | |
| 1.00 | `b3_380_00um` | yes |
| 1.00 | `b3_383_75um_ZOOM` | |
| 1.00 | `b3_385_63um_ZOOM` | |
| 1.00 | `b3_388_13um_LARGE_2` | yes |
| 1.00 | `b3_388_13um_ZOOM` | |
| 1.00 | `wrought_316L_fatigue_1200_cycles_crack` | |
| 1.00 | `wrought_316L_fatigue_1250_cycles_crack` | |
| 1.00 | `wrought_316L_fatigue_1260_cycles_crack` | |
| 1.00 | `wrought_316L_fatigue_1270_cycles_crack` | |
| 1.00 | `wrought_316L_fatigue_1280_cycles_crack` | yes |
| 1.00 | `wrought_316L_fatigue_1290_cycles_crack` | yes |
| 1.00 | `wrought_316L_fatigue_1300_cycles_crack` | yes |
| 1.41 | `b2_335_31um_FRFR` | |
| 1.41 | `HC_316L_fatigue_1760_tip_zoom` | yes |
| 1.41 | `HC_316L_fatigue_1770_tip_zoom` | yes |
| 1.41 | `HC_316L_fatigue_700_cycles` | yes |
| 1.41 | `HC_316L_fatigue_800_cycles` | |
| 1.41 | `wrought_316L_fatigue_1100_cycles_crack` | |
| 1.41 | `wrought_316L_fatigue_800_cycles` | |
| 2.00 | `HC_316L_fatigue_1250_cycles` | |
| 2.00 | `HC_316L_fatigue_1400_cycles_tip` | |
| 2.00 | `HC_316L_fatigue_1790_tip_zoom_2` | |
| 2.24 | `HC_316L_fatigue_1750_tip_zoom` | yes |
| 2.24 | `HC_316L_fatigue_1780_tip_zoom_2` | |
| 2.24 | `HC_316L_fatigue_1790_cycles` | yes |
| 2.24 | `HC_316L_fatigue_600_cycles` | yes |
| 2.24 | `wrought_316L_fatigue_900_cycles` | |
| 2.83 | `B2_333_75_um_zoom` | |
| 2.83 | `HC_316L_fatigue_1100_cycles` | yes |
| 2.83 | `HC_316L_fatigue_1400_cycles` | |
| 3.00 | `HC_316L_fatigue_1200_cycles` | |

(Full filenames and per-frame `half_width` / `contrast` values are in
`augment_results.json` under `frames`.)

Worth knowing about this criterion: it measures the width of the **darkest 20% core** of a
stroke, not the stroke itself, so it reads small on any frame whose darkest pixels form a
filament. Fifteen frames read exactly 1.0 px. It is a usable thin/thick split — 34/60, not
degenerate — but it is not a crack-width measurement, and I would not quote these numbers
as physical widths.

### FAINT — an addition, not a substitute

The brief defines THIN but the question is about thin **and faint** cracks, and thinness
carries no amplitude information (a 1 px crack can be pitch black). So I added a faintness
axis at zero extra cost, from columns already in the cached matrix: per frame,
`median(smooth_s64 − intensity)` over its crack rows — how much darker a crack pixel is
than its own broad neighbourhood. FAINT = lowest tertile, cut at 0.00885 (20 frames);
thin AND faint = 14 frames. These are the rows in the "faint" and "thin&faint" columns
above, and they are the rows on which every augmented arm did **worse** than the baseline.

## Cost

| stage | wall clock | CPU-seconds |
|---|---|---|
| feature extraction, 66 frames (60 labelled + 6 crack-free), 37 columns | 259 s (5 workers) | 1213 |
| evaluation: 25 fold fits + 5 guardrail fits + 4 importance runs | 514 s (6 workers) | 2961 (folds 2015, guardrail 590, RF 356) |

Per-arm fit cost was 348–468 s of CPU across its 5 folds; the 34-dim arm was the most
expensive at 468 s, i.e. 32% more than the baseline's 355 s. Extraction is dominated by the
largest frames (65 s for a 23 Mpx mosaic; CLAHE plus a second 17-filter stack roughly
triples the per-frame feature cost over the baseline 17).

## Conclusion

Adding contrast-derived features alongside the existing 17 changes nothing measurable about
this detector's accuracy. Across 960,000 grouped-by-image rows and four different ways of
adding contrast — from one extra column to a full duplicate 17-feature stack on the CLAHE
image — the mean IoU moves by at most +0.0086, against a baseline fold-to-fold spread of
±0.0275, and on the thin-and-faint frames where the effect was expected to be largest every
arm is at or below the baseline. The result is not that contrast is invisible to the model:
`lcn_w151` ranks first of 20 features and the CLAHE stack absorbs 55% of RandomForest
importance, which means the model reaches for these channels readily. It is that they
carry no information the shipped features lack — `intensity`, six smoothed intensities and
two local standard deviations are already enough for the network to construct a local
contrast internally, so handing it one pre-computed changes which columns get credit and
not what the model can do. The one place the numbers move a lot, the crack-free
false-positive fraction (0.219 → 0.093), is off-specimen background rejection: on the
specimen itself the arms are flat at 0.019–0.022, and the app already suppresses background
structurally. The honest recommendation is not to add these channels — they cost 32% more
feature computation per frame plus a CLAHE pass, and buy nothing. The 41%-importance
finding that motivated the caution about flat-fielding survives intact and is reproduced
here at 42.2%; what this arm adds is that the complementary move, *adding* contrast rather
than *replacing* intensity with it, is not harmful either — it is simply inert.

## What I did not run, and why

- **No CLAHE parameter sweep.** One setting only (`kernel_size=256, clip_limit=0.01`).
  A different kernel or clip limit could change the CLAHE arms; I have no evidence either
  way. 256 px was chosen to match the shipped features' reach, not tuned.
- **One MLP seed** (`random_state=0`, as the protocol fixes). The fold-to-fold spread is
  reported, but not seed-to-seed variance, so "indistinguishable" is a statement about
  fold variation only. Both of my eval runs reproduced identical per-fold IoUs, which
  confirms determinism but is not a variance estimate.
- **No pixel-level ground truth anywhere.** Accuracy is measured against the owner's own
  corrections, which is what the protocol specifies. It inherits whatever the corrections
  inherit; the crack-free specimens are the only ground truth in play, and they are
  negative-only by construction.
- **Full-frame prediction and overlay inspection were not done.** These numbers are on
  sampled labelled rows. A change can hold IoU on the labelled 8000+8000 rows and still
  alter overlays elsewhere in the frame — this project has been burned by exactly that
  before. Given that no arm improved, I did not think it worth the compute, but if any arm
  were ever considered for deployment that check would have to happen first.
- **Nothing was written outside `research/contrast/`.** The 37-column cache lives in the
  session scratchpad (`--cache`), `app/`, `code/`, `models/` and `app_data/` were read
  only, and no git command was run.
