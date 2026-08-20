# TXM Crack Detection — Quickstart

Detects cracks in transmission X-ray microscopy images. Drag images in, look at
what the model found, fix what it got wrong, press Retrain. That is the whole loop.

## What it looks like

One window, three regions. The **sidebar** lists your images with the result burned into
each thumbnail, `% crack`, and a marker for anything still on an older model. The
**toolbar** is the three correction tools (Add crack, Erase, Flip region), the two view
toggles, Advanced, Export and Retrain. The **model picker** is bottom left, under the
image list. Every stroke saves itself — there is no save button.

The figures below are generated from the app's own data by
`python3 code/make_readme_figures.py`, so they cannot drift from what the code does. A
screenshot of the window is the one thing that tool cannot produce; there was one here,
and it was removed because it had gone stale in a way that misrepresented the tool — it
was rendered by a model since measured at 22% false positives on crack-free specimen, and
it predated the display contrast change described below. Replacing it needs someone at a
browser, which is the honest reason it is absent rather than wrong.

### Every upload is destitched and flat-fielded automatically

![Preprocessing](docs/img/preprocessing.png)

Drop an image in and two corrections run before you ever see it, with nothing to
configure:

1. **FFT destitch** — the mosaic tile grid is periodic, so it is one to two frequency
   bins in the row/column profile. `code/destitch.py` notches exactly those bins, with a
   protection pass that blends the correction back wherever real structure would be
   damaged.
2. **Flat-field** — divide by an anisotropic Gaussian blur (σ_y=16, σ_x=22, both
   measured for this dataset's tile pitch) to remove the macro brightness gradient.

Left is what came off the instrument: a 5×3 tile grid, a bright blob, and a crack you can
barely see. The middle panel is what you mark on, and the right one is the same crack at
native resolution. Both steps preserve geometry, so a mask still registers pixel-for-pixel
on the original.

**Contrast is stretched for display.** Flat-fielding leaves the specimen inside a narrow
bright band — measured across these 71 images, a standard deviation of 7 to 14 grey levels
out of 255 — so a crack a few counts deep is nearly invisible if the array is shown as-is.
The app stretches each image's own 1st–99th percentile, measured over the specimen and
cached, which takes that spread to 41–47. That is a *viewing* change only. It also makes
the destitcher's residual visible: the notch removes 91–99% of the tile-pitch amplitude,
and the faint grid still discernible in the middle panel above is the 1–9% that is left.

**The model is still fed the raw image**, deliberately: flat-fielding as *model input* was
tried and cost 0.169 IoU, because large-radius intensity features are ~41% of the model's
importance and flat-fielding removes exactly those. So what the human sees and what the
model sees are different on purpose. If preprocessing ever fails, the app falls back to
raw and says so in red rather than letting you mark an unprocessed frame believing
otherwise.

### What it finds, and what it misses

![Detection example](docs/img/example_detection.png)

A native-resolution window — 1013×760 of a 4376×2363 frame — because a hairline crack does
not survive being downsampled to README width. Left, the image as you see it in the app;
right, the shipped baseline's output. This is an AM/HC specimen — the group where the model
is weakest: masked IoU **0.32** against 0.82 on B2, precision 0.355, on four frames the owner
labelled 90–96% densely. Nothing about *this* frame was used to fit or validate the model.

Four things are visible here, and the last two are why the correction tools exist:

1. It follows the main crack closely, including the fine strands that fray off its left end.
2. It stays off the mottled background texture, which is the failure mode you would expect
   from a pixel classifier on a noisy frame.
3. **It misses the thin crack across the top.** That one is shallower and it is not marked
   at all.
4. **The isolated red sliver on the left is a false positive** — an elongated pore, not a
   crack.

### Correcting it

![Correction example](docs/img/example_correction.png)

One of the four ground-truth images, so this correction can be checked rather than
asserted. The model has run 17,983 px past the end of a real crack — its documented main
failure mode is running wide of a crack, not inventing one elsewhere — and ground truth
says **0.00%** of the cyan region is crack. **Flip region** takes that whole connected blob
in one click; the real crack in the same window keeps its red.

**Erase** reaches the same result by brushing, which on a blob this size is about a minute
of work. That is the only difference between the two: Erase is a brush, Flip region is one
click on a connected component. Both write the same not-crack label, and Retrain learns
from it.

### How well does it do, honestly

![Model against ground truth](docs/img/ground_truth.png)

The one comparison that does not rely on your eye: the model's output beside the
hand-labelled truth for the same window. Agreement is high where the crack is wide open,
which is what all four ground-truth images are.

On the four ground-truth images (all one specimen group, B2) under leave-one-image-out:
mean IoU **0.821**, recall **0.914**.

**The false-alarm figure, with its definition, because it appeared five different ways in
earlier drafts.** *Share of total pixels predicted crack, after speck pruning — which is the
mask you actually see — measured per specimen on the six the owner confirmed crack-free,
then averaged over specimens (n=6):*

| model | mean | range across the six |
|---|---|---|
| shipped baseline | **0.076%** | 0.000 – 0.185% |
| currently deployed retrain | **0.106%** | 0.000 – 0.262% |

Quote the mean *and* the range; one specimen is always near zero and one carries most of it.
Numbers elsewhere in this repo around 0.19–0.26% are the same quantity measured **before**
pruning, which is why they are larger — if you compare two figures, check they are on the
same side of the filter. For external calibration, MIL-HDBK-1823A treats ≤1% probability of
false calls as the NDT yardstick; both models are well inside it.

> **Check which model you are on.** Those numbers are the shipped baseline's. A model
> retrained in the app can be far worse at background and still deploy, because until
> recently the gate only compared IoU on the four B2 ground-truth images. A retrain on a
> single image's corrections measured **22.4%** of crack-free specimen area marked as
> crack -- 107x the baseline -- and passed. The gate now also refuses any candidate whose
> false-positive rate on the crack-free specimens rises by more than 0.5 points, but
> models deployed before that fix are still in your history. The model picker's
> `shipped baseline` entry is the measured-good one. Zero-shot SAM, prompted the way SAM is
designed to be prompted, scores 0.23-0.36 on the same images.

**Performance varies enormously by specimen group, and that is the most important thing on
this page.** Masked IoU is **0.82 on B2** and **0.32 on AM/HC** — measured on four AM/HC
frames the owner labelled 90–96% densely, held out by image (`code/eval_dense_labels.py`).
The AM/HC failure is precision, 0.355 mean and as low as 0.082: it finds the crack and marks
three to twelve times too much material with it. A second, independent protocol agrees the
group does not transfer — leave-one-specimen-group-out gives crack recall 0.836 (B2), 0.795
(B3), 0.763 (wrought), 0.397 (AM/HC).

So treat 0.821 as a B2 number, not a general one. The remaining unknown is whether AM/HC's
0.32 is a model failure or a labelling disagreement: precision is scored against one person's
labels, and a second annotator on a subset is what would settle it. `docs/SAM_COMPARISON.md` has the full study
and `docs/HANDOFF.md` records four approaches that were tried and reverted.
`docs/SAM_COMBINATION_SWEEP.md` tests 78 feature/classifier/ensembling combinations
under leave-one-image-out and finds none that beats the deployed one — it is worth
reading before trying to improve the model, because it also says which six avenues
are measured dead ends.

## Run it

One command. Nothing to install first, nothing to configure:

```bash
git clone https://github.com/jzhang29-max/TXM_Crack_Detection_Pipeline.git && cd TXM_Crack_Detection_Pipeline && ./run_app.sh
```

Then open **http://127.0.0.1:8800**.

That script creates its own virtualenv, installs dependencies, expands the bundled
reference data, and serves the app. Re-running it later just starts the app -- it
notices the venv already exists and that requirements have not changed.

**Python 3.10 is the floor, 3.12 is what this is tested on.** Not 3.9: `scikit-learn>=1.7`
requires 3.10, and the versions pip resolves today (numpy 2.5, scipy 1.18, tifffile) require
3.12. Debian 11 and Ubuntu 20.04 ship 3.9, and stock macOS `python3` may too — on those,
install a newer Python first or `./run_app.sh` fails in pip's resolver after the clone.
Check with `python3 -V` before you start.

Apple Silicon, CUDA and CPU-only all work; a GPU makes the SAM step ~10x faster but nothing
requires one. `PORT=9000 ./run_app.sh` if 8800 is taken.

Two things happen once, not on every start:

- **SAM ViT-H (~2.4 GB)** downloads from HuggingFace on the first prediction and
  caches in `~/.cache/huggingface`.
- **Reference feature stacks (2.1 GB)** are computed on the first *Retrain*, not at
  startup -- nothing else reads them, and building them eagerly used to add several
  silent minutes before the app would serve.

To check your install rather than trust it:

```bash
python3 app/selftest.py
```

## Extending the ground truth

Dense ground truth exists for **four images, all specimen group B2**. That is the project's
binding limitation: held out, the model recovers **39.7%** of the crack the owner marked on
AM/HC against 75.5–81.6% for the same number of random images — the specimen group, not the
training-set size. Closing it needs dense annotation, and the path is wired:

```bash
python3 code/export_annotation_tiles.py --group AM/HC --tiles 30   # uniform-random tiles
python3 code/annotation_tiles.py load                             # into the app, NO prediction
#   ... paint every pixel: Add crack for crack, Erase for not-crack ...
python3 code/annotation_tiles.py status                           # which tiles are dense enough
python3 code/annotation_tiles.py import                           # -> dataset_cache
python3 code/crossval.py                                          # the first honest AM/HC number
```

Three design choices in there are load-bearing, and each is measured:

- **Tiles, not frames.** A held-out IoU is a statistic, so a uniformly-random tile sample
  estimates it without bias and with a quotable interval. 27 tiles of 512×512 is 7.1 M pixels
  — about 0.9× one 8 MP frame — but spread across all 27 AM/HC frames.
- **Sampled uniformly, not where the model is unsure.** Picking uncertain tiles finds more
  crack per tile and would make the ground truth a *biased* sample, so IoU on it would not
  estimate the frame's IoU. Active learning belongs on training tiles; keep the uniform set
  for evaluation.
- **Loaded without a prediction.** A labeller shown the model's mask is being asked to agree
  with it, and 98.3% of this project's existing crack labels are confirmations of exactly
  that. `annotation_tiles.py load` ingests with `predict=False` so the canvas is blank.

`import` refuses a tile with less than 95% of its pixels judged, because a sparse tile
imported as dense counts every unpainted crack pixel as background. Measured on this
project's own sparse corrections, treating them as dense gives a mean IoU of **0.06**.

`pipeline.GT_STEMS` discovers stems from `dataset_cache/` rather than being hardcoded, so a
newly imported tile is picked up by the retrain gate, the cross-validation and the scorecard
at once.

## Security, plainly

**This app has no authentication and is not built to be exposed.** It binds `127.0.0.1`
with `debug=False`, and it should stay there — do not put it behind a lab reverse proxy or a
tunnel as it is. Anyone who can reach the port can read every image, delete any of them, and
start a retrain.

Three things *are* hardened, because they were reachable even bound to localhost:

- **Image ids are validated at a single chokepoint.** `store.path()` refuses anything
  outside `[A-Za-z0-9._-]` or containing `..`. Before that, `DELETE /api/image/%2e%2e`
  handed `shutil.rmtree` the whole `app_data` directory — every correction mask, the model
  registry and the retrain history. `delete_image()` additionally refuses any real path
  outside `app_data/images/`.
- **Uploaded filenames are escaped where they are rendered.** They persist in `meta.json`
  and were interpolated raw into `innerHTML`, so a file arriving in a collaborator's folder
  named `<img src=x onerror=…>.tif` ran in the app's own origin — and could issue exactly
  the DELETE above.
- **Requests must be addressed to this machine.** A `Host` that is not
  `127.0.0.1`/`localhost`/`[::1]` is refused with 403, which is what stops a page on any
  website resolving a name it controls to 127.0.0.1 and talking to your app as same-origin.
  Non-GET requests carrying a foreign `Origin` are refused too.

Model files are unpickled with `joblib`, which executes arbitrary code by construction — so
only load `.joblib` files you produced or trust, exactly as with any scikit-learn artifact.

## What is in here

This one repo is both the tool and the record of how it was built. A new user only
needs the first four entries:

| | |
|---|---|
| `run_app.sh` | the only command you need |
| `app/` | the server and the single-file frontend |
| `code/` | the feature extraction, preprocessing and measurement modules the app imports, plus the batch utilities (`load_all_images.py`, `import_research_corrections.py`, `backup_labels.py`, `clean_gt_conflicting_labels.py`, `crossval.py`, `make_readme_figures.py`, `make_icon.py`) |
| `images/` | **all 71 raw TXM images**, bit-exact. Deflate-compressed float32 TIFF with the floating-point predictor: 2.26 GB instead of 3.40 GB, every file under GitHub's 100 MB limit (two of the originals were 122 MB and could not be pushed at all). Verified 71/71 identical to the originals. Read them with `tifffile` or GDAL; if a tool cannot handle predictor 3, re-save with `tifffile.imwrite(out, tifffile.imread(src))` |
| `models/`, `dataset_cache/`, `paint/corrections/` | the shipped models, the 4 reference ground-truth images, and the correction labels |
| `docs/` | how the model was arrived at. `HANDOFF.md` is the development record including four approaches that were adopted and then reverted; `SAM_COMPARISON.md` is the zero-shot SAM study |
| `research/` | result sets, figures and one-off experiment scaffolding. Nothing here is needed to run the app |
| `app_data/` | your uploads, predictions and corrections. Gitignored -- it is yours, and it is not backed up by this repo |

It used to be two repos: this archive plus a slimmed-down `txm-crack-detector` that
`make_package.sh` generated from it. That split cost a double push on every change and
silently drifted once -- a utility was committed here and left out of the generated copy
because it was missing from an explicit file list. One repo, one push, no list.

## Load the 71 shipped images

The app starts empty -- drag images in, or load everything that ships with the repo:

```bash
# Use the venv's python, or these die with ModuleNotFoundError — ./run_app.sh installs
# into .venv and never touches your system python.
.venv/bin/python code/load_all_images.py
.venv/bin/python code/import_research_corrections.py   # attach the 264 M not-crack labels
```

The loader skips anything already present by filename, so it is safe to re-run. Both are
optional: the app works on images you drop in yourself.

## Using it — every control, and what it is for

Nothing here needs a config file edited or a script run in the right order.

### 1. Get images in

Drag them onto the window, or click the drop zone to browse. Accepted:
`.tif .tiff .png .jpg .jpeg .bmp` — anything else is refused immediately with a message
naming what is supported, rather than failing later inside a background job.

Each image is then destitched, flat-fielded, embedded with SAM and predicted. Budget
~20 s for a 2.9 MP image, a few minutes for a 30 MP mosaic; the status line names the
stage and, on multi-image jobs, shows elapsed time and an estimate of what is left.

To load the 71 images that ship with the repo, `.venv/bin/python code/load_all_images.py`
is much faster than dragging them in — but **budget real time for it on a fresh clone.** That
script reuses `paint/sam_embcache` when present, and that cache is 2.1 GB of derived data
which is *gitignored* — so a clone does not have it. The honest figure for a first run is
**~1050 SAM ViT-H tile passes**: roughly 26 minutes on a GPU, and about 4.4 hours CPU-only.
It is resumable — rerun it and it skips whatever finished — and it only happens once.

### 2. Look at the result

| control | what it does |
|---|---|
| **Show result** | the red predicted-crack overlay on/off |
| **My labels** | the cyan *marked not-crack* overlay on/off. Some specimens are 95% marked, so being able to hide it matters |
| status bar, right | `width×height · % crack · N regions`, for the sensitivity you are viewing. Hover it to see whether you are looking at the destitched+flat-fielded view or raw |
| sidebar rows | thumbnail with the result burned in, `% crack`, `edited` if you have labelled it, and `older model` if its prediction came from a model other than the current one |
| **×** on a row | removes that image and its corrections from the app. Your original file is untouched |

`?image=<part of a filename>` in the URL opens straight to that image, and `&labels=0`
starts with the cyan layer hidden — useful for pointing a colleague at one frame.

### 3. Correct it

Three tools. Keys **1**, **2**, **3**. The status line describes whichever is active.

| tool | gesture | what it does |
|---|---|---|
| **Add crack** | drag | marks crack the model missed |
| **Erase** | drag | marks *only the pixels your brush passes over* as not-crack |
| **Flip region** | one click | marks a *whole connected region* as not-crack. Click it again to flip it back to crack |

**Flip region** resolves your click in three steps, so it works on things the model got
wrong as well as things it got right:

1. clicked a red blob → that blob
2. clicked somewhere the model only *half* fires (probability > 0.15) → that region.
   These are the most valuable negatives you can give it: hard cases on the decision
   boundary
3. clicked plain background → a flood fill from the click, refused if it runs past 25% of
   the image (brush it instead)

**Every stroke saves itself** the moment you release the mouse. There is no save button
and nothing is held in the browser — the correction is on disk before the request
returns, verified by killing the server mid-session and restarting. **⌘Z / Ctrl+Z**
undoes one stroke at a time, 30 deep, and survives a restart. If a save ever fails you
get a red warning and the stroke stays on screen, rather than silently vanishing.

### 4. Advanced (collapsed by default)

These are tuning knobs, not part of the loop:

| control | notes |
|---|---|
| **Brush** | 2–120 px radius |
| **Zoom** / **Fit** | Fit also re-fits when you resize the window, unless you have set a zoom by hand |
| **Sensitivity** | the probability threshold, default 0.50 (calibrated). Lower marks more |
| **Legacy post-processing** | reproduces the older pipeline's cleanup. Off because it measurably removes thin cracks: −0.08 IoU, −0.07 recall, and hand-painted stroke recall falls from ~0.87 to 0.14–0.40. It computes its own mask, so it **disables Sensitivity** while on |
| **Re-apply model** | re-predicts every image with the current model. The recovery path if a model switch was interrupted and some rows still read `older model` |
| **Undo**, **Reset image** | same as ⌘Z; Reset clears all corrections on the open image (⌘Z restores them) |

### 5. Switch models

The dropdown lists the shipped baseline plus every model you have retrained, each marked
`ready`, `N/M ready`, or `needs a pass`. Switching to a model already computed for your
images is **instant** — predictions are cached per (image, model) and hard-linked, so N
models cost N predictions on disk rather than 2N. A model that has not seen an image yet
gets a prediction pass, and the image you are looking at is predicted first.

**Each entry also shows its measured background error** — the share of crack-free
specimen it marks as crack, averaged over the six specimens confirmed to contain no
cracks, computed from cached predictions so it costs nothing to display:

```
retrained 20260818_123934 · ready · 0.14% bg
retrained 20260817_123341 · ready · 22% bg
retrained 20260817_000321 · 4/71 ready · bg not measured
shipped baseline · ready · 0.19% bg
```

That number exists because the picker used to show names only. Two models in a real
history mark **22%** of blank specimen as crack — they predate the false-positive half of
the gate, so they deployed legitimately and remain selectable forever. Someone switched to
one by accident, got visibly worse masks, and there was nothing anywhere in the interface
that could have told them why. Selecting a model measured much worse than the best one you
have now asks for confirmation first, and says by how much. A model never run over those
six specimens reads `bg not measured` rather than a made-up number.

This is also how you roll back: select an earlier model.

### 6. Retrain

Trains on every correction across every image, plus the reference ground truth, then
deploys only if it passes both halves of the gate.

**Every retrain leaves a scorecard** under the model picker, and it persists — reloading the
page or restarting the server does not lose it, which matters because a retrain takes an
hour or two and people reload:

```
held out     0.815  ±0.05
background   0.26%  +0.03pp
in-sample    0.940  ≈ same
deployed 10:27 · details

```

Three numbers, deliberately. Hovering any row explains it and gives the before/after and
the trend; **details** expands to what it trained on, the per-image held-out scores and the
per-specimen background figures. An earlier version of this card put all of that on screen
at once — 998 characters in a 250 px column, with the background figure repeated four
times — which is not a scorecard, it is a wall.

**Two numbers here that pixel overlap cannot give you**, and that no comparable tool
reports. Measured held-out, a flaw counted as detected if *any* of it is marked:

| flaw size | detected | share of all crack pixels |
|---|---|---|
| under 500 px | **25.3%** | 0.3% |
| 500 – 2 k | 28.6% | 0.4% |
| 2 k – 20 k | 69.2% | 3.1% |
| **over 20 k** | **100.0%** | **96.2%** |

It finds every large flaw and a quarter of the small ones — and the last column is why an IoU
hides this: small flaws are 0.3% of crack area, so missing three quarters of them barely
moves the score. And **4.0 false indications per frame** (worst 11, one of six specimens
completely clean) on material confirmed to contain no crack, because "0.106% of area" does not
tell you whether you will dismiss one artifact or thirty. `code/detection_report.py` prints
both; the false-call figure is in every retrain scorecard.

**Held-out IoU comes first because it is the only number that answers "how will this do on
an image it has not seen".** It refits the model once per ground-truth image, each time
leaving that whole image out, so train and test never share an image. On this data it reads
**0.815** against the in-sample **0.939** — the gap is the overfitting, and it is large.

Do **not** replace this with ordinary k-fold. Shuffling pixels into folds leaks and leaks in
the flattering direction, because the 17 hand-crafted features come from neighbourhoods
reaching 256 px and a SAM embedding is a bilinear lookup into a 64×64 grid per 1024 px tile,
so a 16×16 block of pixels shares essentially one embedding vector. Measured with the
deployed architecture on identical rows:

| protocol | mean IoU | fold sd |
|---|---|---|
| the gate's in-sample figure | 0.939 | — |
| random 4-fold, pixels shuffled | 0.930 | **0.003** |
| grouped 4-fold, split by image | **0.824** | 0.050 |

Random k-fold inflates the score by **0.106** and reports a fold spread of 0.003 while doing
it — four genuinely different specimens cannot agree that closely, and that tightness is the
tell. `python3 code/crossval.py --demo-leak` reproduces both columns.

The trend line is the point. All of this was already measured and then thrown away with the
job, so the interface said only "retrain complete" — and three consecutive retrains here
drifted 0.137% → 0.238% → 0.264% on crack-free specimen while ground-truth IoU sat flat at
0.936–0.940. No single retrain looked wrong; the sequence did. A rejected retrain is
recorded too, with the reason, since that is the entry you most want to re-read.

Clicking through shows what it trained on and the per-specimen breakdown, because a mean
that moves 0.14% → 0.26% does not say whether one specimen fell apart or everything softened
slightly.

Two caps decide how much of your work is used, and they are worth knowing:

- **30,000 crack pixels per image.** More than that on one image is discarded, so strokes
  spread over ten images are worth far more than the same effort on one.
- **Negatives per image = (total crack pixels) ÷ (images with negatives)**, to hold the
  class balance near 50/50. So the amount of *background* the model learns from is
  governed by how much *crack* you have drawn — with crack on one image only, 263 M
  not-crack labels are sampled down to ~35,000.

The gate refuses a candidate that either drops IoU on the ground truth by more than 0.01,
**or** raises false positives on the confirmed crack-free specimens by more than 0.5
percentage points. If it refuses, the message says which half failed and by how much, and
the model file is kept so you can inspect it. A refusal is not proof your labels were
wrong — ground truth exists for four images of one specimen group, so the gate is blind
to improvements elsewhere.

When it does deploy it re-applies the new model to every image **inside the same job**, so
nothing is left stale if you close the tab.

### 8. Back up your labels

`app_data/` is gitignored, and should be -- it is 19 GB of predictions, SAM embeddings and
thumbnails, all regenerable. Your correction labels are the exception: nothing can
regenerate them, and they live only on your disk. They are also nearly free to version,
because a correction mask is uint8 and overwhelmingly zero:

```bash
python3 code/backup_labels.py            # 850 MB of masks -> 4.1 MB in paint/app_labels.npz
python3 code/backup_labels.py --status   # what is saved vs what is live
python3 code/backup_labels.py --restore  # write them back after a loss
python3 code/backup_labels.py --prune    # drop archived images not loaded here
git add paint/app_labels.* && git commit -m "labels" && git push
```

Run it after a labelling session. Keyed by filename, not by image id -- an id is a content
hash, so it changes if an image is ever recompressed, and a backup keyed by id would fail
to restore onto the same picture. Restore refuses to overwrite labels the app already has
unless you pass `--force`, since the app's copy is normally the newer one.

### 7. Export

| item | what you get |
|---|---|
| **Black & white mask** | crack = black, PNG |
| **Overlay image** | the display image with red crack and cyan not-crack burned in |
| **Measurements (CSV)** | one row per crack region: area, skeleton length, mean and max width, tortuosity, branch points, orientation, boundary roughness, centroid. Same column definitions as the sibling SEM pipeline |
| **Everything, all images (.zip)** | the three above for every image, plus `summary.csv`. At 71 images this is ~590 MB and takes about five minutes in one request with no progress bar — the menu says so |

Exports honour the sensitivity you are viewing, so what you see is what you get.

## What the model is

A mean-probability ensemble of two models:

- **17 hand-crafted features → MLP.** Intensity, Gaussian-smoothed intensity at
  σ=2…64, gradient magnitude, Laplacian, and local-standard-deviation texture.
- **SAM + those 17 → MLP.** Meta's Segment Anything ViT-H image embedding (256
  channels) concatenated with the same 17 features.

Measured under leave-one-image-out on the 4 Ilastik ground-truth images, with
false positives measured on 6 specimens confirmed crack-free by the specimen owner:

| approach | mean IoU | pixel-weighted IoU | recall | crack-free FP |
|---|---|---|---|---|
| 17 features alone | 0.744 | 0.721 | 0.891 | 7.43% |
| SAM + 17 (hybrid alone) | 0.795 | 0.719 | 0.894 | 0.14% |
| **ensemble of the two (default)** | **0.821** | **0.777** | **0.914** | **0.11%** |

The hybrid alone only *ties* the simple model once you weight by pixel count,
because it loses badly on the largest image — which is 73% of all labelled
pixels. Averaging wins on every image, on both weightings, with recall going up
rather than being traded away. That is why the default is the ensemble.

Zero-shot SAM, used the way SAM is designed to be used (prompt it, read out
masks), scores 0.23–0.36 here and is not usable. See `SAM_COMPARISON.md` for the
full study, including 33 verified citations.

## Things worth knowing before you trust a number

- **Ground truth is 4 images, all one specimen group, all wide-open cracks**
  (median crack width 65 px). With n=4 an exact sign test cannot go below
  p=0.125, so nothing here can be statistically significant. Treat differences
  under ~0.015 IoU as indistinguishable from reseeding — that is the measured
  run-to-run noise.
- **Specks under 2000 px are pruned, and that is measured.** A minimum-area filter
  improves held-out IoU 0.8317 → 0.8391 on 4 of 4 leave-one-image-out folds and cuts
  false positives on crack-free specimen from 0.264% to 0.106%. It runs *before* your
  corrections are applied, so it can never remove crack you painted — `selftest.py`
  asserts that. 2000 rather than the top-scoring 5000 because your own 30.2 M crack
  labels show 5000 costs the worst image 12.4% of its confirmed crack. Details in
  `docs/SAM_COMBINATION_SWEEP.md`.
- **The LEGACY post-processing is still off by default and still under suspicion.** That
  is a different, compound rule — blur, closing, ring rejection, eccentricity *and*
  hysteresis growth — which measured −0.084 IoU and drops hand-painted stroke recall from
  ~0.87 to 0.14–0.40. Isolating its pieces showed the size filter was never the harmful
  part; hysteresis linking was measured separately and is the worst rule of the six.
- **Retrain refuses to deploy a regression.** A candidate must hold IoU within
  0.01 *and* not raise its false-positive rate on the crack-free specimens by more
  than 0.5 points. Every regression this project has had passed a single-metric check —
  an over-aggressive filter and a good one both reduce predicted area, and only
  recall against ground truth separates them.
- **On AM/HC the model scores IoU 0.32, against 0.82 on B2.** Measured on four AM/HC
  frames the owner labelled 90–96% densely, held out by image, scored over judged pixels
  only (`python3 code/eval_dense_labels.py`). The failure is **precision** — 0.355 mean, as
  low as 0.082 — not recall (0.840): it finds the crack and marks 3–12x too much material
  with it. Caveat that needs a second annotator: precision is measured against one person's
  labels, so if the model marks real crack that was called not-crack, this understates it.
- **The model does not transfer to AM/HC, measured a second way.** Leave-one-specimen-group-out on the
  owner's own labels (71 images, all four groups) gives crack recall 0.836 for B2, 0.795 B3,
  0.763 wrought — and **0.397 for AM/HC**. Controlled: holding out 27 *random* images
  instead of the 27 AM/HC ones gives 0.755–0.816, so it is the specimen group and not the
  training-set size. AM/HC also has the highest not-crack agreement (0.973), so the failure
  is under-marking. `python3 code/crossval_groups.py` reproduces it;
  `docs/SAM_COMBINATION_SWEEP.md` has the method and the caveats — these are sparse labels,
  so it is agreement with a human's judgement, not accuracy against truth.
- **Some imported not-crack labels are on real crack, and most cannot be checked.**
  `code/import_research_corrections.py` brought in 263 M not-crack pixels from the
  research archive. On the four images where pixel truth exists, **22–28% of that
  archive's not-crack labels sit on real crack** — the outlines it came from were drawn
  tight, so crack margins came through as background. That is not a harmless missing
  label: it teaches the model that crack margins are background, and margins are where a
  segmentation is decided. `code/clean_gt_conflicting_labels.py` has cleared the 362,851
  such pixels on those four images. **The other 67 images carry negatives from the same
  archive and there is no truth to check them against**, so assume some contamination
  remains and treat the crack-margin behaviour of any retrain with suspicion. Owner-drawn
  force-crack labels, by contrast, agree with ground truth 92–100%.
- **The gate's IoU is in-sample, and that is a real limitation.** A retrain samples
  100 k crack and 100 k background pixels from each of the four ground-truth images,
  and then the gate scores the candidate on those same four images. So "IoU did not
  drop" can be satisfied by fitting them more closely rather than by generalising, and
  a number measured that way runs high: models here score 0.92–0.96 on the ground-truth
  images they trained on, against the 0.821 that leave-one-image-out gives. Quote 0.821.
  The false-positive check on the crack-free specimens is the part of the gate that is
  actually adversarial, and it is what caught the 22.4% model.
- If you retrain and it says *not deployed*, the model file is still saved so you
  can inspect it. **Roll back** restores the previous model.

## Layout

```
app/server.py          the web app
app/core/model.py      the deployed model, one predict() call
app/core/pipeline.py   ingest + retrain, including the validation gate
app/core/store.py      per-image storage and the model registry
app/static/index.html  the whole frontend
code/                  the research pipeline: features, destitch, flatfield, SAM harness
dataset_cache/         the 4 ground-truth images (needed to validate a retrain)
models/                shipped model weights
app_data/              your uploads, embeddings and retrained models (gitignored)
```

Your data lives in `app_data/` inside the checkout. Nothing points at an absolute
path outside it, so moving or deleting your original files cannot break the app.

## Licence

Two licences, because this repository holds both software and experimental data:

- **Code** — MIT, see [LICENSE](LICENSE). Use it, change it, ship it; no warranty.
- **Data** — CC BY 4.0, see [LICENSE-DATA](LICENSE-DATA). That covers `images/`,
  `dataset_cache/`, `paint/corrections/` and the derived results and figures. Free to
  reuse with credit; please cite the repository, or the associated publication once it
  exists.

The distinction matters: a code licence does not grant rights to data, and the raw images
here are experimental measurements rather than software. If you use the labels, read
`docs/HANDOFF.md` section 4 first -- it records which labels are hand-drawn, which are
geometric, and which sources proved untrustworthy.
