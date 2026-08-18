# TXM Crack Detection — Quickstart

Detects cracks in transmission X-ray microscopy images. Drag images in, look at
what the model found, fix what it got wrong, press Retrain. That is the whole loop.

## What it looks like

![The app](docs/img/app.png)

Sidebar lists your images with the result burned into each thumbnail; the toolbar is the
three correction tools, the two view toggles, and Retrain. The model picker is bottom
left. Every stroke saves itself -- there is no save button.

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
barely see. Right is what you mark on. Both steps preserve geometry, so a mask still
registers pixel-for-pixel on the original.

**The model is still fed the raw image**, deliberately: flat-fielding as *model input* was
tried and cost 0.169 IoU, because large-radius intensity features are ~41% of the model's
importance and flat-fielding removes exactly those. So what the human sees and what the
model sees are different on purpose. If preprocessing ever fails, the app falls back to
raw and says so in red rather than letting you mark an unprocessed frame believing
otherwise.

### A crack it finds well

![Detection example](docs/img/example_detection.png)

Left, the image as you see it (destitched and flat-fielded). Right, the model's output.
It traces the fine branching hairline accurately, down to individual branches a few
pixels wide -- and this is an AM/HC specimen, a group with **no** pixel-level ground
truth, so nothing about this frame was used to fit or validate the model.

Look at the top and right edges: the dark off-specimen background is also marked red.
That is a false positive, and it is the honest reason the correction tools exist.

### Correcting it

![Correction example](docs/img/example_correction.png)

Red is predicted crack, cyan is what a human marked as *not* crack. **Flip region** takes
a whole connected blob in one click, so a false positive the size of the frame is one
click rather than a minute of brushing. Press Retrain and those labels become training
data.

### How well does it do, honestly

On the four ground-truth images (all one specimen group, B2) under leave-one-image-out:
mean IoU **0.821**, recall **0.914**, and on six specimens the owner confirmed
crack-free the **shipped baseline** marks **0.21%** of area as crack (measured).

> **Check which model you are on.** Those numbers are the shipped baseline's. A model
> retrained in the app can be far worse at background and still deploy, because until
> recently the gate only compared IoU on the four B2 ground-truth images. A retrain on a
> single image's corrections measured **22.4%** of crack-free specimen area marked as
> crack -- 107x the baseline -- and passed. The gate now also refuses any candidate whose
> false-positive rate on the crack-free specimens rises by more than 0.5 points, but
> models deployed before that fix are still in your history. The model picker's
> `shipped baseline` entry is the measured-good one. Zero-shot SAM, prompted the way SAM is
designed to be prompted, scores 0.23-0.36 on the same images.

Performance varies a lot by specimen group, and the examples above were chosen to show
both ends. Some frames over-predict substantially -- wide red bands around a crack rather
than the crack itself. There is no ground truth outside B2, so outside B2 those numbers
are unverified and your eye is the only judge. `docs/SAM_COMPARISON.md` has the full study
and `docs/HANDOFF.md` records four approaches that were tried and reverted.

## Run it

One command. Nothing to install first, nothing to configure:

```bash
git clone https://github.com/jzhang29-max/TXM_Crack_Detection_Pipeline.git && cd TXM_Crack_Detection_Pipeline && ./run_app.sh
```

Then open **http://127.0.0.1:8800**.

That script creates its own virtualenv, installs dependencies, expands the bundled
reference data, and serves the app. Re-running it later just starts the app -- it
notices the venv already exists and that requirements have not changed.

Python 3.9+ is the only prerequisite. Apple Silicon, CUDA and CPU-only all work; a
GPU makes the SAM step ~10x faster but nothing requires one. `PORT=9000 ./run_app.sh`
if 8800 is taken.

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

## What is in here

This one repo is both the tool and the record of how it was built. A new user only
needs the first four entries:

| | |
|---|---|
| `run_app.sh` | the only command you need |
| `app/` | the server and the single-file frontend |
| `code/` | the feature extraction, preprocessing and measurement modules the app imports, plus the batch utilities (`load_all_images.py`, `import_research_corrections.py`) |
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
python3 code/load_all_images.py          # ~45 min for all 71, reusing the cached SAM embeddings
python3 code/import_research_corrections.py   # attach the 264 M not-crack labels
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

To load the 71 images that ship with the repo, `python3 code/load_all_images.py` is much
faster than dragging them, because it reuses the cached SAM embeddings.

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

This is also how you roll back: select an earlier model. **Check which model you are on
before trusting an overlay** — see the warning under *How well does it do*.

### 6. Retrain

Trains on every correction across every image, plus the reference ground truth, then
deploys only if it passes both halves of the gate.

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
- **Post-processing is off by default and is under suspicion.** The
  shape-validation and minimum-size filter measurably removes thin crack: on one
  ground-truth image it costs 0.084 IoU and 0.072 recall, and hand-painted stroke
  recall drops from ~0.87 at a raw threshold to 0.14–0.40 after it. Toggle it on
  if you want the old behaviour.
- **Retrain refuses to deploy a regression.** A candidate must hold IoU within
  0.01. Every regression this project has had passed a single-metric check —
  an over-aggressive filter and a good one both reduce predicted area, and only
  recall against ground truth separates them.
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
