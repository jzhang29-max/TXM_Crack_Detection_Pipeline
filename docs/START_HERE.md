> **HISTORICAL — written before the app existed.** The commands in this file refer to
> the research-era paint tools (`research/code/paint_server.py`, `research/code/make_worklist.py`), which
> now live in `research/code/` and are superseded by the web app. To label, run
> `./run_app.sh` from the repo root and open http://127.0.0.1:8800. The measurements and
> the priority ordering below are still accurate and still worth reading; only the
> commands and file paths are out of date.

# START HERE

Two ways in, depending on what you want.

## Just use it

```bash
./run_app.sh
```

Opens http://127.0.0.1:8800. Drag TXM images in, the current model predicts them,
paint corrections, press **Retrain on my corrections**. Nothing to configure.

First run creates a virtualenv, installs dependencies, and expands the
compressed ground truth and correction masks. The SAM weights (~2.4 GB) download
from HuggingFace on the first prediction.

Verify your install any time:

```bash
python3 app/selftest.py            # 30 checks
python3 app/selftest.py --retrain   # plus a full retrain (slow)
```

## Hand it to someone else

```bash
./make_package.sh
```

Builds a 55 MB clone-ready repo at `~/Desktop/txm-crack-detector` containing only
what the app needs. Already published at
https://github.com/jzhang29-max/txm-crack-detector

## This folder vs that one

**This** folder is the lab archive: every experiment, every figure, the SAM
comparison study, the research history, and the SAM embedding cache. It is the
record of how the model was arrived at.

**That** folder is the product: 33 files, no dead ends.

## What the model is

A mean-probability ensemble of a 17-hand-crafted-feature MLP and a SAM ViT-H + 17 hybrid.
As deployed, measured by cross-validation grouped by whole image — train and test never share
a frame — with false positives on 6 owner-confirmed crack-free specimens:

| | value |
|---|---|
| held-out IoU (grouped by image) | **0.789** ±0.039, worst fold 0.721 |
| precision / recall | 0.933 / 0.837 |
| crack-free false positives | **0.209%** of area, 1.83 indications/frame |

The per-member comparison that chose the ensemble over either member alone was measured on
leave-one-image-out over 4 externally-labelled frames (17 alone 0.744, hybrid alone 0.795,
ensemble 0.821). Those labels came from another tool and are used nowhere in the project now,
so that table is history, not a current measurement — it is why the ensemble was chosen, and
the numbers above are what the choice delivers on the basis that remains.

## Read next

- `QUICKSTART.md` — using the app, and the caveats that matter
- `SAM_COMPARISON.md` — why not just use Segment Anything, with 33 verified citations
- `APP_COMPARISON.md` — this app vs the SEM one, and the bug that comparison found
- `HANDOFF.md` — the full research record, including four reverted approaches

## Storage note

The 71 correction masks are tracked as `paint/corrections/corrections.npz` (3.2 MB)
rather than as raw `.npy` (850 MB). `code/unpack_package.py` restores them, and
`run_app.sh` calls it automatically. The caches under `paint/` and `app_data/` are
regenerable and untracked.
