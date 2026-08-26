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
| held-out IoU (grouped by image) | **0.811** ±0.023, worst fold 0.778 |
| precision / recall | 0.936 / 0.860 |
| crack-free false positives | **0.174%** of area (0.046% after speck pruning), 2.0 indications/frame |
| mask width | ~5 px half-width, against a 2.5–3 px crack |

Averaging beats either member alone on the same basis:

| arm | mean IoU |
|---|---|
| 17 hand-crafted features | 0.726 |
| SAM + 17 hybrid alone | 0.786 |
| **mean-probability ensemble** | **0.811** |

Read from the deployed model's own gate record (`thincore_v5`, stamp `20260824_225236`) rather
than typed in. This table previously carried the v4 figures (0.651 / 0.778 / 0.792) with a
per-fold column, directly under the v5 headline above — the deploy updated the headline and
left the breakdown behind. The per-fold column is dropped rather than reconstructed: the v5
gate records the mean, spread and worst fold, not the individual folds, and inventing five
numbers to fill a column is how the previous version came to disagree with itself.

The older figures for this comparison (17 alone 0.744, hybrid 0.795, ensemble 0.821) came from
leave-one-image-out over 4 externally-labelled frames. Those labels are used nowhere in the
project now, so they are why the ensemble was originally chosen and the table above is why it
stays. Every retrain re-measures it.

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
