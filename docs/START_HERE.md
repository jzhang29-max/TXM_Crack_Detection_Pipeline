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

A mean-probability ensemble of a 17-hand-crafted-feature MLP and a SAM ViT-H + 17
hybrid. Leave-one-image-out on the 4 Ilastik ground-truth images, with false
positives measured on 6 owner-confirmed crack-free specimens:

| approach | mean IoU | pixel-weighted | recall | crack-free FP |
|---|---|---|---|---|
| 17 features alone | 0.744 | 0.721 | 0.891 | 7.43% |
| SAM + 17 (hybrid alone) | 0.795 | 0.719 | 0.894 | 0.14% |
| **ensemble (deployed)** | **0.821** | **0.777** | **0.914** | **0.11%** |

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
