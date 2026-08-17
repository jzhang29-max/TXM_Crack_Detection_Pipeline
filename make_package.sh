#!/usr/bin/env bash
# Assemble a small, cloneable repo for distribution.
#
#   ./make_package.sh [dest]        default: ~/Desktop/txm-crack-detector
#
# Deliberately the same shape as the SEM project's make_package.sh, and for the
# same reason: this folder is the lab's full data archive (~21 GB of caches,
# embeddings, result sets and git history) and a new user needs almost none of
# it. Copying out just the working parts gives a repo of a few tens of MB.
#
# Included: all code, the two deployed models, the human correction masks and the
# 4 ground-truth images -- the correction masks are the one irreplaceable thing
# here, and the ground truth is what lets a user's retrain be validated at all.
# Excluded: source images, SAM embedding caches, prediction caches, result sets,
# figures -- every one regenerable from the images plus the labels.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HOME/Desktop/txm-crack-detector}"

echo "==> packaging from $SRC"
echo "    into           $DEST"

# Rebuild $DEST from scratch, but keep .git if the destination is already the
# published checkout. Without this, re-running the script to ship an update
# deletes the very history you were about to push to -- recoverable only by
# refetching from the remote, and silently, since a fresh `git init` there looks
# like a brand-new repo rather than a decapitated one.
KEEP_GIT=""
if [ -d "$DEST/.git" ]; then
  KEEP_GIT="$(mktemp -d)"
  mv "$DEST/.git" "$KEEP_GIT"/
  echo "    (preserving existing git history)"
fi
rm -rf "$DEST"
mkdir -p "$DEST"/{code,app/core,app/static,models,dataset_cache,paint/corrections,docs}
if [ -n "$KEEP_GIT" ]; then
  mv "$KEEP_GIT"/.git "$DEST"/
  rmdir "$KEEP_GIT"
fi

# --- code. Only what the app imports, plus the research scripts a user might
# rerun. Everything else in code/ is one-off experiment scaffolding from the
# development history and is deliberately left behind.
for f in txm_features.py destitch.py flatfield.py txm_preprocess.py \
         sem_crack_measurements.py apply_pixel_model.py unpack_package.py \
         generate_benchmark_report.py sam_common.py; do
  [ -f "$SRC/code/$f" ] && cp "$SRC/code/$f" "$DEST"/code/
done
cp "$SRC"/app/server.py      "$DEST"/app/
cp "$SRC"/app/selftest.py    "$DEST"/app/
cp "$SRC"/app/core/*.py      "$DEST"/app/core/
cp "$SRC"/app/static/*.html  "$DEST"/app/static/

# --- the deployed models only. The superseded ones are history, not product. ---
for m in pixel_hgb_final.joblib pixel_sam_hybrid.joblib pixel_flatfield_hgb.joblib; do
  [ -f "$SRC/models/$m" ] && cp "$SRC/models/$m" "$DEST"/models/
done

# --- ground truth + corrections, COMPRESSED ---
# Shipping these raw would be a 5 GB repo: the 17-feature stacks alone are 2.1 GB
# (LARGE_343_75 is 1.5 GB of it) and the correction masks are 1.0 GB of
# overwhelmingly-zero uint8. Packing them cuts the corrections ~346x, and the
# feature stacks are dropped entirely because they are a pure function of the
# image -- code/unpack_package.py recomputes them on first run.
python3 - "$SRC" "$DEST" <<'PYPACK'
import glob, os, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
src, dest = sys.argv[1], sys.argv[2]

# Images go out as 16-bit PNG, not float32 .npy. As float32 they compress badly
# (94 MB -> 78 MB for LARGE_343_75, over GitHub's 100 MB per-file hard limit);
# as uint16 PNG the same image is 37 MB and the whole set is 52 MB instead of 107.
# The cost is one part in 65535: measured max abs error 1.5e-05, which moves the
# 17-feature model's IoU by 9e-06 and flips 100 of 2.86M mask pixels. The source
# arrays are percentile-normalised detector counts, so this is well below the
# precision the data ever had.
n_img = 0
for p in sorted(glob.glob(os.path.join(src, "dataset_cache", "*_img.npy"))):
    stem = os.path.basename(p)[:-len("_img.npy")]
    a = np.load(p)
    out = os.path.join(dest, "dataset_cache", f"{stem}_img.png")
    Image.fromarray((np.clip(a, 0, 1) * 65535).astype(np.uint16)).save(out, format="PNG", optimize=True)
    n_img += 1
    print(f"    {stem}_img -> PNG ({os.path.getsize(out)/1e6:.1f} MB)")

# Masks are boolean and compress to nothing, so one small npz is fine.
gt = {}
for p in sorted(glob.glob(os.path.join(src, "dataset_cache", "*_gt.npy"))):
    gt[os.path.basename(p)[:-4]] = np.load(p)
if gt:
    out = os.path.join(dest, "dataset_cache", "masks.npz")
    np.savez_compressed(out, **gt)
    print(f"    {len(gt)} ground-truth masks -> masks.npz ({os.path.getsize(out)/1e6:.2f} MB)")

corr = {}
for p in sorted(glob.glob(os.path.join(src, "paint", "corrections", "*_correction.npy"))):
    corr[os.path.basename(p)[:-len("_correction.npy")]] = np.load(p)
if corr:
    out = os.path.join(dest, "paint", "corrections", "corrections.npz")
    np.savez_compressed(out, **corr)
    print(f"    {len(corr)} correction masks -> corrections.npz ({os.path.getsize(out)/1e6:.1f} MB)")
PYPACK

m="$SRC/dataset_cache/manifest.json"; [ -f "$m" ] && cp "$m" "$DEST"/dataset_cache/

# --- docs and entry points ---
# QUICKSTART becomes README.md: what a new user needs is "how do I run this",
# not the research history. The history is real and worth keeping (it records
# four reverted approaches so nobody repeats them) but it belongs one level down.
cp "$SRC"/QUICKSTART.md "$DEST"/README.md
for f in requirements.txt run_app.sh; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DEST"/
done
mkdir -p "$DEST"/docs
[ -f "$SRC/HANDOFF.md" ]        && cp "$SRC/HANDOFF.md"        "$DEST"/docs/RESEARCH_NOTES.md
[ -f "$SRC/SAM_COMPARISON.md" ] && cp "$SRC/SAM_COMPARISON.md" "$DEST"/docs/
[ -f "$SRC/APP_COMPARISON.md" ] && cp "$SRC/APP_COMPARISON.md" "$DEST"/docs/
cat > "$DEST"/docs/README.md <<'DOCEOF'
# Background documents

These record how the model was arrived at. None of it is needed to use the app --
start with the README in the parent directory.

- **SAM_COMPARISON.md** — the full study behind the model choice: zero-shot SAM
  measured against the deployed classifier, with 33 verified citations. Answers
  "why not just use Segment Anything?"
- **RESEARCH_NOTES.md** — the development record, including four approaches that
  were adopted and then reverted as regressions (flat-fielding as model input,
  geometric masking, a curvilinearity gate, algorithmic crack labels). Kept so
  they are not retried, and because the reasoning behind the metric rules lives
  here: an over-aggressive filter and a good one both reduce predicted area, and
  only recall against ground truth separates them.
- **APP_COMPARISON.md** — how this app compares to the sibling SEM pipeline's,
  what was copied in each direction, and the shared layout both now use.
DOCEOF
chmod +x "$DEST"/run_app.sh 2>/dev/null || true

cat > "$DEST"/.gitignore <<'EOF'
__pycache__/
*.pyc
.venv/
.DS_Store
# per-user runtime data: uploads, SAM embeddings, predictions, retrained models
app_data/
# unpacked artifacts -- regenerated from the shipped PNG/npz by unpack_package.py.
# Must stay ignored: run_app.sh expands these on first start, and the 17-feature
# stacks alone are 2.1 GB, so without this a user's first `git status` offers a
# repo-breaking commit.
dataset_cache/*.npy
paint/corrections/*_correction.npy
EOF

echo
echo "==> done: $(du -sh "$DEST" | cut -f1)"
echo "    files: $(find "$DEST" -type f | wc -l | tr -d ' ')"
echo
echo "    A new user runs:"
echo "      cd $(basename "$DEST") && ./run_app.sh"
echo
echo "    Note: the SAM weights (~2.4 GB) are NOT bundled -- they download from"
echo "    HuggingFace on first prediction and cache in ~/.cache/huggingface."
