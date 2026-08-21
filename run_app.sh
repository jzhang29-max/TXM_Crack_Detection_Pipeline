#!/usr/bin/env bash
# Start the TXM crack-detection app. Nothing to configure.
#
#   ./run_app.sh
#
# Creates a virtualenv on first run, installs dependencies, then serves the app
# at http://127.0.0.1:8800 -- drag images in, correct them, press Retrain.
#
# Deliberately the same shape as the SEM project's run_app.sh so the two behave
# identically: same venv-on-first-run, same requirements stamp, same "warn about
# the optional heavy dependency rather than failing" behaviour.
set -euo pipefail
cd "$(dirname "$0")"

# Exported, not just set: server.py reads PORT from the environment, so without
# this the "serving on ..." line below and the port actually bound could disagree
# any time this default is edited.
export PORT="${PORT:-8800}"
VENV="${VENV:-.venv}"

if [ ! -d "$VENV" ]; then
  echo "==> creating virtualenv in $VENV (first run only)"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"

# Only reinstall when requirements change -- a stamp file keeps startup fast.
STAMP="$VENV/.req-stamp"
if [ ! -f "$STAMP" ] || ! cmp -s requirements.txt "$STAMP"; then
  echo "==> installing dependencies (this takes a few minutes the first time)"
  python3 -m pip install --quiet --upgrade pip
  python3 -m pip install --quiet -r requirements.txt
  cp requirements.txt "$STAMP"
fi

mkdir -p app_data/images app_data/models models dataset_cache paint/corrections

# Expand the compressed ground truth / corrections a distributed checkout ships with.
# Idempotent -- a no-op on every run after the first.
#
# --skip-features on purpose: the 17-feature reference stacks are 2.1 GB and take
# minutes to compute, and NOTHING except retraining reads them. Building them here
# meant the first `./run_app.sh` sat silently for several minutes before serving.
# pipeline.ensure_gt_features() builds them on the first Retrain instead, with
# progress in the UI.
python3 code/unpack_package.py --skip-features || echo "==> WARNING: unpack step failed; retrain validation may be unavailable"

# SAM is optional, and as of this version that is TRUE rather than aspirational: if the
# import fails or the weights cannot be fetched, ingest catches it, predicts with the
# 17-feature model alone, and records the reason in each image's model line so the app says
# which model produced the mask. This message used to promise a fallback that did not
# exist, and a machine behind a firewall got a red job error on every single image.
HFHUB="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--facebook--sam-vit-huge"
if ! python3 -c "import torch" 2>/dev/null; then
  echo "==> NOTE: PyTorch not installed, so SAM is unavailable."
  echo "    The app runs on the 17-feature model alone"
  echo "    (held-out mean IoU 0.744 vs 0.821 for the SAM ensemble)."
  echo "    To enable it:  pip install torch transformers"
elif [ -n "${TXM_NO_SAM:-}" ]; then
  echo "==> TXM_NO_SAM is set: predicting with the 17-feature model only."
elif [ ! -d "$HFHUB" ] && ! python3 -c "import socket;socket.setdefaulttimeout(4);socket.create_connection(('huggingface.co',443)).close()" 2>/dev/null; then
  # Weights absent AND the hub unreachable: say so NOW, not after the user drops in an
  # image and waits through a failing 2.4 GB download.
  echo "==> NOTE: SAM weights are not cached and huggingface.co is unreachable."
  echo "    The app will start and predict with the 17-feature model alone"
  echo "    (held-out mean IoU 0.744 vs 0.821). Each image will say so in its model line."
  echo "    For the full model: fetch it on a connected machine and copy"
  echo "    ~/.cache/huggingface across, or set TXM_NO_SAM=1 to stop retrying."
fi

if [ ! -f models/hybrid_nogt_20260821.joblib ] && [ ! -f models/pixel_hgb_final.joblib ]; then
  echo "==> WARNING: no model found in models/."
  echo "    The app will start but cannot predict until one is present."
fi

# KMP_DUPLICATE_LIB_OK: scikit-learn and torch each vendor an OpenMP runtime, and
# loading both in one process aborts on macOS without this.
export KMP_DUPLICATE_LIB_OK=TRUE
# Let the SAM pass use all of unified memory rather than a fraction of it.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

echo "==> serving on http://127.0.0.1:$PORT"
echo "    drop images onto the window; press Ctrl-C to stop"
exec python3 app/server.py
