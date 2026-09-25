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
# NOT activated. bin/activate hardcodes VIRTUAL_ENV as an absolute path written when the
# venv was created; move the checkout and sourcing it prepends a directory that no longer
# exists, so a bare `python3` falls through to the next thing on PATH -- a conda install
# here. The sibling app failed exactly this way on 2026-09-24: it served HTTP 200 while
# running scikit-learn 1.7.2 against bundles pickled by 1.9.0, and every inference died.
# $PY resolves from its own location and is correct even when bin/activate is stale.
PY="$PWD/$VENV/bin/python3"
if [ ! -x "$PY" ]; then
  echo "==> no interpreter at $PY -- rebuild the venv:  rm -rf $VENV && ./run_app.sh"
  exit 1
fi

# Only reinstall when requirements change -- a stamp file keeps startup fast.
STAMP="$VENV/.req-stamp"
if [ ! -f "$STAMP" ] || ! cmp -s requirements.txt "$STAMP"; then
  echo "==> installing dependencies (this takes a few minutes the first time)"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt
  cp requirements.txt "$STAMP"
fi

mkdir -p app_data/images app_data/models models dataset_cache paint/corrections

# Expand the compressed correction masks a distributed checkout ships with
# (1.0 GB of .npy from a ~3 MB archive).
# Idempotent -- a no-op on every run after the first.
#
# --skip-features on purpose: the 17-feature reference stacks are 2.1 GB and take
# minutes to compute, and NOTHING except retraining reads them. Building them here
# meant the first `./run_app.sh` sat silently for several minutes before serving.
"$PY" code/unpack_package.py || echo "==> WARNING: unpack step failed; the shipped correction labels may be missing"

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

# Ask whether ANY model is present, not whether two specific 2026-08-22 filenames are.
# Those two were superseded by the v5 pair on 2026-08-24, so this warning had been firing on
# every start for a month while the app could predict perfectly well -- and, worse, it would
# have stayed SILENT if the models had genuinely gone missing under any other name. A check
# that is wrong in both directions is worse than no check.
if ! ls models/*.joblib >/dev/null 2>&1; then
  echo "==> WARNING: no model found in models/."
  echo "    The app will start but cannot predict until one is present."
fi

# KMP_DUPLICATE_LIB_OK: scikit-learn and torch each vendor their own OpenMP runtime, and on
# macOS both are LLVM libomp, which refuses to initialise twice in one process.
#
# This used to claim that loading both "aborts on macOS without this". That does not
# reproduce with the pinned wheels (torch 2.13.0, scikit-learn 1.9.0): checked in both import
# orders, with the variable unset and with it explicitly FALSE, driving real OpenMP work
# through both runtimes -- no OMP Error #15, exit 0 every time. It is also redundant, because
# sklearn sets it itself at sklearn/__init__.py:56 with an unconditional setdefault, on every
# platform, before it loads its OpenMP-linked extensions.
#
# Kept anyway, as insurance rather than as a known fix: it costs nothing, and the failure it
# guards against is real for other wheel combinations even if this one no longer trips it.
# On Linux it is inert -- both wheels vendor GNU libgomp there, which has no duplicate-library
# check and ignores KMP_*.
export KMP_DUPLICATE_LIB_OK=TRUE
# Let the SAM pass use all of unified memory rather than a fraction of it.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

echo "==> serving on http://127.0.0.1:$PORT"
echo "    drop images onto the window; press Ctrl-C to stop"
exec "$PY" app/server.py
