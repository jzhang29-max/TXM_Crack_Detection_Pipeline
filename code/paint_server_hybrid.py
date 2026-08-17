"""
Launch the paint tool in HYBRID mode (SAM+17 predictions on raw input).

A thin wrapper rather than a flag because the launcher config cannot pass
environment variables, and TXM_PAINT_HYBRID has to be set BEFORE paint_common
is imported -- it decides MODEL_PATH and PREDICTED_CACHE_DIR at import time.

Equivalent to:
    TXM_PAINT_HYBRID=1 python3 code/paint_server.py

Serves masks from paint/predicted_cache_hybrid/ (pre-filled by
populate_hybrid_paint_cache.py, so images open instantly). The default
per-group 17-feature path is untouched -- run paint_server.py directly to get
it back.
"""

import os
import sys

os.environ["TXM_PAINT_HYBRID"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paint_server  # noqa: E402  -- import AFTER the env var is set

if __name__ == "__main__":
    # paint_server has no main(); it starts the app under its own __main__
    # guard, which does not run on import. So serve its Flask app directly,
    # mirroring the same PORT handling.
    port = int(os.environ.get("PORT", "8766"))
    print(f"[paint_server_hybrid] HYBRID mode listening on http://127.0.0.1:{port}", flush=True)
    paint_server.app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
