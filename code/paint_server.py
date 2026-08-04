"""
Local web app for manually correcting the TXM pixel classifier's crack mask
directly in the browser -- for the "middle parts" the model didn't cover
well (missed interior gaps) or over-covered (scattered false-positive
speckle, seen especially on the LARGE image).

Adapted from the CBS SEM project's interior_active_learning paint tool
(same canvas-painting mechanic: a transparent paint layer is composited
server-side onto the current template and diffed to find fresh strokes),
simplified because TXM has no discrete candidate-region list or artifact
class to manage -- just a binary crack/not-crack mask per pixel.

Run:
    python3 paint_server.py

then open http://127.0.0.1:8766 in a browser. (Port 8766, not 8765, so it
can run alongside the CBS project's own paint server without clashing.)
"""

import os
import sys

from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc
from paint_frontend import INDEX_HTML

import io
import base64
import numpy as np
from PIL import Image

app = Flask(__name__)


def _png_response(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    from flask import Response
    return Response(buf.getvalue(), mimetype="image/png")


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/images")
def api_images():
    out = []
    for info in pc.list_images():
        try:
            stats = pc.region_stats(info["name"])
        except Exception as e:
            stats = {"n_regions": -1, "area_fraction": 0.0}
        out.append({"name": info["name"], **stats})
    return jsonify(out)


@app.route("/api/template/<name>")
def api_template(name):
    return _png_response(pc.build_template(name))


@app.route("/api/paintlayer/<name>")
def api_paintlayer(name):
    """Resume a previous session: return just the saved strokes as a
    transparent PNG (diffed against the current template)."""
    painted_path = os.path.join(pc.CORRECTIONS_DIR, f"{name}_painted.png")
    if not os.path.exists(painted_path):
        return ("", 204)

    template = np.array(pc.build_template(name))
    painted = np.array(Image.open(painted_path).convert("RGB"))
    if painted.shape != template.shape:
        return ("", 204)

    red_mask = pc.color_mask(painted, template, pc.RED)
    cyan_mask = pc.color_mask(painted, template, pc.CYAN)

    h, w = template.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[red_mask] = [255, 0, 0, 255]
    rgba[cyan_mask] = [0, 204, 255, 255]
    return _png_response(Image.fromarray(rgba, mode="RGBA"))


@app.route("/api/save/<name>", methods=["POST"])
def api_save(name):
    try:
        data = request.get_json()
        header, b64 = data["dataURL"].split(",", 1)
        layer = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

        template = pc.build_template(name)
        if layer.size != template.size:
            return jsonify({"ok": False, "error": f"layer size {layer.size} != template size {template.size}"}), 400

        composited = template.copy()
        composited.paste(layer, (0, 0), mask=layer)
        painted_path = os.path.join(pc.CORRECTIONS_DIR, f"{name}_painted.png")
        composited.save(painted_path)

        result = pc.apply_paint_layer(name, np.array(composited))
        stats = pc.region_stats(name)
        return jsonify({"ok": True, **result, **stats})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/flip_region/<name>", methods=["POST"])
def api_flip_region(name):
    try:
        data = request.get_json()
        x, y = int(round(data["x"])), int(round(data["y"]))
        result = pc.flip_region(name, x, y)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8766, debug=False, threaded=True)
