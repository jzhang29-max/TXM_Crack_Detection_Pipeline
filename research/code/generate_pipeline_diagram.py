"""Full-workflow figure for the TXM crack-detection pipeline, as deployed.

Six inference stages in a serpentine grid, then the human-correction and gated-retrain loop.
Every thumbnail is a real array produced by the app's own modules -- see
compute_diagram_stages.py, which imports app/core/model.py and app/core/pipeline.py rather
than reimplementing them, so this figure cannot quietly describe a system that is no longer
shipping.

WHAT CHANGED FROM THE PREVIOUS VERSION, and why the old figure was wrong:
  * destitch and flat-field were missing entirely, and (A) implied the percentile stretch fed
    the model. It does not -- the stretch is for the human, and the model is fed the raw
    normalised array on purpose.
  * SAM was absent. The deployed model is a mean-probability ensemble of MLP(17) and
    MLP(17 + SAM ViT-H 256-d); the old figure showed one MLP on 17 features.
  * (D) documented the legacy hysteresis post-processing, which is off by default because it
    measurably deletes thin crack (-0.08 IoU). The default is a threshold plus speck pruning.
  * the gate had five checks and trained on Ilastik-derived bootstrap labels. It now has
    three recipe-aware axes, and those reference frames are held OUT of training.

Usage:
    python3 generate_pipeline_diagram.py [dataset_cache_stem]
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagram_helpers import (
    SVG, esc, wrap_tspans, rounded_rect, draw_card, draw_document,
    doc_cy_above, doc_arrow_start_y, stat_card, DOC_COLOR, INK, SUBTEXT, _darken,
)
from compute_diagram_stages import compute_stages, ROOT

# The SVG is a working file: it embeds every thumbnail as base64, so it is ~30 MB and each
# regeneration would add another 30 MB blob to git history forever. It is gitignored. The PNG
# is the deliverable and lives with the other README figures.
OUT_DIR = os.path.join(ROOT, "pipeline_diagram")
PNG_DIR = os.path.join(os.path.dirname(ROOT), "docs", "img")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)
IMAGE_KEY = sys.argv[1] if len(sys.argv) > 1 else "338_13"

STAGE_COLORS = ["#264653", "#287271", "#2A9D8F", "#E9B44C", "#E76F51", "#A63A50"]
PART2_COLORS = ["#7A4C9E", "#0F6E56"]
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"



def _shrink(a, how="mean", target=700):
    """Downsample a panel before it is base64-embedded.

    Thumbnails render about 270 px wide in a 1350-px-wide figure, so embedding
    1700-px arrays put ~40x more data in the SVG than any pixel of it could show. The cost
    was not just size: one data URI reached 22 million characters and libxml2 refused to
    parse the file at all ("Buffer size limit exceeded"), so rsvg-convert produced no PNG.

    `how` matters, and a single rule would corrupt half the panels. Thin crack is one or two
    pixels wide, so whichever direction represents crack has to survive the reduction:
      mean -- continuous greyscale, where averaging is the honest resampling
      max  -- probability maps, where crack is the HIGH value
      min  -- masks and overlays, where crack is black or red and averaging washes it out
    """
    a = np.asarray(a)
    h, w = a.shape[:2]
    f = int(np.ceil(max(h, w) / target))
    if f <= 1:
        return a
    h2, w2 = (h // f) * f, (w // f) * f
    a = a[:h2, :w2]
    op = {"mean": np.mean, "max": np.max, "min": np.min}[how]
    if a.ndim == 2:
        out = op(a.reshape(h2 // f, f, w2 // f, f), axis=(1, 3))
    else:
        out = op(a.reshape(h2 // f, f, w2 // f, f, a.shape[2]), axis=(1, 3))
    return out.astype(a.dtype) if a.dtype == np.uint8 else out


def main():
    print(f"Building pipeline diagram for '{IMAGE_KEY}' (runs the deployed model) ...")
    s = compute_stages(IMAGE_KEY)
    name = s["name"]

    stages = [
        dict(key="A", title="Destitch", icon="sliders",
             subtitle="The mosaic tile grid is periodic, so it is one to two frequency bins; "
                      "those bins are notched out",
             thumbs=[("As received", _shrink(s["img01"]), None),
                     ("Destitched", _shrink(s["destitched"]), None)]),
        dict(key="B", title="Flat-field, for the human only", icon="sun",
             subtitle="Divided by an anisotropic Gaussian, then stretched 1st-99th "
                      "percentile. Geometry preserved",
             thumbs=[("Destitched", _shrink(s["destitched"]), None),
                     ("Display view: what you mark on", _shrink(s["display"]), None)]),
        dict(key="C", title="17 hand-crafted features", icon="network",
             subtitle="Intensity, Gaussian smooths, gradient magnitude, Laplacian and local "
                      "std, sigma 1-64 px",
             thumbs=[("Model input: the RAW array", _shrink(s["img01"]), None),
                     (f"{s['feature_name']} (1 of 17)", _shrink(s["feature_map"]), "viridis")]),
        dict(key="D", title="SAM ViT-H embedding", icon="grid",
             subtitle=f"{s['n_tiles']} tiles of 1024 px, each giving a 64x64 grid of "
                      f"{s['emb_channels']}-d vectors -- one per 16x16 block",
             thumbs=[("Model input: the RAW array", _shrink(s["img01"]), None),
                     (f"SAM embedding: 3 PCs of {s['emb_channels']}", s["sam_rgb"], None)]),
        dict(key="E", title="Ensemble prediction", icon="classifier",
             subtitle="Mean probability of MLP(17) and MLP(17+SAM). Averaging is what wins "
                      "on the large mosaics",
             thumbs=[("MLP(17) alone", _shrink(s["p17"], "max"), "inferno"),
                     ("Averaged with MLP(17+SAM)", _shrink(s["p_ens"], "max"), "inferno")]),
        dict(key="F", title="Threshold and speck pruning", icon="magnifier",
             subtitle="Probability > 0.50, then blobs under 2000 px dropped. Legacy "
                      "hysteresis cleanup is OFF by default",
             thumbs=[("Raw > 0.50 threshold", _shrink(s["raw_thresh_display"], "min"), None),
                     ("Final mask, crack = black", _shrink(s["final_mask_display"], "min"), None)]),
    ]

    # icon_svg returns nothing for an unknown kind, which draws a blank white badge and is
    # easy to miss in a 2700x6750 figure -- (D) shipped that way once. Check the names.
    VALID_ICONS = {"sliders", "sun", "magnifier", "network", "classifier", "grid",
                   "check", "brush", "shield", "loop"}
    for st in stages:
        if st["icon"] not in VALID_ICONS:
            raise SystemExit(f'stage ({st["key"]}) uses icon "{st["icon"]}", which icon_svg '
                             f'does not implement -- it would render as an empty badge. '
                             f'Valid: {sorted(VALID_ICONS)}')

    gate_lines = s["gate_lines"] or [
        ("Reference frames (held out)", "not measured yet", True),
        ("Crack-free specimens", "not measured yet", True),
        ("Grouped-by-image cross-val", "not measured yet", True),
    ]
    gate_card = stat_card("Retrain gate: three axes", gate_lines, PART2_COLORS[1],
                          FONT_PATH, FONT_BOLD_PATH)

    part2_stages = [
        dict(key="G", title="Manual correction", icon="brush",
             subtitle="Add missed crack, erase false positives, or flip a whole region. "
                      "Every stroke saves itself",
             thumbs=[("Model output on the display view", _shrink(s["overlay_model"], "min"), None),
                     ("Hand-labelled truth", _shrink(s["overlay_gt"], "min"), None)]),
        dict(key="H", title="Gated retrain", icon="shield",
             subtitle="Trains on your corrections ONLY. The four reference frames are the "
                      "held-out test set, not training data",
             thumbs=[("Gate report (real run)", np.array(gate_card), None),
                     ("Deployed result", _shrink(s["overlay_model"], "min"), None)]),
    ]

    # ---------------------------------------------------------- geometry
    W = 1350
    MARGIN = 70
    CARD_GAP = 60
    CARD_W = (W - 2 * MARGIN - CARD_GAP) / 2
    CARD_H = 470
    ROW_GAP = 190
    TITLE_H = 260
    PART2_TITLE_H = 110
    PART2_CARD_H = 460
    CAPTION_H = 320

    n_rows = (len(stages) + 1) // 2
    col_x = [MARGIN, MARGIN + CARD_W + CARD_GAP]
    row_y = [TITLE_H + r * (CARD_H + ROW_GAP) for r in range(n_rows)]
    # Serpentine: row 0 left->right, row 1 right->left, and so on. Index i sits in
    # row i//2, and in the column that continues the snake.
    positions = []
    for i in range(len(stages)):
        r = i // 2
        left_to_right = (r % 2 == 0)
        c = (i % 2) if left_to_right else (1 - (i % 2))
        positions.append((col_x[c], row_y[r], c, r))

    last_row_cols = [p[2] for p in positions if p[3] == n_rows - 1]
    out_col = col_x[last_row_cols[-1]]
    out_y = row_y[-1] + CARD_H + 90
    part2_y0 = out_y + 200
    p2_row_y = part2_y0 + PART2_TITLE_H
    p2_bottom = p2_row_y + PART2_CARD_H
    loop_y = p2_bottom + 55
    H = loop_y + 40 + 50 + CAPTION_H

    svg = SVG(W, H)
    svg.add_shadow_filter("cardShadow", dy=10, blur=16, opacity=0.16)
    svg.add_shadow_filter("thumbShadow", dy=4, blur=7, opacity=0.22)
    svg.add_shadow_filter("badgeShadow", dy=3, blur=5, opacity=0.30)
    for i, color in enumerate(STAGE_COLORS):
        svg.add_gradient(f"grad{i}", color, _darken(color, 0.72))
        svg.add_arrowhead(f"arrow{i}", color)
    for i, color in enumerate(PART2_COLORS):
        svg.add_gradient(f"grad2_{i}", color, _darken(color, 0.72))
        svg.add_arrowhead(f"arrow2_{i}", color)
    svg.add_gradient("bg", "#F7FAFC", "#EAF1F3")
    svg.add_arrowhead("arrowDoc", DOC_COLOR)

    svg.raw(f'<rect x="0" y="0" width="{W}" height="{H}" fill="url(#bg)"/>')
    dots = []
    for gx in range(0, W, 26):
        for gy in range(0, H - CAPTION_H, 26):
            dots.append(f'<circle cx="{gx}" cy="{gy}" r="1" fill="#1B242C" opacity="0.035"/>')
    svg.raw("".join(dots))

    svg.raw(f'<text x="{W/2}" y="58" text-anchor="middle" font-family="Georgia, \'Times New Roman\', serif" '
            f'font-size="32" font-weight="700" fill="{INK}">Crack Detection in TXM Images</text>')
    svg.raw(f'<text x="{W/2}" y="90" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="15.5" font-style="italic" fill="{SUBTEXT}">Destitch → features + SAM '
            f'embedding → ensemble → correction → gated retrain. Worked example: {esc(name)}</text>')

    boxes = {}
    for i, stage in enumerate(stages):
        x, y, c, r = positions[i]
        boxes[i] = (x, y, x + CARD_W, y + CARD_H, c, r)
        draw_card(svg, x, y, CARD_W, CARD_H, stage, STAGE_COLORS[i], f"grad{i}")

    # horizontal connector inside a row, in whichever direction the snake runs
    for i in range(len(stages) - 1):
        xi0, yi0, xi1, yi1, ci, ri = boxes[i]
        xj0, yj0, xj1, yj1, cj, rj = boxes[i + 1]
        if ri == rj:
            ay = yi0 + 60
            if xj0 >= xi1:
                x_from, x_to = xi1 - 6, xj0 + 6
            else:
                x_from, x_to = xi0 + 6, xj1 - 6
            svg.raw(f'<path d="M {x_from} {ay} L {x_to} {ay}" stroke="{STAGE_COLORS[i+1]}" '
                    f'stroke-width="4" fill="none" marker-end="url(#arrow{i+1})"/>')
        else:
            cxw = (xi0 + xi1) / 2
            svg.raw(f'<path d="M {cxw} {yi1} C {cxw+40} {yi1+ROW_GAP*0.4}, '
                    f'{cxw-40} {yj0-ROW_GAP*0.4}, {cxw} {yj0}" stroke="{STAGE_COLORS[i+1]}" '
                    f'stroke-width="4" fill="none" marker-end="url(#arrow{i+1})"/>')

    # ---- input document above A ----
    docA_size = 78
    docA_x = boxes[0][0] + CARD_W * 0.5
    docA_y = doc_cy_above(row_y[0], docA_size)
    draw_document(svg, docA_x, docA_y, docA_size, "Raw TXM TIFF\n(float32)", "image")
    svg.raw(f'<path d="M {docA_x} {doc_arrow_start_y(docA_y, docA_size)} L {docA_x} {row_y[0]-4}" '
            f'stroke="{DOC_COLOR}" stroke-width="3" fill="none" marker-end="url(#arrowDoc)"/>')

    # ---- documents feeding the prediction stage (E) ----
    ex0, ey0, ex1, ey1, ec, er = boxes[4]
    mcx = (ex0 + ex1) / 2
    doc_size = 66
    doc_y = doc_cy_above(row_y[er], doc_size)
    doc_arrow_y = doc_arrow_start_y(doc_y, doc_size)
    spacing = (CARD_W - 40) / 2
    for x, label, glyph in [(mcx - spacing, "17-D feature\nstack", "table"),
                            (mcx + spacing, "SAM embedding\n(emb.npz)", "model")]:
        draw_document(svg, x, doc_y, doc_size, label, glyph)
        svg.raw(f'<path d="M {x} {doc_arrow_y} L {x} {row_y[er]-4}" stroke="{DOC_COLOR}" '
                f'stroke-width="3" fill="none" marker-end="url(#arrowDoc)"/>')

    # ---- output documents below the last stage ----
    fx0, fy0, fx1, fy1, fc, fr = boxes[len(stages) - 1]
    out_cx = (fx0 + fx1) / 2
    out_size = 66
    out_docs = [("Black & white\nmask (.png)", "image"), ("Overlay\n(.png)", "image"),
                ("Region stats\n(.csv)", "table")]
    out_spacing = (CARD_W - 60) / 2
    svg.raw(f'<path d="M {out_cx} {fy1} L {out_cx} {out_y - out_size/2 - 8}" '
            f'stroke="{DOC_COLOR}" stroke-width="3" fill="none" marker-end="url(#arrowDoc)"/>')
    for k, (label, glyph) in enumerate(out_docs):
        x = out_cx + (k - 1) * out_spacing
        draw_document(svg, x, out_y, out_size, label, glyph)
        if k != 1:
            svg.raw(f'<path d="M {out_cx} {fy1 + 30} L {x} {out_y - out_size/2 - 8}" '
                    f'stroke="{DOC_COLOR}" stroke-width="2.5" fill="none" '
                    f'marker-end="url(#arrowDoc)" opacity="0.85"/>')

    # ---- part 2 ----
    svg.raw(f'<path d="M {MARGIN} {part2_y0 - 40} L {W-MARGIN} {part2_y0 - 40}" '
            f'stroke="#C7D2D7" stroke-width="2" stroke-dasharray="7 7" fill="none"/>')
    svg.raw(f'<text x="{W/2}" y="{part2_y0 + 18}" text-anchor="middle" '
            f'font-family="Georgia, serif" font-size="24" font-weight="700" '
            f'fill="{PART2_COLORS[1]}">Correction and gated retraining</text>')
    svg.raw(f'<text x="{W/2}" y="{part2_y0 + 44}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="14" font-style="italic" '
            f'fill="{SUBTEXT}">The candidate deploys itself only if all three axes hold — '
            f'and the reference frames it is judged on are never trained on</text>')
    p2_boxes = {}
    for i, stage in enumerate(part2_stages):
        x = col_x[i]
        p2_boxes[i] = (x, p2_row_y, x + CARD_W, p2_row_y + PART2_CARD_H)
        draw_card(svg, x, p2_row_y, CARD_W, PART2_CARD_H, stage, PART2_COLORS[i], f"grad2_{i}")
    ay = p2_row_y + 60
    svg.raw(f'<path d="M {p2_boxes[0][2]-6} {ay} L {p2_boxes[1][0]+6} {ay}" '
            f'stroke="{PART2_COLORS[1]}" stroke-width="4" fill="none" marker-end="url(#arrow2_1)"/>')
    svg.raw(f'<path d="M {(p2_boxes[1][0]+p2_boxes[1][2])/2} {p2_boxes[1][3]} '
            f'L {(p2_boxes[1][0]+p2_boxes[1][2])/2} {loop_y} '
            f'L {(p2_boxes[0][0]+p2_boxes[0][2])/2} {loop_y} '
            f'L {(p2_boxes[0][0]+p2_boxes[0][2])/2} {p2_boxes[0][3]}" '
            f'stroke="{PART2_COLORS[1]}" stroke-width="3" stroke-dasharray="8 6" fill="none" '
            f'marker-end="url(#arrow2_1)"/>')
    svg.raw(f'<text x="{W/2}" y="{loop_y+22}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="13" font-style="italic" fill="{SUBTEXT}">a deployed model re-predicts '
            f'every image inside the same job, so no mask is left stale</text>')

    # ---------------------------------------------------------- caption
    cap_y = H - CAPTION_H + 30
    svg.raw(rounded_rect(MARGIN, cap_y - 24, W - 2 * MARGIN, CAPTION_H - 40, 10, "white",
                         stroke="#D8E0E4", sw=1.5, filter_id="cardShadow"))
    caption = (
        f"Figure 1. Crack-detection pipeline for transmission X-ray microscopy (TXM), as "
        f"deployed. Every panel is a real array from the app's own modules. A raw float32 "
        f"tile is destitched by notching the one-to-two frequency bins the periodic mosaic "
        f"grid occupies (A), then flat-fielded and percentile-stretched (B) -- that pair is "
        f"the DISPLAY view only, and both steps preserve geometry so a mask still registers "
        f"pixel-for-pixel. The model is fed the raw normalised array on purpose: "
        f"flat-fielding the model input was tried and cost 0.169 IoU, because large-radius "
        f"intensity features carry ~41% of the model's importance and flat-fielding removes "
        f"exactly those. Each pixel is described by 17 multi-scale hand-crafted features (C) "
        f"and by a {s['emb_channels']}-dimensional SAM ViT-H image embedding, computed over "
        f"{s['n_tiles']} tiles of 1024 px so that one vector covers each 16x16 block and is "
        f"read back by bilinear lookup (D); SAM contributes its frozen encoder only, and its "
        f"prompt encoder and mask decoder are never called. Two MLPs -- one on the 17 "
        f"features, one on all 273 -- are averaged (E); the average, not the hybrid alone, is "
        f"what wins on the largest mosaics. A 0.50 threshold and removal of blobs under 2000 "
        f"px give the final mask (F), while the older hysteresis cleanup is off by default "
        f"because it measurably deletes thin crack. Corrections are painted in the browser "
        f"(G) and are the ONLY thing a retrain learns from (H): the four dense reference "
        f"frames, and corrections on their specimens, are held out as the test set, which "
        f"costs nothing across specimen groups and is what makes the gate's number mean "
        f"something -- the same architecture scores 0.921 on those frames while training on "
        f"them and 0.714 held out. The gate is three recipe-aware axes, and a candidate is "
        f"only ever compared against a baseline measured the same way. Worked example: "
        f"{esc(name)}."
    )
    tspans, _ = wrap_tspans(caption, 148, MARGIN + 24, 21)
    svg.raw(f'<text x="{MARGIN+24}" y="{cap_y+6}" font-family="Georgia, serif" font-size="13.2" '
            f'fill="{INK}">{tspans}</text>')

    svg_path = os.path.join(OUT_DIR, f"full_workflow_{name}.svg")
    with open(svg_path, "w") as f:
        f.write(svg.render())
    print(f"Saved SVG: {svg_path}")
    png_path = os.path.join(PNG_DIR, "pipeline.png")
    subprocess.run(["rsvg-convert", "-w", str(int(W * 2)), "-h", str(int(H * 2)),
                    svg_path, "-o", png_path], check=True)
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
