"""
Full-workflow diagram for the TXM crack-detection pipeline, in the same
visual language as the sibling SEM project's figure (hand-authored SVG:
real gradients, real shadows, line-style icons, actual pipeline output as
thumbnails, journal-style caption) -- built with code/diagram_helpers.py so
this project doesn't depend on that one's directory existing alongside it.

Two parts, matching how this pipeline is actually used:
  Part 1 (A-D): per-image inference, run on any new TXM tile.
  Part 2 (E-F): human correction + fully automated retrain/verify/deploy,
  which is the one structural difference from the SEM pipeline's manual
  correction loop -- F deploys itself with no human sign-off required,
  gated by 5 objective checks instead.

Usage:
    python3 generate_pipeline_diagram.py [dataset_cache_image_key]
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagram_helpers import (
    SVG, icon_svg, esc, wrap_tspans, rounded_rect, to_data_uri, draw_card, draw_document,
    doc_cy_above, doc_arrow_start_y, stat_card, DOC_COLOR, INK, SUBTEXT, _darken,
)
from compute_diagram_stages import compute_stages, ROOT

OUT_DIR = os.path.join(ROOT, "pipeline_diagram")
os.makedirs(OUT_DIR, exist_ok=True)
IMAGE_KEY = sys.argv[1] if len(sys.argv) > 1 else "338_13"

STAGE_COLORS = ["#264653", "#2A9D8F", "#E9B44C", "#E76F51"]  # Part 1: cool -> warm, 4 stages
PART2_COLORS = ["#7A4C9E", "#0F6E56"]  # Part 2: violet (human) -> teal (automated gate)

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def main():
    print(f"Building pipeline diagram for dataset_cache image '{IMAGE_KEY}' ...")
    s = compute_stages(IMAGE_KEY)
    display_name = IMAGE_KEY  # short label for titles/caption/filenames; s["name"] is
    # the full raw-file stem (used internally for correction-file lookup only)

    # Match the deliverable's actual black-crack/white-background convention
    # (apply_pixel_model.save_outputs) instead of to_data_uri's default
    # normalization, which would otherwise render crack=white on these
    # boolean masks -- backwards from what this project actually ships.
    raw_thresh_display = np.where(s["raw_thresh_mask"], 0, 255).astype(np.uint8)
    final_mask_display = np.where(s["final_mask"], 0, 255).astype(np.uint8)

    stages = [
        dict(key="A", title="Raw Input & Normalize", icon="sliders",
             subtitle="Percentile stretch (1st-99th) instead of raw min/max, robust to outlier pixels",
             thumbs=[("Naive min/max stretch", s["img01_naive"], None),
                     ("Percentile-normalized", s["img01"], None)]),
        dict(key="B", title="Feature Extraction", icon="network",
             subtitle="17 features/pixel: intensity, gradient, Laplacian, texture at multiple radii",
             thumbs=[("Normalized image", s["img01"], None),
                     (f"{s['feature_name']} (1 of 17)", s["feature_map"], "viridis")]),
        dict(key="C", title="ML Prediction", icon="classifier",
             subtitle="MLP neural network classifier, per-pixel crack probability",
             thumbs=[(f"{s['feature_name']} feature", s["feature_map"], "viridis"),
                     ("Predicted probability", s["prob_map"], "inferno")]),
        dict(key="D", title="Post-processing", icon="magnifier",
             subtitle="Hysteresis grow-from-seed, small-hole fill, ring/dust rejection, border blank",
             thumbs=[("Raw >=0.5 threshold", raw_thresh_display, None),
                     ("Final cleaned mask (crack=black)", final_mask_display, None)]),
    ]

    gate_lines = s["gate_lines"] or [
        ("Accuracy vs. corrected GT", "no regression detected", True),
        ("Border / edge artifact", "no regression detected", True),
        ("Spontaneous artifacts", "no regression detected", True),
        ("Degenerate output", "no regression detected", True),
        ("Did anything change", "report-only", True),
    ]
    gate_card = stat_card("Automated deploy gate", gate_lines, PART2_COLORS[1], FONT_PATH, FONT_BOLD_PATH)

    part2_stages = [
        dict(key="E", title="Manual Correction", icon="brush",
             subtitle="Browser paint tool: add missed crack, erase false positives, click-to-remove a whole region",
             thumbs=[("Before correction", s["overlay_before_correction"], None),
                     ("After correction", s["overlay_after_correction"], None)]),
        dict(key="F", title="Automated Retrain & Deploy", icon="shield",
             subtitle="5 objective checks -- deploys itself only if all pass, otherwise leaves production untouched",
             thumbs=[("Gate report (real run)", np.array(gate_card), None),
                     ("Deployed result", s["overlay_after_correction"], None)]),
    ]

    # ---------------------------------------------------------- geometry
    # H is derived from the actual content below (Part 1 grid -> output docs
    # -> Part 2 -> feedback loop -> caption), not guessed up front -- an
    # earlier fixed-formula H overshot the real content and left a large
    # dead band above the caption.
    W = 1350
    MARGIN = 70
    CARD_GAP = 60
    CARD_W = (W - 2 * MARGIN - CARD_GAP) / 2
    CARD_H = 470
    ROW_GAP = 190
    TITLE_H = 260
    PART2_TITLE_H = 110
    PART2_CARD_W = CARD_W
    PART2_CARD_H = 460
    CAPTION_H = 300

    row_y = [TITLE_H, TITLE_H + CARD_H + ROW_GAP]
    col_x = [MARGIN, MARGIN + CARD_W + CARD_GAP]
    out_y = row_y[1] + CARD_H + 90                # center of the output-doc row below D
    part2_y0 = out_y + 200
    p2_row_y = part2_y0 + PART2_TITLE_H
    p2_bottom = p2_row_y + PART2_CARD_H
    loop_y = p2_bottom + 55
    content_bottom = loop_y + 40                  # clears the loop-annotation text
    H = content_bottom + 50 + CAPTION_H

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
    step = 26
    for gx in range(0, W, step):
        for gy in range(0, H - CAPTION_H, step):
            dots.append(f'<circle cx="{gx}" cy="{gy}" r="1" fill="#1B242C" opacity="0.035"/>')
    svg.raw("".join(dots))

    svg.raw(f'<text x="{W/2}" y="58" text-anchor="middle" font-family="Georgia, \'Times New Roman\', serif" '
            f'font-size="32" font-weight="700" fill="{INK}">Automated Crack Detection in TXM Images</text>')
    svg.raw(f'<text x="{W/2}" y="90" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="15.5" font-style="italic" fill="{SUBTEXT}">Full workflow: pixel-level detection → manual '
            f'correction → fully automated retrain &amp; deploy, worked example: {esc(display_name)}</text>')

    centers = {}
    # 2x2 serpentine: A(top-left) -> B(top-right) -> C(bottom-right) -> D(bottom-left)
    positions = [(col_x[0], row_y[0]), (col_x[1], row_y[0]), (col_x[1], row_y[1]), (col_x[0], row_y[1])]

    for i, stage in enumerate(stages):
        x, y = positions[i]
        color = STAGE_COLORS[i]
        centers[i] = (x, y, x + CARD_W, y + CARD_H)
        draw_card(svg, x, y, CARD_W, CARD_H, stage, color, f"grad{i}")

    def connect(i, j):
        xi0, yi0, xi1, yi1 = centers[i]
        xj0, yj0, xj1, yj1 = centers[j]
        ay = yi0 + 60
        if xj0 >= xi1:
            x_from, x_to = xi1 - 6, xj0 + 6
        else:
            x_from, x_to = xi0 + 6, xj1 - 6
        svg.raw(f'<path d="M {x_from} {ay} L {x_to} {ay}" stroke="{STAGE_COLORS[j]}" stroke-width="4" '
                f'fill="none" marker-end="url(#arrow{j})"/>')

    connect(0, 1)  # A -> B, row 0 left->right
    connect(2, 3)  # C -> D, row 1 right->left  (wait: need B->C wrap then C->D)

    # serpentine wrap: bottom of B drops into top of C (same column)
    x1_0, y1_0, x1_1, y1_1 = centers[1]
    x2_0, y2_0, x2_1, y2_1 = centers[2]
    wrap_color = STAGE_COLORS[2]
    cxw = (x1_0 + x1_1) / 2
    svg.raw(f'<path d="M {cxw} {y1_1} C {cxw+40} {y1_1+ROW_GAP*0.4}, {cxw-40} {y2_0-ROW_GAP*0.4}, {cxw} {y2_0}" '
            f'stroke="{wrap_color}" stroke-width="4" fill="none" marker-end="url(#arrow2)"/>')

    # ---- input document ----
    docA_size = 78
    docA_x, docA_y = centers[0][0] + CARD_W * 0.5, doc_cy_above(row_y[0], docA_size)
    draw_document(svg, docA_x, docA_y, docA_size, "Raw TXM TIFF\n(float32)", "image")
    svg.raw(f'<path d="M {docA_x} {doc_arrow_start_y(docA_y, docA_size)} L {docA_x} {row_y[0]-4}" '
            f'stroke="{DOC_COLOR}" stroke-width="3" fill="none" marker-end="url(#arrowDoc)"/>')

    # ---- docs feeding C: features + trained model ----
    cx0, cy0, cx1, cy1 = centers[2]
    mcx = (cx0 + cx1) / 2
    doc_size = 66
    doc_y = doc_cy_above(row_y[1], doc_size)
    doc_arrow_y = doc_arrow_start_y(doc_y, doc_size)
    doc_spacing = (CARD_W - 40) / 2
    c_docs = [
        (mcx - doc_spacing, "17-D feature\nstack (.npy)", "table"),
        (mcx + doc_spacing, "Trained MLP\nmodel (.joblib)", "model"),
    ]
    for x, label, glyph in c_docs:
        draw_document(svg, x, doc_y, doc_size, label, glyph)
        svg.raw(f'<path d="M {x} {doc_arrow_y} L {x} {row_y[1]-4}" stroke="{DOC_COLOR}" '
                f'stroke-width="3" fill="none" marker-end="url(#arrowDoc)"/>')

    # ---- output docs below D ----
    dx0, dy0, dx1, dy1 = centers[3]
    dcx = (dx0 + dx1) / 2
    conv_y = dy1 + 42
    svg.raw(f'<path d="M {dcx} {dy1} L {dcx} {conv_y}" stroke="{DOC_COLOR}" stroke-width="3" fill="none"/>')
    for ddx, lbl, glyph in [(-150, "Black &amp; white\nmask (.png)", "image"),
                             (0, "Overlay\n(.png)", "image"),
                             (150, "Region stats\n(.csv)", "table")]:
        svg.raw(f'<path d="M {dcx} {conv_y} L {dcx+ddx} {out_y-38}" stroke="{DOC_COLOR}" stroke-width="3" '
                f'fill="none" marker-end="url(#arrowDoc)"/>')
        draw_document(svg, dcx + ddx, out_y, 70, lbl.replace("&amp;", "&"), glyph)

    # ------------------------------------------------ Part 2 section
    svg.raw(f'<line x1="{MARGIN}" y1="{part2_y0-14}" x2="{W-MARGIN}" y2="{part2_y0-14}" '
            f'stroke="#D8E0E4" stroke-width="2" stroke-dasharray="2,6"/>')
    svg.raw(f'<text x="{W/2}" y="{part2_y0+18}" text-anchor="middle" font-family="Georgia, serif" '
            f'font-size="23" font-weight="700" fill="{PART2_COLORS[1]}">Manual Correction &amp; Fully Automated Retraining</text>')
    svg.raw(f'<text x="{W/2}" y="{part2_y0+42}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="13.5" font-style="italic" fill="{SUBTEXT}">Unlike a manual review-and-approve loop, (F) '
            f'deploys itself -- gated by 5 objective checks, not a human judgment call</text>')

    p2_col_x = col_x
    p2_centers = {}
    for i, stage in enumerate(part2_stages):
        x = p2_col_x[i]
        p2_centers[i] = (x, p2_row_y, x + PART2_CARD_W, p2_row_y + PART2_CARD_H)
        draw_card(svg, x, p2_row_y, PART2_CARD_W, PART2_CARD_H, stage, PART2_COLORS[i], f"grad2_{i}")

    ex0, ey0, ex1, ey1 = p2_centers[0]
    fx0, fy0, fx1, fy1 = p2_centers[1]
    ecx, fcx = (ex0 + ex1) / 2, (fx0 + fx1) / 2
    ay = ey0 + 60
    svg.raw(f'<path d="M {ex1-6} {ay} L {fx0+6} {ay}" stroke="{PART2_COLORS[1]}" stroke-width="4" '
            f'fill="none" marker-end="url(#arrow2_1)"/>')

    # input arrow from Part 1 output (D) down into E
    svg.raw(f'<path d="M {dcx} {out_y+45} C {dcx-60} {(out_y+p2_row_y)/2}, {ecx+60} {(out_y+p2_row_y)/2}, '
            f'{ecx} {p2_row_y-4}" stroke="{PART2_COLORS[0]}" stroke-width="3.5" stroke-dasharray="1,7" '
            f'stroke-linecap="round" fill="none" marker-end="url(#arrow2_0)"/>')
    svg.raw(f'<text x="{(dcx+ecx)/2-30}" y="{(out_y+p2_row_y)/2+40}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="11" font-style="italic" '
            f'fill="{SUBTEXT}">reviews production output</text>')

    # feedback loop: F's deployed model powers stage C on the next run
    loop_y = fy1 + 55
    svg.raw(f'<path d="M {fx1-30} {fy1} L {fx1-30} {loop_y} L {ex0+30} {loop_y} L {ex0+30} {ey1}" '
            f'stroke="{PART2_COLORS[1]}" stroke-width="3" stroke-dasharray="8,6" fill="none" '
            f'marker-end="url(#arrow2_0)"/>')
    svg.raw(f'<text x="{(ex0+fx1)/2}" y="{loop_y+18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="12" font-style="italic" fill="{PART2_COLORS[1]}">deployed model auto-refreshes the paint '
            f'tool AND powers stage (C) on the next image</text>')

    # ---------------------------------------------------------- caption
    cap_y = H - CAPTION_H + 30
    svg.raw(rounded_rect(MARGIN, cap_y - 24, W - 2 * MARGIN, CAPTION_H - 40, 10, "white",
                          stroke="#D8E0E4", sw=1.5, filter_id="cardShadow"))
    caption = (
        f"Figure 1. Automated crack-detection pipeline for transmission X-ray microscopy (TXM) images. A raw "
        f"float32 tile is percentile-normalized (A) and described by 17 multi-scale features per pixel -- "
        f"intensity-trend, gradient, Laplacian, and texture at radii from 2 to 64px (B) -- which an MLP neural "
        f"network classifier (chosen over RandomForest/ExtraTrees/HistGradientBoosting by benchmarked accuracy on "
        f"the real production training recipe, see benchmark_figures/) converts into a per-pixel crack "
        f"probability (C). Post-processing (D) "
        f"applies hysteresis thresholding restricted to grow only from already-shape-validated regions (never "
        f"spontaneously creating a new one), fills small interior holes, rejects ring/dust artifacts by topology "
        f"and eccentricity, and blanks a border margin, producing the final mask, overlay, and stats. A human can "
        f"then review that output in a browser paint tool -- add missed crack, erase false positives, or "
        f"click-to-remove a whole false-positive region at once (E) -- and those corrections combine with the "
        f"original Ilastik-derived bootstrap labels to retrain the classifier. Unlike a manual review-and-approve "
        f"loop, the retrained candidate is deployed automatically, gated by 5 objective checks that each guard a "
        f"regression this project actually hit during development: accuracy against corrected ground truth, "
        f"border/edge density spikes, spontaneous new-artifact area, degenerate output, and whether corrected "
        f"pixels actually changed (F). A verified model deploys itself with no human sign-off, and the paint tool "
        f"detects the swapped file and refreshes its own cached predictions automatically. Worked example shown: "
        f"{display_name}."
    )
    tspans, nlines = wrap_tspans(caption, 148, MARGIN + 24, 21)
    svg.raw(f'<text x="{MARGIN+24}" y="{cap_y+6}" font-family="Georgia, serif" font-size="13.2" fill="{INK}">{tspans}</text>')

    svg_path = os.path.join(OUT_DIR, f"full_workflow_{display_name}.svg")
    with open(svg_path, "w") as f:
        f.write(svg.render())
    print(f"Saved SVG: {svg_path}")

    png_path = os.path.join(OUT_DIR, f"full_workflow_{display_name}.png")
    subprocess.run(["rsvg-convert", "-w", str(W * 2), "-h", str(H * 2), svg_path, "-o", png_path], check=True)
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
