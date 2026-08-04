"""
Generic hand-authored SVG drawing primitives for pipeline figures: real
gradients, real blurred drop shadows, line-style icons, "document" glyphs,
and card/thumbnail layout helpers. Ported from the sibling SEM project's
generate_scientific_diagram.py (../../CBS_Crack_Detection_Pipeline/code/)
so this project can build its own pipeline diagram without depending on
that project's directory existing alongside this one -- this repo is
self-contained.

Built directly in SVG rather than matplotlib so coordinates are natively
isotropic (no aspect-ratio correction hacks) and shadows/gradients are real
rather than faked with stacked patches. Requires the `rsvg-convert` CLI to
rasterize the final PNG; the SVG itself is also kept as a deliverable since
it's fully vector.
"""
import base64
import io
import textwrap

import numpy as np
from PIL import Image

try:
    from matplotlib import colormaps as _colormaps
    def get_cmap(name):
        return _colormaps[name]
except ImportError:
    import matplotlib.cm as cm
    def get_cmap(name):
        return cm.get_cmap(name)

DOC_COLOR = "#A9812F"
INK = "#1B242C"
SUBTEXT = "#5B6B76"


def to_data_uri(arr, cmap=None):
    """numpy array -> base64 PNG data URI, ready to drop straight into an <image> tag."""
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        a = (a - a.min()) / (np.ptp(a) + 1e-9)
        if cmap:
            arr = (get_cmap(cmap)(a)[..., :3] * 255).astype(np.uint8)
        else:
            arr = (a * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}", img.size


def _darken(hexcolor, factor):
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


# ------------------------------------------------------------------ icons
def icon_svg(kind, color, size=1.0):
    """Icons are authored in a -1..1 box and scaled/translated by the caller
    via a <g transform>, so SVG's native isotropic coordinates keep every
    shape a true circle/square -- no aspect-ratio correction needed."""
    sw = 0.16 * size
    if kind == "sliders":
        parts = []
        for x, ky in zip([-0.55, 0, 0.55], [0.25, -0.35, 0.4]):
            parts.append(f'<line x1="{x}" y1="-0.7" x2="{x}" y2="0.7" stroke="{color}" '
                         f'stroke-width="{sw}" stroke-linecap="round"/>')
            parts.append(f'<circle cx="{x}" cy="{ky}" r="0.16" fill="white" stroke="{color}" stroke-width="{sw}"/>')
        return "".join(parts)
    if kind == "sun":
        parts = [f'<circle cx="0" cy="0" r="0.42" fill="none" stroke="{color}" stroke-width="{sw}"/>']
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x0, y0 = np.cos(ang) * 0.6, np.sin(ang) * 0.6
            x1, y1 = np.cos(ang) * 0.95, np.sin(ang) * 0.95
            parts.append(f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" '
                         f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>')
        return "".join(parts)
    if kind == "magnifier":
        return (f'<circle cx="-0.15" cy="-0.15" r="0.45" fill="none" stroke="{color}" stroke-width="{sw}"/>'
                f'<line x1="0.18" y1="0.18" x2="0.6" y2="0.6" stroke="{color}" '
                f'stroke-width="{sw * 1.3}" stroke-linecap="round"/>')
    if kind == "network":
        pts = [(-0.6, 0.55), (0.6, 0.6), (-0.4, -0.5), (0.55, -0.45), (0, 0.05)]
        parts = []
        for i, (x0, y0) in enumerate(pts):
            for (x1, y1) in pts[i + 1:]:
                parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" '
                             f'stroke-width="{sw * 0.4}" opacity="0.55"/>')
        for (x0, y0) in pts:
            parts.append(f'<circle cx="{x0}" cy="{y0}" r="0.13" fill="{color}"/>')
        return "".join(parts)
    if kind == "classifier":
        left = [(-0.55, 0.6), (-0.55, 0), (-0.55, -0.6)]
        right = [(0.55, 0.35), (0.55, -0.35)]
        parts = []
        for (x0, y0) in left:
            for (x1, y1) in right:
                parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" '
                             f'stroke-width="{sw * 0.4}" opacity="0.6"/>')
        for (x0, y0) in left:
            parts.append(f'<circle cx="{x0}" cy="{y0}" r="0.15" fill="white" stroke="{color}" stroke-width="{sw}"/>')
        for (x0, y0) in right:
            parts.append(f'<circle cx="{x0}" cy="{y0}" r="0.17" fill="{color}"/>')
        return "".join(parts)
    if kind == "grid":
        parts = []
        for gx in (-0.62, 0, 0.62):
            for gy in (-0.62, 0, 0.62):
                parts.append(f'<rect x="{gx-0.24}" y="{gy-0.24}" width="0.48" height="0.48" '
                             f'fill="none" stroke="{color}" stroke-width="{sw}"/>')
        return "".join(parts)
    if kind == "check":
        return (f'<circle cx="0" cy="0" r="0.85" fill="none" stroke="{color}" stroke-width="{sw}"/>'
                f'<polyline points="-0.38,0 -0.05,0.35 0.5,-0.35" fill="none" stroke="{color}" '
                f'stroke-width="{sw * 1.3}" stroke-linecap="round" stroke-linejoin="round"/>')
    if kind == "brush":
        return (f'<path d="M -0.55 0.6 L -0.15 0.15 L 0.15 0.45 L -0.25 0.85 Z" fill="{color}"/>'
                f'<rect x="-0.05" y="-0.65" width="0.75" height="0.32" rx="0.08" '
                f'transform="rotate(45 -0.05 -0.65)" fill="none" stroke="{color}" stroke-width="{sw}"/>'
                f'<circle cx="0.55" cy="-0.55" r="0.1" fill="{color}"/>')
    if kind == "shield":
        return (f'<path d="M 0 -0.85 L 0.62 -0.55 L 0.62 0.15 Q 0.62 0.65 0 0.9 '
                f'Q -0.62 0.65 -0.62 0.15 L -0.62 -0.55 Z" fill="none" stroke="{color}" stroke-width="{sw}"/>'
                f'<polyline points="-0.28,0.02 -0.05,0.3 0.35,-0.3" fill="none" stroke="{color}" '
                f'stroke-width="{sw * 1.2}" stroke-linecap="round" stroke-linejoin="round"/>')
    if kind == "loop":
        parts = [f'<path d="M -0.6 -0.15 A 0.65 0.65 0 1 1 -0.6 0.2" fill="none" '
                 f'stroke="{color}" stroke-width="{sw*1.2}" stroke-linecap="round"/>']
        parts.append(f'<polygon points="-0.6,0.55 -0.85,0.1 -0.35,0.15" fill="{color}"/>')
        return "".join(parts)
    return ""


def doc_glyph_svg(kind, color):
    if kind == "image":
        return (f'<rect x="-0.5" y="-0.42" width="1" height="0.84" fill="none" stroke="{color}" stroke-width="0.06"/>'
                f'<circle cx="-0.22" cy="-0.14" r="0.09" fill="{color}"/>'
                f'<polygon points="-0.5,0.05 -0.05,-0.28 0.15,-0.05 0.5,-0.3 0.5,0.42 -0.5,0.42" fill="{color}" opacity="0.35"/>')
    if kind == "table":
        parts = []
        for x in (-0.45, 0, 0.45):
            parts.append(f'<line x1="{x}" y1="-0.45" x2="{x}" y2="0.45" stroke="{color}" stroke-width="0.055"/>')
        for y in (-0.45, 0, 0.45):
            parts.append(f'<line x1="-0.45" y1="{y}" x2="0.45" y2="{y}" stroke="{color}" stroke-width="0.055"/>')
        return "".join(parts)
    if kind == "model":
        parts = [f'<rect x="-0.32" y="-0.32" width="0.64" height="0.64" fill="none" stroke="{color}" stroke-width="0.07"/>']
        for dx, dy in [(-1, 0.5), (-1, -0.5), (1, 0.5), (1, -0.5), (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1)]:
            x0, y0 = dx * 0.32, dy * 0.32
            x1 = x0 + (0.18 if abs(dx) > abs(dy) else 0) * np.sign(dx)
            y1 = y0 + (0.18 if abs(dy) >= abs(dx) else 0) * np.sign(dy)
            parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="0.06"/>')
        return "".join(parts)
    return ""


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap_tspans(text, width, x, line_h, first_dy=0):
    lines = textwrap.wrap(text, width=width)
    out = []
    for i, ln in enumerate(lines):
        dy = first_dy if i == 0 else line_h
        out.append(f'<tspan x="{x}" dy="{dy}">{esc(ln)}</tspan>')
    return "".join(out), len(lines)


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.defs = []
        self.body = []

    def add_gradient(self, gid, c0, c1, angle="vertical"):
        x2, y2 = ("0%", "100%") if angle == "vertical" else ("100%", "0%")
        self.defs.append(
            f'<linearGradient id="{gid}" x1="0%" y1="0%" x2="{x2}" y2="{y2}">'
            f'<stop offset="0%" stop-color="{c0}"/><stop offset="100%" stop-color="{c1}"/>'
            f'</linearGradient>')

    def add_shadow_filter(self, fid, dx=0, dy=6, blur=10, opacity=0.22):
        self.defs.append(f'''
        <filter id="{fid}" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="{dx}" dy="{dy}" stdDeviation="{blur}" flood-color="#1B242C" flood-opacity="{opacity}"/>
        </filter>''')

    def add_arrowhead(self, mid, color):
        self.defs.append(f'''
        <marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/>
        </marker>''')

    def raw(self, s):
        self.body.append(s)

    def render(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">'
                f'<defs>{"".join(self.defs)}</defs>{"".join(self.body)}</svg>')


def rounded_rect(x, y, w, h, r, fill, stroke=None, sw=0, filter_id=None, opacity=1.0):
    f = f'filter="url(#{filter_id})"' if filter_id else ""
    s = f'stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" fill="{fill}" {s} {f} opacity="{opacity}"/>'


def draw_card(svg, x, y, w, h, stage, color, grad_id):
    svg.raw(rounded_rect(x, y, w, h, 16, "white", stroke="#E1E8EB", sw=1.5, filter_id="cardShadow"))
    header_h = 108
    svg.raw(f'<path d="M {x} {y+16} Q {x} {y} {x+16} {y} L {x+w-16} {y} Q {x+w} {y} {x+w} {y+16} '
            f'L {x+w} {y+header_h} L {x} {y+header_h} Z" fill="url(#{grad_id})"/>')

    badge_r = 34
    bcx, bcy = x + 58, y + header_h / 2
    svg.raw(f'<circle cx="{bcx}" cy="{bcy}" r="{badge_r}" fill="white" filter="url(#badgeShadow)"/>')
    svg.raw(f'<g transform="translate({bcx},{bcy}) scale({badge_r*0.62})">{icon_svg(stage["icon"], color)}</g>')

    svg.raw(f'<text x="{x+104}" y="{bcy-6}" font-family="Helvetica, Arial, sans-serif" font-size="19" '
            f'font-weight="700" fill="white">({stage["key"]}) {esc(stage["title"])}</text>')
    sub_tspans, _ = wrap_tspans(stage["subtitle"], 38, x + 104, 16)
    svg.raw(f'<text x="{x+104}" y="{bcy+14}" font-family="Helvetica, Arial, sans-serif" font-size="12.0" '
            f'fill="#EAF3F1">{sub_tspans}</text>')

    thumb_y = y + header_h + 18
    thumb_h = h - header_h - 18 - 34
    gap = 12
    thumb_w = (w - 2 * 18 - gap) / 2
    for j, (cap, arr, cmap) in enumerate(stage["thumbs"]):
        tx = x + 18 + j * (thumb_w + gap)
        uri, (pw, ph) = to_data_uri(arr, cmap=cmap)
        ar = ph / pw
        draw_w, draw_h = thumb_w, thumb_w * ar
        if draw_h > thumb_h:
            draw_h = thumb_h
            draw_w = thumb_h / ar
        ox = tx + (thumb_w - draw_w) / 2
        oy = thumb_y + (thumb_h - draw_h) / 2
        svg.raw(f'<g filter="url(#thumbShadow)"><rect x="{ox-3}" y="{oy-3}" width="{draw_w+6}" height="{draw_h+6}" '
                f'fill="white"/></g>')
        svg.raw(f'<image x="{ox}" y="{oy}" width="{draw_w}" height="{draw_h}" href="{uri}" '
                f'style="image-rendering:auto"/>')
        svg.raw(f'<rect x="{ox}" y="{oy}" width="{draw_w}" height="{draw_h}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        svg.raw(f'<text x="{tx+thumb_w/2}" y="{thumb_y+thumb_h+20}" text-anchor="middle" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{SUBTEXT}">{esc(cap)}</text>')


# A document's 2-line label sits below it; an outgoing arrow must clear that
# text, not just the doc's own bottom edge. These helpers keep every
# doc-to-box placement/arrow consistent with the actual label height.
DOC_LABEL_GAP = 16
DOC_LINE_H = 14
DOC_LABEL_PAD = 10
DOC_ARROW_MIN = 22


def doc_label_block_h(n_lines=2):
    return DOC_LABEL_GAP + n_lines * DOC_LINE_H + DOC_LABEL_PAD


def doc_cy_above(row_top, doc_size, n_lines=2):
    """Center-y for a document so its label clears row_top with room for an arrow."""
    return row_top - DOC_ARROW_MIN - doc_label_block_h(n_lines) - doc_size / 2


def doc_arrow_start_y(doc_cy, doc_size, n_lines=2):
    """Where an outgoing arrow below this doc may safely start (below its label)."""
    return doc_cy + doc_size / 2 + doc_label_block_h(n_lines)


def draw_document(svg, cx, cy, size, label, glyph):
    w = h = size
    x0, y0 = cx - w / 2, cy - h / 2
    fold = 0.16 * w
    svg.raw(f'<g filter="url(#thumbShadow)">'
            f'<polygon points="{x0},{y0} {x0+w-fold},{y0} {x0+w},{y0+fold} {x0+w},{y0+h} {x0},{y0+h}" '
            f'fill="#FFF8E4" stroke="{DOC_COLOR}" stroke-width="1.6"/></g>')
    svg.raw(f'<polygon points="{x0+w-fold},{y0} {x0+w-fold},{y0+fold} {x0+w},{y0+fold}" '
            f'fill="#F0E0AE" stroke="{DOC_COLOR}" stroke-width="1"/>')
    svg.raw(f'<g transform="translate({cx},{cy-h*0.08}) scale({w*0.38})">{doc_glyph_svg(glyph, DOC_COLOR)}</g>')
    lines = label.split("\n")
    tspans = "".join(f'<tspan x="{cx}" dy="{0 if i==0 else 14}">{esc(l)}</tspan>' for i, l in enumerate(lines))
    svg.raw(f'<text x="{cx}" y="{y0+h+16}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="11.5" fill="{SUBTEXT}">{tspans}</text>')


def stat_card(title, lines, accent, font_path=None, font_bold_path=None):
    """A small 'figure' summarizing a handful of stat lines as plain text --
    kept as a raster image so it slots into draw_card's image-thumbnail
    contract unchanged."""
    from PIL import ImageDraw, ImageFont
    W, H = 520, 430
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    def _font(size, bold=False):
        path = (font_bold_path if bold else font_path) or "/System/Library/Fonts/Supplemental/Arial.ttf"
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return ImageFont.load_default()

    draw.rectangle([0, 0, W - 1, H - 1], outline="#E1E8EB", width=2)
    draw.rectangle([0, 0, W - 1, 64], fill=accent)
    draw.text((22, 18), title, fill="white", font=_font(22, bold=True))
    y = 90
    for line, sub, ok in lines:
        color = "#1E7A3C" if ok is True else ("#B23A2E" if ok is False else INK)
        # Draw the check/cross as actual line strokes rather than a unicode
        # glyph -- Arial.ttf has no U+2713/2717 glyph and silently renders a
        # tofu box instead, which was invisible-looking in the diagram.
        mx, my = 30, y + 12
        if ok is True:
            draw.line([(mx - 8, my), (mx - 2, my + 7), (mx + 10, my - 8)], fill=color, width=4, joint="curve")
        elif ok is False:
            draw.line([(mx - 7, my - 7), (mx + 7, my + 7)], fill=color, width=4)
            draw.line([(mx - 7, my + 7), (mx + 7, my - 7)], fill=color, width=4)
        draw.text((48, y), line, fill=color, font=_font(21, bold=True))
        draw.text((48, y + 30), sub, fill=SUBTEXT, font=_font(13))
        y += 62
    return img
