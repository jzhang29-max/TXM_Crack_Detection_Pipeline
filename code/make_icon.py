"""Generate the app icon: app/static/icon.svg plus PNG fallbacks.

    python3 code/make_icon.py            # write the icon files
    python3 code/make_icon.py --preview  # also write a magnified sheet to eyeball

The icon has to say what the app is at 16x16 in a browser tab, which rules out anything
with detail. What survives at that size is a silhouette and at most three colours, so the
design is the three things the app actually does, and nothing else:

    a light grey field   the specimen
    a dark jagged line   the crack
    a red band under it  the model's overlay

Drawn from one shared polyline so the crack and its overlay cannot drift apart, and the
overlay is deliberately WIDER than the crack, because that is the model's real behaviour --
it runs wide of a crack rather than inventing one elsewhere.

The SVG is the icon browsers use. The PNGs exist for Safari's touch icon and for anything
that still refuses an SVG favicon. All of them are generated from the same coordinates here
rather than drawn by hand twice.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(PROJECT, "app", "static")

# One crack, in a 64x64 box. Every rendering below uses exactly these points.
#
# The first attempt was a single clean diagonal with one perpendicular branch, and at 64 px
# it read unmistakably as a SWORD -- the branch became a crossguard and the red halo became
# a glowing blade. What makes a line read as a crack instead is irregularity: it has to
# meander, with short back-steps, and branch at ACUTE angles rather than square ones. Four
# geometries were rendered at 16/32/64 px and compared before this one was picked.
CRACK = [(7, 55), (17, 46), (23, 47), (30, 37), (36, 38), (45, 26), (50, 27), (57, 9)]
BRANCHES = [[(30, 37), (25, 30)], [(45, 26), (50, 33)]]
# Darker than the app's own specimen grey on purpose: a browser tab is often white, and at
# #D7D9D7 the tile edge dissolved into it. Checked at true 16 px on both white and dark.
FIELD = "#C6C9C4"      # specimen: the flat-fielded grey, darkened for tab contrast
CRACK_C = "#191B1F"    # crack: near-black, as a real crack reads after flat-fielding
OVER = "#E8574C"       # the red overlay, same hue as the mask in the app
BOX = 64


def svg():
    pts = " ".join(f"{x} {y}" for x, y in CRACK)
    brs = "\n    ".join(
        f'<polyline points="{" ".join(f"{x} {y}" for x, y in b)}" '
        f'stroke="{CRACK_C}" stroke-width="3"/>' for b in BRANCHES)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {BOX} {BOX}"
     width="{BOX}" height="{BOX}" role="img" aria-label="TXM Crack Detection">
  <title>TXM Crack Detection</title>
  <rect x="2" y="2" width="{BOX-4}" height="{BOX-4}" rx="13" fill="{FIELD}"/>
  <g fill="none" stroke-linecap="round" stroke-linejoin="round">
    <!-- the model's overlay: wider than the crack, on purpose -->
    <polyline points="{pts}" stroke="{OVER}" stroke-width="12" opacity="0.48"/>
    <!-- the crack, and two branches at acute angles -->
    <polyline points="{pts}" stroke="{CRACK_C}" stroke-width="5"/>
    {brs}
  </g>
</svg>
'''


def png(size, ss=8):
    """Raster at `size`, supersampled `ss`x then reduced, so edges are antialiased."""
    from PIL import Image, ImageDraw
    S = size * ss
    k = S / BOX
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([2 * k, 2 * k, (BOX - 2) * k, (BOX - 2) * k],
                        radius=13 * k, fill=FIELD)
    over = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    def stroke(dr, pts, col, wid):
        dr.line([(x * k, y * k) for x, y in pts], fill=col,
                width=max(1, int(wid * k)), joint="curve")
        r = wid * k / 2                                  # emulate SVG round caps and joins
        for x, y in pts:
            dr.ellipse([x * k - r, y * k - r, x * k + r, y * k + r], fill=col)

    stroke(od, CRACK, OVER, 12)
    over.putalpha(over.getchannel("A").point(lambda a: int(a * 0.48)))
    im.alpha_composite(over)
    stroke(d, CRACK, CRACK_C, 5)
    for b in BRANCHES:
        stroke(d, b, CRACK_C, 3)
    # clip to the rounded field so nothing bleeds outside the tile
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([2 * k, 2 * k, (BOX - 2) * k, (BOX - 2) * k],
                                           radius=13 * k, fill=255)
    im.putalpha(mask)
    return im.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true",
                    help="also write a magnified sheet, to check it reads at 16 px")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    p = os.path.join(OUT, "icon.svg")
    with open(p, "w") as f:
        f.write(svg())
    print(f"  wrote {os.path.relpath(p, PROJECT)}  {os.path.getsize(p)} bytes")

    for size, name in ((32, "icon-32.png"), (180, "apple-touch-icon.png")):
        q = os.path.join(OUT, name)
        png(size).save(q, optimize=True)
        print(f"  wrote {os.path.relpath(q, PROJECT)}  {size}x{size}  "
              f"{os.path.getsize(q)/1e3:.1f} kB")

    if a.preview:
        from PIL import Image
        sizes = [16, 24, 32, 64]
        pad, scale = 12, 6
        w = sum(s * scale + pad for s in sizes) + pad
        h = max(sizes) * scale + pad * 2
        sheet = Image.new("RGB", (w, h), (28, 30, 34))
        x = pad
        for s in sizes:
            tile = png(s).resize((s * scale, s * scale), Image.NEAREST)   # show real pixels
            sheet.paste(tile, (x, (h - s * scale) // 2), tile)
            x += s * scale + pad
        sp = os.environ.get("SP", "/tmp")
        out = os.path.join(sp, "icon_preview.png")
        sheet.save(out)
        print(f"\n  preview (16/24/32/64 px, magnified {scale}x, nearest-neighbour so you "
              f"see the actual pixels): {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
