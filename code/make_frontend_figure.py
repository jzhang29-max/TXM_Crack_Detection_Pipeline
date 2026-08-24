"""Screenshot the running app for the README, driven by its own deep link.

Generated, not pasted, for the same reason as the other figures: a screenshot cropped by hand
documents whatever happened to be on screen that day. The app takes ?image=<substring> and
&labels=0, so the exact frame and the exact layer state are part of the command.

    ./run_app.sh                       # in one terminal
    python3 code/make_frontend_figure.py
"""
import argparse
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "docs", "img", "app.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8800")
    ap.add_argument("--image", default="HC_316L_fatigue_1200_cycles",
                    help="substring of the filename to select")
    ap.add_argument("--width", type=int, default=1680)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright
    url = f"{args.url}/?image={args.image}&labels=0"
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": args.width, "height": args.height},
                        device_scale_factor=2)
        pg.goto(url, wait_until="networkidle", timeout=120000)
        # the mask is a second <img> fetched after the display image; wait for both to decode
        pg.wait_for_function(
            "() => [...document.querySelectorAll('img')].filter(i => i.naturalWidth > 400)"
            ".length >= 2", timeout=120000)
        pg.wait_for_timeout(2500)
        got = pg.evaluate("() => document.body.innerText.split('\\n').slice(-2)")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pg.screenshot(path=args.out)
        b.close()
    print(f"  showing: {got}")
    print(f"  wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
