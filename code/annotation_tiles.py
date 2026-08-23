"""Load exported tiles for annotation, then fold the finished masks into the ground truth.

    python3 code/annotation_tiles.py load            # tiles -> the app, WITHOUT predicting
    python3 code/annotation_tiles.py status          # how densely each tile is annotated

STEP 1, `load`. Puts every tile from code/export_annotation_tiles.py into the app as a
normal image so it can be painted with the tools already learned -- but ingested with
`predict=False`, so there is no model mask on the canvas. That is the point: a labeller shown
the model's output is being asked to agree with it, and 98.3% of this project's existing
crack labels are confirmations of exactly that. Dense ground truth has to be drawn without
it, or it cannot be used to evaluate the thing that drew it.

STEP 2, paint. Add crack covers crack; Erase covers not-crack. Every pixel needs one or the
other -- that is what makes it *dense*, and dense is what makes IoU computable, because IoU
needs the false negatives. `status` reports the fraction still untouched per tile.

NO STEP 3. `import` used to copy finished tiles into dataset_cache as ground-truth arrays for
the retrain gate to score against. Nothing is scored against a separate labelled set any
more, so a painted tile feeds training exactly the way every other image does -- through the
corrections drawn on it -- and there is nothing left to import.

"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import store as S            # noqa: E402
import pipeline as P         # noqa: E402

TILES = os.path.join(PROJECT, "paint", "annotate")
MANIFEST = os.path.join(TILES, "manifest.json")
PREFIX = "ANNOT_"


def manifest():
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"no {os.path.relpath(MANIFEST, PROJECT)} -- run "
                         f"python3 code/export_annotation_tiles.py --group AM/HC first")
    with open(MANIFEST) as f:
        return json.load(f)


def loaded():
    """Tile name -> image meta, for tiles already in the app."""
    out = {}
    for m in S.list_images():
        fn = m.get("filename") or ""
        if fn.startswith(PREFIX):
            out[fn[len(PREFIX):]] = m
    return out


def cmd_load(a):
    man = manifest()
    have = loaded()
    n_new = 0
    for t in man["tiles"]:
        if t["tile"] in have:
            continue
        path = os.path.join(TILES, t["tile"])
        if not os.path.exists(path):
            print(f"  missing {t['tile']}")
            continue
        with open(path, "rb") as f:
            iid, _ = S.save_upload(PREFIX + t["tile"], f.read())
        P.ingest(iid, predict=False)
        n_new += 1
        print(f"  loaded {t['tile']}  ({t['group']}, from {t['filename'][22:50]})")
    print(f"\n{n_new} tile(s) loaded, {len(have)} already present.")
    print("Open the app: they appear in the sidebar as ANNOT_tile_NNN.png with NO red")
    print("overlay, which is deliberate. Paint every pixel -- Add crack for crack, Erase")
    print("for not-crack -- then run:  python3 code/annotation_tiles.py status")
    return 0


def _judged(iid):
    corr = S.load_npy(iid, "correction.npy", mmap=True)
    if corr is None:
        return 0.0, 0.0
    c = np.asarray(corr)
    return float((c != 0).mean()), float((c == 1).mean())


def cmd_status(a):
    man = manifest()
    have = loaded()
    print(f"{'tile':<16} {'group':<8} {'judged':>8} {'crack':>7}  state")
    done = 0
    for t in man["tiles"]:
        m = have.get(t["tile"])
        if m is None:
            print(f"{t['tile']:<16} {t['group']:<8} {'—':>8} {'—':>7}  not loaded")
            continue
        cov, cr = _judged(m["id"])
        state = "READY" if cov >= a.min_covered else ("untouched" if cov < 0.01
                                                      else "partial")
        done += state == "READY"
        print(f"{t['tile']:<16} {t['group']:<8} {cov*100:>7.1f}% {cr*100:>6.2f}%  {state}")
    print(f"\n{done}/{len(man['tiles'])} tile(s) dense enough to import "
          f"(>= {a.min_covered*100:.0f}% of pixels judged)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("load", "status"):
        p = sub.add_parser(name)
        p.add_argument("--min-covered", type=float, default=0.95,
                       help="fraction of pixels that must be judged for a tile to count")
    a = ap.parse_args()
    return {"load": cmd_load, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
