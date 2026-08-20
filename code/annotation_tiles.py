"""Load exported tiles for annotation, then fold the finished masks into the ground truth.

    python3 code/annotation_tiles.py load            # tiles -> the app, WITHOUT predicting
    python3 code/annotation_tiles.py status          # how densely each tile is annotated
    python3 code/annotation_tiles.py import          # finished tiles -> dataset_cache

STEP 1, `load`. Puts every tile from code/export_annotation_tiles.py into the app as a
normal image so it can be painted with the tools already learned -- but ingested with
`predict=False`, so there is no model mask on the canvas. That is the point: a labeller shown
the model's output is being asked to agree with it, and 98.3% of this project's existing
crack labels are confirmations of exactly that. Dense ground truth has to be drawn without
it, or it cannot be used to evaluate the thing that drew it.

STEP 2, paint. Add crack covers crack; Erase covers not-crack. Every pixel needs one or the
other -- that is what makes it *dense*, and dense is what makes IoU computable, because IoU
needs the false negatives. `status` reports the fraction still untouched per tile.

STEP 3, `import`. Writes each finished tile into dataset_cache as `<stem>_img.npy` and
`<stem>_gt.npy`, which pipeline.GT_STEMS now discovers automatically -- so the retrain gate,
crossval_grouped, crossval_groups and the scorecard all pick it up with no further wiring.
The 17-feature stack is built by ensure_gt_features on the next retrain.

REFUSES A TILE THAT IS NOT DENSE. Below --min-covered (default 0.95) of pixels judged, the
tile is skipped with a message. A sparsely-painted tile imported as dense ground truth would
count every unpainted crack pixel as background -- a false negative that is really a missing
label -- and that silently deflates every recall number computed afterwards. Measured on this
project's own sparse corrections: treating them as dense gives a mean IoU of 0.06.
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


def cmd_import(a):
    man = manifest()
    have = loaded()
    os.makedirs(P.GT_CACHE, exist_ok=True)
    wrote, skipped = 0, []
    for t in man["tiles"]:
        m = have.get(t["tile"])
        if m is None:
            skipped.append((t["tile"], "not loaded")); continue
        cov, cr = _judged(m["id"])
        if cov < a.min_covered:
            skipped.append((t["tile"], f"only {cov*100:.1f}% judged")); continue
        corr = np.asarray(S.load_npy(m["id"], "correction.npy"))
        img = np.asarray(S.load_npy(m["id"], "img.npy"))
        if corr.shape != img.shape:
            skipped.append((t["tile"], "shape mismatch")); continue
        stem = f"annot_{os.path.splitext(t['tile'])[0]}"
        gt = (corr == 1)
        np.save(os.path.join(P.GT_CACHE, f"{stem}_img.npy"), img.astype(np.float32))
        np.save(os.path.join(P.GT_CACHE, f"{stem}_gt.npy"), gt)
        # provenance next to the data, so nobody has to guess where a stem came from
        with open(os.path.join(P.GT_CACHE, f"{stem}_source.json"), "w") as f:
            json.dump(dict(tile=t["tile"], source_image=t["filename"], group=t["group"],
                           y=t["y"], x=t["x"], size=t["size"],
                           judged_fraction=round(cov, 4), crack_fraction=round(cr, 4),
                           sampling="uniform-random inside specimen support",
                           annotated_by="owner, without a model overlay"), f, indent=1)
        wrote += 1
        print(f"  {stem}  {cr*100:5.2f}% crack, {cov*100:.1f}% judged")
    if skipped:
        print(f"\n  skipped {len(skipped)}:")
        for n, why in skipped[:12]:
            print(f"    {n}  --  {why}")
    if wrote:
        stems = P._discover_gt_stems()
        print(f"\n{wrote} tile(s) written to dataset_cache.")
        print(f"pipeline.GT_STEMS now discovers {len(stems)} stems "
              f"({len(stems)-len(P.GT_STEMS_SHIPPED)} new).")
        print("\nThe 17-feature stacks build on the next Retrain (ensure_gt_features), and")
        print("from then on the gate, crossval_grouped and the scorecard all use them.")
        print("For the first honest AM/HC number without waiting for a retrain:")
        print("  python3 code/crossval.py")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("load", "status", "import"):
        p = sub.add_parser(name)
        p.add_argument("--min-covered", type=float, default=0.95,
                       help="fraction of pixels that must be judged for a tile to count")
    a = ap.parse_args()
    return {"load": cmd_load, "status": cmd_status, "import": cmd_import}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
