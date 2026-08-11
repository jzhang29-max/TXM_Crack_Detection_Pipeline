"""
Quality-control gate on FLATFIELDING, not on the model.

Diagnosis this encodes: 13 of 27 AM images were flooded with predicted crack
(21-65% of frame) while the other 14 were fine (0.7-18.9%). The cause is not
the classifier -- it is that flatfielding FAILED on those 13, so the mosaic
tile-grid texture is still present in the "flatfielded" image and the model
is faithfully reporting it.

Measured separation, over the specimen interior only (pixels > 0.25 so the
off-specimen field does not skew it):

                   median brightness   median IQR
    flooded (13)         0.548            0.195
    working (14)         0.933            0.015

A correctly flatfielded specimen normalises to near-uniform bright: high
median, tiny spread. The failures are mid-grey with ~13x the internal
variation. Every working image has IQR <= 0.051; 11 of 13 failures have
IQR >= 0.131. The failures are concentrated in the "_tip" / "_tip_zoom"
zoomed acquisitions, and the giveaway pair is 1650_cycles (IQR 0.051,
0.7% predicted) vs 1650_cycles_tip (IQR 0.198, 60.8% predicted) -- same
specimen, same cycle count, different acquisition.

Such images should be RE-FLATFIELDED upstream, not compensated for
downstream. Labelling their texture as not-crack would teach the model to
suppress genuine texture in correctly-processed images too, and training on
them at all pollutes the model with a preprocessing artifact.

Usage:
    python3 qc_flatfield_quality.py [--json out.json]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc

PREDCACHE = os.path.join(pc.PROJECT_DIR, "paint", "flatfield_predcache")
SPECIMEN_MIN = 0.25    # ignore off-specimen field when measuring uniformity
IQR_MAX = 0.10         # above this, flatfielding did not flatten the tile structure
BRIGHT_MIN = 0.70      # a flatfielded specimen should normalise bright, not mid-grey


def assess(name):
    ip = os.path.join(PREDCACHE, f"{name}_img.npy")
    if not os.path.exists(ip):
        return None
    img = np.load(ip)
    body = img[img > SPECIMEN_MIN]
    del img
    if body.size < 1000:
        return dict(name=name, ok=False, reason="almost no specimen interior",
                    brightness=None, iqr=None)
    p25, p50, p75 = (float(x) for x in np.percentile(body, [25, 50, 75]))
    iqr = p75 - p25
    reasons = []
    if iqr > IQR_MAX:
        reasons.append(f"IQR {iqr:.3f} > {IQR_MAX} (tile-grid texture not flattened)")
    if p50 < BRIGHT_MIN:
        reasons.append(f"median brightness {p50:.3f} < {BRIGHT_MIN} (specimen not normalised bright)")
    return dict(name=name, ok=not reasons, brightness=p50, iqr=iqr,
                reason="; ".join(reasons) or None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(pc.PROJECT_DIR, "results", "qc_flatfield.json"))
    args = ap.parse_args()

    rows = []
    for info in pc.list_images():
        a = assess(info["name"])
        if a is None:
            continue
        a["group"] = info.get("group", "?")
        rows.append(a)

    bad = [r for r in rows if not r["ok"]]
    print(f"{len(rows)} images assessed; {len(bad)} FAILED flatfield QC\n")
    print(f"{'image':44s} {'bright':>7s} {'IQR':>7s}  status")
    for r in sorted(rows, key=lambda r: -(r["iqr"] or 0)):
        st = "FAIL" if not r["ok"] else "ok"
        b = f"{r['brightness']:.3f}" if r["brightness"] is not None else "   n/a"
        i = f"{r['iqr']:.3f}" if r["iqr"] is not None else "   n/a"
        print(f"{r['name'].split('_idx')[0][-42:]:44s} {b:>7s} {i:>7s}  {st}")

    print(f"\nFAILED by group:")
    by = {}
    for r in bad:
        by.setdefault(r["group"], []).append(r)
    for g, rs in sorted(by.items()):
        print(f"  {g:26s} {len(rs)} image(s)")
    print("\nThese need RE-FLATFIELDING upstream. Compensating downstream would teach")
    print("the model to suppress genuine texture in correctly-processed images too.")
    json.dump(rows, open(args.json, "w"), indent=2)
    print(f"\nSaved {args.json}")


if __name__ == "__main__":
    main()
