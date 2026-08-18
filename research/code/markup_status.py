"""
Markup progress across all 71 images -- what has been hand-marked, what is
still only carrying auto-generated labels, and what is untouched.

Distinguishes three states, because they mean very different things for
training:

  HAND-MARKED   the image has force-CRACK labels. Only a human puts these here
                now: every automatically generated positive label was reverted
                after two of them turned out to be wrong (elongated inclusions
                mislabelled as crack; the dark wedge mislabelled as a thick
                crack). So force-crack == the owner's own judgment.
  NEG-ONLY      only force-NOT-crack labels. These come from the automatic
                off-specimen / crack-free passes and are trustworthy, but they
                teach the model where cracks are ABSENT, never what one looks
                like. An image in this state contributes no positive signal.
  UNTOUCHED     no labels at all.

The gap that matters: every crack pixel the model has ever learned comes from
the 12 original B2 images. AM and Wrought have ZERO positive examples, which is
why they still mark the wedge rim instead of the thin centre cracks.

Usage:
    python3 markup_status.py            # full table
    python3 markup_status.py --todo     # only what still needs hand-marking
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo", action="store_true")
    args = ap.parse_args()

    rows = []
    for info in pc.list_images():
        nm, grp = info["name"], info.get("group", "?")
        cp = os.path.join(pc.CORRECTIONS_DIR, f"{nm}_correction.npy")
        n_crack = n_neg = 0
        if os.path.exists(cp):
            c = np.load(cp)
            n_crack, n_neg = int((c == 1).sum()), int((c == 2).sum())
            del c
        state = "HAND-MARKED" if n_crack else ("NEG-ONLY" if n_neg else "UNTOUCHED")
        rows.append(dict(name=nm, group=grp, crack=n_crack, neg=n_neg, state=state,
                         input="flatfielded" if pc.uses_flatfield(nm) else "raw"))

    by = {}
    for r in rows: by.setdefault(r["group"], []).append(r)

    if not args.todo:
        print(f"{'state':13s} {'+crack px':>11s} {'-crack px':>12s} {'input':>12s}  image")
        for g, rs in sorted(by.items()):
            print(f"\n--- {g} ({len(rs)} images, {rs[0]['input']} input) ---")
            for r in sorted(rs, key=lambda r: r["name"]):
                print(f"{r['state']:13s} {r['crack']:11,d} {r['neg']:12,d} {r['input']:>12s}  "
                      f"{r['name'].split('_idx')[0][-40:]}")

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"{'group':26s} {'n':>3s} {'hand-marked':>12s} {'neg-only':>9s} {'untouched':>10s}")
    for g, rs in sorted(by.items()):
        h = sum(1 for r in rs if r["state"] == "HAND-MARKED")
        n = sum(1 for r in rs if r["state"] == "NEG-ONLY")
        u = sum(1 for r in rs if r["state"] == "UNTOUCHED")
        print(f"{g:26s} {len(rs):3d} {h:12d} {n:9d} {u:10d}")
    tot_h = sum(1 for r in rows if r["state"] == "HAND-MARKED")
    print(f"\n{tot_h} of {len(rows)} images have force-CRACK labels (i.e. a human marked a crack).")

    need = [r for r in rows if r["state"] != "HAND-MARKED" and r["group"] in
            {"AM 316LH Fatigue", "Wrought 316L H Fatigue"}]
    if need:
        print(f"\nHIGHEST VALUE TO MARK NEXT -- {len(need)} AM/Wrought images with no positive")
        print("examples. These groups have ZERO crack examples in training, which is why they")
        print("mark the wedge rim instead of the thin centre cracks. Even 2-3 per group helps:")
        for r in need[:8]:
            print(f"   {r['group'][:22]:24s} {r['name'].split('_idx')[0][-42:]}")
        if len(need) > 8:
            print(f"   ... and {len(need)-8} more")

if __name__ == "__main__":
    main()
