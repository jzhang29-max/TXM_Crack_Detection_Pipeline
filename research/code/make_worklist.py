"""
Regenerate WORKLIST.md — an ordered, tickable checklist of all 71 images.

Ordering is deliberate rather than alphabetical:
  1. Groups with ZERO positive crack examples first (Wrought, then AM). Every
     crack pixel the model has ever learned comes from B2, which is why those
     two groups trace wedge rims instead of real cracks. A mark there teaches
     the model something it cannot currently know.
  2. Within a group, worst predicted output first (highest predicted area).
     Those frames are where the current model is most wrong, so a correction
     carries the most information.

Checkboxes reflect whether the image has any force-CRACK label yet, i.e.
whether a human has marked a crack on it. Re-run after a markup session to
refresh the ticks.

Usage:
    python3 code/make_worklist.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_common as pc

PRIORITY = {"Wrought 316L H Fatigue": 0, "AM 316LH Fatigue": 1,
            "B3 316L Amb Tension": 2, "B2 316L H Tension": 3}


def main():
    root = pc.PROJECT_DIR
    sp = os.path.join(root, "results", "final_71_pergroup", "summary.json")
    summ = {r["name"]: r for r in json.load(open(sp))} if os.path.exists(sp) else {}

    rows = []
    for info in pc.list_images():
        nm, grp = info["name"], info.get("group", "?")
        cp = os.path.join(pc.CORRECTIONS_DIR, f"{nm}_correction.npy")
        n_crack = 0
        if os.path.exists(cp):
            c = np.load(cp)
            n_crack = int((c == 1).sum())
            del c
        s = summ.get(nm, {})
        rows.append(dict(name=nm, group=grp, done=n_crack > 0,
                         area=s.get("area_fraction"), rg=s.get("n_regions"),
                         inp=s.get("input", "?")))
    rows.sort(key=lambda r: (PRIORITY.get(r["group"], 9), -(r["area"] or 0)))

    total_done = sum(1 for r in rows if r["done"])
    out = ["# WORKLIST — all 71 images", "",
           f"**{total_done} / {len(rows)} marked** (has at least one force-CRACK label)", "",
           "Ordered by value, not alphabetically: groups with ZERO crack examples come",
           "first (Wrought, AM), then within each group worst-output-first, since those",
           "frames are where the model is most wrong and a correction teaches the most.",
           "",
           "Progress saves per image — stop and resume freely. Refresh the ticks with:",
           "`python3 code/make_worklist.py`", "",
           "Columns: predicted crack area, number of separate pieces found.", ""]
    cur = None
    for r in rows:
        if r["group"] != cur:
            cur = r["group"]
            d = sum(1 for x in rows if x["group"] == cur and x["done"])
            t = sum(1 for x in rows if x["group"] == cur)
            note = "  ← no crack examples yet, highest value" if PRIORITY.get(cur, 9) < 2 else ""
            out += ["", f"## {cur}  ({d}/{t} marked)  · {r['inp']} input{note}", ""]
        box = "[x]" if r["done"] else "[ ]"
        a = f"{r['area']*100:5.1f}%" if r["area"] is not None else "    ?"
        out.append(f"- {box} `{a}` {str(r['rg'] or '?'):>4s} pieces — {r['name'].split('_idx')[0]}")
    open(os.path.join(root, "WORKLIST.md"), "w").write("\n".join(out) + "\n")
    print(f"WORKLIST.md written — {total_done}/{len(rows)} marked")
    for g in sorted(set(r["group"] for r in rows), key=lambda g: PRIORITY.get(g, 9)):
        d = sum(1 for r in rows if r["group"] == g and r["done"])
        t = sum(1 for r in rows if r["group"] == g)
        print(f"  {g:26s} {d:2d}/{t:2d}")


if __name__ == "__main__":
    main()
