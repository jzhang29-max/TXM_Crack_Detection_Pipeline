"""Specimen-group assignment, and the leave-one-specimen-group-out splitter.

WHY THIS FILE EXISTS. The previous run scored every arm with GroupKFold(5) grouped by
IMAGE. That guarantees train and test never share a frame -- and guarantees nothing about
specimens. The 60 labelled frames come from four specimen families and most families
contribute 12-24 frames, so a random 5-way partition by image puts frames of the SAME
specimen on both sides of every split. Frames of one specimen share its exact grey level,
its noise floor, its crack morphology and its tiling; a model that has seen 20 of a
specimen's 24 frames is close to in-sample on the other 4. Specimens differ far more than
frames do, so a +0.02 measured that way can be a +0.02 on interpolating within a specimen
and nothing at all on the next specimen the owner loads.

THE GROUPS, and how every frame is assigned. Purely from the filename, in this order (the
order matters: `b3_3_18lbf` contains both `b3_` and `lbf`, and `B2_333_75_um_zoom`
contains `B2_`), and each frame gets exactly one label:

    "HC"       filename contains "hc_"        -- HC_316L_fatigue_* additively manufactured
    "B2"       filename contains "b2_"        -- B2_*/b2_* (upper and lower case both occur)
    "B3"       filename contains "b3_"        -- b3_*
    "wrought"  filename contains "wrought_"   -- wrought_316L_fatigue_*

Matching is case-insensitive, which is required: the corpus contains both `B2_3_1_lbf`
and `b2_340_00`. assign() raises rather than guessing if a filename matches none or more
than one of the four, so an unassignable frame cannot silently end up in a group.

Note "HC" and "B2"/"B3" are different specimens, not different imaging days: the
260618/260619/260620 date stems correlate with group but are not what defines it.
"""

import re

GROUPS = ("HC", "B2", "B3", "wrought")

_PATTERNS = {
    "HC": re.compile(r"hc_", re.I),
    "B2": re.compile(r"b2_", re.I),
    "B3": re.compile(r"b3_", re.I),
    "wrought": re.compile(r"wrought_", re.I),
}


def assign(filename):
    """Filename -> one of GROUPS. Raises on ambiguity or no match."""
    hits = [g for g, p in _PATTERNS.items() if p.search(filename or "")]
    if len(hits) != 1:
        raise ValueError("cannot assign specimen group (%r matches %s)"
                         % (filename, hits or "nothing"))
    return hits[0]


def assign_all(filenames):
    return [assign(f) for f in filenames]


def loso_splits(frame_groups, row_group_index):
    """Leave-one-specimen-group-out splits over ROWS.

    frame_groups: list, one specimen-group name per frame, indexed like the per-frame
                  arrays in the cache.
    row_group_index: int array, the frame index each row came from.

    Yields (held_out_group_name, train_row_idx, test_row_idx) with every frame of the
    held-out specimen entirely on the test side.
    """
    import numpy as np
    fg = np.asarray(frame_groups)
    per_row = fg[np.asarray(row_group_index)]
    out = []
    for g in GROUPS:
        te = np.flatnonzero(per_row == g)
        tr = np.flatnonzero(per_row != g)
        if len(te) and len(tr):
            out.append((g, tr, te))
    return out
