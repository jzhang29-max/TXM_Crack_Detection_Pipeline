"""
First-run unpack for a distributed checkout. Idempotent; run_app.sh calls it
every start and it does nothing once expanded.

Why the repo ships compressed instead of ready-to-use: a git repo has to stay
small, and the arrays this project needs do not.

  paint/corrections/*.npy   1.0 GB raw  ->  ~3 MB compressed  (346x; they are
                            uint8 and overwhelmingly zero)
                            stack is a pure function of the image, so it is
                            recomputed here rather than stored. LARGE_343_75's
                            alone is 1.5 GB.

So the package carries dataset_cache/packed.npz plus corrections.npz, and this
expands them into the plain .npy layout every other module expects. Nothing else
in the codebase needs to know packaging happened.

Usage:
    python3 code/unpack_package.py            # expand + build missing features
    python3 code/unpack_package.py --check    # report what is present, change nothing
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
CORR_DIR = os.path.join(PROJECT, "paint", "corrections")
PACKED_CORR = os.path.join(CORR_DIR, "corrections.npz")


def expand_corrections():
    if not os.path.exists(PACKED_CORR):
        return 0
    z = np.load(PACKED_CORR)
    os.makedirs(CORR_DIR, exist_ok=True)
    n = 0
    for key in z.files:
        out = os.path.join(CORR_DIR, f"{key}_correction.npy")
        if os.path.exists(out):
            continue
        np.save(out, z[key])
        n += 1
    return n




def status():
    corr = len(glob.glob(os.path.join(CORR_DIR, "*_correction.npy")))
    return dict(correction_masks=corr, packed_corr=os.path.exists(PACKED_CORR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what is already expanded and exit")
    args = ap.parse_args()

    if args.check:
        for k, v in status().items():
            print(f"  {k}: {v}")
        return
    n = expand_corrections()
    print(f"==> unpacked {n} correction mask(s)" if n
          else f"==> nothing to unpack; {status()['correction_masks']} correction mask(s) "
               f"already expanded")


if __name__ == "__main__":
    main()
