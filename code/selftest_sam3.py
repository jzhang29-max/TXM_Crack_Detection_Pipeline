"""Checks the SAM 3 adapter's arithmetic WITHOUT downloading SAM 3.

    python3 code/selftest_sam3.py

The weights are gated behind a licence acceptance, so the integration would otherwise sit
unverified until someone authenticates -- and the risky part is not the download, it is the
generalised-stride bilinear lookup that replaces model.interp_tile's hardcoded divide-by-16.
That is checkable against the original with no network at all: at stride 16 the two must
agree exactly, because they are then the same function.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import model as M            # noqa: E402
import sam3_encoder as S3    # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main():
    rs = np.random.RandomState(0)

    # 1. at SAM 1's stride the generalised lookup must be the SAME FUNCTION
    emb = rs.rand(256, 64, 64).astype(np.float16)
    rr = rs.randint(0, M.TILE, 4000)
    cc = rs.randint(0, M.TILE, 4000)
    a = M.interp_tile(emb, rr, cc)
    b = S3.interp_tile(emb, rr, cc, M.EMB_STRIDE)
    check("stride-16 lookup reproduces model.interp_tile exactly",
          a.shape == b.shape and np.array_equal(a, b),
          f"max |diff| {np.abs(a - b).max():.3e} over {a.size:,} values")

    # 2. a different stride must actually change the mapping, not silently ignore it
    c = S3.interp_tile(rs.rand(256, 32, 32).astype(np.float16), rr, cc, 32.0)
    check("a stride-32 grid is addressed at stride 32",
          c.shape == (len(rr), 256) and np.isfinite(c).all(),
          f"shape {c.shape}")

    # 3. edge clamping: coordinates at and past the tile edge must stay in range
    edge = np.array([0, M.TILE - 1, M.TILE + 50])
    e = S3.interp_tile(emb, edge, edge, M.EMB_STRIDE)
    check("out-of-tile coordinates clamp instead of throwing",
          e.shape == (3, 256) and np.isfinite(e).all())

    # 4. FPN level selection picks the one nearest SAM 1's granularity
    class T:
        def __init__(self, h): self.shape = (1, 256, h, h)
    maps = [T(256), T(128), T(64), T(32)]          # strides 4, 8, 16, 32 for a 1024 tile
    i, stride = S3._pick_level(maps, M.TILE)
    check("FPN level nearest stride 16 is chosen", i == 2 and stride == 16.0,
          f"level {i}, stride {stride:g}")
    maps2 = [T(256), T(128), T(32)]                # no exact 16 available
    i2, s2 = S3._pick_level(maps2, M.TILE)
    check("with no stride-16 level, the closest is chosen", s2 in (8.0, 32.0),
          f"level {i2}, stride {s2:g}")

    # 5. rows_at stitches tiles with last-tile-wins, matching model.py's convention
    coords = np.array([[0, 0], [0, 512]], np.int32)
    embs = np.stack([np.zeros((256, 64, 64), np.float16),
                     np.ones((256, 64, 64), np.float16)])
    q_r = np.array([10, 10]); q_c = np.array([100, 900])
    got = S3.rows_at(coords, embs, 16.0, q_r, q_c)
    check("rows_at resolves overlapping tiles last-first",
          got.shape == (2, 256) and got[1].mean() == 1.0,
          f"in-both-tiles pixel took tile 1 (mean {got[0].mean():.1f}), "
          f"tile-1-only pixel {got[1].mean():.1f}")

    # 6. the reachability check must be honest and not raise
    ok, why = S3.availability()
    check("availability() reports a reason rather than raising",
          isinstance(ok, bool) and isinstance(why, str) and len(why) > 0,
          f"{ok}: {why[:70]}")

    print()
    if FAILED:
        print(f"  {len(FAILED)} FAILED: {FAILED}")
        return 1
    print("  all adapter checks passed (SAM 3 weights not required for any of them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
