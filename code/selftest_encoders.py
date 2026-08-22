"""Checks the encoder adapters' arithmetic WITHOUT downloading anything.

    python3 code/selftest_encoders.py

The risky code is not the download, it is the generalised-stride bilinear lookup replacing
model.interp_tile's hardcoded divide-by-16, and the pyramid/level handling that has to cope
with three APIs returning three shapes. All of that is checkable offline.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import model as M            # noqa: E402
import encoders as E         # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


class T:
    """Stand-in for a torch tensor: shape plus the permute the adapter may call."""
    def __init__(self, shape): self.shape = tuple(shape)
    def permute(self, *o): return T([self.shape[i] for i in o])


def main():
    rs = np.random.RandomState(0)

    emb = rs.rand(256, 64, 64).astype(np.float16)
    rr = rs.randint(0, M.TILE, 4000); cc = rs.randint(0, M.TILE, 4000)
    a = M.interp_tile(emb, rr, cc)
    b = E.interp_tile(emb, rr, cc, M.EMB_STRIDE)
    check("stride-16 lookup reproduces model.interp_tile exactly",
          a.shape == b.shape and np.array_equal(a, b),
          f"max |diff| {np.abs(a - b).max():.3e} over {a.size:,} values")

    c = E.interp_tile(rs.rand(256, 32, 32).astype(np.float16), rr, cc, 32.0)
    check("a stride-32 grid is addressed at stride 32",
          c.shape == (len(rr), 256) and np.isfinite(c).all())

    edge = np.array([0, M.TILE - 1, M.TILE + 50])
    check("out-of-tile coordinates clamp instead of throwing",
          E.interp_tile(emb, edge, edge, M.EMB_STRIDE).shape == (3, 256))

    i, s = E._pick_level([T((1, 256, 256, 256)), T((1, 256, 128, 128)),
                          T((1, 256, 64, 64)), T((1, 256, 32, 32))], M.TILE, 256)
    check("level nearest stride 16 is chosen", i == 2 and s == 16.0, f"level {i}, stride {s:g}")

    i2, s2 = E._pick_level([T((1, 256, 256, 256)), T((1, 256, 128, 128)),
                            T((1, 256, 32, 32))], M.TILE, 256)
    check("with no stride-16 level, the closest is taken", s2 in (8.0, 32.0),
          f"level {i2}, stride {s2:g}")

    # a 1x1 global-pooled level must never be selected as a dense feature map
    i3, s3 = E._pick_level([T((1, 256, 1, 1)), T((1, 256, 64, 64))], M.TILE, 256)
    check("a degenerate 1x1 level is skipped", i3 == 1 and s3 == 16.0, f"level {i3}")

    # channels-last must be normalised, not silently mis-read
    ch_last = E._to_bchw(T((1, 64, 64, 256)), 256)
    ch_first = E._to_bchw(T((1, 256, 64, 64)), 256)
    check("channels-last maps are transposed to channels-first",
          ch_last.shape == (1, 256, 64, 64) and ch_first.shape == (1, 256, 64, 64),
          f"{ch_last.shape} / {ch_first.shape}")

    # the three return shapes the SAM APIs actually use
    class Out1:  # SAM 3
        fpn_hidden_states = [T((1, 256, 64, 64))]
    class Out3:  # a bare last_hidden_state
        last_hidden_state = T((1, 256, 64, 64))
        fpn_hidden_states = None
    ok = (len(E._pyramid(Out1())) == 1
          and len(E._pyramid([T((1, 256, 64, 64)), T((1, 256, 32, 32))])) == 2   # SAM 2 list
          and len(E._pyramid(Out3())) == 1)
    check("all three pyramid return shapes are handled", ok)

    coords = np.array([[0, 0], [0, 512]], np.int32)
    embs = np.stack([np.zeros((256, 64, 64), np.float16), np.ones((256, 64, 64), np.float16)])
    got = E.rows_at(coords, embs, 16.0, np.array([10, 10]), np.array([100, 900]))
    check("rows_at resolves overlapping tiles last-first",
          got.shape == (2, 256) and got[1].mean() == 1.0)

    for n in ("sam2", "sam3"):
        ok_n, why = E.availability(n)
        check(f"availability({n}) reports rather than raises",
              isinstance(ok_n, bool) and isinstance(why, str) and why,
              f"{ok_n}: {why[:60]}")

    print()
    if FAILED:
        print(f"  {len(FAILED)} FAILED: {FAILED}")
        return 1
    print("  all adapter checks passed (no encoder weights required)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
