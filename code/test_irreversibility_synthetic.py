"""Synthetic control with a specimen edge and a FIXED crack root.

The previous control had no specimen boundary, so it could not test the anchor. This one is
built the way the real thing is: dark outside, bright specimen, crack initiating at the
boundary and growing inward. The root is fixed by construction; the tip advances.
"""
import sys
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "core"))
import sequence as Q

H, W = 700, 1100
EDGE = 150                     # specimen starts at x=EDGE

def scene(length, seed=1, grow_seed=None):
    """(mask, image). Crack root fixed at (350, EDGE); tip at EDGE+length."""
    r = np.random.RandomState(seed)
    m = np.zeros((H, W), bool)
    y = 350.0
    for i, x in enumerate(range(EDGE, min(W, EDGE + length))):
        y += r.normal(0, 0.6)
        y = max(10, min(H - 11, y))
        m[int(y) - 3:int(y) + 4, x] = True
    img = np.zeros((H, W), np.float32)
    img[:, EDGE:] = 0.75                                  # specimen
    img += np.random.RandomState(7).normal(0, 0.02, (H, W))
    img[m] = 0.15                                          # crack is dark
    return m, img

def place(arr, dy, dx, fill=0):
    out = np.full_like(arr, fill)
    ys0, ys1 = max(0, dy), min(H, H + dy)
    xs0, xs1 = max(0, dx), min(W, W + dx)
    out[ys0:ys1, xs0:xs1] = arr[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out

m0, i0 = scene(600)
print("  SYNTHETIC CONTROL WITH A SPECIMEN EDGE AND A FIXED CRACK ROOT")
print("  earlier: crack length 600.  later: length 800 (real growth) plus a known shift.\n")
print(f"  {'true shift':>14}{'containment-only':>20}{'anchored':>18}{'cont. (anch.)':>15}{'method':>18}")
m1_base, i1_base = scene(800)
for (dy, dx) in [(0, 0), (12, -30), (-120, 240), (300, -500), (-40, 620)]:
    m1 = place(m1_base, dy, dx); i1 = place(i1_base, dy, dx, fill=0.0)
    c = Q.register_by_containment(m0, m1)
    r = Q.register_anchored(m0, m1, i0, i1)
    ok_c = "OK" if abs(c[0]-dy)<=8 and abs(c[1]-dx)<=8 else "wrong"
    ok_a = "OK" if abs(r['dy']-dy)<=8 and abs(r['dx']-dx)<=8 else "wrong"
    print(f"  {str((dy,dx)):>14}{str((c[0],c[1]))+' '+ok_c:>20}"
          f"{str((r['dy'],r['dx']))+' '+ok_a:>18}{r['containment']*100:>14.1f}%{r['method']:>18}")

print("\n  negative control -- a genuinely DIFFERENT crack (different root height):")
r2 = np.random.RandomState(0)
mD = np.zeros((H, W), bool); y = 150.0
for x in range(EDGE, EDGE + 600):
    y += r2.normal(0, 0.6); y = max(10, min(H-11, y)); mD[int(y)-3:int(y)+4, x] = True
iD = np.zeros((H, W), np.float32); iD[:, EDGE:] = 0.75; iD[mD] = 0.15
rD = Q.register_anchored(m0, mD, i0, iD)
print(f"    anchored containment {rD['containment']*100:.1f}%   "
      f"mouths {rD.get('mouth_earlier')} vs {rD.get('mouth_later')}")
print("    (a low number here is the CORRECT answer -- these are not the same crack)")
