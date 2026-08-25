"""Where the probability mass sits, and whether the probabilities are calibrated.

Two things the sweep exposed and could not explain:
  THE CLIFF. Whole-frame FP on crack-free specimen goes 27.1% at t=0.42 -> 23.1% at 0.43
  -> 4.8% at 0.44 -> 0.21% at 0.50. A 130x change over 8 hundredths means a large mass of
  pixels is piled up in a narrow probability band just below the shipped threshold.

  CALIBRATION. index.html tells the user 0.50 is "the calibrated default". A reliability
  curve on the 4 densely-labelled frames tests that literally: of the pixels the model
  scores p, what fraction are actually crack?
"""
import sys, os, json
import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "research", "oppoint"))
os.chdir(P0)
from analyze import load, NB, CENTERS, above, idx

OUT = os.path.join(P0, "research", "oppoint")
imgs = load()

# ---------------------------------------------------------------- 1. the cliff
print("=" * 78)
print("1. WHERE THE PROBABILITY MASS SITS (crack-free specimens, on-specimen only)")
print("=" * 78)
hs = np.zeros(NB, np.int64)
for d in imgs:
    if d["clean"] and "h_spec" in d:
        hs = hs + d["h_spec"]
tot = hs.sum()
print(f"on-specimen pixels over the 6 crack-free specimens: {tot:,}")
print(f"\n{'band':>12} {'pixels':>14} {'% of on-spec':>13}")
bands = [(0.0, .20), (.20, .30), (.35, .40), (.40, .42), (.42, .43), (.43, .44),
         (.44, .45), (.45, .46), (.46, .47), (.47, .48), (.48, .50), (.50, .60),
         (.60, .80), (.80, 1.0)]
for a, b in bands:
    n = int(hs[idx(a):idx(b)].sum())
    print(f"{a:5.2f}-{b:4.2f} {n:>14,} {n/tot*100:12.4f}%")
peak = int(np.argmax(hs))
print(f"\nmodal bin: p={CENTERS[peak]:.4f} holding {hs[peak]/tot*100:.3f}% of on-specimen px")
top = np.argsort(hs)[::-1][:8]
print("8 heaviest bins:", ", ".join(f"{CENTERS[i]:.3f}({hs[i]/tot*100:.2f}%)" for i in top))

# same for the whole frame, all 71 images
ha = np.zeros(NB, np.int64)
for d in imgs:
    ha = ha + d["h_all"]
ta = ha.sum()
print(f"\nALL 71 frames, whole-frame, {ta:,} px. Fraction in 0.42-0.48: "
      f"{ha[idx(.42):idx(.48)].sum()/ta*100:.3f}%")
pk = int(np.argmax(ha))
print(f"modal bin over all frames: p={CENTERS[pk]:.4f} ({ha[pk]/ta*100:.3f}%)")

# ------------------------------------------------- 2. reliability on dense GT
print("\n" + "=" * 78)
print("2. RELIABILITY ON THE 4 DENSELY-LABELLED FRAMES (every pixel labelled)")
print("=" * 78)
g1 = np.zeros(NB, np.int64); g0 = np.zeros(NB, np.int64)
for d in imgs:
    if "h_gt1" in d:
        g1 = g1 + d["h_gt1"]; g0 = g0 + d["h_gt0"]
edges = [0, .05, .1, .2, .3, .4, .45, .5, .55, .6, .7, .8, .9, 1.0]
print(f"{'prob band':>13} {'n px':>13} {'predicted':>10} {'actual':>8} {'gap':>8}")
rel = []
for a, b in zip(edges[:-1], edges[1:]):
    lo, hi = idx(a), idx(b)
    n1, n0 = int(g1[lo:hi].sum()), int(g0[lo:hi].sum())
    n = n1 + n0
    if n == 0:
        continue
    # mass-weighted mean predicted prob inside the band
    w = (g1 + g0)[lo:hi]
    pred = float((CENTERS[lo:hi] * w).sum() / max(w.sum(), 1))
    act = n1 / n
    rel.append(dict(lo=a, hi=b, n=n, pred=pred, act=act))
    print(f"{a:5.2f}-{b:4.2f} {n:>13,} {pred:10.4f} {act:8.4f} {act-pred:+8.4f}")
ece = sum(r["n"] * abs(r["act"] - r["pred"]) for r in rel) / sum(r["n"] for r in rel)
print(f"\nexpected calibration error (mass-weighted) = {ece:.4f}")
# where does actual probability cross 0.5?
for r in rel:
    if r["act"] >= 0.5:
        print(f"actual P(crack) first reaches 0.50 in the predicted band "
              f"{r['lo']:.2f}-{r['hi']:.2f} (predicted {r['pred']:.3f}, actual {r['act']:.3f})")
        break

# the decision-relevant question: at which predicted prob is P(crack|p)=0.5 ?
cum1 = g1.astype(float); cum0 = g0.astype(float)
frac = cum1 / np.maximum(cum1 + cum0, 1)
ok = (cum1 + cum0) > 10000
cand = [i for i in range(NB) if ok[i] and frac[i] >= 0.5]
if cand:
    print(f"finest bin where P(crack|p)>=0.5 with >10k px: p={CENTERS[min(cand)]:.4f}")

json.dump(dict(reliability=rel, ece=ece,
               on_spec_hist=hs.tolist(), all_hist=ha.tolist()),
          open(os.path.join(OUT, "calib.json"), "w"), indent=2, default=float)
print("\nwrote calib.json")
