"""Exhaustive edge-case exercise of app/core/sequence.py public API. READ-ONLY: builds
synthetic arrays in memory, touches no file and no HTTP endpoint."""
import sys, os, traceback, warnings
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app", "core"))
import sequence as Q

PUB = [n for n in dir(Q) if not n.startswith("_") and callable(getattr(Q, n))
       and getattr(getattr(Q, n), "__module__", None) == "sequence"]
print("PUBLIC CALLABLES:", sorted(PUB))
print("MODULE CONSTANTS: COARSE_RANGE=%s COARSE_STEP=%s REFINE=%s MOUTH_BAND=%s"
      % (Q.COARSE_RANGE, Q.COARSE_STEP, Q.REFINE, Q.MOUTH_BAND))
print()

FAIL = []


def raises(name, fn, exc=ValueError):
    """Malformed input that SHOULD be rejected. A clear exception is the PASS here.

    Without this distinction the suite printed "2 hard failures" for the two cases where the
    module behaves correctly, which is the same trap as treating exit 0 as success -- just
    inverted. A guard that fires is not a defect.
    """
    try:
        r = fn()
    except exc as e:
        print(f"[ok ] {name}\n       -> correctly rejected: {type(e).__name__}: {e}")
        return None
    except Exception as e:                                            # noqa: BLE001
        FAIL.append((name, f"wrong exception type: {e!r}"))
        print(f"[XXX] {name}\n       !! expected {exc.__name__}, got {type(e).__name__}: {e}")
        return None
    FAIL.append((name, f"no exception raised; returned {r!r}"))
    print(f"[XXX] {name}\n       !! expected {exc.__name__}, but it returned a value")
    return r


def case(name, fn):
    """Run fn(); print result or the exception. Capture warnings too."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = fn()
            ws = "; ".join(sorted({f"{x.category.__name__}: {x.message}" for x in w}))
            print(f"[ok ] {name}\n       -> {r}" + (f"\n       !warn {ws}" if ws else ""))
            return r
        except Exception as e:
            FAIL.append((name, repr(e)))
            print(f"[XXX] {name}\n       !! {type(e).__name__}: {e}")
            print("       " + traceback.format_exc().splitlines()[-3].strip())
            return None


# ---------------------------------------------------------------- scene builders
H, W, EDGE = 700, 1100, 150


def scene(length, y0=350.0, seed=1):
    r = np.random.RandomState(seed)
    m = np.zeros((H, W), bool)
    y = y0
    for x in range(EDGE, min(W, EDGE + length)):
        y += r.normal(0, 0.6)
        y = max(10, min(H - 11, y))
        m[int(y) - 3:int(y) + 4, x] = True
    img = np.zeros((H, W), np.float32)
    img[:, EDGE:] = 0.75
    img += np.random.RandomState(7).normal(0, 0.02, (H, W))
    img[m] = 0.15
    return m, img


def place(arr, dy, dx, fill=0):
    out = np.full_like(arr, fill)
    ys0, ys1 = max(0, dy), min(H, H + dy)
    xs0, xs1 = max(0, dx), min(W, W + dx)
    out[ys0:ys1, xs0:xs1] = arr[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


m600, i600 = scene(600)
m800, i800 = scene(800)

print("=" * 100)
print("A. GROUND-TRUTH SANITY: pair_consistency / monotone_repair / sequence_report on a")
print("   pair whose true shift is KNOWN. These three were not re-tested after the")
print("   specimen_mask change; note they never call the anchor at all.")
print("=" * 100)

case("A1 pair_consistency(m600, m800) no shift -- truth dy=dx=0",
     lambda: Q.pair_consistency(m600, m800))

m800s = place(m800, -120, 240)
r = case("A2 pair_consistency(m600, shifted m800) truth (-120,240) -- register_anchored gets"
         "\n     this right; pair_consistency uses containment only",
         lambda: Q.pair_consistency(m600, m800s))
if r:
    print(f"       CHECK truth (-120,240) vs reported ({r['dy']},{r['dx']})  "
          f"-> {'MATCH' if (r['dy'], r['dx']) == (-120, 240) else 'MISREGISTERED'}")
    print(f"       CHECK violation={r['violation']} vs vanished_px/earlier_px="
          f"{r['vanished_px'] / max(r['earlier_px'], 1):.4f}  (should agree)")

r = case("A3 pair_consistency(identical masks) -- must be dy=dx=0, containment 1.0",
         lambda: Q.pair_consistency(m600, m600))
if r:
    print(f"       CHECK containment={r['containment']} (expect 1.0), "
          f"violation={r['violation']} (expect 0.0), "
          f"vanished_px={r['vanished_px']} (expect 0)")

r = case("A4 monotone_repair(m600, m800) no shift", lambda: Q.monotone_repair(m600, m800))
if r:
    rep, info = r
    print(f"       repaired shape={rep.shape} dtype={rep.dtype}; superset of m800? "
          f"{bool((m800 & ~rep).sum() == 0)}; contains registered m600? "
          f"{bool((m600 & ~rep).sum() == 0)}")

r = case("A5 sequence_report 3 frames in order",
         lambda: Q.sequence_report([("600", m600), ("800", m800), ("800shift", m800s)]))
if r:
    for d in r:
        print(f"       {d['pair']}: dy={d['dy']} dx={d['dx']} cont={d['containment']} "
              f"area_decreased={d['area_decreased']}")

print()
print("=" * 100)
print("B. specimen_mask AND ITS CALLERS after the binary_closing + binary_fill_holes change")
print("=" * 100)


def spec_stats(img):
    sp = Q.specimen_mask(img)
    return dict(shape=sp.shape, dtype=str(sp.dtype), frac_true=round(float(sp.mean()), 4),
                edge_pos=Q._edge_position(img))


case("B1 specimen_mask on the good synthetic frame (specimen from x=150)",
     lambda: spec_stats(i600))
case("B2 does the closing+fill actually cover the dark crack? "
     "(crack px inside specimen_mask)",
     lambda: f"{int((m600 & Q.specimen_mask(i600)).sum())} of {int(m600.sum())} crack px "
             f"are inside the support ({100 * (m600 & Q.specimen_mask(i600)).sum() / m600.sum():.1f}%)")
case("B3 crack_mouth on the good frame (should find the root near y=350, x~=EDGE)",
     lambda: Q.crack_mouth(m600, Q.specimen_mask(i600)))
case("B4 register_anchored good frames, truth (0,0)",
     lambda: {k: v for k, v in Q.register_anchored(m600, m800, i600, i800).items()})

print("\n-- B5 MOUTH_BAND documented as 60 px; measure the band the code actually builds --")


def band_width():
    from scipy.ndimage import binary_dilation
    sp = np.zeros((H, W), bool)
    sp[:, EDGE:] = True            # perfectly clean vertical boundary at x=EDGE
    outside = binary_dilation(~sp, iterations=3)
    band = outside & sp
    for _ in range(max(1, Q.MOUTH_BAND // 8)):
        band = binary_dilation(band, iterations=4) & sp
    cols = np.nonzero(band.any(axis=0))[0]
    return (f"band spans x=[{cols.min()},{cols.max()}] -> width {cols.max() - cols.min() + 1} px "
            f"inside the specimen, while MOUTH_BAND says {Q.MOUTH_BAND}")


case("B5 effective mouth-band width vs MOUTH_BAND", band_width)

print()
print("=" * 100)
print("C. DEGENERATE MASKS")
print("=" * 100)

empty = np.zeros((H, W), bool)
allT = np.ones((H, W), bool)
one = np.zeros((H, W), bool); one[350, 500] = True          # even coords
one_odd = np.zeros((H, W), bool); one_odd[351, 501] = True   # odd coords

case("C1 register_by_containment(empty earlier, m800)", lambda: Q.register_by_containment(empty, m800))
r = case("C2 pair_consistency(EMPTY earlier, m800) -- garbage in", lambda: Q.pair_consistency(empty, m800))
if r:
    print(f"       NOTE containment={r['containment']} violation={r['violation']} "
          f"from a completely empty earlier mask")
r = case("C3 pair_consistency(m600, EMPTY later) -- crack vanished entirely",
         lambda: Q.pair_consistency(m600, empty))
if r:
    print(f"       containment={r['containment']} area_decreased={r['area_decreased']}")
case("C4 pair_consistency(empty, empty)", lambda: Q.pair_consistency(empty, empty))
case("C5 register_by_containment(allTrue, allTrue) -- expect dy=dx=0, cont 1.0",
     lambda: Q.register_by_containment(allT, allT))
case("C6 pair_consistency(allTrue, allTrue)", lambda: Q.pair_consistency(allT, allT))
case("C7 pair_consistency(m600, allTrue) -- later covers everything, expect cont 1.0",
     lambda: Q.pair_consistency(m600, allT))
r = case("C8 pair_consistency(allTrue, m600) -- massive shrink",
         lambda: Q.pair_consistency(allT, m600))
if r:
    print(f"       containment={r['containment']} area_decreased={r['area_decreased']}")

r = case("C9 SINGLE PIXEL, identical earlier and later, even coords (350,500)",
         lambda: Q.pair_consistency(one, one))
if r:
    print(f"       containment={r['containment']} (a mask identical to itself: expect 1.0)")
r = case("C10 SINGLE PIXEL, identical earlier and later, ODD coords (351,501)",
         lambda: Q.pair_consistency(one_odd, one_odd))
if r:
    print(f"       containment={r['containment']} violation={r['violation']} "
          f"earlier_px={r['earlier_px']} later_px={r['later_px']} "
          f"vanished_px={r['vanished_px']}  <-- identical masks")
one_moved = np.zeros((H, W), bool); one_moved[350 + 40, 500 + 60] = True
r = case("C11 SINGLE PIXEL shifted by a known (40,60): can the search find it?",
         lambda: Q.pair_consistency(one, one_moved))
if r:
    print(f"       truth (40,60) reported ({r['dy']},{r['dx']}) cont={r['containment']}")

thin = np.zeros((H, W), bool); thin[301, EDGE:EDGE + 600] = True   # 1-px-tall crack, odd row
thin2 = np.zeros((H, W), bool); thin2[301, EDGE:EDGE + 800] = True
r = case("C12 1-PIXEL-WIDE crack, later is a strict superset (truth (0,0), cont must be 1.0)",
         lambda: Q.pair_consistency(thin, thin2))
if r:
    print(f"       containment={r['containment']} violation={r['violation']} "
          f"vanished_px={r['vanished_px']} of earlier_px={r['earlier_px']}")

print()
print("=" * 100)
print("D. MASKS / IMAGES OF DIFFERENT SHAPES")
print("=" * 100)

small = m800[:400, :600].copy()
r = case("D1 pair_consistency(700x1100 earlier, 400x600 later)",
         lambda: Q.pair_consistency(m600, small))
if r:
    a_in_common = int(m600[:400, :600].sum())
    print(f"       earlier_px={r['earlier_px']} (FULL 700x1100 mask) but only {a_in_common} px "
          f"lie in the 400x600 common region")
    print(f"       later_px={r['later_px']} area_decreased={r['area_decreased']} "
          f"<-- areas compared over DIFFERENT fields of view")
r = case("D2 monotone_repair(700x1100, 400x600) -- what shape comes back?",
         lambda: Q.monotone_repair(m600, small))
if r:
    rep, info = r
    print(f"       repaired.shape={rep.shape}  input later.shape={small.shape}  "
          f"earlier.shape={m600.shape}")
case("D3 sequence_report with mismatched shapes",
     lambda: [(d['pair'], d['dy'], d['dx'], d['containment']) for d in
              Q.sequence_report([("a", m600), ("b", small), ("c", m800)])])
case("D4 register_anchored with mask/image shape mismatch (mask 700x1100, img 400x600)",
     lambda: Q.register_anchored(m600, small, i600, i800[:400, :600]))

print()
print("=" * 100
      )
print("E. IMAGES WITH NO SPECIMEN, NO BOUNDARY, OR NaN")
print("=" * 100)

uniform = np.full((H, W), 0.5, np.float32)
case("E1 specimen_mask(uniform 0.5 image) -- no specimen at all", lambda: spec_stats(uniform))
case("E2 _edge_position(uniform)", lambda: Q._edge_position(uniform))
case("E3 crack_mouth(m600, specimen_mask(uniform))",
     lambda: Q.crack_mouth(m600, Q.specimen_mask(uniform)))
case("E4 register_anchored with uniform images -> must fall back and say so",
     lambda: Q.register_anchored(m600, m800, uniform, uniform))

noise = np.random.RandomState(3).normal(0.5, 0.02, (H, W)).astype(np.float32)
case("E5 specimen_mask(PURE NOISE, no specimen) -- does it invent a specimen?",
     lambda: spec_stats(noise))
case("E6 _edge_position(PURE NOISE) -- garbage in, what comes out?",
     lambda: Q._edge_position(noise))
case("E7 register_anchored on PURE NOISE images (masks are real cracks)",
     lambda: Q.register_anchored(m600, m800, noise, noise))

full_spec = np.full((H, W), 0.75, np.float32) + np.random.RandomState(7).normal(0, 0.02, (H, W))
full_spec = full_spec.astype(np.float32); full_spec[m600] = 0.15
case("E8 specimen_mask(frame that is ENTIRELY specimen, no boundary in view)",
     lambda: spec_stats(full_spec))
case("E9 crack_mouth on entirely-specimen frame (crack touches no boundary)",
     lambda: Q.crack_mouth(m600, Q.specimen_mask(full_spec)))
case("E10 register_anchored when the crack touches no boundary",
     lambda: Q.register_anchored(m600, m800, full_spec, full_spec))

interior = np.zeros((H, W), bool)
interior[350:357, 500:900] = True        # crack fully interior, nowhere near x=150
img_int = i600.copy(); img_int[m600] = 0.75; img_int[interior] = 0.15
case("E11 crack fully INTERIOR (never reaches the specimen edge) -> mouth must be None",
     lambda: Q.crack_mouth(interior, Q.specimen_mask(img_int)))
case("E12 register_anchored, interior crack",
     lambda: Q.register_anchored(interior, interior, img_int, img_int))

nan_all = np.full((H, W), np.nan, np.float32)
case("E13 specimen_mask(ALL NaN)", lambda: spec_stats(nan_all))
one_nan = i600.copy(); one_nan[0, 0] = np.nan
case("E14 specimen_mask with ONE NaN pixel in an otherwise perfect frame",
     lambda: spec_stats(one_nan))
case("E15 crack_mouth with ONE NaN pixel in the frame",
     lambda: Q.crack_mouth(m600, Q.specimen_mask(one_nan)))
r = case("E16 register_anchored with ONE NaN pixel in img_later -- silent fallback?",
         lambda: Q.register_anchored(m600, m800, i600, one_nan))
if r:
    print(f"       method={r.get('method')} reason={r.get('reason')}")
case("E17 _edge_position with ONE NaN pixel", lambda: Q._edge_position(one_nan))
inf_img = i600.copy(); inf_img[5, 5] = np.inf
case("E18 specimen_mask with +inf in the frame", lambda: spec_stats(inf_img))
case("E19 pair_consistency with NaN INSIDE THE MASK argument (float mask, not bool)",
     lambda: Q.pair_consistency(np.where(m600, 1.0, np.nan), m800.astype(float)))

print()
print("=" * 100)
print("F. OTHER INPUT ABUSE")
print("=" * 100)
case("F1 sequence_report([])", lambda: Q.sequence_report([]))
case("F2 sequence_report(one frame)", lambda: Q.sequence_report([("only", m600)]))
case("F3 sequence_report(None)", lambda: Q.sequence_report(None))
case("F4 specimen_mask on a 3-channel RGB image", lambda: Q.specimen_mask(
    np.stack([i600, i600, i600], axis=-1)))
case("F5 specimen_mask on a uint8 image", lambda: spec_stats((i600 * 255).astype(np.uint8)))
raises("F6 specimen_mask on a 1-D array", lambda: Q.specimen_mask(np.linspace(0, 1, 100)))
case("F7 pair_consistency on tiny 4x4 masks (smaller than every decimation step)",
     lambda: Q.pair_consistency(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0],
                                          [0, 0, 0, 0]], bool),
                                np.ones((4, 4), bool)))
case("F8 register_by_containment with coarse_range=0", lambda: Q.register_by_containment(
    m600, m800s, coarse_range=0))
case("F9 monotone_repair(empty, empty)", lambda: Q.monotone_repair(empty, empty))
case("F10 crack_mouth(empty mask, good specimen)",
     lambda: Q.crack_mouth(empty, Q.specimen_mask(i600)))
case("F11 crack_mouth(good mask, empty specimen)",
     lambda: Q.crack_mouth(m600, np.zeros((H, W), bool)))
raises("F12 crack_mouth with mask/spec shape mismatch",
       lambda: Q.crack_mouth(m600, Q.specimen_mask(i600)[:400, :600]))
case("F13 register_anchored(refine=0)",
     lambda: Q.register_anchored(m600, m800s, i600, place(i800, -120, 240, fill=0.0), refine=0))

print()
print("=" * 100)
print("G. INTERNAL CONSISTENCY OF register_anchored's REPORTED dy AGAINST ITS OWN SCORE")
print("=" * 100)


def odd_anchor():
    """dy0 odd -> the loop scores dy//2*2 but reports dy. Force an odd mouth offset."""
    mA, iA = scene(600, y0=350.0)
    mB, iB = scene(800, y0=350.0)
    mB2, iB2 = place(mB, 51, 0), place(iB, 51, 0, fill=0.0)
    r = Q.register_anchored(mA, mB2, iA, iB2)
    dy, dx = r["dy"], r["dx"]
    ac, bc = Q._crop_common(mA, mB2)
    ad, bd = ac[::2, ::2], bc[::2, ::2]
    tot = max(int(ad.sum()), 1)
    sh_rep, _ = Q._shift_into(bd, dy // 2, dx // 2, ad.shape)      # what the code scores
    c_rep = int((ad & sh_rep).sum()) / tot
    # recompute at FULL resolution with the reported integer shift
    mv, val = Q._shift_into(bc, dy, dx, ac.shape)
    c_full = int((ac & mv).sum()) / max(int(ac.sum()), 1)
    return (f"truth dy=51 dx=0 | reported ({dy},{dx}) anchor_shift={r['anchor_shift']} "
            f"| containment reported {r['containment']} | dy//2*2 actually scored = "
            f"{(dy // 2) * 2} | recomputed at full res with the REPORTED shift = "
            f"{c_full:.4f} (decimated {c_rep:.4f})")


case("G1 odd anchor offset: reported dy vs the shift actually evaluated", odd_anchor)


def cont_full_vs_dec():
    r = Q.pair_consistency(m600, m800s)
    ac, bc = Q._crop_common(m600, m800s)
    mv, val = Q._shift_into(bc, r["dy"], r["dx"], ac.shape)
    av = ac & val
    return (f"pair_consistency containment={r['containment']} (2x-decimated) vs full-res "
            f"containment with the same shift={int((ac & mv).sum()) / int(ac.sum()):.4f}; "
            f"1-vanished/earlier={1 - r['vanished_px'] / r['earlier_px']:.4f}")


case("G2 is the reported containment the same at full resolution?", cont_full_vs_dec)

print()
print("=" * 100)
print("H. DOES pair_consistency / sequence_report EVER USE THE ANCHOR?")
print("=" * 100)
# Ask the RUNNING code, not its source text. The first version of this section grepped
# inspect.getsource() for "register_anchored", which reported False for all three functions the
# moment the calls were routed through a _register() helper -- a false negative that would have
# hidden a real regression. Every result carries `method`, so just read it.
_probe = [("pair_consistency", lambda *a: Q.pair_consistency(*a)),
          ("monotone_repair", lambda *a: Q.monotone_repair(*a)[1])]
for fname, fn in _probe:
    without = fn(m600, m800s).get("method")
    with_img = fn(m600, m800s, i600, i800).get("method")
    ok = "OK" if (without == "containment_only" and with_img == "anchored") else "PROBLEM"
    print(f"   {ok:>7}  {fname}: no images -> {without!r}, with images -> {with_img!r}")
    if ok == "PROBLEM":
        FAIL.append((f"H {fname}", f"no-images={without!r} with-images={with_img!r}"))
_sr2 = [r["method"] for r in Q.sequence_report([("a", m600), ("b", m800s)])]
_sr3 = [r["method"] for r in Q.sequence_report([("a", m600, i600), ("b", m800s, i800)])]
ok = "OK" if _sr2 == ["containment_only"] and _sr3 == ["anchored"] else "PROBLEM"
print(f"   {ok:>7}  sequence_report: 2-tuples -> {_sr2}, 3-tuples -> {_sr3}")
if ok == "PROBLEM":
    FAIL.append(("H sequence_report", f"2-tuple={_sr2} 3-tuple={_sr3}"))
print("   The anchor is reachable through the ordinary API, not only by calling it by name.")

print()
print("=" * 100)
print("SUMMARY: %d unexpected failures" % len(FAIL))
for n, e in FAIL:
    print("   FAIL", n, "->", e)
