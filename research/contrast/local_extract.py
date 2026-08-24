"""Phase 1 of the LOCAL / ADAPTIVE contrast-enhancement arm.

For every image that carries a correction.npy with BOTH classes, sample the
protocol's rows (<=8000 correction==1 + <=8000 correction==2, RandomState(0)),
and for each contrast arm: transform img01 -> rescale to [0,1] ->
compute_feature_stack -> sample the 17 features at those pixels.

Also samples 200_000 uniformly-random pixels from each of the 6 crack-free
specimens (pipeline.CLEAN_SPECIMENS) for the false-positive guardrail, and
measures each labelled frame's median crack half-width so the thin-frame
subset can be split out downstream.

Writes ONLY into research/contrast/. Nothing under app/, code/, models/ or
app_data/ is touched (all reads).

MEMORY NOTE. compute_feature_stack() materialises an (H, W, 17) float32 cube,
which is 2.2 GB on the 32 MP mosaics -- times 11 arms that is not a good idea.
sample_features() below computes the identical channels one at a time and
samples each immediately. It is verified bit-for-bit equal to
compute_feature_stack(img)[ys, xs, :] (see _selftest()), so this is a memory
optimisation and not a change of protocol.

RESCALE NOTE. Every arm's output is mapped into [0,1] by a per-image affine
min-max. That specific choice is close to free: every one of the 17 features is
an affine function of the image (gaussian smoothing is linear; gradient
magnitude, Laplacian and local std are homogeneous), so a per-image affine
rescale followed by the classifier's StandardScaler is very nearly a no-op.
img.npy is itself already per-image min-max normalised to [0,1], so for the
identity arm the rescale is exactly the identity map. One extra arm
(lcn_w51_robust) instead uses the project's own robust 1st-99th percentile
clip+stretch, purely to show that the rescale rule is not what drives the
result.
"""

import argparse
import json
import os
import sys
import time

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, morphology

import pipeline as P  # noqa: E402
import store as S  # noqa: E402
from txm_features import (  # noqa: E402
    GRADIENT_SIGMAS,
    LAPLACIAN_SIGMAS,
    N_FEATURES,
    SMOOTH_SIGMAS,
    TEXTURE_SIGMAS,
    compute_feature_stack,
    local_std,
    robust_normalize,
)

OUT = os.path.join(P0, "research", "contrast")
CACHE = os.path.join(OUT, "local_cache")
CLEAN_CACHE = os.path.join(OUT, "local_cache_clean")

MAX_PER_CLASS = 8000
CLEAN_SAMPLE = 200_000
LCN_EPS = 1e-3          # local std runs 0.011-0.25 on this data, so this is a
                        # genuine normalisation, not a smoothed difference
THIN_MAX_HALFWIDTH = 3.0


# ---------------------------------------------------------------- features

def sample_features(img01, ys, xs):
    """compute_feature_stack(img01)[ys, xs, :] without the (H,W,17) cube."""
    out = np.empty((len(ys), N_FEATURES), dtype=np.float32)
    i = 0
    out[:, i] = img01[ys, xs]
    i += 1
    for s in SMOOTH_SIGMAS:
        out[:, i] = ndi.gaussian_filter(img01, sigma=s)[ys, xs]
        i += 1
    for s in GRADIENT_SIGMAS:
        out[:, i] = ndi.gaussian_gradient_magnitude(img01, sigma=s)[ys, xs]
        i += 1
    for s in LAPLACIAN_SIGMAS:
        out[:, i] = ndi.gaussian_laplace(img01, sigma=s)[ys, xs]
        i += 1
    for s in TEXTURE_SIGMAS:
        out[:, i] = local_std(img01, sigma=s)[ys, xs]
        i += 1
    assert i == N_FEATURES
    return out


def _selftest():
    rng = np.random.RandomState(1)
    img = rng.rand(300, 400).astype(np.float32)
    ys = rng.randint(0, 300, 200)
    xs = rng.randint(0, 400, 200)
    a = compute_feature_stack(img)[ys, xs, :]
    b = sample_features(img, ys, xs)
    assert np.array_equal(a, b), np.abs(a - b).max()
    print("selftest: sample_features == compute_feature_stack (bit-exact)")


# -------------------------------------------------------------------- arms

def t_identity(img):
    return img


def t_clahe(img, clip, k):
    ks = (max(1, img.shape[0] // k), max(1, img.shape[1] // k))
    return exposure.equalize_adapthist(img, kernel_size=ks,
                                       clip_limit=clip).astype(np.float32)


def t_clahe_blend(img, clip, k, alpha):
    return ((1.0 - alpha) * img + alpha * t_clahe(img, clip, k)).astype(np.float32)


def t_lcn(img, w):
    mean = ndi.uniform_filter(img, size=w)
    msq = ndi.uniform_filter(img.astype(np.float64) ** 2, size=w)
    std = np.sqrt(np.clip(msq - mean.astype(np.float64) ** 2, 0, None))
    return ((img - mean) / (std + LCN_EPS)).astype(np.float32)


def t_unsharp(img, sigma, amount):
    return (img + amount * (img - ndi.gaussian_filter(img, sigma))).astype(np.float32)


def to01_minmax(a):
    lo, hi = float(np.min(a)), float(np.max(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


ARMS = {
    "identity":                 (t_identity, {}, to01_minmax),
    "clahe_c0.01_k8":           (t_clahe, dict(clip=0.01, k=8), to01_minmax),
    "clahe_c0.01_k16":          (t_clahe, dict(clip=0.01, k=16), to01_minmax),
    "clahe_c0.03_k8":           (t_clahe, dict(clip=0.03, k=8), to01_minmax),
    "clahe_c0.03_k16":          (t_clahe, dict(clip=0.03, k=16), to01_minmax),
    "lcn_w51":                  (t_lcn, dict(w=51), to01_minmax),
    "lcn_w151":                 (t_lcn, dict(w=151), to01_minmax),
    "lcn_w51_robust":           (t_lcn, dict(w=51), robust_normalize),
    "unsharp_s2_a1.0":          (t_unsharp, dict(sigma=2, amount=1.0), to01_minmax),
    "unsharp_s8_a1.5":          (t_unsharp, dict(sigma=8, amount=1.5), to01_minmax),
    # Own arm. The project's own flat-fielding result says the model leans hard
    # on ABSOLUTE large-radius intensity, so the reasoned way to add local
    # contrast is to add it on top of the original rather than in place of it:
    # a half-and-half blend keeps the DC term while still boosting a faint
    # crack against its own neighbourhood.
    "clahe_c0.01_k8_blend0.5":  (t_clahe_blend, dict(clip=0.01, k=8, alpha=0.5), to01_minmax),
}

ARM_ORDER = list(ARMS)


def apply_arm(name, img01):
    fn, kw, rescale = ARMS[name]
    return rescale(fn(img01, **kw))


# ------------------------------------------------------------- thin cracks

def crack_halfwidth(img01, corr):
    """Median crack half-width in px, per the protocol's thin-frame recipe."""
    m1 = corr == 1
    n1 = int(m1.sum())
    if n1 == 0:
        return None, 0
    thr = float(np.percentile(img01[m1], 20.0))
    core = m1 & (img01 < thr)
    kw = P._skimage_size_kw(morphology.remove_small_objects, 64)
    core = morphology.remove_small_objects(core, **kw)
    if not core.any():
        return None, 0
    dt = ndi.distance_transform_edt(core)
    sk = morphology.skeletonize(core)
    if not sk.any():
        return None, int(core.sum())
    return float(np.median(dt[sk])), int(core.sum())


# ------------------------------------------------------------------ driver

def pick(rng, flat_idx, cap):
    if flat_idx.size <= cap:
        return flat_idx
    return flat_idx[rng.choice(flat_idx.size, cap, replace=False)]


def labelled_images():
    out = []
    for m in S.list_images():
        iid = m["id"]
        c = S.load_npy(iid, "correction.npy", mmap=True)
        if c is None:
            continue
        c = np.asarray(c)
        if (c == 1).any() and (c == 2).any():
            out.append(iid)
    return sorted(out)


def clean_images():
    out = []
    for m in S.list_images():
        low = (m["id"] + " " + str(m.get("filename", ""))).lower()
        for cs in P.CLEAN_SPECIMENS:
            if cs.lower() in low:
                out.append((cs, m["id"]))
                break
    return sorted(out)


def do_labelled(iid, arms, force=False):
    dst = os.path.join(CACHE, iid + ".npz")
    meta_dst = os.path.join(CACHE, iid + ".json")
    have = set()
    if os.path.exists(dst) and not force:
        with np.load(dst) as z:
            have = {k[2:] for k in z.files if k.startswith("X_")}
    todo = [a for a in arms if a not in have]
    if not todo and os.path.exists(meta_dst):
        return json.load(open(meta_dst)), 0.0

    t0 = time.time()
    corr = np.asarray(S.load_npy(iid, "correction.npy"))
    img = np.asarray(S.load_npy(iid, "img.npy"), dtype=np.float32)
    h, w = img.shape

    rng = np.random.RandomState(0)
    i1 = pick(rng, np.flatnonzero(corr.ravel() == 1), MAX_PER_CLASS)
    i2 = pick(rng, np.flatnonzero(corr.ravel() == 2), MAX_PER_CLASS)
    flat = np.concatenate([i1, i2])
    y = np.concatenate([np.ones(i1.size, np.int8), np.zeros(i2.size, np.int8)])
    ys, xs = np.unravel_index(flat, (h, w))

    hw, core_px = crack_halfwidth(img, corr)

    store = {}
    if os.path.exists(dst) and not force:
        with np.load(dst) as z:
            store = {k: z[k] for k in z.files}
    store["y"] = y
    store["ys"] = ys.astype(np.int32)
    store["xs"] = xs.astype(np.int32)

    per_arm_time = {}
    for a in todo:
        ta = time.time()
        tr = apply_arm(a, img)
        store["X_" + a] = sample_features(tr, ys, xs)
        del tr
        per_arm_time[a] = round(time.time() - ta, 2)

    np.savez(dst, **store)
    meta = dict(id=iid, h=int(h), w=int(w), mp=round(h * w / 1e6, 2),
                n_crack_rows=int(i1.size), n_not_rows=int(i2.size),
                n_crack_px=int((corr == 1).sum()), n_not_px=int((corr == 2).sum()),
                median_halfwidth_px=hw, thin_core_px=core_px,
                thin=(hw is not None and hw <= THIN_MAX_HALFWIDTH),
                arm_seconds=per_arm_time)
    json.dump(meta, open(meta_dst, "w"), indent=1)
    return meta, time.time() - t0


def do_clean(cs, iid, arms, force=False):
    dst = os.path.join(CLEAN_CACHE, iid + ".npz")
    have = set()
    if os.path.exists(dst) and not force:
        with np.load(dst) as z:
            have = {k[2:] for k in z.files if k.startswith("X_")}
    todo = [a for a in arms if a not in have]
    if not todo:
        return 0.0

    t0 = time.time()
    img = np.asarray(S.load_npy(iid, "img.npy"), dtype=np.float32)
    h, w = img.shape
    rng = np.random.RandomState(0)
    flat = rng.choice(h * w, CLEAN_SAMPLE, replace=False)
    ys, xs = np.unravel_index(flat, (h, w))

    # Secondary (non-protocol) view: off-specimen background is 20-40% of these
    # frames and dilutes a rate computed over the whole frame, so record which
    # sampled pixels sit on the specimen and report both numbers.
    try:
        on_spec = P.specimen_support(img)[ys, xs]
    except Exception as e:                                   # noqa: BLE001
        print("   specimen_support failed:", e)
        on_spec = np.ones(len(ys), bool)

    store = {}
    if os.path.exists(dst) and not force:
        with np.load(dst) as z:
            store = {k: z[k] for k in z.files}
    store["ys"] = ys.astype(np.int32)
    store["xs"] = xs.astype(np.int32)
    store["on_specimen"] = on_spec
    for a in todo:
        tr = apply_arm(a, img)
        store["X_" + a] = sample_features(tr, ys, xs)
        del tr
    np.savez(dst, **store)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    _selftest()
    arms = ARM_ORDER if a.arms == "all" else [x for x in a.arms.split(",") if x in ARMS]
    print("arms:", arms)

    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(CLEAN_CACHE, exist_ok=True)

    ids = labelled_images()
    if a.limit:
        ids = ids[:a.limit]
    print(f"{len(ids)} labelled images")

    t_start = time.time()
    metas = []
    for k, iid in enumerate(ids, 1):
        m, dt = do_labelled(iid, arms, force=a.force)
        metas.append(m)
        print(f"[{k}/{len(ids)}] {m['mp']:5.1f}MP {dt:6.1f}s hw="
              f"{m['median_halfwidth_px']} thin={m['thin']} {iid[:52]}", flush=True)

    cleans = clean_images()
    print(f"{len(cleans)} crack-free specimens")
    for k, (cs, iid) in enumerate(cleans, 1):
        dt = do_clean(cs, iid, arms, force=a.force)
        print(f"[clean {k}/{len(cleans)}] {dt:6.1f}s {cs}", flush=True)

    manifest = dict(
        arms=arms, arm_params={k: {kk: vv for kk, vv in ARMS[k][1].items()} for k in arms},
        rescale={k: ARMS[k][2].__name__ for k in arms},
        lcn_eps=LCN_EPS, max_per_class=MAX_PER_CLASS, clean_sample=CLEAN_SAMPLE,
        thin_max_halfwidth=THIN_MAX_HALFWIDTH,
        images=metas, clean=[dict(specimen=c, id=i) for c, i in cleans],
        extract_seconds=round(time.time() - t_start, 1))
    json.dump(manifest, open(os.path.join(OUT, "local_manifest.json"), "w"), indent=1)
    print("total extract %.1f s" % (time.time() - t_start))
    thin = [m["id"] for m in metas if m["thin"]]
    print(f"thin frames: {len(thin)}/{len(metas)}")
    for t in thin:
        print("  ", t)


if __name__ == "__main__":
    main()
