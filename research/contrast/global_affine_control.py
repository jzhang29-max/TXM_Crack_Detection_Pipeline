"""Control for the GLOBAL contrast study: how much of a z-space difference does
a transform the model provably CANNOT see still produce?

Any affine map x -> a*x + b (a > 0) is absorbed exactly by StandardScaler:
each of the 17 features is affine in its untransformed counterpart -- intensity
and the 6 Gaussian smooths pick up (a, b); the 4 gradient magnitudes, 4
Laplacians and 2 local-stds pick up (a, 0) since they annihilate constants --
and StandardScaler standardises each column independently. So an affine arm
must score identically, and "unchanged" is the correct answer, not a bug.

In float32 it is not *bit* identical, and this script measures the floor. The
answer matters for reading the main table: `stretch_1_99` shows a non-zero
max|z - z_identity| and that number needs a scale to be judged against.

Writes research/contrast/global_affine_control.json.
"""
import json
import os
import sys

import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
sys.path.insert(0, os.path.join(P0, "research", "contrast"))

import store as S  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from txm_features import FEATURE_NAMES  # noqa: E402
import global_contrast_arms as G  # noqa: E402

N = 60_000


def main():
    ims = S.list_images()
    ims.sort(key=lambda m: m["width"] * m["height"])
    m = next(x for x in ims
             if S.load_npy(x["id"], "correction.npy", mmap=True) is not None)
    img = np.asarray(S.load_npy(m["id"], "img.npy"), np.float32)
    rng = np.random.RandomState(1)
    idx = rng.choice(img.size, N, replace=False)
    z0 = StandardScaler().fit_transform(G.featurize_at(img, idx).astype(np.float64))

    cases = {
        # provably affine -> the model cannot see it. This is the noise floor.
        "exact_affine_0.5x+0.25": np.clip(0.5 * img + 0.25, 0, 1).astype(np.float32),
        "exact_affine_0.9x+0.05": np.clip(0.9 * img + 0.05, 0, 1).astype(np.float32),
        # the study arms, for scale
        "stretch_1_99": G.ARMS["stretch_1_99"](img),
        "gamma_2.0": G.ARMS["gamma_2.0"](img),
        "equalize_hist": G.ARMS["equalize_hist"](img),
    }
    out = {"frame": m["filename"], "n_sample_px": N, "cases": {}}
    for k, x in cases.items():
        z = StandardScaler().fit_transform(
            G.featurize_at(np.ascontiguousarray(x, np.float32), idx).astype(np.float64))
        d = np.abs(z - z0).max(axis=0)
        out["cases"][k] = dict(
            max_abs_z_diff=float(d.max()),
            per_feature={FEATURE_NAMES[i]: float(d[i]) for i in np.argsort(-d)[:4]},
        )
        print(f"{k:24s} max|z-z0| = {d.max():.4g}   "
              + ", ".join(f"{FEATURE_NAMES[i]}={d[i]:.2g}" for i in np.argsort(-d)[:3]))

    floor = max(out["cases"][k]["max_abs_z_diff"]
                for k in ("exact_affine_0.5x+0.25", "exact_affine_0.9x+0.05"))
    out["float32_noise_floor_max_abs_z_diff"] = floor
    out["note"] = (
        "The floor is set by local_std (the texture_s* features): it evaluates "
        "sqrt(E[x^2] - E[x]^2), and that cancellation loses precision differently "
        "once x is rescaled. A provably-invisible affine map still moves z by "
        f"~{floor:.3g}, so any arm at that level is a no-op, while gamma_2.0 at "
        f"{out['cases']['gamma_2.0']['max_abs_z_diff']:.3g} is a genuine change.")
    with open(os.path.join(P0, "research", "contrast",
                           "global_affine_control.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nfloat32 noise floor: {floor:.4g}")
    print("wrote research/contrast/global_affine_control.json")


if __name__ == "__main__":
    main()
