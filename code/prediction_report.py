#!/usr/bin/env python3
"""Report exactly what the model did to one ingested image, and in what environment.

WHY THIS EXISTS. CI ingested one real frame with the full ensemble on a macos-26-arm64
runner and recorded a predicted area of 9.25%, where this project's development machine
records 18.80% for the same frame, the same model files and the same torch version. Same
code, same input, twice the area. MPS versus CPU was ruled out first -- forcing CPU locally
reproduces 18.7972% exactly, with zero pixels flipping across the 0.60 threshold -- so the
cause is elsewhere and was not diagnosable from the workflow log, which printed a single
number.

So this prints the intermediate quantities instead of the conclusion: image shape and
checksum, the raw probability distribution, the area before and after speck pruning, and the
component count. Whichever of those first disagrees between two machines is where the
divergence lives. A difference in the checksum means the input differs; in the probability
mean means the model or its features differ; only in the pruned area means the mask is
fragmenting differently.

Also useful outside CI: run it when someone reports a mask that looks wrong, and compare.

    python3 code/prediction_report.py                 # every ingested image
    python3 code/prediction_report.py <image_id>      # one
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "app", "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline as P                                            # noqa: E402
import store as S                                               # noqa: E402


def environment():
    """The versions that could plausibly move a prediction, and the compute device."""
    import importlib
    print("environment")
    print(f"  python           {sys.version.split()[0]}  {sys.platform}")
    for mod in ("numpy", "scipy", "sklearn", "skimage", "tifffile", "imagecodecs",
                "joblib", "PIL", "torch", "transformers"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<16} {getattr(m, '__version__', '?')}")
        except Exception as e:                                   # noqa: BLE001
            print(f"  {mod:<16} absent ({type(e).__name__})")
    try:
        import torch
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"  SAM would use    {dev}")
    except Exception:                                            # noqa: BLE001
        print("  SAM would use    n/a (no torch)")
    print(f"  threads          OMP={os.environ.get('OMP_NUM_THREADS', 'unset')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS', 'unset')}")
    print(f"  TXM_NO_SAM       {os.environ.get('TXM_NO_SAM', 'unset')}")


def report(image_id):
    from skimage.measure import label

    meta = S.read_meta(image_id)
    img = S.load_npy(image_id, "img.npy")
    prob = S.load_npy(image_id, "prob.npy")
    print(f"\n{meta.get('filename') or image_id}")
    print(f"  model            {meta.get('model')}")
    print(f"  model_key        {meta.get('model_key')}")
    print(f"  recorded area    {meta.get('predicted_area')}")
    if img is None or prob is None:
        print("  (no img.npy / prob.npy -- not ingested)")
        return

    img = np.asarray(img, np.float32)
    prob = np.asarray(prob, np.float32)
    # Checksum the INPUT, so a divergence in the decoded image is distinguishable from a
    # divergence in the model. Rounded first: a bit-level hash of float32 would differ on
    # any last-bit change and could not tell "different image" from "different arithmetic".
    h = hashlib.sha1(np.round(img.astype(np.float64), 6).tobytes()).hexdigest()[:16]
    print(f"  image            {img.shape} {img.dtype}  sha1(round 6)={h}")
    print(f"  image stats      min {img.min():.6f}  max {img.max():.6f}  "
          f"mean {img.mean():.6f}  std {img.std():.6f}")
    print(f"  prob stats       min {prob.min():.6f}  max {prob.max():.6f}  "
          f"mean {prob.mean():.6f}  std {prob.std():.6f}")

    thr = P.DEFAULT_THRESHOLD
    raw = prob > thr
    pruned = P.prune_specks(raw)
    n_raw = int(label(raw, connectivity=2).max())
    n_pruned = int(label(pruned, connectivity=2).max())
    print(f"  threshold        {thr}")
    print(f"  area raw         {raw.mean():.6f}   components {n_raw}")
    print(f"  area pruned      {pruned.mean():.6f}   components {n_pruned}   "
          f"(MIN_BLOB_PX={P.MIN_BLOB_PX})")
    print(f"  pruning removed  {(raw.mean() - pruned.mean()):.6f} "
          f"({100 * (1 - pruned.mean() / max(raw.mean(), 1e-12)):.1f}% of the raw area)")
    # Where the probability mass sits relative to the cut, which is what decides whether a
    # small shift in prob can move the area a lot.
    for lo, hi in ((0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70)):
        print(f"  prob in [{lo:.2f},{hi:.2f})  {float(((prob >= lo) & (prob < hi)).mean()):.6f}")


def main():
    environment()
    ids = sys.argv[1:] or [m["id"] for m in S.list_images()
                           if "SELFTEST" not in (m.get("filename") or "")]
    if not ids:
        print("\nno ingested images")
        return
    for iid in ids:
        report(iid)


if __name__ == "__main__":
    main()
