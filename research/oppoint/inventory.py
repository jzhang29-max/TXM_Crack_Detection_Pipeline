"""Inventory: which images have prob.npy, from WHICH model, and what labels exist.

Read-only. Writes only research/oppoint/inventory.json.
"""
import sys, os, json
import numpy as np

P0 = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"
sys.path.insert(0, os.path.join(P0, "app", "core"))
sys.path.insert(0, os.path.join(P0, "code"))
import store as S, pipeline as P, model as M

OUT = os.path.join(P0, "research", "oppoint")

reg = S.registry()
cur = reg.get("current")
cur_key = S.model_key(cur)
print("current model entry:", json.dumps(cur, indent=2, default=str))
print("current model_key:", cur_key)

rows = []
for m in S.list_images():
    iid = m["id"] if isinstance(m, dict) else m
    meta = S.read_meta(iid)
    fn = (meta.get("filename") or "")
    mk = meta.get("model_key")
    has_prob = os.path.exists(S.path(iid, "prob.npy"))
    cached = []
    pd = S.path(iid, "probs")
    if os.path.isdir(pd):
        cached = sorted(f[:-4] for f in os.listdir(pd) if f.endswith(".npy"))
    corr = S.load_npy(iid, "correction.npy", mmap=True)
    n1 = n2 = 0
    shape = None
    if corr is not None:
        corr = np.asarray(corr)
        shape = list(corr.shape)
        n1 = int((corr == 1).sum())
        n2 = int((corr == 2).sum())
    pshape = None
    if has_prob:
        pr = S.load_npy(iid, "prob.npy", mmap=True)
        if pr is not None:
            pshape = list(np.asarray(pr).shape)
    clean = any(k.lower() in fn.lower() for k in P.CLEAN_SPECIMENS)
    rows.append(dict(iid=iid, filename=fn, model_key=mk, has_prob=has_prob,
                     stale=(has_prob and mk != cur_key), cached=cached,
                     n_crack=n1, n_not=n2, corr_shape=shape, prob_shape=pshape,
                     clean=clean))

json.dump(dict(current_key=cur_key, current=cur, rows=rows),
          open(os.path.join(OUT, "inventory.json"), "w"), indent=2, default=str)

nprob = sum(r["has_prob"] for r in rows)
nstale = sum(r["stale"] for r in rows)
nlab = sum(1 for r in rows if r["n_crack"] or r["n_not"])
print(f"\nimages={len(rows)} with_prob={nprob} stale_prob={nstale} labelled={nlab}")
print(f"clean_specimen_images={sum(r['clean'] for r in rows)}")
print("\nmodel_key histogram:")
from collections import Counter
for k, v in Counter(r["model_key"] for r in rows).most_common():
    print(f"  {k!r}: {v}")
print("\nSTALE (prob.npy not from current model):")
for r in rows:
    if r["stale"]:
        print(f"  {r['iid'][:60]} mk={r['model_key']} cached={r['cached']}")
print("\nlabelled images (crack px, notcrack px):")
for r in sorted(rows, key=lambda r: -(r["n_crack"] + r["n_not"])):
    if r["n_crack"] or r["n_not"]:
        print(f"  {r['n_crack']:>9} {r['n_not']:>9}  clean={int(r['clean'])} "
              f"prob={int(r['has_prob'])} {r['filename'][:60]}")
