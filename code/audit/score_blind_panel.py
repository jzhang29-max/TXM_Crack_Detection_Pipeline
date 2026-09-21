#!/usr/bin/env python3
"""Unblind a blind-panel run and compute every statistic the docs quote from it.

Takes the crop key (which class each field really was) and the panel votes, and emits the
contingency table, the instrument-validation test on the controls, the test-class comparison,
per-panel breakdowns, and -- the part run 1 got wrong -- a check for CONFOUNDING between the
test class and the controls on specimen group.

Usage:  python3 code/audit/score_blind_panel.py <crop_key.json> <votes.json> [out.json]

  crop_key.json : {"crops": [{field, cls, iid, px, grp, ...}]}  or a bare list of those
  votes.json    : {"byField": {"<field>": [{panel, crack, confidence, evidence}, ...]}}

The VOTES themselves are produced by a blind multi-agent panel, which this script does not
run. That is a real reproducibility limit and it is stated in code/audit/README.md rather
than hidden: the scoring is deterministic and re-runnable, the votes are not.
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

SCORE = {"yes": 1.0, "unsure": 0.5, "no": 0.0}


def load_key(path):
    d = json.load(open(path))
    return d["crops"] if isinstance(d, dict) and "crops" in d else d


def main(keyfile, votefile, outfile=None):
    key = {k["field"]: k for k in load_key(keyfile)}
    votes = json.load(open(votefile))["byField"]
    rows = []
    for fld, k in key.items():
        vs = votes.get(fld, [])
        if not vs:
            continue
        rows.append(dict(field=fld, cls=k["cls"], iid=k.get("iid"), px=k.get("px"),
                         grp=k.get("grp"), n=len(vs),
                         s=float(np.mean([SCORE.get(v["crack"], 0.0) for v in vs])),
                         yes=sum(1 for v in vs if v["crack"] == "yes"),
                         no=sum(1 for v in vs if v["crack"] == "no"),
                         unsure=sum(1 for v in vs if v["crack"] == "unsure"),
                         panels={v["panel"]: v["crack"] for v in vs}))
    g = lambda c: [r for r in rows if r["cls"] == c]                   # noqa: E731
    maj = lambda rs: sum(1 for r in rs if r["yes"] >= 2)               # noqa: E731
    out = dict(n_fields=len(rows), votes_per_field=dict(Counter(r["n"] for r in rows)))

    print(f"fields {len(rows)}   votes/field {out['votes_per_field']}")
    print(f"\n{'class':7s} {'n':>3s} {'score':>7s} {'majority YES':>14s}")
    for c in sorted({r["cls"] for r in rows}):
        rs = g(c)
        out.setdefault("by_class", {})[c] = dict(
            n=len(rs), score=float(np.mean([r["s"] for r in rs])),
            majority_yes=maj(rs))
        print(f"{c:7s} {len(rs):3d} {np.mean([r['s'] for r in rs]):7.3f} "
              f"{maj(rs):5d} ({100*maj(rs)/len(rs):3.0f}%)")

    P, T, M = g("POSS") or g("POS"), g("TEST") or g("UNLAB"), g("NEGM")
    if P and M:
        a, c_ = maj(P), maj(M)
        p = fisher_exact([[a, len(P) - a], [c_, len(M) - c_]], alternative="greater")[1]
        out["instrument"] = dict(pos=f"{a}/{len(P)}", neg=f"{c_}/{len(M)}", fisher_p=float(p),
                                 validated=bool(p < 0.05))
        print(f"\nINSTRUMENT: control {a}/{len(P)} vs matrix {c_}/{len(M)}  Fisher p={p:.3g}  "
              f"{'VALIDATED' if p < 0.05 else 'NOT VALIDATED -- nothing below counts'}")
    if T and P and M:
        t = maj(T)
        out["test"] = dict(
            majority_yes=f"{t}/{len(T)}", rate=float(t / len(T)),
            vs_control_p=float(mannwhitneyu([r["s"] for r in T], [r["s"] for r in P]).pvalue),
            vs_matrix_p=float(mannwhitneyu([r["s"] for r in T], [r["s"] for r in M]).pvalue))
        print(f"TEST: {t}/{len(T)} = {100*t/len(T):.0f}%   vs control p="
              f"{out['test']['vs_control_p']:.3g}   vs matrix p={out['test']['vs_matrix_p']:.3g}")

    # CONFOUND CHECK. Run 1 of this design matched on size and not on specimen group, and
    # standardising the control to the test group mix moved its yes-rate 0.562 -> 0.313 --
    # which was the entire reported effect. Always report both.
    if T and P and all(r.get("grp") for r in T + P):
        tg, pg = Counter(r["grp"] for r in T), Counter(r["grp"] for r in P)
        out["group_mix"] = dict(test=dict(tg), control=dict(pg), matched=bool(tg == pg))
        print(f"\nGROUP MIX  test {dict(tg)}   control {dict(pg)}   "
              f"{'MATCHED' if tg == pg else 'NOT MATCHED -- standardising below'}")
        strata = defaultdict(dict)
        for name, rs in (("test", T), ("control", P)):
            for grp in set(r["grp"] for r in rs):
                sub = [r for r in rs if r["grp"] == grp]
                strata[grp][name] = (maj(sub), len(sub))
        tot = sum(tg.values())
        std = sum((strata[grp]["control"][0] / strata[grp]["control"][1]) * (tg[grp] / tot)
                  for grp in tg if strata.get(grp, {}).get("control", (0, 0))[1])
        out["control_standardised_to_test_mix"] = float(std)
        out["control_raw"] = float(maj(P) / len(P))
        print(f"  control yes-rate raw {maj(P)/len(P):.3f}   standardised to test mix {std:.3f}"
              f"   test {maj(T)/len(T):.3f}")
        for grp in sorted(strata):
            s = strata[grp]
            print(f"    {grp}: control {s.get('control', ('-', 0))[0]}/{s.get('control', ('-', 0))[1]}"
                  f"   test {s.get('test', ('-', 0))[0]}/{s.get('test', ('-', 0))[1]}")

    for pn in sorted({p for r in rows for p in r["panels"]}):
        line = []
        for c in sorted({r["cls"] for r in rows}):
            v = [r["panels"].get(pn) for r in g(c) if pn in r["panels"]]
            line.append(f"{c} {100*sum(1 for x in v if x == 'yes')/max(len(v), 1):3.0f}%")
        out.setdefault("per_panel", {})[pn] = line
        print(f"  {pn}: " + "   ".join(line))

    out["per_field"] = rows
    if outfile:
        json.dump(out, open(outfile, "w"), indent=1, default=str)
        print(f"\nwrote {outfile}")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:4])
