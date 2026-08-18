"""
Harvest results from ANY workflow run's journal, including runs that are still
in flight or that died partway.

Why this exists: a large agent workflow can lose most of its agents to session
limits (one run here lost 51 of 92). Every agent that DID finish has already
had its return value appended to journal.jsonl, so nothing is actually lost --
but only if it gets pulled out and committed. This script does that, so partial
progress survives a session ending mid-run and never needs re-running.

Two ways progress is preserved:
  1. THIS SCRIPT -- read the journal at any time, even mid-run, and write the
     completed results to results/. Safe to run repeatedly.
  2. WORKFLOW RESUME -- re-invoking a workflow with the same scriptPath and
     resumeFromRunId replays every completed agent() call from cache
     instantly; only failed or newly-added calls actually execute. Same script
     + same args means a 100% cache hit.

Usage:
    python3 harvest_workflow_results.py                  # newest run
    python3 harvest_workflow_results.py --run wf_xxxxx   # a specific run
    python3 harvest_workflow_results.py --list           # what runs exist
"""
import argparse, glob, json, os, sys

WF_ROOT = os.path.expanduser(
    "~/.claude/projects/-Users-jiamingzhang-Desktop-APP/"
    "ca4727e5-ac25-49c5-94b5-b5cabff138cc/subagents/workflows")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "harvested")


def runs():
    out = []
    for d in glob.glob(os.path.join(WF_ROOT, "wf_*")):
        jp = os.path.join(d, "journal.jsonl")
        if os.path.exists(jp):
            out.append((os.path.getmtime(jp), os.path.basename(d), jp))
    return sorted(out, reverse=True)


def harvest(jp):
    """Every {"type":"result"} line holds one finished agent's return value.
    Note the key is 'result', not 'value' -- getting that wrong silently
    yields zero findings, which cost time once already."""
    started, results = 0, []
    for line in open(jp):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        t = rec.get("type")
        if t == "started":
            started += 1
        elif t == "result":
            r = rec.get("result")
            if r is not None:
                results.append(r)
    return started, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    R = runs()
    if args.list or not R:
        for mt, name, jp in R:
            s, res = harvest(jp)
            print(f"  {name}  started={s:4d}  completed={len(res):4d}")
        if not R:
            print("no workflow runs found")
        return

    if args.run:
        match = [x for x in R if args.run in x[1]]
        if not match:
            print(f"no run matching {args.run}"); sys.exit(1)
        mt, name, jp = match[0]
    else:
        mt, name, jp = R[0]

    started, results = harvest(jp)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(OUT_DIR, f"{name}.json")

    # Split structured findings from bookkeeping returns so the useful payload
    # is easy to consume downstream.
    findings = [r for r in results if isinstance(r, dict) and (
        "fraction_of_red_that_is_false_positive" in r or "overall_verdict" in r
        or "findings" in r or "agree_fraction_false_positive" in r)]
    json.dump({"run": name, "agents_started": started, "agents_completed": len(results),
               "findings": findings, "all_results": results},
              open(out, "w"), indent=1, default=str)
    print(f"{name}: {started} started, {len(results)} completed, {len(findings)} structured findings")
    print(f"saved -> {os.path.relpath(out)}")

    if findings and "fraction_of_red_that_is_false_positive" in findings[0]:
        fps = sorted(f["fraction_of_red_that_is_false_positive"] for f in findings
                     if isinstance(f.get("fraction_of_red_that_is_false_positive"), (int, float)))
        if fps:
            print(f"median false-positive fraction so far: {fps[len(fps)//2]:.2f}  (n={len(fps)})")
        wedge = [f for f in findings if f.get("wedge_rim_still_traced") is not None]
        if wedge:
            n = sum(1 for f in wedge if f["wedge_rim_still_traced"])
            print(f"wedge rim still traced: {n}/{len(wedge)}")


if __name__ == "__main__":
    main()
