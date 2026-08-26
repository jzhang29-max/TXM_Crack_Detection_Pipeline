# Research scripts

One-off scripts from the development phase: dataset builders, experiment runners,
comparison harnesses, figure generators, and the two paint tools (`paint_server.py`,
`paint_server_hybrid.py`) that the `app/` in this repo replaced.

**None of this is needed to run the app.** They were moved here out of `code/` because a
future reader opening `code/` should see the nine modules the product actually uses, not
sixty-six files of which two are superseded copies of the tool they are looking at.

They were written to run from the repo root with `code/` on the import path, so they now
need it supplied:

    PYTHONPATH=code python3 research/code/<script>.py

That fixes the IMPORTS. It does not fix the DATA PATHS, and most of these scripts also need
those supplied. They compute their project root as `dirname(__file__)/..`, which pointed at the
repo root when they lived in `code/` and now points at `research/` -- so `paint_common.py` and
the twelve other files that do this look for `research/paint/corrections`,
`research/dataset_cache` and `research/models`, none of which exist. Nothing raises: the
directories are created empty on import, so a script runs, finds zero labels, and reports
nothing wrong.

They are kept as the provenance of the numbers in `docs/`, not as a runnable pipeline, so this
is documented rather than repaired -- rewriting the path resolution in forty archived scripts
would change what they do without anyone re-running the analyses that cite them. If you need to
re-run one, point it at the real directories explicitly and check it found your labels before
believing its output.

Some also expect caches or result directories that are gitignored and were regenerated
locally; treat them as a record of what was done rather than as a runnable suite.
