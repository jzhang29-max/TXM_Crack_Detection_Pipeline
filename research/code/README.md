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

Some also expect caches or result directories that are gitignored and were regenerated
locally; treat them as a record of what was done rather than as a runnable suite.
