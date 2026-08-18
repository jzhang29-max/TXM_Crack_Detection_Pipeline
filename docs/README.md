# How the model was arrived at

None of this is needed to run the app -- start with the README one level up.

- **HANDOFF.md** — the development record. Includes the label inventory (which labels
  exist, who made them, and how much to trust each source) and four approaches that were
  adopted and then reverted as regressions: flat-fielding as model input, geometric
  masking, a curvilinearity gate, and algorithmic crack labels. Kept so they are not
  retried, and because the reasoning behind the metric rules lives here: an
  over-aggressive filter and a good one both reduce predicted area, and only recall
  against ground truth separates them.
- **SAM_COMPARISON.md** — the full study behind the model choice: zero-shot Segment
  Anything measured against the deployed classifier, with verified citations. Answers
  "why not just use SAM?"
- **APP_COMPARISON.md** — how this app compares to the sibling SEM pipeline's, what was
  borrowed in each direction, and the layout both now share.
- **MARKUP_GUIDE.md**, **WORKLIST.md**, **START_HERE.md** — earlier working notes.
