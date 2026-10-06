# Evaluation evidence

Every `den evaluate` writes here, one folder per run (`runs/kev-recipe/4-skills` -> `kev-recipe/4-skills/`,
`release:v1` -> `release-v1/`): per evaluated file, `<split>/<file>[+augmentation]/report.json` (every metric) and
`rows.json` (each question's calibrated probabilities, label and the links Kev's checks pair rows by), plus one
`provenance.json` (the run, its base, LoRA, head and temperature, the commits it was trained and evaluated at, and the
sha256 of every scored file). No weights: commit it, so results outlive the machine that produced them.

`den compare --paired A B` compares two runs question by question with 95% intervals and saves
`comparison-vs-<A>.json` in B's folder; `den release` records the comparison with the parent version.
`reports/claims.json` ties every number printed in the docs to a file here (`make check` verifies them).
