# Releases

One JSON record per released version (`v1.json`, `v2.json`, ...), written by `den release create` (`make release`)
and committed. A record says what the version is and where it lives: the Hub repo and the commit its version tag
points at, its parent version, the base, LoRA and head, every training stage's data (each file's sha256 and the lines
used), the data repo commit it was trained from, dev and locked-test results, and every model file's sha256.

Records are immutable: a new model is a new version. `release:<version>` works wherever a run does
(`den train --init-from release:v1`, `den evaluate --run release:v1`, `den serve --run release:v1`).
