# Little-Go optimized submission ready — 2026-10-13

User resumed paused work with “继续开”. Local optimization, complete default-budget tournaments, final validation and packaging are complete. This checkpoint records the delivered version; the previous paused checkpoint and baseline remain preserved.

## Delivered files

- submission/my_player3.py: 21,951 bytes, SHA256 dda83b2ac22036ea4bdc49418193aa38893fe4f1f6c64d06494e2a45156bff36.
- little-go-submission.zip: 5,782 bytes; contains only my_player3.py, byte-identical to submission.
- experiments/candidate_pvs.py is the frozen identical tested source. experiments/baseline_player3.py and candidate_fast.py remain historical comparisons.
- reports/OPTIMIZATION.md and STAGE1-pvs-resume-heldout.md contain the current outcome; old RESULTS.md / STAGE1.md describe the baseline.

## Completed validation

Actual CPython3.7.5: 47 tests passed after promotion. Internal optimized rules:25,000 comparisons, zero mismatches versus independent matrix rules and original GO. Same-seed Alpha proxy20/20 (baseline16/20, bitboards19/20). Held-out seed762000: original Random19/20, GreedyProxy20/20, AggressiveProxy20/20, AlphaProxy20/20, black/white10 games each. All100 full games and2400 moves replayed using original GO; zero invalid actions, timeouts, crashes. Held-out80 CPU1866.275s, maxmove3.145s, maxgame25.824s. Local Stage1 formula90/90, not an official grade.

Final original file-host deployment probe:2 complete games, both colors won, default budget, fresh process every turn. These unseeded games verify interfaces rather than estimate win rate. Final ZIP membership and extracted bytes verified.

## Boundaries and remaining external work

No Vocareum submission was performed. Formal Alpha-beta, Q-learning, Championship opponents and actual grading CPU wrapper remain unavailable. No 100/100 guarantee. User can upload only submission/my_player3.py and run the official evaluation; capture that report before attributing official scores or making further changes.

No Git repository, commits, external messages, or changes to synced sources/original downloads. The CPython runtime remains in .runtime/Python-3.7.5/python.exe (WSL ELF with .exe suffix). The paused backup archive excludes runtime, caches and backup archives.

## Restart guidance

Read this checkpoint and reports/OPTIMIZATION.md. Do not restore the old baseline over the delivered PVS version. Do not merge old interrupted progress counts into complete results. The Stage1 tool now atomically checkpoints full completed traces with source hash and seed for future long runs. All resumed tournament workers and final deployment probe finished normally.
