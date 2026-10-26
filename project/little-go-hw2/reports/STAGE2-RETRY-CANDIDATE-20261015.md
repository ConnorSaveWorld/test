# Stage2 retry candidate — 2026-10-15

## Change

The previous Stage2 candidate limited itself to 3 CPU seconds per move and reserved only 65 CPU seconds per game. The revised candidate allocates up to 7 CPU seconds per move and up to 82 CPU seconds per game, spreading remaining time across the agent's remaining turns. It leaves 8 seconds below the assignment's approximate 90-second game allowance and caps wall time at 9 seconds per move. The state/protocol/rules/search logic and Stage1 delivery were not changed.

This targets the observed Stage2 gap (Q-learning 10/10, Championship 0/10) by using more of the available search time against the stronger opponent. It is a plausible improvement, not a measured fix: the official Championship is unavailable locally, and this candidate has not yet been run against it. An earlier fixed 5-second-budget variant scored 34/64 versus 36/64 for the baseline on a small, forced-opening proxy pool, so more search time alone is not proven to improve win rate.

## Verification

- New budget tests: 3 passed; they were first run before the helper existed and failed as expected.
- Candidate rule/protocol/search suite: 27 passed.
- Full existing local suite: 63 passed, 1 platform-specific test skipped.
- Candidate transitions: 25,000 checked against both the independent matrix rules and the course GO; 0 mismatches.
- Windows Python 3.13 syntax compile passed; source, Downloads copy, and the single-file ZIP member have identical SHA-256: `9eae2f97961f95ea4d3ec3c34bee4478fffbc44b2eccf125492a9d205d82c791`.

## Files

- Stage2 source: `stage2_candidate/my_player3.py`
- Upload-ready copy: `C:\Users\paul2\Downloads\my_player3.py`
- One-file archive: `C:\Users\paul2\Downloads\little-go-stage2-candidate.zip`
- Previous failed Stage2 copy preserved as `C:\Users\paul2\Downloads\my_player3_stage2_20261015_failed.py`
- Stage1 file and archive remain unchanged.
