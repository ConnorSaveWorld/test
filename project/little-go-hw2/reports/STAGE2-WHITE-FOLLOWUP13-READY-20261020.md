# Stage 2 White follow-up counterwall candidate — 2026-10-20

## Candidate

New SHA-256: `cefadb4f805e5d95e2f157a5b50619e476ed05dea47bef36e6ac1a0a56661698`.

The candidate keeps the tested White reply17 after the exact opening `B12,W6,B7`, then plays point13 only after the exact seven-ply sequence `B12,W6,B7,W17,B18,W16,B11`. All other positions use the existing search. The condition comes from the local pressure-profile loss trace; it does not affect a nearby sequence such as the same line ending in Black10.

## Local evidence

- CPython 3.7.5 source-parametric rule, protocol and search suite: **27/27 passed**.
- Matched D4 screen: 8 orientations against each of four local profiles, 200,000 White and 48,000 Black nodes per move. The follow-up candidate won **32/32** with no illegal moves, timeouts or crashes. Reply17-only won31/32; the new branch changed its one pressure-profile loss into a win. Aggregate White margin stayed+72.
- Paired against the currently delivered reply18 candidate on the same seeds: both won32/32; new candidate gained+17 aggregate White margin (baseline0, pressure+2, space0, wall3+15), with no proxy win-count regressions.
- Fresh official-file-host checks against Random: one game as Black and one as White, both won; every move passed the course GO legality check. Maximum agent move wall time8.59s; reported CPU was about51.5s/game. This small Random sample checks protocol and timing, not Championship strength.

These profiles are local proxies, not Vocareum's hidden Championship agent. No official full-score claim is made.

## Packaging and provenance

The source was installed as `my_player3.py` in Downloads, its candidate and extracted folders, and the project candidate folder. All four source copies match SHA-256 `cefadb4f...`. The Downloads and project ZIPs have SHA-256 `26d7fc73a315255f213c9bea1721700b3a8eb1921896d53d73112fd8d657d1e2`; the ZIP round-trip was verified and contains only `my_player3.py`.

Previous copies were backed up in `reports/delivery-backups/20261020-second-reply17-followup-prepromotion/`. Stage1 and `sources/` were not changed.

The latest official result remains submission9, score6.2. Its recorded submitted source hash is `4edca990...`, different from the prior staged candidate `122a9c0b...`; therefore that result does not measure this candidate. The new candidate still needs the user's next Vocareum run and report.
