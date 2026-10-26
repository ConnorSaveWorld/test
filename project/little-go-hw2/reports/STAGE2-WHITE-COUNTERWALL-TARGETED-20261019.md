# Stage 2 White counterwall candidate — 2026-10-19

## Starting evidence

Vocareum still shows submission 8 (submitted Oct 18, 2026 at 6:15:07 pm PDT; report executed 6:37:27 pm PDT): Q-learning 10/10, Championship Black 5/5 and White 0/5, final score 6. The browser report was read-only checked on Oct 19; no newer submission exists. The uploaded file is still the prior `4edca990...` source.

The five recorded Championship games all used White's early reply at point 18. Four traces began with Black 12, White 6, Black 11; the fifth began Black 12, White 6, Black 7. The exact hidden opponent remains unavailable locally, so this is a targeted hypothesis, not a reproduced official win.

## Candidate change

In the exact three-ply center approach position, answer the recorded Black-11 approach with point 13. Preserve the prior point-18 reply after Black 7 and use the existing search everywhere else. The branch checks the full previous/current boards, player, move count and pass state; the selected point is checked for legality before it is returned.

## Validation

- Test-driven behavior checks first failed against the old source (Black 11 still returned 18), then passed after the isolated change. Both targeted replies are legal.
- Full CPython 3.7.5 test suite: 97 tests passed.
- Paired local screen: 8 D4 opening orientations against each of four local search profiles, at 200,000 White nodes and 48,000 Black nodes per move. Both old and targeted candidates won all 32 games; no illegal, timeout or crash results. The targeted source had no win-rate regressions. Its aggregate White margin improved by 7 points over the 32 games: 0 for baseline, +2 across pressure, 0 for space, and +5 across the wall-building profile. These agents are local proxies, not Vocareum's Championship player.
- Actual CPython 3.7.5 fresh-process file-host check: one game as Black and one as White, both won against Random. Each output was checked with the course GO implementation. Maximum agent-reported CPU was 52.15 seconds/game; maximum single-move wall time was 8.47 seconds. This small sample validates the interface and basic operation, not the hidden opponent.
- Source SHA-256: `122a9c0bc0e668687106cc1939aecb60cdf34851fb26b44669cf9fe891e1e24d`.
- The Downloads source, extracted copy, project copy and sole ZIP entry were byte-checked against that hash. The prior `4edca990...` source and ZIP were backed up under `reports/delivery-backups/20261019-targeted13-prepromotion/`.

## Status and limit

The candidate is staged as `my_player3.py` in Downloads for the user's next run. It has not been submitted to Vocareum. Official Stage 2 remains 6 until the user runs this file; no full-score claim is made.
