# Stage2 tie-order experiment — 2026-10-17

## Hypothesis

The move generator sorts equal tactical-priority moves by ascending point index. A state-derived deterministic tie order might reduce repeated low-index bias when a time-limited search cannot distinguish moves.

## Result

The all-color tie order scored 34/64 on seed 630000 at 0.25 CPU seconds per move, versus 41/64 for the existing connectivity candidate at the same configuration. It gained one White win (29/32 versus 28/32) but lost eight Black wins (5/32 versus 13/32), so it was rejected.

A White-only version preserved the previous Black ordering and scored 40/64 on independent seed 530000 at 0.25 CPU seconds per move. The same-seed baseline scored 42/64 (Black 15/32, White 27/32); the variant scored Black 14/32, White 26/32. The experiment did not improve the local proxy results and was not promoted to Downloads.

## Verification and limits

- Tie-order tests passed after failing against the old ordering as expected.
- Full local unittest suite: 65 passed, 1 Windows-specific CPU accounting test skipped.
- Strong league used four local full-legal-search proxies, paired forced two-move openings, 64 games, and 0.25 CPU seconds per move. It is not the official Championship opponent; no official conclusion follows.
- Official Vocareum submission 6 remains the latest file: Q-learning 10/10, Championship Black 5/5 / White 0/5, score 6. The report gives no per-game traces or source fingerprint.
