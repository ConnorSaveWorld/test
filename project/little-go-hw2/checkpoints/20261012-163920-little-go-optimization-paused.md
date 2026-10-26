# Little-Go optimization checkpoint

Pause requested on 2026-10-12; documentation completed during resume on 2026-10-13. The original JSON manifest is retained unchanged. No Git repository exists. Synced sources and original downloads are read-only.

## Verified state

- Submission and ZIP still contain the original baseline, SHA256 b5ec69b0c8cf0e016d5f7fb32486058be2c3b1534df9842badd46aa052b01d62.
- Frozen bitboard candidate: experiments/candidate_fast.py, SHA256 98f7ce51ad9a872b27542dce7d5148817f0b18525374dab6df89e184ca9bf540.
- Frozen PVS candidate: experiments/candidate_pvs.py, SHA256 dda83b2ac22036ea4bdc49418193aa38893fe4f1f6c64d06494e2a45156bff36.
- Actual CPython 3.7.5 runtime: .runtime/Python-3.7.5/python.exe under WSL Ubuntu-24.04. The ELF executable has a .exe suffix because the build lives on a Windows filesystem.
- Baseline full fresh-process Stage1 local rehearsal: Random 20/20, Greedy proxy 20/20, Aggressive proxy 20/20, Alpha-beta proxy 16/20. Zero invalid moves, crashes, or timeouts. Only Random is the original course opponent.
- Bitboard candidate completed Alpha-beta proxy 19/20, Black 10/10, White 9/10. Zero invalid moves, crashes, or timeouts; max move CPU 3.123 seconds, max game CPU 25.615 seconds.
- Last full suite: 46 tests passed on actual CPython 3.7.5. Both optimized internal rule engines matched independent and original-host oracles on 25,000 transitions.
- Independent reviewer found no remaining major issue; PVS, full-window, and no-TT values agreed across 30 additional fixtures. Check cold-process timing before promotion.
- Local TD learning regressed badly and was rejected. No learned model is deployed.

## Interrupted work

All five active tournament process groups were stopped for the pause. No candidate was promoted. Partial progress counts are preserved in the adjacent JSON manifest, but the previous harness did not save full partial-game traces. Do not combine them into complete win-rate claims. In particular, held-out Random had one loss (8/9); do not omit it from the historical record.

## Resume work

1. Re-run full 20-game PVS batches with unchanged frozen code: development Alpha-beta seed 561100, and four held-out opponents seed 762000. Use fresh tags pvs-resume-dev and pvs-resume-heldout, retaining old partial files.
2. Check full outcomes by color, error counts, CPU per move and game, and every terminal trace. Seek at least 18/20 per Stage1 group. Local proxies cannot establish the hidden official score.
3. Promote only a validated candidate, rerun the full suite against the promoted submission, repeat internal rule validation and fresh-process deployment probes, then regenerate the ZIP with only my_player3.py.
4. Update README, progress.md and the final comparison report. Formal Alpha-beta, Q-learning, Championship and Vocareum CPU-wrapper behavior remain untested externally.

Invoke tournaments from the little-go-hw2 directory via WSL:

    .runtime/Python-3.7.5/python.exe tools/stage1.py --opponent alphabeta-proxy --games 20 --seed 561100 --source experiments/candidate_pvs.py --tag pvs-resume-dev

Use the same command with seed 762000, tag pvs-resume-heldout, and each opponent official-random, greedy-proxy, aggressive-proxy, alphabeta-proxy.

The archive excludes the compiled runtime, caches, and backup archives; these remain in the local workspace. Source, tests, tools, references, and completed reports are preserved.
