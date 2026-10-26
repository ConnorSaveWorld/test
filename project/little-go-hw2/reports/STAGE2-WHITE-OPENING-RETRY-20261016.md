# Stage2 white-opening retry — 2026-10-16

## Trigger

The latest official run scored 6: Q-learning 10/10, Championship 5/10. Against Championship, Black won 5/5 and White won 0/5. The platform report contains aggregate results only and no move traces, so the exact losing move cannot be diagnosed from available evidence.

## Change

The previous candidate evenly divided an 86-second game CPU budget. This candidate weights White's first three decisions 1.25x, giving each about 8.43 CPU seconds on a full game, subject to an 8.5-second move cap. Other decisions share the remaining game budget, with a 7-second move cap. The total cap stays 86 seconds (four seconds below the approximate 90-second game allowance); wall time is capped at 9.5 seconds per move. The change targets the repeatable White-only loss pattern by spending more search on early responses. It is a hypothesis, not a demonstrated Championship fix.

## Verification

- Added weighted-budget tests; they failed before the function accepted color and passed after implementation (3/3).
- Candidate rule/protocol/search suite: 27 passed.
- Full project suite: 63 passed; one platform-specific Unix CPU accounting test skipped on Windows.
- 25,000 candidate transitions checked against the independent matrix rules and course GO: zero mismatches.
- Python 3.13 compilation passed.
- Source, Downloads file, and one-file ZIP member all match SHA-256 `604ea3ee7228a790cbcef60c2a8a6dfac043acb83d4a72ce30c2568466bd1dcb`.

## Delivery

- Source: `stage2_candidate/my_player3.py`
- Upload-ready file: `C:\Users\paul2\Downloads\my_player3.py`
- Archive: `C:\Users\paul2\Downloads\little-go-stage2-candidate.zip`
- Official Championship remains unverified for this revision; only a new platform run can establish whether White's result changes.
