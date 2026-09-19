# f3d_close cover knife @ n=1 — 2026-09-09 (Fire #18)

| cover | PASS | exam mean | span | leak max | knife |
|---:|:---:|---:|---:|---:|:---:|
| 1.0 | 5/6 | 0.8433 | 0.7906 | 0.2645 | YES |
| 1.5 | 5/6 | 0.8878 | 0.6158 | 0.0670 | YES |

### Finding

- Both covers fragile on f3d_close @ n=1 — investigate.
- Verdict: `close_fragile_both`
- Music: Do not smoke Music close pairs until close cell solid.

## Detail

Both covers fail **only seed0** (1.0: exam=0.20 leak=0.26; 1.5: exam=0.38 leak=0.03).
Seeds {1,2,3,7,42} PASS both. Not a cover=1.0-vs-1.5 recipe break — a **seed0
close-cell flake** under n=1. Music: keep pole=1.0 lock; require multi-seed close
PASS (drop seed0 or demand 6/6 with margin). Raising cover to 1.5 does **not**
fix seed0 here.
