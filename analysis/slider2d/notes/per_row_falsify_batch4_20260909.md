# Per-row falsify batch4 (M28–M31) — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 1786.7s. **Locked recipe unchanged. merge=NO.**

Expect: **M29 clears** (DoF cascade); **M28/M30/M31 stay biting** (eoc/teacher).
M14 = content_deleted_under_declare_lie (M20 family) — out of scope here.

| mid | shared pass | per_row pass | clears? | expect | match |
|---|---|---|:---:|:---:|:---:|
| M28 | 0/3 | 0/6 | no | bite | True |
| M29 | 0/3 | 6/6 | YES | clear | True |
| M30 | 0/3 | 1/6 | no | bite | True |
| M31 | 0/3 | 0/6 | no | bite | True |

CTRL leftover per_row: 3/3 ok=True
cleared=['M29']; uncleared=['M28', 'M30', 'M31']; overpowered=False

**Verdict:** YES — per-row clears M29 (DoF) without clearing M28/M30/M31 (eoc/teacher); CTRL ok; merge=NO

JSON: `per_row_falsify_batch4_20260909.json`

## Notes

- **M29**: shared multi 0/3 → per-row **6/6** — DoF/cascade bite (same class as M24/M27).
- **M28**: leak≈0.65 under shared+per-row — guard-refuse raw poles; heads do not help.
- **M30**: per-row **1/6** (leak_max≈0.205) — near-threshold knife around analytic floor 0.202; still **not cleared**.
- **M31**: multi recovers under per-row but leak≈0.207 stays — eoc floor like M21.
- Aligns Fire #25 (`per_row_falsify_m29_fire25`): M29 YES, M20/M28/M30 no.
- **M14** = content_deleted_under_declare_lie (M20 family) — separate from eoc floor.
- merge=NO; recipe_change=NO.
