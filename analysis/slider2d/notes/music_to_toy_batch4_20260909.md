# Music→toy batch4 — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 1428.2s. CPU only. No Music train.

## New cells (M28–M31)

| ID | Cell | Music / mech symptom |
|---|---|---|
| M28 | `guard_refuse_hot_eoc` | hot eoc → blend guard refuses → raw-pole leak (M21 dig refuse band) |
| M29 | `content_cascade_rows` | û-stable ascending content (verse→chorus) |
| M30 | `eoc_threshold_edge` | M21 pool, eoc=0.33 just over analytic ê-floor |
| M31 | `leftover_hot_eoc_declare` | leftover amps + admitting hot eoc (ê-floor, not YAML blowup) |

## Guard contrast (teacher, no train)

| cell | admissible | teacher_lr | eoc |
|---|:---:|---:|---:|
| M21 | True | 0.2050 | 0.35 |
| M28 | False | 0.6500 | 0.85 |
| M30 | True | 0.2016 | 0.33 |
| M31 | True | 0.2036 | 1.15 |

## Bite / no-bite table

| cell | mid | posture | pass | exam_mean | leak_max | fail_seeds | verdict |
|---|---|---|:---:|---:|---:|---|---|
| `guard_refuse_hot_eoc` | M28 | locked | **0/6** | 1.0000 | 0.6629 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `guard_refuse_hot_eoc` | M28 | music | **0/6** | 0.9999 | 0.6598 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `content_cascade_rows` | M29 | locked | **0/6** | 0.8807 | 0.0084 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `content_cascade_rows` | M29 | music | **0/6** | 0.8769 | 0.0103 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `eoc_threshold_edge` | M30 | locked | **0/6** | 0.7100 | 0.2088 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `eoc_threshold_edge` | M30 | music | **0/6** | 0.7127 | 0.2142 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `leftover_hot_eoc_declare` | M31 | locked | **0/6** | -0.1550 | 0.2101 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `leftover_hot_eoc_declare` | M31 | music | **0/6** | -0.1469 | 0.2152 | [0, 1, 2, 3, 7, 42] | **HARD_BITE** |
| `leftover_regression` | CTRL | locked | **3/3** | 0.9304 | 0.0004 | [] | **NO_BITE** |

## vic0@n1 smoke (must not false-fix hard bites)

- `guard_refuse_hot_eoc` vic0 n1: 0/3 leak_max=0.6619 (HARD_BITE)
- `eoc_threshold_edge` vic0 n1: 0/3 leak_max=0.2076 (HARD_BITE)

## Registry

Registered in `CELLS_3D` this fire:

- `guard_refuse_hot_eoc` (M28)
- `content_cascade_rows` (M29)
- `eoc_threshold_edge` (M30)
- `leftover_hot_eoc_declare` (M31)

## Verdict

- Recipe change: **NO**
- Leftover regression: **3/3** (must stay NO_BITE)
- HARD_BITEs: ['M28_guard_refuse_hot_eoc/locked: 0/6 HARD_BITE', 'M28_guard_refuse_hot_eoc/music: 0/6 HARD_BITE', 'M29_content_cascade_rows/locked: 0/6 HARD_BITE', 'M29_content_cascade_rows/music: 0/6 HARD_BITE', 'M30_eoc_threshold_edge/locked: 0/6 HARD_BITE', 'M30_eoc_threshold_edge/music: 0/6 HARD_BITE', 'M31_leftover_hot_eoc_declare/locked: 0/6 HARD_BITE', 'M31_leftover_hot_eoc_declare/music: 0/6 HARD_BITE']
- KNIVEs: []
- NO_BITEs: []

JSON: `music_to_toy_batch4_20260909.json`

## Interpretation

- **M28**: Music hot hold-ê YAML that breaks `lm_blend_guard` — teacher falls back to raw poles → full leak. Distinct from M21 (admits, floor≈0.205).
- **M29**: verse→chorus content cascade — shared δ covers only 2/4 rows (DoF). Leak tiny. Expect per-row clears (like M24/M27); next falsify.
- **M30**: eoc=0.33 sits just over analytic floor (0.32 pass / 0.33 fail) — confirms sharp threshold from M21 dig.
- **M31**: leftover amps + admitting hot eoc — ê-floor leak≈0.21 **and** exam collapse (content survival). Not M20 YAML blowup (e_unused still 0.45).
- **vic0@n1**: does not false-fix M28/M30 (ADOPT posture safe for these bites).
- **M14 reminder**: `content_deleted_under_declare_lie` (M20 family) — unrelated to batch4 eoc floor.

**Wins (Music bug now fails in-toy):** M28, M29, M30, M31.

No recipe change. Keep FM0, l2=0.02, locked 1200+c1.5.
