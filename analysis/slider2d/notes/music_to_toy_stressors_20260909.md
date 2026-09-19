# Music → toy stressors — results 2026-09-09 (box-cpu)

**Host:** shared box `/workspace/sliders-conceptmod` (laptop/pop-os hop DOWN; no SSH).
**SHA:** `435e87363bf1` (+ local `field3d.py` Music hooks + `CELLS_3D` registry).
**No Music GPU train.** CPU harness only via `/workspace/.venv-2d` (torch 2.13.0+cpu).

**Locked toy recipe:** steps=1200, `cover_weight=1.5`, `teacher=faithful_guard_e`,
FM=0, `n_particles≤12`, `particle_l2=0.02`, `b_cap=1`. Reject 800×cover3.0.

**Music-posture:** `n_particles=1` (parts0 proxy), `cover_weight=1.0`.

Catalog: `music_to_toy_stressor_catalog_20260909.md`
Full grid JSON/MD: `music_to_toy_stressors_run_20260909.{json,md}` (wall **1034.5s**)
Dig: `dig_close_cross_music_stress_20260909.{json,md}` (wall **351.5s**)
Prior Fire #19: `music_toy_dig_fire19_20260909.{json,md}`

---

## Integration (box)

Hooks from package `field3d_music_stressor_hooks.py` already present as ctors in
`analysis/slider2d/field3d.py`; this job also registered them in `CELLS_3D`:

- `lyric_span_entangle` → `lyric_span_entangle_field3d` (M1)
- `close_live_noise` → `close_live_noise_field3d` (M3/M11)
- `dual_arm_leftover_geom` → `dual_arm_leftover_geom_field3d` (M4/M7/M8)

`__all__` already exported the three ctors. Scripts/notes copied under
`analysis/slider2d/notes/` with REPO path rewritten to `/workspace/sliders-conceptmod`.

---

## Wins — Music bugs that now fail / knife in-toy

| ID | Cell | Locked / posture | PASS | primary mean | leak max | Notes |
|---|---|---|:---:|---:|---:|---|
| **M1** | `lyric_span_entangle` | locked c1.5 n12 | **0/6** | 0.9903 | 0.1923 | bites via `pass_multi_row` (rows_covered 0/5) despite high u_kept |
| **M1** | `lyric_span_entangle` | locked c1.5 n1 | **0/6** | 0.9887 | 0.2011 | same |
| **M1** | `lyric_span_entangle` | music c1.0 n12 | **0/6** | 0.9882 | 0.1880 | same |
| **M1** | `lyric_span_entangle` | music c1.0 n1 | **0/6** | 0.9911 | 0.2076 | same |
| **M2** | `cross_axis_rows` | locked n12 / n1 | **0/6** / **0/6** | 0.5848 / 0.5759 | 0.3839 / 0.4385 | **hard boundary** (Fire #13 confirmed) |
| **M2** | `cross_axis_span_sample` | locked n12 / n1 | **0/6** / **0/6** | 0.9522 / 0.9567 | 0.2310 / 0.2656 | bites (multi-row / leak) |
| **M3** | `close_clean` @ n=1 | c1.0 / c1.5 | **5/6** knife | 0.8605 / 0.8911 | 0.2024 / 0.0537 | **seed0 only** fail |
| **M11** | `close_live_noise` @ n=1 | c1.0 / c1.5 | **5/6** knife | 0.8837 / 0.9115 | 0.1164 / 0.0376 | same seed0 knife; n=12 is 6/6 |
| **M7** | dual listen `faithful` cover1.5 n1 | score mode | **0/3** | 0.9923 | **0.4565** | Arm T alone FAIL (leak) |
| **M8** | dual leftover_only guard cover0 n1 | score mode | **0/3** | 0.1499 | 0.1419 | Arm L alone FAIL (undershoot) |
| **M4+** | dual locked guard∧cover n1 | score mode | **3/3** | 0.9947 | 0.0089 | positive control — both required |

### Dig harden (f3d_close seed0 @ n=1 cover=1.5)

- More steps 1600/2000: still FAIL (exam≈0.37)
- `n_particles=4` or `12`: recovers PASS (exam≈0.97 / 0.98)
- Alt seeds 5–11: all PASS; geom jitter seed0: still FAIL
- **Music rule:** multi-seed close; do not raise pole on seed0 alone; parts0 (n=1) exposes knife that n≥4 hides.

---

## Positive controls (must stay green)

| cell | recipe | PASS | primary mean | leak max |
|---|---|:---:|---:|---:|
| leftover_baseline | locked c1.5 n12 | 6/6 | 0.9920 | 0.0004 |
| leftover_baseline | music c1.0 n1 | 6/6 | 0.9977 | 0.0097 |
| close_clean / close_live_noise | locked c1.5 n12 | 6/6 / 6/6 | 0.9819 / 0.9871 | ≤0.013 |
| dual locked guard∧cover | n1 | 3/3 | 0.9947 | 0.0089 |

---

## Catalog completeness (M1–M12)

| ID | In-toy status after this job |
|---|---|
| M1 | **BITES** — `lyric_span_entangle` 0/6 all postures |
| M2 | **BITES / hard boundary** — `cross_axis_rows` 0/6 n=1 and n=12 |
| M3 | **Knife** — close n=1 seed0-only 5/6 |
| M4 | Encoded as dual-arm cells (L+T both required) |
| M5 | Documented: leftover still 6/6 at n=1; knives on close/cross |
| M6 | Neg controls already in Fire #13 (`axis_leak_primary`, mismatch_declare) |
| M7 | **BITES** score-mode listen/cover_only 0/3 leak≈0.46 |
| M8 | **BITES** gate-only cover0 0/3 kept≈0.15 |
| M9 | Reject cell (doc) — do not adopt 800×cover3.0 |
| M10 | Doc gap only |
| M11 | **Knife** — `close_live_noise` 5/6 @ n=1 |
| M12 | Encoding gap (tx / SpanTransformerD) |

---

## Pytest

- `tests/test_field3d.py` — Music ctors + `CELLS_3D` registry tests added; run focused.
- `tests/test_music_to_toy_stressors.py` — short-smoke (400 steps) for new cells.

---

## Sync

See `/workspace/sliders-conceptmod/SYNC_TO_POPOS.md` for scp list when laptop returns.
