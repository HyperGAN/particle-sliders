# Analytic-valley emp + sheet/highd A/B/C ports — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 565.9s. CPU only. **No Music GPU.**
Locked shared AdvResidual **unchanged**. merge=**NO**.
Does **not** redo `declare_lie_eoc_family_theory` — fills emp valley + sheet/highd ports.

## 1. Analytic valley emp (M21 pool)

Prior theory dig: analytic lr < 0.20 for eoc∈[0.55,0.65] (admit) before refuse@0.68.
Emp was missing in that band (only floor@0.32 / refuse@0.68 anchors).

| cell | PASS | exam | leak | ana_lr | ana_pass |
|---|:---:|---:|---:|---:|:---:|
| `CTRL_leftover` | 3/3 | 0.9304 | 0.0003 | None | None |
| `M21pool_eoc0.55_valley` | 0/3 | 0.4146 | 0.2012 | 0.1959 | True |
| `M21pool_eoc0.58_valley` | 0/3 | 0.3796 | 0.1944 | 0.1892 | True |
| `M21pool_eoc0.6_valley` | 0/3 | 0.3571 | 0.1891 | 0.1843 | True |
| `M21pool_eoc0.62_valley` | 0/3 | 0.3368 | 0.1834 | 0.179 | True |
| `M21pool_eoc0.65_valley` | 0/3 | 0.3058 | 0.1753 | 0.1703 | True |
| `M21pool_eoc0.66_pre_refuse` | 0/3 | 0.2955 | 0.1717 | 0.1673 | True |
| `M21pool_eoc0.68_refuse_onset` | 0/3 | 1.0 | 0.6568 | 0.161 | True |

**Valley class:** `valley_emp_BITES_mixed_mech` — analytic_all_pass_leak=True; emp_all_fail=True.

- **eoc=0.55:** residual leak **overshoot** (0.2012 > 0.20) while ana_lr=0.1959.
- **eoc∈[0.58,0.65]:** leak **under** gate (0.194→0.175) but **exam/content collapses** (exam≈0.38→0.31) under tilted admit teacher — still BITES.
- **eoc≥0.68:** refuse → raw leak≈0.66 (exam=1.0).
- Analytic 'valley' is **teacher-only**; emp never clears the band under locked shared.

## 2. Sheet A/B/C declare ports

Teacher=`faithful_guard_e` on `leaky_field` with declared leak_dir variants.

| name | PASS | leak_tok | on_sheet | off_caption | branch |
|---|:---:|---:|---:|---:|---|
| `CTRL_leftover_e` | 3/3 | 0.0 | 0.9391 | 0.3734 | CTRL |
| `A_on_u` | 0/3 | 0.2277 | 0.9926 | 0.0 | A_or_C_refuse_raw |
| `A_on_sheet` | 0/3 | 0.2277 | 0.9926 | 0.0 | A_or_C_refuse_raw |
| `B_tilt_mild` | 3/3 | 0.0375 | 0.9494 | 0.334 | B_tilt_admit |
| `B_tilt_mid` | 3/3 | 0.098 | 0.9645 | 0.264 | B_tilt_admit |
| `B_tilt_hot` | 3/3 | 0.1943 | 0.986 | 0.1243 | B_tilt_admit |
| `C_hot_sheet` | 0/3 | 0.2186 | 0.9908 | 0.064 | A_or_C_refuse_raw |

### Sheet DoF CELLS_SHEET_DOF (NO_BITE)

Scale / row_leaks proxies of M16/M24/M27/M29 — sheet verdicts stay green (shared residual still covers; no content-axis hetero). Prefer Field3D for DoF bites.

| cell | PASS | note |
|---|:---:|---|
| `scale_stagger_sheet` | 3/3 | NO_BITE expected — scale/leak-amp alone does not break sheet verdicts |
| `scale_descent_sheet` | 3/3 | NO_BITE expected — scale/leak-amp alone does not break sheet verdicts |
| `content_leak_flip_sheet` | 3/3 | NO_BITE expected — scale/leak-amp alone does not break sheet verdicts |
| `content_cascade_sheet` | 3/3 | NO_BITE expected — scale/leak-amp alone does not break sheet verdicts |

## 3. Highd A ports (hold-λ)

Highd has no `faithful_guard_e` teacher — A maps via hold-λ content-heavy declare.
Note: `CTRL_leftover_only` also fails highd `LEAK_LOCK=0.20` (leak≈0.50) — the A-port signal is **leak magnitude** (synonym/hot ≈4.8–5.2 vs leftover ≈0.50), not pass/fail alone.

| name | PASS | leak | hold_on_content | cover |
|---|:---:|---:|---:|---:|
| `CTRL_leftover_only` | 0/3 | 0.504 | 0.0 | 0.5873 |
| `A_synonym_content` | 0/3 | 4.7744 | 0.9207 | 0.8412 |
| `A_on_u_lie` | 0/3 | 0.9557 | 0.0 | 0.0 |
| `A_hot_content` | 0/3 | 5.215 | 0.9536 | 0.8 |
| `A_medium_pin` | 0/3 | 4.8027 | 0.9218 | 0.8401 |

## Family port map

```
Field3D A/B/C  ──sheet──►  A_on_u / C_hot FAIL leak_tok; B_tilt often PASS;
                           DoF sheet = NO_BITE (encoding gap)
Field3D A      ──highd──►  synonym / hot_content FAIL leftover_leak (hold-λ)
Field3D B/C               highd lacks admit/refuse guard teacher
```

## Verdict

valley=valley_emp_BITES_mixed_mech; sheet_A_bites=YES; sheet_DoF=NO_BITE; highd_A_bites=YES; recipe_change=NO; merge=NO

- suite notes: valley emp filled; sheet/highd ports documented
- recipe_change=**NO**; merge_to_trainer=**NO**; No Music train

JSON: `valley_emp_sheet_abc_ports_20260909.json`
