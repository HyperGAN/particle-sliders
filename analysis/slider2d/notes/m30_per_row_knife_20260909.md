# M30 per-row knife — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 2000.6s. merge=NO. recipe_change=NO.

## Per-seed M30 per_row (eoc=0.33)

| seed | pass | leak_ratio | pass_leak | u_kept |
|---:|:---:|---:|:---:|---:|
| 0 | False | 0.2016 | False | 0.9935 |
| 1 | True | 0.1992 | True | 0.9926 |
| 2 | False | 0.2045 | False | 0.9993 |
| 3 | False | 0.2028 | False | 0.9957 |
| 7 | False | 0.2028 | False | 0.9920 |
| 42 | False | 0.2053 | False | 0.9905 |

pass_seeds=[1]; fail_seeds=[0, 2, 3, 7, 42]

## Analytic lr vs eoc (M21 pool amps)

| eoc | analytic_lr | pass_analytic |
|---:|---:|:---:|
| 0.30 | 0.1948 | True |
| 0.32 | 0.1996 | True |
| 0.33 | 0.2016 | False |
| 0.34 | 0.2034 | False |
| 0.35 | 0.2050 | False |

## Empirical eoc sweep

| eoc | mode | pass | mean_leak | max_leak | fail |
|---:|---|:---:|---:|---:|---|
| 0.30 | shared | 2/3 | 0.1995 | 0.2006 | [2] |
| 0.30 | per_row | 3/3 | 0.1967 | 0.1987 | [] |
| 0.32 | shared | 0/3 | 0.2046 | 0.2063 | [0, 1, 2] |
| 0.32 | per_row | 2/3 | 0.1988 | 0.2004 | [2] |
| 0.33 | shared | 0/3 | 0.2065 | 0.2079 | [0, 1, 2] |
| 0.33 | per_row | 1/6 | 0.2027 | 0.2053 | [0, 2, 3, 7, 42] |
| 0.34 | shared | 0/3 | 0.2089 | 0.2100 | [0, 1, 2] |
| 0.34 | per_row | 0/3 | 0.2042 | 0.2071 | [0, 1, 2] |
| 0.35 | shared | 0/3 | 0.2105 | 0.2113 | [0, 1, 2] |
| 0.35 | per_row | 0/6 | 0.2062 | 0.2070 | [0, 1, 2, 3, 7, 42] |

## Couple weight on M30 per_row

| w | pass | mean_leak | max_leak | fail |
|---:|:---:|---:|---:|---|
| 0.0 | 1/6 | 0.2027 | 0.2053 | [0, 2, 3, 7, 42] |
| 0.1 | 0/6 | 0.2020 | 0.2058 | [0, 1, 2, 3, 7, 42] |
| 0.3 | 0/6 | 0.2026 | 0.2049 | [0, 1, 2, 3, 7, 42] |

## Verdict

**M30 per-row knife = seeds [1] (leak near floor); couple does not clear 6/6; eoc sweep confirms threshold; NOT a recipe fix — keep HARD_BITE**

- Do not chase M30 with couple/per-row/recipe knobs.
- M14 = content_deleted_under_declare_lie (M20 family).

JSON: `m30_per_row_knife_20260909.json`
