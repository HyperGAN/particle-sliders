# Per-row falsify A — M21/M24/M27 vs M20 — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 1578.8s. CPU only. **NON_DEFAULT_falsify_A**.
Locked recipe unchanged. merge_to_trainer=NO.

## Hypothesis

Per-row AdvResidual can clear DoF / content↔leak flip / scale-descent
bites (M21/M24/M27) **without** clearing amp-lie declare (M20).
If M20 clears → heads overpowered (falsified).

## Grid

| cell | mid | mode | pass | multi | bite | exam | leak_max | fail |
|---|---|---|:---:|:---:|:---:|---:|---:|---|
| `hold_e_lyric_mix` | M21 | shared | **0/3** | 3/3 | 0/3 | 1.0 | 0.2113 | [0, 1, 2] |
| `hold_e_lyric_mix` | M21 | per_row | **0/6** | 6/6 | 0/6 | 0.9921 | 0.207 | [0, 1, 2, 3, 7, 42] |
| `content_leak_flip_rows` | M24 | shared | **0/3** | 3/3 | 3/3 | 0.6472 | 0.0014 | [0, 1, 2] |
| `content_leak_flip_rows` | M24 | per_row | **6/6** | 6/6 | 6/6 | 0.9868 | 0.005 | [] |
| `scale_descent_homo` | M27 | shared | **0/3** | 0/3 | 0/3 | 0.7028 | 0.0007 | [0, 1, 2] |
| `scale_descent_homo` | M27 | per_row | **6/6** | 6/6 | 6/6 | 0.9811 | 0.0056 | [] |
| `amp_lie_leftover_declare` | M20 | shared | **0/3** | 3/3 | 0/3 | 0.986 | 0.3855 | [0, 1, 2] |
| `amp_lie_leftover_declare` | M20 | per_row | **0/6** | 6/6 | 0/6 | 0.9908 | 0.3819 | [0, 1, 2, 3, 7, 42] |
| `leftover` | CTRL | shared | **3/3** | 3/3 | 3/3 | 0.9926 | 0.0002 | [] |
| `leftover` | CTRL | per_row | **3/3** | 3/3 | 3/3 | 0.9926 | 0.0002 | [] |

## Verdict

- **YES — per-row clears ['M24', 'M27'] without clearing M20 (amp-lie still bites); uncleared=['M21']; CTRL ok; merge=NO**
- ideal=True; overpowered=False; m20_still_bites=True
- cleared_geom=['M24', 'M27']; uncleared_geom=['M21']
- leftover CTRL ok=True
- recipe_change=NO; merge_to_trainer=NO

JSON: `per_row_falsify_m21_m24_m27_20260909.json`
