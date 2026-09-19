# Fire #20 — lyric_span_entangle harden dig (2026-09-09)

Host: box-cpu @ `435e873` (+ local field3d.py). Wall 300.0s. CPU only.
pop-os SSH hop DOWN (laptop+LAN timeout).

## Hypothesis

Shared `AdvResidual` + multi-row cover cannot hit heterogeneous `row_amps`
when `e_on_content>0` (Music span-entangled leftover / lyric pool mix).

## Grid

| cell | PASS | multi_row | mean exam | leak max | mean rows_cov | fail |
|---|:---:|:---:|---:|---:|---:|---|
| `leftover_n12_c1.5` | 3/3 | 3/3 | 0.9304 | 0.0004 | 1.0 | [] |
| `lyric_m1_locked_n12_c1.5` | 0/6 | 0/6 | 0.9903 | 0.1923 | 0.0 | [0, 1, 2, 3, 7, 42] |
| `hetero_eoc0.0_n12_c1.5` | 0/3 | 0/3 | 0.9355 | 0.0016 | 1.0 | [0, 1, 2] |
| `hetero_eoc0.15_n12_c1.5` | 0/3 | 0/3 | 0.9641 | 0.0775 | 0.0 | [0, 1, 2] |
| `hetero_eoc0.25_n12_c1.5` | 0/3 | 0/3 | 0.9625 | 0.037 | 0.0 | [0, 1, 2] |
| `hetero_eoc0.45_n12_c1.5` | 0/3 | 0/3 | 0.9902 | 0.1923 | 0.0 | [0, 1, 2] |
| `homo_eoc0.0_n12_c1.5` | 0/3 | 0/3 | 0.9325 | 0.0024 | 3.0 | [0, 1, 2] |
| `homo_eoc0.45_n12_c1.5` | 0/3 | 0/3 | 0.5174 | 0.1509 | 3.0 | [0, 1, 2] |
| `m1_n4_c1.5` | 0/3 | 0/3 | 0.9886 | 0.1662 | 0.0 | [0, 1, 2] |
| `m1_n12_c2.0` | 0/3 | 0/3 | 0.9906 | 0.1887 | 0.0 | [0, 1, 2] |
| `m1_n12_steps1600` | 0/3 | 0/3 | 0.9893 | 0.1705 | 0.0 | [0, 1, 2] |
| `m1_n12_c1.5_eunused1` | 0/3 | 0/3 | 0.9598 | 0.0132 | 0.0 | [0, 1, 2] |
| `lyric_m1_music_n1_c1.0` | 0/3 | 0/3 | 0.9921 | 0.2076 | 0.0 | [0, 1, 2] |

## Isolation (what actually bites)

- Fail mode is almost always `pass_multi_row` (rows_covered ≪ rows), not row0 u_kept.
- Heterogeneous `row_amps` alone → ~1/5 rows covered (Music multi-span).
- Homogeneous amps + staggered `row_scales` → ~3/5 covered (still short of full).
- `e_on_content=0.45` mainly inflates leak / can break leftover gate on homo cells.
- Recipe knobs (n, cover, steps) do not unlock multi-row — shared residual limit.

## Verdict

- **leftover regression?** NO (3/3)
- **locked M1 still bites?** YES (0/6, multi 0/6)
- **e_on_content soft boundary?** False (hetero eoc0 0/3 vs eoc0.45 0/3)
- **homo+eoc0.45 alone OK?** False (0/3)
- **cross-axis drives fail more than entangle?** False
- **any cover/n/steps harden full-pass?** False
- **hard boundary (document)?** True
- **Recipe change?** NO
- **Ping user?** YES — Music→toy M1 bite + harden diagnosis

JSON: `lyric_span_harden_fire20_20260909.json`
