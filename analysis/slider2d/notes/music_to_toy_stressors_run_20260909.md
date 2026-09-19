# Music→toy stressors run — 2026-09-09

Host: pop-os CPU @ `435e87363bf1`. Wall 1034.5s. No Music train.

## Recipe

- Locked: 1200 / cover=1.5 / faithful_guard_e / FM0 / n≤12 / l2=0.02 / b_cap=1
- Music-posture: n_particles=1 (parts0 proxy), cover=1.0

## Pass/fail grid (summary)

| cell | recipe | PASS | primary mean | leak max | knife |
|---|---|:---:|---:|---:|:---:|
| lyric_span_entangle/c1.5_n12 | locked | 0/6 | 0.9903 | 0.1923 | no |
| lyric_span_entangle/c1.5_n1 | c1.5_n1 | 0/6 | 0.9887 | 0.2011 | no |
| lyric_span_entangle/c1.0_n12 | music-posture | 0/6 | 0.9882 | 0.1880 | no |
| lyric_span_entangle/c1.0_n1 | music-posture | 0/6 | 0.9911 | 0.2076 | no |
| dual_arm/leftover_only_guard_cover0_n1 | dual | 0/3 | 0.1499 | 0.1419 | no |
| dual_arm/listen_cover_only_faithful_c1.5_n1 | dual | 0/3 | 0.9923 | 0.4565 | no |
| dual_arm/locked_guard_cover1.5_n1 | dual | 3/3 | 0.9947 | 0.0089 | no |
| close_live_noise/c1.5_n12 | locked | 6/6 | 0.9871 | 0.0083 | no |
| close_live_noise/c1.5_n1 | c1.5_n1 | 5/6 | 0.9115 | 0.0376 | yes |
| close_live_noise/c1.0_n12 | music-posture | 6/6 | 0.9707 | 0.0123 | no |
| close_live_noise/c1.0_n1 | music-posture | 5/6 | 0.8837 | 0.1164 | yes |
| close_clean/c1.5_n12 | locked | 6/6 | 0.9819 | 0.0129 | no |
| close_clean/c1.5_n1 | c1.5_n1 | 5/6 | 0.8911 | 0.0537 | yes |
| close_clean/c1.0_n12 | music-posture | 6/6 | 0.9650 | 0.0199 | no |
| close_clean/c1.0_n1 | music-posture | 5/6 | 0.8605 | 0.2024 | yes |
| leftover_baseline/c1.5_n12 | locked | 6/6 | 0.9920 | 0.0004 | no |
| leftover_baseline/c1.5_n1 | c1.5_n1 | 6/6 | 0.9975 | 0.0053 | no |
| leftover_baseline/c1.0_n12 | music-posture | 6/6 | 0.9908 | 0.0004 | no |
| leftover_baseline/c1.0_n1 | music-posture | 6/6 | 0.9977 | 0.0097 | no |
| cross_axis_rows/c1.5_n12 | locked | 0/6 | 0.5848 | 0.3839 | no |
| cross_axis_rows/c1.5_n1 | c1.5_n1 | 0/6 | 0.5759 | 0.4385 | no |
| cross_axis_span_sample/c1.5_n12 | locked | 0/6 | 0.9522 | 0.2310 | no |
| cross_axis_span_sample/c1.5_n1 | c1.5_n1 | 0/6 | 0.9567 | 0.2656 | no |

## Wins — Music bug now fails (or knives) in-toy

- lyric_span_entangle/c1.5_n12: 0/6 — M1 span-entangled leftover bites
- lyric_span_entangle/c1.5_n1: 0/6 — M1 span-entangled leftover bites
- lyric_span_entangle/c1.0_n12: 0/6 — M1 span-entangled leftover bites
- lyric_span_entangle/c1.0_n1: 0/6 — M1 span-entangled leftover bites
- dual_arm/leftover_only_guard_cover0_n1: 0/3 — M4/M7/M8 dual-arm arm must not pass alone
- dual_arm/listen_cover_only_faithful_c1.5_n1: 0/3 — M4/M7/M8 dual-arm arm must not pass alone
- dual_arm/locked_guard_cover1.5_n1: 3/3 — gate∧cover required (positive control)
- close_live_noise/c1.5_n1: 5/6 — M3/M11 close+live noise knife
- close_live_noise/c1.0_n1: 5/6 — M3/M11 close+live noise knife

## Still hard to encode

- tx∩guard SystemExit / SpanTransformerD lyric positions
- parts ↔ n_particles object mismatch (proxy only)
- lyrichold ≠ cover/pole without live listen metrics

Catalog: `music_to_toy_stressor_catalog_20260909.md`
JSON: `music_to_toy_stressors_run_20260909.json`
