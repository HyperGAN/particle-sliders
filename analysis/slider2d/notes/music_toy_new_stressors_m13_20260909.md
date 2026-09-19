# Music→toy new stressors M13–M19 — 2026-09-09

Host: box-cpu @ `435e87363bf1`. Wall 3108.2s. CPU only. No Music train.

## New cells (CELLS_3D)

| ID | cell | Music proxy |
|---|---|---|
| M13 | `tiny_slider_dom` | harder close (û=0.05, content=1.2) |
| M14 | `e_on_u_declare_lie` | YAML ê restates û |
| M15 | `prefix_shared_proxy` | whole-prefix hold / shared neu |
| M16 | `scale_stagger_homo` | mild multi-row (homo amps + scale stagger) |
| M17 | `roles_split_proxy` | role-split UNI (û vs content rows) |
| M18 | `close_with_leak` | close + small unused ê |
| M19 | `grit_content_dom` | grit/distortion content-dom |

## Results

| cell | PASS | exam | u | leak | rows | fail | flag |
|---|:---:|---:|---:|---:|---:|---|---|
| `leftover_locked_n12` | 3/3 | 0.9304 | 0.9918 | 0.0004 | 1.0 | [] | ok |
| `close_locked_n12` | 3/3 | 0.9839 | 1.0024 | 0.0129 | 3.0 | [] | ok |
| `M13_tiny_slider_dom_locked_n12_c1.5` | 6/6 | 0.9743 | 1.0136 | 0.034 | 3.0 | [] | ok |
| `M13_tiny_slider_dom_music_n1_c1.0` | 5/6 | 0.8152 | 0.9399 | 0.2611 | 3.0 | [0] | KNIFE |
| `M14_e_on_u_declare_lie_locked_n12_c1.5` | 0/6 | 0.1817 | 0.9878 | 0.1016 | 3.0 | [0, 1, 2, 3, 7, 42] | BITES |
| `M14_e_on_u_declare_lie_music_n1_c1.0` | 0/6 | 0.1702 | 0.9805 | 0.1114 | 3.0 | [0, 1, 2, 3, 7, 42] | BITES |
| `M15_prefix_shared_proxy_locked_n12_c1.5` | 6/6 | 0.965 | 1.1642 | 0.0031 | 4.0 | [] | ok |
| `M15_prefix_shared_proxy_music_n1_c1.0` | 6/6 | 0.9637 | 1.1776 | 0.0281 | 4.0 | [] | ok |
| `M16_scale_stagger_homo_locked_n12_c1.5` | 0/6 | 0.9434 | 1.4521 | 0.0021 | 3.0 | [0, 1, 2, 3, 7, 42] | BITES |
| `M16_scale_stagger_homo_music_n1_c1.0` | 0/6 | 0.9434 | 1.5022 | 0.0086 | 3.0 | [0, 1, 2, 3, 7, 42] | BITES |
| `M17_roles_split_proxy_locked_n12_c1.5` | 0/6 | 0.6266 | 0.6266 | 0.0067 | 0.0 | [0, 1, 2, 3, 7, 42] | BITES |
| `M17_roles_split_proxy_music_n1_c1.0` | 0/6 | 0.6417 | 0.6417 | 0.0243 | 0.0 | [0, 1, 2, 3, 7, 42] | BITES |
| `M18_close_with_leak_locked_n12_c1.5` | 6/6 | 0.9816 | 1.0003 | 0.0129 | 3.0 | [] | ok |
| `M18_close_with_leak_music_n1_c1.0` | 5/6 | 0.857 | 0.933 | 0.2024 | 3.0 | [0] | KNIFE |
| `M19_grit_content_dom_locked_n12_c1.5` | 6/6 | 0.9433 | 0.9963 | 0.0058 | 3.0 | [] | ok |
| `M19_grit_content_dom_music_n1_c1.0` | 6/6 | 0.9483 | 1.0132 | 0.0245 | 3.0 | [] | ok |
| `M13_tiny_music_n1_vic0` | 6/6 | 0.9672 | 1.0019 | 0.03 | 3.0 | [] | ok |
| `M18_closeleak_music_n1_vic0` | 6/6 | 0.9762 | 0.9879 | 0.0076 | 3.0 | [] | ok |
| `M13_tiny_music_n2` | 6/6 | 0.9801 | 1.0176 | 0.0592 | 3.0 | [] | ok |
| `probe_leftover_bcap0_n12` | 3/3 | 0.93 | 0.9483 | 0.0008 | 1.0 | [] | ok |
| `probe_close_n1_end0` | 5/6 | 0.9006 | 0.9443 | 0.0642 | 3.0 | [0] | KNIFE |
| `probe_close_n1_span1` | 2/3 | 0.7967 | 0.8326 | 0.0743 | 3.0 | [0] | KNIFE |
| `probe_live_n1_vic0` | 6/6 | 0.9866 | 0.9912 | 0.0161 | 3.0 | [] | ok |

## Wins (biting / knife)

- M13_tiny_slider_dom_music_n1_c1.0: 5/6 KNIFE fail=[0]
- M14_e_on_u_declare_lie_locked_n12_c1.5: 0/6 BITES
- M14_e_on_u_declare_lie_music_n1_c1.0: 0/6 BITES
- M16_scale_stagger_homo_locked_n12_c1.5: 0/6 BITES
- M16_scale_stagger_homo_music_n1_c1.0: 0/6 BITES
- M17_roles_split_proxy_locked_n12_c1.5: 0/6 BITES
- M17_roles_split_proxy_music_n1_c1.0: 0/6 BITES
- M18_close_with_leak_music_n1_c1.0: 5/6 KNIFE fail=[0]
- probe_close_n1_end0: 5/6 KNIFE fail=[0]
- probe_close_n1_span1: 2/3 KNIFE fail=[0]

## Verdict

- Recipe change? **NO**
- lyric_span still HARD BOUNDARY (not re-chased)
- close parts0 rule from harden bites still stands (multi-seed / n≥2 / vic0@n1 candidate)

JSON: `music_toy_new_stressors_m13_20260909.json`

## Close / parts0 decision (locked elsewhere)

- multi-seed gate for close-family @ n=1
- **n≥2 harden** (Fire #21)
- **vicreg_weight=0 @ n=1 propose-only** (NOT in defaults) — confirmed here: M13/M18/live vic0 → 6/6

## Parallel (do not merge)

- per-row AdvResidual clears lyric_span 6/6 analysis-only — shared residual remains default.
