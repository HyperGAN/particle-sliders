# M20 thread — tiny/closeleak n≥2 transfer + M14/M19 (2026-09-09)

Wall 1065.1s. Fire #21 n≥2 rule transfer to M13/M18.

| cell | PASS | exam | leak | fail | flag |
|---|:---:|---:|---:|---|---|
| `leftover_n12` | 3/3 | 0.9304 | 0.0004 | [] | ok |
| `M13_tiny_n1_c1.0` | 5/6 | 0.8152 | 0.2611 | [0] | KNIFE |
| `M13_tiny_n2_c1.0` | 6/6 | 0.9801 | 0.0592 | [] | ok |
| `M13_tiny_n2_c1.5` | 6/6 | 0.9881 | 0.0403 | [] | ok |
| `M13_tiny_n1_vic0` | 6/6 | 0.9672 | 0.03 | [] | ok |
| `M18_n1_c1.0` | 5/6 | 0.857 | 0.2024 | [0] | KNIFE |
| `M18_n2_c1.0` | 6/6 | 0.9838 | 0.0363 | [] | ok |
| `M14_n12` | 0/6 | 0.1817 | 0.1016 | [0, 1, 2, 3, 7, 42] | BITES |
| `M14_n2` | 0/6 | 0.1841 | 0.1037 | [0, 1, 2, 3, 7, 42] | BITES |
| `M19_n12` | 6/6 | 0.9433 | 0.0058 | [] | ok |
| `M19_n1_c1.0` | 6/6 | 0.9483 | 0.0245 | [] | ok |

Wins: ['M13_tiny_n1_c1.0: 5/6', 'M18_n1_c1.0: 5/6', 'M14_n12: 0/6', 'M14_n2: 0/6']

Recipe change? NO

JSON: `music_toy_m20_sheet_span_20260909.json`


## Takeaway

- Fire #21 **n≥2** transfers to M13 tiny + M18 close_with_leak (both 6/6).
- vic=0@n=1 propose-only also clears M13 (6/6) — **not** a default flip.
- M14 declare-lie still **0/6 at n=2** — not a parts0 knife; keep as bite cell.
- Recipe change: NO.
