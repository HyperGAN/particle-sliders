# Fire #21 — f3d_close / close_live_noise harden (2026-09-09)

Host: box-cpu @ `435e873`. Wall #21d 219.6s (plus partial 21a/21b). CPU only. pop-os SSH DOWN.

## Hypothesis

Music-posture `n_particles=1` causes the f3d_close seed0-only knife; **n≥2** may already harden (tighter than prior n≥4 note).

## Grid

| cell | PASS | mean exam | fail | seed0-only |
|---|:---:|---:|---|:---:|
| `leftover_n12_c1.5` | 6/6 | — | [] | False |
| `close_n1_c1.0` | 5/6 | 0.8605 | [0] | True |
| `close_n1_c1.5` | 5/6 | 0.8911 | [0] | True |
| `close_n2_c1.0` | 6/6 | 0.9891 | [] | False |
| `close_n2_c1.5` | 6/6 | 0.9923 | [] | False |
| `live_n1_c1.0_default` | 3/4 | 0.8283 | [0] | True |
| `live_n2_c1.0_default` | 4/4 | 0.993 | [] | False |

## Verdict

- **leftover regression?** NO (6/6)
- **seed0-only knife @ n=1?** YES (c1.0 5/6, c1.5 5/6, fail=[0])
- **n≥2 harden?** YES (c1.0 6/6, c1.5 6/6)
- **live n1→n2?** 3/4 → 4/4
- **Recipe change?** NO — document Music close: avoid n=1/parts0 for close; **n≥2** clears knife
- **Ping user?** YES

JSON: `close_harden_fire21_20260909.json`
