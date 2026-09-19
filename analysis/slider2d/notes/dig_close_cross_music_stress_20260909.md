# Dig: f3d_close knife + cross_axis + Music→toy stressors — 2026-09-09

Host: pop-os CPU @ `435e87363bf1`. Locked recipe; Music proxy n_particles=1; FM0; b_cap=1.
Wall: 351.5s. No Music train. Servers untouched.

## A) f3d_close @ n=1

| cover | PASS | exam mean | span | leak max | knife |
|---:|:---:|---:|---:|---:|:---:|
| 1.0 | 5/6 | 0.8605 | 0.7800 | 0.2024 | yes |
| 1.5 | 5/6 | 0.8911 | 0.6147 | 0.0537 | yes |

### Seed0 harden probes (cover=1.5)

| probe | pass | exam | leak |
|---|:---:|---:|---:|
| steps_1600 | False | 0.3768 | 0.0666 |
| steps_2000 | False | 0.3706 | 0.0340 |
| n_particles_4 | True | 0.9719 | 0.0033 |
| n_particles_12 | True | 0.9829 | 0.0129 |
| alt_seed_5 | True | 0.9950 | 0.0575 |
| alt_seed_6 | True | 0.9977 | 0.0229 |
| alt_seed_8 | True | 0.9952 | 0.0302 |
| alt_seed_9 | True | 0.9954 | 0.0319 |
| alt_seed_10 | True | 0.9936 | 0.0504 |
| alt_seed_11 | True | 0.9889 | 0.0126 |
| geom_jitter | False | 0.3634 | 0.0393 |

## B) cross_axis_rows (hard boundary check)

| n_particles | PASS | exam mean | span | leak max |
|---:|:---:|---:|---:|---:|
| 1 | 0/6 | 0.5759 | 0.0227 | 0.4385 |
| 12 | 0/6 | 0.5848 | 0.0348 | 0.3839 |

## C) NEW Music→toy stressors

- `lyric_span_entangle` @ n=1 cover=1.5: **0/6** mean=0.9887 leak_max=0.2011 bites=True
- `dual_arm_listen_cover_only` (teacher=faithful): **3/3** mean=0.9923 leak_max=0.4565
- same geom + guard: **3/3** mean=0.9289 leak_max=0.0089

## Finding

- Prefer harden path if any seed0 probe recovers 6/6 close; else document flake + multi-seed Music rule.
- cross_axis_rows: if still 0/6 at n=1 and n=12 → **hard boundary** (heterogeneous per-row axis mix does not port).
- New stressors that FAIL under locked recipe are biting Music→toy cells — keep as regression suite.

JSON: `dig_close_cross_music_stress_20260909.json`
