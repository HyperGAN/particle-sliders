# FM-on-normalized vs FM-off (2026-09-09 fire #6)

SHA `435e873` on box-cpu (laptop local-exec unavailable; pop-os Tailscale timed out).

Locked recipe: steps=1200, cover_weight=1.5, b_cap=1.0, n_particles=12, teacher=`faithful_guard_e`, FM off by default.

## Grid (seed=0)

| label | fm_weight | normalize | kept | leak | garble | swing | pass | sec |
|---|---:|:---:|---:|---:|---:|---:|:---:|---:|
| `baseline_fm0` | 0.0 | True | 0.9306 | -0.0003 | 0.0009 | 1.0899 | PASS | 7.1 |
| `fm_norm_0.1` | 0.1 | True | 0.9298 | -0.0002 | 0.0009 | 1.0881 | PASS | 7.4 |
| `fm_norm_0.5` | 0.5 | True | 0.9305 | -0.0001 | 0.0009 | 1.0896 | PASS | 7.0 |
| `fm_norm_1.0` | 1.0 | True | 0.9295 | +0.0001 | 0.0009 | 1.0875 | PASS | 7.4 |
| `fm_raw_0.5_control` | 0.5 | False | 0.9282 | -0.0001 | 0.0010 | 1.0849 | PASS | 7.1 |

### Residual / train fields (seed=0)

| label | pole_rel_err± | covered | cap | collapse | sheet_dir_kept |
|---|---|:---:|---:|---:|---:|
| `baseline_fm0` | 0.0198/0.0255 | True | 0.0021 | 0.0840 | 0.9704 |
| `fm_norm_0.1` | 0.0202/0.0266 | True | 0.0023 | 0.0845 | 0.9701 |
| `fm_norm_0.5` | 0.0197/0.0261 | True | 0.0023 | 0.0833 | 0.9703 |
| `fm_norm_1.0` | 0.0212/0.0310 | True | 0.0025 | 0.0806 | 0.9680 |
| `fm_raw_0.5_control` | 0.0236/0.0326 | True | 0.0024 | 0.0790 | 0.9650 |

## Multi-seed

Ran seeds `{0,1,2,3,7,42}` for competitive FM weights vs FM-off.

- fm_weight=0.0: **6/6 PASS**, kept mean `0.9303`, span `0.0017`

| seed | kept | leak | pass |
|---:|---:|---:|:---:|
| 0 | 0.9306 | -0.0003 | PASS |
| 1 | 0.9309 | -0.0000 | PASS |
| 2 | 0.9308 | -0.0004 | PASS |
| 3 | 0.9292 | -0.0006 | PASS |
| 7 | 0.9302 | -0.0000 | PASS |
| 42 | 0.9301 | +0.0004 | PASS |

- fm_weight=0.1: **6/6 PASS**, kept mean `0.9303`, span `0.0009`

| seed | kept | leak | pass |
|---:|---:|---:|:---:|
| 0 | 0.9298 | -0.0002 | PASS |
| 1 | 0.9307 | -0.0001 | PASS |
| 2 | 0.9306 | -0.0001 | PASS |
| 3 | 0.9298 | +0.0000 | PASS |
| 7 | 0.9305 | -0.0004 | PASS |
| 42 | 0.9305 | +0.0003 | PASS |

- fm_weight=0.5: **6/6 PASS**, kept mean `0.9304`, span `0.0007`

| seed | kept | leak | pass |
|---:|---:|---:|:---:|
| 0 | 0.9305 | -0.0001 | PASS |
| 1 | 0.9303 | +0.0003 | PASS |
| 2 | 0.9302 | -0.0004 | PASS |
| 3 | 0.9309 | -0.0002 | PASS |
| 7 | 0.9304 | +0.0000 | PASS |
| 42 | 0.9303 | +0.0004 | PASS |

- fm_weight=1.0: **6/6 PASS**, kept mean `0.9302`, span `0.0011`

| seed | kept | leak | pass |
|---:|---:|---:|:---:|
| 0 | 0.9295 | +0.0001 | PASS |
| 1 | 0.9305 | +0.0001 | PASS |
| 2 | 0.9305 | +0.0004 | PASS |
| 3 | 0.9299 | +0.0002 | PASS |
| 7 | 0.9303 | -0.0002 | PASS |
| 42 | 0.9303 | +0.0010 | PASS |

## Verdict

**keep_fm_off_flat**: FM-on PASSes but is flat vs off (mean 0.9304 vs 0.9303); no recipe change.

Raw FM control (w=0.5, normalize=False): kept=0.9282 PASS — known-bad uncapped path (document only).

## Next thread

particle_l2 micro-sweep at high n, or multi-pair/cross-axis stress, or exam_score at locked recipe (prefer particle_l2 micro-sweep @ n=12 first)
