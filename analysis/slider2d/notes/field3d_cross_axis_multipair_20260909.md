# Field3D cross-axis multipair @ locked 1200+c1.5 (Fire #13)

Host: pop-os CPU @ `435e87363bf1`. Locked leftover recipe; FM off; n=12; l2=0.02; b_cap=1.
Seeds: `[0, 1, 2, 3, 7, 42]`.

## Ask

Does the locked recipe survive **true cross-axis** R3 geometry (`Field3D.row_amps` — different rows on different axis mixes), beyond PairField multi-row exam (#12) and hard_multipair declare/amplitude mismatch?

## Per-cell summary

| cell | PASS | primary mean | primary span | u_kept mean | leak abs max | knife_edge |
|:---|:---:|---:|---:|---:|---:|:---:|
| f3d_leftover | 6/6 | 0.9917 | 0.0015 | 0.9917 | 0.0009 | no |
| f3d_divergent | 6/6 | 0.9210 | 0.0090 | 0.9891 | 0.0008 | no |
| f3d_close | 6/6 | 0.9829 | 0.0098 | 0.9998 | 0.0081 | no |
| f3d_unused_e | 6/6 | 0.9205 | 0.0013 | 0.9824 | 0.0019 | no |
| axis_u_primary | 6/6 | 0.9328 | 0.0096 | 0.9328 | 0.0011 | no |
| axis_content_primary | 6/6 | 0.9288 | 0.0092 | 0.9354 | 0.0097 | no |
| axis_leak_primary | 0/6 | 0.9481 | 0.0099 | 0.9481 | 2.1479 | no |
| cross_axis_rows | 0/6 | 0.5814 | 0.0316 | 0.5814 | 0.3959 | no |
| cross_axis_span_sample | 0/6 | 0.8308 | 0.0218 | 0.8308 | 0.2197 | no |
| cross_axis_mismatch_declare | 0/6 | 0.6432 | 0.0222 | 0.6432 | 0.7396 | no |

## Finding

- Verdict: **`cross_axis_partial`**
- Wall: 664.8s
- Locked recipe **fails solidly (not knife-edge)** on: `axis_leak_primary` (leak≫û), `cross_axis_rows` (heterogeneous per-row axis mix), `cross_axis_span_sample` (live-like span entanglement). `cross_axis_mismatch_declare` fails as expected negative control. Document as Music transfer risk; do **not** weaken portable recipe.

## Transfer interpretation (Music LM)

| Toy knob | Music LM knob | Lesson |
|---|---|---|
| `faithful_guard_e` | leftover gate / `--recipe` | refuse bad `leak_*` YAML; honest unused_e subtracts |
| `cover_weight=1.5` | `--pole_weight` (~1.0 Music posture) | cover still required with gate |
| `b_cap=1` | `b_cap=1` | keep |
| `n_particles=12` | `--parts 0` (do not map) | particle budget is toy; Music stays parts0 |
| `fm_weight=0` | FM / txfm off | keep FM-off |
| residual RpGAN | `lm_adv` mlp | transferable core |

### Cover vs residual under cross-axis

- `axis_u_primary`: primary=0.9328 leak_max=0.0011 pass=6/6
- `axis_content_primary`: primary=0.9288 leak_max=0.0097 pass=6/6
- `axis_leak_primary`: primary=0.9481 leak_max=2.1479 pass=0/6
- `cross_axis_rows`: primary=0.5814 leak_max=0.3959 pass=0/6
- `cross_axis_span_sample`: primary=0.8308 leak_max=0.2197 pass=0/6
- `cross_axis_mismatch_declare`: primary=0.6432 leak_max=0.7396 pass=0/6

## Recipe change?

None adopted — document failing cells only.
