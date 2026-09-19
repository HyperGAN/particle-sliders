# multi-pair / cross-axis stress @ locked 1200+c1.5 (Fire #8)

Host: pop-os CPU @ `435e873`. Locked leftover recipe; FM off; n=12; l2=0.02; b_cap=1.
Seeds: `[0, 1, 2, 3, 7, 42]`.

## Per-pair multi-seed

| pair | family | pass | metric mean | metric span | notes |
|---|---|---:|---:|---:|---|
| `sheet_leftover` | sheet | 6/6 | 0.9302 (kept) | 0.0016 | portable |
| `sheet_gender` | sheet | 6/6 | 0.9955 (kept) | 0.0010 | portable |
| `exam_divergent` | exam | 6/6 | 1.0000 (overlap) | 0.0000 | portable |
| `exam_close` | exam | 6/6 | 0.9939 (overlap) | 0.0156 | portable |
| `exam_unused_e` | exam | 6/6 | 0.9792 (overlap) | 0.0417 | portable |

## Compiled + exam_score by seed

| seed | leftover | gender | divergent | close | unused_e | exam_score | compiled |
|---:|:---:|:---:|:---:|:---:|:---:|---:|---|
| 0 | P | P | P | P | P | 1.0000 | `works` |
| 1 | P | P | P | P | P | 0.9896 | `works` |
| 2 | P | P | P | P | P | 0.9844 | `works` |
| 3 | P | P | P | P | P | 0.9948 | `works` |
| 7 | P | P | P | P | P | 0.9948 | `works` |
| 42 | P | P | P | P | P | 0.9937 | `works` |

## Finding

- Verdict: **`portable_recipe_solid`**
- Compiled WORKS: **6/6**; some=0; fail=0
- exam_score mean=0.9929 min=0.9844 max=1.0000
- Fluke pairs: none
- Portable across pairs: **True**

Do **not** declare 2D done unless leftover sheet is seed-stable (≥6), exam_score strong, no known false locks, AND multi-pair looks solid.

Next: 3D leftover sheet scaffold / highd stress, or fix highd window-mean flake, or LR micro-sweep only if multi-pair is solid
