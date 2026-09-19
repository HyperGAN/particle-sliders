# particle_l2 micro-sweep @ locked 1200+c1.5 n=12 (Fire #7)

Host: pop-os CPU @ `435e873`. Default `particle_l2=0.02`.

## Seed-0 grid

| particle_l2 | kept | leak | garble | swing | residual_norm | pass | sec |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 0.0 | 0.9288 | -0.0000 | 0.0009 | 1.0859 | — | PASS | 8.1 |
| 0.005 | 0.9302 | +0.0001 | 0.0009 | 1.0889 | — | PASS | 7.5 |
| 0.01 | 0.9303 | +0.0001 | 0.0009 | 1.0888 | — | PASS | 7.4 |
| 0.02 | 0.9295 | +0.0001 | 0.0009 | 1.0874 | — | PASS | 7.7 |
| 0.05 | 0.9302 | -0.0002 | 0.0009 | 1.0890 | — | PASS | 7.7 |
| 0.1 | 0.9308 | -0.0001 | 0.0009 | 1.0900 | — | PASS | 7.4 |

## Multi-seed

- l2=0.0: 6/6 PASS, kept mean 0.9303 (span 0.0025)
- l2=0.02: 6/6 PASS, kept mean 0.9302 (span 0.0016)
- l2=0.05: 6/6 PASS, kept mean 0.9306 (span 0.0019)
- l2=0.1: 6/6 PASS, kept mean 0.9306 (span 0.0010)

## Verdict

**keep_default_0.02** — default l2=0.02 seed-stable 6/6 mean kept 0.9302 span 0.0016; l2=0 ablation 6/6 mean kept 0.9303 (Δ vs 0.02 = +0.0000); best multi-seed l2=0.1 6/6 mean 0.9306 (Δ=+0.0004); flat vs default (Δkept ≪ 0.01) — no recipe change; seed0 grid: 0.0->0.9288P, 0.005->0.9302P, 0.01->0.9303P, 0.02->0.9295P, 0.05->0.9302P, 0.1->0.9308P

## Next

multi-pair/cross-axis stress, or exam_score at locked recipe, or LR/b_cap micro-sweep that preserves seed stability
