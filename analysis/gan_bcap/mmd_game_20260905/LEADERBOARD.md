# MMD game leaderboard

Fixed render-heuristic-v2; two prompts × seeds 7/23, 20 seconds, GPU 1. These are repeated development fixtures. The score is a heuristic, not a human-quality certificate.

Overall leader: **gan_v2_baseline**. MMD personal best: **mmd_seed_720**.
Active round: none. Completed rounds: 4. Historical opponents still awaiting this board: 25.

| Rank | Entry | Family | Score ↑ | Worst clip | Eligible | Round |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | gan_v2_baseline | reference | 0.799029 | 0.434939 | True | bootstrap |
| 2 | gan_fm_cap_1950 | reference | 0.796622 | -0.301422 | True | bootstrap |
| 3 | mmd_seed_720 | mmd | 0.749177 | 0.154681 | True | bootstrap |
| 4 | round0001_adaptive | mmd | 0.749177 | 0.154681 | True | round-0001 |
| 5 | round0004_strength125 | calibration | 0.589273 | -0.340354 | True | round-0004 |
| 6 | round0002_float32 | mmd | 0.530924 | -0.598124 | True | round-0002 |
| 7 | gan_bounded_1200 | reference | 0.529191 | -0.584009 | True | bootstrap |
| 8 | round0003_coverage | mmd | 0.494746 | 0.016326 | True | round-0003 |
| 9 | gan_original_600 | reference | 0.367456 | -0.084563 | True | bootstrap |

Historical scores from different fixtures are listed only in `opponents.json`; they are not leaderboard entries. Each entry retains its component scores, all four examples, control hashes and score reports. A higher mean may coexist with individual regressions; those remain visible.

## Next opponents to evaluate

- `gan_v2_decay` — `/ml2/music/sliders-conceptmod/models/gan-v2/decay-660-20260905/decay-660-20260905_step660.safetensors`
- `gan_v2_ema` — `/ml2/music/sliders-conceptmod/models/gan-v2/repaired-660-20260905/repaired-660-20260905_ema_step660.safetensors`
- `gan_v2_fm_capped` — `/ml2/music/sliders-conceptmod/models/gan-v2/fm_capped-660-20260905/fm_capped-660-20260905_step660.safetensors`
- `gan_v2_fm_normalized` — `/ml2/music/sliders-conceptmod/models/gan-v2/fm_normalized-660-20260905/fm_normalized-660-20260905_step660.safetensors`
- `gan_v2_repaired` — `/ml2/music/sliders-conceptmod/models/gan-v2/repaired-660-20260905/repaired-660-20260905_step660.safetensors`
- `learned_energy_cfg_720` — `/ml2/music/sliders-conceptmod/models/conditional-energy-cfg-research-20260905-attempt2/conditional-energy-cfg-research-20260905-attempt2_step720.safetensors`
- `learned_energy_conditional_720` — `/ml2/music/sliders-conceptmod/models/conditional-energy-research-20260905/conditional-energy-research-20260905_step720.safetensors`
- `gan_bounded_1050` — `/ml2/music/sliders-conceptmod/models/gan-v2/baseline-to1050-20260905/baseline-to1050-20260905_step1050.safetensors`

## Completed rounds

- round-0001: no new record; next experiment: Test bf16 rounding as the obstruction: compare repeated losses and directional finite differences at MMD 720 under bf16 and float32, then train one float32 paired-MMD continuation with fixed histories and a separately calibrated fixed objective; retain the locked bf16 audio judge.
- round-0002: no new record; next experiment: From retained MMD 720, test fixed history coverage: collect one extra seeded history per training prompt with that frozen parent active at +1, archive it before optimization, and mix it equally with the original neutral histories. Recompute and freeze float32 teachers and calibration, use fresh AdamW, and train one 150-attempt paired-MMD candidate. Keep prompts, rank, kernel and locked audio judge unchanged; measure original/added-history losses separately. This is a frozen-history surrogate, not full trajectory-distribution MMD.
- round-0003: no new record; next experiment: Test one preset 1.25 strength calibration of the strongest measured adapter, with zero optimizer updates, separate calibration-family bookkeeping, and unchanged locked renderer at +1. Confirm any new overall lead on fresh rows 4/5, seeds 515/727, 30 seconds against the starting leader.
- round-0004: no new record; next experiment: Deferred until user review. No new experiment or opponent benchmark is queued; inspect the matched audio and per-clip concept/production regressions before choosing a follow-up.
