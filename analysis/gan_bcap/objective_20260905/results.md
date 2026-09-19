# Objective experiments — complete results

These are research screens, not a universal stability result. Every completed run is included. Gaussian HQ and core width are distribution metrics; saved-span error and live teacher error are not audio-quality scores.

| Gaussian run | Steps | Batch | Schedule | EMA modes | EMA HQ | EMA core width / truth | EMA mode TV | Live HQ |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| bcap-batch32-seed7 | 7000 | 32 | delayed | 94 | 0.2292 | 6.714 | 0.1848 | 0.2461 |
| bcap-constant-batch32-seed23 | 14000 | 32 | constant | 16 | 0.0353 | 9.941 | 0.2441 | 0.0391 |
| bcap-constant-seed7 | 7000 | 256 | constant | 100 | 0.7790 | 1.623 | 0.1159 | 0.2433 |
| bcap-fixed-prior-seed7 | 7000 | 256 | delayed | 43 | 0.0777 | 9.595 | 0.1205 | 0.0763 |
| bcap-seed7 | 7000 | 256 | delayed | 100 | 0.9876 | 0.868 | 0.1189 | 0.9828 |
| energy-power0.25-seed7 | 7000 | 256 | delayed | 100 | 0.3973 | 4.511 | 0.0316 | 0.4136 |
| energy-power0.5-seed7 | 7000 | 256 | delayed | 100 | 0.2643 | 6.690 | 0.0316 | 0.2831 |
| energy-power1.5-seed7 | 7000 | 256 | delayed | 9 | 0.0250 | 11.612 | 0.0639 | 0.0258 |
| energy-seed7-repaired-audit | 7000 | 256 | delayed | 62 | 0.0546 | 10.107 | 0.0383 | 0.0575 |
| learned-energy-batch1024-seed7 | 7000 | 1024 | delayed | 92 | 0.5292 | 3.861 | 0.1467 | 0.5227 |
| learned-energy-seed7 | 7000 | 256 | delayed | 100 | 0.7499 | 4.112 | 0.1249 | 0.7406 |
| learned-energy-spectral2-seed7 | 7000 | 256 | delayed | 26 | 0.4015 | 4.977 | 0.5399 | 0.3989 |
| learned-energy-vicreg-seed7 | 7000 | 256 | delayed | 100 | 0.7297 | 4.929 | 0.1250 | 0.7149 |
| learned-mmd-seed7 | 7000 | 256 | delayed | 100 | 0.6903 | 3.641 | 0.1309 | 0.6930 |
| mmd-batch1024-constant-seed7 | 7000 | 1024 | constant | 100 | 0.3576 | 3.533 | 0.0274 | 0.1133 |
| mmd-batch1024-extended-seed101 | 28000 | 1024 | delayed | 100 | 0.9336 | 0.991 | 0.0337 | 0.9293 |
| mmd-batch1024-extended-seed23 | 28000 | 1024 | delayed | 100 | 0.9356 | 0.994 | 0.0328 | 0.9275 |
| mmd-batch1024-extended-seed7 | 28000 | 1024 | delayed | 100 | 0.9355 | 0.989 | 0.0276 | 0.9285 |
| mmd-batch1024-seed7 | 7000 | 1024 | delayed | 100 | 0.8224 | 0.959 | 0.0272 | 0.8288 |
| mmd-batch256-extended-seed7 | 28000 | 256 | delayed | 100 | 0.8000 | 1.162 | 0.0293 | 0.7929 |
| mmd-seed7 | 7000 | 256 | delayed | 100 | 0.4497 | 3.492 | 0.0321 | 0.4684 |
| r1r2-constant-batch32-seed23 | 14000 | 32 | constant | 21 | 0.0295 | 11.246 | 0.2574 | 0.0232 |
| r1r2-seed7 | 7000 | 256 | delayed | 100 | 0.9546 | 0.892 | 0.1259 | 0.9539 |

Mode TV is total-variation error from uniform nearest-mode probabilities, including tail samples. At 20,000 independent uniform assignments its simulated sampling reference averages .0281. The initial `energy-seed7` run hit an audit-argument error at final evaluation. It has no completed checkpoint and was rerun as `energy-seed7-repaired-audit`; its intermediate log is retained. The constant batch-32 seed-23 stress arms jointly change batch, schedule, seed and horizon. The later seed-7 cap controls isolate removing decay (live HQ .9828→.2433) and changing batch 256→32 (EMA HQ .9876→.2292). At fixed MMD batch 1024, removing decay changes EMA HQ .8224→.3576 and live HQ .8288→.1133.

| Saved-span game | Seeds | Mean relative error at 1200 | Range |
| --- | ---: | ---: | --- |
| b_cap | 3 | 1.1999 | 1.0793–1.3713 |
| b_cap_fm | 3 | 1.1971 | 1.1193–1.2793 |
| conditional_energy | 3 | 0.0098 | 0.0096–0.0099 |
| r1r2_0.02 | 3 | 1.2219 | 1.0687–1.3982 |
| r1r2_0.1 | 3 | 1.1232 | 1.1028–1.1563 |
| r1r2_1.0 | 3 | 1.1659 | 1.1424–1.1860 |

The span canary optimizes free hidden values from a 64-coordinate slice of archived states. It uses fixed source-600 outputs as critic context, not the actual neutral hidden context. It is a falsification fixture, not a LoRA capacity or audio benchmark. A discriminator loss near log(2) with large target error is not successful matching.

| Real LoRA continuation | Teacher branches | Imported G moments | Train error before → after | Heldout error before → after | Shortened / rejected proposals | Maximum parameter step |
| --- | --- | --- | --- | --- | ---: | ---: |
| conditional-energy-research-20260905 | conditional | yes | 0.6633 → 0.4329 | 0.7988 → 0.7333 | 0 / 0 | 0.1373 |
| conditional-energy-cfg-research-20260905-attempt2 | conditional, unconditional | no | 0.9223 → 0.4275 | 0.9515 → 0.6433 | 107 / 0 | 1.4416 |
| conditional-mmd-cfg-research-20260905 | conditional, unconditional | no | 0.9223 → 0.7598 | 0.9515 → 0.8799 | 89 / 21 | 1.6202 |

All live runs start from the same source-600 LoRA. The second energy trial changes both branch coverage and optimizer initialization; it is a combined implementation test, not an isolated CFG ablation. The fixed-MMD trial uses the same fresh moments and both branches as the second energy trial. The first CFG attempt was stopped before training to move source capture ahead of model I/O. The two heldout prompts are diagnostic development fixtures. No run here establishes free-running convergence.
