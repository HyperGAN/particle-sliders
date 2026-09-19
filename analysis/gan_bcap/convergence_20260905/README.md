# Convergence criteria study — 2026-09-05

The user paused catalog training and assigned GPU 1 to this study. Seven catalog controls are complete; Distortion has a full recoverable state at update 300. The catalog service is stopped and its default-target symlink removed. Research completion must not resume the catalog automatically.

## Empirical result

Six disposable probes used the unchanged trainer and original full GAN states. The source critic was frozen for generator optimization; discriminator searches used separate copies and the original detached training spans. Source states and frozen critics were verified unchanged.

| Source | G loss reduction in 20 updates | Best D objective reduction | G initial → best |
| --- | ---: | ---: | ---: |
| early300 | 84.35% | 0.2919717 | 1.8478 → 0.2891 |
| original600 | 92.32% | 0.76232378 | 2.9972 → 0.2301 |
| bounded1350 | 91.60% | 0.058757511 | 4.2999 → 0.3613 |
| failed900 | 5.42% | 2.1456348e-07 | 123.1206 → 116.4474 |
| original600-lr1 | 94.98% | 0.76232378 | 2.9972 → 0.1505 |
| bounded1350-lr1 | 73.93% | 0.058757511 | 4.2999 → 1.1209 |

G uses quarter LR except the two full-LR checks, restored optimizer moments, and actual parameter-step cap 2. D uses 60 updates, both quarter/full LR and fresh/restored moments. The generator objective includes the original adversarial, batch feature-matching, end-margin and zero-scale terms. Training logs are pre-update, so each probe executes 21 updates to measure losses at the initial point and after updates 1–20. The final 21-update weights are disposable diagnostic artifacts, never catalog candidates.

The 600 and 1350 games have substantial unilateral improvement available. The historical failed 900 critic has almost none because it already separates fake and real nearly perfectly. Its generator objective remains over 116. Low movement or small critic improvement alone must not imply convergence. Finite best-response searches give lower bounds on available improvement; failure to find improvement does not prove stationarity.

## Working operational rule

1. Track instability separately and retain the previous usable checkpoint when it occurs.
2. Lock an incumbent, metric, prompt design and comparison budget before collecting confirmation samples.
3. Compare candidates on identical prompt/seed fixtures. Average seeds within each prompt before estimating uncertainty; seeds do not substitute for prompt coverage.
4. Require at least eight prompt groups and four seeds per group. Use a one-sided paired t upper bound across prompt means, allocating alpha across the predeclared comparisons. This relies on representative independent prompt groups and an approximate prompt-level sampling model.
5. Stop extending only after three evaluated checkpoints have upper gain bounds at or below 0.05. Wide intervals request more evaluation; absence of a significant gain is not evidence of no worthwhile gain.
6. Separate quality-validated completion from a statistical plateau. Failed concept/preservation checks mean stalled; an unvalidated judge means validation is still required.
7. Reserved final examples must not have participated in the preceding selection. If their result changes the winner, that set becomes development validation and a new reserved check is required.
8. A resource cap is a budget stop. It never changes the scientific label to converged.

The 0.05 tolerance is inherited from the fixed research score and is a working operational threshold, not a calibrated perceptual just-noticeable difference. Game thresholds (G fractional gain 5%, D absolute gain 0.02) are diagnostic tolerances only. They do not enter the quality stopping rule.

## Prospective audio test

The running audio study locks 600 as incumbent and compares saved 1050/1200/1350 weights. It uses eight entirely new arrangements/lyric sheets and four seeds, 32 matched samples per checkpoint. If a comparison remains inconclusive, it adds eight predeclared prompts, reaching 64 samples per checkpoint. The maximum does not force a convergence decision. CPU measurement overlaps subsequent GPU rendering. All first renders are retained; the V2 metric is unchanged.

The audio judge is not yet calibrated as an absolute voice classifier. The report keeps negative/positive description margins and reference results visible and will not automatically call a checkpoint quality-validated. This remains required work before resuming fully automatic catalog acceptance.

## Validation and reproducibility

- 11 focused tests passed (convergence probes/criteria and parameter-step limits).
- 20,000 Gaussian prompt-level simulations at the worthwhile-gain boundary yielded a 4.95% probability of at least one false no-gain decision across 12 comparisons, under the stated model. This is not the error rate of the full three-checkpoint stopping policy and does not validate the music metric.
- SciPy 1.16.3 is installed only under `/ml2/music/.cache/gan-convergence/python`, with no dependency changes to the training environment.
- All 16 prospective prompt files passed the existing artist-name validator.
- Core trainer, GAN, step-limit and earlier experiment sources remain unchanged.

The game-theoretic motivation is related to [GAN duality-gap monitoring](https://proceedings.neurips.cc/paper_files/paper/2019/hash/692baebec3bb4b53d7ebc3b9fabac31b-Abstract.html) and [proximal duality gaps](https://proceedings.mlr.press/v139/sidheekh21a.html). The actual training game here includes asymmetric feature matching and regularizers; this probe is not claimed to instantiate those formal guarantees.

Live page: http://100.90.104.57:8888/gan-convergence-20260905/index.html

## Paused catalog and background work

The research audio and page services are installed under the user systemd configuration and survive logout. The audio service has at most three rapid failure retries, reusing completed artifacts. The catalog service remains inactive and has no autostart symlink.

When the user later resumes the catalog, restore Distortion from its full 300-update state into a fresh attempt name. Preserve its original 300-update inference snapshot as an eligible comparison alongside later snapshots; the current controller only searches the resumed folder for early weights, so that cross-attempt candidate needs handling before resumption.
