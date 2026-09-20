# v2 vs the 2D winner — release note (additive, weights unchanged)

Question: does this release follow a 2D winner?

Verdict: **PARTIAL**. The core training game follows the named 2D transfer
parent; the v2 head is a honestly-documented native-scale choice, not a 2D
winner. Nothing below changes any weight, catalog entry, sample or evidence
file. This note only makes the relationship explicit.

## Transfer parent (named)

**UniPG `particle_bridge`** — `anneal-routed-particle-error-yue2-v1`
(`--recipe particle_bridge` in `mikkel/sliders-conceptmod`,
`docs/yue2-particle-bridge.md`): model-glue `df70ccb…`, config SHA256
`1ef39a62…`, ParticleGAN `441fdf42`. Its toy (seeds 0/1/7) FAILs at 600 and
HITs both UNI cells, live and EMA, at 3400 and 8000. The recipe name, config
SHA and model-glue ref recorded in `catalog.json` and every
`evidence/particle-gmix-1600-v2/<control>/{manifest,teacher-audit}.json`
identify this parent exactly.

Not the parent: locked `AdvConfig()` (guardrail, unchanged by design),
formulation-leaderboard toy winners without particles
(`faithful_plus_neu`, `rpgan_bcap_plus_neu`), or production Arm B
(`unipolar-rpgan-bcap-yue2-v3`, no particles, different LRs/critic).

## What matches the parent (core game)

`Rp` paired relativistic game with D-first/frozen-D G order, GAN-only (all
aux weights 0, `adv_weight` 1); `b_cap` (L2, threshold 1, coeff 1, every 4th
update ×4, `normalized_paired_error_plus_shared_gaussian`); 128×4 shared
particle cloud with routed 8→4→12→8 bridge; particle VIC (coeff 1, target std
1, eps 1e-4, batch 64); constant LRs G 0.0006 / D 0.0009 / particles 0.006,
Adam (0, 0.999), no weight decay, EMA 0.995; unipolar (`faithful_plus_neu`,
trained scales `[1.0]`, range `[0,1]`, scale-0 exact base, negative
unsupported); rank/alpha 8/8 over 112 AR q/k/v/o branches, critic train-only.
Equations: `FORMULATION.md` / `MATH.md`. Per-run proof: `teacher-audit.json`
+ `status.json` + `training-summary.json`.

## What is v2-native (not a 2D winner)

- **Critic `gmix_t8_w48_l1`** (8 tokens × width 48, 1 layer, 4 heads, score
  bound 8, mean+max pool) instead of the bridge doc's 3×48 MLP. A legal
  trainer option, with native evidence (metal survived; hiphop held-out
  residual RMS 0.997→0.144, gain →0.989), but never crowned on any 2D board —
  and its female sibling collapsed before the h13 fix. "Later trainer
  defaults are not a description of these trained files."
- **Paired-edit normalization** (scales from `std(t−n)` + median-RMS gain;
  critic sees `(g−t)/s`) instead of absolute-positive whitening.
- **Seedbank + batch**: 512 sources (128 seeds × 4 templates, 32-token
  histories), D/G batches of 8 with replacement — not the doc's 4 native
  rows / batch 64.
- **Noise**: `σ₀ = max(E/0.28, 0.03)` (metal 4.11); T = 8000 metal only,
  T = 1600 otherwise; holds 1.0 (pop/hiphop) or 1.3×E (other 13). The
  release does **not** reach 0.03 at update 1600 (metal final σ ≈ 1.537).

## Consequences

- Use native teachers at scales `[0,1]`; `-1` is an unsupported canary.
- Distilled rank-8 LoRAs approximate these exact teachers (see
  `DISTILLATION.md`); conversion is exact before BF16, distillation is an
  approximation.
- No 2D HIT is claimed for the v2 head; 2D HITs belong to the parent toy.
- No retrain, no deletion, no bipolar flip. Full knob-by-knob audit with
  file pointers: `mikkel/sliders-conceptmod`, `docs/yue2-gmix-2d-audit.md`.
