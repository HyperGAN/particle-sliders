# YuE2 v2 (particle-gmix-1600-v2) vs the 2D winner — audit

Question from Mikkel: does the released Hub model
[`ntc-ai/yue2-concept-sliders`](https://huggingface.co/ntc-ai/yue2-concept-sliders)
(`particle-gmix-1600-v2`, 16 controls, final EMA at update 1600) follow a 2D winner?

Short answer: **PARTIAL**. The released training formulation follows the named
2D transfer parent for its core ParticleGAN game, but the v2 critic,
normalization, noise schedule, seedbank and batch size are native-scale
choices validated on native — not 2D winners. No retrain is proposed; the fix
is an additive honesty note (this file, plus the Hub-side copy
`V2_VS_2D_WINNER.md`). No bipolar default flip. v2 is kept as-is.

## Intended transfer parent (named)

**UniPG `particle_bridge`** — `anneal-routed-particle-error-yue2-v1`
(`--recipe particle_bridge`), documented in
[`docs/yue2-particle-bridge.md`](yue2-particle-bridge.md):

- Reference: model-glue `df70ccb2ca8f532bdcc07a343fd12bec77362523`,
  config SHA256 `1ef39a623505691b8710cd37cb768452cd79f6666d109614af297e9d270d88bb`,
  pinned ParticleGAN `441fdf42`.
- 2D result: routed low-rank student + PairField backend, seeds 0/1/7 —
  FAIL at 600, **HIT both UNI cells (divergent + close), live and EMA, at 3400
  and 8000** (off-caption 0, neutral hold 1). Same weights fail the bipolar
  exam (expected: `-1` is an unscored UNI canary there).

Why not the other candidates:

- **Locked Field3D / `AdvConfig()`**: untouched by design (bridge doc:
  "existing recipes, Music bipolar `ARM_B`, live `--lm_target v9`, and locked
  `AdvConfig()` defaults are unchanged"). It is a guardrail, not the parent.
- **Formulation-leaderboard UNI winners** (`faithful_plus_neu`,
  `rpgan_bcap_plus_neu`, both HIT in
  [`FORMULATION_LEADERBOARD_UNIPOLAR.md`](FORMULATION_LEADERBOARD_UNIPOLAR.md)):
  toy winners **without particles**. The Hub release is particle-based, so
  neither is its parent.
- **ParticleGAN-faithful Arm B** (production
  `unipolar-rpgan-bcap-yue2-v3`: MLP 2×256 critic, raw-positive teacher, no
  particles, G 5e-4 / D 7.5e-4, 600 steps; FAIL toys at 600/1200, PASS at
  3400 — see [`MUSIC_UNIPOLAR_GAN_NEXT.md`](MUSIC_UNIPOLAR_GAN_NEXT.md),
  [`yue2-arm-b-verification.md`](yue2-arm-b-verification.md)): the Hub release
  differs on particles, critic, teacher scaling and LRs. Not the parent.

Hub provenance agrees on the parent: `catalog.json` / per-run `manifest.json` /
`teacher-audit.json` all record `name: anneal-routed-particle-error-yue2-v1`,
the same `config_sha256` and `model_glue_reference`, and
`generator_objective: paired_error_rpgan_plus_particle_vic`.

## Knob-by-knob: Hub v2 vs the 2D bridge spec

Sources: Hub `FORMULATION.md` (= `MATH.md`), `catalog.json` (female record),
`evidence/particle-gmix-1600-v2/metal/{manifest,status,teacher-audit}.json`,
`training-summary.json` vs repo `docs/yue2-particle-bridge.md`,
`conceptmod/textsliders/{yue2_particle_bridge,particle_bridge_gan}.py`,
`train_lora_yue2_arm_b.py`, UniPG scoreboards, critic-search records.

| Knob | 2D bridge spec (parent) | Hub v2 (released) | Verdict |
|---|---|---|---|
| Rp paired game, GAN-only | `mean softplus(D(fake)-D(real))` / `mean softplus(D(real)-D(fake))`, D-first then frozen-D G, no MSE/FM/ending/hold/anchor/reconstruction | Same objective; all aux weights 0, `adv_weight` 1, same D→G order | **MATCH** |
| `b_cap` | exact autograd both inputs, L2 threshold 1, coeff 1, every 4th update ×4 | `adv_b_cap` 1.0, `adv_reg_kappa` 1.0, `penalty_lazy_k` 4, `penalty_method` autograd, `penalty_anneal` none, `cap_coordinates normalized_paired_error_plus_shared_gaussian` | **MATCH** |
| Particles 128×4 | 128 learned × 4, std-normal init, shared cloud, softmax route `qPᵀ/√4`, VIC sample 64 w/o replacement | `parts` 128, `particle_dim` 4, same init/routing/sharing, `particle_vic_batch` 64 | **MATCH** |
| VIC | std hinge + off-diag covariance, coeff 1, target std 1, eps 1e-4, cov/dim | Same (`vicreg_weight` 1, `particle_vic_target_std` 1, `particle_vic_eps` 1e-4; equation in FORMULATION.md matches `particle_vic()`) | **MATCH** |
| LRs / optim / EMA | G 0.0006 / D 0.0009 / particles 0.006, Adam (0, 0.999), no wd, constant, EMA 0.995 incl. routers+cloud | Identical values in catalog + `training-summary.json` (`g_lr`/`d_lr`/`particle_lr`); `schedule` constant; EMA 0.995; release = final EMA | **MATCH** |
| Critic | 3 hidden × 48 LeakyReLU-0.2 MLP (bridge doc) | `gmix_t8_w48_l1`: full-state linear → 8×48 tokens + positions, 1× 4-head attention block, mean+max RMS pool, `8·tanh(score/8)` | **DRIFT** — legal `--critic gmix` option in current code, but never a 2D winner (critic-search scoreboard: no eligible winner; editnorm compare: metal gmix survived, female gmix collapsed @200 pre-h13; disc sweep tested mix/hybrid/patch/bottleneck, not this gmix as winner). Native validation only (metal survived, hiphop held-out probes). Hub is honest that "later trainer defaults are not a description of these trained files". |
| Polarity | propose-only unipolar bridge; toy gated UNI (+1 cover / 0 hold, −1 canary) | `polarity` unipolar, `lm_target faithful_plus_neu`, `trained_scales [1.0]`, range [0,1], negative strengths unsupported, −1 canary | **MATCH** |
| Noise anneal | `σ(t) = 0.03^min(t/8000,1)`, start 1.0, T = 8000, no hold | `σ₀ = max(E/0.28, 0.03)` (metal 4.11, female 5.29); T = 8000 metal only, T = 1600 others; holds: pop/hiphop 1.0, other 13 at 1.3×E; release never reaches 0.03 at 1600 (metal final σ ≈ 1.537) | **DRIFT** — honestly documented in FORMULATION.md ("three recorded schedules", "does NOT reach 0.03"); matches current `noise_std()` + CLI (`noise_decay_steps = steps`) but not the bridge-doc anneal |
| Teacher contract | `T(h) = (h − mean_train_pos)/std_train_pos`, stats from absolute positives | Paired-edit whitening: scales from `std(t−n)` + median-RMS gain to 1, critic sees `(g−t)/s` (mean cancels); 512 seedbank sources (128 seeds × 4 templates, 32-token histories, D/G batches of 8 with replacement) vs doc's 4 native rows / batch 64 | **DRIFT** — documented evolution; matches current `register_paired_error_norm()` + `prepare()` + CLI overrides (`--sample_seeds 128 --history_tokens 32 --adv_batch 8`), not the original doc contract |
| Scales / adapter | rank/alpha 8/8, 112 AR q/k/v/o branches, shared cloud, train at +1, scale-0 exact base, critic train-only | Identical (`adapter_format …ar-v1`, rank/alpha 8/8, `zero_behavior exact_base_by_adapter_scale`, same inference equations) | **MATCH** |

Missing from the release (by design, not an omission): per-run 2D gate
scores — native runs are fixed-budget (1600 EMA, no listening selection);
2D HIT claims live in the bridge doc toy table, not in Hub evidence. No MISSING
knob changes the formulation verdict.

## Verdict

**PARTIAL — follows the named 2D winner for the game, not for the v2 head.**

The core (`Rp`, `b_cap`, 128×4 particles, VIC, LRs, polarity, scales, EMA,
D→G order, GAN-only) is the UniPG `particle_bridge` transfer, byte-traceable
via recipe name + config SHA + model-glue ref. The v2 head (gmix critic,
paired-edit norm, seedbank-512/batch-8, T=1600 + holds) is a native-scale
configuration with native evidence (per-run traces/audits, metal survival,
hiphop held-out residual 0.997→0.144 / gain 0.989) and honest Hub math — but it
was never crowned on any 2D board, and one of its siblings (female gmix
pre-h13) collapsed.

## Fix applied (additive, no retrain, no bipolar flip)

- Weights/catalog/prompts/samples/evidence untouched — teacher-audit, status,
  manifest and catalog agree with each other and with the code paths that
  produce them.
- This file is the permanent repo audit note.
- Hub-side copy `V2_VS_2D_WINNER.md` (same verdict + table pointers) is
  proposed as an additive commit on `ntc-ai/yue2-concept-sliders`; it deletes
  nothing and changes no weight, catalog, README or MATH content. If the Hub
  write window lapses, the Hub-ready text ships in this PR unchanged.
- Do not invent a bipolar Music default flip; do not delete v2; retrain only
  if Mikkel's tip requires a different formulation (it does not — the
  formulation is self-consistent and honestly documented).
