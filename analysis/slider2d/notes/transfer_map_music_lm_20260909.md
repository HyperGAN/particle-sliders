# Transfer map: locked 2D/3D slider toy → Music LM (2026-09-09)

Parallel track (no Cursor). Host: pop-os `/ml2/music/sliders-conceptmod` @ SHA `435e873`
(+ local fires). **GPUs left alone.** Scope: code/docs alignment + cheap CPU
microchecks — not Field3D multi-pair sprint duplication.

## Locked recipe (source of truth)

From `analysis/slider2d/gan_bcap_findings.md` + Fires #1–#12
(`notes/research_log_20260909.md`):

| knob | locked value |
|---|---|
| teacher | `faithful_guard_e` |
| steps | **1200** |
| `cover_weight` | **1.5** |
| `b_cap` | **1.0** (κ=1) |
| `fm_weight` | **0.0** (off) |
| `n_particles` | **≤12** (default 12) |
| `particle_l2` | **0.02** |
| Adam β1 | **0** |
| Field3D | same recipe transfers (Fires #10–#12) |

**Do NOT adopt** `800×cover3.0` (Fire #4 false lock, 1/6 seeds).

## Transfer table

| toy knob (2D `AdvConfig` / `gan.py`) | Music LM analogue | risk if wrong | recommended default |
|---|---|---|---|
| `teacher=faithful_guard_e` | `--lm_target faithful_guard_e` (blend-guards leftover ê; refuses when ê restates axis) | Raw `faithful` / `faithful_raw` copies unused ê → leak ≈0.23 fails sheet | **`faithful_guard_e`** for leftover axes; clean gender may stay `v9` / UNI with no ê |
| `b_cap=1.0` | `lm_adv.CAP_COEFF` / `--adv_reg_coeff` + `CAP_KAPPA` / `--adv_reg_kappa` (both 1.0) | Cap too soft → D races; off → mode collapse / unbounded steepness | **coeff=1, kappa=1** in scaled teacher-RMS coords (`--adv_in scaled`) |
| `fm_weight=0` | `--fm_weight` (provenance default 0; **smoke launcher env default 1**; `gan_v2.Recipe.fm_weight=1`) | Raw FM uncapped by b_cap (arch review #1); Music `feature_matching_loss` has **no normalize** — feature scale free | **FM off (0)** for #94 transfer. If probing FM: normalized features only; never raw. Do not copy smoke/`gan_v2` FM=1 blindly |
| `fm_normalize=True` (toy only) | `gan_v2` critic `normalized_features=True` on repaired/fm_normalized arms; legacy helper is raw batch-mean MSE | Enabling normalize flag without cutting weight still flat on 2D (Fire #6); raw path known-bad | Prefer **off**; if on, require **normalized** features + separate FM grad budget |
| `cover_weight=1.5` | Mode pin on shared residual ≈ `--pole_weight` MSE / HQ residual cover — **not** `--parts` | Zero cover → residual undershoot (kept~0.23); short high-cover false locks | Keep a **modest MSE pin** (~HQ analogue). Do **not** chase kept with 800×high-cover |
| `n_particles≤12` + `ParticlePrior` | Toy: jitter prior around residual. Music `--parts` / `ParticleBatch` = **row miner**, not latent prior | Treating `--parts` as ParticlePrior → wrong dial; raising latent particle count steals residual | Toy: **n≤12, l2=0.02**. Music: **`--parts 0`** unless explicitly mining rows; do not map particle_l2→parts_vic |
| `particle_l2=0.02` | Toy L2 on ParticlePrior; Music `parts_vic` is KL balance on row miners (legacy name) | Confusing VICReg/parts_vic with residual L2 | Keep toy **0.02**. Music miners: leave `parts_vic` alone unless mining on |
| Adam β1=0 + D lr 1.5× | `--gan_beta1 0` / `--adv_beta1 0`; `D_LR_MULT=1.5` / `--adv_lr` default | β1>0 on sparse D grads slowed ParticleGAN; mismatched D/G lr | **β1=0**, D lr **1.5×** LoRA lr |
| span/end real cloud | TX `--adv_arch tx` + `--adv_readout mean_last` (lyric span + audio-start) | MLP-only last-token blind to prefix; mean-only readout merges EOS channel | Prefer **tx + mean_last** for lyric recipes |
| steps=1200 (toy) | Music smoke card historically **120** updates (listen-preferred); not the same budget | Equating 1200 toy steps to Music step count → over/under-train | Transfer **ratios**, not step count. Re-tune Music length with listen + multi-seed |
| Field3D û/ĉ/ê leftover | Live leak_* + leftover-gated teachers; highd leftover tests | Dropping gate on multi-row / content↔ê mixes reintroduces leak | Keep **leftover gate** when transferring multi-row/exam cells |

## Wiring gap (important)

Current working-tree `conceptmod/textsliders/train_lm_slider_music3.py` **no longer
exposes** `--adv_weight` / `--fm_weight` / `--adv_reg_*` (microcheck
`current_train_missing_adv_cli`). Those flags still exist in:

- provenance trainers under `models/gan-bcap-repair/*/provenance/...`
- `scripts/train_lm_gan_bcap.sh` (still passes them)
- primitives in `conceptmod/textsliders/lm_adv.py` (`rp_*`, `cap_penalty`, `feature_matching_loss`)
- alternate path `conceptmod/textsliders/gan_v2/` (`Recipe.fm_weight=1` by default)

**Transfer implication:** map defaults into the live entrypoint you actually run
(`gan_v2` or a restored adv CLI). Do not assume `train_lm_slider_music3.py`
parser matches the smoke script today.

## Cheap CPU microchecks (this track)

```text
CUDA_VISIBLE_DEVICES= pytest tests/test_lm_2d_adv.py -q
# → 10 passed in ~97s

CUDA_VISIBLE_DEVICES= PYTHONPATH=. python /tmp/transfer_map_microchecks.py
# → 38 pass / 0 fail  (JSON: /tmp/transfer_map_microchecks.json)
```

Validated: AdvConfig lock cells; `lm_adv` CAP_COEFF/KAPPA/D_LR_MULT; Music FM
helper lacks normalize and scales with feature magnitude; toy normalized FM is
scale-invariant; `ParticleBatch` is a row sampler not ParticlePrior; smoke /
`gan_v2` FM=1 disagree with locked FM=0; provenance CLI defaults fm=0 /
adv_reg_coeff=1; docs forbid 800×c3.0.

## Top 5 transfer recommendations

1. **Keep FM off (`--fm_weight 0`)** when porting #94. Smoke/`gan_v2` FM=1 is a
   listen-era default, not the locked 2D recipe. Raw Music FM is uncapped by b_cap.
2. **Keep b_cap at coeff=1, kappa=1** on `--adv_in scaled` inputs (`CAP_COEFF` /
   `--adv_reg_coeff`).
3. **Keep leftover teacher `faithful_guard_e`** (or equivalent gate) — cover alone
   does not stop ê copy.
4. **Do not map `--parts` to toy `n_particles`.** Music parts are row miners; toy
   particles are residual jitter. Prefer parts=0; if residual undershoots, cut
   particle count / raise particle_l2 *in the toy*, or raise a true mode-pin
   (pole/cover) on Music — not miner count.
5. **Reject short-budget high-cover false locks** (800×c3.0). Require multi-seed
   PASS with margin (kept ≫ 0.90). Also: restore or explicitly route through an
   entrypoint that still accepts adv/FM flags before claiming a live Music transfer.

## Non-goals / leave alone

- No Music LM GPU trains; GPU0 ~30 GB reserved left untouched.
- No Field3D multi-pair sprint duplication (already solid in Fires #10–#12).
- Energy/MMD `objective_20260905` is a rejected replacement, not this map.
