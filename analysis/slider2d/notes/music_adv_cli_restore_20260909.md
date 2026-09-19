# Music adv/FM CLI restore (2026-09-09)

Host: pop-os `/ml2/music/sliders-conceptmod` @ SHA `435e873` (+ local restore).
**No Music LM GPU train.** nano-work-server left on GPU 0.

## What was broken

`transfer_map_music_lm_20260909.md` microcheck `current_train_missing_adv_cli`:
live `conceptmod/textsliders/train_lm_slider_music3.py` no longer exposed the
ParticleGAN knobs that `scripts/train_lm_gan_bcap.sh` and provenance trainers
pass. Diff vs launcher / provenance / `lm_adv.py`:

| surface | before restore |
|---|---|
| live `train_lm_slider_music3.py` | **missing** `--adv_weight`, `--fm_weight`, `--fm_mode`, `--adv_reg_coeff`, `--adv_reg_kappa`, `--adv_arch`, `--adv_in`, `--adv_readout`, `--adv_batch`, `--gan_beta1`, `--pole_weight`, `--lyrichold_weight`, `--parts*`, `--txfm_weight`, `--grad_account`, … and the RpGAN+b_cap+FM train loop |
| `scripts/train_lm_gan_bcap.sh` | still passes those flags (would `argparse` fail) |
| `scripts/train_lm_gan_v2.sh` | separate `gan_v2.train` path (`Recipe.fm_weight=1` default) — not the live music3 entrypoint |
| `conceptmod/textsliders/lm_adv.py` | primitives intact (`rp_*`, `cap_penalty`, `feature_matching_loss`, SpanTransformerD) |
| provenance under `models/gan-bcap-repair/*/provenance/...` | full adv CLI + loop (source for restore) |
| local stash `adv-loop-20260909` | same adv port against older HEAD (`63c5d08` / PR #50) |

Blocked live Music LM transfer of locked **#94** recipe through the music3
entrypoint (FM off + b_cap=1 + leftover teacher).

## What we fixed (paths)

1. **3-way merge** of adv loop from stash/`train_from_stash.py` onto current
   trainer (kept roles/orth/`faithful_guard_e` from HEAD):
   - **before:** `conceptmod/textsliders/train_lm_slider_music3.py` (2672 lines, 0 adv flags)
   - **backup:** `/tmp/train_lm_slider_music3.py.before_adv_restore_20260909`
   - **after:** same path (~3650 lines) with argparse + train loop wired to `lm_adv` / `lm_particles`
2. Conflict resolution kept **both** lyric-hold fields (`pos_hidden`, `role_spans`,
   `neu_lyric`, `lyric_mask`) and adv span fields (`pos_full`, `pos_span_mask`).
3. Threaded roles/orth tensors through the stash `out` dict into `losses_from`
   (`pred_lyric` / `pred_concept` / `pred_plus_lyric` NameError fix).
4. `_assert_lyric_span` again returns `(neu_mask, pos_mask)` for span-D; updated
   `tests/test_lm_lyric_hold.py` accordingly (+ pre-existing `cannot be found` match).

Defaults after restore (provenance-aligned, **not** smoke launcher FM=1):

| flag | default |
|---|---|
| `--fm_weight` | **0.0** (locked #94 FM off) |
| `--adv_weight` | 0.0 (opt-in; set 1 for GAN) |
| `--adv_reg_coeff` / `--adv_reg_kappa` | **1.0** / **1.0** (b_cap) |
| `--gan_beta1` | **0.0** |
| `--lm_target` | `v9` (use `faithful_guard_e` for leftover) |

## Exact flags for a future #94-matching smoke (do not run GPU now)

Locked 2D recipe → Music LM entrypoint (hypotheses from transfer map). Prefer
**longer** Music steps over short×high-cover. FM **off**. b_cap c=κ=1.
`faithful_guard_e` needs declared `leak_*` (and slider axis) in the prompts YAML.

```bash
# CPU dry-parse only today; future GPU smoke on a free device (not GPU0 / nano-work-server):
CUDA_VISIBLE_DEVICES=1 MUSIC3_PYTHON=/home/mikkel/anaconda3/envs/minimax-music3/bin/python \
FM_WEIGHT=0 ADV_WEIGHT=1 MINERS=0 STEPS=120 SEED=7 \
  bash scripts/train_lm_gan_bcap.sh 1 locked94-fm0-bcap1-guard

# Explicit equivalent (leftover teacher; swap prompts YAML that declares leak_*):
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -u \
  conceptmod/textsliders/train_lm_slider_music3.py \
  --name locked94-fm0-bcap1-guard \
  --prompts_file conceptmod/textsliders/data/prompts-gender-uni-v2.yaml \
  --save_dir models/gan-bcap-repair/locked94-fm0-bcap1-guard \
  --lm_target faithful_guard_e \
  --pole_mode hidden --rank 8 --alpha 8 --lr 5e-4 --steps 120 --seed 7 \
  --no-early_stop --endreg_weight 1 --save_every 0 --device 0 \
  --adv_arch tx --adv_in scaled --adv_readout mean_last --adv_condition none \
  --adv_weight 1 --fm_weight 0 --fm_mode batch --pole_weight 0 --lyrichold_weight 1 \
  --adv_reg_coeff 1 --adv_reg_kappa 1 --adv_batch 4 \
  --gan_beta1 0 --gan_lr_schedule constant --grad_account --parts 0
```

Notes:

- Smoke script historically defaults `FM_WEIGHT=1`; **override to 0** for #94.
- `--adv_arch tx` requires a lyric UNI recipe (`faithful_plus_neu_lyric`). For
  leftover-only `faithful_guard_e` without lyric span, use `--adv_arch mlp`
  (tx will `SystemExit` at train start). Gender lyric smoke can keep
  `faithful_plus_neu_lyric` as in `train_lm_gan_bcap.sh` while still setting
  `FM_WEIGHT=0` + b_cap 1.
- Cover analogue on Music is modest pole/lyric hold (`--pole_weight` /
  `--lyrichold_weight`), not toy `cover_weight=1.5` by name.
- Do **not** map `--parts` to toy `n_particles`; keep `--parts 0` unless mining.

## Verification (CPU only)

```text
parse_args locked knobs + --help          OK
pytest tests/test_lm_lyric_hold.py tests/test_lm_adv.py \
       tests/test_lm_adv_tx.py tests/test_lm_adv_condition.py \
       tests/test_lm_gan.py -q            49 passed
pytest tests/test_lm_trainer_v9.py -q -k "parse or help or faithful_guard or lm_target"
                                          5 passed
pytest tests/test_lm_gan_trainer_integration.py -q
                                          28 passed (~42s)
```

## Non-goals

- No Music LM GPU train; nano-work-server / busy GPUs untouched.
- Did not change `gan_v2` defaults (still FM=1 research arms).
- Energy/MMD `objective_20260905` still rejected vs RpGAN+b_cap.
