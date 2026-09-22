# Supra2-IMG slider

Opt-in rectified-flow DiT slider on
[`SupraLabs/Supra2-IMG`](https://huggingface.co/SupraLabs/Supra2-IMG)
(~104.1M SupraDiT, frozen Flan-T5-Base, SD-VAE-FT-MSE). **Does not change
the default Music 3 trainer** (`train_lora_music3.py` / `--lm_target v9`).
Not Anima, Sana, Krea, or Z-Image — those trainers refuse a `supra` model id.

CPU tests use `--dummy` (tiny `SupraDiT` with the live LoRA path suffixes).
No GPU train and no Hub weights in CI. This repo does not vendor
`model_final_ema.pt`.

## Architecture

Matches the Hub `inference.py` card:

| field | value |
|---|---|
| hub id | `SupraLabs/Supra2-IMG` |
| checkpoint | `model_final_ema.pt` (local or `--allow_hub`; not vendored) |
| params | ~104.1M |
| encoder | frozen `google/flan-t5-base`, context 128, `D_CTX=768` |
| VAE | `stabilityai/sd-vae-ft-mse`, scale `0.18215` |
| resolution | **256²**, latent **32²**, patch **2** (256 tokens) |
| DiT | `D_MODEL=576`, `DEPTH=14`, `N_HEADS=9`, `HEAD_DIM=64`, `MLP_RATIO=4` |
| sampler | Euler flow, `t = i/K`, `z <- z + (1/K) * v` |
| sample CFG | **3.0**, **50** steps: `v_u + cfg * (v_c - v_u)` |

Train-time UNI still uses conceptmod's unscaled direction
`v(z, t, c) - v(z, t, '')`. The Euler grid is the Hub schedule (t runs
0 → 1), not Anima's FlowMatch σ schedule (1 → 0).

## UNI + unused-token hold

Same contract as the Anima trainer, on this backbone:

- student **+1** stays on the **neu / infer** caption
- the **+** caption is the teacher only
- scale **0** stays on neu
- pinned `attributes` are unused-token hold bookkeeping, **not** caption prefixes
- `concept_words` (`warm, golden, sunlit, glow`) are **not** held
- minus is a canary only

The concept is **lighting** (neutral daylight → warm sun). Not age.

| scale | student | `--lm_target trajectory` (default) |
|---|---|---|
| **+1** | infer / neu | K-step Hub Euler of frozen **plus** from the same `z_0` |
| **0** | infer / neu | light identity vs frozen **neu** traj (`--traj_identity_weight`, default 0.25) |
| **−1** | unscored canary | — |

```
MSE(x_student, x_plus) + λ_id * MSE(x_zero, x_neu)
```

`--lm_target direct` and `--lm_target cfg_delta` are the 1-step recipes.
`--teacher_gap_boost` (default 1, off) applies to those only.
Anima `embed_struct` / `same_crop` and Music 3 `v9` are rejected here.

## LoRA

Flan-T5 has no separate text conditioner. Caption signal enters through
`ctx_proj` and cross-attention. `proj` exists on both self- and
cross-attn, so targets are path suffixes.

| `--lora_targets` | trained | frozen |
|---|---|---|
| **`cross`** (default) | `ctx_proj`, `cross_attn.q`, `cross_attn.kv`, `cross_attn.proj` | Flan-T5, VAE, self-attn |
| `dit` | `self_attn.qkv`, `self_attn.proj` | Flan-T5, VAE, cross-attn, `ctx_proj` |
| `dit+cross` | both | Flan-T5, VAE |

## Live train card

```bash
HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_supra.py \
  --name lighting-supra \
  --prompts_file conceptmod/textsliders/data/prompts-supra.yaml \
  --model_id SupraLabs/Supra2-IMG \
  --lora_targets cross --rank 16 --resolution 256 \
  --sample_steps 50 --cfg 3 \
  --lr 1e-4 --lm_target trajectory --traj_steps 4 \
  --sample_every 100 \
  --device cuda:0 --save_dir models/lighting-supra
```

Print the card without training:

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_supra.py --print_card
```

Dummy (CI / no GPU / no Hub):

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_supra.py \
  --dummy --steps 8 --device cpu --save_dir /tmp/supra-dummy
PYTHONPATH=. pytest tests/test_supra_slider.py -q
```

Live `--dummy` off expects a **local** `model_final_ema.pt`
(`--checkpoint`) or an explicit `--allow_hub` download. CI does not set
`--allow_hub`. Frozen Flan-T5 encode and SD-VAE decode are not executed
in this backend PR; the dummy `predict_v` + Hub Euler is the CPU contract.

## Product handoff

This is the train/infer backend inside particle-sliders, parallel to
Anima (`anima_slider.py`, `train_lora_anima.py`). A later
**HyperGAN/supra-concept-sliders** (or similar) product repo would
consume it the way anima-concept-sliders consumes the Anima backend.

Not in this PR:

- Comfy plugin / studio UI
- vendored Hub weights (`model_final_ema.pt`, Flan-T5, SD-VAE)
- the product repo itself

## Yaml

`conceptmod/textsliders/data/prompts-supra.yaml` — woman and man, neutral
daylight vs warm sun. `conceptmod/textsliders/data/config-supra.yaml`
records the same card the CLI defaults use.

## Related

- [docs/anima-slider.md](anima-slider.md) — the trainer contract this card follows
- [docs/sana-slider.md](sana-slider.md) — different backbone; refuses `supra`
- [docs/krea-slider.md](krea-slider.md) — different backbone; refuses `supra`
