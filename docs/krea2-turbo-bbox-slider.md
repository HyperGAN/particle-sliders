# Krea2 turbo-bbox slider

Opt-in distilled image trainer for
[`jimmycarter/krea2-turbo-bbox`](https://huggingface.co/jimmycarter/krea2-turbo-bbox).
**Does not change the Music 3 default** and does not change the stock
Krea Raw card in [`train_lora_krea.py`](krea-slider.md) (CFG 4.5 / 28
steps). That trainer **refuses** this model id. Sana, Anima, and Supra
refuse it the same way they refuse each other's ids.

CPU tests use `--dummy`. No Hub weights and no GPU train in CI. This
repo does not vendor the transformer.

## Why a separate entrypoint

Raw and this finetune do not share defaults:

| | Raw (`train_lora_krea.py`) | turbo-bbox (`train_lora_krea2.py`) |
|---|---|---|
| weights | `krea/Krea-2-Raw` | transformer from `jimmycarter/krea2-turbo-bbox` |
| skeleton | the pipeline itself | `krea/Krea-2-Raw` (VAE, Qwen3-VL, scheduler) |
| steps | **28** | **8** |
| CFG | **4.5** | **0** |
| timestep shift | `calculate_shift` | **mu=1.15** (distilled) |

`krea_looks_turbo` treats a `turbo` substring as the stock Turbo *run*
card. This Hub id contains `turbo`, but it is a transformer-only
upload: `Krea2Pipeline.from_pretrained` on the id is the wrong load,
and Raw CFG 4.5 is the wrong teacher. The stock trainer raises instead
of guessing.

## Load

The Hub repo is the DiT only. Latest pushed variant is
`epoch-14-step-73184/transformer`. `is_distilled` is a pipeline flag
and is not stored on that upload, so the live loader sets it.

```python
tf = Krea2Transformer2DModel.from_pretrained(
    "jimmycarter/krea2-turbo-bbox",
    subfolder="epoch-14-step-73184/transformer",
    torch_dtype=torch.bfloat16,
)
pipe = Krea2Pipeline.from_pretrained(
    "krea/Krea-2-Raw", transformer=tf, torch_dtype=torch.bfloat16,
)
```

| input | how it loads |
|---|---|
| default Hub id | `from_pretrained(model_id, subfolder=--transformer_subfolder)` |
| local diffusers directory | that directory, or `<dir>/<subfolder>` when `config.json` is nested |
| `--transformer` Comfy `.safetensors` | `krea2-bbox-turbo-comfy-latest.safetensors` (or any local file), remapped onto the Raw skeleton |

`krea/Krea-2-Raw` is gated. `--allow_hub` is required for the first
download of the transformer and the skeleton. A warm cache can run
with `HF_HUB_OFFLINE=1` and no flag. CI does not set `--allow_hub`.

ComfyUI's Krea-2 Turbo template labels guidance-off as **CFG 1.0**.
This trainer uses the diffusers convention **guidance_scale 0**
(`v = v(cond)`, no `v(cond) - v('')` term). Do not pass Raw 4.5.
mu=1.15 is the distilled timestep shift, not a CFG scale.

## UNI

Same contract as stock Krea where it still applies. Not Music 3
lyric-hold. Not Anima `embed_struct` / `same_crop`.

| scale | teacher |
|---|---|
| **+1** | `v(z, t, pos)` at CFG 0 |
| **0** | `v(z, t, neu)` |
| **−1** | canary only (`v(neg) − v('')` is logged, never a loss) |

Unused tokens (pinned yaml `attributes`) hold to encode(neu). Concept
words are not held. Happy yaml is **bare captions**: attributes are
not prefixed onto target / positive / neutral. The fruit-bowl
`control_prompt` is verify-only.

Yaml `guidance_scale` is not the train/sample card. The CLI defaults
are CFG 0 / 8 / mu 1.15 even if a row still says 4.5. An explicit
`--sample_guidance` / `--sample_steps` / `--mu` overrides those
defaults; nothing fills them in from the Raw card.

`--lm_target embed` is the stock TE-only path (forces `--lora_targets te`).
Sample guidance stays 0.

LoRA targets match stock Krea: DiT `to_q/to_k/to_v/to_out.0` (default,
rank 16) or Qwen3-VL `q_proj/k_proj/v_proj/o_proj`.

About 90% of the base model's training used the grounding DSL in the
Hub [PROMPTING.md](https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md):
plain text, boxes `[x0,y0,x1,y1]` on a 0–1000 grid, **x first**.
Captions are passed through unchanged. The shipped yaml includes one
grounded row so a product repo can see that shape. Prefixing
`male` / `female` onto those strings would corrupt the DSL, which is
why bare captions stay on.

## Live train card

The shipped yaml is bare smile captions, so `--hold_weight` defaults
to **0.1**. The stock Krea trainer still defaults to 1.0 for its age
yaml. Gate on the smile grid, not on a crop-purity metric.

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea2.py \
  --name smile-krea2-bbox \
  --prompts_file conceptmod/textsliders/data/prompts-krea2-bbox.yaml \
  --model_id jimmycarter/krea2-turbo-bbox --allow_hub \
  --transformer_subfolder epoch-14-step-73184/transformer \
  --skeleton_model krea/Krea-2-Raw \
  --lora_targets dit --rank 16 --resolution 512 \
  --sample_steps 8 --sample_guidance 0 --mu 1.15 \
  --hold_weight 0.1 --steps 800 --lr 1e-4 --seed 7 --device 0 \
  --save_dir models/smile-krea2-bbox
```

Print the card without training:

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_krea2.py --print_card
```

Local Comfy file (skeleton still needs a Raw cache or `--allow_hub`):

```bash
python conceptmod/textsliders/train_lora_krea2.py \
  --name smile-krea2-bbox \
  --transformer /path/to/krea2-bbox-turbo-comfy-latest.safetensors \
  --allow_hub \
  --sample_steps 8 --sample_guidance 0 --mu 1.15
```

Dummy (CI / no GPU / no Hub):

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_krea2.py \
  --dummy --steps 8 --device 0 --save_dir /tmp/krea2-bbox-dummy
PYTHONPATH=. pytest tests/test_krea2_bbox_slider.py tests/test_krea_slider.py -q
```

## Product handoff

[HyperGAN/krea2-concept-sliders](https://github.com/HyperGAN/krea2-concept-sliders)
wraps this entrypoint the way a Supra product wraps
`train_lora_supra.py` and an Anima product wraps `train_lora_anima.py`.
The product repo is not created in this change. Weights are not vendored.

## Yaml

`conceptmod/textsliders/data/prompts-krea2-bbox.yaml` — bare smile rows
plus one grounded caption. `conceptmod/textsliders/data/config-krea2-bbox.yaml`
records the distilled card.

## Related

- [docs/krea-slider.md](krea-slider.md) — Raw train / stock Turbo run
- [docs/anima-slider.md](anima-slider.md) — refuses this model id
- [docs/supra-slider.md](supra-slider.md) — refuses this model id
- [docs/sana-slider.md](sana-slider.md) — refuses this model id
