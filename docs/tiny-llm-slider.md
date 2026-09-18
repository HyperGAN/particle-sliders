# Tiny-LLM concept-slider test target

Opt-in only. Not a default, not a production recipe.

## Which model

Pinned: **`Qwen/Qwen3-0.6B-Base`** (Hub safetensors total **596,049,920**
params BF16, Apache-2.0, ungated).

- Qwen3 release April 2025 → satisfies the 2025+ recency bar.
- Live config: 28 layers, hidden 1024, 16 query / 8 KV heads (GQA,
  head_dim 128), SwiGLU intermediate 3072, vocab 151936 tied,
  context 40960, RoPE theta 1e6.
- LoRA attaches to attention `q_proj` / `k_proj` / `v_proj` / `o_proj`
  only (live `Qwen3Attention` names). MLP, norms, embeddings and
  `lm_head` stay frozen. Rank 8, alpha 8.0, LoRA-up `N(0, 0.02)`.

## Why this one (rejected candidates)

- `google/gemma-3-270m(-it)`: smaller (268M, June 2025) but **gated**
  (manual approval) — a fresh VM cannot download it, so it fails the
  smoke-run requirement.
- `HuggingFaceTB/SmolLM2-135M(-Instruct)`: smaller still (134M,
  ungated) but a **November-2024** release — fails the 2025+ bar.
- `Qwen/Qwen3-0.6B` (instruct, 751M): same family and date, but larger
  than the base checkpoint (596M), so the base wins on "smallest".
- Anything over ~1B params is rejected in code (`_assert_qwen3_shape`).

## Smoke-run

CPU / CI (no weights, no Hub):

```bash
python3 conceptmod/textsliders/train_lora_tiny_llm.py --dummy \
  --steps 20 --name tiny-llm-smoke --save_dir models/tiny-llm-smoke
```

Reload + report only:

```bash
python3 conceptmod/textsliders/train_lora_tiny_llm.py --dummy \
  --steps 0 --load_tiny_lora models/tiny-llm-smoke \
  --name tiny-llm-reload --save_dir models/tiny-llm-reload
```

Real weights (single small GPU or CPU, needs `transformers>=4.51`):

```bash
python3 conceptmod/textsliders/train_lora_tiny_llm.py \
  --allow_hub --device cuda:0 --steps 20 --name tiny-llm-real \
  --save_dir models/tiny-llm-real
```

Omit `--allow_hub` to use a local Hub cache snapshot only. The trainer
writes `{name}_last.json` (sidecar) plus `{name}_lora.safetensors`
(custom `lora_tiny-*` keys, not PEFT) and a `report/hidden_delta.json`
grid at scales 0 / 0.5 / 1.

Verified 2026-09-18 on CPU against the real pinned weights: 112 LoRA
modules (28 layers x q/k/v/o), hidden shape `(1, 7, 1024)`, scale 0
bit-exact vs base, 2-step smoke loss 0.0408 -> 0.0327, report row-0
scale-1 `delta_cos` 0.93 after 2 steps (already along the concept axis).

## UNI recipe

Student scale +1 fits the raw + caption last-hidden states, student
scale 0 fits the neutral caption (`tiny_llm_uni_hidden`, full-sequence
MSE). No minus teacher; the `unconditional` prompt row is a canary only.
Scale 0 is exact base (adapters rest at multiplier 0).

## What it is NOT

- Not Music 3, not YuE2, not H3, not Music Arm B.
- Not the locked recipe: it does not read `locked_shared` and does not
  change any live `--lm_target` default (Music 3 stays `v9` / `hidden`).
- Not a default: nothing imports this backend unless the tiny trainer
  (or its test) asks for it.
- Not a quality slider: the dummy stand-in is randomly initialized, so
  its "concepts" are smoke only. Real concept tests need `--allow_hub`.
