# Tiny-LLM concept-slider test target

Opt-in only. Not a default, not a production recipe. Propose-only: does not
flip Music ARM_B, live `--lm_target`, locked AdvConfig, or YuE2 defaults.

## Which model

Pinned: **`Qwen/Qwen3-0.6B-Base`** (Hub safetensors total **596,049,920**
params BF16, Apache-2.0, ungated).

- Qwen3 release April 2025 → satisfies the 2025+ recency bar.
- Live config: 28 layers, hidden 1024, 16 query / 8 KV heads (GQA,
  head_dim 128), SwiGLU intermediate 3072, vocab 151936 tied,
  context 40960, RoPE theta 1e6.
- Branches attach to attention `q_proj` / `k_proj` / `v_proj` / `o_proj`
  only (live `Qwen3Attention` names). MLP, norms, embeddings and
  `lm_head` stay frozen. Rank 8, alpha 8.0.

## Why this one (rejected candidates)

- `google/gemma-3-270m(-it)`: smaller (268M, June 2025) but **gated**
  (manual approval) — a fresh VM cannot download it, so it fails the
  smoke-run requirement.
- `HuggingFaceTB/SmolLM2-135M(-Instruct)`: smaller still (134M,
  ungated) but a **November-2024** release — fails the 2025+ bar.
- `Qwen/Qwen3-0.6B` (instruct, 751M): same family and date, but larger
  than the base checkpoint (596M), so the base wins on "smallest".
- Anything over ~1B params is rejected in code (`_assert_qwen3_shape`).

## Working game: YuE2 routed particle bridge (transferred, not reinvented)

The trainer runs YuE2's current working recipe
`anneal-routed-particle-error` (`--recipe particle_bridge`) through the
**shared** `conceptmod/textsliders/particle_bridge_gan` module — the same
Rp paired-error GAN on `e = T(student) - T(positive)` (same noise on
real/fake, sigma anneals 0.03 over 8000), lazy b_cap every 4th update x4,
particle VIC, EMA 0.995. The earlier hidden-state MSE UNI recipe is gone
(the trainer rejects any other `--recipe`). There is no output MSE, FM,
ending, hold, or anchor anywhere. Unipolar (+ vs raw positive); scale 0
bypasses every branch exactly; `unconditional` rows are unscored canaries.

### MATCH (read from the YuE2 source, never copied)

- Game math, LRs (G/D/particle 0.0006/0.0009/0.006), Adam betas (0,
  0.999), constant schedule, 128x4 standard-normal cloud, VIC coeff 1
  (std hinge + off-diag cov, 64-particle sample), b_cap L2 k=1 coeff 1
  lazy-4 x4, EMA 0.995, noise horizon 8000 — all imported from
  `particle_bridge_gan.REFERENCE` / `shared.update` / `shared.build_game`
  (tests assert module identity plus every number).
- Adapter math: per-projection router + `softmax(q P^T / sqrt(4))` route
  over one shared cloud, `up` zero-initialized, rank/alpha pinned 8/8.
- Full 128x4 cloud on the default path — no shrink, no smoke flag needed
  (the cloud is 512 params; the host forward dominates CPU cost).

### DRIFT (deliberate tiny-model transfers)

- Host: Qwen3 last-hidden instead of YuE2 AR hidden; readout is the last
  caption-token hidden instead of YuE2's prefix-boundary token (same kind
  of readout — last prompt token — minus lyrics/music structure).
- Prompts: 3 plain caption rows instead of 4 sound-only lyric rows.
- Adapter dtype follows the host (fp32 on CPU); YuE2 runs fp32 LoRA over
  a bf16 host. Identical on every fp32 path.
- **Normalization:** `tiny_llm_particle.build_game` calls
  `shared.build_game(network, targets)` **without** `neutrals`. Whitening
  stays absolute-target (`std` of positive states). Live YuE2
  `particle_bridge` now passes neutrals and uses paired-edit whitening.
  Do not treat a tiny-LLM smoke as a replay of current YuE2 native.
- Scope: no `state.pt` resume, no campaign/queue machinery, no lyric
  guards, no audio render — this is a game smoke target, not a release
  pipeline. Save format is its own `conceptmod-tiny-llm-particle-v1`
  (EMA `_last` + live `_live_last`, same convention as the YuE2 run).
- `sound_only` prompt validation is YuE2-music-specific and not applied.

## Smoke-run

CPU / CI (no weights, no Hub):

```bash
python3 conceptmod/textsliders/train_lora_tiny_llm.py --dummy \
  --steps 20 --name tiny-llm-particle --save_dir models/tiny-llm-particle
```

Reload + report only:

```bash
python3 conceptmod/textsliders/train_lora_tiny_llm.py --dummy \
  --steps 0 --load_tiny_lora models/tiny-llm-particle \
  --name tiny-llm-reload --save_dir models/tiny-llm-reload
```

Real weights (single small GPU or CPU, needs `transformers>=4.51`):

```bash
python3 conceptmod/textsliders/train_lora_tiny_llm.py \
  --allow_hub --device cuda:0 --steps 20 --name tiny-llm-real \
  --save_dir models/tiny-llm-real
```

Omit `--allow_hub` to use a local Hub cache snapshot only. The trainer
writes `{name}_last.json` (sidecar, REFERENCE numbers echoed) plus EMA
`{name}_last.safetensors`, live `{name}_live_last.safetensors` (custom
`lora_tiny-*` keys, not PEFT), and a `report/hidden_delta.json` diagnostic
grid at scales 0 / 0.5 / 1 (report-only; the game has no MSE).

The 1200-update budget in the YuE2 queue is a run choice, not a game
number; the noise horizon stays 8000 regardless of `--steps`.

Verified 2026-09-18: dummy CPU smoke (g_adv 0.6931 = ln 2 at init,
VIC 0.127 -> 0.066, sigma 0.9996 -> 0.9974) and a 3-step live smoke on the
real pinned weights (g_adv 0.6931 -> 0.6915, d_loss 0.6931 -> 0.7053,
VIC 0.0823 -> 0.0801, cos_pos 0.993, EMA + live exports written).

## What it is NOT

- Not Music 3, not YuE2, not H3, not Music Arm B.
- Not the locked recipe: it does not read `locked_shared` and does not
  change any live `--lm_target` default (Music 3 stays `v9` / `hidden`).
- Not a default: nothing imports this backend unless the tiny trainer
  (or its test) asks for it.
- Not the old unipolar-rpgan-bcap loop, not c9 4x LR, not MSE.
- Not a quality slider: the dummy stand-in is randomly initialized, so
  its "concepts" are smoke only. Real concept tests need `--allow_hub`.

## Related

- [docs/README.md](README.md) — backend map
- [docs/prompts.md](prompts.md) — `prompts-tiny-llm.yaml`
- [yue2-slider.md](yue2-slider.md) — live YuE2 particle CLI (paired-edit)
- [bonsai-gguf-slider.md](bonsai-gguf-slider.md) — frozen-readout sibling smoke
