# Native image bindings for the shared PG0.9 GAN

`conceptmod.textsliders.pg09_images` adds thin response bindings to `particle_sliders.pg09.ConditionalPG09`. It owns no loss, optimizer or training loop. Both `bispf` and `bispf_optimized` use the released PG0.9 public adversarial losses, critic penalty and required prior regularizer. There is no optimized MSE fallback or auxiliary reconstruction loss. Native image fields remain spatial through the public penalty; only the critic flattens internally.

The implementation is isolated at `/mnt/nvme/sliders/image-adapters-pg09-20261017`, branch `integration/pg09-image-adapters`, based on existing adapters `92f6e3f`. It reuses the exact shared repeatable-checkpoint fix `ec34e4ecc68aedc88bb5cd44a96d4bd78af99fa9` (local dependency cherry-pick `983f94c`, already present in canonical core as `331c3ff`). No alternate checkpoint implementation was invented. Shared core is pinned at **`ec234e184676c46060ecbf6122df0e7b726f45a7`** in a separate immutable checkout `/mnt/nvme/sliders/image-core-pin-7884c8d` (the directory name is historical; HEAD is ec234e1). No shared core checkout file was edited here.

## Discovered repository inventory

This inventory comes from the actual GitHub HyperGAN organization API, followed through every page, rather than guessed family repository names. All branches were `main` at inspection on 2026-10-17. Sana, ZImage, SD/SDXL/SD3, FLUX and Cascade exist as modules in `particle-sliders`; the API returned no separate family repository for them. Local historical qualification-head pins and remotes are separately recorded in `evidence/pg09-images/local-inspection-pins.json`.

| Actual repository | Exact discovered HEAD | Decision |
| --- | --- | --- |
| HyperGAN/particle-sliders | `20fdf98d25455582e7e7191eef8ef51cf73173ae` | Shared source modules: Sana/ZImage/SD/SDXL/SD3/FLUX/Krea/Cascade bindings implemented. |
| HyperGAN/anima-particle-sliders | `778ef38be9f1bec4e35cb02ade3a95e9c8dc8c14` | Excluded: separate existing adapter owner. |
| HyperGAN/supra-particle-sliders | `a908cc8ace35e4019a19bd25c1af7fd7e9ec72b6` | Excluded: separate existing adapter owner. |
| HyperGAN/krea2-particle-sliders | `0b489f63388b000f18242c5d401d8b7709c27dee` | Implemented: actual product DiT/PEFT binding; native tiny Krea2 CPU qualified. |
| HyperGAN/ming-particle-sliders | `93cb5bfa9e6585da3dc0a89327fb56dc276e8cc5` | Blocked: train/infer are stamp-only stubs, no model forward/runtime/bank. |
| HyperGAN/music3-particle-sliders | `b40d2aa6fed745eda3043d3d3b11e362ec2f8dfe` | Excluded: audio owner. |
| HyperGAN/yue2-particle-sliders | `b33794d81b92c125891a011531f55554d65e2082` | Excluded: audio owner. |
| HyperGAN/iris-particle-sliders | `6a226126623fb17c87fe22b86173fe2fcff4bf3e` | Implemented: native RGB velocity plus actual product hooked bank/particles. |
| HyperGAN/qwen-image-particle-sliders | `b418ff5b709dd687fdc7d9fbb5498c592ec21a58` | Implemented: native VAE field plus actual product hooked bank/particles. |
| HyperGAN/moss-image-sliders | `477278a2e87f77d8a56a38bd6b952741231f1b3e` | Excluded: separate existing adapter owner. |

Ming evidence: at its pinned HEAD, `ming/train.py` describes a “train stub” and rejects normal execution with “Ming training is not wired”; `ming/infer.py` states “Adapter execution ... is not implemented in this scaffold.” Its tracked tree has no native model/runtime/adapter code. A thin PG adapter cannot reuse an absent forward. This is an implementation dependency, not a permission denial. The core owner was told that Ming is covered by the image inventory and currently blocked; no model family was silently left unowned.

## Bindings and exact coordinates

`image_lora_parameters(model, native_lora=None)` registers the existing FP32 adapter leaves exactly once. It supports PEFT LoRA A/B leaves and the existing conceptmod `LoRANetwork`; frozen base weights stay out of optimizers and exports. Unsupported trainable base/text-encoder leaves are rejected. Enabled native checkpointing uses the common per-forward captured scale hook, including repeatable protected-gradient replays.

`ImageResponseAdapter(family, model, network, scheduler=..., pipeline=..., model_identity=...)` accepts staged models only; construction never loads/downloads weights or starts a worker. `prepare(ResponseRow)` deeply freezes conditioning dictionaries, latents, times and teacher targets. The student always re-evaluates the native model on neutral conditioning. Targets are frozen positive responses, with explicit CFG when requested; the negative CFG branch remains at adapter scale zero. These are **unipolar** bindings (scale+1 training), not a new signed-teacher assumption.

| Family | Response reaching PG | Native time and forward |
| --- | --- | --- |
| Sana | `[1,C,H,W]` raw flow velocity | Raw scheduler t multiplied by model `timestep_scale`; exact attention mask retained; split learned-variance half only when output has 2C channels. |
| ZImage | `[1,C,H,W]` flow velocity | Native list-of-CHW-frame model call at `(1000-t)/1000`; negate native prediction as in existing ZImage backend. |
| SD | `[1,C,H,W]` native prediction | Actual staged scheduler input scaling and explicit epsilon/v_prediction configuration. No guessed conversion to flow velocity. |
| SDXL | `[1,C,H,W]` native prediction | Same scheduler rule, plus native pooled text and six time IDs. |
| SD3 | `[1,C,H,W]` raw flow velocity | Raw scheduler t; native pooled projection and joint text encodings. No `scheduler.step` masquerading as a prediction. |
| FLUX | `[1,C,H,W]` latent flow velocity | Actual FluxPipeline 2x2 spatial pack/unpack with declared text/image position IDs; cast native timestep to latent dtype before t/1000. |
| Krea Raw / Krea2 Turbo-BBox | `[1,C,1,H,W]` VAE latent flow velocity | Actual Krea2Pipeline spatial pack/unpack and position IDs; t/1000 then cast to latent dtype. `from_krea2_product` reuses the product backend's actual DiT/PEFT parameters. |
| Cascade Stage C | `[1,C,H,W]` native spatial image-embedding model output | Native DDPMWuerstchen continuous timestep ratio r in [0,1], text/pooled/image embeddings. Not RGB or an arbitrary vector. |
| Iris | `[1,3,H,W]` RGB pixel velocity | Native IrisDiT raw t=1000*sigma and text-layer embeddings/mask; open native CFG interval endpoints. `IrisResponseAdapter` reuses the product AdapterBank and the same actual participating 128x4 particle parameter. |
| Qwen-Image2.1 | `[1,C,1,H,W]` VAE latent flow velocity | Actual QwenImage21Pipeline unpatched spatial pack/unpack at t/1000, native product `with_encoding`/`predict`, uncached text-to-image/true_cfg_scale1 only. Actual hooked product AdapterBank and participating128x4particles. |

FLUX/Krea/Qwen response fields are genuine native spatial lattices, reversibly converted by their official packing methods. No generic continuous vector is reshaped into an image, and the core's wide-vector>=170 block is unchanged. The first eight bindings are explicitly **adapter GANs with particles=None**; only Iris and Qwen claim a routed particle GAN, with identity and gradient tests on the actual hooked bank.

`image_condition_features(prepared)` provides fixed numeric time, scale, CFG and strength coordinates plus a source-bound32D Rademacher projection for each neutral/positive/negative native conditioning dictionary. Every native tensor coordinate participates, including embeddings, attention masks, pooled vectors and position/time IDs; private per-key CPU generators and bounded16384-element chunks preserve training RNG and bound temporary memory. The declared version/seed/rank/chunk policy is included in backend metadata. These are input context coordinates, never row IDs.

An exact collision gate compares the complete dataset's numeric features against the exact conditioning/timestep/sign/CFG identity before constructing the game. Different contexts with identical critic features fail closed; repeated context at a different sampled latent is valid. The equal-moment token/channel permutation regression distinguishes contexts that collided under the original mean/RMS policy, and a deliberately forced sketch collision verifies rejection. For lazy caches, prepare the conditions over the full immutable dataset, persist their tensors/policy/hashes in the index and validate globally; a minibatch-only collision check is insufficient. A finite sketch cannot guarantee semantic separation on unseen prompts or exclude near collisions; held-out conditioning transfer remains unqualified. Full conditioning bytes, native geometry, helper source hashes and staged model identity are checkpoint-bound by core. Production non-CPU models require a verified staged model identity.

Example binding after the admission owner stages a frozen model and its native LoRA:

```python
from conceptmod.textsliders.pg09_images import ImageResponseAdapter, image_lora_parameters, image_condition_features
from particle_sliders.pg09 import ConditionalPG09
network = image_lora_parameters(model)
backend = ImageResponseAdapter('sd3', model, network, model_identity=verified_identity)
prepared = [backend.prepare(row) for row in frozen_rows]
conditions = image_condition_features(prepared)
game = ConditionalPG09(backend, prepared, critic, conditions=conditions,
    particles=None, response_domain='continuous_field', preset='bispf_optimized',
    total_steps=steps, batch_size=2, response_scale=1., seed=37)
```

## CPU evidence and reproduction

All fixtures use actual tiny native model architectures with random weights and synthetic/precomputed text; no pretrained model quality is claimed. Every environment command hides CUDA. The core verifies all49PG0.9Python source hashes against the official release. The source/evidence receipt records implementation commit, exact core source, native product/upstream heads, runtime versions, resolved recipes, source hashes and test XML. The final conditioning follow-up supersedes the initial moment-only feature policy following independent QA review. Existing shared environments were not changed.

Native products are isolated sparse checkouts under `/mnt/nvme/sliders/image-inspect-*`. Iris uses upstream `speridlabs/iris-3b@a8d15239dea469aba042cfa56ca3bb4e450d5ebc`. Qwen requires isolated Diffusers source `1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458`, Transformers5.17.0, huggingface-hub1.32.0 and tokenizers0.23.1. Tiny Iris parsing uses OmegaConf2.3.0 and antlr4-python3-runtime4.9.3. These extra wheels were staged under `/tmp/pg09-image-deps` and `/tmp/pg09-qwen-deps`, without installation into a shared environment.

From this checkout, using the existing PG0.9 qualification interpreter:

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/tmp/pg09-image-deps:/mnt/nvme/sliders/image-core-pin-7884c8d/packages/particle-sliders-core/src:/mnt/nvme/sliders/image-inspect-iris3b-a8d1523/src:/mnt/nvme/sliders/image-inspect-iris-particle-sliders-20261017/src:/mnt/nvme/sliders/image-inspect-krea2-particle-sliders-20261017:.:tests
/mnt/nvme/sliders/qualification/envs/pg090/bin/python -m pytest tests/test_pg09_images.py -q -o pythonpath='' -p no:cacheprovider
/mnt/nvme/sliders/qualification/envs/pg090/bin/python scripts/qualify_image_responses_pg09_cpu.py --suite images --output /tmp/pg09-image-fresh-process

export PYTHONPATH=/tmp/pg09-qwen-deps:/mnt/nvme/sliders/image-inspect-diffusers-1d5d056/src:/mnt/nvme/sliders/image-core-pin-7884c8d/packages/particle-sliders-core/src:/mnt/nvme/sliders/image-inspect-qwen-image-particle-sliders-20261017/src:.:tests
/mnt/nvme/sliders/qualification/envs/pg090/bin/python -m pytest tests/test_pg09_qwen_image.py -q -o pythonpath='' -p no:cacheprovider
/mnt/nvme/sliders/qualification/envs/pg090/bin/python scripts/qualify_image_responses_pg09_cpu.py --suite qwen --output /tmp/pg09-qwen-fresh-process
```

The50image checks and2Qwen checks cover native fields, frozen dictionary snapshots/base weights, connected/updated adapter leaves, actual routed-bank participation, exact checkpoint/export reload, input/time/sign guards, real conceptmod LoRA reuse, Krea product reuse, native checkpoint repeated backward, FP32 shared CFG/render-loop equivalence and BF16 native timestep ordering. Krea's BF16 tiny fixture emits a PyTorch RMSNorm dtype performance warning; the arithmetic test passes. Fresh-process qualification runs both presets for every implemented family: **five uninterrupted updates equal two plus three in a fresh process**, including the lazy fourth-step critic penalty, complete checkpoint state, each metric/loss and evaluation output. A discovered SD scheduler metadata ordering issue was fixed by sorting its nonsemantic `_use_default_values` list; actual scheduler settings are retained and bound.

## Remaining qualification and limits

- Ming needs a real native model forward, adapter/runtime and image-conditioning/layer contract before it can be adapted. Its scaffold is not a runnable backend.
- Full pretrained model memory, sample quality/convergence and held-out prompt transfer are unmeasured. This branch ran CPU fixtures only. The core owner remains sole GPU/process admission owner; no GPU admission was requested or consumed here.
- SD/SDXL/Cascade and Iris sampler trajectories are rejected until native scheduler/FlowDPMSolver parity is qualified. Raw native response GAN training is implemented. Flow families support only an explicitly supplied FP32 Euler schedule and the same training/render response loop; mixed-precision trajectories fail closed. Official production sampler/render parity and staged VAE decode remain pending. No fake image decoder is supplied.
- Text-encoder LoRA/live encoding, Qwen image-conditioned/Edit/Layered modes or CFG other than1, and alternative Cascade scheduler-to-ratio transforms are outside these bindings and rejected where applicable.
- No large eager production response cache is constructed here; production cache wiring can use the owner's lazy provider API. No MSE launcher was restarted, no service was stopped, no permission was changed and nothing was published or merged.
