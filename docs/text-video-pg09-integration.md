# Text/video shared PG 0.9 integration, 2026-10-17

Historical initial integration receipt. Current explicit-profile adapters,
independent LTX review, local staged assets and admitted real cached Bonsai
results are recorded in [text-video-pg09-current-status.md](text-video-pg09-current-status.md).
The original wide-domain and local-LTX-discovery blockers below describe the
earlier release-only checkpoint, not the later explicit-profile qualification.

Implemented in isolated checkout `/mnt/nvme/sliders/text-video-adapters-pg09-20261017`,
branch `integration/text-video-pg09`, implementation commit
`56c91a8` (adapters, tests, verifier), based on adapter commit
`92f6e3f87328e4216b5462c5de0319d89289adac` (implementation `182db8e`).
The final tested shared core is immutable
`ec234e184676c46060ecbf6122df0e7b726f45a7`, including balanced sampling and
lazy-cache support. The owner's common checkpoint helper fix `ec34e4e` /
canonical cherry-pick `331c3ff` is included here as dependency commit `6e8ba6a`;
no alternate checkpoint helper was invented.
No shared core files were edited. The core/process owner was contacted directly.

## Repository discovery and coverage

Queried all pages of the unauthenticated public GitHub
`GET /orgs/HyperGAN/repos?per_page=100` inventory and resolved each matching
default branch through the branch API. This is a public inventory, not a claim
about unseen private repositories. Pinned README files were read for classification;
no model weights were downloaded. Machine-readable evidence is
`evidence/pg09-text-video/repository-inventory.json`.

| Actual repository | Exact default-branch head | Family and handoff |
|---|---|---|
| [particle-sliders](https://github.com/HyperGAN/particle-sliders) | `20fdf98d25455582e7e7191eef8ef51cf73173ae` | Shared package and embedded text/video; owned here |
| [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders) | `778ef38be9f1bec4e35cb02ade3a95e9c8dc8c14` | Image; existing integrated adapter, image owner |
| [supra-particle-sliders](https://github.com/HyperGAN/supra-particle-sliders) | `a908cc8ace35e4019a19bd25c1af7fd7e9ec72b6` | Image; existing integrated adapter, image owner |
| [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders) | `0b489f63388b000f18242c5d401d8b7709c27dee` | Image; outside this task, owner confirmation needed |
| [ming-particle-sliders](https://github.com/HyperGAN/ming-particle-sliders) | `93cb5bfa9e6585da3dc0a89327fb56dc276e8cc5` | Image decomposition scaffold; projection hooks unwired, image-owner confirmation needed |
| [music3-particle-sliders](https://github.com/HyperGAN/music3-particle-sliders) | `b40d2aa6fed745eda3043d3d3b11e362ec2f8dfe` | Audio; outside this task |
| [yue2-particle-sliders](https://github.com/HyperGAN/yue2-particle-sliders) | `b33794d81b92c125891a011531f55554d65e2082` | Audio; outside this task |
| [iris-particle-sliders](https://github.com/HyperGAN/iris-particle-sliders) | `6a226126623fb17c87fe22b86173fe2fcff4bf3e` | Native pixel-space image; image-owner confirmation needed |
| [qwen-image-particle-sliders](https://github.com/HyperGAN/qwen-image-particle-sliders) | `b418ff5b709dd687fdc7d9fbb5498c592ec21a58` | Image; image-owner confirmation needed |

No public standalone TinyQwen, Bonsai, LTX or H3 `-particle-sliders` repository
was returned. All four actual backend implementations are embedded in the shared
repository. No additional public text/video/other family was found. Ownership
of the image/audio families is delegated elsewhere, not verified by this task;
Ming/Iris and the other unconfirmed image families were flagged to the core owner
for parent handoff. A direct message to parent thread
`01a10ecb-09b2-741c-82de-b44dd08060c4` returned `thread not found`; no retry/send
is claimed. Core-owner coordination succeeded.

## Compatibility decisions

| Backend | Architecture / response | Decision |
|---|---|---|
| TinyQwen | Existing `TinyParticleSlider`, attention q/k/v/o, actual `[128,4]` dense-routed bank, last-token hidden vector | Thin `TinyQwenResponseAdapter` uses shared `ConditionalPG09`. Native 32-wide CPU fixture passes. Standard live 1024-wide response is rejected before response preparation/optimizer construction. |
| Bonsai | Frozen ternary GGUF fork-server 5120-wide last-token readout; existing torch residual particle head | Blocked by unchanged PG 0.9 wide-vector domain. No GGUF autograd or in-model LoRA claim, no adapter substitution. |
| LTX default | Gemma TE/video-connector adapters, post-connector embedding sequence | Blocked: sequence is not an admitted native field, and generic reshaping/pooling would change the protocol. Existing MSE embedding trainer is preserved and never invoked. |
| LTX explicit velocity | Existing video-attention LoRA, video volume `[B,C,T,H,W]`; coupled transformer consumes fixed audio context | Thin `LTX25VideoResponseAdapter` qualifies the one-step raw video-velocity arm on the native CPU backend. Public penalty sees genuine 5D video volumes. `particles=None` means adapter GAN only. Full AV trajectory/render and pretrained quality remain unqualified. |
| MiniMax H3 | Omni-transformer jointly packs text/video/audio with token tags, modality indices and MM-RoPE; returns separate packed video/audio velocities | Blocked pending reviewed native geometry and a declared multimodal response/critic protocol. No concatenation, vector-to-image reshape or invented `predict_v`. |

The existing TinyQwen port `7039652` was inspected before implementation. It uses
`tiny_llm_pg09.PG09TinyRun` and the separate `pg09_game.PairedPG09` trainer. This
change reuses its native slider and backend through the shared core; it neither
duplicates that loop nor alters its legacy checkpoints or controller dispatch.

Both new adapters own only responses and adapter leaves. The canonical core owns
all losses, optimizers, penalties, required public prior regularization, sampling
and checkpoint state. Tests forbid MSE calls. Critic coordinates are the shared
Gaussian-reference paired residual. Frozen critic conditions contain the native
neutral response as context and explicit scale/time coordinates, never row IDs.
The Tiny bank is the actual bank used in `generated`, owned once; LTX has none.

LTX uses native time `1000*sigma` for both video and audio. The CPU test preserves
the exact time-major/spatial-major patch=1 video coordinate permutation. Live
binding requires the staged pipeline's own `_pack_latents`/`_unpack_latents`,
explicit patch=1 configuration, verified model identity, FP32 adapter leaves and
frozen base modules. The pipeline codec file is hashed into backend metadata.
The upstream codec was inspected at Diffusers commit
`b800f828f3951c4d206e154411bcbfa693cd322e`,
`src/diffusers/pipelines/ltx2/pipeline_ltx2.py` lines 648–682.
No full-model pipeline was imported or run. Native checkpoint recomputation
captures the adapter scale; the existing public hook is reused.

## Model identity, licenses and hardware gates

Lightweight model metadata was queried; these are source pins for discovery,
not verified local weight-file manifests:

| Model | Hub revision observed | License / remaining gate |
|---|---|---|
| `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | Apache-2.0; live width blocked. |
| `prism-ml/Ternary-Bonsai-2-27B-gguf` | `b072e1d3b35a0a630cece372c2127528e0994386` | Apache-2.0; width blocked. Existing backend pins 5,946,648,928-byte PTQ1_0 file and Prism runtime `prism-b10685-7dffb15`; stock llama.cpp is unsuitable. No server started. |
| `Lightricks/LTX-2.5-Diffusers` | `da0b690960026eab794b8ceaf841233cf7064478` | Custom `ltx-2.x-community-license-agreement`; model configuration fetch returned HTTP401. Gated access/terms, staged config/source/weights, memory and quality validation remain pending. |
| `MiniMaxAI/MiniMax-H3` | `42ed227ee7df40d41602854ae760620d6eb651fe` | Custom `minimax-h3-community-license-agreement`; architecture/domain gate plus staged weights/source, terms and hardware fit remain pending. |

Custom community terms are not treated as the repository's MIT license.
Source-code licenses are recorded separately in the inventory. Full-model
memory/throughput were not measured. Existing H3 documentation describes a
large BF16 stack requiring split/offload or substantially larger devices; this
task makes no new fit claim. LTX's old embedding/velocity quality observations
are historical evidence, not validation of this GAN arm. In particular, the
historical velocity teacher gap was weak; CPU software compatibility does not
establish a useful trained slider.

No GPU run, process/service stop, permission change, MSE restart, publication or
merge occurred. GPU/process admission remains with thread
`01a147d7-f540-778e-ad79-4d8eb267403d`.

## CPU evidence and reproduction

New suite: **23 passed, zero failures/skips**, against immutable core `ec234e1`.
It covers both public presets, actual-bank and optimizer ownership, FP32 adapters
on a BF16 frozen Tiny host, base invariance, neutral student, scale-zero parity,
MSE prohibition, native 5D penalty shape, coordinate ordering, frozen audio/time
binding, checkpoint scale capture (including a complete protected-gradient GAN update), transactional restore rejection, exact
continuation, source identity and inference export/reload at 0/.5/1.

Four additional two-process replay cases pass: Tiny and LTX, each with `bispf`
and `bispf_optimized`. Producer checkpoints after two updates; a separate Python
process restores and finishes two more. Losses, full optimizer/regularizer/RNG
state and seeded sample match exactly. Evaluation leaves training state intact;
CUDA stays uninitialized. Receipts: `evidence/pg09-text-video/tests.xml` and
`fresh-process-resume.json`. Test checkpoint tensors stay under `/tmp`, outside
the commit. An earlier replay against the actively edited canonical checkout
correctly rejected a changed core-source signature; final evidence uses only the
immutable sparse checkout `/mnt/nvme/sliders/text-video-core-ec234e1-20261017`.

```bash
SLIDERS_CORE=/mnt/nvme/sliders/text-video-core-ec234e1-20261017
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$SLIDERS_CORE/packages/particle-sliders-core/src:$PWD" \
  /mnt/nvme/sliders/qualification/envs/pg090/bin/python -m pytest -q \
  -p no:cacheprovider -o pythonpath="$SLIDERS_CORE/packages/particle-sliders-core/src" \
  tests/test_pg09_text_video.py
/mnt/nvme/sliders/qualification/envs/pg090/bin/python \
  scripts/qualify_text_video_pg09_cpu.py --core "$SLIDERS_CORE" \
  --output /tmp/text-video-pg09-replay
```

The repository's default pytest path prepends its historical core package.
The explicit `pythonpath` override above selects the reviewed core and avoids
silently importing that older package. No dependency installation was performed.
