# Calculated memory requirements

Updated 2026-11-14. All **28 existing inventory repositories** have a root
**`MEMORY.md`** and a prominent numeric README summary. Follow each link for
training, sampling and preparation profiles, GPU layout, input limits, precision,
residency, source revisions and the calculation ledger.

These are **calculated planning budgets, not measured peaks or hard ceilings**.
No model, benchmark, sampling or training job was run for this documentation.
Each number applies only to its stated finite profile. Larger/unbounded inputs
have no finite universal maximum. A conservative dense-attention fallback can be
much larger than the documented efficient/checkpointed alternative.

The GPU column shows the **largest per-device budget among the documented
profiles**, with that profile's GPU count and limits. Host RAM is the largest
separate host budget across its listed profiles; it need not occur in the same
phase. Read the linked table to select a useful execution profile. Sequential
phases use their maximum; simultaneously resident components/jobs add. Retired,
unshipped or external-reference rows retain those explicit qualifications.

| Repository / full profiles | Largest documented GPU profile | Largest documented host RAM | GPU profile scope |
| --- | --- | --- | --- |
| [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 204 GiB | 24 GiB | Train 768px dense fallback; 768×768; ≤512 tokens; B=2; BF16 frozen; FP32 adapter/critic/activation envelope; no checkpointing |
| [arete-particle-sliders](https://github.com/HyperGAN/arete-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 420 GiB | 68 GiB | Default uncheckpointed response training: single GPU; batch 2; ≤ 4096 tokens; BF16 base/FP32 adapters; eager; checkpointing=False |
| [evidence-dial](https://github.com/HyperGAN/evidence-dial/blob/main/MEMORY.md) | 1 GPU(s): 18 GiB | 20 GiB | Expanded behavior response training; batch 8; prompts≤ 512 tokens including target instruction; BF16/SDPA; all 36 blocks checkpointed; 138 cachedpairs |
| [flux-particle-sliders](https://github.com/HyperGAN/flux-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 160 GiB | 48 GiB | Native cached-response training, largest declared canvas; B2; 1024²; text256; BF16; no checkpointing |
| [glide-particle-sliders](https://github.com/HyperGAN/glide-particle-sliders/blob/main/MEMORY.md) | 0 GPU(s): 0 GiB | 22 GiB | Native encoder cache preparation / decision; CPU FP32; SDPA; batch 2; bounded text≤ 512 tokens including rendered choices; two choices only |
| [glimmer-particle-sliders](https://github.com/HyperGAN/glimmer-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 28 GiB | 14 GiB | Native NF4 response training; batch 2; complete input ≤ 128 tokens; all 52 blocks checkpointed; FP32 routed trainables |
| [iris-particle-sliders](https://github.com/HyperGAN/iris-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 80 GiB | 112 GiB | Train512px GPU, CPU encoder; 512×512; ≤300 tokens; B=2; FP32 DiT/adapter/activation envelope; BF16 TE; no checkpointing |
| [klein-particle-sliders](https://github.com/HyperGAN/klein-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 24 GiB | 32 GiB | Finished-image sampling; B1; 256²/512²; text128; BF16; all components resident |
| [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 60 GiB | 40 GiB | Train default512px; 512×512; ≤512 tokens; B=2; BF16 frozen; FP32 LoRA/critic/activation envelope; checkpointed |
| [ltx25-particle-sliders](https://github.com/HyperGAN/ltx25-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 128 GiB | 64 GiB | Native training, colocated GPU; B2;512×768;17frames;text128;BF16;all components on one GPU |
| [meta-glimmer-particle-slider](https://github.com/HyperGAN/meta-glimmer-particle-slider/blob/main/MEMORY.md) | 1 GPU(s): 28 GiB | 184 GiB | Native NF4 response training; batch 2; complete input ≤ 128 tokens; all 52 blocks checkpointed; FP32 routed trainables |
| [ming-particle-sliders](https://github.com/HyperGAN/ming-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 132 GiB | 136 GiB | Train staged512px no checkpointing; 512×512; ≤256 tokens; B=2; BF16 staged DiT; FP32 LoRA/critic/activation envelope; no checkpointing |
| [minimax-h3-image-particle-sliders](https://github.com/HyperGAN/minimax-h3-image-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 40 GiB | 64 GiB | **Unshipped/local native implementation**; Separate native FP4 training, full text bound; B2; 2048²; text≤4096; checkpointed; FP4/BF16 |
| [minimax-h3-particle-sliders](https://github.com/HyperGAN/minimax-h3-particle-sliders/blob/main/MEMORY.md) | 2 GPU(s): GPU0: 96, GPU1: 96 GiB | 128 GiB | Public original BF16 AV training; B2;5s480p/124frames; text≤256; saved24/26; group5/FF512/row512 |
| [moss-image-sliders](https://github.com/HyperGAN/moss-image-sliders/blob/main/MEMORY.md) | 1 GPU(s): 52 GiB | 76 GiB | Render1024px all components GPU; 1024×1024; ≤4096 tokens; B=1; BF16 DiT/TE; FP32 VAE/adapter/critic/activation envelope; no checkpointing |
| [music3-particle-sliders](https://github.com/HyperGAN/music3-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 32 GiB | 32 GiB | Native LM response training (default checkpointing=False); batch 2, total prefix+history ≤ 256 tokens; BF16 base/FP32 trainables; eager; no checkpointing |
| [p-h3-image-particle-sliders](https://github.com/HyperGAN/p-h3-image-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 64 GiB | 32 GiB | Original fixed-bank cached-response training; B2; native2048²+4audio rows; text≤256; int8/BF16; checkpointed |
| [particle-slider-runner](https://github.com/HyperGAN/particle-slider-runner/blob/master/MEMORY.md) | Product-dependent | 1 GiB controller; product +1 GiB worker | Complete model budget lives in product MEMORY.md; infrastructure overhead only |
| [particle-sliders](https://github.com/HyperGAN/particle-sliders/blob/main/MEMORY.md) | Product-dependent | 1 GiB controller; product +4 GiB example | Complete model budget lives in product MEMORY.md; infrastructure overhead only |
| [qwen-image-particle-sliders](https://github.com/HyperGAN/qwen-image-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 196 GiB | 184 GiB | Train512px dense fallback; 512×512; ≤512 tokens; B=2; BF16 frozen; FP32 routed adapter/critic/activation envelope; no checkpointing |
| [qwen35-particle-sliders](https://github.com/HyperGAN/qwen35-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 12 GiB | 12 GiB | Particle hidden-response training (RAW planning policy); scopeall/rank8; accumulated batch 4,≤ 384 tokens; BF16 full model, FP32 adapters; no checkpointing; no KV |
| [ripple-particle-sliders](https://github.com/HyperGAN/ripple-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 10 GiB | 12 GiB | NVFP4 file-backed hidden-response training; one CUDA GPU; batch 2; ≤ 128 tokens; packed MLP block16 FP8 scales, BF16 dense attention; all 80 blocks checkpointed; chunk256 |
| [sd-particle-sliders](https://github.com/HyperGAN/sd-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 8 GiB | 12 GiB | **Historical reference; current repo is a retired scaffold**; Historical SD1.5 sampling planning reference; B1;512²;CLIP77;FP16;SDPA;no offload;not implemented on main |
| [sophon-particle-sliders](https://github.com/HyperGAN/sophon-particle-sliders/blob/main/MEMORY.md) | 2 GPU(s): 56/66 GiB | 1420 GiB | Original speech teacher/preparation or finished audio: explicit 32/32; two GPUs; full 64-layer BF16 base; FP32 talker/E2 upper; total context≤ 4096 positions and≤ 512nativeframes per example; no CPU offload |
| [supra-particle-sliders](https://github.com/HyperGAN/supra-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 8 GiB | 8 GiB | Train native default256px; 256×256; ≤128 tokens; B=4; FP32 storage/activation envelope; BF16 CUDA autocast; no checkpointing |
| [voxcpm2-particle-sliders](https://github.com/HyperGAN/voxcpm2-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 60 GiB | 18 GiB | Full-waveform response training (native defaults); one GPU; ≤ 512 cachedpairs; batch 2; 4 frames×4patch=30,720 samples(.64s48k); 10 Euler; text/referenceprefix≤ 256 positions; BF16 base/FP32 VAE+adapters; no checkpointing |
| [yue2-particle-sliders](https://github.com/HyperGAN/yue2-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 92 GiB | 28 GiB | Separate eager finished-audio pipeline: AR/NAR then VAE; batch 1; semantic generation ≤ 9000 tokens, total AR context/latent frames ≤ 24576; BF16 model + FP32 VAE; math-attention upper envelope, no AR offload |
| [zimage-particle-sliders](https://github.com/HyperGAN/zimage-particle-sliders/blob/main/MEMORY.md) | 1 GPU(s): 144 GiB | 64 GiB | Train staged512px no checkpointing; 512×512; ≤256 tokens; B=2; FP32 staged model/LoRA/critic/activation envelope; no checkpointing |

The shared engine's [memory standard](https://github.com/HyperGAN/particle-sliders/blob/main/MEMORY.md)
defines the required root file, bounded profiles, state/activation/cache/load
ledger and update policy. Pretrained model bytes alone are not total training
memory. No weight EMA is included. Existing rates, weights, checkpoints, runtime
sources and qualification results were preserved.

Deleted Fibo, Cascade and TinyQwen have no current repository to update. SD is
included as a retired scaffold with a separately labeled historical reference.
Glimmer and Meta Glimmer share a base but each interface has its own document.
Evidence Dial is included as the separately inventoried integration; its memory
profiles do not merge its training/selection evidence into the slider campaign.
