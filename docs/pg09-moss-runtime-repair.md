# MOSS cache-producer runtime repair

The first pretrained GAN probe made zero optimizer updates. It rejected teacher/cache parity in the newer Torch 2.11 / Diffusers 0.40 development stack, especially at the low-noise cached state. The strict relative-RMS limit remains 0.001; caches, old adapters and failed-run receipts are preserved.

A separate two-stack diagnostic (`296582f`) made zero updates. Both stacks loaded identical frozen parameters and RoPE buffer values. The original Torch 2.9.1 / Diffusers 0.38 / Transformers 5.3 stack reproduced cached neutral and positive outputs bit for bit at both inspected states, including the deterministic policy used by the GAN worker. The newer stack differed; math-only SDPA differed on both. This supports a runtime/backend arithmetic explanation, without identifying a single kernel or package as the cause. Raw outputs, repeat checks and file hashes are bound in `evidence/pg09-moss-runtime/diagnosis.json`.

`scripts/guarded_moss_original_pg09.py` retains the existing leases, physical-card identity/resident checks, kernel/resource guards, checkpoint/recipe phase proofs and no-retry behavior. It launches the shared GAN worker through `scripts/moss_original_runtime.py`, which verifies the exact original executable, native module origins/versions and the 49 official PG source hashes before CUDA/model loading. Only the extracted pure-Python PG wheel is added to the original interpreter path; no packages are installed or replaced.

New `*-original-runtime.json` recipes use separate output directories, the same fixed GAN-only 3,840-update/batch-eight budget and held-out protocol, and a three-step probe before training. An initial qualification callback additionally compares zero-adapter generated responses with grad enabled and disabled against the cached neutral response, under the unchanged parity threshold. It contributes no optimized term and performs no backward or optimizer call.

The diagnostic is not a training or image-quality result. Sustained training and matched exported/reloaded held-out comparisons remain required.
