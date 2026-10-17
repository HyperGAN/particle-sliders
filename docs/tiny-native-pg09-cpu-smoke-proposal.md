# Actual Qwen3 native-attention CPU smoke — awaiting admission

This proposal implements the next real LLM gate, without extending failed
Bonsai training. It does not authorize itself. Independent QA and the core
owner's explicit admission of the two frozen commands are required before
loading the staged pretrained model. No pretrained weights were loaded during
preparation, and no GPU, server, download, service change or package write ran.

The exact recipe and argv arrays are in
`evidence/pg09-text-video/tiny-native-cpu-smoke-proposal.json`.
Proposal SHA256: `837db81ff679cd169b805e82585c44135566c51726174c9474865ef716cf1518`.
Worker SHA256: `a9ba85e9003536a3fca24d99d48285aa580e4d7c82937f136832a0265fb64d3a`.
Dataset SHA256: `1a2f44c69da30e4da8bd88e36ba74b80b96b368d13ad0298a06d29d4301d8f2e`.

## Proposed execution

Use local `Qwen/Qwen3-0.6B-Base` revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, Apache-2.0. The existing
1,192,135,096-byte safetensors file is freshly stream-hashed only after process
admission, before model allocation. No discovery download or new terms acceptance
is required. The expected weight hash is
`cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.

The actual 596,049,920-parameter base remains frozen in CPU FP32. Its complete
state is stream-hashed before attachment and after execution without cloning the
base. Native `TinyParticleSlider` rank8/alpha8 wraps all112 Qwen3 attention
q/k/v/o projections, including the actual asymmetric Q/KV dimensions, and owns
the actual128x4 routed bank once. Adapters and critic are FP32.

The existing `TinyQwenResponseAdapter` supplies frozen positive teachers and a
live **neutral-input student**, retaining all1024 last-token coordinates.
Conditions contain the actual frozen neutral vector, sign1 and time0. The
predeclared raw-logit critic uses hidden16. These choices do not establish
conditioning quality; this is a runtime smoke gate.

Only immutable core `3e94b13eb5d6cd1ceee3f018f019a68603f35696`
`ConditionalPG09` owns losses, optimizers, sampling and updates. Use the public
`bispf_optimized` recipe without overrides, seed7, batch2, balanced rows,
sigma1/response_scale1, **total_steps1200 but stop_after4**. No schedule is
compressed to a four-step horizon. G is public adversarial plus required public
prior; D is public adversarial plus public continuous penalty. Optimized MSE is
forbidden. The explicit reviewed `alz-continuous-v1`/`continuous` pair is passed
to the public validator before the first model load and to the game. The default
release/auto guard remains intact; the isolated single ALZ overlay is uninstalled.

Produce checkpoints at2 and finishes at4. A separately launched process rebuilds
the same model and resumes2→4. It must match full state, both optimizer/controller
states, sampling/model RNG, losses, hidden vectors and generated text exactly.
The restored whole final packet is saved independently. The adapter is then
zeroed before inference-export reload, followed by another exact sample check.
The frozen base and zero-scale output must remain exact throughout.

## Text and split evidence

The dataset freezes the three original native training pairs and canaries.
Three historically inspected comparison pairs are explicitly **validation**;
they are not described as unseen. Three new final-holdout pairs are declared now
but are never tokenized, forwarded or generated during the smoke. No final-heldout
model selection is possible in this worker.

Initial and final hidden vectors cover every training, validation and canary
row at0/-1/.5/+1. Actual greedy Hugging Face generation covers the first training,
validation and canary prompt plus one plain prose completion, each at the same
four strengths, with max32 new tokens. The worker retains complete token IDs,
text, emitted-token self-NLL/entropy and repetition diagnostics. Scale0 is
compared against the unattached base's hidden states and greedy text. Negative
strength is extrapolation from unipolar positive training, with no antonym or
negative-pole claim. Hidden-state errors and text diagnostics are not optimized,
and four updates cannot establish convergence, held-out transfer or semantic
quality.

## Process and source bounds

Exactly two sequential commands: produce≤900s then restore≤600s. CUDA is hidden;
Torch intra/inter-op, BLAS and tokenizer pools are restricted to one thread.
Peak RSS must stay≤8GiB, host reserve≥8GiB (startup requires16GiB available), and
the actual output filesystem must retain≥100GiB free. The reused stdlib resource
guard resolves the output's nearest existing parent and verifies exact mount
`/mnt/nvme`, device `/dev/nvme0n1p2`; it has no `/tmp` or filesystem fallback.
Checks run before Torch/model payload and after each phase/update/evaluation.
The external timeout is separate. Post-phase RSS checks cannot interrupt a native
allocation immediately; the peak is checked before proceeding.

Output:
`/mnt/nvme/sliders/qualification/runs/tiny-qwen-native-pg09-cpu-smoke-20261017-v1/bispf_optimized`.
Producer output and per-mode start markers are exclusive creations. A failed
command stops the sequence; no retry, budget extension, sweep or fallback is
admitted. A subsequent sustained run requires measured timing/memory, reviewed
actual text/validation evidence, a frozen longer recipe and new owner admission.

The proposal hashes86 source/small metadata files: worker/tests/builder, native
backend/adapter/bridge/critic, dataset, staged metadata/tokenizer/license, native
Transformers/Torch execution sources, core and all49 PG sources. Interpreter,
Python/Torch/numpy/Transformers/Diffusers/safetensors versions, generation and
runtime controls are pinned. Core also source/runtime/architecture-binds every
training checkpoint. No frozen Bonsai/native/core file was edited for this work.

## Preparation evidence and remaining gates

Synthetic CPU suite: **17 passed in4.33s**, including seven new worker tests and
ten existing wide-domain cases. The new test uses actual randomly initialized
Transformers `Qwen3ForCausalLM`/Qwen3Attention, with asymmetric Q projection width,
native hooks and autoregressive generation. It checks GAN-only updates, actual
bank changes, exact base/Off, state/loss replay, zero-before-export reload and
final-holdout exclusion. It loads no pretrained model. Receipt:
`evidence/pg09-text-video/tiny-native-cpu-smoke-tests.xml`.

Actual-host metadata-only preflight passes: exactNVMe device, about1974GiB free,
about45GiB host available, no Torch import or model payload. The large weights are
explicitly stat-only at this stage. Receipt:
`evidence/pg09-text-video/tiny-native-cpu-smoke-preflight.json`.

Independent source review and explicit process admission are **pending**. Actual
pretrained loader/FP32 memory, generation/Off parity, gradient behavior and exact
fresh-process runtime remain unqualified until the admitted commands execute.
The separate Tiny integration auditor found that the native constructor leaves
its routed MLP on CPU for non-CPU hosts. The current all-CPU proposal does not
exercise that defect; the reviewed constructor-device fix and new source pins
are required before any future CUDA proposal. Its production controller/export
fixes are separate audit work and have not been silently incorporated here.

Bonsai's completed1200-update held-out FAIL remains unchanged. Neither this
proposal nor a future corrected-input result can isolate GAN-vs-MSE causality.
LTX actual loader/memory/render gates and H3 native AV/license/hardware blockers
also remain unchanged. MOSS retains GPU priority under the sole coordinator.
