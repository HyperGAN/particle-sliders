# TinyQwen package integration audit — isolated proposal

Historical fe54d05 PG0.9.0 audit receipt. Current installed API0.6.0 / PG0.9.1
cohort and public legacy CLI retirement are documented in
[text-package-acceptance.md](text-package-acceptance.md). These older receipts
are preserved and do not qualify the replacement package candidate.

Repository: `https://github.com/HyperGAN/particle-sliders.git`.
TinyQwen/native LLM is an embedded backend, not a distinct public derivative
repository. Source identity was verified before edits: clean
`/mnt/nvme/sliders/text-video-adapters-pg09-20261017`, branch
`integration/text-video-pg09`, commit
`7f018780c073790e1d706c31716d720302750a08`.
Audit work is isolated in task-13 `llm-audit`, branch
`audit/tinyqwen-shared-pg09`. No source pinned by an active process was edited.

## Demonstrated defects and changes

| Finding | Change / evidence |
|---|---|
| Reachable PG09 qualifier used local `PairedPG09`, bypassing the shared game | `PG09TinyRun.step` now executes shared `ConditionalPG09.step`; tests forbid local-loop/MSE calls and assert actual class and bank identity |
| Native inference instantiated the local `particle_bridge_gan.RoutedMLP` copy | Native projection executes `particle_sliders.reference.RoutedMLP`; exact seeded keys/tensors/output parity regression |
| FP32 checkpoint loaded into BF16 intermediate leaves | Explicit `adapter_dtype=torch.float32` loads after allocation in FP32; previous sample particle `0.12345679104328156` rounded to `0.12353515625`; full tensor and signed-scale replay regressions |
| PG09 export loader attached before all target/tensor validation | Reuses the validated generic loader; malformed targets, shapes, dtypes, values and metadata reject without binding, base, tokenizer or RNG mutation |
| Dummy inference extended vocabulary but resume omitted it | Wrapper checkpoint retains the validated vocabulary, restoring it only after transactional shared-game restore; fresh-process CLI full-state replay regression |
| Export loader ignored conditioning/source/recipe labels | Requires current semantic and inference-source contracts, public preset recipe, model fingerprint and step bounds before attachment |
| Dictionary config silently ignored dummy context limit | Native encode reads the actual dict/object config; context-overflow regression rejects before model execution |
| Router allocated on CPU while other native leaves used host device | Constructor moves every adapter leaf to the host device; CPU/meta regression covers device placement without CUDA |

The shared v2 protocol deliberately changes local-loop coordinates, critic
conditioning and sampler. It is a new cohort; old local PG09/archival states
cannot resume silently. The public PG version remains **0.9.0**, never 0.10.

## Entry-point audit

| Entry point | Executed path / status |
|---|---|
| `tiny_llm_pg09.PG09TinyRun` | Shared `ConditionalPG09`, actual native bank, neutral student/frozen positive target, GAN-only objective, shared checkpoint/sampler |
| `scripts/qualify_tiny_pg09_cpu.py` | Explicit `--core`, verified import path/hash, synthetic CPU only, six updates maximum; independent-process continuation |
| `pg09_text_video.TinyQwenResponseAdapter` | Shared-game response bridge; native neutral readout, exact frozen zero scale; actual adapter ownership and source/domain gate |
| `TinyParticleSlider.save/load/scaled` | Native q/k/v/o projection weights and real bank; FP32 opt-in portable inference load, signed scales, exact zero bypass |
| `train_lora_tiny_llm.py` | Archival positive-caption Rp/VIC trainer, not PG09; its game is preserved, native routed forward reuses package; `_last` EMA and `_live_last` live weights remain explicit |
| `scripts/qualify_captions.py` / `caption_quality.py` | Archival native campaign/recovery and autoregressive `model.generate` within active adapter scale; not newly qualified shared PG09 generation |
| `qualify_text_video_pg09_cpu.py` / `qualify_wide_text_pg09_cpu.py` | Existing shared test factories; release/native-width profile, replay and inference checks rerun |
| Family owner's `verify_tiny_pg09_native.py` | Separate new actual-model worker/proposal; owned elsewhere and untouched by this audit |

The archival positive-student objective is not presented as neutral shared-GAN
training. Bonsai's readout-head transfer failure is not evidence about native
TinyQwen text generation. Existing held-out readout data were not optimized here.

## Integration and remaining gates

CPU tests use immutable shared core
`3e94b13eb5d6cd1ceee3f018f019a68603f35696`, PG module SHA256
`426abc25323aea96be91e92f1f4101063d5b3c4062f042e869c729027b4f9256`.
Wide-vector compatibility uses only the already reviewed isolated ALZ
`alz-continuous-v1` / `continuous` profile; release/auto rejects native width1024
before adapter attachment or response execution. No installed package was changed.

The historical in-repository core is version0.3.0 and does not expose PG09.
The qualifier now selects its actual shared core explicitly. Shared package
owner task14 is separately extracting the response/native helpers and exact-object
compatibility shims into API0.6.0 (still PG0.9). Retarget to the exact reviewed
wheel/commit only after wheel CPU gates and Ada review; moving source-bound
classes requires a new cohort or separately reviewed migration.

The same owner received a real random-Qwen3 CPU reproduction showing that
`bind_native_gradient_checkpointing` expects Diffusers
`enable_gradient_checkpointing`, while Transformers exposes
`gradient_checkpointing_enable(gradient_checkpointing_kwargs=None)`. The current
path fails closed; packaged support is tracked by the shared owner. This audit
does not silently disable native checkpointing.

Required native gates remain pretrained loader/model identity, frozen zero-scale
parity, live export identity verification, signed actual text generation, useful
held-out transfer, full-horizon training quality, memory/throughput and coordinator
process/GPU admission. The family owner prepares the bounded real-model worker;
no real weights were loaded, no GPU job/service/permissions/credentials changed,
no model was published and no PR was merged in this audit.

Receipts are in `evidence/tiny-pg09-audit/`. Synthetic optimizer updates establish
software operation and continuation, not pretrained-model quality.
