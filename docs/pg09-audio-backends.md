# Audio response bindings for shared ConditionalPG09

Current integration: exact reviewed `alz-continuous-v1` profile is supported
through core `3e94b13eb5d6cd1ceee3f018f019a68603f35696`. Native random
YuE2-2048 and Qwen3-4096 CPU gates pass on their separate runtime stacks.
Pretrained/GPU/listening qualification awaits core-owner admission. The
released default still rejects wide vectors. See [live validation](pg09-audio-live-validation.md).

## Scope and compatibility

GitHub's public `HyperGAN` repository API was enumerated on 2026-10-17: page 1
returned 20 repositories and page 2 returned 0. Exactly two audio/music product
repositories with the `-particle-sliders` suffix were found. Their actual
default branches were cloned rather than guessing repository names.

| Repository | Inspected `main` HEAD | Native training response | Decision |
|---|---|---|---|
| `HyperGAN/yue2-particle-sliders` | `b33794d81b92c125891a011531f55554d65e2082` | Final AR hidden state `[1,2048]`, neutral/positive captions with identical continuation history | Native backend/ParticleSlider binding and explicit-profile CPU gates pass; pretrained lease pending |
| `HyperGAN/music3-particle-sliders` | `b40d2aa6fed745eda3043d3d3b11e362ec2f8dfe` | Final Qwen3 prompt hidden state `[1,4096]` | Product live trainer is explicitly a skeleton; reuse existing shared Music3 `ParticleNetwork` and native Qwen3 forward; explicit-profile CPU gates pass; pretrained lease pending |

The Music3 product also names diffusion-transformer and condition-encoder
hosts. Its published routed-particle host is the LM. Those other hosts have
no completed live product binding and are not silently substituted for it.
No additional HyperGAN audio `-particle-sliders` repository was found. Other
modalities were excluded from source inspection and implementation.

Both product repositories pin the older `winning_formulation()` package at
`a119ca1ecd3d5d6c437065839d22739b04f2f4d8`. Those product training loops are
not invoked by the new adapters. YuE2's existing native slider obtains its
router architecture through the shared stamp, while all updates, objectives,
regularization, optimizers, sampling and checkpoints come from the single
`particle_sliders.pg09.ConditionalPG09` game.

## Binding and response contract

`conceptmod/textsliders/pg09_audio.py` adds:

- `YuE2ARResponseAdapter(backend, slider, model_identity=...)`, reusing the
  product `YuE2Backend.hidden` and the attached `ParticleSlider`.
- `Music3ARResponseAdapter(model, slider, model_identity=...)`, reusing the
  native Qwen3 causal model and existing
  `conceptmod.textsliders.music3_particle_bridge.ParticleNetwork`.
- `AudioRow` and `audio_conditions`, with exact neutral/positive token tuples,
  shared continuation-history length and unipolar training scale +1.

Prepare caches frozen positive and neutral final-token hidden states. Student
forwards always use neutral tokens. The response is the genuine last AR
hidden state, with no projection, truncation, field reshape or logits
substitution. Frozen context features for the conditional critic are the
entire neutral hidden state plus neutral/positive token lengths, continuation
length and adapter sign. The core independently binds the exact input token
bytes, targets and condition bytes in its checkpoint signature.

The adapter-only parameter view reuses every existing native trainable and
exposes the **same participating 128×4 bank** as `adapter.particles`. Pass that
exact object as `particles=adapter.particles` to `ConditionalPG09`. It enters
the actual hooked model response and is owned once by the public optimizer.
No extra bank or prior is manufactured. The generator objective is public
adversarial loss plus public prior regularization; the critic objective is
public adversarial loss plus public penalty. No MSE or auxiliary objective is
called.

`sample(prepared, scale=...)` is explicitly a hidden-state probe. It preserves
RNG and checks scale 0 exactly against the frozen neutral response. It does
not claim to generate tokens, invoke CFG, decode a VAE, or render audio.
Audio generation/decoding and full-model quality remain unqualified.

## Released-default domain guard and pretrained gates

Released unchanged PG0.9 and the shared contract reject flat continuous
vectors of width 170 or greater. Both native pretrained responses exceed this:

- YuE2 model `m-a-p/YuE2-3B` at
  `14fc6c6f146441b1dd6363fcb2e01e82a6914cb7`: hidden width 2048;
  config SHA256 `ad3477bbef890bf98ae196c1e4b44779494a6231c4ab66f32708eabadf265329`.
  Native runtime pin `multimodal-art-projection/YuE` at
  `ef1936f2ee39fe8de486a0f47a481c95f8d4da87`.
- Music3 model `MiniMaxAI/MiniMax-Music3` at
  `bd348f9c49ea3c1b39f33ace3436f8fad435f24e`: native Qwen3 hidden width 4096.

`require_audio_width` delegates to the shared public `validate_pg09_domain`
before any forward. The default `release`/`auto` pair rejects these widths.
Only the explicitly selected `alz-continuous-v1`/`continuous` pair accepts
them when all 49 imported sources match the reviewed manifest. The single
ALZ overlay remains isolated and uninstalled. The same options are required
on `ConditionalPG09`; source/profile/domain identities are checkpoint-bound.
No response is projected, truncated or reshaped. A later admitted audio render must also qualify its actual
AR/token-sampling/decoder path; hidden probes are not listening evidence.

Only CPU tiny random models were constructed. No pretrained weights were
loaded, no GPU was initialized, and no service, hold, permissions, publication
or merge was changed. The core owner remains sole GPU/process admission
owner.

## CPU validation

`scripts/pg09_audio_cpu_fixtures.py` constructs the discovered product's
32-wide `TinyYuE2` stand-in and an actual random 32-wide native Qwen3 model.
Both reuse the existing native routed sliders. This validates software
connectivity; neither fixture is pretrained audio-quality evidence.

The test suite covers both `bispf` and `bispf_optimized`, complete native
trainable updates, a nonzero directional response gradient to the actual
bank, frozen host weights, conditioning, token snapshots/history validation,
unipolar targets, per-forward scale checkpoint recomputation, portable exports,
transactional incompatible-checkpoint rejection and independent probe RNG.
Three fresh CPU processes compare uninterrupted 3 updates against 1+2 updates
across both families and both presets. Every loss, state tensor, optimizer,
regularizer, RNG, sample and export field is compared exactly.

Final dependencies:

- Shared core: `3e94b13eb5d6cd1ceee3f018f019a68603f35696`, frozen in the
  isolated checkout `/mnt/nvme/sliders/audio-core-pin-86fc4e0-20261017`
  (the directory retains its initial pin in its name; its HEAD is 3e94b13).
- Common checkpoint helper: `ec34e4ecc68aedc88bb5cd44a96d4bd78af99fa9`,
  cherry-picked as `fbf97b3557779d97cb06a9c1b3194abcb5c21a82` in this
  branch. The helper is byte-identical; no audio-local context wrapper remains.
- YuE2 product and Music3 product: exact heads in the inventory above.
  No product requirements were installed over the isolated released PG0.9 env.

The initial native audio milestone passed **17 tests with no skips in 17.25 seconds**,
including all four fresh-process replay combinations. Native Transformers
`id2label` integer keys are retained. The final receipt stores the exact
source files, environment versions, resolved recipes and replay digests.
The shared checkpoint-helper and conditional-core regression suite also passes
**49 tests with no skips in 4.71 seconds**. Lint and `git diff --check` pass.

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/nvme/sliders/audio-core-pin-86fc4e0-20261017/packages/particle-sliders-core/src:/mnt/nvme/sliders/audio-yue2-source-20261017:/mnt/nvme/sliders/audio-adapters-pg09-20261017:/mnt/nvme/sliders/audio-adapters-pg09-20261017/scripts
cd /mnt/nvme/sliders/audio-adapters-pg09-20261017
/mnt/nvme/sliders/qualification/envs/pg090/bin/python -m pytest -q \
  tests/test_pg09_audio.py -o pythonpath='' -p no:cacheprovider \
  --basetemp=/tmp/pg09-audio-final --junitxml=/tmp/pg09-audio-tests.xml
```

JUnit and source receipts are retained in `evidence/pg09-audio/`. The checkpoint
fix is shared infrastructure authored by its owner and applied as a dependency;
the audio implementation commit changes only audio files.

Explicit-profile follow-up: 4 native YuE2 width 2048 tests passed in 18.78s
and 4 native Qwen3 width 4096 tests passed in 15.71s, both presets and exact
fresh-process resume on the pinned native interpreters. The updated released
path and output suite passes 23 tests in 18.42s. These are random-model CPU
engineering checks; no pretrained quality claim follows. Receipts live under
`evidence/pg09-audio/domain-integration/`.
