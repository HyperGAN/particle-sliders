# Shared boundary and state-machine follow-ups

The original audit and 13 priority fixes merged into `main` on 2026-11-06.
The following new PRs target `develop`, created from reviewed main
`285180ed9813f12e912bb6bce6bada00496fd19a`. They are separate review units and
remain open. The product PRs target their own repository's `develop`.

## Shared core and Music3

- [Shared-core boundary #191](https://github.com/HyperGAN/particle-sliders/pull/191):
  core0.14 installs generic lifecycle, adapters, execution and tuning contracts;
  excludes embedded native products, prompts and legacy namespace hooks from
  regular and strict editable installs. Generic flow conventions are explicit
  inputs. Replaced source and native tests remain archived on their old cohort.
- [Music3 extraction #16](https://github.com/HyperGAN/music3-particle-sliders/pull/16):
  product0.4 owns its native loader, routed architecture and response subclass,
  consumes the shared game and pins reviewed core runtime `9e9dcea`. The complete
  previous0.3 source/lock is retained; captured rates/checkpoints are not rebound.
- [Shared hyperparameter search #195](https://github.com/HyperGAN/particle-sliders/pull/195):
  a new raw-only schema/cohort port of preserved prototype `98b2c0b`. It binds
  actual optimizer rates, retains full response coordinates and exact resume,
  and ranks strictly by the one FP64 metric. Media, OFF, native availability,
  boundaries and alignment remain diagnostics. No elapsed-time cap is imposed.
  The PR depends on #191; original source and native receipts remain historical.

## Separate follow-up PRs

| Finding | PR | Worker validation |
|---|---|---|
| C18/C19: reader failure can omit EOF and strand the controller | [#185](https://github.com/HyperGAN/particle-sliders/pull/185) | 39 passed, 5 optional skips; real malformed-output children |
| O03: DSP cache omits content and measurement identity | [#186](https://github.com/HyperGAN/particle-sliders/pull/186) | 11 regressions; real WAV bytes |
| A05: newly attached adapter recaptures already-scaled peer baselines | [#187](https://github.com/HyperGAN/particle-sliders/pull/187) | 24 CPU checks |
| A03: zero strength still executes nonfinite adapter math | [#188](https://github.com/HyperGAN/particle-sliders/pull/188) | 63 CPU checks |
| A04: empty/nonfinite/unknown envelope mutates activation | [#189](https://github.com/HyperGAN/particle-sliders/pull/189) | 56 CPU checks |
| O20: existing files impersonate completed distillation | [#190](https://github.com/HyperGAN/particle-sliders/pull/190) | 10 stdlib fault/adoption checks |
| T17: teacher-conditioning cache omits model/adapter/settings | [#192](https://github.com/HyperGAN/particle-sliders/pull/192) | 25 passed, 5 subtests |
| T12: multi-file GANv2 publication exposes mixed generations | [#193](https://github.com/HyperGAN/particle-sliders/pull/193) | 87 passed, 5 optional skips |
| O16: stale training marker waits forever; zero-step evidence is deleted | [#194](https://github.com/HyperGAN/particle-sliders/pull/194) | 42 CPU process/ownership checks |
| A08: Music3 scale scope loses heterogeneous effective strengths | [Music3 #17](https://github.com/HyperGAN/music3-particle-sliders/pull/17) | 39 CPU checks; depends on extraction #16 |

The T12 migration is bounded to GANv2 plus a reusable generation publisher.
Other historical multi-file formats/captured callers keep their original pins
and remain migration input. O16/O20/T17 repair retained historical consumers;
those model-specific consumers are excluded from the0.14 wheel. Their generic
helpers live in the already-packaged execution namespace. Historical O16/O20
selection formulas are not promoted to the current scientific protocol.

## Verification and integration

Core0.14's active installed-package suite passed **554 tests, 23 skipped** both
outside the checkout and through the default repository command. The skips are
17 optional HNDL cases, 5 pinned-reporting-runtime opt-ins and 1 collection skip.
The final build used setuptools84.2.1 and ParticleGAN0.9.5. All39 core submodules
imported with CUDA uninitialized. The wheel has41 Python files and53 runtime/data
files; rebuilding through sdist preserved the complete runtime map. A later
stale-directory-symlink guard passed its targeted real-wheel regression and left
the runtime map identical. Default tests no longer inject the legacy source tree.

Music3 passed123 source tests (97 explicit historical/native skips) and115
isolated installed-wheel tests (3 repository-only archive assertions covered by
the source suite). CPU evidence covers owned gradients, true OFF, raw reload,
malformed/late load rollback, RNG, exact continuation and shared LR controls.
It does not qualify pretrained/GPU admission, acoustic generation or new rates.

Root independently reran reader tests26, WAV-cache tests11, A05 tests2 (the optional
PEFT module skipped in that environment), true-OFF tests30, envelopes34,
distillation10, teacher-cache17, generation/execution87 (5 optional skips),
owner recovery42 and Music3 scale tests4. These overlap the worker suites and are
not additive totals. The previously excluded old-cohort editable test was also
closed: its full install/uninstall suite passed3 checks in a clean private setup.

A private combined preview at `2dce2b09` passed **624 installed tests, 60 skipped**
and **79 retained-source tests plus3subtests**. All42 submodules imported from
private site-packages without excluded native products or CUDA initialization.
The37 additional skips identify the historical LoRA/DoRA variants excluded by
0.14; all source0.13 adapter cases still run. Two expected native-source receipt
conflicts were recomputed from the actual combined bytes; there were no code
conflicts. The preview was not pushed or merged.

Before release, retain the reviewed receipt resolutions and refresh product
source locks for the newly combined core. Product #17 follows #16; tuning follows #191. Existing packets,
measurements and native captures keep their original immutable environment.
All prospective work stays raw-only (`ema_decay=None`), keeps upstream scalar
optimizer statistics and ranks solely by operand-first FP64 per-example pooled
relative SSE, equally averaged across examples. No training deadline is added.


The tuning PR passed **627 installed tests, 23 skipped** and **73 independent
focused tests**. Its exact accumulator matched100 independent rational-sum probes.
Review fixed optional-media drift blocking offline raw audits and incorrect
optimizer base-rate admission. Root reran29 metric tests and10 ranking/admission
regressions. Both CPU CLI examples completed plan/run/audit/idempotent-repeat;
their32 trials consumed256/240 physical updates respectively, and repeat consumed
zero extra updates. A separate build with setuptools84.2.1 preserved all7 search
module bytes and excluded product implementations. This qualifies generic CPU
search infrastructure, not pretrained/native rates or quality.
