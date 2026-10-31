# H3 Flex shared response session — source draft

This adds an H3-owned consumer of the real `particle-sliders-core` response
lifecycle. Both the slider's down/up projections and the conditional critic
execute HNDL graphs built by **HyperGAN develop**. It does not copy a GAN
objective, regularizer, optimizer, checkpoint engine or training loop.

**Native H3 training is still refused.** Official ParticleGAN0.9.4 release/auto
does not admit the genuine wide continuous response. No driver, permissions,
GPU jobs, pretrained weights, native output or model publication are involved
in the software qualification below.

## Exact source cohort

| Component | Reviewed identity |
| --- | --- |
| Preserved H3 native-boundary commit | `3e252a9b18197cbd79a943b47db37b3fea0a385d` |
| Shared core candidate | `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, core0.12.0 |
| Shared candidate wheel receipt | `9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad` |
| Official ParticleGAN | `0.9.4`, `release-0.9.4/auto` |
| HyperGAN develop | `5f368a3770cf5bdae0448995c9f76e29af8049f7` |
| HNDL | `0.10.0` |
| Native Pika model/runtime | Original exact pins in [the native-boundary guide](pika-h3-flex.md) |

The candidate is a separate, parent-owned dependency/publication. This H3
change does not merge, vendor or modify it. The source-checkout consumer
requires the separately installed candidate and verifies its critical source
hashes from `data/pika-h3-flex-session.json`. It does not change `sys.path` or
install dependencies at runtime. The common verified builder gate currently
lives in `particle_sliders.fibo_hndl.verified_builder`; H3 calls that gate,
which verifies HyperGAN's clean build provenance and HNDL's complete source
closure. The executed graph sources are H3-specific, not Fibo's graphs.

## Actual execution and ownership

`HNDLH3FlexBackend` inherits pinned local loading, frozen Pika-at-1 behavior,
native packing and the guarded lifecycle from the original H3 boundary. It
overrides slider construction with `pika-h3-flex-slider.hndl`: two bias-free
linear projections with rank8/alpha8, the existing adapter-only LoRA
architecture. They run in float32, casting the resulting delta to the frozen
native host dtype. Scale zero calls the complete original PEFT forward.
No particle bank is fabricated; the declared arm is `adapter_gan`.

`pika-h3-flex-critic.hndl` describes the conditional raw-logit critic. Both
graphs are resolved and built through the actual verified HyperGAN/HNDL
builder. Graph source hashes and runtime provenance enter authoritative
architecture/checkpoint identity. A unit test compares HNDL projection values
and input derivatives with the matching native linear expression, including
a bfloat16 host with float32 trainable leaves.

`H3ResponseAdapter` exposes the complete packed video velocity followed by
audio velocity as a genuine continuous vector. Positive and neutral rows
retain their own native conditioning/layout and share video/audio noise and
times. Teacher responses use frozen base plus Pika, with slider zero. Student
responses use neutral conditioning at slider+1. Row identity is an explicit
one-hot critic condition; no response projection or image reshape is used.
This draft binds the **first native t2va evaluation point**, prepared by the
existing native boundary. It does not qualify control over all six sampling
steps or audiovisual quality.

`Binding` transfers the actual `ConditionalPG09` game, observation hook and
cleanup callback to the shared `ResponseSession`. The shared package owns
updates, EMA, observation RNG, checkpoint/resume and v1 numerical response
export/reload. Optional recomputation uses its reversible
`checkpoint_with_adapter_scale` handles; native internal checkpointing must
be off. Cleanup closes those handles before removing owned HNDL forwards and
retains ownership errors rather than restoring unrelated owners.

The custom slider context remains necessary. Its trainable wrappers are not
named PEFT adapters; Pika is the only PEFT name and remains fixed at1. The
shared PEFT-name selection helper cannot scale these custom graph leaves.
Existing fixed-companion and signed/fractional-scale tests remain intact.

## Native preflight and API

The following uses the public session configuration and delegates preparation,
packet validation and restore to the exact shared lifecycle implementation:

```python
from particle_sliders.lifecycle import SessionConfig
from conceptmod.textsliders.pika_h3_flex_session import open_session

config = SessionConfig("pika-h3-flex", "fresh-h3-response-cohort",
    snapshot="/path/to/pinned-pika-snapshot",
    rows=[{"neutral": "A fox in a forest.", "positive": "A red fox in a forest."},
          {"neutral": "A quiet room.", "positive": "A brightly lit quiet room."}],
    duration=5, resolution="480p", steps=4, batch=2, seed=7)
open_session(config)  # currently refuses before native loading
```

The declaration fixes genuine geometry: 124 prepared frames at832×480 produce
`(1,1398528)`. Source tests execute the official gate and verify that the native
loader is never called:

```text
ValueError: released PG09 cannot admit flat continuous vectors with width >=170
```

Public options reject fixture substitution, response width/shape overrides,
source-profile overrides and penalty-domain overrides. The in-progress
upstream continuous-domain proposal is not consumed. Removing this refusal
requires a separately reviewed source/profile adoption. Full-width critic
allocation, upstream optimizer memory behavior, hardware fit, pretrained
loading, native-only v2 export, standard-LoRA interchange, six-step media
sampling and quantized training remain unqualified. The standalone GGUF is
still an unqualified inference candidate, not a supported session input.

## Executed software checks

The shared lane uses installed candidate0.12, officialPG0.9.4, actual
HyperGAN5f368a/HNDL0.10, Python3.12.15, Torch2.14.1+cpu, PEFT0.21.2 and
Transformers5.22.1 with the pinned Diffusers source. Selected existing requests
dependencies are reused read-only in a task-local dependency view. This is not
a pristine full Pika environment or pretrained qualification.

The six-coordinate **random miniature native-H3 architecture fixture** has
its own explicit name/identity and is not selectable through public H3
options. It exercises actual HNDL projection/critic execution, shared PG
updates, scale-captured recomputation, unchanged frozen base/Pika weights,
observation isolation, exact four-step versus2+2 resume, shared export/reload
and changed-cohort refusal. Its artificial geometry never licenses the native
1.4M-wide model. No actual H3 training run is performed.

Run from this repository with the separately installed shared candidate:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  -o pythonpath=. tests/test_pika_h3_flex_session.py tests/test_pika_h3_flex.py
```

`-o pythonpath=.` prevents the repository's older core0.11 source path from
shadowing the installed candidate. The source hash gate rejects a mixed cohort.
The original Pika/legacy H3 suite is also retained in its original environment.

## Publication overlap

Historical draft [PR140](https://github.com/HyperGAN/particle-sliders/pull/140)
is a divergent stacked LTX/H3 audit. It overlaps
`minimax_h3_backend.py` and `docs/minimax-h3-slider.md` from the preserved native
commit. This focused change does not merge that branch, its historical shared
core, its LTX payload or its qualification claims. No other H3 Flex PR was found
at the pre-publication check. Parent controls actual-diff review, dependency
assembly, CI/release-impact checks and any merge.
