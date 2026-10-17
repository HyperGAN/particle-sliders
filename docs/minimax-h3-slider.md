# MiniMax-H3 migration status

MiniMax-H3 training and native execution are **closed** pending a reviewed
native jointly packed audio/video response and critic protocol for the shared
ParticleGAN API. The embedded backend is not a completed GAN migration.

`train_lora_minimax_h3.py` rejects direct `train(...)`, ordinary CLI commands,
`--steps 0 --load_h3_lora`, and sample export before backend construction,
adapter loading or output writes. The legacy supervised UNI/MSE optimizer has
been removed. `--dummy` does not enable training. Native backend construction,
packing, forwards, adapter save/load, generation and non-dummy diagnostics reject
as well. Old flags remain parseable so stale commands explain the missing gates.
The closed trainer can report those gates with standard Python alone.

The remaining gates are reviewed native AV geometry and response/critic
semantics; exact staged source, asset and FL2VA/Ref2VA variant identity; the
applicable MiniMax-H3 community license; measured hardware fit; and admission by
the runtime coordinator. No model, GPU job, service or permission change is
needed to inspect this status. Hardware or license availability alone would not
supply the missing multimodal GAN protocol.

The source identifies `MiniMaxAI/MiniMax-H3`, a jointly packed text/video/audio
Omni-Transformer with modality indices, token tags and MM-RoPE. Its output has
separate video and audio velocities. The existing two-video-token/two-audio-token
layout is a synthetic architecture fixture; it does not establish native AV
geometry. Flattening those outputs or reshaping them as an image would not
establish a reviewed response domain. `predict_v(...)` continues to reject.

The removed trainer compared the adapted positive input against the frozen
teacher on that same positive input with MSE, so zero adapter was already an
optimum. Its frozen embedding hold had no adapter gradient. Neither loss nor
legacy adapter-only files provide a valid shared GAN training or resume route.
Future support must use the installable shared API with actual active adapter
weights, neutral student conditioning, declared response coordinates/signs,
GAN-only objectives and transactional checkpoint/resume validation.

Historical configurations (`config-minimax-h3.yaml` and
`config-minimax-h3-chiaroscuro.yaml`), prompts and native source details remain
for audit. Their UNI recipe names, training schedules and old hardware notes are
historical records, not executable recommendations or current admission.
Existing native adapters do not establish a reviewed shared API checkpoint.
The previous MP4 fallback wrote a container header after codec failure; public
sample export is now closed and cannot manufacture a successful video artifact.

CPU diagnostic fixtures remain available:

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_minimax_h3.py --dummy --diag \
  --save_dir /tmp/h3-diag
PYTHONPATH=. python conceptmod/textsliders/diag_minimax_h3_uni.py --dummy \
  --save_dir /tmp/h3-diag-direct
PYTHONPATH=. pytest tests/test_minimax_h3_slider.py tests/test_h3_embedded_closure.py -q
```

These diagnostics measure synthetic packed velocities and embedding hold using
small CPU modules. They offer no native runtime or quality qualification. Dummy
LoRA save/load is retained for architecture tests and diagnostic fixtures only;
it is not optimizer resume, and synthetic frame helpers are not AV render tests.

The independent closure tests also run without installed model dependencies:

```bash
python -S tests/test_h3_embedded_closure.py
```

See [text/video integration](text-video-pg09-integration.md),
[current status](text-video-pg09-current-status.md) and the
[real-model gates](text-video-pg09-real-model-plan.md) for the retained protocol,
source, asset, license and coordinator constraints. No current runtime or
quality admission is claimed here.
