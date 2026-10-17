# LTX-2.5 shared connector GAN API

`particle_sliders.ltx25_cli` is the installable API behind the historical
`conceptmod/textsliders/train_lora_ltx25.py` entrypoint. It executes
`ConditionalPG09` for every update. The old embedding MSE/relative-L2 and
velocity/hold optimizer loops have been removed from that entrypoint.

The current CLI admits synthetic CPU fixtures only. It does not qualify the
pretrained model, convergence, hardware fit, license applicability or audiovisual
rendering. The original native helper/diagnostic fixtures remain available;
they are not a second training route. The production text-only loader at
`c7096d475f70daee41fc161150cf8eeba1dfffc5` must be reused and reviewed before
admitting a new native runtime. No active source pin or old checkpoint is retargeted.

## Explicit source profile

The default connector output retains its complete native token/feature sequence
and fixed common mask. It is a continuous vector, not an image. Released
ParticleGAN 0.9.1 still misclassifies wide vectors through its automatic penalty
heuristic. Release/auto therefore rejects before backend construction. An
explicit `alz-continuous-v1` / `continuous` pair requires the exact reviewed
isolated 0.9.1 overlay. This CLI never installs or patches that profile.

With the package and that profile already installed in an isolated environment:

```sh
python -m particle_sliders.ltx25_cli --dummy --steps 4 --stop_after 2 \
  --rank 2 --alpha 2 --te_last_n 1 \
  --pg_source_profile alz-continuous-v1 --penalty_input_domain continuous \
  --save_dir /tmp/ltx-gan
```

Resume with the identical command/session options, `--resume
/tmp/ltx-gan/checkpoint.pt`, and `--stop_after 4`. `--steps` is the fixed full
horizon. Full checkpoints include game/optimizer/sampling/RNG state. Inference
exports contain the actual adapter leaves and source/protocol signature:

```sh
python -m particle_sliders.ltx25_cli --dummy --steps 0 \
  --load_export /tmp/ltx-gan/adapter-export.pt --save_dir /tmp/ltx-probes
```

Inference reconstructs the stored session and validates it through the same
shared core before loading weights. It cannot resume training. Old
`--load_ltx_lora` files are rejected because they lack this binding.

Training uses frozen positive teacher features with native preconnector
nonconcept hold, live neutral student conditioning, balanced independent D/G
row streams, raw conditional critic logits and only the public GAN objectives.
The actual adapter leaves are FP32; frozen BF16 hosts retain their dtype. There
is no routed particle bank in this adapter GAN arm (`particles=None`).

Output consists of `checkpoint.pt`, `adapter-export.pt`, `result.json` and
optional `response-probes.pt` at scales 0, 0.5 and 1. These are connector response
probes. No MP4 or audio is fabricated. Full native shapes/masks, staged loader
identity, memory/runtime, denoise/decode parity, AV preservation and useful
transfer still require coordinator admission and review. Only coordinator
`01a147d7-f540-778e-ad79-4d8eb267403d` owns runtime admission.

## Review composition

The video delta stacks on the frozen family source `ed3e4a9` and depends on
the task14 installable API. `scripts/build_video_candidate_wheel.py` composes
exact core/video Git objects into a review candidate without editing either
checkout. `scripts/verify_ltx25_sourcefree.py` tests an installed candidate from
an empty temporary working directory with isolated Python. Both presets must
match full four-step training to two-plus-two fresh-process continuation and
inference probe reload exactly, and reject altered source signatures before
writing output. The receipt records the exact wheel and source hashes; the
first task14 wheel alone does not include this video delta.

Current acceptance uses replacement shared core
`4be0b5b29d2e2c741431f468d71ba8a7c884b724`; the earlier `42a2e371` candidate is
superseded. Exact producer/consumer wheel hashes and preserved prior lineage are
in `evidence/video-audit/current-dependency-lock.json`. The four video module
sources are unchanged. No active cohort or historical checkpoint is retargeted.
