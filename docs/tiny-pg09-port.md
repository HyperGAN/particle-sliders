# TinyQwen PG0.9 port milestone

This is a new explicit training arm, preserving the existing caption trainer
and its completed checkpoints. `tiny_llm_pg09.PG09TinyRun` uses the real existing
TinyQwen backend and routed projection adapter. Its CPU qualification uses the
backend's random stand-in; it does not download pretrained weights or establish
language/audio quality.

The legacy trainer prepares `ids=plus_ids` and feeds those positive-caption IDs
to its adapted student. Inference tests neutral captions. This new arm feeds
the neutral caption to the student and keeps the frozen positive caption as
the target, making conditioning match its intended inference call. This is an
explicit change of experimental protocol, not a retroactive claim about PG0.9
being used in old runs.

The port calls released `get_recipe('bispf_optimized',...)` and public optimizer,
loss, penalty, prior-regularizer, scheduler and constraint-backward APIs through
`pg09_game.PairedPG09`. The MLP error critic is an explicit architecture choice;
no old rank-3 mixed critic is silently substituted. One `[128,4]` particle bank
is shared across the rank8 projection branches, and dense routing binds all128
active table rows before the PG optimizer step. The SBNP weight is applied once.

The adapter is explicitly FP32 even when its frozen host is BF16. The new export
loader keeps that precision, avoiding an intermediate copy into BF16 adapter
weights. The CPU fixture's tokenizer vocabulary is retained for exact reload;
this fixture detail is distinct from the pretrained model's fixed tokenizer.

Checkpoints retain exact resolved recipe fields, source hashes, model fingerprint,
teacher/conditioning identity, network/critic weights, PG optimizer state and
private sampling/model RNG streams. Legacy states are rejected. Sampling does
not consume the training streams. Tests cover real parameter changes in up,
down, routing, bridge and particle groups; a single ownership entry per particle
bank; frozen-base exclusion; exact zero scale; exact resume; and export/reload
at0,.5,1 on FP32 and BF16 frozen hosts.

Use the isolated verified0.9 environment:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/nvme/sliders/qualification/envs/pg090/bin/python -m pytest -q tests/test_tiny_pg09_port.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /mnt/nvme/sliders/qualification/envs/pg090/bin/python scripts/qualify_tiny_pg09_cpu.py \
  --output /tmp/tiny-pg09-new-run --until 2
```

The CPU CLI refuses budgets beyond six updates. It has no live-download flag.
The class requires an externally verified staged-model fingerprint for a live
backend; its live export loader deliberately remains blocked pending that
verification. Pretrained-model GPU admission, generation/readout QA, memory and
throughput qualification remain undone. No native held controller uses this
new arm yet, and no old run's budget or optimizer state is replaced.
