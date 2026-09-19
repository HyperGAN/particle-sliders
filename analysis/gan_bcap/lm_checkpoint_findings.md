# Checkpoint reload verification

`lm_evaluate.py` loaded the bf16 language model once, cached pristine caption
teachers before attaching LoRA, then strictly loaded four final checkpoint
files. Evaluation used all four training rows and four new arrangement/lyric
combinations at scale +1. Every neutral/positive lyric span was verified to
contain identical token IDs. All **32 zero-scale checks were bitwise identical**
to the pristine model, including the full hidden sequence.

The repaired model without the conflicting neutral lyric hold produces a
substantial caption-directed change, including on heldout inputs. The held
variant does not match the intended caption change despite using the same
120-update budget.

| Checkpoint | Updates | Training last-delta cosine | Training normalized error | Heldout last-delta cosine | Heldout normalized error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `repaired-tx-smoke` | 120 | **0.913** | **0.412** | **0.860** | **0.517** |
| `repaired-tx-hold` | 120 | 0.359 | 1.400 | 0.270 | 1.365 |
| `baseline-tx` | 32 | 0.072 | 0.997 | 0.081 | 0.996 |
| Existing `gender-uni-tx2` | 400 | 0.170 | 1.059 | 0.142 | 1.051 |

Cosine compares `h(neutral + LoRA) - h(neutral)` against
`h(positive) - h(neutral)`. Error is
`norm(h(neutral + LoRA) - h(positive)) / norm(h(positive) - h(neutral))`;
an unchanged neutral model scores 1, and a perfect caption match scores 0.
The repaired smoke checkpoint's last-token magnitude ratio is 0.887 on
training rows and 0.790 on heldout rows. These are measurements of loaded
weights, not averages copied from training logs.

| Checkpoint | Training lyric-delta cosine | Heldout lyric-delta cosine | Training lyric normalized error | Heldout lyric normalized error |
| --- | ---: | ---: | ---: | ---: |
| `repaired-tx-smoke` | **0.827** | **0.700** | **0.584** | **0.726** |
| `repaired-tx-hold` | 0.131 | 0.275 | 0.996 | 0.961 |
| `baseline-tx` | 0.096 | 0.132 | 0.996 | 0.991 |
| Existing `gender-uni-tx2` | 0.044 | 0.104 | 0.999 | 0.995 |

The smoke checkpoint moves the lyric hidden states toward the positive
caption's aligned lyric states. The neutral hold opposes that motion: it
reduces relative lyric hidden RMS drift from 0.364 to 0.087 on training rows,
but also removes most of the teacher-directed span change. A small neutral
hidden drift therefore cannot serve as the sole success criterion for a
discriminator trained to match positive-caption span states. Conversely,
matching those hidden states does not establish transcription fidelity;
generated audio needs a separate check.

The baseline and older checkpoint have unequal training budgets, shown
explicitly above. This audit demonstrates failure versus successful movement
and the matched-budget hold conflict; it is not a matched-budget sweep of
every historical implementation.

Full per-row metrics, exact evaluated heldout prompts/lyrics, topology,
checkpoint SHA-256 digests, and prompt-source provenance are in
`lm_checkpoint_comparison.json`. Runtime was 10.2 seconds on GPU 0 after
which the process exited and released its GPU allocation.

Run from `/ml2/music/sliders-conceptmod` on an assigned free GPU:

```bash
CUDA_VISIBLE_DEVICES=0 /home/mikkel/anaconda3/envs/minimax-music3/bin/python \
  analysis/gan_bcap/lm_evaluate.py \
  --weights models/gan-bcap-repair/repaired-tx-smoke/repaired-tx-smoke_last.safetensors \
            models/gan-bcap-repair/repaired-tx-hold/repaired-tx-hold_last.safetensors \
            models/gan-bcap-repair/baseline-tx/baseline-tx_last.safetensors \
            /tmp/opencode/lm-adv/gender-uni-tx2/gender-uni-tx2_last.safetensors \
  --output analysis/gan_bcap/lm_checkpoint_comparison.json
```

The evaluator supports subsequent final checkpoints through `--weights` and
can evaluate multiple files in one model load when their LoRA topology matches.
