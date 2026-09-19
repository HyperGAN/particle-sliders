# Uni-v2 plus: last-token hit, lyric garble

Hand this to someone who has not been in the train thread.
Live listens, not the CPU 2-d fixture. Default trainer is still
`--lm_target v9` / `--pole_mode hidden`. This page does not change
that, does not update the bipolar board, and does not rewrite v4.

**Claim:** same-room uni-v1 yamls + `--lm_target faithful_plus_neu`
still shred lyrics on several plus LoRAs. Train `c+` / `p%` will
call those runs a hit. Whisper lyric recall will not.

## What we trained

UNI plus only. One concept per yaml. Teacher is raw `h+` (the +
caption). Scale 0 is taught `h0`. Minus is not a teacher.

```bash
CUDA_VISIBLE_DEVICES=N python conceptmod/textsliders/train_lm_slider_music3.py \
  --name energy-lm-uni-v2 \
  --prompts_file conceptmod/textsliders/data/prompts-energy-uni-v1.yaml \
  --lm_target faithful_plus_neu --pole_mode hidden \
  --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 \
  --no-early_stop --endreg_weight 1.0 --device 0
```

Same card for the other plus axes (`--name <axis>-lm-uni-v2`, matching
`prompts-<axis>-uni-v1.yaml`). Do not pass v4. Sidecars on this round
record `lm_target=faithful_plus_neu`, `plus_neu=true`,
`prompts_file=…-uni-v1.yaml`.

Weights: `models/<axis>-lm-uni-v2/`.
Listens: `eval/listen/uni-v2/<axis>-lm-uni-v2/` (scales 0, 1, 2 plus
the + REF and Off REF). Driver: `scripts/train_uni_v2_plus.sh`.

## How a listen is built

Slider clips encode the **neutral** caption + yaml lyrics, LoRA scale
only. REF clips change the **prompt** with the slider off.

That is the product path: user prompt stays the song; the fader is
supposed to add the concept. If +1 garbles and the + REF still sings
the yaml line, the caption is singable and the LoRA is what broke AR.

## Two different failures (do not mix them)

1. **Two-song yaml (v4).** Genre / BPM / mix / kit ride with the
   axis. `h+` is another track. High `c+` means “arrived at the other
   song.” Lyrics from the yaml get replaced or shredded. Fix attempted
   in uni-v1: pin room, flip only the + words in one field.
2. **Last-token transplant (uni-v2, this page).** Yaml is same-room.
   Loss is last-hidden MSE only:

   student = encode(neu tokens, LoRA @ +1)
   teacher = encode(pos tokens)   # same lyrics, + words flipped
   loss    = MSE(last, h+) + MSE(LoRA @ 0, h0)

   Last real token is `<|audio_start|>`, which is what AR continues
   from. The LoRA still rewrites **every** prefix token, including
   lyrics, because it sits on Qwen3Attention. Matching the last
   hidden to `h+` does not match the KV cache of the neu prefix.
   Generation then runs with a pos-like last hidden and a neu KV.
   That is lyric garble even when the + REF (pos caption, no LoRA)
   still sings the line.

Gender mostly survives (1): woman/man is an on-manifold LM
attribute. Grit does not survive (2): fry / rasp / gravel is a
delivery transplant that the last-token loss will happily “hit.”

## Ears: whisper lyric recall at +1

`openai/whisper-tiny` on CPU, yaml lyrics from each `LISTEN.md`,
word recall of the transcript against that line. Tiny-whisper on
singing is noisy; the top and bottom of this table are not close
calls.

Ranked worst `+1` first. Eight finished plus axes.

| rank | axis | 0 | **+1** | +2 | REF+ | +1 transcript (head) |
|---:|---|---:|---:|---:|---:|---|
| 1 | grit | 0.62 | **0.00** | 0.15 | 0.85 | `Those are those are those are` |
| 2 | distortion | 0.62 | **0.08** | 0.69 | 0.85 | `I` |
| 3 | joy | 0.85 | **0.23** | 0.85 | 0.62 | chorus loop, verse gone |
| 4 | hurt | 0.85 | **0.54** | 0.54 | 0.62 | `I still want to, you're such a store…` |
| 5 | energy | 0.85 | 0.62 | 0.46 | 0.85 | first line only, chorus gone |
| 6 | rapslow | 0.75 | 0.75 | **0.00** | 1.00 | verse ok at +1; +2 is `Yeah.` |
| 7 | gender | 0.54* | 0.85 | 0.85 | 0.92 | yaml line intact |
| 8 | tempo | 0.69 | 0.92 | 0.85 | 0.69 | yaml line intact |

\*gender 0 is a whisper miss (`in the end too now`); +1 and REF+ are
clean. Grit REF+ still sings the yaml line (0.85) while grit +1 is a
loop (0.00): that is failure (2), not a bad + caption.

Do not ship grit / distortion / joy on this card. Gender and tempo
are the only two that keep lyrics at +1. Rapslow is fine at +1 and
dead at +2.

## Train `c+` / `p%` will not show this

Logged `c+` is cosine of `(pred_+1 − h0)` with `(h+ − h0)`.
Logged `p%` is `‖pred_+1 − h+‖ / ‖h+ − h0‖`. Both are last-hidden
geometry. Last-50 window on the same eight runs:

| axis | whisper +1 | last-50 c+ | last-50 p% | mean ‖h+−h0‖ |
|---|---:|---:|---:|---:|
| grit | **0.00** | 0.945 | 32% | 8.1 |
| rapslow | 0.75 | 0.948 | 32% | 7.6 |
| gender | 0.85 | 0.990 | 15% | 13.6 |
| distortion | 0.08 | 0.812 | 60% | 2.0 |

Grit and rapslow are the same hidden run. Opposite lyrics.
Spearman vs whisper +1 across the eight: last-50 `c+` / `p%` / L2
all ρ ≈ +0.52 (one story: big on-manifold deltas land and look
good). Absolute miss `p% × L2` ≈ 0. Collapse ≈ 0. Planreg `pdrift`
goes the **wrong** way (gender has the most drift and the best
lyrics).

High `p%` on a *small* `‖h+−h0‖` (distortion 60% on L2 2.0, energy
69% on L2 2.9) is a dirty miss: the adapter moved the LM and never
landed. That proxy catches distortion/energy. It does **not** catch
grit, which landed.

This is the same fact as [lm-sheet-goodhart.md](lm-sheet-goodhart.md):
hidden-only fields cannot see lyric garble. The plus+neu CPU exam
([lm-plus-neu-exam.md](lm-plus-neu-exam.md)) ranked
`faithful_plus_neu` first on **caption-word cover** and **neu_hold
at scale 0**. It does not score yaml-lyric survival at +1 on a live
rollout. Cover ≠ lyrics.

## What to measure instead of ears, if you must

1. **Whisper lyric recall** on +1 vs 0 vs REF+, yaml line as
   reference. Live gate. `scripts/blindspot_whisper.py` is the
   large-v3-turbo version of the same number; the table above used
   tiny on CPU so we did not steal the train GPUs.
2. **Off-caption / off-sheet mass on a continuation**, not on the
   last hidden. That is the sheet cell (`on-sheet` vs `garble`
   tokens) and the plus-exam gate `off-caption ≤ 0.05`. Toy vocab,
   not live Qwen.
3. **Free-run AR** (`scripts/probe_lm_policy_ood.py`,
   `scripts/probe_lm_free_run.py`). First-token KL to `encode(pos)`
   is *another last-token metric*: if grit really hit `h+`, it looks
   clean and the garble is later in the rollout. Score sampled
   semantic tokens against the lyric sheet, or compare the roll to
   the + REF roll, not only to scale 0.

Until (3) is a logged train number, **whisper (or a coworker with
headphones) is the gate.** Do not pick checkpoints by last-50 `c+`.

## Reproduce

```bash
# train + sample (refuses v4)
./scripts/train_uni_v2_plus.sh 0 energy gender tempo distortion
# play
#   eval/listen/uni-v2/<axis>-lm-uni-v2/02_*_plus1.wav
#   vs 01_*_zero.wav and 04_REF_*_no_slider.wav
```

A pass is: +1 still sings the yaml line, and leans toward the + REF
on the concept (loud / female / fast / …) without becoming a
different song.

## Related

- [lm-uni-v1.md](lm-uni-v1.md) — uni yaml shape (one concept, no `leak_*`).
- [lm-plus-neu-exam.md](lm-plus-neu-exam.md) — CPU plus+neu rank (cover / neu_hold).
- [lm-sheet-goodhart.md](lm-sheet-goodhart.md) — why hidden MSE cannot see garble.
- [lm-plus-exam.md](lm-plus-exam.md) — plus-only cover / off-caption, leftover-gate card.
- [lm-pair-exam.md](lm-pair-exam.md) — bipolar continuation gates (not this).
