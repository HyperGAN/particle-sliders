# Predictive KL and the smoke listening preference

The user likes the 300-update smoke audio and disputes the earlier suggestion
to prefer 120 based on last-token hidden error. The subsequent instruction is
to explore further by manual sampling; 300 is not an established optimum.

The measured predictive KL does **not** recover that listening preference.
It is useful as teacher-policy agreement, but it is not being installed as a
stopping rule or a training objective.

## What was measured

For each of four development heldout prompts, the pristine model composes two
reference continuations: one from the neutral caption and one from the positive
caption. Seed 101 and a 250-frame cap are fixed. Every checkpoint is evaluated
on those same eight histories. Both the positive-caption teacher and the
neutral-caption adapter receive the identical appended audio-frame embeddings
within each comparison.

At each position, the actual frozen LM head maps hidden states to the allowed
semantic audio-code logits plus the audio-end logit. Softmax at temperature 1
defines the distribution. The evaluator records KL in both directions,
Jensen–Shannon divergence, predictive entropy and EOS probability. It separates
the first audio prediction from predictions after audio frames have arrived.
Mean KL is per position, then averaged equally across the four prompt rows;
neutral and positive reference histories are reported separately.

These are **raw conditional policies before CFG and top-k**. The metric does
not include the depth-code decoder, the flow transformer, or audio perception.
It is not full sequence KL: the histories are fixed reference continuations,
not trajectories sampled from each candidate. This avoids comparing different
songs token by token, while also excluding candidate-specific rollout errors.
The forwards use shared teacher-forced geometry; bf16 cached sampling can differ
numerically. The exact identity controls refer to this evaluation geometry.
The positive-caption base model is a behavioral reference, not a quality
oracle. Closer imitation of its token distribution need not be preferred music.

## Initial results: the completed 30–300 sweep

| Checkpoint | First-token KL on positive histories | Continuation KL on neutral histories | Continuation KL on positive histories |
| --- | ---: | ---: | ---: |
| Original 120-update smoke | 0.03904 | 0.01196 | 0.00915 |
| 60 updates | 0.05307 | 0.01198 | 0.00905 |
| 120 updates (repeat) | 0.03167 | 0.01163 | 0.00896 |
| 150 updates | 0.02836 | 0.01142 | 0.00891 |
| 180 updates | 0.03167 | 0.01218 | 0.00935 |
| 300 updates | 0.03915 | 0.01271 | 0.01051 |

Lower means closer to the positive-caption teacher, not better sounding.
The continuation minimum among the measured checkpoints is 150, rather than
the user-liked 300. Reverse KL and JS show the same ordering for 120 versus
300. No temperature, vocabulary subset or weighting was tuned to make 300 win.

For comparison, the unchanged neutral model has continuation KL 0.01895 on
neutral histories and 0.01323 on positive histories. All these adapters move
closer to the positive-caption policy than the slider-off control. That is
evidence of a conditioning change, not an audio-quality ranking.

All **88/88** zero-scale policy-logit comparisons are bitwise identical to the
pristine baseline. Analytic tests check KL direction, identity, invariance to
constant logit shifts, JS symmetry/bounds, and per-position normalization.

The broader reason to keep the interpretation narrow is that matching a
reference distribution and producing preferred samples are different objectives.
[A note on the evaluation of generative models](https://arxiv.org/abs/1511.01844)
demonstrates why generative likelihood and sample fidelity should not be
treated as interchangeable. This project's listening feedback supplies the
specific counterexample to the proposed checkpoint ranking.

## Completed 600-update experiment

A fresh continuous run retained the smoke recipe through 600 updates and
saved inference weights at 150/300/450/600. Manual comparisons use the old
preferred 300-update weights alongside the new 300/450/600 checkpoints, with
fixed generation seeds 7/23. The extra new 300 checkpoint controls for repeat
variation before attributing an audio change to extra updates.

| Checkpoint | Continuation KL on neutral histories | Continuation KL on positive histories |
| --- | ---: | ---: |
| Previous preferred 300 | 0.012706 | 0.010510 |
| New run: 300 | 0.011486 | 0.009401 |
| New run: 450 | 0.011637 | 0.009214 |
| New run: 600 | 0.012037 | 0.010007 |

These use the same saved reference histories as the initial sweep. The old
300 checkpoint reproduces its previous per-position results exactly, and all
**32/32** new zero-scale policy comparisons are exact. The different new/old
300 results show that the repeated run is not numerically identical. The
continuation minimum also depends on which reference histories are used;
neither picks a listening winner.

The [side-by-side listening page](../../eval/listen/gan-bcap-steps600-20260904/index.html)
contains 20-second samples from one prompt row, two generation seeds and four
checkpoints, plus slider-off and positive-caption references. The old 300 audio
is reused byte for byte. There are no seed retries. This is a first manual
comparison, not a broad prompt or training-seed evaluation. Subsequent user
feedback reports stronger feminine voice character at 600 than 450, while
remaining unsure whether overall music is improving or just changing. This
establishes a perceived change in the intended direction, not an overall
quality winner or an optimal update count. Both continuation KL columns rise
from 450 to 600 despite that perceived increase in voice femininity.

For the next listening pass, record pairwise preference within each generation
seed and note overall music, intended voice change, lyric clarity and artifacts
separately. If later steps keep improving, extend the saved game; if they become
worse, compare more closely spaced snapshots around the last preferred region.
Confirm the apparent winner on additional prompts and seeds before adopting a
default update count. KL and hidden error remain diagnostics throughout.

The original run did not save its critic or optimizer state, so it cannot be
continued exactly from the inference adapter. The trainer now has opt-in full
GAN-state saving and continuation. A constant-LR four-update tiny-model run
matches a two-update run resumed for two more updates, including complete logs
and final weights, in both cached-frame and empty-frame cases. Changed LR is
rejected. The focused training checks record 31 passes, and the policy-metric
checks record two passes.

`SAVE_STATE=1` retains a local `*_state.pt` containing LoRA, critics, optimizers,
training history and Python/Torch RNG states. The latest state is atomically
replaced at snapshot intervals and at completion. These are local research
states; inference safetensors remain LoRA-only. To continue after 600, use a
fresh run name and `RESUME_STATE` pointing to that state; `STEPS` specifies the
total budget. Constant LR is required, and training settings, prompts and
source fingerprints must match. This does not claim GPU bitwise reproducibility
across different software or hardware.

The real run completed all 600 updates with finite training logs. The retained
300, 450 and 600 full states have LoRA tensors exactly matching their respective
inference weights. The final render and KL jobs completed successfully on GPU 1.

Artifacts:

- [Combined KL curves through 600 updates](policy_comparison_20260904.png)
- [Combined KL table](policy_comparison_20260904.tsv)
- [30–300 per-position KL, diagnostics and provenance](policy_steps300_20260904.json)
- [300/450/600 per-position KL, diagnostics and provenance](policy_steps600_20260904.json)
- [New manual comparison](../../eval/listen/gan-bcap-steps600-20260904/index.html)
- [Original manual comparison](../../eval/listen/gan-bcap-steps-20260904/index.html)
