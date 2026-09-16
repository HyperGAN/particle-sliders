# Unipolar GAN + neutral — opt-in trial

`rpgan_bcap_plus_neu` passes the CPU **unipolar** leaderboard's original
gates at 400 updates. It is a GAN-only candidate; the existing
`faithful_plus_neu` row remains a supervised MSE baseline.

| Seed 0, 400 updates | Cover >= 0.85 | Off-caption <= 0.05 | Neutral hold >= 0.85 | Result |
|---|---:|---:|---:|---|
| divergent | 0.928790 | 0.000000 | 0.993213 | HIT |
| close | 0.926515 | 0.000000 | 0.982018 | HIT |

The same gates pass for seeds 0, 1, and 7 in `tests/test_unipolar_gan.py`.
The student retains the leaderboard's **trainable origin**: neutral hold is
learned, not forced by removing that parameter. Scores at 0 and 1 decide
acceptance. Half-scale is diagnostic; -1 and antipodal cosine never gate UNI.
The same weights fail the bipolar continuation exam, as expected for a
model that never trains the negative endpoint. Both board outputs include
this split; existing bipolar recipes and their defaults are unchanged.

## Exact objective

For endpoint `s` in `{0, 1}`, real is the raw caption hidden delta relative
to neutral: zero at 0, `positive - neutral` at 1. Fake is the student delta.
The critic receives the endpoint label along with that delta.

```text
D = mean_s [ mean_rows softplus(D(fake, s) - D(real, s)) + b_cap_s ]
G = mean_s mean_rows softplus(D(real, s) - D(fake, s))
b_cap_s = 0.5 * (mean_real relu(norm(grad_hidden D) - 1)^2
              + mean_fake relu(norm(grad_hidden D) - 1)^2)
```

The cap differentiates hidden coordinates, **not the fixed scale label**.
Its coefficient and threshold are 1, with exact autograd every D update.
One D update precedes one G update; G uses the updated D. The critic is a
two-layer, 256-wide LeakyReLU MLP, calibrated once with positive-teacher RMS.

There is no MSE, ending loss, cover loss, feature matching, lyric hold,
particle regularizer, EMA, gradient clipping, or negative-endpoint training.
Adam uses betas (0, 0.99), equal G/D rates, and the existing delayed cosine
schedule (80-update hold, final ratio 0.05). The loss does not change phases.

## Implementation and transfer checks

The CPU fixture and YuE2 adapter both call
`conceptmod/textsliders/unipolar_gan.py`; the trainer does not reimplement the
game. Tests compare the full D/G update with independent equations, including
active and inactive cap penalties, and compare native eager/checkpointed
gradients. Resume restores the exact model/critic/optimizer/sampler/RNG state;
the schedule horizon is pinned. Toy tests forbid MSE calls during GAN fitting.

The toy's direct residual uses LR **0.005**. Native rank-8/alpha-8 attention
LoRA uses **0.0005**: these parameterizations have different units. This is an
explicit transfer setting, not a claim that the native 600-update trajectory
has already been validated by a 400-update toy. Native scale zero is exactly
the base model, so its G term is the constant log(2) with zero adapter
gradient; the cached zero delta avoids an unused native forward.

The new arm is propose-only on the board (`MERGE_TO_TRAINER=False`); it does
not promote itself into trainer defaults. The user separately requested the
opt-in YuE2 metal trial below. Music `ARM_B`, live `--lm_target v9`, and the
locked bipolar `AdvConfig()` remain unchanged.

## Reproduce and run the requested metal trial

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python analysis/slider2d/run_formulation_leaderboard.py \
  --polarity uni --out docs/formulation-leaderboard
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python analysis/slider2d/run_formulation_leaderboard.py \
  --polarity both --out docs/formulation-leaderboard
PYTHONPATH=. pytest tests/test_formulation_leaderboard.py \
  tests/test_lm_plus_neu_exam.py tests/test_unipolar_gan.py -q
```

Fresh run only; the older plus-only GAN state is incompatible:

```bash
python scripts/train_yue2_arm_b_campaign.py --recipe gan_plus_neu \
  --name metal-yue2-gan-plus-neu --steps 600 --gpu 1 \
  --save_dir models/metal-yue2-gan-plus-neu-600-20260916 \
  --output_dir eval/listen/yue2-metal-gan-plus-neu-600-20260916
```

The four sound-only training rows are shuffled in balanced passes. There is
no generated-history sampling in this prompt-state objective. Evaluation uses
two held-out prompts and seeds 1709/2903, with 0/0.5/1 and the positive caption
reference. Native audio quality remains a separate listening check.
