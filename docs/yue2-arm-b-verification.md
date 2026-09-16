# YuE2 unipolar GAN formulation verification

The user requires unipolar, GAN-only training. The current recipe is
`unipolar-rpgan-bcap-yue2-v3`. It keeps Arm B's paired logistic game and
vendored discriminator gradient cap; it does not claim parity with the full
bipolar Music Arm B objective.

## Exact objective

For each of four prompt rows, let `real = h_positive - h_neutral` and
`fake = h_student(+1, neutral_prompt) - h_neutral`. The positive teacher is
raw `h_positive`, following the shared `lm_faithful_plus_neu` function.
The critic divides inputs by a fixed RMS calibrated from all real deltas.

```text
L_D = mean softplus(D(fake) - D(real))
    + 0.5 * (mean relu(||grad D_core(real / RMS)||_2 - 1)^2
           + mean relu(||grad D_core(fake / RMS)||_2 - 1)^2)

L_G = mean softplus(stopgrad(D(real)) - D(fake))
```

D updates first, using detached fake vectors. G then updates at +1 with D
frozen. Averages are over the four rows. There is no factor 0.5 on G, no
negative branch, and no positive MSE, ending, cover, FM, lyric hold, plan,
VICReg, or zero-anchor loss. Weight decay and clipping retain their documented
optimizer settings; no new optimizer terms are introduced.

## Checks against the earlier mistakes

- The old bipolar run stopped at update 142 and is preserved. New training
  starts from the base model with a different recipe identity.
- Prompt loading rejects negative captions, bipolar ranges, and leak-axis
  metadata. No opposite teacher is encoded or silently ignored.
- Scale 0 is the base model by adapter construction. It needs no learned
  neutral anchor. Only scale +1 enters the discriminator and generator phases.
- `rp_g_loss` uses the toy helper's `(real, fake)` argument order.
- The vendored cap receives scaled vectors and `critic.net`, preserving its
  gradient units without scaling twice. Zero fake vectors remain included.
- The cap is exact autograd on every update, threshold/coefficient 1, no
  annealing or finite differences.
- Generator backward retains +1 during checkpoint recomputation. Base weights
  stay frozen; D parameters have no gradients after its own update.
- Training uses prompt states only. It never calls audio sampling or computes
  ending margins. Shuffled prompt batches and the RNG state resume exactly.

## Executable evidence

`tests/test_yue2_arm_b.py` compares two full D/G updates with independent,
full-batch equations. It covers active/inactive gradient caps, non-unit teacher
scale, losses, generator gradients, and updated G/D parameters. The reference
G objective contains only paired logistic loss; adding any MSE or end term
would fail numeric parity. The fake adapter rejects negative scales.

Native tiny-model tests verify raw-positive targets, checkpointed/eager
backward agreement, frozen base weights, and exact scale-zero output. Recovery
tests compare uninterrupted and split runs including both optimizers, prompt
sampling and RNG, while audio sampling is patched to fail if called.
Bipolar prompts and incompatible resumes are rejected.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_yue2_arm_b.py \
  tests/test_music_arm_b.py tests/test_music_arm_b_gates.py \
  tests/test_yue2_slider.py tests/test_yue2_uni.py tests/test_yue2_fresh.py
```

A live GPU preflight checks finite updates, export/state agreement, and native
0/0.5/1 rendering. These tests verify implementation, not audible quality.
