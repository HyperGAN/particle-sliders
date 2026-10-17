# Bonsai read-only diagnosis: unstable fit and an almost constant edit

This is a saved-checkpoint diagnosis, not further training. The final script ran
in 1.47s on one CPU thread with CUDA hidden, gradients disabled and optimizer
construction explicitly forbidden. It loaded the existing native head/critic
checkpoints and cached readouts only. All 76 opened artifact/cache files and all 96
source/metadata pins matched before/after. No game/optimizer, GGUF/server, model
download, GPU launch or held-out tuning occurred.

Script: `scripts/diagnose_bonsai_pg09_readonly.py`, SHA256
`18709d3d31338f2bfe893ad6c5d43ebfe1e0484790f576359e18953b4459d5e3`.
Exact output: `evidence/pg09-text-video/bonsai-readonly-diagnosis.json`.
The loaded final head reproduces saved held-out zero/+1 vectors bit for bit.

## Fit versus generalization

At strength +1, final training mean relative-edit MSE is **1.225932**, with 2/3
cases satisfying the per-case edit criterion. Held-out mean is **1.943358**, 0/3.
The no-edit baseline is 1. Training fit is therefore incomplete at the final
checkpoint; this is not solely a cleanly learned mapping failing to generalize.

| Case | Training relative-edit MSE | Training cosine | Held-out relative-edit MSE |
|---|---:|---:|---:|
|Garden|1.964946|0.511718|2.108203|
|Sea|0.895702|0.672162|1.944979|
|Street|0.817148|0.559320|1.776892|

Training did learn something: the saved step 900 checkpoint has training mean
0.702492 and 3/3 per-case successes. It then worsens again. This is a training-only
diagnostic, not checkpoint selection; no new held-out evaluation was performed
at 900. The existing fixed evaluation checkpoints remain the held-out evidence.
The record supports partial learning, unstable amplitude/fit, and worse transfer.

## Sign, strength and conditioning

There is no observed sign reversal. Final training cosines are positive; at -1
they reverse and training mean error rises to 4.147493. Strength 0 is exact Off;
0/.5/1 follow the native linear strength rule to ≤2.98e-8 absolute error. The
trained +1 edit magnitudes are 1.619/1.262/0.920 times the three teacher edit norms.
At the already-declared 0.5 diagnostic strength, training mean error is 0.691288.
This shows amplitude miscalibration; it is not a selected deployment strength or
a held-out-based tuning recommendation.

The stronger finding is a nearly input-independent edit: the three generated
training edit directions have pairwise cosines **0.999994–0.999999**, whereas the
three frozen teacher edit directions have cosines **0.209–0.314**. Cross-target
errors are nearly unchanged when the generated row changes. This resembles a
common offset rather than three well-separated conditional edits.

The critic is not literally disconnected from context, but its effective row
sensitivity is weak in this probe. With 256 fixed common Gaussian draws, cycling
only the three training contexts changes its fake-minus-real gap by mean absolute
**1.49e-6–1.03e-5**. The matched gaps are roughly -0.000123 to 0.001075, against
real-logit standard deviation ≈0.01125. Full context weights are nonzero, so this
is not proof of a missing tensor connection. Unit-norm neutral inputs and
teacher-edit norms 0.302/0.388/0.532 also sit beside a sigma 1 reference whose sampled
L2 norm averages 71.61. Those scales are consistent with weak conditional signal,
but the norm comparison alone does not establish the cause of collapse.

## Defect assessment and smallest next test

No concrete wrong-row, wrong-input, sign, strength-factor or checkpoint-loading
defect was found. Source inspection shows neutral-student/positive-target pairing
and shared row indices for target/context; independent forward reproduction
confirms the checkpoint mapping. Existing connectivity tests establish a graph
path, not that a model uses context effectively. The missing training-fit and
conditioning-sensitivity measurements were a qualification/instrumentation gap.
The evidence does not establish a bug in PG's public GAN formula or identify one
optimizer/architecture choice as the cause.

The smallest justified next test is **zero-update, training-only bias/input
ablation on the saved final head**: compare each neutral-input edit with its
zero-input edit and record route distributions. This can localize the common
offset to bias/shared-bank versus input-dependent paths before proposing any new
training arm. It requires no held-outs, optimizer or additional training budget.
It has not been executed as part of this frozen diagnosis.

The archived run fed positive students; this corrected run feeds neutral students
and also changes the public optimizer/objective/schedule. Preserve that confound:
these observations do not isolate GAN-versus-MSE causality. Tiny/LTX metadata and
software preparations remain available, with pretrained parity/shape/memory/render
and coordinator model/GPU admission gates held; H3 native AV/license/hardware
blockers remain unchanged. No GPU launch or new training is requested here.

Reproduce only the read-only diagnosis:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
timeout 120 /mnt/nvme/sliders/qualification/envs/pg090/bin/python \
  scripts/diagnose_bonsai_pg09_readonly.py --code "$PWD" \
  --run /mnt/nvme/sliders/qualification/runs/bonsai-pg09-gan-1200-20261017-v2 \
  --output /tmp/bonsai-readonly-diagnosis.json
```
