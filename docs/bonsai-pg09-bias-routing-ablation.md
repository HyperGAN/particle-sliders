# Bonsai common-edit ablation: bias-carried offset, almost shared routing

The saved final head carries its common edit mainly through the bridge decoder's
bias paths. Input-specific routing contributes almost no differentiation between
the three training rows. This is a forward-path attribution on a saved model,
not proof of what caused that model to learn the behavior.

The final parent-authorized analysis ran in **1.121 seconds**, one CPU thread,
`no_grad`, with optimizer construction forbidden and CUDA hidden. Eleven
predeclared ablations operated only on in-memory saved-head copies and the three
training pairs. **Zero held-out tensors were read, zero optimizer updates ran.**
All 7 opened checkpoint/cache tensors and all 96 source/metadata pins matched
before/after. Original head state, checkpoints and runtime configuration were
unchanged. The manual native-path decomposition matches the native forward
bit for bit before any ablation.

Script: `scripts/ablate_bonsai_pg09_readonly.py`, SHA256
`301ed3477a81cff357d72c7ac83457e608a373a085f6bbd8cd91f7993891ba8f`.
Full output: `evidence/pg09-text-video/bonsai-bias-routing-ablation.json`.
Source/data remain pinned to the completed v2 recipe and native model files.

The initial executed/QA-reviewed version (script SHA256 `0003c91d...`, output
SHA256 `5196be4c...`,1.164s) is also preserved as
`bonsai-bias-routing-initial-executed.py` and
`bonsai-bias-routing-initial-output.json` in the same evidence directory.
The final version adds explicit checkpoint loss=`non_saturating`, sigma=1 and
response_scale=1 assertions and fields; its bounded repeat returned identical
measurements. Only script/hash/runtime metadata differ. Both executed sources
and outputs are retained; neither constructed an optimizer or read held-out tensors.

Independent QA source/algebra/hash review **PASS** is frozen at
`27f5e1142fd0b355445429878ee8b02533cc1f6d`. It independently verified all 96
source and 7 training-artifact hashes, both preserved executions, and equality
of all common measurements. QA ran no forwards, optimizers or GPU actions.
The receipt is preserved as
`evidence/pg09-text-video/bonsai-bias-routing-independent-review.json`.
This qualification does not change the original held-out FAIL or establish
historical training causality.

## Bias and input dependence

The native three edit norms are approximately 0.489, with **99.99976%** of their
total squared magnitude in the common mean component.

| Fixed-head intervention | Edit L2 norm across training rows | Change versus native edit |
|---|---:|---:|
|Native|0.488917–0.489124|0|
|Zero input|0.486686|5.60–5.94%|
|Remove final bridge-decoder bias|0.213592–0.213973|107.88–107.92%|
|Remove all bridge-decoder biases|0.058196–0.060078|89.18–89.50%|
|Uniform particle routes|0.475142–0.475934|3.55–3.77%|
|Zero routed-bank readout z|0.466351–0.467736|6.35–6.69%|

Zero input retains 99.5% of the native edit norm, and only modestly changes its
vector. Removing all bridge-decoder biases shrinks the norm by about **88%**.
For the zero-input diagnostic, only the residual-head input is zeroed: the
counterfactual residual is applied to the original neutral base when computing
training metrics. The teacher and neutral base are not replaced with zeros.
The final decoder bias alone maps through `up` to an additive vector of norm
0.527654, cosine 0.914284 with the mean edit, and projection coefficient 0.986446
along that mean. Its exact additive identity is verified within 3.73e-9.

These are **not additive percentage attributions**. The final bias vector is
larger than the net edit because other paths cancel part of it. Hidden-bias
ablations also change nonlinear activations; removing only hidden decoder biases
actually increases edit norm to about 0.579. Removing biases does not restore
conditional behavior: even the all-decoder-biases-zero edit directions remain
nearly parallel (minimum cosine 0.999906). Bias carries much of the common
magnitude; the learned input/decoder mapping is also poorly differentiated.

## Routing versus capacity

Route entropy is about 4.822 nats, near the uniform maximum `log(128)=4.852`.
More directly, the largest route-weight L1 distance between training contexts is
only **2.17e-4**. Replacing all row routes by their training mean changes the
output by at most **9.56e-6 relative**; cycling routes between the three rows
changes it by at most **1.70e-5 relative**. Thus the routed bank participates,
but the final input-specific routing is effectively shared in this probe.
Uniform routing and zero-z interventions show a modest shared bank contribution;
they do not demonstrate a disconnected or unused bank.

The learned `up` matrix is not rank-one collapsed: its 8 singular values are
approximately 5.128, 4.016, 2.162, 1.593, 1.350, 1.125, 0.969 and 0.743; the
leading squared singular value accounts for only **48.5%** of their sum.
The three teacher edits span rank 3, versus nominal rank 8. No hard rank-capacity
limit is demonstrated. However, algebraic projection onto this *learned* frozen
output subspace leaves 59.3%, 37.5% and 47.1% of teacher-edit energy outside it.
That is a learned representation mismatch, not evidence that the architecture
could never learn the three targets. Projection is an algebraic diagnostic;
no new parameters or fitted head were constructed.

## What the final critic rewards

With the same 256 Gaussian draws and matched original training contexts, the
frozen critic/public non-saturating G formula gives:

| Counterfactual response | Mean adversarial G loss (lower is preferred by this frozen critic) |
|---|---:|
|Native learned offset|0.693681|
|Exact teacher / zero residual oracle|0.693968|
|All decoder biases removed|0.694742|
|No edit|0.694838|

The native offset is slightly preferred over exact teacher matching in this
snapshot (mean paired difference -2.8765e-4). For garden and sea the rowwise
paired differences are -5.3778e-4 and -3.8599e-4, with sampling SEMs 1.72e-5 and
2.08e-5; street is near neutral (+6.08e-5, SEM 5.06e-5). The zero-residual oracle
is verified to produce exactly the real-reference logits. The prior is unchanged
in these output-path interventions; this comparison concerns the adversarial
signal only.

Removing all decoder biases reduces training distance error from 1.226 to 0.900
in this ablation but worsens the frozen critic score. That is consistent with
the current critic rewarding the shared offset rather than precise rowwise
targets. It does **not** establish historical training causality: a discriminator
snapshot can lag the generator, these are finite fixed-noise probes, and no
controlled retraining was performed. It is not evidence of a public GAN-formula
implementation error. The oracle is a mathematical reference, not a replacement
model; no ablation is selected for deployment or a new training arm.

## Supported conclusion and limits

The common edit is predominantly **bias-carried**, with nearly shared routing
and weak learned input differentiation. The frozen critic signal is compatible
with maintaining that behavior. Nominal capacity alone does not explain it, and
no sign, pairing, scale or checkpoint defect has been established. Bias removal
alone is not demonstrated to fix conditional learning or transfer.

The original held-out FAIL remains unchanged and was not used to select arms.
The corrected neutral-student input and public recipe both differ from the
archived positive-student run, so preserve the causal confound. No training
extension, restart, tuning, full-model/GGUF/server/GPU execution or new runtime
configuration was performed or requested. Core owner and QA were notified;
Tiny/LTX live gates and H3 blockers remain separate.

Exact read-only command, from the isolated code checkout:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
timeout 120 /mnt/nvme/sliders/qualification/envs/pg090/bin/python \
  scripts/ablate_bonsai_pg09_readonly.py --code "$PWD" \
  --run /mnt/nvme/sliders/qualification/runs/bonsai-pg09-gan-1200-20261017-v2 \
  --output /tmp/bonsai-bias-routing-ablation.json
```
