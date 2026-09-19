# YuE2 particle-to-LoRA distillation

The user requested an attempt to distill all 16 released particle sliders into
ordinary LoRAs, using both GPUs. These experimental candidates preserve the
original particle checkpoints and do not replace the Space or studio defaults.

[Live measurements, downloads and listening comparisons](http://100.90.104.57:8888/yue2-lora-distill-20260917/)

[Hugging Face ordinary-LoRA option](https://huggingface.co/ntc-ai/yue2-concept-sliders#choose-your-adapter)
is published at `72131f4f954db1e2d809a52bb84d6a79aa9e5f5f`, under
`distilled-rank8-v1/`: all 16 native and ComfyUI exports, 128 FLAC/MP3 comparisons,
the standard workflow, loader, method and validation records. The existing
converter CLI reproduced every converted checkpoint byte for byte. All 481
uploaded file hashes passed verification, as did real model-card audio playback.
Publication evidence is in `releases/yue2-concept-sliders/verification/distilled-hf-20260917/`.

## What is being removed

The teacher projection adds

$$\Delta_T(x)=U f(Vx,P).$$

Here the learned cloud $P$ and routed nonlinear bridge $f$ are the existing
particle implementation. The distilled student adds only

$$\Delta_S(x)=U A x.$$

Its saved tensors contain ordinary `lora_down`, `lora_up`, and `alpha` entries.
There is no router, bridge, particle cloud, custom ComfyUI node, or inference
MLP in the student. Every native projection has rank 8 and alpha 8.

## First pass: projection regression

Keep each teacher up matrix $U$. On real AR activations, find a linear down
matrix $A$ that approximates the teacher's eight output features $f(Vx,P)$.
Put input vectors $x_i^T$ in the rows of $X$ and corresponding teacher features
$f(Vx_i,P)^T$ in the rows of $Y$. Let
$D_{ii}=\sqrt{(X^T X)_{ii}}$ and $Z=XD^{-1}$. Solve

$$B=(Z^T Z+\lambda I)^{-1}Z^T Y,\qquad A=B^T D^{-1}.$$

The implementation floors the squared diagonal at $10^{-12}$ for numerical
stability. This is regression through the origin: it does not add a bias to
the LoRA. The teacher is nonlinear and has biases, so an exact match on all
possible inputs is not promised.

Each slider uses neutral captions from the audited training catalog, excluding
the lyric sheet reserved for validation, plus teacher-generated continuations
on its first three training rows. Each continuation has a 384-token guard and
a fixed seed from 4100–4102. Activations are collected with the teacher at
strength 0 and 1. Q/K/V share their input covariance; the O projection has its
own covariance. No held-out evaluation lyric sheet is used for fitting.

Choose ridge strength from 0.0001, 0.001 and 0.01 using the fourth training
row's continuation, seed 4103. Model selection uses the mean normalized error
at strengths 0.5 and 1, both at the first music-token boundary and throughout
the music continuation.

## Second pass: full-model distillation

Starting from the selected regression, optimize both ordinary LoRA matrices
for 100 Adam steps, learning rate 0.0001, gradient norm cap 1. The teacher and
base model stay frozen. Alternate continuation examples and broader training
captions, sampling strengths 0.5 and 1 with a fixed RNG seed. The loss is

$$L=\frac12\sum_{r\in\{boundary,music\}}
\frac{\mathrm{MSE}(h_S^{(r)},h_T^{(r)})}
{\max(\mathrm{MSE}(h_T^{(r)},h_0^{(r)}),10^{-7})}.$$

Evaluate the reserved training row every ten updates. Save the best checkpoint,
including step zero as a candidate. Refinement is discarded when it does not
improve that validation score. Test-set measurements never choose the winner.

## Evaluation and its limits

The existing two held-out prompts, seed 1709, supply both base-generated and
teacher-generated token trajectories. At strengths 0.5 and 1, measure:

$$E=\frac{\|h_S-h_T\|^2}{\|h_T-h_0\|^2},\qquad
C=\frac{(h_S-h_0)\cdot(h_T-h_0)}{\|h_S-h_0\|\|h_T-h_0\|}.$$

Error zero is an exact teacher match; error one is what leaving the slider off
would produce. Cosine one means aligned steering. Report the first-token
boundary separately from the music sequence. Also report teacher-to-student
and teacher-to-base KL over the codec vocabulary at every 32nd music position,
before sampling penalties and classifier-free guidance.

These metrics concern teacher fidelity, not perceptual quality. Small teacher
effects can make BF16 rounding a substantial part of the normalized error.
Autoregressive sampling can diverge despite small distribution errors. The
experiment does not establish full-song quality, mixed-slider behavior,
negative strengths, or strengths above one.

Render two fresh matched groups per slider: Off, original particles, plain
LoRA during AR only, and the same plain semantic sequence with its LoRA also
active during acoustic-prefix conditioning. Every clip uses seed 1709, a
500-token / approximately 20-second diagnostic guard and 16 acoustic ODE steps.
Check finite stereo 48 kHz audio and nonzero signal. The acoustic-prefix test
mirrors standard ComfyUI scope in the native runtime; it is not a full ComfyUI
audio render or a blind listening evaluation.

## ComfyUI

Use the **standard Load LoRA node**. Connect MODEL and CLIP from the YuE2
checkpoint loader. Set model strength **0**, CLIP strength **1**; reduce CLIP
strength toward zero to reduce the effect. Use `off` planning mode for the
same semantic-only setup used in this experiment. The workflow keeps the
native 360-second upper guard; it is not the 20-second diagnostic setting.

Place a `*_comfyui.safetensors` download in `ComfyUI/models/loras/` and load
`workflow.json`. The ordinary LoRA also changes the acoustic-prefix conditioning
computed by YuE2's text encoder. The fourth player makes that difference
reviewable. The acoustic denoiser and VAE receive no LoRA tensors.

Conversion extends the existing `mikkel/conceptmod` converter, published in
[a5c3dd8](https://github.com/mikkel/conceptmod/commit/a5c3dd8), from an isolated
worktree at `/ml2/music/.cache/conceptmod-yue2-lora`. There is no second
converter in this repository. Q/K/V are fused by concatenating the down
matrices and block-diagonalizing the up matrices, preserving alpha/rank.
Their fused rank is 24; O stays rank 8. This is an exact rearrangement of the
linear updates before BF16 export rounding.

The native regression recovery test passes. Converter tests include unequal
Q/K/V output widths, unequal alpha scales, rejection of nonlinear particle
checkpoints and incomplete QKV groups, and application through the real
ComfyUI YuE2 LoRA loader. The relevant converter suite has 32 passing tests.

## Reproducibility

Sources: `scripts/distill_yue2_particles.py`,
`scripts/refine_yue2_distillation.py`, `scripts/report_yue2_distillation.py`.
Artifacts: `models/yue2-lora-distill-20260917/{control}/`.
Each selected checkpoint records its teacher hash, model identity, recipe,
hyperparameters and source hashes. Each control preserves its original ridge
fit, all refinement validation scores, final held-out measurements and clips.

GPU 0: female, pop, rnb, pop-punk, country, house, kpop, afrobeats.
GPU 1: male, hiphop, indie-rock, metal, acoustic-folk, disco-funk, reggaeton, lofi.
Both cards share with the existing studio. Base model and VAE revisions are
pinned by the published catalog; weights are read from the local HF cache.
The studio queue and original release remain available throughout.

Weights retain the original release's CC BY-NC 4.0 terms. The ComfyUI workflow
is derived from the existing workflow distributed with the particle release;
its original workflow license accompanies this comparison.

## Completion and GitHub mirror

All 16 distillations, 128 audio comparisons and 896 ComfyUI patch bindings
passed final verification. Both GPU workers exited successfully. The local
comparison page serves the complete ComfyUI and native download archives.

GitHub documentation and the standalone ordinary loader are published at
`75c55aaf278157ce96a68649fc841290fa5f01fc`. Large GitHub release assets are still
uploading. The user service `music-yue2-distilled-publish-20260917-v2.service`
waits for the current uploader, retries incomplete transfers, verifies every
server-side SHA256 against `artifact-manifest.json`, then publishes the
`distilled-rank8-20260917` prerelease. It leaves the release in draft on failure.
Current state is in `github-publication.json` here and on the listening page.
