# YuE2 female slider, 600 updates

This experiment ports the winning UNI16 **initial 600-update warm-up** to
YuE2's native AR attention path. The reference is
[the selected formulation](../../docs/hub-formulation-fresh-selected.md) and
`warmup_args` in `analysis/uni16_fresh3400_20260912/common.py`.

- Rank 8, alpha 8; zero-initialized LoRA-up matrices; 112 AR projections.
- Four matched neutral/female prompt pairs in every update, with identical lyrics.
- RpGAN + exact batch-mean critic feature matching + frozen-base end-margin MSE,
  each with coefficient 1; no direct hidden-state MSE or explicit lyric hold.
- Shared two-layer, width-128 transformer critic; fixed RMS calibration;
  input-gradient cap threshold/coefficient 1; mean/last readout.
- Generator AdamW LR 0.0005, critic Adam LR 0.00075; beta1 0, beta2 0.999;
  constant rates; generator gradient value clip 1; generator weight decay 1e-6.
- Fixed 250-token histories per row. The later fresh-history continuation and
  update-norm cap begin after step 600 in the reference campaign and are outside
  this experiment.

The model-specific changes are token packing, causal AR forwarding, and YuE2's
native music-end versus semantic-vocabulary margin. Training and rendering use
physical GPU 1. CPU diagnostics use cached CLAP and Audiobox Aesthetics models.

The four training lyric sheets and four evaluation sheets are disjoint. All
authored prompts pass the project's artist-name validation. `source-sha256.json`
pins the training source, shared objective helpers, prompts and reference recipe.

Run directory: `models/female-yue2-uni16-600-20260916/`.
Final export: `female-yue2-uni16-600_600.safetensors`.
The full `_state.pt` retains the critic, both optimizers, prompt teachers,
base histories and RNG state. `verification.json` records tensor-by-tensor
export/full-state equality and render-manifest checks.

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME=/ml2/music/.cache/huggingface HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /ml2/music/.cache/yue2-test-env/bin/python conceptmod/textsliders/train_lora_yue2.py \
  --config_file conceptmod/textsliders/data/config-yue2-female-uni16.yaml \
  --save_dir models/female-yue2-uni16-600-20260916
```

`render.py` makes ten 750-token previews: two held-out arrangements, seeds 7/23,
off/on at identical neutral prompts, and an explicit female-caption reference
for each arrangement at seed 7. Rendering records truncation; these are excerpts,
not complete songs. The native acoustic model and VAE remain unadapted.
The listening page lives in `eval/listen/yue2-female-uni16-600-20260916/`.

`measure.py` compares cached, fixed audio-model diagnostics on measurement copies
normalized to stereo RMS 0.1. It preserves original audio. Female-minus-male CLAP
similarity and aesthetics scores are proxies, not calibrated judgments of vocal
gender or audio quality. No ASR or human listening result is claimed here.

Validation before the live run: 36 tests passed across the YuE2 integration,
UNI16 port, shared feature-matching loss and transformer critic tests. They
cover native end-margin values/gradients, strict checkpoint loading, exact off
behavior, frozen base weights, and exact full-state resume.

## Completed result

All 600 updates finished on GPU 1 in **900.86 seconds** (15.0 minutes, excluding
model loading and frozen-teacher preparation). The 13 MB export has SHA-256
`81a6b52f22140ac4d62f60dcccd17496865cd887b5acacd268a9378d21c36190`.
All 336 tensors are finite and exactly match the final full training state.
Training source hashes, prompt-name validation and all render manifests pass.

The adversarial game became less well aligned after about step 340. At step 600,
the mean last-token target cosine was 0.245 and the delta magnitude ratio was
5.24. These are training-state diagnostics, not measurements of audible quality.
The requested recipe and step budget were kept unchanged.

All ten previews are finite, non-silent 48 kHz stereo, about 30 seconds long,
and explicitly marked as truncated at the semantic-token cap. All four off/on
pairs differ in semantic tokens and waveform. The page and every WAV URL returned
HTTP 200 on the existing listening server.

Matched on-minus-off diagnostic changes:

| Arrangement | Seed | Female CLAP margin | Enjoyment proxy | Production proxy |
|---|---:|---:|---:|---:|
| Small band | 7 | -0.0653 | +0.1240 | -0.0750 |
| Small band | 23 | +0.0864 | +0.1592 | +0.1030 |
| Upright piano | 7 | +0.1150 | +0.1886 | +0.2150 |
| Upright piano | 23 | +0.1203 | -0.1521 | -0.3939 |

The female-vocal proxy improved in **3/4** pairs, with a mean change of +0.0641.
Quality-proxy changes are mixed. This is an experimental listening candidate;
there is no human listening or lyric-preservation verdict. Full values are in
`diagnostic-summary.json` and the listening page's `audio-diagnostics.json`.

[Listen to all ten previews](http://100.90.104.57:8888/yue2-female-uni16-600-20260916/)
