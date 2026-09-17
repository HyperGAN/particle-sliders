# YuE2: current 16-slider catalog to 1200

The user requested all current MiniMax slider concepts on YuE2, 1200 updates
each, training on both GPUs. Both cards share with the existing studio; the
studio queue and registry are preserved.

Live queue, training curves, and final listening grids:
<http://100.90.104.57:8888/yue2-particle-uni16-1200-20260917/>

## Fixed recipe and budget

Every job uses the existing `--recipe particle_bridge`,
`anneal-routed-particle-error-yue2-v1`, with training sources identical to the
accepted metal source `2067705`. No trainer/loss/optimizer code changes are
part of this queue. G / D / particle learning rates remain 0.0006 / 0.0009 /
0.006, constant. Seed 7, 128×4 routed particles, particle VIC weight 1,
paired-error Gaussian-noise critic, lazy b_cap every fourth update, EMA 0.995.
The noise horizon stays 8000; it is not compressed to the new stopping budget.

Metal resumes its complete live adapter, cloud, critic, optimizers, EMA,
sampler and RNG from 600 to **1200 total**, preserving its original four
prompt pairs. Its 600-step run and listening grid remain available. The other
15 start fresh; historical UNI16 checkpoints are a different recipe.
This queues 18,600 additional updates, with 19,200 total across the final set.

| GPU 0, in order | GPU 1, in order |
|---|---|
| Female | Metal (600 → 1200) |
| Male | Pop |
| Hip-Hop | R&B |
| Indie Rock | Pop Punk |
| Country | Acoustic Folk |
| House | Disco Funk |
| K-pop | Reggaeton |
| Afrobeats | Lo-fi |

Each card runs one campaign at a time. GPU 1 first waits for the existing
metal-600 renderer to finish, identified by PID and process start time, so
resuming cannot overwrite the renderer's source export midway through its
grid. The workers cooperate with existing `.music-gpu-N.lock` leases and
wait for at least 10,000 MiB free before loading. They do not stop the studio.

## Prompt transfer and evaluation

The registry snapshot SHA, MiniMax sidecar provenance, exact argv for every
job, source/prompt hashes, context lengths and selected source rows are in
`manifest.json` (also published as `commands.json`). All 16 train/eval files
passed the application's name check, YuE2 validation, disjoint-lyric check,
and CPU tokenizer preflight. Maximum training prefix length is 246 tokens,
well within the unchanged 1024-token trainer limit.

For the 15 fresh sliders, neutral/positive captions and lyrics are copied
verbatim from the four prompt pairs used by the current MiniMax weights.
Only unused legacy YAML fields are removed. Evaluation uses MiniMax's held-out
rows **2/3** and seeds **1709/2903**. Metal retains its approved native prompts
and two held-out rows to preserve exact resume and the earlier comparison.

After each job reaches 1200, the existing campaign renders 20 clips: two rows
× two seeds × 0/0.5/1, positive-caption reference, and unscored -1 canary.
Final listening weights are EMA; `_live_*` exports preserve raw weights.
Saved states and exports remain available every 100 updates. The queue does
not automatically promote a checkpoint or change the studio's MiniMax catalog.
Slider pages now use the actual concept label instead of calling every slider
Metal. Hidden caption geometry remains diagnostic, not an audio-quality gate.

## Persistence and failure behavior

The two user services call `scripts/queue_yue2_particle_catalog.py worker`,
which invokes the existing `train_yue2_arm_b_campaign.py`. This is scheduling
only; there is no additional training implementation. Both workers and the
page publisher have exclusive locks. A child campaign inherits the worker/GPU
leases so a worker crash cannot launch a second model over an orphaned child.

A finished job is skipped on restart. An interrupted job resumes exact state.
A failed job retains its logs/state and is marked failed; other sliders
continue. Failed jobs are not silently retried. A zero process exit after
interruption is not considered success: the worker checks both the 1200-update
budget and successful render completion. Service failure restarts the worker;
intentional shutdown requests a graceful checkpoint.

```bash
systemctl --user status music-yue2-particle-1200@0.service \
  music-yue2-particle-1200@1.service music-yue2-particle-1200-page.service
journalctl --user -u music-yue2-particle-1200@0.service -n 30 --no-pager
```

Source and prompts live in the frozen checkout
`/ml2/music/.cache/sliders-yue2-uni16-1200-20260917`.
Do not run these queued jobs from a changing checkout. Workers verify pinned
files before each campaign. Runtime job/worker JSON and logs are ignored by git.

Queue preparation command (CPU only; does not launch training):

```bash
CUDA_VISIBLE_DEVICES='' HF_HOME=/ml2/music/.cache/huggingface HF_HUB_OFFLINE=1 \
  /ml2/music/.cache/yue2-test-env/bin/python scripts/queue_yue2_particle_catalog.py prepare \
  --queue_dir analysis/yue2_uni16_1200_20260917 \
  --models_root /ml2/music/sliders-conceptmod/models/yue2-particle-uni16-1200-20260917 \
  --output_root /ml2/music/sliders-conceptmod/eval/listen/yue2-particle-uni16-1200-20260917 \
  --metal_run /ml2/music/sliders-conceptmod/models/metal-yue2-particle-bridge-s7-20260917 \
  --wait_gpu1_pid 375268
```

The published dashboard copies `scripts/assets/yue2-catalog-dashboard.html`.
The worker and publisher unit templates here pin the checked-out source paths.

Validation: 29 selected queue/display/native-recipe checks passed, followed by
the additional worker lifecycle test (device isolation, failure continuation,
and restart skipping), bringing the relevant passing checks to 30. The five
queue tests run without allocating a GPU. Dashboard JavaScript syntax checked
with `node --check`. The original recipe's CPU gate proof and deployment checks
remain in [the particle-bridge audit](../../docs/yue2-particle-bridge.md).
