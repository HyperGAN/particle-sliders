# Bounded native audio validation — preparation and admission gates

Audio ownership continues through the continuous-domain fix and real-model
validation. The earlier fail-closed adapter milestone is software groundwork.
No full-model job has run and no GPU lease has been granted to this task.

## Existing offline assets and runtimes

`scripts/inspect_audio_pg09_assets.py` completed on both native stacks, loading
only configs and tokenizers. It verifies config hashes and staged asset byte
sizes, records existing weight hashes without re-reading the weight tensors,
and records exact executable/module origins plus all 49 ParticleGAN source
hashes. Neither invocation initialized CUDA or loaded model weights.

| Family | Interpreter | Actual Torch / Transformers | Staged native response |
|---|---|---|---|
| YuE2 | `/ml2/sliders/.venv-yue2-retrain/bin/python` | `2.10.0+cu128` / `4.57.6` | `[1,2048]` |
| Music3 | `/mnt/nvme/sliders/qualification/envs/pg090/bin/python` | `2.11.0+cu128` / `5.15.0` | `[1,4096]` |

The committed earlier audio receipt also recorded Torch 2.11/Transformers 5.15.
An intermediate coordination message incorrectly described the newer runtime
as 2.12/5.13; the actual module origins in these receipts supersede that message.

YuE2's runtime and product both use the package name `yue2`. The preflight
imports the official installed runtime first, then adds the isolated product's
directory for its nonconflicting backend/slider/defaults/prompts submodules.
No installed package is edited. For this interpreter, only the PG0.9 package
and its metadata are read from the isolated PG environment's site-packages;
Torch and Transformers remain the native YuE2 versions. No ABI replacement
or global environment write is used.

The four fixed metal training rows and two fixed heldout rows fit the 512-token
cap without truncation: YuE2 maximum 180 tokens; Music3 maximum 167 tokens.
YuE2 can additionally use a fixed 16 semantic-token shared continuation history
and remain within that cap. Music3 preserves its last-real-token
`<|audio_start|>` boundary and does not substitute generated history for it.

## Reproduce CPU-only preflight

Run from the isolated audio checkout. These commands use the released
unchanged PG0.9 package for metadata inspection, not the pending overlay.

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /mnt/nvme/sliders/audio-adapters-pg09-20261017

PYTHONPATH=/mnt/nvme/sliders/qualification/envs/pg090/lib/python3.11/site-packages:. \
/ml2/sliders/.venv-yue2-retrain/bin/python scripts/inspect_audio_pg09_assets.py \
  --family yue2 --product /mnt/nvme/sliders/audio-yue2-source-20261017 \
  --model-dir /mnt/nvme/ai/caches/huggingface/music/hub/models--m-a-p--YuE2-3B/snapshots/14fc6c6f146441b1dd6363fcb2e01e82a6914cb7 \
  --manifest /mnt/nvme/sliders/qualification/models/yue2-cached/qualification-manifest.json \
  --train-prompts conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml \
  --heldout-prompts conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml \
  --max-seq-len 512 --output /tmp/audio-yue2-assets-preflight.json

PYTHONPATH=. /mnt/nvme/sliders/qualification/envs/pg090/bin/python \
  scripts/inspect_audio_pg09_assets.py --family music3 \
  --model-dir /mnt/nvme/sliders/qualification/models/music3-bd348f9 \
  --manifest /mnt/nvme/sliders/qualification/models/music3-bd348f9/qualification-manifest.json \
  --train-prompts conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml \
  --heldout-prompts conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml \
  --max-seq-len 512 --output /tmp/audio-music3-assets-preflight.json
```

## Proposed bounded stages after host-profile QA and GPU admission

These are proposed bounds, not an active reservation. Core owner
`01a147d7-f540-778e-ad79-4d8eb267403d` assigns the physical GPU, process slot,
memory reserve and time window. MOSS priority jobs remain ahead of audio.
Legacy dispatch holds remain active; none of the old train/queue launchers
will be resumed.

1. Freeze QA-approved core/adapter/PG source profile. Bind the exact 49 source
   hashes, released base-wheel digest, single ALZ patch digest, explicit
   declared domain and resolved continuous domain. Requalify both presets on
   CPU, including each native runtime and cross-profile/domain restore
   rejection. This CPU/source gate is now qualified against immutable core
   `3e94b13eb5d6cd1ceee3f018f019a68603f35696`; the release/auto width guard
   remains active by default. No alternate patch, projection or field reshape.
2. Request a separate native training smoke lease: four train/two heldout
   rows, batch 2, rank 8, existing 128×4 bank, max 512 tokens, seed 7,
   `bispf_optimized`, total budget 20 updates. Compare fresh-process 1+2 against 3
   updates with the same declared 20-update recipe. Write checkpoints every 2
   updates, never extend the budget silently. Hard wall limit 900 seconds,
   failed gates stop only this task's own child process.
3. On successful smoke and renewed admission, continue that same checkpoint
   to 20 updates. Report actual elapsed time, peak allocated/reserved device
   memory and resource telemetry. A larger training budget requires a new
   explicit decision; 20 updates alone do not establish musical quality.
4. Request a separate listening lease. Use two predeclared heldout neutral
   captions/lyrics with seeds 1709/1710. Save matched base, adapter-off,
   scales 0.5 and 1, plus frozen positive-caption reference. YuE2 uses the pinned
   eager AR path with `cot=off`, max 200 semantic tokens, then frozen NAR/VAE;
   record truncation rather than hiding it. Music3 uses the staged native
   modular pipeline, max 8 seconds per clip. Preserve the actual sampling,
   CFG and decoder configuration; no unrelated image/flow adapter path.

Exact proposed argument vectors are frozen in the command cards. They remain
pending the core owner's external guardian and explicit GPU admission.

## Representative output checks

Training must show finite generator adversarial/prior and critic
adversarial/continuous-penalty components, nonzero response-direction gradients
to the actual bank, adapter changes, and unchanged frozen base weights. Report
row exposures and exact conditioning. Require exact native scale 0 replay,
adapter-only export/reload parity, and exact fresh-process continuation of
model/optimizer/controller/prior/RNG/loss state. Heldout response change must
be finite and nonzero, with no quality claim from that fact alone.

Audio outputs must have finite nonempty mono/stereo arrays, correct declared
sample rate, recorded duration/token bounds, deterministic base/off replay,
and waveform-effect measurements for the trained adapter. Report clipping,
silence, RMS/peak values, lengths and all truncation flags. Musical concept
direction, intelligible lyrics, voice consistency and artifacts require
listening review of the two predeclared cases. A waveform difference alone
does not pass semantic quality.

## Reviewed host integration and outstanding GPU gates

QA has passed eight optimizer/fresh-process cases for the single ALZ proposal
at widths 1024/2048/4096/5120, both presets. Patch SHA256:
`c6531a21f62fa96682f5aece517a6a862bb849ab971aa2766b3946b9a7543571`.
The proposal remains an isolated overlay. QA independently requalified
immutable core `3e94b13eb5d6cd1ceee3f018f019a68603f35696`; its public
validator and source/domain/runtime-bound checkpoint contract are wired into
the audio adapters. Native YuE2-2048 and Qwen3-4096 random-model CPU gates
pass under both presets on their separate runtime stacks, including exact
fresh-process continuation and safe checkpoint loading. **GPU admission and
pretrained training/listening qualification remain pending.**

Additional preparation checks: 16 existing audio tests passed in 7.03 seconds
under the native YuE2 Torch 2.10/Transformers 4.57.6 interpreter on the released
PG0.9 path, using the namespace bootstrap (fresh-process child bootstrap is
deferred to final profile qualification). Six matched-waveform checker CPU
tests pass in 0.03 seconds, including exact-off, finite/channel, duration-bound
and truncation-reporting cases. These tests synthesize arrays, not audio.

## Reviewable bounded workers

`scripts/qualify_audio_pg09_live.py` loads only the requested native model,
prepares the predeclared rows and delegates every update to ConditionalPG09.
It validates the exact domain profile before CUDA initialization/model loading,
requires an explicit admission identifier and fresh free-memory bound, reuses
the existing resource guard, hashes every frozen host parameter before/after,
checks actual-bank response gradients, actual bank weight changes with delta
RMS, critic/adapter changes, and tests
adapter reload after deliberately perturbing adapter state. Checkpoints remain
resumable after inference-export checks. No legacy trainer is invoked.

Before constructing YuE2, the worker resolves the pinned model ID/revision
offline, requires the actual directory to agree with both the supplied path
and manifest, and recomputes the native weight/config identity. It then passes
the pinned model ID/revision/cache to the native constructor and checks the
loaded identity again. A read-only asset check hashes the 7,261,441,640-byte
weight file and confirms an exact manifest match without loading tensors or
initializing CUDA. The critic has a stable import identity in `pg09_audio`,
allowing public inference-export reload from an imported renderer. CPU tests
exercise that separate import path and all three bank-update stage checks.

`scripts/run_audio_pg09_smoke.py` runs full 3, first 1 and resumed 3 as serial
children within a 900-second budget; no simultaneous CUDA children. It
compares every state/metric/heldout-probe/export field exactly. A CPU random
fixture exercises this supervisor; native/full-width source-profile gates
exercise both actual model classes on their corresponding CPU runtime stacks.

`evidence/pg09-audio/audio-admission-drafts.json` contains exact argument
vectors, worker/supervisor/resource-guard hashes, core/profile pins and the
proposed resource bounds. An admission ID is provenance, not lease proof.
Core owner owns the external guardian: both physical-card leases, GPU UUID
and resident admission, per-child process-group ownership, kernel telemetry,
resource guards, a 900-second total and no retry. This supervisor's
`subprocess.run` children inherit its process group; the external owner must
establish the required ownership or invoke the three worker argument vectors
directly. No alternate local lease mechanism is provided.

YuE2 requests at least 16 GiB fresh free memory; Music3 at least
26 GiB. The existing runtime guard retains its 6 GiB free reserve, thermal and
host/NVMe checks. Each smoke window is capped at 900 seconds. Subsequent
continuation to 20 updates and listening runs need separate renewed admission.

The corrected frozen worker SHA256 is
`019a56ef3475f2d876e6a3c0affee2c4029095d6e5e2bc92ada9ca2c8a4fa3ec`.
It supersedes the worker in commit `90c17d0`. Final repair evidence is in
`evidence/pg09-audio/live-worker-repair/source-receipt.json`: 29 release/worker
checks, four native-width checks per runtime, six focused worker checks, and
an exact serial supervisor CPU replay. These checks do not establish
pretrained behavior or audio quality.
