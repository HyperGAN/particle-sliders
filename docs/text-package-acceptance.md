# Installed TinyQwen/Bonsai package cohort

The public Tiny/Bonsai surfaces use `particle-sliders-core` API 0.6.0 with official
ParticleGAN 0.9.1. This is a new source-bound cohort. No active PG 0.9.0 environment,
checkpoint, worker or service is retargeted. The candidate source commit, exact
wheel hashes and critical native implementation hashes are recorded in
`conceptmod/textsliders/text_package_pin.json`; Ada review status is explicit.
Candidate qualification does not claim review approval or a merge.

The final frozen candidate is `cdfb80a76b882b4b73861b12f555263bd64ffeb7`.
Its core wheel SHA256 is
`4abc83988fdeb6d7478de830e7ee8ec5d9bd99c20d52f89e4ce594d50dfd39a3`;
the official PG wheel SHA256 is
`b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec`.
The consumer pin manifest SHA256 is
`d4733bb038c78c049274ffca8aaffdd8aa94ce4e47eda0c559300c07a186e104`.
The earlier `42a2e371` receipts are retained under `superseded-42a/` and do not
constitute acceptance of this replacement. The `replacement-4be/` receipts remain
scoped to that exact earlier wheel; they are not silently rebound to CDF.

## Installation and runtime identity

Use the two exact offline wheels named by `requirements-text-pg09.txt` in a new
isolated host with already qualified Torch/NumPy/safetensors dependencies:

```bash
python -m pip install --no-deps --find-links /verified/offline/wheels \
  -r requirements-text-pg09.txt
python scripts/qualify_tiny_pg09_cpu.py --output /fresh/tiny-software-probe --until 2
python scripts/qualify_tiny_pg09_cpu.py --output /fresh/tiny-software-probe --until 4
```

The requirements enable `--no-index` and `--require-hashes`; they cannot silently
select another release or wheel. `requirements-tiny-pg09.txt` includes this lock.
Do not install the repository's historical embedded 0.3 core into this cohort.
The bounded Tiny CLI imports the installed package and refuses `--core` checkout
injection. Distribution versions alone are insufficient: the product verifies
wheel metadata, owned file paths, critical source hashes and actual imported
module origins before native execution. Source checkout shadows fail even when
installed version metadata is visible.

Release/auto remains blocked for native-width 1024 Tiny and 5120 Bonsai continuous
training. The only alternative is an explicitly selected, fully source-verified
`alz-continuous-v1` / `continuous` profile. The product permits that reviewed PG
source overlay only when requested; the core/native helpers must still originate
from the locked installed wheel. No reshape, projection, automatic fallback or
installed-source patch is performed.

`PG09TinyRun` is protocol `tiny-qwen-shared-package-v3`. Shared `ConditionalPG09`
owns the entire GAN update/sampler/checkpoint; native Tiny backend, routed bank,
projection, response adapter and critic are package objects. PG 0.9.1's optimized
preset defaults to BF16, while this native response loop actually trains FP32
adapter leaves; resolved recipe and export validation explicitly bind FP32.
New checkpoints/exports bind a portable exact package pin. Old local/v2 states
cannot be relabeled as this cohort.

## Public archived-export behavior

`train_lora_tiny_llm.py` and `train_lora_bonsai_gguf.py` retire positive-step local
Rp/VIC training before model, prompt, readout, download or output access. Their
`--steps 0` modes require an existing explicit export (or a directory with exactly
one export). Ambiguous EMA/live directories require a filename. Old
`qualify_captions.py` and `queue_caption_qualification.py` launchers reject before
reading pinned recipes or creating queue/process state. Existing forensic helper
functions remain archival and are not new shared training entry points.

The old `qualify_text_video_pg09_cpu.py` and `qualify_wide_text_pg09_cpu.py`
Tiny/Bonsai routes also reject before source/output/process access. Their LTX
harnesses remain separately owned and require explicit `--family ltx` or
`--family ltx_connector`. `verify_bonsai_pg09_cached.py` retains metadata-only
preflight, while produce/restore are retired. The diagnostic/ablation launchers
are archived here because their absolute source manifest did not bind an
arbitrary `--code` checkout's actual imports. Historical reproduction requires
the original immutable cohort source; it is not new package qualification.

For Bonsai, reporting an explicit head is:

```bash
python -m conceptmod.textsliders.train_lora_bonsai_gguf \
  --steps 0 --load_head /saved/selected_head.safetensors \
  --name review --save_dir /fresh/report --no_report
```

`--no_report` validates and records the head without creating a backend, fetching
readouts or constructing a critic/optimizer/sampler/EMA. Omit it only with the
explicit existing fork-server/readout setup (or `--dummy` synthetic fixtures).
Tiny's corresponding flag is `--load_tiny_lora`; protocol-bearing exports use the
strict current source/model/vocabulary loader. Unsupported protocols fail before
attachment. Raw archival exports without a verified base fingerprint are labeled
`base_identity_verified=false` and never become quality-qualified.

Both public modes preserve source bytes and adjacent metadata. They emit an
inference receipt and optional diagnostics, never a new step-zero weight export.
The original weights kind, training step, model/protocol metadata, failed quality
flags and source checksum are retained. Receipt ownership is reserved before
report output; concurrent calls cannot publish another operation's report.
Checksums are checked around loading/reporting. Nonfinite responses/metrics fail
without publishing nonfinite report JSON, and failed operations are recorded as
failed rather than completed.

## Acceptance boundary

Prior 4be receipts in `evidence/text-package-acceptance/replacement-4be/` cover installation from hashed
wheels, actual imported package paths, source-free shared GAN execution, fresh
process continuation, exact FP32 reload on a BF16 frozen host, signed/zero scale,
transactional invalid-load rejection, meta-device placement, source/profile
shadows, native 1024 explicit-profile replay/export, and actual random Qwen3 public
checkpoint-hook gradient parity. Only synthetic CPU software updates run.

Replacement receipts are in `evidence/text-package-acceptance/replacement-4be/`:
93 consumer/public tests passed with two expected native weight/server skips;
one explicit native-width 1024 replay/inference test passed; and 133 frozen-package
optimizer/native/Qwen tests passed. The latter include both G/D negative second
moments, incomplete/extra/erased histories, mandatory history presence, valid
initial empty history, honestly unused critic parameters and exact next-update
resume. Rejected states leave the receiver unchanged. Installed imports were
checked independently of the frozen source checkout used to select these tests.
The source-free subprocess blocks repository and native runtime imports while
executing two real shared GAN updates against a dummy native model.
Eleven additional subprocess regressions confirm retirement before source,
artifact, output or process access, including preservation of existing reports.
The combined 4be acceptance was 238 passed, two expected skips.

Final CDF receipts in `evidence/text-package-acceptance/final-cdf/` record a narrow
29-test installed-wheel acceptance with no failures or skips: 14 consumer cases,
14 focused optimizer cases and one explicit native-width 1024 case. These verify
the exact new pin/import origins, source-shadow rejection, source-free actual
GAN updates, fresh-process continuation, FP32 reload, Bonsai selected-weight
inference and native signed export/reload. A native Tiny regression rejects zero
and future G/D Adam clocks atomically, then resumes the valid snapshot to the
exact next update. Shared tests retain valid initial empty, never-used and
partially-used critic histories. All fixtures are freshly constructed under CDF.
The earlier checkpoints and running cohorts remain source-bound to their own
packages. No unrelated native/Qwen broad suite was rerun for this narrow adoption.

Bonsai's previous 1200-update held-out failure remains unchanged: 0/3 successful
cases, mean relative-edit MSE1.9433580629, `quality_qualified=false`. No extension,
retry or metric rewrite is performed. The family owner separately owns pretrained
Tiny generation/quality, live model/export identity, resource/memory/throughput
and coordinator process/GPU admission. LTX/H3 are assigned separately. No model
weights/public samples are published and no merge is performed here.
