# Text/video PG09 bounded real-model verification preparation

This is preparation, not a launch receipt. Parent requested retained text/video
ownership through integration and validation. Core owner confirmed integration
of implementation `56c91a8` and evidence `8c31940` as canonical `e985db9` and
`b40d6e6`. The tested core remains `ec234e1`; its release-auto wide-vector gates
remain intact. No live adapter guard has been removed.

QA independently qualified the single existing ALZ overlay at
`/tmp/alz-pg09-domain-proposal`, patch SHA256
`c6531a21f62fa96682f5aece517a6a862bb849ab971aa2766b3946b9a7543571`.
Its eight engineering cases cover both presets at widths 1024/2048/4096/5120,
actual G/D/bank updates and fresh-process continuation. QA reports 42 exact
default-path comparisons. This does not admit a host profile or a model run.
The explicit source profile, all49 source hashes/base wheel/patch digest,
declared and resolved penalty domain, and cross-profile/domain transactional
rejection must be frozen and independently reviewed by the core owner/QA first.
No installed PG package or second patch was changed or created here.

## Existing assets and completed preparation

`scripts/prepare_text_video_pg09_verification.py` uses only the Python standard
library. It performs no model imports, network calls or process dispatch. It
pins the existing asset manifests; verifies small configuration/tokenizer files;
checks the presence and sizes of large weights; and labels their prior SHA
attestations accurately. Existing exact LTX metadata-cache links are accepted
only within their selected cache root and with matching metadata hashes. Other
escaping paths/links are rejected. No permissions or symlinks were changed.

The receipt is `evidence/pg09-text-video/assets-prepared.json`; eight negative
and positive audit tests pass (`asset-preparation-tests.xml`). Results:

| Asset | Existing source | Preparation result |
|---|---|---|
| TinyQwen | `models/tiny-qwen-da87bfb`, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | Seven metadata/tokenizer/license hashes verified; 1,192,135,096-byte weight stat matches prior manifest. Actual config declares native hidden width1024. |
| Bonsai | `models/bonsai-b072e1d`, revision `b072e1d3b35a0a630cece372c2127528e0994386` | 5,946,648,928-byte PTQ1_0 file stat matches. Manifest pins Prism runtime `7dffb158de30ebb8ef9d64f33c6b0b2d7c1e6313`. Neither binary nor server was executed. |
| Bonsai readouts | `models/bonsai-caption-readouts-20261017` | Metadata for **15** cached tensors inspected, not16. All bind model identity `358f97c219524fd96f9d510824f4bc7f874ad4e44927359e0911a3cb95bde5c7`, matching the archived failed run. All historical held-out neutral/positive captions are available. No tensor payload loaded. |
| LTX | `models/ltx25-staged-da0b690`, revision `da0b690960026eab794b8ceaf841233cf7064478` | Fifteen metadata hashes verified; fourteen weight-file stats match the existing 70,057,304,830-byte manifest. Local config confirms patch1, native128 video channels, VAE factors8/32/32. This resolves the prior discovery-only remote HTTP401 configuration uncertainty using previously staged assets. |

All model paths above are under `/mnt/nvme/sliders/qualification`.
The existing LTX manifest records private R&D use and the exact community
license source. No agreement was accepted and no new commercial/distribution
applicability is claimed. Full weight-file rehashing, source/runtime compatibility
and model placement remain prelaunch gates; current stat checks are not a new
70GB weight verification.

## Concrete bounded sequence after admission

1. Receive the core owner's frozen host API/profile and QA host-admission
   receipt. Keep release-auto behavior unchanged. Bind the exact approved overlay
   and both declared/resolved domains in every checkpoint and export. Do not infer
   API names or copy a source validator into an adapter.
2. Reuse that core gate to enable TinyQwen's existing actual routed adapter at its
   native1024 response width. Add the thin Bonsai readout adapter around the
   existing `BonsaiParticleHead` and actual128x4 bank; use immutable cached real
   readouts before considering a private fork-server process. No legacy trainer
   or old optimizer state is reused.
3. Qualify both presets on CPU at the real response widths, including all G/D/bank
   ownership/connectivity, frozen base/readout invariance, public continuous
   penalty, forbidden optimized MSE, profile/domain mismatch rejection before
   mutation, and independent-process2+2 continuation. Cache provenance and prompt
   identities must bind exact tensor payload hashes when loading is admitted.
4. Present the exact frozen runner command, source/profile/asset pins, selected
   device and budget to the core/process owner. Only request GPU admission after
   all applicable software gates pass. The owner alone selects process placement.
   A proposed hardware pilot is two updates, batch2, 600seconds maximum, no retry
   or budget extension; software continuation uses four total updates with a
   checkpoint after two. These are preparation budgets, not permission to run.
5. After admission, verify required local file hashes before model allocation,
   load only authorized local snapshots (`local_files_only`, remote code off),
   preserve BF16 frozen host/FP32 adapter precision, and record actual memory and
   timing. Compare Off, initial+1 and trained outputs, export/reload and exact zero.
   No resident service is stopped or altered. No speculative download or fallback
   model is permitted.

The current script deliberately has no `--run`, `--allow-hub` or GPU flag. Model
actions stay stopped until the exact host API and admission are provided.

## Comparable held-out evidence

The preparation receipt binds recipe and final-metrics hashes from the failed
`tiny-caption-native-1200-20261017` and
`bonsai-caption-native-1200-20261017` runs. Preserve the original three training
rows, seed7 and held-out strength grid0/.5/1. Historical held-out pairs are:

- `a garden beside a stone wall` → `a blooming rose garden beside a stone wall`
- `a sea beside a fishing boat` → `a calm turquoise sea beside a fishing boat`
- `a street with lamps beside a bakery` → `a cozy rainy street with warm lamps beside a bakery`

Retain canary/positive-input identity diagnostics and the original frozen-positive
reference. Report relative edit MSE/cosine only as diagnostics; never optimize
them. Tiny generation controls use original greedy `do_sample=False`,
`max_new_tokens=32`. Changed student conditioning (neutral rather than the legacy
positive input) is an explicit experimental correction. Nonzero hidden edits or
a four-update pilot do not establish language or semantic quality. A substantive
1200-update comparison requires its own measured budget and owner admission.

## Dependent actions still stopped

| Target / exact action | Missing condition | State |
|---|---|---|
| TinyQwen native1024 shared PG training | Reviewed host source profile/domain binding, backend opt-in and real-width gates, exact model/device budget admission | Pending; not launched |
| Bonsai native5120 head training or loading cached tensor payloads | Same host/profile/backend gates; core owner's current instruction is metadata inspection only | Stopped; not denied |
| Bonsai fork server | Explicit process admission and exact pinned command/runtime/weight verification; cached inputs should avoid this need for the first software pilot | Stopped; no server started |
| LTX default TE/video-connector GAN arm | A thin native post-connector sequence/mask response adapter preserving preconnector hold and source `c7096d4`, plus reviewed domain/shape contract. The continuous patch alone does not supply this adapter. | Stopped; optional velocity arm is not a substitute |
| LTX real AV denoising/decoding | Admitted native default or explicit velocity protocol, verified source/components/private terms, measured feasible process placement, matching full49-frame544x960 reference controls/seeds123/124/125, owner admission | Stopped; old MSE queue remains held |
| MiniMax H3 adapter/model execution | Reviewed genuine AV geometry and response/critic protocol, exact staged asset/variant manifest and applicable community terms, feasible measured placement and owner admission | Stopped; no generic flatten/image disguise or fake `predict_v` |

No GPU admission has been requested yet. Parent callbacks replace the unavailable
direct parent messaging route. Image owner now covers Krea2/Ming/Iris/Qwen-Image;
those families are outside this retained text/video ownership.

Reproduce metadata preparation without any model execution:

```bash
python3 scripts/prepare_text_video_pg09_verification.py \
  --output /tmp/text-video-pg09-assets-prepared.json
```
