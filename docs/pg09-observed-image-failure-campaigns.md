# Prepared reruns of the actual Sana Happy / SD3 Smile MSE failures

This isolated branch prepares the two observed failure cases requested by the parent. It does not broaden the task to every held image queue. The archived500-update MSE results stay unchanged; neither old trainer is invoked or restarted. No model/checkpoint contents have been loaded, no cache has been extracted, and no GPU worker/process has been dispatched. The proposals are not admitted and are not image-quality evidence.

Checkout: `/mnt/nvme/sliders/image-failure-campaigns-pg09-20261017`, branch `campaigns/sana-happy-sd3-smile-pg09`. Only new files are added. Canonical shared core and active MOSS sources are untouched. The sole core/GPU/process owner remains thread `01a147d7-f540-778e-ad79-4d8eb267403d`.

The complete machine-readable proposals are in `evidence/pg09-image-failure-campaigns/proposals.json`. They bind archived recipes/progress/disposition, original initial-checkpoint **metadata only**, repaired source files, runtime library sources/versions, model manifest metadata and15 frozen native baseline PNG hashes. Core is the requested immutable `3e94b13eb5d6cd1ceee3f018f019a68603f35696`, module SHA256 `426abc25323aea96be91e92f1f4101063d5b3c4062f042e869c729027b4f9256`. Preparation independently verifies all49 installed PG0.9 Python sources against its released source manifest. These genuine4D fields use the unchanged `release` / `auto` profile. No wide-vector overlay or PG mathematics is changed.

## Preserved cases

| Setting | Sana Happy | SD3 Smile |
| --- | --- | --- |
| Archive | `runs/sana-happy-20261017` | `runs/sd3-smile-20261017` |
| Model revision | `83d7a190bfd1fd070570a793d2dab5c7a3231b9d` | `ea42f8cef0f178587cf766dc8129abd379c90671` |
| Actual archived source | Original `a6f94e7`; completed repaired implementation `9d093415230ace2f94377464047f1da77b1eaba9` | `526e0365dc66094ba9030659637e7667b62152c2` |
| Training contexts | Both original YAML rows and attributes; person / portrait neutral captions, happy positives; sad minus is canary only | All4 actual archived recipe contexts; the unrelated checked-in example YAML is not substituted |
| Student response | Raw neutral velocity at fixed native t500, training attention mask of ones | Native one-step scheduler endpoint at CFG1, after the original CURRENT-adapter no-grad neutral prefix at selected index1..49 of50 |
| Frozen teacher | Empty-prompt velocity +4.5*(positive velocity-empty velocity), with original training mask convention | Original BF16 neutral endpoint +(positive endpoint-negative endpoint); subtraction/addition are retained rather than simplified to positive |
| Architecture | Same complete attn2 set,148,764,672 trainables in the original run | Same Attention/full rank4 alpha1 LoRA,1,179,648 trainables in the original run |
| Precision | Original frozen FP32 base + FP32 full-weight delta, exact same parameter capacity; no LoRA reduction | Frozen BF16 host; FP32 LoRA leaves required by shared PG0.9. This training-leaf precision change is explicit, not a claim of old BF16-update equivalence |
| Response field | `[1,32,16,16]` native velocity | `[1,16,64,64]` native scheduler-step latent |
| Native render |512px,20 steps,CFG4.5, actual cached masks and explicit empty negative |512px,28 steps,CFG3, original provided embeddings/pooled text and explicit empty negative; no T5 substitution |
| Adapted CFG branches |Both native branches, as in archive |Both native branches, as in archive |
| Held-out people |Exact archived female/male neutral and smiling prompts, each seeds123 and456 |Exact archived female blue-coat / brick-wall prompt at123, male bookcase prompt at124 |
| Fruit control |`a bowl of fruit on a table`, seed789 |`a ceramic bowl of fruit on a wooden table`, seed125 |
| Strength grid |Off0,0.5,1 plus frozen positive teacher |Off0,-1,0.5,1 plus frozen positive CFG3 teacher |

The exact prompt strings and baseline file hashes are in the proposals. Fruit remains evaluation-only: no new fruit teacher, reconstruction penalty or optimized hold loss is added. Original teacher snapshots and failed controls remain available for matched panels.

Sana's archived quality review says happy-expression transfer failed for both people/seeds and fruit/bowl content changed at strength0.5/1 despite completed500 updates and successful software/zero/export checks. The parent selected SD3 Smile as an observed failure; its archived receipt completed500 updates and explicitly left image quality unqualified. This preparation does not invent a new visual judgment from scalar losses or the CPU tests.

## Thin campaign-specific ports

`conceptmod/textsliders/pg09_image_failure_campaigns.py` supplies response/generator state and lifecycle code only. All D/G losses, penalties, optimizer steps, public constraint-geometry backward, recipe LR policy, balanced D/G sampling and checkpoint state remain in `ConditionalPG09`.

`SanaHappyCampaignBackend` retains the actual full-xattn function class. `XattnDeltaNetwork` represents each original attn2 parameter as frozen W plus trainable FP32 delta using `torch.func.functional_call`; base weights never enter the optimizer/export. At scale0 it calls exact base W. It does not substitute a tiny/random model, low-rank adapter or a generic CFG-student response. Sad minus conditioning cannot accidentally replace the empty teacher: the input row has an explicit `unconditional` field. Native gradient-checkpoint replay of functional weights is rejected until separately qualified; the archive did not enable it. Native pipeline/offload/compile-wrapper compatibility must pass the real parity gate before admission.

`SD3SmileCampaignBackend` retains real `scheduler.step`, captured native timesteps/sigmas/cursor and the archive's BF16 endpoint rounding. It does not treat the old scheduler-step output as a raw velocity. The original `diffusion_sd3` prefix helper is injected by source identity and reused under `no_grad` at current adapter strength1. `build_sd3_current_prefix_snapshot` then detaches that prefix and computes matched frozen teacher endpoints at the identical x/t/cursor. No old MSE training function is called. Enabled LoRA native checkpointing reuses the existing shared repeatable scale-capture hook.

`render_archived_case` calls the actual staged Sana/SD3 pipeline with exact archived prompts/embeddings, sampler steps, CFG, empty negative, CPU-generator seed and scale. It changes neither the sampler nor decoder. Sana's functional transformer is installed only for the call and restored in finally. Both branches see the same archive strength. Torch CPU/owned CUDA, Python and NumPy training RNG are restored around evaluation. This is an implementation proposal: no actual native model/pixel parity or full decoder has run in this task.

Known archived protocol gaps are preserved and declared: Sana trains a raw neutral student toward a CFG-positive teacher, and its training wrapper uses ones masks while rendering uses true masks; SD3 training is CFG1 one-step endpoint while rendering is CFG3 with adapter on both branches. Altering these would be a distinct protocol. It cannot be presented as a matched loss-only rerun without an explicit decision.

`ImageFailureCritic` consumes every native field coordinate and the useful full-conditioning features, flattens only internally, and produces raw unbounded logits using hidden widths256/128. It has no MOSS hardcoded state counts or fake particle bank. Both campaigns are explicitly **adapter GAN** arms with `particles=None`; the public prior regularizer is applicable only if a separately qualified, actually participating bank is supplied. No loss or optimizer is implemented locally.

## Online per-update snapshot provider: new contract, CPU prototype only

The core owner accepted this **new provider contract for CPU prototype/testing**, not live admission and not a claim that the old immutable-target-cache contract already covers it.

SD3's original neutral prefix runs with the current learned adapter. Thus, different learned G states naturally generate different prefixes. The preserved matching objects are initial-noise tensors/seeds, selected prompt/time/interval, guidance, source/prefix algorithm and held-out protocol. Silently replacing it with an initially frozen-prefix cache would change the original state distribution. Frozen-prefix caching remains an explicit unselected alternative in the proposals.

`OnlineSnapshotProvider` protocol **`online-current-adapter-snapshot-proposal-v2`** binds immutable descriptor/noise/context/time/guidance/source/condition hashes and a versioned algorithm identity to core's existing lazy-provider signature. Its target values are **not** described as immutable for the whole run. `online_campaign_step` only arms one snapshot, invokes **one shared core step**, then invalidates the deterministic row cache. It is not a second training loop.

Within an armed logical step, each selected unique G/preflight/D row is built once with no gradient into prefix/teacher. Before arming, the complete private descriptor set, isolated public diagnostic copies and condition table are checked against their declared hashes. Each row fetch rechecks its descriptor and the full condition table; each condition fetch checks the entire fixed identity. Builders receive defensive snapshots and any attempted descriptor mutation is rejected without changing the bound originals. Mid-step condition drift fails closed, including after a row was cached.

The provider freezes complete `network.state_dict()` tensor values and every named buffer (including nonpersistent buffers) on their native device. Every row/condition fetch and the end of row construction verifies parameter identity/version plus exact state/buffer names, shapes, dtypes, devices, layouts and values. This catches buffer changes, `.data` writes and added/removed state. Full start-of-step state and buffer SHA256 digests identify the receipts; counters alone are insufficient. Selected-row receipts include logical step, actual G/buffer digests, immutable descriptor/noise identity, actual prefix/paired-state/target/neutral hashes. State snapshots and cached rows are released at `end_step`. New targets for the next step use updated G. Outside the armed step, row and condition access fail closed. Builder use of Torch CPU/owned CUDA, Python or NumPy global RNG is rejected and its state restored; no independent evolving provider RNG needs checkpointing.

Loading G and completed step reconstructs the selected rows and receipts exactly in the bounded CPU contract tests under both public presets. The actual file-backed **separate-process GPU** version remains mandatory and unrun. Core's signature binds the deterministic provider algorithm/descriptors/conditions, while the core checkpoint includes the actual generator state; it does not pretend to hash every future dynamic target up front. `core.evaluate` cannot fetch this provider outside an armed step; held-out native pipeline evaluation is a separate path.

The proposed in-memory input-row descriptors are a CPU API prototype. Before production, root must stage a source/checksum-bound descriptor/noise reader and immutable complete condition index, with selected-row loading rather than duplicating encodings/inputs on GPU. No cache extraction or reader execution occurred here. Full generator protection has a real cost: Sana's148M FP32 leaves require roughly595MB of extra device snapshot storage plus595MB of state readback/hash per logical update; all named buffers are additionally snapshotted and hashed. Exact device comparisons run per fetch. This in-memory prototype also clones descriptor inputs and validates descriptor/condition hashes, so its time and device/host peaks require admission measurement before selecting a production reader. There is no unreviewed incremental-hash shortcut.

## Proposed trained comparison, not a canary result

For **each** of the two observed cases, the proposals freeze the public `bispf` and `bispf_optimized` recipes, with batch2, seed7 and a candidate **1,000 G /1,000 D update** horizon. A500-descriptor matched draw plan gives2,000 exposures and four complete balanced cycles per independent D/G stream. This budget exceeds the historical500-update floor and is derived from this image plan, not copied from MOSS's3,840 count. It is a concrete proposal for root review, not proof that the budget is sufficient for convergence.

The actual model runs must complete the authorized horizon and matched held-out/fruit panels before a quality claim. Checkpoints/panels at0,20,250,500,750,1000 preserve a direct historical500-update comparison and final trained endpoint. The native no-update/prefix-resume checks and canary stage are admission gates only; none replaces the trained endpoint. Nonfinite/invariant/resource failures may stop a run; no automatic retries or silent objective/budget substitution is proposed.

Each public recipe uses the historical G base LR as an explicit allowed override (Sana2e-5,SD31e-4). Other optimizer/penalty/LR-law choices stay public. One finite positive response-scale scalar must be frozen from **training-only initial frozen-teacher edit RMS** before the final recipe/run identity can be frozen; it is a detached response coordinate, never an optimized MSE/auxiliary loss. Its value is intentionally null while actual staged teacher calibration is unavailable. Gaussian reference sigma is1. There is no extra source-domain patch.

Seed7 and the original selected-time/noise-generation policy are retained, but the original per-update tensor trace has not been extracted or loaded. Initial checkpoint metadata hashes are recorded; checkpoint content remains untouched. Root must stage and checksum the actual draw descriptors after admission, binding the precise initial-noise bytes and selected times across both arms. No claim of byte-identical historical training-noise replay is made from a seed alone.

## Frozen evidence and outstanding gates

The CPU checks cover archive metadata/15 baseline bindings, exact prompt/control/seed/grid retention, refusal to read checkpoint/weight content, Sana teacher/raw-student algebra/full-xattn gradients, actual public native scheduler-step algebra/BF16 target rounding, native render arguments/RNG preservation, current-G prefix-helper calls, same-step/version/RNG isolation, actual G/prefix/target provenance and reconstructed core checkpoint continuation. Tiny arithmetic doubles are labeled CPU contract-only. These checks do not measure native model memory, image quality, transfer or convergence.

Run the bounded suite with CUDA hidden and the immutable core:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/mnt/nvme/sliders/image-campaign-core-pin-3e94/packages/particle-sliders-core/src:. \
/mnt/nvme/sliders/qualification/envs/pg090/bin/python -m pytest \
  tests/test_pg09_image_failure_campaigns.py -q -o pythonpath='' -p no:cacheprovider
```

Pending, without denial or uncertain execution: root GPU/process admission; exact noise/time/condition asset staging and global collision audit; independent online-contract review; real native frozen Off/teacher/pipeline/mask/sampler/512px decoder parity; actual parameter/source/runtime validation; memory/wall-time/hash-cost measurement; separate-process file-backed GPU resume; full adequately trained comparison and matched held-out/fruit review. Actual GPU slots are reserved by root's MOSS full jobs. This branch schedules nothing and changes no service, permissions, old results, active queue or publication state.


## Provenance repair and preserved failure evidence

Original proposal `78087def8d6cd6888be7eec6cadadb7ee094215b` passed15 own CPU checks but failed3 independent provenance tests: mutable descriptor identity, builder mutation of a live descriptor, and generator buffer drift. QA preserved the regressions in commit `3b3be3cc29c1b09ee3a3e973aa779945fcdb175c` and its original failure XML in `/ml2/codex-storage/future-workspaces/2026-10-17/task-3/qa_pg09/online-snapshot-78087de-independent-review.xml`. A later independent regression also exposed condition drift after a cached fetch. Protocol v2 supersedes the v1 proposal; the old failure receipts remain evidence, not rewritten passes. The repair adds private descriptor ownership, defensive builders, fixed-identity revalidation and full generator/buffer state guards described above. The original15-test report must not be used to claim v2 qualification.
