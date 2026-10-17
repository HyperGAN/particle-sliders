# New CPU-only private-generator draw reconstruction proposal

This change adds **new files only** after immutable v3 schema implementation `4ee0d9d`. Core, previous audit/proposal scripts, archive packets, active MOSS and earlier failure evidence stay unchanged. Root remains the sole actual packet/process/GPU admission owner. No actual packet reload, private assignment of archived states, global RNG restoration, CUDA noise, text encoding, native forward, cache extraction or model worker dispatch occurs in this deliverable.

## Completed prerequisite and current limits

Root's actual v3 static receipt is SHA256 `843b6966b7501068d8baf982fcd6f52561975684b0a690a6eee4cffac4c8aae1` (1.417s,22.7MB RSS). Its separately admitted restricted RNG receipt is `1ee854f66d368389b39c0cd08399a16b2ee0df4e873330c437e656c591cf9ed2` (2.492s,534,241,280B peak), bound to exact plan `40534a425e868d7bd49c9d07ccae78ba3322e2ca6eccb36da02e716a8aabe706`. Four initial/step500 packets passed strict case/schema/CPU-mmap checks, including SD3 LR scheduler counters0/500 and1/501. Independent QA froze source/AST/import-fit and root-receipt review in `93455ba8eac0cdb2301e7b2db889958476b1e6b0`. V2's failed common-schema attempt remains preserved. None of this establishes historical draw replay.

Only the **metadata JSON receipts** are copied into `evidence/pg09-image-failure-campaigns/private-draw-proposal/`. They show:

| Substream | Sana step0 to500 | SD3 step0 to500 |
| --- | --- | --- |
| Torch CPU byte-state hash | unchanged | changed |
| Python random-state hash | unchanged | unchanged |
| One saved CUDA byte-state hash | changed | changed |

This is consistent with the visible source, not a reconstructed trace. The receipt contains RNG hashes/schema, not raw CPU/Python states sufficient to run a private generator. A separately admitted bounded reader must reload only those already checksum-bound packets to obtain the values; **that reader has not run here**. Seed7 cannot stand in for post-construction checkpoint states.

## Implemented private comparison, unexecuted actual cohort

`conceptmod/textsliders/pg09_image_draw_replay.py` implements a source-only CPU substream check:

- Create local `torch.Generator(device='cpu')` and assign a cloned recovered CPU state to that object only. Create a separate `random.Random(0)` and assign recovered Python state to that object only. Never call global `manual_seed`, `set_rng_state`, `random.seed`, `random.setstate` or any global-state restoration.
- Sana alternates its2 source prompt rows for500 updates and uses fixed rawt500. The visible loop draws no CPU or Python randomness; native FP32 noise belongs to CUDA and is omitted from this CPU check.
- SD3 performs exactly500 ordered pairs of private CPU draws: `torch.randint(0,4,(1,))` for prompt, then `torch.randint(1,50,(1,))` for prefix index. No batched replacement, reversed order or extra draws. These are indices, not actual native scheduler timesteps/sigmas.
- Compare the produced private CPU and Python end states **exactly** with recovered step500 states and report both hashes, ordered indices and schedule hash. Mismatch is a reportable blocker; there is no search for hidden consumers, state repair, seed substitution or automatic retry. A candidate is marked unusable on either mismatch. `historical_draw_replay_qualified` remains false even on agreement.

`scripts/verify_image_draw_replay_pg09.py` is a reviewable but **unadmitted** next-cohort command. It reuses the unchanged fresh/disjoint/no-replace output guard, checks plan/actual module/source/recovered-receipt/packet/sidecar hashes, reloads only four known packets with `weights_only=True,map_location='cpu',mmap=True`, validates strict case fields and SD3 scheduler metadata, and compares CPU/Python/CUDA byte identities against root's recovered receipt. CUDA byte vectors are only compared as CPU bytes, never restored/interpreted by a CUDA generator. It copies the tiny CPU byte vector and Python tuple, drops packet mappings, then invokes the private comparison. It instantiates no model, optimizer or LR scheduler.

It checks that global Torch CPU, Python and NumPy RNG states stayed unchanged and CUDA remained uninitialized. An unexpected change aborts without attempting a global restoration. Caps remain one thread, CUDA hidden,120s wall/CPU, hard8GiB address space/RSS<=8GiB and>=16GiB host available. Output publication is exclusive and race safe. The new plan is a distinct root-reviewed CPU cohort based on completed v3 recovery; it does not weaken the prior static/RNG same-plan gate or grant itself execution.

The **implemented metadata-only preparation command** is:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /mnt/nvme/sliders/qualification/envs/pg090/bin/python \
 scripts/prepare_image_draw_replay_pg09.py \
 --output-dir evidence/pg09-image-failure-campaigns/private-draw-proposal
```

It checks only source and completed receipt JSON, records the actual visible source AST call order/locations, and freezes the new driver/module/source/input hashes and exact unexecuted argv in `private-draw-plan.json` / `stage-command.json`. This preparation and CLI help were run; the command that reloads packets was not.

## Hidden-consumer and save-timing limits

Source inspection covers the exact training/qualification/save/native-helper files already bound in the v3 plan. Sana checkpoints initial state after construction/encoding and before diagnostics/render; every20 saves follow optimizer and precede checkpoint diagnostics/render. SD3 initial save follows construction/encoding, then evaluation; later saves follow optimizer **and LR scheduler**. Resumed workers can perform extra diagnostics/renders between a save and the next draw. Those paths use explicit private CPU render/diagnostic generators, but a source-level argument is not proof about every downstream implementation/kernel. SD3 wraps Torch/Python/CUDA evaluation RNG, not NumPy. Sana's renderer lacks an all-stream restore wrapper.

Endpoint agreement strongly supports the proposed CPU substream but does not independently prove all intermediate draws or rule out compensating/irrelevant consumers. Endpoint mismatch must remain a concrete mismatch; this proposal performs no adaptive search. CUDA noise-state progression, native stochastic kernels/dropout, device/runtime identity and exact CUDA-generator mapping remain a **separate tiny GPU-noise admission**, then native text encoding/masks/pooled vectors and original sampler/teacher/decoder parity. SD3 prefixes additionally depend on the actual learned adapter state and captured native cursor/interval. No CPU Gaussian replacement or frozen-prefix substitution is supplied.

## File-backed descriptor reader progress: CPU fixtures only

`FileDescriptorAssets` implements lazy selected-row reads from a checksum-bound JSON index, with no eager response/target cache. Each selected row loads only a bounded raw-noise packet and its native condition packet, using exact checksum-verified bytes through restricted CPU loading. Index/draw/source/conditioning-policy/declared collision-receipt bindings, source-specific draw fields, file sizes, path containment/no-symlink rules, native shapes/dtypes/role keys/masks and finite values are validated. Private index bytes and returned copies prevent mutable live descriptor aliases; index drift is rejected. The reader has no evolving RNG, model, optimizer, target builder or GPU movement.

The returned `initial_noise_fp32` is the source's raw FP32 native Gaussian **asset**, with an explicitly named raw byte hash. Sana consumes FP32; SD3's later root-admitted native worker must perform the original BF16 model-input conversion and bind both raw and converted input hashes. The raw byte hash must not be mistaken for the existing backend's typed/digested `initial_noise_sha256`. The reader binds a declared complete-condition-gate checksum; it does not itself execute or qualify the global collision audit. All fixture gate/source hashes are labeled synthetic and cannot qualify production conditions.

Both public GAN presets were exercised through the existing `OnlineSnapshotProvider` and `ConditionalPG09` using native-shaped **synthetic** reader assets: selected-row caching, CPU file checks, core-owned preflight/D/G updates and exact in-process5 versus2+3 checkpoint continuation including the fourth penalty pass. The fixture critic/teacher/backend are explicitly arithmetic test doubles. No loss/optimizer/training loop was copied into this reader. This advances the reader/provider seam, not the native production worker or quality claim.

Remaining blockers: root source/plan review and admission for actual private CPU comparison; any CPU/Python endpoint mismatch; separately admitted CUDA noise and native conditions; actual complete collision audit/calibration; production reader-to-native worker/lease/guardian wiring; native teacher/CFG/mask/sampler/decoder parity; actual snapshot/hash/resource measurement and separate-process GPU disk resume; full1,000-update arms and matched held-out/fruit review. MOSS GPU priority and all frozen source pins remain in effect. No automation or process watch is created.
