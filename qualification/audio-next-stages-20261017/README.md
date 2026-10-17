# Proposed sequential audio admission

These are frozen CPU/offline proposals. They do not grant admission or dispatch
anything. Root thread `01a147d7-f540-778e-ad79-4d8eb267403d` owns every guardian,
physical-card lease and resource decision. All MOSS jobs retain priority and
their existing pins. No admitted worker, adapter, core, native runtime or
guardian source was edited by this preparation.

YuE2's actual pretrained smoke already passed: uninterrupted three updates and
fresh-process one plus two updates have identical complete state, metrics,
probes and export. Root completed in 177.615 seconds. This qualifies software
behavior, without a listening or musical-quality result.

| Phase | Proposed work | Fresh free GPU / host | Bound | Remaining gate |
|---|---|---|---|---|
| Y20 | Resume the exact full3 checkpoint for 17 additional updates; keep the original 20-update recipe | 16 / 12 GiB | One CUDA child; 720 seconds compute + 180 shutdown | Root must bind a continuation-specific guardian/card, check capacity and grant new admission |
| YR0 | Heldout case 0, seed 1709: clean base, off, half, on, frozen positive-caption reference | 20 / 16 GiB | Five serial clips, each at most 200 semantic tokens and 8.1 seconds; 720 + 180 seconds | Bounded shared-checkpoint renderer implementation and CPU qualification, accepted Y20 export/hash, renderer guardian and new admission |
| YR1 | Same matrix for heldout case 1, seed 1710 | 20 / 16 GiB | Same separate 720 + 180-second lease | All YR0 gates plus accepted case 0 and a fresh root lease |
| M3S | Music3 LM-only native smoke: full3 versus fresh1+2 in the original 20-update recipe | 30 / 24 GiB | One supervisor, three serial CUDA children; 720 + 180 seconds total | Root must bind a Music3-specific guardian/card; first pretrained GPU memory/runtime qualification and new admission |

The preferred order is Y20, YR0, YR1, M3S. Root may instead review a separate
M3S window after Y20 while the renderer is being implemented. Audio phases
never overlap each other or displace MOSS priority. There is no automatic next
phase or retry. The existing root guardian is intentionally restricted to the
completed YuE2 smoke; it rejects these other commands unchanged. A fresh
capacity window alone does not remove that scope requirement.

## What is ready

Actual SHA256 and size checks passed for **36,312,138,722 bytes** of existing
assets, including the YuE2 model/tokenizer/VAE and all staged Music3 LM,
tokenizer, condition encoder, RVQ decoder, flow transformer and vocoder files.
No missing download or native API import was found. No pretrained tensor was
deserialized and no model was constructed during preparation.

Both native interpreters passed the public continuous-domain validator and
recorded all 49 reviewed PG source hashes. YuE2 retains Torch 2.10.0 and
Transformers 4.57.6; Music3 retains Torch 2.11.0 and Transformers 5.15.0.
The environment includes only the reviewed ALZ overlay, the four original
wheel metadata files, immutable core source, and isolated adapter/scripts.
Whole site-packages injection remains excluded. Native pipeline signatures,
sources and heldout token boundaries were inspected without initializing CUDA.

Y20 and M3S contain exact proposed argument vectors for the existing immutable
workers, with explicit new-admission placeholders. Their roots must create
phase-specific guarded cards before those vectors become executable.

## Actual implementation and producer gaps

YR0 and YR1 are protocol specifications with `command: null`; they cannot be
mistaken for executable render cards. The current legacy product renderer
uses a different checkpoint format and inherits the native CFG 1.01 default.
It cannot be invoked unchanged for shared PG09 exports.

A new bounded renderer must retain the public inference export contract,
stable source/critic identity, actual routed bank, and captured adapter scale.
It must generate the clean baseline before attaching the slider to the same
host, then qualify exact off replay, RNG isolation and native host transfers.
AR generation uses explicit CFG 1.0, `cot=off`, eager BF16, 32–200 semantic
tokens and the predeclared seed. NAR/VAE remain at adapter scale zero, with
native 32-step midpoint synthesis and FP32 48 kHz stereo decoding. This CFG
choice is explicit and still requires renderer review and native qualification.

The 20-step export does not exist yet. Its producer path is declared, but its
hash is intentionally absent; it must be frozen only after Y20 passes. All
rendered tokens, latents, samples, truncation flags, model/export/source hashes
and resource telemetry must be retained. Difference measurements do not pass
semantic quality; heldout listening remains a separate review.

Every card retains both physical-card leases, fresh UUID/resident admission,
one owned process group, at most one CUDA child, kernel checks, 6 GiB running
GPU reserve, 4 GiB running host reserve, 100 GiB NVMe reserve, thermal/power
guards and root-controlled shutdown. The validator does not reserve capacity.

`sequential-plan.json` indexes all four card hashes. `validation-receipt.json`
records successful CPU scope/source/input checks. The asset/runtime receipts
and the small reproducible offline validators are stored alongside the cards.
