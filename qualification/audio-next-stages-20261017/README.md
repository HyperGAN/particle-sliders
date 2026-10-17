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
| YR0 | Heldout case 0, seed 1709: clean base, off, half, on, frozen positive-caption reference | 20 / 16 GiB | Five serial clips, each at most 200 semantic tokens and 8.1 seconds; 720 + 180 seconds | Independent review of the new CPU-qualified renderer, accepted Y20 export/qualification hashes, renderer guardian and new admission |
| YR1 | Same matrix for heldout case 1, seed 1710 | 20 / 16 GiB | Same separate 720 + 180-second lease | All YR0 gates plus accepted case 0 and a fresh root lease |
| M3S | Music3 LM-only native smoke: full3 versus fresh1+2 in the original 20-update recipe | 30 / 24 GiB | One supervisor, three serial CUDA children; 720 + 180 seconds total | Root must bind a Music3-specific guardian/card; first pretrained GPU memory/runtime qualification and new admission |

The preferred order is Y20, YR0, YR1, M3S. Root may instead review a separate
M3S window after Y20 while renderer review/producer/guard gates remain pending. Audio phases
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

## Separate renderer and remaining review/producer gates

YR0 and YR1 now contain frozen renderer command vectors and unreviewed root
card templates. They remain `PROPOSED_NOT_ADMITTED`, with no admission ID and
no accepted export/qualification hashes. The separate implementation is
`scripts/render_yue2_pg09.py`; no admitted worker, adapter, core or guardian
was changed. The legacy renderer remains unused.

The new bounded renderer retains the public inference export contract,
stable source/critic identity, actual routed bank, and captured adapter scale.
It generates the clean baseline before attaching the slider to the same
host, and gates exact off replay, RNG isolation and native host transfers.
AR generation uses explicit CFG 1.0, `cot=off`, eager BF16, 32–200 semantic
tokens and the predeclared seed. NAR/VAE remain at adapter scale zero, with
native 32-step midpoint synthesis and FP32 48 kHz stereo decoding. This CFG
choice is explicit and still requires independent review and native qualification.

CPU tests use actual random YuE2 AR models with the public GAN producer and
synthetic acoustic output. They cover all five variants, AR-only scale and
NAR/VAE zero, exact clean/off tokens/latents/waveform, Python/NumPy/Torch RNG,
token/domain/duration/prefix failures, public export source/probe/step rejection,
bank/frozen-host invariants, and fresh-process safe export reload. A direct
comparison against the unchanged producer loader confirms identical public
game signatures and exact 20-step export/probe replay on random native hosts.
No pretrained weights, native acoustic synthesis, GPU or dispatch were used.

The 20-step export does not exist yet. Its producer path is declared, but its
hash is intentionally absent; it must be frozen only after Y20 passes. All
rendered tokens, latents, samples, truncation flags, model/export/source hashes
and resource telemetry must be retained. Difference measurements do not pass
semantic quality; heldout listening remains a separate review.

Every card retains both physical-card leases, fresh UUID/resident admission,
one owned process group, at most one CUDA child, kernel checks, 6 GiB running
GPU reserve, 8 GiB continuous running host reserve, 100 GiB NVMe reserve, thermal/power
guards and root-controlled shutdown. The validator does not reserve capacity.
The 8 GiB host floor is continuous and separate from the stricter 12/16/24 GiB
startup floors. Only the new phase guards enforce this policy; already-bound
MOSS guards remain untouched. The new renderer also checks 8 GiB during its
own stage/forward checks, supplementing the unchanged resource helper.

`sequential-plan.json` indexes all four card hashes. `validation-receipt.json`
records successful CPU scope/source/input checks. The asset/runtime receipts
and the small reproducible offline validators are stored alongside the cards.

Every YuE phase binds the exact product root
`/mnt/nvme/sliders/audio-yue2-source-20261017` and all nine product Python
source paths and SHA256 values from the completed native smoke. The proposal
validator requires those nine pins and the fixed continuation `--product`
argument. The renderer also requires that exact root and all nine files before
model loading. CPU negative tests reject a missing product pin, a substituted
product root and a return to the obsolete 4 GiB continuous host floor.
