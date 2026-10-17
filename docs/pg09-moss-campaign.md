# Live MOSS campaign bridge: GAN-only rerun preparation

This branch prepares Bad Intent and Afterimage for an admitted full-model rerun. It does not start that rerun, load pretrained weights, choose a GPU, cancel a process or revive a historical launcher. The core/process owner controls those actions. The old recipes, caches, MSE checkpoints and as-recorded images remain unchanged.

## Frozen data and useful conditions

`MossCampaignRows(directory)` reads the old `recipe.json`, `target-index.json`, cached prompt embeddings and target sidecars. It verifies canonical recipe/prompt/index identities, model revision and target asset hashes. Every target file is checksum-verified and schema/finite-checked on bounded LRU admission. It retains lightweight cache addresses and materializes only selected native rows. Core must use the lazy-provider API; converting it to an eager list/tuple is inappropriate for the live dataset.

Each concept has 24 training captions ×16 seeds ×2 frozen trajectories ×10 positions =7,680 unsigned states. Every state has both signed targets, giving 15,360 rows. Native responses remain `[1,128,32,32]` through the released PG penalty. No projection, downsampling, arbitrary image reshaping or categorical fallback is introduced.

The cached target convention is **frozen neutral + sign × strength × (frozen positive − frozen neutral)**. The negative teacher pole is the neutral caption. The empty prompt belongs only to CFG rendering and must not replace that teacher pole.

Conditions contain 4,101 float32 features: mask-weighted mean neutral Qwen context (4,096 coordinates), divided by its own pooled RMS, followed by raw native timestep/1,000, sign, strength, training guidance/3, and frozen-trajectory path. There is no row-ID feature. Context, time, sign and target stay paired for real/fake/penalty evaluations. The small native-field critic uses context-dependent FiLM, so condition features influence field gradients rather than only adding a scalar score offset.

Training requires `split='train'`. Held-out providers remain readable for evaluation, but both `build_campaign_run` and `response_scale()` reject them. Frozen response-scale calibration is training-only edit RMS; it is not an optimized MSE loss.

Read-only CPU calibration of all 384 training bundles per case produced:

| Case | Fixed response scale, strength 1 | Native signed rows |
|---|---:|---:|
| Bad Intent | 0.18700603425178916 | 15,360 |
| Afterimage | 0.19194795196490175 | 15,360 |

Both audits took about 30 seconds each, with peak process RSS about 700 MiB including Torch and the bounded cache. Reports live under `evidence/pg09-moss-campaign/`. No pretrained model weights were read.

## Runner integration

```python
from conceptmod.textsliders.pg09_moss_campaign import MossCampaignRows, build_campaign_run

rows = MossCampaignRows('/path/to/original/campaign', split='train')
run = build_campaign_run(
    admitted_transformer, rows,
    model_identity=verified_staged_model_identity,  # model + exact matching revision; coordinator verified
    response_scale=verified_training_scale,
    batch_size=8, total_steps=3840,
)
metrics = run.step()
packet = run.game.state_dict()                  # shared-core complete optimizer/RNG/sampling checkpoint
run.game.load_state_dict(packet)                # shared-core transactional restore
run.export_adapter('/new/run/adapter.safetensors')
cases = list(run.render_cases())                # no inference, RNG consumption or model call
image = run.render(admitted_pipeline_with_vae, cases[0])
```

The model/pipeline must already be admitted and staged by the coordinator. The factory creates only an FP32 ordinary MOSS LoRA and a small condition-aware critic. It installs the common repeatable scale-captured checkpoint wrapper, explicitly selects `bispf_optimized`, and calls the single shared `ConditionalPG09` with lazy rows, lazy conditions and balanced sampling. It owns no training loss or optimizer. MOSS is explicitly an **adapter-GAN arm**, without a synthetic or disconnected particle bank.

No MSE, distillation, reconstruction or preservation loss is added. The optimized objectives remain public PG adversarial generator/critic losses and PG's required constraints. Old MSE optimizer state cannot resume into this arm. Calibration and later residual diagnostics must remain labeled as diagnostics.

Portable exports include the adapter-only safetensors file, its checksum, exact core configuration/resolved recipe, model/cache identities and completed step. Export paths are immutable. Reload validates provenance and uses the core's inference-only export gate; calling a training step after inference loading is rejected.

## Coverage and budget proposal

The old 1,600 updates ×8 unsigned states evaluated both signs, totaling25,600 signed generator forwards. Simply keeping1,600 GAN updates at batch8 would provide only 12,800 signed generator exposures. Independent random sampling with replacement would leave many rows unseen.

| Generator batch | One full signed cycle | Match old 25,600 G forwards | Proposed two full cycles |
|---:|---:|---:|---:|
| 8 | 1,920 updates | 3,200 updates | 3,840 updates |
| 4 | 3,840 updates | 6,400 updates | 7,680 updates |
| 2 | 7,680 updates | 12,800 updates | 15,360 updates |

Use independent balanced shuffled D/G cycles, with exact order/cursor/exposure checkpointing owned by core. Proposed sustained budget is two full G cycles; this is 20%more generator exposures than the old run, plus the adversarial critic forwards and protected-gradient recomputation cost. A coordinator-admitted bounded canary must establish actual batch size, memory and time before a sustained budget is approved. No wall-time or quality claim follows from the CPU fixtures. A 20-step canary is a software/resource gate, not a completed rerun.

## Matched evaluation and protocol correction

Preserve the two archived held-out captions, evaluation seed 29001, CPU initial noise, official native 50-step scheduler, 512×512 resolution, CFG 3 and strengths−1/0/+1. Preserve the direct-positive baseline at both CFG 3 and CFG 1. The second seed listed in source manifests, 29027, was not used by the old campaign renderer; adding it is a separate expansion, not a replacement for matched evaluation.

Archived rendering adapted both conditional and negative branches and skipped block 22 in the negative branch. Corrected inference adapts only the conditional branch and uses a frozen empty-prompt negative without layer skipping. This is an intentional protocol change, recorded in each returned case specification. For objective comparisons, the coordinator should re-render the archived MSE adapter under the same corrected inference protocol, separately from its archived as-recorded images. This helper never silently substitutes corrected images for originals.

The generic render loop reuses exact native scheduler timesteps/sigma intervals and official MOSS FLUX.2 VAE normalization/unpacking. The VAE is injected by the admitted pipeline; there is no fake-image fallback.

## Remaining legacy cases and ports

The independent six-case reference manifest still marks all sustained GAN reruns unverified. CPU ports do not change that status.

| Case/group | Current port state |
|---|---|
| MOSS Bad Intent, Afterimage | This lazy live helper; pretrained GAN rerun and held-out quality pending admission |
| MOSS Final Form | Same cache/recipe format supported; stopped MSE checkpoint retained; no corrected sustained rerun yet |
| Supra Final Boss | Native runtime adapter exists; dedicated lazy campaign/positive-CFG teacher and preservation-control rerun still needs integration |
| Sana Happy | No released-PG live campaign port in this branch; preserve failed smile and fruit-control tests |
| SD3 Smile | No released-PG live campaign port in this branch; preserve scheduler-step teacher coordinates and fruit control |
| Krea Smile, Cascade Pixel Art, SD1 Age, Flux Green→Red, LTX2.5 Smile, Z-Image Age | Historical MSE entrypoints held by the coordinator; dedicated corrected live ports/reruns remain pending, without assuming each is an observed quality failure |

Anima's product routed adapter is already adversarial, so it must not be classified as MSE solely because its released-PG0.9 live rerun is pending. Music3/YuE2 and language-model comparisons have separate owners and domain/admission gates; this helper does not dispatch them.

Reference: QA's `qa_pg09/rerun-reference-manifest.json` and `RERUN-RELEASE-GATES.md` in its isolated qualification checkout. User intent remains actual corrected sustained reruns and held-out outcomes, not CPU qualification alone.
