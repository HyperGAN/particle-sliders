# Stable source integration: evidence boundaries

The core0.8.3 main-target integration preserves the officialPG0.9.1 reviewed
lifecycle history (core0.8.1 `32f6830`, core0.8.2 `f6807a9`) and exact upstream
source gates. It adds historical stamp/constructor compatibility, not a new GAN
law or checkpoint migration. Product drafts are independent source changes;
publication/merge of this core does not merge or repin any product.

Selected completed product software checks at immutable handoff heads:

| Product | Exact handoff head | Evidence | Core |
|---|---|---|---|
| Anima | c24005bb45e901b7343e0f06d64b93b8259fc021 | 16 installed lifecycle/native fixtures +16 preserved PR5 dispatch checks | 0.8.1 |
| Krea2 | b1f42f17f028cab70d3b4517e8f51aa3b5396355 | 37 installed +6 dependency guards | 0.8.1 |
| Supra | 8cffbadc95befd196fb4e77acb69ff181ee097f5 | 18 installed fixtures | 0.8.1 |
| Iris | fb5acc391ae7e32ad077364df2b1cbfc4c9ae158 | 60 installed fixtures | 0.8.1 |
| Qwen Image | cd0da29985ce017134dc436d3c5db91e237c05d3 | 26 installed fixtures | 0.8.1 |
| Fibo | 8b8adb7956adafa90b683cf3bb2e7689269e2df5 | 48 installed fixtures | 0.8.2 |
| Music3 | b16b756e08fa55004fd3dc93e699b5e8cd4a38e0 | 22 installed numerical fixtures | 0.8.2 |
| YuE2 | 25977f1f55be02d64a125f4b3d88ad1a126edac9 | 13 installed numerical fixtures | 0.8.2 |
| MOSS | 614cd9b52d305252776324a1b00e738539b548e4 | 31 installed packaging/session cases, including the previous25 source cases | separate officialPG0.9.3 line |
| Ming | 4c314cd6db29872f5e796f5b09f12786659820c6 | static audit, disabled operations; no native implementation | audit only |

MOSS packaging used the isolated declared Hatchling1.28.0 backend and preserves
core18a9aa8. Native/sampling/compact inference remain gated; this removes the
previous packaging blocker, not the separate PG-line boundary.

These are selected receipts, not one universal all-products suite. Some handoff
heads contain only docs/evidence after unchanged tested runtime; their owning PR
must name both. No code from those product workspaces is copied into this core.
Sana is in this repository and has a separate archive-faithful migration plan.

Iris rendering remains refused; Krea is adapter-only with no routed bank.
Official wide continuous Music3/YuE2 states remain refused. CPU hidden/latent
fixtures do not demonstrate pretrained output quality, actual image sampling,
acoustic/waveform output, accelerator fit, research-cohort equivalence, or matched
HNDL network migration. The separate unofficial PG-domain proposal and its
synthetic qualifications are outside this stable source integration.

All consumers keep source/model/architecture/profile/cohort identity. Historical
checkpoints and samples remain owned by their old pins; no permissive restore or
silent model/recipe substitution is introduced. HyperGAN reporting and matched-
weight HNDL network migrations are separate integration work, not implied by a
shared import. No weights, model downloads, deployment or package release is
part of this Git integration.
