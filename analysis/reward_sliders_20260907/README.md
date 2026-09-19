# reward-ce-v1

Content Enjoyment activation experiment with gated, ordinary rank-8/alpha-8 Music 3 LoRA training.

Current stage: **complete**. Completed observations: 257. Failed observations: 47.

[Frozen protocol](manifest.json) · [Status](status.json) · [Capture audit](audit/parity.json) · [Prompt audit](prompt-audit.json)

The pilot fixes eight training, four development and four test families before scoring. Each exact caption/sheet/style cell has four seeds. CE is the only selection reward; other axes remain diagnostics. Both CFG branches are captured and steered after the complete prompt prefill. The captured frame feedback supplies shared histories for teacher/student residuals.

The live GPU audit compares decoded PCM, because FLOAT WAV files contain a timestamp in their PEAK chunk. The original file hashes remain part of each observation. The first engineering audit and its correction are retained under [amendments](amendments/0001-pcm-parity/record.json).

[Listen to all eight matched development A/B pairs](http://192.168.1.90:7860/reward-ce-v1/). Playback includes optional level matching and original WAV downloads.

## Direction fit

| Layer | Training rank agreement | Dev rank agreement | Median residual L2 |
|---|---:|---:|---:|
| 11 | 0.771 | 0.458 | 95.806 |
| 23 | 0.812 | 0.500 | 277.551 |

Association alone does not establish that steering improves audio.

Development selection: **layer23-s0.01**.

| Arm | Mean normalized CE |
|---|---:|
| off | 7.7925 |
| layer11-s0.01 | 7.7996 |
| layer11-s0.03 | 7.7701 |
| layer23-s0.01 | 7.8008 |
| layer23-s0.03 | 7.3663 |

## Causal result

LoRA training gate: **passed**.

| Comparison | Paired CE change | Seed wins | Family bootstrap 95% interval |
|---|---:|---:|---|
| positive vs off | +0.2263 | 56.2% | -0.0453, +0.4583 |
| positive vs random | +0.1122 | 37.5% | -0.0381, +0.2625 |
| reversed vs off | +0.1238 | 56.2% | -0.0743, +0.2721 |

The pilot gate requires at least +0.02 CE against both Off and the fixed random direction, more than half of seed pairs won against Off, and no unresolved arm failures. Only a fresh full-song transfer study can validate the whole selected method.

The pilot meets its declared continuation rule, but the family bootstrap intervals still overlap zero. Proceeding to a student does not establish a reliable or broad quality gain.

### Matched family effects

| Family | Positive − Off CE | Positive − random CE | Reversed − Off CE |
|---|---:|---:|---:|
| pilot-12 | +0.4352 | -0.0268 | +0.2857 |
| pilot-13 | +0.4814 | +0.1888 | +0.2585 |
| pilot-14 | -0.2054 | +0.3361 | -0.1852 |
| pilot-15 | +0.1941 | -0.0494 | +0.1364 |

These four families, not their individual windows or seeds, are the independent pilot units.

### Other scorer axes

| Axis | Positive − Off, equal family weight |
|---|---:|
| CE | +0.2263 |
| PQ | +0.0268 |
| PC | +0.2466 |
| CU | +0.1035 |

Only CE selected the teacher. These diagnostics are not a combined reward. Every raw and normalized window score remains in the [observations](observations/).

## Pilot intent diagnostics

Valid matched pairs: 16/16. Invalid diagnostic observations: 0.

| Measurement | Positive − Off, equal family weight |
|---|---:|
| style similarity | -0.00286 |
| voice similarity | -0.03194 |
| phrase accuracy | +0.14268 |
| rms db | +0.52260 |
| crest db | -0.66728 |
| window rms cv | -0.02996 |
| duration s | +0.00000 |
| silent fraction | -0.00001 |
| clipped fraction | +0.00000 |
| instrumental word rate | +93.49459 |

These post-selection measurements describe the short pilot audio; they do not establish full-song preservation and do not change the causal gate. [Pair and family details](evaluation/causal-intent/summary.json) include output embedding diversity.

## LoRA checkpoints

Latest exported step: 660. [reward-ce-v1_step660.safetensors](student/reward-ce-v1_step660.safetensors).

Full training states remain separate from inference safetensors. Checkpoints and effective strengths are selected with development audio CE, followed by matched composition controls and fresh transfer songs.

| Step | Prompt relative RMS range | Generation residual relative error range | Teacher residual cosine range |
|---|---:|---:|---:|
| [300](student/diagnostics-step300.json) | 0.098–0.146 | 1.268–2.746 | 0.197–0.519 |
| [600](student/diagnostics-step600.json) | 0.057–0.151 | 1.679–3.123 | 0.232–0.447 |
| [660](student/diagnostics-step660.json) | 0.164–0.225 | 2.198–4.295 | 0.167–0.354 |

These are conditional-branch, same-history diagnostics on one row per training family at multiplier 1. Passing the gross-divergence bounds does not establish preserved words or improved free-running audio.

## LoRA development selection

Selected arm: **lora-step600-m1**.

| Arm | Mean normalized CE |
|---|---:|
| off | 7.7925 |
| lora-step300-m0.1 | 7.6144 |
| lora-step300-m0.3 | 7.7066 |
| lora-step300-m1 | 7.7180 |
| lora-step600-m0.1 | 7.7735 |
| lora-step600-m0.3 | 7.8363 |
| lora-step600-m1 | 7.8901 |
| lora-step660-m0.1 | 7.7942 |
| lora-step660-m0.3 | 7.6052 |
| lora-step660-m1 | 7.6465 |

[Frozen evaluation recipe](evaluation/manifest.json). Solo effective strength is controlled through host energy; a lone fader does not sweep strength.

## Composition controls

| Comparison | Valid pairs | CE change | Seed wins | Family bootstrap 95% interval |
|---|---:|---:|---:|---|
| Added LoRA, original style multipliers fixed | 4 | -0.0767 | 25.0% | -0.1379, -0.0156 |
| Added LoRA, total energy fixed | 4 | -0.4574 | 0.0% | -0.6857, -0.2290 |
| Added LoRA versus matched reduced styles | 4 | -0.2932 | 25.0% | -0.3939, -0.1925 |

[Exact resolved multipliers and observations](evaluation/composition-results.json).

## Pilot LoRA comparison

| Comparison | Valid pairs | CE change | Seed wins | Family bootstrap 95% interval |
|---|---:|---:|---:|---|
| LoRA versus Off | 16 | +0.3998 | 75.0% | +0.1537, +0.6458 |

These pilot families were already used by the activation gate; not fresh final validation.

## Matched runtime

| Treatment | Median end-to-end overhead | 10th–90th percentile | Median overhead excluding setup | Median setup seconds |
|---|---:|---|---:|---:|
| Activation teacher | -1.3% | -22.1%, +0.3% | -0.3% | 0.00 |
| Merged LoRA | +14.9% | +6.2%, +35.8% | -0.3% | 12.43 |

Engineering target: at most 2% median overhead. sequential matched runs with concurrent studio load on another GPU; report variability, not zero-cost claims.
The development grid changes the adapter mix between LoRA/teacher samples; consecutive Off seeds can reuse a mix. End-to-end differences therefore include unequal setup-cache patterns. The separately reported time excluding setup isolates that component, but does not remove concurrent-load variability.
[Setup, merge, generation and peak-memory records](evaluation/runtime.json).

## Full-song completion

| Arm | Valid duration | Cap hits | Too short | Other failures | Running |
|---|---:|---:|---:|---:|---:|
| off | 0 | 24 | 0 | 0 | 0 |
| lora | 1 | 23 | 0 | 0 | 0 |

Each arm has 24 declared outputs. Cap hits and short outputs remain failures even when the scorer returns a CE value; their raw scores and audio remain in the [transfer observations](transfer/observations/).

## Fresh full-song transfer

Twelve fresh families, two fresh seeds, and one frozen checkpoint/strength rule. Primary CE covers every disjoint ten-second window, including the actual tail, weighted by duration.

| Comparison | Valid pairs | CE change | Seed wins | Family bootstrap 95% interval |
|---|---:|---:|---:|---|
| LoRA versus Off | 0 | missing | missing | missing |
| beginning | 0 | missing | missing | missing |
| middle | 0 | missing | missing | missing |
| ending | 0 | missing | missing | missing |

There are no valid full-song pairs, so full-song CE improvement and intent preservation cannot be estimated from this run. Cap failures in either arm do not establish a LoRA-specific completion problem. The short-clip results remain separate evidence.

### By voice

| Comparison | Valid pairs | CE change | Seed wins | Family bootstrap 95% interval |
|---|---:|---:|---:|---|
| female | 0 | missing | missing | missing |
| instrumental | 0 | missing | missing | missing |
| male | 0 | missing | missing | missing |
| unspecified | 0 | missing | missing | missing |

### By genre

| Comparison | Valid pairs | CE change | Seed wins | Family bootstrap 95% interval |
|---|---:|---:|---:|---|
| acoustic ballad | 0 | missing | missing | missing |
| acoustic chamber folk | 0 | missing | missing | missing |
| acoustic soul | 0 | missing | missing | missing |
| arpeggiated dance | 0 | missing | missing | missing |
| chamber pop | 0 | missing | missing | missing |
| dance funk | 0 | missing | missing | missing |
| fast guitar pop | 0 | missing | missing | missing |
| guitar pop | 0 | missing | missing | missing |
| orchestral dance | 0 | missing | missing | missing |
| percussion pop | 0 | missing | missing | missing |
| small ensemble jazz | 0 | missing | missing | missing |
| syncopated electronic pop | 0 | missing | missing | missing |

Preservation checks: **not assessable**. Reported issues: 24. Generation failures: 47.

[Pair-level intent changes, failures, diversity and decision](transfer/results.json) · [Frozen transfer fixtures and tolerances](transfer/manifest.json)

Short-pilot variability calibrates intent tolerances; duration, silence and diversity bounds are declared engineering limits, not estimates of long-song population variability.

Decision: **inconclusive_or_failed_transfer**.

See the machine-readable decision for all statistical results.

[Decision details](decision.json)

## Failed observations

| Observation | Error |
|---|---|
| transfer-00-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-00-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-00-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-00-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-01-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-01-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-01-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-01-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-02-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-02-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-02-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-02-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-03-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-03-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-03-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-03-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-04-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-04-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-04-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-04-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-05-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-05-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-05-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-05-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-06-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-06-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-06-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-06-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-07-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-07-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-07-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-07-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-08-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-08-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-08-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-08-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-09-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-09-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-09-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-09-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-10-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-10-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-10-s6607-lora | Full song hit cap or ended before half its intended duration |
| transfer-10-s6607-off | Full song hit cap or ended before half its intended duration |
| transfer-11-s5501-lora | Full song hit cap or ended before half its intended duration |
| transfer-11-s5501-off | Full song hit cap or ended before half its intended duration |
| transfer-11-s6607-off | Full song hit cap or ended before half its intended duration |

[Implementation test record](audit/tests.json) · [Test details](audit/unit-tests.xml)

[Final artifact and protocol integrity audit](audit/final-integrity.json)

The studio uses GPU 0 and the investigation uses physical GPU 1. Production slider weights and registry settings are not changed by this research pipeline.

Run or resume the frozen investigation with:

```bash
systemctl --user status music-reward-ce-v1-20260907.service
journalctl --user -u music-reward-ce-v1-20260907.service -n 30 --no-pager
```
