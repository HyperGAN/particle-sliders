# Matched audio results for the GAN repairs

Two reserved prompts × four matched seeds. A screening experiment; no convergence or musical-quality certificate.

All five arms completed 60 updates from the original 600 state. All 300 updates passed the finite-gradient and declared-update-bound checks. The main regression suite has 124 passes, with three additional audio-audit tests.

| Candidate | Proxy score | Paired delta vs 600 | Worst clip | Flagged clips | Consensus diversity alarms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 600 | 0.5300 | +0.0000 | -0.0846 | 2/8 | 0/2 |
| Bounded baseline 660 | 0.7957 | +0.2656 | -0.2947 | 1/8 | 0/2 |
| Normalized FM 660 | 0.4550 | -0.0751 | -0.3817 | 2/8 | 0/2 |
| FM gradient limit 660 | 0.7804 | +0.2504 | -0.2322 | 2/8 | 0/2 |
| LR decay 660 | 0.6279 | +0.0978 | -0.8605 | 1/8 | 0/2 |
| Combined repairs 660 | 0.4911 | -0.0389 | -0.1930 | 1/8 | 0/2 |
| Combined repairs · EMA 660 | 0.6301 | +0.1001 | -0.4373 | 1/8 | 0/2 |

The proxy score is the unchanged, predeclared heuristic relative to slider-off. A higher number does not certify better music. Flags are technical/ASR checks; word matching and lyric coverage are exposed separately because slower phrasing can lower coverage. The diversity checks measure near-duplicates in two representations, not known musical modes.

No automatic catalog promotion or convergence claim follows from two prompts. The combined arm changes several safeguards, the critic and the training data together; it cannot isolate the benefit of each component. EMA still has 74% nominal initial-600 mass before compression.

[Listen with matched controls](../../../eval/listen/gan-v2-20260905/compare.html) · [Full measurements](final-audit.json) · [Implementation and validation](README.md)
