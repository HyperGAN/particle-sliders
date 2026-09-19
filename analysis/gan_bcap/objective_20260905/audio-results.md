# Matched audio diagnostics

Every first clip is retained. These fixed transcription and audio-model scores are diagnostics on two development seeds; no thresholds were fitted here. CLAP margin is feminine-voice descriptor minus masculine-voice descriptor; it is not a probability. PQ is the frozen production-quality estimate.

| Candidate / control | Seed | ASR precision | Phrase match | Sheet coverage | Voice margin | PQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 7 | 1.000 | 1.000 | 1.000 | -0.1167 | 8.378 |
| positive_reference | 7 | 1.000 | 0.800 | 0.769 | 0.0856 | 8.172 |
| Energy · both branches | 7 | 0.923 | 0.923 | 0.923 | 0.0719 | 8.295 |
| Energy · conditional branch | 7 | 0.708 | 0.625 | 1.000 | 0.0903 | 8.266 |
| Fixed MMD · both branches | 7 | 0.575 | 0.325 | 0.923 | 0.0736 | 8.142 |
| Reference · 600 | 7 | 0.750 | 0.625 | 0.923 | 0.1122 | 8.278 |
| off | 23 | 1.000 | 1.000 | 1.000 | 0.0385 | 8.203 |
| positive_reference | 23 | 1.000 | 1.000 | 1.000 | 0.0302 | 8.360 |
| Energy · both branches | 23 | 0.688 | 0.594 | 1.000 | 0.0840 | 8.341 |
| Energy · conditional branch | 23 | 0.538 | 0.385 | 0.615 | 0.0817 | 8.412 |
| Fixed MMD · both branches | 23 | 0.966 | 0.966 | 1.000 | 0.0743 | 6.948 |
| Reference · 600 | 23 | 0.870 | 0.870 | 0.923 | 0.1211 | 8.326 |

The 20-second cap does not test natural endings or long-form stability. A correct transcript does not certify the intended vocal change, and a higher voice margin does not certify lyric fidelity.

[Listen to all candidates](../../../eval/listen/gan-objective-20260905/index.html)
