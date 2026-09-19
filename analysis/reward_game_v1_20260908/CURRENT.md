# Current reward-slider evidence

[Live combined results](http://192.168.1.90:8888/reward-game-current/)

**Research stopped at the user’s request after preservation finished. No confirmed improving incumbent.** Off remains eligible.

**Listen to the [selected refined block results](http://192.168.1.90:8888/reward-game-best/).** The page includes all sixteen fresh comparisons, the only Off regression, and the preservation limitations.

**Best fixed-development result: the refined block slider on [block-v1](http://192.168.1.90:8888/block-v1/).** It wins all 16 cases and averages +0.191 CE, with a smallest gain of +0.057. It is also the first candidate to pass a fresh CE batch; preservation failed and independent replication was not run. Joint has the largest development mean; the game prioritizes consistency and worst regression before mean.

| Candidate | Development wins vs Off | Mean CE gain | Worst delta | Development gate |
|---|---:|---:|---:|---|
| [Refined block](block-v1/evaluations/eval-ed9b1ec4311767beabac/scorecard.md) | 16/16 | +0.191 | +0.057 | Pass |
| [Original block](block-v1/evaluations/eval-1d60eebe79ac75da4077/scorecard.md) | 16/16 | +0.168 | +0.032 | Pass |
| [acoustic gain150](acoustic-v1/evaluations/eval-ab2f4a2cc90e440992e6/scorecard.md) | 16/16 | +0.137 | +0.021 | Pass |
| [joint-v1](joint-v1/evaluations/eval-aa34ad592ff039eb295e/scorecard.md) | 14/16 | +0.218 | -0.410 | Pass |
| [ff-v1](ff-v1/evaluations/eval-9f0a9ab616376dc37bf6/scorecard.md) | 16/16 | +0.053 | +0.011 | Reject |

These are the same exposed development cases, with compatibility established by exact Off checks in each renderer version. They are not untouched validation.

| First fresh batch | Valid comparisons | Wins vs Off | Mean gain vs Off | Outcome |
|---|---:|---:|---:|---|
| [Acoustic gain150](http://192.168.1.90:8888/acoustic-fresh-v1/) | 16/16 | 16 | +0.085 | Failed minimum mean gain and original-control evidence |
| [Joint](http://192.168.1.90:8888/joint-fresh-v1/) | 15/16 | 11 | +0.079 observed | One short candidate; batch invalid and observed gains insufficient |
| [Block](http://192.168.1.90:8888/block-fresh-v1/) | 16/16 | 15 | +0.088 | Failed minimum mean gain and original-control uncertainty |
| [Refined block](http://192.168.1.90:8888/robust-block-fresh-v1/) | 16/16 | 15 | +0.110 | CE gates and uncertainty passed; preservation failed |

Each fresh row uses different families and seeds. Their counts and means do not constitute matched candidate comparisons. Every failure and recording is retained. The refined block preservation check finished and failed. Independent replication and studio composition were not run.

The joint and acoustic playback links are repaired. All 64 joint and 80 acoustic URLs passed HTTP checks; downloaded examples matched and decoded as the original raw WAVs. Headless Chrome also decoded and started both tested clips, including the exact reported joint URL, with no media errors. Reload [joint-v1](http://192.168.1.90:8888/joint-v1/) or [acoustic-v1](http://192.168.1.90:8888/acoustic-v1/).

The latest latent-target adapter failed its first development screen: 1/4 wins, no meaningful wins, and −0.0034 mean CE. Its optimized training targets had improved substantially; those gains did not transfer to ordinary adapter renders. The preceding full-flow attempt stopped at its preset latent-drift limit after two saved updates. Both failures and all artifacts are retained.

The refined block candidate jointly updates attention and feed-forward factors from the previous block parent. Two full-batch updates passed export, native merge and exact Off restoration checks, then passed all 16 development cases. It has 11 wins and +0.195 mean CE against the original slider. The fixed candidate passed its 48-recording fresh batch: +0.110204 CE against Off, 95% whole-family interval [0.080603, 0.140382], and +0.284957 against the original, interval [0.176389, 0.390547]. It won 15/16 cases against Off and 12/16 against the original; its sole Off regression was −0.005768. All 48 locked preservation measurements were valid, with tolerances frozen from Off before candidate diagnostics. The candidate failed one crest-factor check against Off (−1.4495 dB versus a 1.3803 dB tolerance) and checks on twelve comparisons against the original, including lyrics, style, voice and dynamics. No gates were weakened. The user requested a stop after this check; independent replication and composition were not started. See [development scorecard](block-v1/evaluations/eval-ed9b1ec4311767beabac/scorecard.md), [frozen protocol](block-v1/confirmation/robust-block-first-v1/protocol.json) and [operational state](RUNNING.md).

After the first refined-block fresh batch: 430 new recordings, including 32 derived latent-target trials; 150 candidate optimizer updates; 32 separate latent-ascent steps; 5 discarded actual-host audit updates. All 16 new development WAVs and all 48 fresh WAVs passed HTTP and raw-byte hash checks. See [cost record](audit/cumulative-cost-current.json).

Natural full-song completion remains separate from this first-20-second CE research. Production registry and generator behavior are unchanged.
