# Do the saved metrics track the listener?

Partly, but none is validated as a reliable selection rule or causal explanation. This audit uses saved measurements only.

| Metric | Merger pairwise agreement | Preferred clip picked | Earlier energy pairwise agreement |
|---|---:|---:|---:|
| Content Enjoyment | 13/18 | 3/6 | 4/9 |
| Production Quality | 12/18 | 3/6 | 5/9 |
| Production Complexity (exploratory: higher) | 12/18 | 3/6 | 8/9 |
| Content Usefulness | 9/18 | 2/6 | 4/9 |
| ASR phrase match | 12.5/18 | 3.5/6 | 4/9 |
| ASR lyric recall | 12/18 | 2/6 | 4/9 |
| ASR lyric precision | 11/18 | 2.66667/6 | 6/9 |
| First concept CLAP margin | 9/18 | 2/6 | 8/9 |
| Second concept CLAP margin | 10/18 | 3/6 | 6/9 |
| Mean of two CLAP margins (exploratory) | 11/18 | 3/6 | 7/9 |
| Minimum of two CLAP margins (exploratory) | 11/18 | 3/6 | 7/9 |
| Update cosine to ordinary (static method order) | 12/18 | 3/6 | Not applicable |

Tied predictions receive half pairwise credit and split top-choice credit. A ranking of three clips creates three comparisons; six rankings are not eighteen independent listening tests.

Content Enjoyment (CE) gives the largest aggregate agreement among these saved single metrics: 13/18, but selects the listener’s favorite in only 3/6 groups. A fixed ordinary > KnOTS-TIES > TIES order gives 12/18 and also selects 3/6 favorites. Update cosine produces exactly that fixed order here; it contributes no song-dependent selection.

Excluding the uncertain female + pop confirmation group gives CE 11/15 and the fixed method order 12/15. Excluding the close country confirmation top comparison as well gives CE 10/14 and the fixed method order 11/14. CE gets only 4/9 comparisons and 1/3 favorites right in the earlier energy study. Production Complexity gets 8/9 in that energy study and 12/18 here, making it another candidate to evaluate prospectively. Complexity is not inherently better, and these two studies share clips. No fitted blend of metrics is justified by these few labels.

## What matches, and what fails

- Female + pop: CE and PQ both put KnOTS-TIES above ordinary on both songs, agreeing with the listener on that comparison. Neither chooses TIES as the confirmation favorite, where the listener was uncertain.
- Country + indie rock: both concept margins recover the entire first-song ranking. On confirmation, they put ordinary first but incorrectly put KnOTS-TIES above TIES. CE recovers the full confirmation ranking and its small top-score gap, but puts ordinary last on the first song.
- House + acoustic folk: CE and PQ recover the first-song order. On confirmation, both make errors below or above ordinary; neither recovers the full ranking.

## Weight geometry

Source-update cosines are approximately 0.0127 for female + pop, 0.0612 for country + indie rock, and 0.0434 for house + acoustic folk. That is a possible association to investigate, not a validated explanation from only three pairs. These values are fixed across songs, while the house preference reverses. All merger conditions also match each projection’s update norm, so Frobenius energy cannot distinguish them. Raw weight cosine is not the same as activation alignment.

## Implication

Use CE/PQ, complexity and each requested concept margin as separate diagnostics. A predictive rule needs new preferences on songs and seeds not used to select or tune it, and must beat a simple fixed-method baseline. A mechanistic explanation would additionally need measurements of the model’s behavior on the actual song/seed context; this audit does not establish one.

Definitions: [Audiobox Aesthetics](https://arxiv.org/abs/2502.05139) separates enjoyment from technical production quality. [KnOTS](https://arxiv.org/abs/2410.19735) studies alignment for LoRA merging; its motivation is not proof that raw weight cosine predicts these listening choices.

The earlier energy check shares the first-song fixture and three recordings with this study. It is useful evidence of a failure to transfer across tested settings, not an independent test set. All metrics were measured on the first 20 seconds.
