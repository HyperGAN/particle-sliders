# Experiment record

The repeatable game is implemented. Research stopped at the user’s request after the first refined-block preservation check completed. No broadly improving incumbent is confirmed. The benchmark is exposed development material; every candidate uses one fixed checkpoint and multiplier throughout its scorecard.

All rows retain the same original 16 cases, physical GPU assignments, controls, CE procedure and fixed 4/8/16 gates. Acoustic and joint candidates have separately audited native host bridges and their own game ledgers. Partial-stage results remain partial. Off stays eligible.

| Attempt | Method | Local updates | New training clips | New development clips | Final development result vs Off | Decision |
|---|---|---:|---:|---:|---|---|
| `f03e6c5c5fa63a5d` | Supplied gentle preference continuation | 0 | 0 | 8 | 5/8, mean +0.319456, worst −0.330153 | Rejected on wins |
| `4f0bd6298d0067e4` | Cached original at half strength | 0 | 0 | 0 | 6/8, mean +0.209659, worst −0.548287 | Rejected on loss floor |
| `b4c790ef8fddfce6` | Semantic positive imitation, merged numerical forward | 10 | 0 | 4 | 2/4, mean −0.041023, worst −1.050335 | Early rejected |
| `5a605ee341b7872e` | Direct CE search over two depth gain patterns | 0 | 8 | 0 | No development submission | Training objective selected Off |
| `a0d34ac693e0ea02` | Semantic plus residual positive imitation | 10 | 0 | 16 | 11/16, mean +0.075333, worst −0.587728 | Rejected |
| `d0986266be642e9a` | Matched parent/Off positive targets | 10 | 8 | 4 | 2/4, mean −0.308346, worst −2.969716 | Early rejected |
| `2fef1e8749204c12` | Direct CE search over two projection gain patterns | 0 | 8 | 0 | No development submission | Training objective selected Off; gain grid closed |
| `d21fc4af9b9a0b74` | Signed clipped semantic/residual policy update | 10 | 0 | 16 | 12/16, mean +0.134689, worst −0.531847 | Rejected on wins and loss floor |
| `01894958f3db39c0` | Full-batch signed policy update | 10 | 0 | 4 | 2/4, mean +0.162187, worst −0.237593 | Early rejected; signed training pilots closed |
| `cd23c91b67d1e7b7` | Fresh acoustic attention LoRA, truncated direct CE gradient | 8 | 0 | 16 | 16/16 meaningful wins, mean +0.095898, smallest gain +0.024003 | Rejected on mean |
| `fc4e20db06661f0a` | One acoustic calibration folded into ordinary factors | 0 | 0 | 16 | 16/16 meaningful wins, mean +0.136709, smallest gain +0.021300 | Development passed; fresh confirmation failed |
| `681bbec245e13053` | Fixed pair of signed LM policy and calibrated acoustic LoRA | 0 | 0 | 16 | 14/16 meaningful wins, mean +0.218012, worst −0.409758 | Development passed; fresh confirmation failed |
| `708e99657d4674b2` | Fresh acoustic feed-forward LoRA, truncated direct CE gradient | 8 | 0 | 16 | 16/16 wins (15 meaningful), mean +0.053453, smallest gain +0.010612 | Rejected on mean |
| `ae49598815a2c0df` | Fixed attention gain150 plus FF step-eight factors on disjoint projections | 0 | 0 | 16 | 16/16 meaningful wins, mean +0.167911, smallest gain +0.032461 | Development passed; fresh confirmation failed |
| `96696ec3f6d8fa72` | Fresh attention LoRA with complete within-chunk flow gradients | 2 | 0 | 0 | No development submission | Stopped at the predeclared latent-drift limit before update three |
| `6867a1c81f47121c` | CE latent targets distilled into transported velocity targets | 80 | 32 derived target trials | 4 | 1/4 wins, no meaningful wins, mean −0.003362, worst −0.023046 | Early rejected |
| `1a6369e04e9c82ca` | Joint refinement of the fixed block parent with balanced regression-sensitive CE | 2 | 8 parent captures | 16 | 16/16 meaningful wins, mean +0.190882, smallest gain +0.057367 | Development and first fresh CE passed; preservation failed; user stopped search |

Local updates exclude inherited parent training. The latest row includes the completed September 9 development result; use [live progress](http://192.168.1.90:8888/reward-game-current/) for fresh confirmation progress. Its initial graph was resource-aborted before a complete case or update. Explicit recomputation and runtime amendments preserve the scientific recipe and resume saved partial gradients. They do not represent additional scientific candidates. See [operational state and exact execution identities](RUNNING.md).

The original full-batch signed-policy row performs 80 rollout forward/backward evaluations for ten optimizer steps. Direct-search rows score fixed proposals on training families and do not claim optimizer training. The initial acoustic, feed-forward and current robust derivatives cover the final two flow steps of each chunk, holding earlier trajectory states and overlap fixed. The full-flow attempt covers all thirty steps within each chunk, still holding conditioning and reference overlap fixed. Ordinary development evaluation runs the complete generation process.

The full-flow attempt preserved two actual updates and stopped at its fixed 0.1 relative latent-drift limit. The source did not log the triggering value, so that value is unknown. Exact Off restoration passed; no candidate was submitted to development. Its earlier cache-gradient parity failure and memory-aborted graph are retained separately.

The latent-target attempt made 32 latent-ascent steps on eight existing Off training examples. These are separate from its 80 candidate optimizer updates. Selected targets averaged +0.615224 training CE with maximum relative latent L2 0.016252, but these gains did not transfer to ordinary adapter renders. All 32 derived target recordings and all four rejected development clips remain available. Export, native merging and frozen-base checks passed; those integrity checks do not establish better music.

## Best development and fresh confirmation

The refined block candidate leads the fixed development ranking with 16/16 meaningful Off wins, mean +0.190882 and minimum +0.057367. Against the original it has 11 wins and mean +0.195210. The preceding block parent had 16/16 meaningful Off wins, mean +0.167911 and minimum +0.032461. Joint-v1 has the largest development mean (+0.218012), with 14/16 Off wins and a worst loss of −0.409758. Consistency and worst loss rank ahead of mean under this game's fixed rule. [Current comparison table](CURRENT.md).

| First fresh batch | Valid pairs | Off wins | Mean vs Off | 95% whole-family interval vs Off | Original wins | Mean vs original | 95% interval vs original | Decision |
|---|---:|---:|---:|---|---:|---:|---|---|
| Acoustic gain150 | 16/16 | 16 | +0.084564 | [0.067051, 0.108237] | 8 | +0.114078 | [−0.133661, 0.355353] | Failed Off mean, original wins and original uncertainty |
| Joint | 15/16 | 11 | +0.078545 observed | Incomplete; no interval | 12 | +0.184946 observed | Incomplete; no interval | One short candidate; invalid batch and insufficient observed Off gains |
| Block | 16/16 | 15 | +0.087521 | [0.036343, 0.136969] | 11 | +0.140558 | [−0.026153, 0.314670] | Failed Off mean and original uncertainty |
| Refined block | 16/16 | 15 | +0.110204 | [0.080603, 0.140382] | 12 | +0.284957 | [0.176389, 0.390547] | First fresh CE passed; preservation failed |

These batches contain 48 recordings each and use different families and seeds. Their means are not matched candidate comparisons. The joint invalid candidate ended at 17.415 seconds. The block's worst Off delta was −0.215814. Every failed batch is exposed and remains in the ledger. All four batches restored Off exactly. Refined block reached preservation, which failed; independent replication and composition were not run.

The refined candidate is frozen as `robust-block-first-v1`, using Bank B and seeds 196613/262147. Its 48 fresh recordings completed, all valid, and passed the practical and uncertainty gates. The protocol was frozen before audio and checks exposure across all three previous failed batches. Its sole Off loss was −0.005768; all eight family-average gains are positive against both controls. All 48 preservation measurements completed with the locked CLAP and Whisper models. Baseline tolerances were frozen before candidate measurements. One dynamics check failed against Off, and twelve comparisons had preservation issues against the original. Bank C remains unused. The user requested stopping after this check, and no new experiment, replication or composition was launched.

Any new development winner still needs a separately frozen first fresh batch, its independent preservation checks, an unchanged-candidate replication on new families/seeds, the replication's preservation checks, and the fixed per-host-energy composition check. No production default has changed. The research scope is the frozen first 20 seconds; natural full-song completion remains unestablished.

## Audio and costs

Playback is repaired for [joint development](http://192.168.1.90:8888/joint-v1/), [acoustic development](http://192.168.1.90:8888/acoustic-v1/), [block development](http://192.168.1.90:8888/block-v1/) and [feed-forward development](http://192.168.1.90:8888/ff-v1/). Failed fresh recordings remain available for [acoustic](http://192.168.1.90:8888/acoustic-fresh-v1/), [joint](http://192.168.1.90:8888/joint-fresh-v1/) and [block](http://192.168.1.90:8888/block-fresh-v1/). Players retain raw WAV bytes and include regressions.

After refined-block first fresh CE: **430 new recordings** = 152 development + 64 training (including 32 derived latent-target trials) + 22 duplicate engineering + 192 fresh recordings (144 from failed batches, 48 from the current CE pass). Candidate optimizer updates: **150**, plus **5 discarded actual-host audit updates** and **32 separate latent-ascent steps**. Completed candidate update-wall segments total 24439.380359 seconds. Fresh confirmation work is added as recordings complete. These segments are not GPU utilization and do not constitute complete wall-time accounting. [Rebuildable costs](audit/cumulative-cost-current.json).

All prior progress prose is preserved in [the archived experiment record](audit/EXPERIMENTS-before-current-table-v1.md). Exact hypotheses, recipes, source hashes, full states, observations and decisions remain under their attempt and training directories. [CLI documentation](README.md) and [current operational state](RUNNING.md) are maintained separately.

The [selected listening page](http://192.168.1.90:8888/reward-game-best/) presents the refined block slider with all16fresh comparisons and its limitations. It is a recommendation for listening, not a confirmed production default. Stop records and STOP markers exist in the root and all four game homes; the listener and read-only overview remain available.
