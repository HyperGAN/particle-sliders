# YuE2 hip-hop training evidence

Snapshot of `hiphop-yue2-gmix-1600` from the
[September 18 gmix catalog dashboard](http://100.90.104.57:8888/yue2-gmix-catalog-20260918/#job-hiphop).
The run completed 1,600 updates with initialization seed 7. It was selected
as an example of strong teacher alignment at full strength, not as a random
sample or evidence that every catalog run succeeded.

The figures are redrawn from the dashboard's exact JSON series. Training
includes every update without smoothing. The held-out figure shows every
100-update EMA probe at strength +1; the archive also retains the 0.25 and 0.5
probes. No values were invented, interpolated, or filtered out of the archive.

- [Training metrics](training-metrics.json): all 1,600 updates.
- [Probe metrics](probe-metrics.json): all 48 strength/checkpoint combinations,
  plus the original stopping status and selected step.
- [Run provenance](run.json): retrieval time, source URLs, SHA-256 hashes,
  model identity, recipe, initialization seed, and training-source hashes.
- [Plot script](plot.py): generates [training](training.svg) and
  [held-out](heldout.svg) SVGs from these local files using Matplotlib.

Rebuild in an environment with Matplotlib:

```bash
python docs/assets/yue2-gmix-hiphop-20260918/plot.py
```

This experiment uses paired-edit normalization and a global-mix (`gmix`)
critic with 8 tokens, width 48, 1 attention layer, 4 heads and score bound 8.
Four caption templates each have 128 generated continuation seeds with
32 history tokens. Training uses 8 noise-row pairs per D/G minibatch. The
reported noise standard deviation finishes at 1.0; the dashboard's generic
“noise reaches the floor” blurb is not a measurement of this run's actual noise.
See `run.json` and `training-metrics.json` for the recorded configuration and
observed values.

At strength +1, from update 100 to 1600, held-out residual RMS falls from
0.996702 to 0.144063, teacher SWD from 0.663255 to 0.049993, and mean gain rises
from 0.183105 to 0.989067. Final training edit cosine is 0.994790.

The [training evidence section](../../math.md#training-evidence) defines the graph terms.
The [probe implementation](../../../conceptmod/textsliders/distribution_probe.py)
defines the residuals, projection distances and gain. The probe uses fixed
random projections and continuation seeds disjoint from training; it does not
train or reuse a critic to compute the plotted metrics.

The stopping record remains `not_plateaued`: the full-strength match does not
establish calibration at intermediate strengths. At update 1600, strength 0.5
has mean gain 0.777875 rather than 0.5. The independent classifier evaluator
has not been trained (`evaluator_auc: null`). These graphs document hidden-state
learning, not audio quality or a controlled comparison against other recipes.
This run is separate from the public September 17 1,200-update particle release.
