# First automatic audio comparison

Four fixed prompt/seed examples per candidate. The overall heuristic is provisional; individual listening preferences may differ.

| Candidate | Heuristic score | Concept contribution | Enjoyment contribution | Production contribution | Lyric contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| 600 original | 0.953 | +0.895 | +0.098 | +0.050 | -0.090 |
| 750 hold + limit | 0.896 | +0.939 | +0.085 | -0.010 | -0.118 |
| 750 limit only | 0.697 | +0.901 | -0.012 | -0.103 | -0.088 |
| 900 original update | 0.656 | +0.907 | +0.084 | -0.067 | -0.268 |
| 900 limit only | 0.603 | +0.825 | -0.007 | -0.138 | -0.078 |
| 900 hold + limit | -0.005 | +0.651 | -0.140 | -0.239 | -0.276 |

600 leads this first comparison. The 750 hold-plus-limit candidate is close and has slightly stronger average concept contribution. Step-limit-only 750/900 retain a strong measured concept change but score lower on predicted production quality. No candidate was near-silent or automatically declared broken.

The controller continues the step-limit-only trajectory in 150-update blocks to check whether its score recovers or improves. This report is the initial comparison, not the final duration suggestion.

[Raw components and individual examples](pilot-initial.json). [Component plot](pilot-initial-components.png).
