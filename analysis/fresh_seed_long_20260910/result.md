# Longer fresh-seed Lo-fi results

Status: Complete.

All four monitoring cases receive equal weight. The reference is fresh step 660.

| Step | Mean CE | CE change vs 660 | Wins | Production quality | Lyric agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| 660 | 7.796379 | +0.000000 | 0/4 | 8.145867 | 0.860838 |
| 800 | 7.557598 | -0.238781 | 0/4 | 8.248975 | 0.873661 |
| 1000 | 7.639942 | -0.156436 | 1/4 | 8.213364 | 0.831554 |

## Step 800: every paired CE difference

- Arrangement 1, seed 101: -0.072769
- Arrangement 1, seed 303: -0.276301
- Arrangement 2, seed 101: -0.332442
- Arrangement 2, seed 303: -0.273612

## Step 1000: every paired CE difference

- Arrangement 1, seed 101: -0.128308
- Arrangement 1, seed 303: -0.110752
- Arrangement 2, seed 101: +0.003217
- Arrangement 2, seed 303: -0.389902

These reused monitoring cases do not independently confirm checkpoint selection.
One training prompt and one continued trajectory; style teachers remain fixed.

[Listen](http://100.90.104.57:8888/fresh-seed-long-lofi-20260910/)
