# Field3D PairField exam — 2026-09-09 (Fire #12 / 2D→3D)

Locked recipe: steps=1200, cover_weight=1.5, teacher=faithful_guard_e,
fm_weight=0, n_particles=12, particle_l2=0.02, b_cap=1.

Ask: port a true PairField-style exam cell into multi-row Field3D (divergent / close / unused_e) using the locked adversarial core.

Seeds: [0, 1, 2, 3, 7, 42]. Stage: full.

## Per-cell summary

| cell | PASS | u_kept mean | content_kept mean | leak_ratio max | exam_score mean | exam_score min | fail c/s/l/m |
|:---|:---:|---:|---:|---:|---:|---:|:---:|
| divergent | 6/6 | 0.9894 | 0.9216 | 0.0011 | 0.9216 | 0.9192 | 0/0/0/0 |
| close | 6/6 | 0.9987 | 0.9828 | 0.0097 | 0.9828 | 0.9783 | 0/0/0/0 |
| unused_e | 6/6 | 0.9826 | 0.9959 | 0.0017 | 0.9203 | 0.9198 | 0/0/0/0 |

## Detail tables

### divergent

| seed | exam_pass | u_kept | content_kept | leak_ratio | exam_cont | exam_swing | exam_score | rows | s |
|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 0 | PASS | 0.9878 | 0.9192 | 0.0005 | 0.9192 | 0.9996 | 0.9192 | 3/3 | 9.9 |
| 1 | PASS | 0.9881 | 0.9205 | 0.0006 | 0.9205 | 0.9996 | 0.9205 | 3/3 | 8.6 |
| 2 | PASS | 0.9912 | 0.9192 | 0.0011 | 0.9192 | 0.9996 | 0.9192 | 3/3 | 8.6 |
| 3 | PASS | 0.9913 | 0.9257 | 0.0005 | 0.9257 | 0.9997 | 0.9257 | 3/3 | 8.4 |
| 7 | PASS | 0.9868 | 0.9201 | 0.0004 | 0.9201 | 0.9996 | 0.9201 | 3/3 | 8.1 |
| 42 | PASS | 0.9914 | 0.9250 | 0.0000 | 0.9250 | 0.9996 | 0.9250 | 3/3 | 8.4 |

### close

| seed | exam_pass | u_kept | content_kept | leak_ratio | exam_cont | exam_swing | exam_score | rows | s |
|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 0 | PASS | 0.9890 | 0.9783 | 0.0063 | 0.9783 | 1.0000 | 0.9783 | 3/3 | 8.7 |
| 1 | PASS | 1.0014 | 0.9846 | 0.0015 | 0.9846 | 1.0000 | 0.9846 | 3/3 | 8.3 |
| 2 | PASS | 1.0106 | 0.9877 | 0.0097 | 0.9877 | 1.0000 | 0.9877 | 3/3 | 8.3 |
| 3 | PASS | 0.9975 | 0.9843 | 0.0043 | 0.9843 | 1.0000 | 0.9843 | 3/3 | 9.1 |
| 7 | PASS | 0.9911 | 0.9808 | 0.0016 | 0.9808 | 1.0000 | 0.9808 | 3/3 | 14.3 |
| 42 | PASS | 1.0025 | 0.9807 | 0.0009 | 0.9807 | 1.0000 | 0.9807 | 3/3 | 11.2 |

### unused_e

| seed | exam_pass | u_kept | content_kept | leak_ratio | exam_cont | exam_swing | exam_score | rows | s |
|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 0 | PASS | 0.9824 | 0.9933 | 0.0010 | 0.9824 | 0.9200 | 0.9200 | 3/3 | 14.9 |
| 1 | PASS | 0.9862 | 1.0000 | 0.0006 | 0.9862 | 0.9206 | 0.9206 | 3/3 | 13.9 |
| 2 | PASS | 0.9858 | 1.0064 | 0.0001 | 0.9858 | 0.9204 | 0.9204 | 3/3 | 9.0 |
| 3 | PASS | 0.9813 | 0.9902 | 0.0017 | 0.9813 | 0.9198 | 0.9198 | 3/3 | 9.2 |
| 7 | PASS | 0.9792 | 0.9902 | 0.0001 | 0.9792 | 0.9204 | 0.9204 | 3/3 | 9.2 |
| 42 | PASS | 0.9807 | 0.9951 | 0.0001 | 0.9807 | 0.9205 | 0.9205 | 3/3 | 17.6 |

### Finding

- Locked recipe + multi-row PairField cells (divergent/close/unused_e) seed-stable on Field3D. Guard refuses on divergent; unused_e gates ê; close keeps small-û + content. Multi-row coverage holds.
- Verdict: `field3d_pairfield_exam_solid`
- Next: phase milestone: PairField exam cells transfer to multi-row R³; next highd/live bridge
- Wall: 185.6s
