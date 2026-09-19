# Field3D exam-entangle — 2026-09-09 (Fire #11 / 2D→3D)

Locked recipe: steps=1200, cover_weight=1.5, teacher=faithful_guard_e,
fm_weight=0, n_particles=12, particle_l2=0.02, b_cap=1.

Ask: does leftover gating stay seed-stable when content vs leftover
amplitudes vary (exam-port analogues), or does content↔ê entanglement
create false locks / leak failures?

Seeds: [0, 1, 2, 3, 7, 42].

## Per-family summary

| family | content | leak | PASS | u_kept mean | content_kept mean | leak_ratio max | fail u/c/leak |
|:---|---:|---:|:---:|---:|---:|---:|:---:|
| baseline | 0.55 | 0.45 | 6/6 | 0.9917 | 0.9948 | 0.0009 | 0/0/0 |
| close_like | 0.90 | 0.10 | 6/6 | 0.9929 | 0.9937 | 0.0009 | 0/0/0 |
| unused_e_like | 0.10 | 0.90 | 6/6 | 0.9919 | 1.0016 | 0.0003 | 0/0/0 |
| entangled | 0.70 | 0.70 | 6/6 | 0.9925 | 0.9947 | 0.0004 | 0/0/0 |
| content_zero | 0.00 | 0.45 | 6/6 | 0.9920 | n/a | 0.0002 | 0/0/0 |
| leak_zero | 0.55 | 0.00 | 6/6 | 0.9917 | 0.9948 | 0.0009 | 0/0/0 |

## Detail tables

### baseline (content=0.55 leak=0.45)

| seed | pass | u_kept | content_kept | leak_ratio | resid | collapse | pair_odd_cos | s |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | PASS | 0.9921 | 0.9953 | 0.0002 | 1.133 | -1.000 | 0.9304 | 6.8 |
| 1 | PASS | 0.9909 | 0.9948 | 0.0009 | 1.132 | -1.000 | 0.9306 | 6.3 |
| 2 | PASS | 0.9914 | 0.9938 | 0.0002 | 1.132 | -1.000 | 0.9302 | 6.2 |
| 3 | PASS | 0.9912 | 0.9942 | 0.0001 | 1.132 | -1.000 | 0.9303 | 6.2 |
| 7 | PASS | 0.9924 | 0.9954 | 0.0004 | 1.133 | -1.000 | 0.9302 | 6.2 |
| 42 | PASS | 0.9924 | 0.9955 | 0.0000 | 1.133 | -1.000 | 0.9303 | 6.2 |

### close_like (content=0.90 leak=0.10)

| seed | pass | u_kept | content_kept | leak_ratio | resid | collapse | pair_odd_cos | s |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | PASS | 0.9925 | 0.9933 | 0.0004 | 1.336 | -1.000 | 0.9973 | 6.2 |
| 1 | PASS | 0.9929 | 0.9938 | 0.0006 | 1.336 | -1.000 | 0.9972 | 6.3 |
| 2 | PASS | 0.9934 | 0.9939 | 0.0009 | 1.337 | -1.000 | 0.9972 | 6.2 |
| 3 | PASS | 0.9922 | 0.9933 | 0.0001 | 1.336 | -1.000 | 0.9973 | 6.2 |
| 7 | PASS | 0.9935 | 0.9939 | 0.0001 | 1.337 | -1.000 | 0.9972 | 6.2 |
| 42 | PASS | 0.9928 | 0.9942 | 0.0000 | 1.336 | -1.000 | 0.9973 | 6.2 |

### unused_e_like (content=0.10 leak=0.90)

| seed | pass | u_kept | content_kept | leak_ratio | resid | collapse | pair_odd_cos | s |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | PASS | 0.9914 | 0.9976 | 0.0003 | 0.996 | -1.000 | 0.7451 | 6.1 |
| 1 | PASS | 0.9920 | 1.0058 | 0.0002 | 0.997 | -1.000 | 0.7448 | 6.0 |
| 2 | PASS | 0.9926 | 1.0028 | 0.0001 | 0.998 | -1.000 | 0.7449 | 6.1 |
| 3 | PASS | 0.9905 | 0.9972 | 0.0000 | 0.996 | -1.000 | 0.7450 | 6.1 |
| 7 | PASS | 0.9926 | 1.0020 | 0.0001 | 0.998 | -1.000 | 0.7450 | 6.2 |
| 42 | PASS | 0.9923 | 1.0043 | 0.0003 | 0.997 | -1.000 | 0.7448 | 6.1 |

### entangled (content=0.70 leak=0.70)

| seed | pass | u_kept | content_kept | leak_ratio | resid | collapse | pair_odd_cos | s |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | PASS | 0.9922 | 0.9945 | 0.0004 | 1.212 | -1.000 | 0.8676 | 6.2 |
| 1 | PASS | 0.9928 | 0.9942 | 0.0004 | 1.212 | -1.000 | 0.8677 | 6.0 |
| 2 | PASS | 0.9928 | 0.9959 | 0.0000 | 1.213 | -1.000 | 0.8675 | 6.0 |
| 3 | PASS | 0.9917 | 0.9940 | 0.0002 | 1.211 | -1.000 | 0.8676 | 5.8 |
| 7 | PASS | 0.9931 | 0.9949 | 0.0000 | 1.213 | -1.000 | 0.8675 | 6.0 |
| 42 | PASS | 0.9924 | 0.9948 | 0.0000 | 1.212 | -1.000 | 0.8675 | 6.0 |

### content_zero (content=0.00 leak=0.45)

| seed | pass | u_kept | content_kept | leak_ratio | resid | collapse | pair_odd_cos | s |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | PASS | 0.9918 | 0.0000 | 0.0002 | 0.992 | -1.000 | 0.9120 | 6.1 |
| 1 | PASS | 0.9921 | 0.0000 | 0.0001 | 0.992 | -1.000 | 0.9119 | 6.2 |
| 2 | PASS | 0.9916 | 0.0000 | 0.0001 | 0.992 | -1.000 | 0.9119 | 6.1 |
| 3 | PASS | 0.9911 | 0.0000 | 0.0002 | 0.991 | -1.000 | 0.9118 | 6.0 |
| 7 | PASS | 0.9929 | 0.0000 | 0.0001 | 0.993 | -1.000 | 0.9119 | 6.1 |
| 42 | PASS | 0.9926 | 0.0000 | 0.0001 | 0.993 | -1.000 | 0.9119 | 6.1 |

### leak_zero (content=0.55 leak=0.00)

| seed | pass | u_kept | content_kept | leak_ratio | resid | collapse | pair_odd_cos | s |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | PASS | 0.9921 | 0.9953 | 0.0002 | 1.133 | -1.000 | 1.0000 | 6.0 |
| 1 | PASS | 0.9909 | 0.9948 | 0.0009 | 1.132 | -1.000 | 1.0000 | 6.0 |
| 2 | PASS | 0.9914 | 0.9938 | 0.0002 | 1.132 | -1.000 | 1.0000 | 6.1 |
| 3 | PASS | 0.9912 | 0.9942 | 0.0001 | 1.132 | -1.000 | 1.0000 | 6.1 |
| 7 | PASS | 0.9924 | 0.9954 | 0.0004 | 1.133 | -1.000 | 1.0000 | 6.0 |
| 42 | PASS | 0.9924 | 0.9955 | 0.0000 | 1.133 | -1.000 | 1.0000 | 6.0 |

## exam_proxy (per seed)

Mean u_kept over families with content>0; max leak_ratio across all families.
Not a transferred scoreboard gate — diagnostic only.

| seed | u_kept_mean (content fams) | leak_ratio max | PASS fams |
|---:|---:|---:|:---:|
| 0 | 0.9920 | 0.0004 | 6/6 |
| 1 | 0.9919 | 0.0009 | 6/6 |
| 2 | 0.9924 | 0.0009 | 6/6 |
| 3 | 0.9914 | 0.0002 | 6/6 |
| 7 | 0.9928 | 0.0004 | 6/6 |
| 42 | 0.9925 | 0.0003 | 6/6 |

### Finding

- Locked leftover gating stays seed-stable across exam-port amplitude mixes (baseline/close/unused_e/entangled + content_zero/leak_zero). No false locks.
- content_zero: content gate intentionally skipped (a_c≈0).
- leak_zero: declared_e → None; leak_dir=None path.
- Verdict: `field3d_exam_entangle_solid`
- Next: port a true PairField-style exam cell into R3 or multi-row Field3D
- Wall: 220.9s
