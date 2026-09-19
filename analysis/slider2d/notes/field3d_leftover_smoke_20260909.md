# Field3D leftover smoke — 2026-09-09 (Fire #9 / 2D→3D)

Locked recipe: steps=1200, cover_weight=1.5, teacher=faithful_guard_e,
fm_weight=0, n_particles=12, particle_l2=0.02, b_cap=1.

Geometry: orthonormal u-hat (concept), c-hat (content/intended-off-u), e-hat (leftover),
plus one lyric row -> dim=4. Poles h+/- = neu +/- a (no common s-hat).

Seeds: [0, 1, 2, 3, 7, 42].

| seed | pass | u_kept | content_kept | leak_ratio | resid | collapse | s |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | PASS | 0.9921 | 0.9953 | 0.0002 | 1.133 | -1.000 | 6.6 |
| 1 | PASS | 0.9909 | 0.9948 | 0.0009 | 1.132 | -1.000 | 6.0 |
| 2 | PASS | 0.9914 | 0.9938 | 0.0002 | 1.132 | -1.000 | 6.1 |
| 3 | PASS | 0.9912 | 0.9942 | 0.0001 | 1.132 | -1.000 | 6.2 |
| 7 | PASS | 0.9924 | 0.9954 | 0.0004 | 1.133 | -1.000 | 6.2 |
| 42 | PASS | 0.9924 | 0.9955 | 0.0000 | 1.133 | -1.000 | 6.3 |

Summary: **6/6 PASS**; u_kept mean=0.9917 span=0.0015; content_kept mean=0.9948;
leak_ratio mean=0.0003 max=0.0009; wall=37.4s.

### Finding

- **LOCKED RECIPE TRANSFERS** to R³ leftover: û+content recovered, leftover ê gated.
- Gate fails: u=0 content=0 leak=0 (of 6).
- Verdict: `field3d_leftover_transfers`
- Next: harden Field3D gates / exam-port rollout on same geometry, or stress content vs leftover entanglement
