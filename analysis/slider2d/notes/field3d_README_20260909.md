# Field3D scaffold (runnable)

Module: `analysis/slider2d/field3d.py` (imported by `gan.py`).

```bash
# Unit tests (CPU, short fits)
PYTHONPATH=. python -m pytest tests/test_field3d.py -q

# Leftover smoke @ locked recipe (6 seeds)
PYTHONPATH=. python analysis/slider2d/notes/field3d_leftover_smoke_20260909.py

# PairField exam cells
PYTHONPATH=. python analysis/slider2d/notes/field3d_pairfield_exam_20260909.py

# Harder multipair / mismatch (Fire #13)
PYTHONPATH=. python analysis/slider2d/notes/field3d_hard_multipair_20260909.py

# Span × entanglement diagnostics (Fire #14)
PYTHONPATH=. python analysis/slider2d/notes/span_entangle_diag_20260909.py
```

Locked recipe: 1200 steps, cover_weight=1.5, `faithful_guard_e`, FM off,
n_particles≤12, particle_l2=0.02, b_cap=1. Do not adopt 800×cover3.0.

Transfer map: `music_lm_transfer_map_20260909.md`.
