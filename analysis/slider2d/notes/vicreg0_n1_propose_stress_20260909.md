# vicreg0 @ n=1 propose stress — 2026-09-09

**Host:** box-cpu @ `435e87363bf1`. CPU harness only. **No Music train.**
**Locked defaults UNCHANGED** (vicreg_weight stays 0.05 in AdvConfig).

## Recommendation: **ADOPT** (Music-posture n=1 only)

Adopt as **Music-posture scoring harness** conditional only:

```python
# Music-posture harness — NOT locked AdvConfig default
if n_particles <= 1:
    cfg = replace(cfg, vicreg_weight=0.0)  # VICReg std ill-posed at n=1
```

Music-posture scoring harness only: if n_particles<=1: set vicreg_weight=0. Do NOT change locked default vicreg_weight=0.05. Do NOT silence multi-seed gate. Fire #21 n>=2 remains primary harden.

### Warnings (non-blocking if ADOPT)

- M22 knife clears under vic0 (4/6→6/6) — soft false-lock risk; document as posture side-effect, not hard bite

---

## Evidence table

| cell | posture | vic=0.05 | vic=0.0 | note |
|---|---|:---:|:---:|---|
| close n1 | Music c1.0 | 19/21 fail=[0, 4] | **21/21** | primary Music knife |
| close_live n1 | Music c1.0 | 20/21 fail=[0] | **21/21** | primary Music knife |
| close n1 | propose c1.5 prior | 20/21 | **21/21** | prior 21-seed cite |
| close_live n1 | propose c1.5 prior | 20/21 | **21/21** | prior 21-seed cite |
| close n1 | propose c1.5 reconfirm | 6/7 fail=[0] | **7/7** | seeds [0, 1, 2, 3, 4, 7, 42] |
| close_live n1 | propose c1.5 reconfirm | 6/7 fail=[0] | **7/7** | seeds [0, 1, 2, 3, 4, 7, 42] |
| leftover n12 | locked c1.5 | 6/6@0.9303 | 6/6@0.9303 | flat=True |
| leftover n1 | Music c1.0 | 6/6 | 6/6 | must stay green |
| dual-arm positive | c1.5 n1 | 3/3 | 3/3 | must PASS |
| dual-arm positive | Music c1.0 | — | 3/3 | must PASS |
| dual leftover_only | n1 vic0 | — | 0/3 | must FAIL |
| dual listen_only | n1 vic0 | — | 0/3 | must FAIL |
| M22 stagger_mild | Music n1 | 4/6 | 6/6 | false_lock_risk=True |
| M23 multipair | Music n1 | 6/6 | 6/6 | must stay green |
| M20_amp_lie HARD_BITE | Music n1 | 0/6 | 0/6 | still_fail=True false_fix=False |
| M21_hold_e HARD_BITE | Music n1 | 0/6 | 0/6 | still_fail=True false_fix=False |
| M24_content_leak_flip HARD_BITE | Music n1 | 0/6 | 0/6 | still_fail=True false_fix=False |

---

## Gate checklist

| gate | result |
|---|---|
| Replicate c1.5 close/live 21/21 under vic0 | YES (prior close 21/21 + live 21/21; reconfirm 7/7 + 7/7) |
| Music c1.0 close/live 21/21 under vic0 | YES (close 21/21 + live 21/21) |
| leftover flat / no regression | YES |
| dual-arm positive PASS / arms FAIL | YES |
| M22 no hard new fail | YES (soft clear risk=True) |
| M23 no new fail | YES |
| M20/M21/M24 still FAIL under vic0 | YES |
| Locked defaults changed? | **NO** |

## Relation to Fire #21

- **n≥2** remains primary harden for close-family (VICReg well-posed).
- **vic=0 @ n=1** is complementary Music parts0 posture (ill-posed VICReg term).
- Multi-seed gate still mandatory under parts0 even if vic0 adopted.

Wall: 1168.8s. JSON: `vicreg0_n1_propose_stress_20260909.json`.
