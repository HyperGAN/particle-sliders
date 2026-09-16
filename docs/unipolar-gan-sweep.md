# Unipolar ParticleGAN toy sweep (UniPG-E, GAN-only)

CPU harness: zero-at-zero residual, paired Rp logistic + GradRegularizer,
raw-positive teacher cloud only. **No supervised MSE / cover / FM on G.**
Scale 0 is exact base by construction; scale -1 is a canary (never a gate);
antipodal cos is never consulted. All arms `propose_only` — Music bipolar
(`ARM_B` / `locked_shared` / live `--lm_target`) untouched.

Gates (both cells `divergent` + `close`): cover@+1 >= 0.85, leak@+1 <= 0.05,
neu_hold@0 >= 0.85. PASS = 6/6 (2 cells x seeds 0/1/7) at that budget.

## Budget ladder

| arm | 600 | 1200 | 2400 | 3400 | earliest |
|---|---|---|---|---|---|
| `uni_locked` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `k05` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `k2` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `coeff05` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `coeff2` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `ginterp` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `delayed` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `lazy2` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `slowprior` | 0/6 | 4/6 | 4/6 | 4/6 | — |
| `frozenprior` | 3/6 | 5/6 | **PASS** (6/6) | **PASS** (6/6) | 2400 |
| `n32` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `vic0` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `vicfaithful` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `pl2off` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `d15` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `lr2x` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `beta999` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `emaoff` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `critic128` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `fullclone` | 0/6 | 0/6 | 0/6 | 0/6 | — |
| `frozen_k2` | 3/6 | 5/6 | **PASS** (6/6) | **PASS** (6/6) | 2400 |
| `frozen_gi` | 5/6 | **PASS** (6/6) | **PASS** (6/6) | **PASS** (6/6) | 1200 |
| `frozen_c05` | 2/6 | 5/6 | 5/6 | 5/6 | — |

## Arm cards

- `uni_locked` [MATCH] vicreg_fn=locked delta={} — locked b_cap k=1 coeff=1, n=12, vic=0.05, pl2=0.02, LR 5e-3 shared; GAN-only baseline
- `k05` [MATCH] vicreg_fn=locked delta={"kappa": 0.5} — tighter cap k=0.5 (D flatter, G signal weaker?)
- `k2` [MATCH] vicreg_fn=locked delta={"kappa": 2.0} — looser cap k=2 (D may stay steep longer)
- `coeff05` [MATCH] vicreg_fn=locked delta={"b_cap": 0.5} — half-strength cap coeff=0.5
- `coeff2` [MATCH] vicreg_fn=locked delta={"b_cap": 2.0} — double-strength cap coeff=2
- `ginterp` [MATCH] vicreg_fn=locked delta={"grad_arm": "g_interp_cap"} — interp-path cap (high-dim ParticleGAN geometry)
- `delayed` [MATCH] vicreg_fn=locked delta={"target_anneal": "delayed"} — delayed center anneal (hold k then slide to R1/R2-like)
- `lazy2` [MATCH] vicreg_fn=locked delta={"grad_lazy": 2} — lazy reg every 2nd step (StyleGAN2-style, coeff x2)
- `slowprior` [MATCH] vicreg_fn=locked delta={"prior_lr_mult": 0.1} — slow prior 0.1x (residual must carry the mode)
- `frozenprior` [MATCH] vicreg_fn=locked delta={"prior_lr_mult": 0.01, "vicreg_weight": 0.0} — near-frozen prior 0.01x + VICReg off (pure jitter prior)
- `n32` [MATCH] vicreg_fn=locked delta={"n_particles": 32} — big particles n=32 (toward ParticleGAN scale)
- `vic0` [MATCH] vicreg_fn=locked delta={"vicreg_weight": 0.0} — VICReg off (particle spread unconstrained)
- `vicfaithful` [MATCH] vicreg_fn=faithful delta={"vicreg_weight": 1.0} — upstream VICReg (var+cov, std_target=1.0, weight=1.0)
- `pl2off` [MATCH] vicreg_fn=locked delta={"particle_l2": 0.0} — particle L2 off (particles free to roam)
- `d15` [MATCH] vicreg_fn=locked delta={"d_lr_mult": 1.5} — ParticleGAN D mult x1.5
- `lr2x` [MATCH] vicreg_fn=locked delta={"d_lr_mult": 1.5, "lr": 0.01} — #116 pg_2x_lr clone (2x base + D 1.5x, prior HOLD 1.0)
- `beta999` [MATCH] vicreg_fn=locked delta={"beta2": 0.999} — Adam beta2 0.999 (upstream momentum)
- `emaoff` [MATCH] vicreg_fn=locked delta={"ema": 0.0} — EMA off (raw last-step residual)
- `critic128` [MATCH] vicreg_fn=locked delta={"critic_hidden": 128} — thicker critic hidden=128
- `fullclone` [MATCH] vicreg_fn=faithful delta={"beta2": 0.999, "n_particles": 32, "particle_l2": 0.0, "vicreg_weight": 1.0} — #116 pg_full_clone thin preset (b2 .999, n=32, vic=1 faithful, pl2=0)
- `frozen_k2` [MATCH] vicreg_fn=locked delta={"kappa": 2.0, "particle_l2": 0.0, "prior_lr_mult": 0.01, "vicreg_weight": 0.0} — HUNT: frozen jitter prior + loose cap k=2
- `frozen_gi` [MATCH] vicreg_fn=locked delta={"grad_arm": "g_interp_cap", "particle_l2": 0.0, "prior_lr_mult": 0.01, "vicreg_weight": 0.0} — HUNT: frozen jitter prior + interp-path cap
- `frozen_c05` [MATCH] vicreg_fn=locked delta={"b_cap": 0.5, "particle_l2": 0.0, "prior_lr_mult": 0.01, "vicreg_weight": 0.0} — HUNT: frozen jitter prior + half-strength cap

## Per-budget detail (cover/off per cell/seed)

### `uni_locked` — locked b_cap k=1 coeff=1, n=12, vic=0.05, pl2=0.02, LR 5e-3 shared; GAN-only baseline

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.588 | 0.000 | — |
| 600 | close/s1 | 0.609 | 0.000 | — |
| 600 | close/s7 | 0.604 | 0.000 | — |
| 600 | divergent/s0 | 0.586 | 0.021 | — |
| 600 | divergent/s1 | 0.606 | 0.010 | — |
| 600 | divergent/s7 | 0.608 | 0.000 | — |
| 1200 | close/s0 | 0.624 | 0.000 | — |
| 1200 | close/s1 | 0.658 | 0.000 | — |
| 1200 | close/s7 | 0.642 | 0.000 | — |
| 1200 | divergent/s0 | 0.624 | 0.000 | — |
| 1200 | divergent/s1 | 0.667 | 0.000 | — |
| 1200 | divergent/s7 | 0.654 | 0.000 | — |
| 2400 | close/s0 | 0.642 | 0.000 | — |
| 2400 | close/s1 | 0.672 | 0.000 | — |
| 2400 | close/s7 | 0.669 | 0.000 | — |
| 2400 | divergent/s0 | 0.647 | 0.000 | — |
| 2400 | divergent/s1 | 0.684 | 0.000 | — |
| 2400 | divergent/s7 | 0.673 | 0.000 | — |
| 3400 | close/s0 | 0.658 | 0.000 | — |
| 3400 | close/s1 | 0.685 | 0.000 | — |
| 3400 | close/s7 | 0.681 | 0.000 | — |
| 3400 | divergent/s0 | 0.659 | 0.000 | — |
| 3400 | divergent/s1 | 0.697 | 0.000 | — |
| 3400 | divergent/s7 | 0.686 | 0.000 | — |

### `k05` — tighter cap k=0.5 (D flatter, G signal weaker?)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.587 | 0.000 | — |
| 600 | close/s1 | 0.630 | 0.000 | — |
| 600 | close/s7 | 0.610 | 0.000 | — |
| 600 | divergent/s0 | 0.591 | 0.021 | — |
| 600 | divergent/s1 | 0.626 | 0.000 | — |
| 600 | divergent/s7 | 0.626 | 0.000 | — |
| 1200 | close/s0 | 0.633 | 0.000 | — |
| 1200 | close/s1 | 0.686 | 0.000 | — |
| 1200 | close/s7 | 0.659 | 0.000 | — |
| 1200 | divergent/s0 | 0.636 | 0.000 | — |
| 1200 | divergent/s1 | 0.688 | 0.000 | — |
| 1200 | divergent/s7 | 0.685 | 0.000 | — |
| 2400 | close/s0 | 0.669 | 0.000 | — |
| 2400 | close/s1 | 0.711 | 0.000 | — |
| 2400 | close/s7 | 0.697 | 0.000 | — |
| 2400 | divergent/s0 | 0.678 | 0.000 | — |
| 2400 | divergent/s1 | 0.720 | 0.000 | — |
| 2400 | divergent/s7 | 0.720 | 0.000 | — |
| 3400 | close/s0 | 0.695 | 0.000 | — |
| 3400 | close/s1 | 0.738 | 0.000 | — |
| 3400 | close/s7 | 0.722 | 0.000 | — |
| 3400 | divergent/s0 | 0.704 | 0.000 | — |
| 3400 | divergent/s1 | 0.742 | 0.000 | — |
| 3400 | divergent/s7 | 0.743 | 0.000 | — |

### `k2` — looser cap k=2 (D may stay steep longer)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.583 | 0.000 | — |
| 600 | close/s1 | 0.616 | 0.000 | — |
| 600 | close/s7 | 0.603 | 0.000 | — |
| 600 | divergent/s0 | 0.582 | 0.021 | — |
| 600 | divergent/s1 | 0.453 | 0.042 | — |
| 600 | divergent/s7 | 0.597 | 0.000 | — |
| 1200 | close/s0 | 0.619 | 0.000 | — |
| 1200 | close/s1 | 0.661 | 0.000 | — |
| 1200 | close/s7 | 0.633 | 0.000 | — |
| 1200 | divergent/s0 | 0.620 | 0.000 | — |
| 1200 | divergent/s1 | 0.449 | 0.042 | — |
| 1200 | divergent/s7 | 0.639 | 0.000 | — |
| 2400 | close/s0 | 0.630 | 0.000 | — |
| 2400 | close/s1 | 0.668 | 0.000 | — |
| 2400 | close/s7 | 0.650 | 0.000 | — |
| 2400 | divergent/s0 | 0.631 | 0.000 | — |
| 2400 | divergent/s1 | 0.449 | 0.042 | — |
| 2400 | divergent/s7 | 0.645 | 0.000 | — |
| 3400 | close/s0 | 0.637 | 0.000 | — |
| 3400 | close/s1 | 0.677 | 0.000 | — |
| 3400 | close/s7 | 0.654 | 0.000 | — |
| 3400 | divergent/s0 | 0.637 | 0.000 | — |
| 3400 | divergent/s1 | 0.292 | 0.708 | — |
| 3400 | divergent/s7 | 0.654 | 0.000 | — |

### `coeff05` — half-strength cap coeff=0.5

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.586 | 0.000 | — |
| 600 | close/s1 | 0.618 | 0.000 | — |
| 600 | close/s7 | 0.602 | 0.000 | — |
| 600 | divergent/s0 | 0.583 | 0.021 | — |
| 600 | divergent/s1 | 0.608 | 0.010 | — |
| 600 | divergent/s7 | 0.608 | 0.000 | — |
| 1200 | close/s0 | 0.624 | 0.000 | — |
| 1200 | close/s1 | 0.667 | 0.000 | — |
| 1200 | close/s7 | 0.643 | 0.000 | — |
| 1200 | divergent/s0 | 0.639 | 0.000 | — |
| 1200 | divergent/s1 | 0.659 | 0.000 | — |
| 1200 | divergent/s7 | 0.654 | 0.000 | — |
| 2400 | close/s0 | 0.649 | 0.000 | — |
| 2400 | close/s1 | 0.680 | 0.000 | — |
| 2400 | close/s7 | 0.669 | 0.000 | — |
| 2400 | divergent/s0 | 0.646 | 0.000 | — |
| 2400 | divergent/s1 | 0.683 | 0.000 | — |
| 2400 | divergent/s7 | 0.674 | 0.000 | — |
| 3400 | close/s0 | 0.655 | 0.000 | — |
| 3400 | close/s1 | 0.694 | 0.000 | — |
| 3400 | close/s7 | 0.679 | 0.000 | — |
| 3400 | divergent/s0 | 0.658 | 0.000 | — |
| 3400 | divergent/s1 | 0.691 | 0.000 | — |
| 3400 | divergent/s7 | 0.682 | 0.000 | — |

### `coeff2` — double-strength cap coeff=2

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.588 | 0.000 | — |
| 600 | close/s1 | 0.623 | 0.000 | — |
| 600 | close/s7 | 0.606 | 0.000 | — |
| 600 | divergent/s0 | 0.584 | 0.021 | — |
| 600 | divergent/s1 | 0.596 | 0.010 | — |
| 600 | divergent/s7 | 0.608 | 0.000 | — |
| 1200 | close/s0 | 0.627 | 0.000 | — |
| 1200 | close/s1 | 0.668 | 0.000 | — |
| 1200 | close/s7 | 0.646 | 0.000 | — |
| 1200 | divergent/s0 | 0.631 | 0.000 | — |
| 1200 | divergent/s1 | 0.656 | 0.000 | — |
| 1200 | divergent/s7 | 0.653 | 0.000 | — |
| 2400 | close/s0 | 0.646 | 0.000 | — |
| 2400 | close/s1 | 0.694 | 0.000 | — |
| 2400 | close/s7 | 0.672 | 0.000 | — |
| 2400 | divergent/s0 | 0.648 | 0.000 | — |
| 2400 | divergent/s1 | 0.675 | 0.000 | — |
| 2400 | divergent/s7 | 0.675 | 0.000 | — |
| 3400 | close/s0 | 0.659 | 0.000 | — |
| 3400 | close/s1 | 0.703 | 0.000 | — |
| 3400 | close/s7 | 0.683 | 0.000 | — |
| 3400 | divergent/s0 | 0.661 | 0.000 | — |
| 3400 | divergent/s1 | 0.694 | 0.000 | — |
| 3400 | divergent/s7 | 0.689 | 0.000 | — |

### `ginterp` — interp-path cap (high-dim ParticleGAN geometry)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.584 | 0.000 | — |
| 600 | close/s1 | 0.604 | 0.000 | — |
| 600 | close/s7 | 0.607 | 0.000 | — |
| 600 | divergent/s0 | 0.588 | 0.021 | — |
| 600 | divergent/s1 | 0.585 | 0.010 | — |
| 600 | divergent/s7 | 0.596 | 0.000 | — |
| 1200 | close/s0 | 0.626 | 0.000 | — |
| 1200 | close/s1 | 0.639 | 0.000 | — |
| 1200 | close/s7 | 0.646 | 0.000 | — |
| 1200 | divergent/s0 | 0.626 | 0.000 | — |
| 1200 | divergent/s1 | 0.626 | 0.000 | — |
| 1200 | divergent/s7 | 0.636 | 0.000 | — |
| 2400 | close/s0 | 0.650 | 0.000 | — |
| 2400 | close/s1 | 0.658 | 0.000 | — |
| 2400 | close/s7 | 0.671 | 0.000 | — |
| 2400 | divergent/s0 | 0.648 | 0.000 | — |
| 2400 | divergent/s1 | 0.644 | 0.000 | — |
| 2400 | divergent/s7 | 0.659 | 0.000 | — |
| 3400 | close/s0 | 0.657 | 0.000 | — |
| 3400 | close/s1 | 0.667 | 0.000 | — |
| 3400 | close/s7 | 0.687 | 0.000 | — |
| 3400 | divergent/s0 | 0.662 | 0.000 | — |
| 3400 | divergent/s1 | 0.660 | 0.000 | — |
| 3400 | divergent/s7 | 0.672 | 0.000 | — |

### `delayed` — delayed center anneal (hold k then slide to R1/R2-like)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.590 | 0.000 | — |
| 600 | close/s1 | 0.608 | 0.000 | — |
| 600 | close/s7 | 0.602 | 0.000 | — |
| 600 | divergent/s0 | 0.588 | 0.021 | — |
| 600 | divergent/s1 | 0.607 | 0.010 | — |
| 600 | divergent/s7 | 0.607 | 0.000 | — |
| 1200 | close/s0 | 0.626 | 0.000 | — |
| 1200 | close/s1 | 0.658 | 0.000 | — |
| 1200 | close/s7 | 0.643 | 0.000 | — |
| 1200 | divergent/s0 | 0.624 | 0.000 | — |
| 1200 | divergent/s1 | 0.666 | 0.000 | — |
| 1200 | divergent/s7 | 0.657 | 0.000 | — |
| 2400 | close/s0 | 0.646 | 0.000 | — |
| 2400 | close/s1 | 0.676 | 0.000 | — |
| 2400 | close/s7 | 0.670 | 0.000 | — |
| 2400 | divergent/s0 | 0.649 | 0.000 | — |
| 2400 | divergent/s1 | 0.688 | 0.000 | — |
| 2400 | divergent/s7 | 0.675 | 0.000 | — |
| 3400 | close/s0 | 0.656 | 0.000 | — |
| 3400 | close/s1 | 0.686 | 0.000 | — |
| 3400 | close/s7 | 0.683 | 0.000 | — |
| 3400 | divergent/s0 | 0.660 | 0.000 | — |
| 3400 | divergent/s1 | 0.700 | 0.000 | — |
| 3400 | divergent/s7 | 0.692 | 0.000 | — |

### `lazy2` — lazy reg every 2nd step (StyleGAN2-style, coeff x2)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.582 | 0.000 | — |
| 600 | close/s1 | 0.616 | 0.000 | — |
| 600 | close/s7 | 0.604 | 0.000 | — |
| 600 | divergent/s0 | 0.587 | 0.021 | — |
| 600 | divergent/s1 | 0.544 | 0.010 | — |
| 600 | divergent/s7 | 0.608 | 0.000 | — |
| 1200 | close/s0 | 0.622 | 0.000 | — |
| 1200 | close/s1 | 0.663 | 0.000 | — |
| 1200 | close/s7 | 0.648 | 0.000 | — |
| 1200 | divergent/s0 | 0.626 | 0.000 | — |
| 1200 | divergent/s1 | 0.648 | 0.000 | — |
| 1200 | divergent/s7 | 0.653 | 0.000 | — |
| 2400 | close/s0 | 0.646 | 0.000 | — |
| 2400 | close/s1 | 0.686 | 0.000 | — |
| 2400 | close/s7 | 0.675 | 0.000 | — |
| 2400 | divergent/s0 | 0.651 | 0.000 | — |
| 2400 | divergent/s1 | 0.673 | 0.000 | — |
| 2400 | divergent/s7 | 0.674 | 0.000 | — |
| 3400 | close/s0 | 0.667 | 0.000 | — |
| 3400 | close/s1 | 0.698 | 0.000 | — |
| 3400 | close/s7 | 0.686 | 0.000 | — |
| 3400 | divergent/s0 | 0.668 | 0.000 | — |
| 3400 | divergent/s1 | 0.684 | 0.000 | — |
| 3400 | divergent/s7 | 0.689 | 0.000 | — |

### `slowprior` — slow prior 0.1x (residual must carry the mode)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.818 | 0.000 | — |
| 600 | close/s1 | 0.418 | 0.000 | — |
| 600 | close/s7 | 0.832 | 0.000 | — |
| 600 | divergent/s0 | 0.820 | 0.000 | — |
| 600 | divergent/s1 | 0.125 | 0.875 | — |
| 600 | divergent/s7 | 0.813 | 0.000 | — |
| 1200 | close/s0 | 0.872 | 0.000 | HIT |
| 1200 | close/s1 | 0.253 | 0.000 | — |
| 1200 | close/s7 | 0.879 | 0.000 | HIT |
| 1200 | divergent/s0 | 0.878 | 0.000 | HIT |
| 1200 | divergent/s1 | 0.000 | 1.000 | — |
| 1200 | divergent/s7 | 0.881 | 0.000 | HIT |
| 2400 | close/s0 | 0.875 | 0.000 | HIT |
| 2400 | close/s1 | 0.446 | 0.000 | — |
| 2400 | close/s7 | 0.881 | 0.000 | HIT |
| 2400 | divergent/s0 | 0.862 | 0.000 | HIT |
| 2400 | divergent/s1 | 0.000 | 1.000 | — |
| 2400 | divergent/s7 | 0.877 | 0.000 | HIT |
| 3400 | close/s0 | 0.878 | 0.000 | HIT |
| 3400 | close/s1 | 0.460 | 0.000 | — |
| 3400 | close/s7 | 0.880 | 0.000 | HIT |
| 3400 | divergent/s0 | 0.868 | 0.000 | HIT |
| 3400 | divergent/s1 | 0.000 | 1.000 | — |
| 3400 | divergent/s7 | 0.879 | 0.000 | HIT |

### `frozenprior` — near-frozen prior 0.01x + VICReg off (pure jitter prior)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.842 | 0.000 | — |
| 600 | close/s1 | 0.855 | 0.000 | HIT |
| 600 | close/s7 | 0.870 | 0.000 | HIT |
| 600 | divergent/s0 | 0.850 | 0.000 | HIT |
| 600 | divergent/s1 | 0.375 | 0.625 | — |
| 600 | divergent/s7 | 0.845 | 0.000 | — |
| 1200 | close/s0 | 0.903 | 0.000 | HIT |
| 1200 | close/s1 | 0.903 | 0.000 | HIT |
| 1200 | close/s7 | 0.902 | 0.000 | HIT |
| 1200 | divergent/s0 | 0.912 | 0.000 | HIT |
| 1200 | divergent/s1 | 0.000 | 1.000 | — |
| 1200 | divergent/s7 | 0.905 | 0.000 | HIT |
| 2400 | close/s0 | 0.904 | 0.000 | HIT |
| 2400 | close/s1 | 0.905 | 0.000 | HIT |
| 2400 | close/s7 | 0.907 | 0.000 | HIT |
| 2400 | divergent/s0 | 0.909 | 0.000 | HIT |
| 2400 | divergent/s1 | 0.909 | 0.000 | HIT |
| 2400 | divergent/s7 | 0.909 | 0.000 | HIT |
| 3400 | close/s0 | 0.907 | 0.000 | HIT |
| 3400 | close/s1 | 0.901 | 0.000 | HIT |
| 3400 | close/s7 | 0.904 | 0.000 | HIT |
| 3400 | divergent/s0 | 0.907 | 0.000 | HIT |
| 3400 | divergent/s1 | 0.909 | 0.000 | HIT |
| 3400 | divergent/s7 | 0.905 | 0.000 | HIT |

### `n32` — big particles n=32 (toward ParticleGAN scale)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.611 | 0.000 | — |
| 600 | close/s1 | 0.645 | 0.000 | — |
| 600 | close/s7 | 0.644 | 0.000 | — |
| 600 | divergent/s0 | 0.617 | 0.000 | — |
| 600 | divergent/s1 | 0.453 | 0.042 | — |
| 600 | divergent/s7 | 0.643 | 0.000 | — |
| 1200 | close/s0 | 0.650 | 0.000 | — |
| 1200 | close/s1 | 0.700 | 0.000 | — |
| 1200 | close/s7 | 0.686 | 0.000 | — |
| 1200 | divergent/s0 | 0.661 | 0.000 | — |
| 1200 | divergent/s1 | 0.450 | 0.042 | — |
| 1200 | divergent/s7 | 0.699 | 0.000 | — |
| 2400 | close/s0 | 0.664 | 0.000 | — |
| 2400 | close/s1 | 0.702 | 0.000 | — |
| 2400 | close/s7 | 0.697 | 0.000 | — |
| 2400 | divergent/s0 | 0.669 | 0.000 | — |
| 2400 | divergent/s1 | 0.450 | 0.042 | — |
| 2400 | divergent/s7 | 0.713 | 0.000 | — |
| 3400 | close/s0 | 0.671 | 0.000 | — |
| 3400 | close/s1 | 0.705 | 0.000 | — |
| 3400 | close/s7 | 0.705 | 0.000 | — |
| 3400 | divergent/s0 | 0.681 | 0.000 | — |
| 3400 | divergent/s1 | 0.451 | 0.042 | — |
| 3400 | divergent/s7 | 0.720 | 0.000 | — |

### `vic0` — VICReg off (particle spread unconstrained)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.588 | 0.000 | — |
| 600 | close/s1 | 0.624 | 0.000 | — |
| 600 | close/s7 | 0.609 | 0.000 | — |
| 600 | divergent/s0 | 0.591 | 0.021 | — |
| 600 | divergent/s1 | 0.604 | 0.010 | — |
| 600 | divergent/s7 | 0.615 | 0.000 | — |
| 1200 | close/s0 | 0.629 | 0.000 | — |
| 1200 | close/s1 | 0.680 | 0.000 | — |
| 1200 | close/s7 | 0.656 | 0.000 | — |
| 1200 | divergent/s0 | 0.634 | 0.000 | — |
| 1200 | divergent/s1 | 0.653 | 0.000 | — |
| 1200 | divergent/s7 | 0.661 | 0.000 | — |
| 2400 | close/s0 | 0.641 | 0.000 | — |
| 2400 | close/s1 | 0.698 | 0.000 | — |
| 2400 | close/s7 | 0.677 | 0.000 | — |
| 2400 | divergent/s0 | 0.649 | 0.000 | — |
| 2400 | divergent/s1 | 0.675 | 0.000 | — |
| 2400 | divergent/s7 | 0.687 | 0.000 | — |
| 3400 | close/s0 | 0.661 | 0.000 | — |
| 3400 | close/s1 | 0.697 | 0.000 | — |
| 3400 | close/s7 | 0.688 | 0.000 | — |
| 3400 | divergent/s0 | 0.668 | 0.000 | — |
| 3400 | divergent/s1 | 0.691 | 0.000 | — |
| 3400 | divergent/s7 | 0.698 | 0.000 | — |

### `vicfaithful` — upstream VICReg (var+cov, std_target=1.0, weight=1.0)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.567 | 0.000 | — |
| 600 | close/s1 | 0.697 | 0.000 | — |
| 600 | close/s7 | 0.612 | 0.000 | — |
| 600 | divergent/s0 | 0.591 | 0.021 | — |
| 600 | divergent/s1 | 0.636 | 0.000 | — |
| 600 | divergent/s7 | 0.631 | 0.000 | — |
| 1200 | close/s0 | 0.583 | 0.010 | — |
| 1200 | close/s1 | 0.607 | 0.000 | — |
| 1200 | close/s7 | 0.634 | 0.000 | — |
| 1200 | divergent/s0 | 0.612 | 0.000 | — |
| 1200 | divergent/s1 | 0.657 | 0.000 | — |
| 1200 | divergent/s7 | 0.624 | 0.000 | — |
| 2400 | close/s0 | 0.526 | 0.000 | — |
| 2400 | close/s1 | 0.476 | 0.000 | — |
| 2400 | close/s7 | 0.520 | 0.000 | — |
| 2400 | divergent/s0 | 0.505 | 0.021 | — |
| 2400 | divergent/s1 | 0.632 | 0.000 | — |
| 2400 | divergent/s7 | 0.546 | 0.000 | — |
| 3400 | close/s0 | 0.652 | 0.000 | — |
| 3400 | close/s1 | 0.520 | 0.000 | — |
| 3400 | close/s7 | 0.508 | 0.000 | — |
| 3400 | divergent/s0 | 0.568 | 0.000 | — |
| 3400 | divergent/s1 | 0.474 | 0.042 | — |
| 3400 | divergent/s7 | 0.516 | 0.000 | — |

### `pl2off` — particle L2 off (particles free to roam)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.586 | 0.000 | — |
| 600 | close/s1 | 0.608 | 0.000 | — |
| 600 | close/s7 | 0.599 | 0.000 | — |
| 600 | divergent/s0 | 0.581 | 0.021 | — |
| 600 | divergent/s1 | 0.608 | 0.000 | — |
| 600 | divergent/s7 | 0.605 | 0.000 | — |
| 1200 | close/s0 | 0.611 | 0.000 | — |
| 1200 | close/s1 | 0.649 | 0.000 | — |
| 1200 | close/s7 | 0.630 | 0.000 | — |
| 1200 | divergent/s0 | 0.617 | 0.000 | — |
| 1200 | divergent/s1 | 0.655 | 0.000 | — |
| 1200 | divergent/s7 | 0.641 | 0.000 | — |
| 2400 | close/s0 | 0.617 | 0.000 | — |
| 2400 | close/s1 | 0.648 | 0.000 | — |
| 2400 | close/s7 | 0.637 | 0.000 | — |
| 2400 | divergent/s0 | 0.614 | 0.000 | — |
| 2400 | divergent/s1 | 0.658 | 0.000 | — |
| 2400 | divergent/s7 | 0.641 | 0.000 | — |
| 3400 | close/s0 | 0.614 | 0.000 | — |
| 3400 | close/s1 | 0.650 | 0.000 | — |
| 3400 | close/s7 | 0.635 | 0.000 | — |
| 3400 | divergent/s0 | 0.609 | 0.000 | — |
| 3400 | divergent/s1 | 0.660 | 0.000 | — |
| 3400 | divergent/s7 | 0.639 | 0.000 | — |

### `d15` — ParticleGAN D mult x1.5

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.585 | 0.000 | — |
| 600 | close/s1 | 0.625 | 0.000 | — |
| 600 | close/s7 | 0.613 | 0.000 | — |
| 600 | divergent/s0 | 0.589 | 0.021 | — |
| 600 | divergent/s1 | 0.616 | 0.000 | — |
| 600 | divergent/s7 | 0.610 | 0.000 | — |
| 1200 | close/s0 | 0.624 | 0.000 | — |
| 1200 | close/s1 | 0.674 | 0.000 | — |
| 1200 | close/s7 | 0.652 | 0.000 | — |
| 1200 | divergent/s0 | 0.628 | 0.000 | — |
| 1200 | divergent/s1 | 0.683 | 0.000 | — |
| 1200 | divergent/s7 | 0.658 | 0.000 | — |
| 2400 | close/s0 | 0.640 | 0.000 | — |
| 2400 | close/s1 | 0.691 | 0.000 | — |
| 2400 | close/s7 | 0.684 | 0.000 | — |
| 2400 | divergent/s0 | 0.647 | 0.000 | — |
| 2400 | divergent/s1 | 0.704 | 0.000 | — |
| 2400 | divergent/s7 | 0.682 | 0.000 | — |
| 3400 | close/s0 | 0.656 | 0.000 | — |
| 3400 | close/s1 | 0.706 | 0.000 | — |
| 3400 | close/s7 | 0.697 | 0.000 | — |
| 3400 | divergent/s0 | 0.666 | 0.000 | — |
| 3400 | divergent/s1 | 0.717 | 0.000 | — |
| 3400 | divergent/s7 | 0.692 | 0.000 | — |

### `lr2x` — #116 pg_2x_lr clone (2x base + D 1.5x, prior HOLD 1.0)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.590 | 0.000 | — |
| 600 | close/s1 | 0.634 | 0.000 | — |
| 600 | close/s7 | 0.617 | 0.000 | — |
| 600 | divergent/s0 | 0.594 | 0.021 | — |
| 600 | divergent/s1 | 0.625 | 0.000 | — |
| 600 | divergent/s7 | 0.648 | 0.000 | — |
| 1200 | close/s0 | 0.615 | 0.000 | — |
| 1200 | close/s1 | 0.654 | 0.000 | — |
| 1200 | close/s7 | 0.622 | 0.000 | — |
| 1200 | divergent/s0 | 0.644 | 0.000 | — |
| 1200 | divergent/s1 | 0.664 | 0.000 | — |
| 1200 | divergent/s7 | 0.695 | 0.000 | — |
| 2400 | close/s0 | 0.666 | 0.000 | — |
| 2400 | close/s1 | 0.704 | 0.000 | — |
| 2400 | close/s7 | 0.678 | 0.000 | — |
| 2400 | divergent/s0 | 0.673 | 0.000 | — |
| 2400 | divergent/s1 | 0.682 | 0.000 | — |
| 2400 | divergent/s7 | 0.733 | 0.000 | — |
| 3400 | close/s0 | 0.719 | 0.000 | — |
| 3400 | close/s1 | 0.716 | 0.000 | — |
| 3400 | close/s7 | 0.703 | 0.000 | — |
| 3400 | divergent/s0 | 0.699 | 0.000 | — |
| 3400 | divergent/s1 | 0.742 | 0.000 | — |
| 3400 | divergent/s7 | 0.746 | 0.000 | — |

### `beta999` — Adam beta2 0.999 (upstream momentum)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.590 | 0.000 | — |
| 600 | close/s1 | 0.609 | 0.000 | — |
| 600 | close/s7 | 0.599 | 0.000 | — |
| 600 | divergent/s0 | 0.589 | 0.021 | — |
| 600 | divergent/s1 | 0.585 | 0.010 | — |
| 600 | divergent/s7 | 0.611 | 0.000 | — |
| 1200 | close/s0 | 0.623 | 0.000 | — |
| 1200 | close/s1 | 0.667 | 0.000 | — |
| 1200 | close/s7 | 0.637 | 0.000 | — |
| 1200 | divergent/s0 | 0.623 | 0.000 | — |
| 1200 | divergent/s1 | 0.654 | 0.000 | — |
| 1200 | divergent/s7 | 0.657 | 0.000 | — |
| 2400 | close/s0 | 0.640 | 0.000 | — |
| 2400 | close/s1 | 0.671 | 0.000 | — |
| 2400 | close/s7 | 0.660 | 0.000 | — |
| 2400 | divergent/s0 | 0.640 | 0.000 | — |
| 2400 | divergent/s1 | 0.670 | 0.000 | — |
| 2400 | divergent/s7 | 0.673 | 0.000 | — |
| 3400 | close/s0 | 0.654 | 0.000 | — |
| 3400 | close/s1 | 0.692 | 0.000 | — |
| 3400 | close/s7 | 0.672 | 0.000 | — |
| 3400 | divergent/s0 | 0.658 | 0.000 | — |
| 3400 | divergent/s1 | 0.686 | 0.000 | — |
| 3400 | divergent/s7 | 0.689 | 0.000 | — |

### `emaoff` — EMA off (raw last-step residual)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.617 | 0.000 | — |
| 600 | close/s1 | 0.652 | 0.000 | — |
| 600 | close/s7 | 0.646 | 0.000 | — |
| 600 | divergent/s0 | 0.618 | 0.000 | — |
| 600 | divergent/s1 | 0.658 | 0.000 | — |
| 600 | divergent/s7 | 0.647 | 0.000 | — |
| 1200 | close/s0 | 0.631 | 0.000 | — |
| 1200 | close/s1 | 0.652 | 0.000 | — |
| 1200 | close/s7 | 0.639 | 0.000 | — |
| 1200 | divergent/s0 | 0.628 | 0.000 | — |
| 1200 | divergent/s1 | 0.673 | 0.000 | — |
| 1200 | divergent/s7 | 0.653 | 0.000 | — |
| 2400 | close/s0 | 0.634 | 0.000 | — |
| 2400 | close/s1 | 0.673 | 0.000 | — |
| 2400 | close/s7 | 0.679 | 0.000 | — |
| 2400 | divergent/s0 | 0.644 | 0.000 | — |
| 2400 | divergent/s1 | 0.687 | 0.000 | — |
| 2400 | divergent/s7 | 0.675 | 0.000 | — |
| 3400 | close/s0 | 0.656 | 0.000 | — |
| 3400 | close/s1 | 0.683 | 0.000 | — |
| 3400 | close/s7 | 0.678 | 0.000 | — |
| 3400 | divergent/s0 | 0.661 | 0.000 | — |
| 3400 | divergent/s1 | 0.691 | 0.000 | — |
| 3400 | divergent/s7 | 0.685 | 0.000 | — |

### `critic128` — thicker critic hidden=128

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.585 | 0.000 | — |
| 600 | close/s1 | 0.641 | 0.000 | — |
| 600 | close/s7 | 0.606 | 0.000 | — |
| 600 | divergent/s0 | 0.590 | 0.021 | — |
| 600 | divergent/s1 | 0.586 | 0.010 | — |
| 600 | divergent/s7 | 0.621 | 0.000 | — |
| 1200 | close/s0 | 0.631 | 0.000 | — |
| 1200 | close/s1 | 0.696 | 0.000 | — |
| 1200 | close/s7 | 0.644 | 0.000 | — |
| 1200 | divergent/s0 | 0.635 | 0.000 | — |
| 1200 | divergent/s1 | 0.652 | 0.000 | — |
| 1200 | divergent/s7 | 0.679 | 0.000 | — |
| 2400 | close/s0 | 0.646 | 0.000 | — |
| 2400 | close/s1 | 0.706 | 0.000 | — |
| 2400 | close/s7 | 0.668 | 0.000 | — |
| 2400 | divergent/s0 | 0.652 | 0.000 | — |
| 2400 | divergent/s1 | 0.678 | 0.000 | — |
| 2400 | divergent/s7 | 0.700 | 0.000 | — |
| 3400 | close/s0 | 0.660 | 0.000 | — |
| 3400 | close/s1 | 0.720 | 0.000 | — |
| 3400 | close/s7 | 0.686 | 0.000 | — |
| 3400 | divergent/s0 | 0.665 | 0.000 | — |
| 3400 | divergent/s1 | 0.691 | 0.000 | — |
| 3400 | divergent/s7 | 0.716 | 0.000 | — |

### `fullclone` — #116 pg_full_clone thin preset (b2 .999, n=32, vic=1 faithful, pl2=0)

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.593 | 0.000 | — |
| 600 | close/s1 | 0.653 | 0.000 | — |
| 600 | close/s7 | 0.627 | 0.000 | — |
| 600 | divergent/s0 | 0.616 | 0.000 | — |
| 600 | divergent/s1 | 0.589 | 0.010 | — |
| 600 | divergent/s7 | 0.661 | 0.000 | — |
| 1200 | close/s0 | 0.525 | 0.000 | — |
| 1200 | close/s1 | 0.645 | 0.000 | — |
| 1200 | close/s7 | 0.540 | 0.000 | — |
| 1200 | divergent/s0 | 0.593 | 0.021 | — |
| 1200 | divergent/s1 | 0.659 | 0.000 | — |
| 1200 | divergent/s7 | 0.666 | 0.000 | — |
| 2400 | close/s0 | 0.492 | 0.000 | — |
| 2400 | close/s1 | 0.568 | 0.000 | — |
| 2400 | close/s7 | 0.483 | 0.000 | — |
| 2400 | divergent/s0 | 0.543 | 0.052 | — |
| 2400 | divergent/s1 | 0.576 | 0.010 | — |
| 2400 | divergent/s7 | 0.517 | 0.010 | — |
| 3400 | close/s0 | 0.497 | 0.000 | — |
| 3400 | close/s1 | 0.590 | 0.000 | — |
| 3400 | close/s7 | 0.486 | 0.000 | — |
| 3400 | divergent/s0 | 0.555 | 0.021 | — |
| 3400 | divergent/s1 | 0.498 | 0.010 | — |
| 3400 | divergent/s7 | 0.485 | 0.198 | — |

### `frozen_k2` — HUNT: frozen jitter prior + loose cap k=2

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.846 | 0.000 | — |
| 600 | close/s1 | 0.850 | 0.000 | — |
| 600 | close/s7 | 0.869 | 0.000 | HIT |
| 600 | divergent/s0 | 0.856 | 0.000 | HIT |
| 600 | divergent/s1 | 0.463 | 0.042 | — |
| 600 | divergent/s7 | 0.851 | 0.000 | HIT |
| 1200 | close/s0 | 0.901 | 0.000 | HIT |
| 1200 | close/s1 | 0.903 | 0.000 | HIT |
| 1200 | close/s7 | 0.901 | 0.000 | HIT |
| 1200 | divergent/s0 | 0.914 | 0.000 | HIT |
| 1200 | divergent/s1 | 0.482 | 0.042 | — |
| 1200 | divergent/s7 | 0.905 | 0.000 | HIT |
| 2400 | close/s0 | 0.903 | 0.000 | HIT |
| 2400 | close/s1 | 0.903 | 0.000 | HIT |
| 2400 | close/s7 | 0.905 | 0.000 | HIT |
| 2400 | divergent/s0 | 0.904 | 0.000 | HIT |
| 2400 | divergent/s1 | 0.912 | 0.000 | HIT |
| 2400 | divergent/s7 | 0.905 | 0.000 | HIT |
| 3400 | close/s0 | 0.907 | 0.000 | HIT |
| 3400 | close/s1 | 0.901 | 0.000 | HIT |
| 3400 | close/s7 | 0.903 | 0.000 | HIT |
| 3400 | divergent/s0 | 0.905 | 0.000 | HIT |
| 3400 | divergent/s1 | 0.908 | 0.000 | HIT |
| 3400 | divergent/s7 | 0.902 | 0.000 | HIT |

### `frozen_gi` — HUNT: frozen jitter prior + interp-path cap

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.843 | 0.000 | — |
| 600 | close/s1 | 0.854 | 0.000 | HIT |
| 600 | close/s7 | 0.869 | 0.000 | HIT |
| 600 | divergent/s0 | 0.854 | 0.000 | HIT |
| 600 | divergent/s1 | 0.859 | 0.000 | HIT |
| 600 | divergent/s7 | 0.855 | 0.000 | HIT |
| 1200 | close/s0 | 0.900 | 0.000 | HIT |
| 1200 | close/s1 | 0.906 | 0.000 | HIT |
| 1200 | close/s7 | 0.899 | 0.000 | HIT |
| 1200 | divergent/s0 | 0.904 | 0.000 | HIT |
| 1200 | divergent/s1 | 0.907 | 0.000 | HIT |
| 1200 | divergent/s7 | 0.900 | 0.000 | HIT |
| 2400 | close/s0 | 0.902 | 0.000 | HIT |
| 2400 | close/s1 | 0.903 | 0.000 | HIT |
| 2400 | close/s7 | 0.906 | 0.000 | HIT |
| 2400 | divergent/s0 | 0.907 | 0.000 | HIT |
| 2400 | divergent/s1 | 0.905 | 0.000 | HIT |
| 2400 | divergent/s7 | 0.907 | 0.000 | HIT |
| 3400 | close/s0 | 0.898 | 0.000 | HIT |
| 3400 | close/s1 | 0.903 | 0.000 | HIT |
| 3400 | close/s7 | 0.900 | 0.000 | HIT |
| 3400 | divergent/s0 | 0.909 | 0.000 | HIT |
| 3400 | divergent/s1 | 0.908 | 0.000 | HIT |
| 3400 | divergent/s7 | 0.903 | 0.000 | HIT |

### `frozen_c05` — HUNT: frozen jitter prior + half-strength cap

| budget | cell/seed | cover | off | hit |
|---|---|---|---|---|
| 600 | close/s0 | 0.845 | 0.000 | — |
| 600 | close/s1 | 0.830 | 0.000 | — |
| 600 | close/s7 | 0.870 | 0.000 | HIT |
| 600 | divergent/s0 | 0.851 | 0.000 | HIT |
| 600 | divergent/s1 | 0.462 | 0.042 | — |
| 600 | divergent/s7 | 0.831 | 0.000 | — |
| 1200 | close/s0 | 0.904 | 0.000 | HIT |
| 1200 | close/s1 | 0.898 | 0.000 | HIT |
| 1200 | close/s7 | 0.901 | 0.000 | HIT |
| 1200 | divergent/s0 | 0.912 | 0.000 | HIT |
| 1200 | divergent/s1 | 0.481 | 0.042 | — |
| 1200 | divergent/s7 | 0.903 | 0.000 | HIT |
| 2400 | close/s0 | 0.903 | 0.000 | HIT |
| 2400 | close/s1 | 0.902 | 0.000 | HIT |
| 2400 | close/s7 | 0.905 | 0.000 | HIT |
| 2400 | divergent/s0 | 0.908 | 0.000 | HIT |
| 2400 | divergent/s1 | 0.418 | 0.083 | — |
| 2400 | divergent/s7 | 0.905 | 0.000 | HIT |
| 3400 | close/s0 | 0.909 | 0.000 | HIT |
| 3400 | close/s1 | 0.904 | 0.000 | HIT |
| 3400 | close/s7 | 0.904 | 0.000 | HIT |
| 3400 | divergent/s0 | 0.908 | 0.000 | HIT |
| 3400 | divergent/s1 | 0.000 | 1.000 | — |
| 3400 | divergent/s7 | 0.904 | 0.000 | HIT |


## Read

- PASS arms (earliest budget): `frozen_gi`@1200, `frozen_k2`@2400, `frozenprior`@2400.
- Best early-pass recipe: `frozen_gi` (frozen jitter prior 0.01x + VICReg/L2 off + `g_interp_cap`) — 6/6 from 1200 through 3400.
- Why the rest fail: live particles eat the pole modes — the fake cloud
  covers the teacher while the residual undershoots ~30% (`uni_locked`
  plateaus at cover 0.66-0.70 even at 3400). Slowing the prior forces the
  residual to carry +1. Divergent seed 1 is the hardest cell (sole miss
  for four arms at 1200; garble-collapse for `slowprior`/`frozen_c05` at
  longer budgets); the interp-path cap is the only geometry robust there
  at every budget.
- No supervised MSE on any G: `cover_weight=0`, `fm_weight=0` (fail-closed).
- Music bipolar untouched: no trainer row, no default, no `--lm_target` flip.

Scales: [0.0, 0.5, 1.0] (gates on 0 and 1; 0.5 diagnostic; -1 canary).
