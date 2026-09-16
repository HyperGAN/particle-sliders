# ParticleGAN formulation gap (CLONE ARM series framing)

Locked baseline (`analysis/slider2d/locked_baseline_defaults.py`, single
source): ParticleGAN-faithful b_cap (arm=b_cap, κ=1, coeff=1, L2, lazy=1,
anneal=none) + Music extras (cover_weight=1.5, teacher=faithful_guard_e,
fm_weight=0 / FM off, n_particles=12, particle_l2=0.02, vicreg=0.05,
Adam β1=0, EMA, delayed cosine).

Where the toy still differs from ParticleGAN proper:

1. **Particle cloud scale (this series, ARM 1/5).** ParticleGAN trains with
   ~20k particles; the CPU toy uses n=12 (two `ParticlePrior`s of 12×dim).
   ARM 1 scales n to 64→256 and measures gates + wall time, propose_only.
2. **Update budget.** ParticleGAN's 100-Gaussians run is ~7k updates; the
   toy fits 1200 steps (exam/sheet) on 2-D fixtures.
3. **Data geometry.** ParticleGAN matches real modes; the toy matches
   leftover-gated caption poles plus a span/end cloud (hidden-state deltas,
   not audio, not images).
4. **Mode pin.** `cover_weight=1.5` is a Music-only MSE onto gated centers —
   pure RpGAN undershoots the shared residual on sheet/exam width. No
   ParticleGAN analogue; not part of this series' deltas.
5. **Critic capacity.** Fourier-2 MLP (hidden=64, n_rand=16) sized for 2-D;
   ParticleGAN critics scale with data dim.

One knob per arm, propose_only, always compared against `locked_cfg` with
the same honesty/leak gates. Production argv and the live trainer default
(`--lm_target v9`) never move.
