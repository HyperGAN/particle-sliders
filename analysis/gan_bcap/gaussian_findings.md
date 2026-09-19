# 100-Gaussian cap reproduction

The reference recipe reproduces on an A6000. At 7,000 updates, `b_cap`
captures all 100 modes with **98.771%** of samples within three data standard
deviations of a mode center. The matched R1/R2 run reaches **93.461%**.
This is a single-seed reproduction, not a new hyperparameter sweep.

| Penalty | Modes | HQ fraction | Core sigma / data sigma | Tail beyond 10 sigma | Sliced W1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b_cap`, coefficient 1 | 100 | 0.98771 | 0.8672 | 0.00628 | 0.2123 |
| R1/R2, coefficient 0.02 | 100 | 0.93461 | 0.9035 | 0.04008 | 0.3207 |

The cap is exactly `0.5 * coefficient * (mean(relu(norm(grad D(real)) - 1)^2)
+ mean(relu(norm(grad D(fake)) - 1)^2))`. It caps excessive slopes; it does
not impose a lower bound. The successful cap run ends with median fake-input
gradient norm **0.0854**, compared with **0.0486** for R1/R2. Thus a norm below
one is compatible with successful training. A zero gradient or a detached
generator still needs a separate fix.

Both runs use actual optimizable latent particles. All 20,000 rows moved;
mean final displacement was 0.4329 for the cap and 0.3872 for R1/R2. This
behavior cannot be replaced by merely resampling fixed training rows.

Recipe held fixed: seed 1234; 20,000 particles with latent dimension 4;
batch 256; Fourier-2 MLP discriminator; relativistic-pair logistic GAN loss;
Adam betas `(0, 0.999)`; G LR 0.0006, D LR 0.0009, particle LR 0.006;
VICReg coefficient 1 on unique sampled particle rows; EMA 0.995 of both
generator and particles; full LR through 60% of training, then cosine decay
to 5% of the initial LR. The cap first passes all 100 modes and 90% HQ at
update 5,501. The R1/R2 run passes at 5,001 but has a lower final HQ fraction.

The harness imports the unchanged upstream `examples/100gaussians.py` at
revision `b5ee35f9b24cf35a6b1346ba2f0856c4877aa336` from
`/tmp/opencode/ParticleGAN`. It replaces plotting with capturing EMA model
references and logs existing coverage evaluations. Epoch grouping changes
to one epoch with the same total updates; the optimizer schedule and existing
100-step coverage evaluation frequency are preserved. Final metrics use
100,000 samples; sliced W1 uses 8,192 of those samples and 128 projections.
The run uses torch 2.11.0+cu128 and takes about 56 seconds per arm on GPU 0.

Reproduce from `/ml2/music`:

```bash
CUDA_VISIBLE_DEVICES=0 /home/mikkel/anaconda3/envs/minimax-music3/bin/python \
  sliders-conceptmod/analysis/gan_bcap/gaussian_repro.py
```

`gaussian_comparison.json` records numerical results and provenance;
`gaussian_comparison.png` plots training coverage and HQ. Each
`gaussian_*_seed1234/` directory contains a trajectory, final metrics,
scatter plot, and toy checkpoint.

This validates the reference cap and movable-prior recipe on the toy
distribution. It does not establish music quality, condition fidelity, or
the correct feature geometry for the language-model adaptation.
