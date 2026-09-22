# concept-slider-core (retired path)

This subdirectory was the Anima extraction of the shared algorithms. The
installable package is now
[`packages/particle-sliders-core`](../particle-sliders-core)
(`particle-sliders-core`, import `particle_sliders`).

`concept_slider_core` remains a deprecated alias inside that package.
New pins use HyperGAN/particle-sliders, not `mikkel/sliders-conceptmod`:

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@<commit>#subdirectory=packages/particle-sliders-core
```

Commits at or before `beaffeb` still contain the old tree. Do not point a
new product pin at this directory.
