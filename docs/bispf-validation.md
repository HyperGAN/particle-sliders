# BISPF validation receipt

Executed in the isolated `codex/bispf-upgrade-20261017` clone, based on
`HyperGAN/particle-sliders` main `20fdf98d25455582e7e7191eef8ef51cf73173ae`.
The original `/ml2/hypergan/particle-sliders` checkout remained clean and
unchanged. No separate MOSS workspace was opened or modified.

Upstream source checkout: `../upstream-particle-gan`, clean develop HEAD
`b5d2372f2739d7579ab66d6dba6e220d2eb776d9`, verified with `git ls-remote`.
Tests below ran against that actual source via `PYTHONPATH`; they were not
just prepared. Dependency-resolver installation from GitHub was not tested.

## Environment and GPU preflight

Actual host `nvidia-smi` and process listings were checked outside the sandbox
before bounded validation. Driver 595.99.02 / CUDA 13.2, two RTX A6000s;
GPU 0 had nano-work-server (262 MiB) and desktop use; GPU 1 had llama-server
(17,154 MiB). No training process appeared in those listings. A later MOSS
single-GPU pilot was announced; BISPF validation stayed CPU-only throughout,
with `CUDA_VISIBLE_DEVICES=''`. No job was stopped, restarted or modified.
The sandbox itself cannot query the NVIDIA driver or see host jobs, so its
initial failed `nvidia-smi` was not treated as a host-driver failure.

Python 3.11; torch 2.11.0+cu128; NumPy 2.4.4; pytest 9.1.1.
`../test-env` was created with `--system-site-packages` from
`/home/mikkel/anaconda3/envs/minimax-music3/bin/python`. SciPy 1.17.1 was
installed only into that workspace virtualenv after the initial broad suite
failed collection for missing SciPy. Shared Python environments were not edited.

## Passing regression commands

All commands in this section ran from the particle-sliders clone.

```bash
PYTHONPATH=../upstream-particle-gan CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  timeout 60 ../test-env/bin/python -m pytest \
  tests/test_music3_bispf.py packages/particle-sliders-core/tests \
  tests/test_particle_sliders_formulation.py tests/test_yue2_particle_bridge.py \
  -q --disable-warnings --tb=short
```

**51 passed, 5 skipped in 4.09s**, exit 0. The skips require the optional
YuE2/native backend environment. Includes:

- Explicit numerical Sinkhorn loss/gradient and minimizing-sign checks;
  original coordinate metadata is retained, and G receives the whole batch.
- One particle Parameter, one optimizer owner, actual particle updates,
  upstream paired penalty/optimizer classes, constant learning rates.
- Stochastic model forwards and exact checkpoint continuation, initial
  prior moments and exact thermal scale, nonreentrant checkpoint replay
  with the scale context retained through backward.
- Music 3's real CLI with a 32-wide weights-free dummy: one update, save,
  reload and two more updates versus three uninterrupted updates.
  Model, critic, both optimizers, prior and streams compare bit exactly.
- Native safetensors export/load, exact scale-zero base identity, nonzero
  scale-one edit, live-weight exports and neutral-caption student inputs.
- Preserved gmix/endpoint/teacher/cap equations and the existing YuE2 tests.

An earlier combined run exposed shared `REFERENCE` mutation in the existing
Music 3 builder. The final implementation passes explicit per-run critic
settings; the combined suite above passes without those side effects.

From the upstream checkout, the following bounded subset also ran:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 timeout 60 \
  ../test-env/bin/python -m pytest -q tests/test_bispf.py \
  -k 'recipe_and_trainer_integration or v2_trainer_integration_and_telemetry or sinkhorn_scale_free_nn_residual or sbnp_capacity_conditioned_covariance_tethering or sbnp_degree_of_freedom_guard_discrete_codebooks or lambda_floor_default' \
  --tb=short
```

**6 passed, 34 deselected in 0.86s**, exit 0.

## Actual CLI smoke commands

The first command used `../smoke-inputs/prompts.yaml`, byte-identical to the
committed [prompt fixture](../conceptmod/textsliders/data/prompts-bispf-smoke.yaml).
The next two commands used the committed fixture path. Each exited 0 under
a 60-second timeout, CPU only. They did not load Music 3 weights.

```bash
PYTHONPATH=../upstream-particle-gan:packages/particle-sliders-core/src \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 timeout 60 ../test-env/bin/python \
  -m conceptmod.textsliders.train_lora_music3_particle \
  --dummy --device cpu --prompts_file ../smoke-inputs/prompts.yaml \
  --save_dir ../smoke-final --steps 3 --until 1 --save_every 1 \
  --max_seq_len 512 --seed 0 --batch_size 4

PYTHONPATH=../upstream-particle-gan:packages/particle-sliders-core/src \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 timeout 60 ../test-env/bin/python \
  -m conceptmod.textsliders.train_lora_music3_particle \
  --dummy --device cpu \
  --prompts_file conceptmod/textsliders/data/prompts-bispf-smoke.yaml \
  --save_dir ../smoke-final --steps 3 --until 3 --save_every 1 \
  --max_seq_len 512 --seed 0 --batch_size 4

PYTHONPATH=../upstream-particle-gan:packages/particle-sliders-core/src \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 timeout 60 ../test-env/bin/python \
  -m conceptmod.textsliders.train_lora_music3_particle \
  --dummy --device cpu \
  --prompts_file conceptmod/textsliders/data/prompts-bispf-smoke.yaml \
  --save_dir ../smoke-final-full --steps 3 --until 3 --save_every 1 \
  --max_seq_len 512 --seed 0 --batch_size 4
```

An actual comparison of the saved BISPF dictionaries, histories excluding
wall-clock timings, and exported tensors passed with `rtol=0, atol=0`.
The same comparison is regression-tested in `tests/test_music3_bispf.py`.
Final dummy metrics: D loss 2.6439573765, G adversarial loss 3.1301112175,
prior regularization .0041523031, sigma 1, cosine .4589060321. Rates remained
(.0006, .0009, .006). These three updates establish finite operation and
recovery, not convergence or native audio quality.

A separate one-update `--dummy --device cpu --formulation legacy` invocation
also exited 0, with the older 64-row RpGAN/VIC update and noise curve. Its
final G adversarial loss was .6931471825 and particle VIC .1044856757.

## Package and broader-suite checks

`pip wheel --no-deps --no-build-isolation ./packages/particle-sliders-core`
produced a 0.4.0 wheel (SHA256
`d5fd7382123d0ebf3aaf2755d381669284e882b28dc94f4910ec1beafd5b3d96`).
Every packaged Python source was compared byte-for-byte with the final source. It was installed with `--no-deps` into the isolated
virtualenv; importing `particle_sliders` from that installed wheel succeeded,
and its `bispf_provenance.json` package data was present. ParticleGAN for that
import came from the verified source checkout via `PYTHONPATH`.
`git diff --check` passed. Bulk logs and checkpoints remain outside Git in
the assigned task workspace.

The broader `timeout 120 ... python -m pytest -q --disable-warnings
--maxfail=5 --tb=short` run **did not complete** (exit 124; about 20% progress).
It showed one failure in the unrelated `tests/test_listen_server.py` because
the sandbox forbids socket creation. A focused rerun reproduced the exact
`PermissionError: [Errno 1] Operation not permitted` at the loopback socket.
The same exact focused test outside the socket-blocking sandbox passed:
`PYTHONPATH=../upstream-particle-gan CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 timeout 20 ../test-env/bin/python -m pytest tests/test_listen_server.py -q --tb=short`
returned **1 passed in 0.53s**, exit 0. Do not interpret the
partial broad run as a full-suite pass. No remaining broad-suite failures
were inferred from tests that did not finish.

## Remaining scope

No native Music 3 model/GPU run or generated audio was tested. Full-batch
Sinkhorn memory use, native checkpoint replay, wide-hidden-state upstream
heuristics and listening quality need validation before any long campaign.
The MOSS pilot is separate and receives no qualification from these receipts.
This change is for draft review; no merge or deployment was performed.
