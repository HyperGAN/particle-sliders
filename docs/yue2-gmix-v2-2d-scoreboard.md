# Released-v2-on-2D scoreboard: `particle-gmix-1600-v2` game vs UNI gates

**Verdict: PASS.** The released-v2-shaped 2D arm clears both UNI cells on all
three seeds at the native 1600 stop (EMA readout, the release-relevant one)
and live+EMA at 3400. No gate was faked, moved, or rescored.

## The arm

`analysis/slider2d/yue2_gmix_v2_exam.py` (`particle_gmix_v2`, propose-only)
reuses the parent bridge toy (`yue2_particle_exam.Student/Backend/scoring`,
`yue2_particle_bridge` game, `particle_bridge_gan` shared losses) and swaps
the bridge-doc MLP-only head for the released v2 head. Knob pins live in
`V2_SPEC`, tested against the Hub golden numbers in
`tests/test_yue2_gmix_v2_exam.py`.

| Knob | Parent bridge toy (2D HIT at 3400/8000) | This arm (released v2) |
|---|---|---|
| Critic | 3x48 MLP | `gmix_t8_w48_l1` (8x48 tokens, 1 layer, 4 heads, bound 8) |
| Normalization | absolute-positive whitening, `noise_start` 1.0 | paired-edit whitening, `sigma0 = edit_rms/0.28` |
| Noise horizon | T=8000, no hold | T=1600, hold 1.3xE (the 13-control schedule incl. female/male) |
| D/G batch | 64 | 8, with replacement |
| Core (unchanged) | Rp paired game, `b_cap` lazy-x4, 128x4, VIC, LRs 6e-4/9e-4/6e-3, Adam (0,0.999), EMA 0.995, GAN-only | identical |
| Gates (unchanged) | cover>=0.85, off<=0.05, neu_hold>=0.85, scales 0/0.5/1, -1 canary | identical |

Fixture gap (honest, not hidden): native trains 512 seedbank sources
(128 seeds x 4 templates x 32-token histories); PairField has 3 rows and no
histories, so the toy draws batches of 8 with replacement from 3 rows under
the same replacement policy. Step budgets are not equivalent convergence
measures across the two fixtures.

## Ladder (seeds 0/1/7, EMA readout; full traces in `yue2-gmix-v2-2d-ladder.json`)

| Updates | Divergent cover | Close cover | Both cells, all seeds |
|---:|---:|---:|---|
| 600 | 0.476–0.485 | 0.493–0.514 | FAIL (cover; off-caption also fails on seeds 0/7 divergent) |
| 1200 | 0.786–0.811 | 0.922–0.948 | FAIL (divergent cover on every seed) |
| 1600 | 0.924–0.941 | 0.954–0.972 | **PASS (EMA, 6/6)**; off-caption 0, neutral hold 1 |
| 3400 | 0.963–0.980 | 0.970–0.992 | **PASS (live and EMA, 6/6)**; off-caption 0, neutral hold 1 |

Live weights at 1600 pass 5/6 (seed-0 divergent live cover 0.829, EMA 0.941);
live catches up everywhere by 3400. The release ships final EMA weights, so
the EMA readout is the release-relevant one — and it matches the native 1600
stop exactly. Peak transient G loss is 3.9–4.6 across runs (finite; passing
endpoints do not mean every update was smooth). The same weights fail the
bipolar exam in all six cases at every budget, as required: `-1` is an
unscored UNI canary, and the Hub documents negative strengths as
unsupported. The -1 endpoint lands nearest `neu` (never `pos`) but sings
off-corpus on divergent rows — unconstrained by design, hence unscored.

## Incidental fix this PR needed

The parent bridge toy (`yue2_particle_exam.py`) is currently broken on main:
`yue2_particle_bridge.update.predict` evaluates `row['prefix']` eagerly as a
`dict.get` default, so any row without `prefix` (both 2D toys) raises
`KeyError: 'prefix'`. One-word lazy fix (`row.get('prefix')`) repairs the
parent and the new arm; native rows always carry `train_ids`/`prefix`, so
native lookup order is unchanged. The bridge-doc 3400/8000 table predates the
regression and is not re-claimed here.

## Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python3 -m analysis.slider2d.yue2_gmix_v2_exam --seeds 0 1 7 \
  --steps 600 1200 1600 3400 --out /tmp/yue2-gmix-v2-ladder.json
PYTHONPATH=. python3 -m pytest tests/test_yue2_gmix_v2_exam.py \
  tests/test_yue2_particle_bridge.py tests/test_yue2_gan_exam.py -q
```

The exam exits 1 unless the declared final budget passes both cells on every
seed; earlier budgets are reported honestly and cannot be hidden by later
success. The 8 pin tests fail closed if the critic, normalization, noise
schedule, batch size, LRs, or GAN-only loss drift from the Hub numbers, and
if Music bipolar `ARM_B`, the live `--lm_target v9` default, the production
YuE2 Arm B recipe, or locked `AdvConfig()` defaults move.

## Related

- [docs/README.md](README.md) — operator map
- [yue2-slider.md](yue2-slider.md) — live CLI vs published vs this arm
- [yue2-particle-bridge.md](yue2-particle-bridge.md) — original MLP / absolute-whitening audit
- Live Music 3 default stays `--lm_target v9` / `--pole_mode hidden`.
