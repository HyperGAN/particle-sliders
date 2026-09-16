# Music Arm B — operator recipe (RpGAN + b_cap)

Winning config from Slider's CPU / Field3D selection work (KEEP / DR✓):
`recommend=locked_shared; posture=hold_ablation; bottleneck=close`.
Lowrank / exam-only arms are DROP_EX — do not adopt.

Single source of truth in code: `conceptmod/textsliders/music_arm_b.py`
(`get_music_arm_b_recipe()`). Demo single-source: `analysis/slider2d/adv.py`.

## Winning argv (copy/paste)

```bash
--adv_loss rpgan_logistic --grad_reg b_cap --grad_coeff 1 --adv_reg_kappa 1 \
  --grad_norm l2 --b_cap 1 --fm_weight 0 --lm_target faithful_guard_e \
  --adv_arch mlp --cover_weight 1.0 --pole_weight 1.0 --parts 0 \
  --vicreg_weight 0 --eval_scales -1.0,0.0,0.5,1.0
```

Or print it any time (no Music weights, no torch needed):

```bash
python scripts/smoke_music_arm_b_argv.py
```

The smoke exits 0 only if the Arm B shape matches the table below, and
proves the validator rejects single-knob drift (FM-on, demo cover 1.5,
`--parts != 0`, wrong teacher, `tx` critic, wrong kappa). Any diff exits 1.

## Handoff table

| Knob | Value |
|---|---|
| Grad regularizer | ParticleGAN `b_cap` — `coeff=1`, `kappa=1`, `norm=l2` |
| Adversarial loss | RpGAN logistic (relativistic pair) |
| Feature matching | OFF — `fm_weight=0` |
| Teacher / leftover | `faithful_guard_e` |
| Critic arch | `mlp` (never `tx` + `faithful_guard_e` — dual-arm incompatible) |
| `b_cap` / `adv_reg_kappa` | `1` / `1` |
| Cover / pole (Music) | `cover_weight=1.0`, `pole_weight=1.0` (Field3D demo uses cover 1.5) |
| Particles | Music: `--parts 0` (toy: `n_particles≤12`, `particle_l2=0.02`) |
| Eval scales | always `-1, 0, 0.5, 1` |

Music-posture ADOPTs: `vicreg_weight=0` at `--parts 0`; close-family cells
prefer `n≥2` or `vic0@n1`; multi-seed before claiming a win. Propose-only
YAML hygiene: `e_on_content=0`; FLP: no lyric/caption row with unused-ê
primary (`se < max(su,sc)`).

## Using it with the supervised trainer today

The adversarial knobs (critic, RpGAN, `b_cap`, parts, cover/pole) are
**propose-only** on `train_lm_slider_music3.py` — that trainer is
supervised and has no critic. The honored subset is just the teacher:

```bash
python conceptmod/textsliders/train_lm_slider_music3.py --arm_b \
  --prompts_file <prompts>.yaml --save_dir <dir> [...]
```

`--arm_b` sets `--lm_target faithful_guard_e` and prints the full winning
argv for before/after verification. It fails closed if you also pass a
conflicting `--lm_target`. Default off — the live default stays `v9`.

## Verify before / after train

1. Print full adv-related argv and confirm every row in the table.
2. Confirm critic is mlp + `faithful_guard_e`, FM=0, b_cap=1, κ=1, parts=0.
3. If anything differs, **stop and report the diff** — do not silently
   train a drifted recipe.

## Explicitly do NOT use

- FM-on / raw feature matching under `b_cap`
- `tx` + `faithful_guard_e` in one argv
- lowrank_k3 / FLP student arms as the Music train recipe (DROP_EX)
- Thinned "b_cap with κ hardcoded and no GradRegularizer"
