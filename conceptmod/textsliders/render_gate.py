"""Closed-loop render gate: score a slider by what its full rollout produces.

The open-loop probe scores velocity deltas at anchored x_t states the base
model visits, and is blind to compounding: checkpoints reading cos 0.85+
render one pole at 3-4x base RMS while the opposite pole collapses toward
silence (see docs/tf-leak.md). This module runs the SHIPPED sampler --
FlowMatchEuler over sigmas linspace(1, 1/N), CFG 1.7 against a zeros
condition, LoRA live on both branches -- from the neutral condition at a
fixed seed for each scale, then measures the finals:

- audio (preferred): decode final latents with the vocoder and report RMS
  ratio vs the scale-0 render. Same quantity scripts/tf_render_gain.py
  reports on ritual ladders, computed in-process.
- latent (fallback): energy ratio of final latents. Cheap enough to run
  every N training steps; whether it tracks audio is measured in
  docs/tf-leak.md before anyone trusts it for selection.

Scores: ``spread_1`` / ``spread_all`` = max |ln rms_ratio| over +/-1 /
all non-zero scales -- the checkpoint-selection score (lower is better,
explode/collapse symmetric: 4x and 1/4x both read 1.39). ``verdict``
matches the established tf_render_gain convention instead: OK iff no pole
exceeds 1.5x base RMS AND none collapses below 1/3x -- so a legitimately
quiet-but-alive pole (yearn -2 reads 0.60x) still ships.
"""

from __future__ import annotations

import numpy as np
import torch

INFERENCE_CFG = 1.7  # MiniMaxMusic3ChunkDenoiseInner guidance_scale default


def rollout_finals(
    transformer: torch.nn.Module,
    network,
    condition: torch.Tensor,
    model_dir,
    device: torch.device,
    seed: int,
    scales: list[float],
    num_steps: int = 30,
    amp: bool = True,
    latent_len: int | None = None,
) -> dict[float, torch.Tensor]:
    """Run the shipped sampler to completion per scale; return final latents.

    Mirrors train_lora_music3.rollout_states exactly (same scheduler, CFG,
    branch handling) but keeps only the terminal state. Uses a private
    generator so neither data_rng nor traj_rng moves.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_dir), subfolder="scheduler", local_files_only=True
    )
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    dtype = torch.bfloat16 if amp else torch.float32
    cond = condition.to(device=device, dtype=dtype)
    if latent_len is None:
        latent_len = int(cond.shape[1])
    zeros = torch.zeros_like(cond)
    finals: dict[float, torch.Tensor] = {}
    for scale in scales:
        scheduler.set_timesteps(sigmas=sigmas, device=device)
        generator = torch.Generator(device=device.type).manual_seed(int(seed))
        latents = torch.randn(
            (1, 128, latent_len), device=device, dtype=dtype, generator=generator
        )
        network.set_lora_slider(float(scale))
        with network:
            for t in scheduler.timesteps:
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                if amp and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        vel_cond = transformer(latents, timestep, cond, return_dict=False)[0]
                        vel_uncond = transformer(latents, timestep, zeros, return_dict=False)[0]
                else:
                    vel_cond = transformer(latents, timestep, cond, return_dict=False)[0]
                    vel_uncond = transformer(latents, timestep, zeros, return_dict=False)[0]
                velocity = vel_uncond + INFERENCE_CFG * (vel_cond - vel_uncond)
                latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
        finals[float(scale)] = latents.detach().float().cpu()
    network.set_lora_slider(1.0)
    return finals


def load_vocoder(model_dir, device: torch.device):
    """Load just the vocoder (~the only pipeline piece missing in-trainer)."""
    from diffusers import MiniMaxMusic3Vocoder

    vocoder = MiniMaxMusic3Vocoder.from_pretrained(
        str(model_dir), subfolder="vocoder", torch_dtype=torch.bfloat16, local_files_only=True
    )
    return vocoder.to(device).eval()


@torch.no_grad()
def decode_rms(vocoder, latent: torch.Tensor) -> float:
    """RMS of the vocoder decode of one final latent."""
    with torch.no_grad():
        wave = vocoder(latent.to(device=vocoder.device, dtype=vocoder.dtype))
    audio = wave.detach().float().cpu().numpy()
    if audio.ndim > 1:
        audio = audio.mean(axis=tuple(range(audio.ndim - 1)))
    return float(np.sqrt(np.mean(np.square(audio))))


def vocoder_wave(vocoder, latent: torch.Tensor):
    """Differentiable decode: latents -> waveform (grad flows through)."""
    return vocoder(latent.to(device=vocoder.device, dtype=vocoder.dtype))


def band_energy(wave: torch.Tensor, sr: int = 44100, n_bands: int = 6):
    """Differentiable log-band energies of a waveform [T] or [C,T]."""
    m = wave.float().mean(dim=tuple(range(wave.ndim - 1)))
    spec = torch.fft.rfft(m).abs() ** 2 + 1e-9
    freqs = torch.fft.rfftfreq(m.shape[-1], 1 / sr)
    edges = np.array([0, 120, 400, 1200, 3500, 8000, 22050][: n_bands + 1])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        out.append(torch.sqrt(spec[mask].sum()))
    return torch.stack(out)


def render_metrics_penalty(
    wave_s,
    wave_0,
    tau: float = 0.30,
):
    """relu(|ln band-energy ratio| - tau)^2 summed over bands.

    Broadband level symmetry and per-band timbre drift in one differentiable
    term; zero inside the caps, steep outside (constraint, not driver).
    """
    bs = band_energy(wave_s)
    b0 = band_energy(wave_0).clamp_min(1e-8)
    ratio = torch.log(bs.clamp_min(1e-8)) - torch.log(b0)
    excess = torch.relu(ratio.abs() - tau)
    return (excess ** 2).sum()


def _latent_energy(latent: torch.Tensor) -> float:
    return float(latent.float().pow(2).mean().sqrt())


def gate_report(
    finals: dict[float, torch.Tensor],
    vocoder=None,
) -> dict:
    """Per-scale ratios vs scale 0 plus spread scores.

    Returns rows [{scale, rms|energy, ratio}], spread_1 (max |ln ratio| over
    +/-1) and spread_all (over every non-zero scale present).
    """
    if 0.0 not in finals:
        raise ValueError(f"gate needs a scale-0 final, got scales {sorted(finals)}")
    if vocoder is not None:
        def measure(latent: torch.Tensor) -> float:
            return decode_rms(vocoder, latent)
        key = "rms"
    else:
        measure = _latent_energy
        key = "energy"
    base = max(measure(finals[0.0]), 1e-9)
    rows = []
    ratios: dict[float, float] = {}
    for scale in sorted(finals):
        if scale == 0.0:
            continue
        value = measure(finals[scale])
        ratio = value / base
        ratios[scale] = ratio
        rows.append({"scale": scale, key: round(value, 5), "ratio_vs_zero": round(ratio, 4)})
    spread_1 = max(
        (abs(np.log(max(r, 1e-9))) for s, r in ratios.items() if abs(s) <= 1.0),
        default=0.0,
    )
    spread_all = max(abs(np.log(max(r, 1e-9))) for r in ratios.values())
    hottest = max((max(r, 1.0 / r) for r in ratios.values()), default=1.0)
    quietest = min((min(r, 1.0 / r) for r in ratios.values()), default=1.0)
    ok = hottest <= 1.5 and quietest >= (1.0 / 3.0)
    return {
        "measure": key,
        "rows": rows,
        "spread_1": round(float(spread_1), 4),
        "spread_all": round(float(spread_all), 4),
        "hottest_inv_ratio": round(float(hottest), 4),
        "verdict": (
            "OK"
            if ok
            else f"FAIL hottest={hottest:.2f}x quietest={quietest:.2f}x (caps: <=1.5x hot, >=1/3x quiet)"
        ),
    }


def run_gate(
    transformer: torch.nn.Module,
    network,
    condition: torch.Tensor,
    model_dir,
    device: torch.device,
    seed: int,
    scales: list[float],
    num_steps: int = 30,
    amp: bool = True,
    vocoder_box: list | None = None,
) -> dict:
    """One gate evaluation. vocoder_box: optional [None] cell that lazily
    holds the vocoder across calls so callers pay the load once."""
    finals = rollout_finals(
        transformer, network, condition, model_dir, device,
        seed=seed, scales=scales, num_steps=num_steps, amp=amp,
    )
    vocoder = None
    if vocoder_box is not None:
        if vocoder_box[0] is None:
            vocoder_box[0] = load_vocoder(model_dir, device)
        vocoder = vocoder_box[0]
    return gate_report(finals, vocoder=vocoder)
