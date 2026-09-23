"""Stamp-backed feature-space train step for winning_formulation().

Products project model residuals into adapter_rank features and call
``FormulationGame.step``. They do not reimplement D/G/VIC/optimizer wiring.

This is the shared product runner (paired-error + particle VIC on gmix).
``EndpointGame`` remains the bipolar teacher/predict path for locked_shared /
Music Arm B style endpoints. Distillation stays in ``fit_routed_down``.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import torch
from torch import nn


def _paired_bank(rank: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed) + 17)
    neutrals = torch.randn(32, rank, generator=generator)
    targets = neutrals + torch.randn(32, rank, generator=generator)
    return {"targets": targets, "neutrals": neutrals}


def _regularizer_penalty(regularizer, critic, real, fake, *, step: int):
    """Accept both GradientPenalty.penalty(...) and a rare callable form."""
    if callable(getattr(regularizer, "penalty", None)):
        out = regularizer.penalty(critic, real, fake, step=step, collect_stats=False)
        return out[0] if isinstance(out, tuple) else out
    out = regularizer(critic, real, fake, step=step)
    return out[0] if isinstance(out, tuple) else out


class FormulationGame:
    """One optimizer step of ``winning_formulation()`` in feature space.

    ``features`` must be ``[batch, adapter_rank]``. The product owns how those
    features are produced (velocity edit, prompt projection, whitened residual).
    Particles and the routed bridge live here unless the product passes its own.
    """

    def __init__(
        self,
        stamp,
        declared: dict[str, Any],
        *,
        seed: int = 7,
        device: str | torch.device = "cpu",
        bridge: nn.Module | None = None,
        particles: nn.Parameter | None = None,
        critic: nn.Module | None = None,
        extra_generator: Sequence[nn.Parameter] | Iterable[nn.Parameter] | None = None,
        critic_targets: torch.Tensor | None = None,
        critic_neutrals: torch.Tensor | None = None,
    ):
        torch.manual_seed(int(seed))
        self.stamp = stamp
        self.declared = dict(declared)
        self.device = torch.device(device)
        self.regularizer = stamp.regularizer()
        self.d_loss, self.g_loss, self.vic = stamp.losses()
        self.rank = int(stamp.spec["adapter_rank"])
        self.vic_weight = float(stamp.spec["vicreg_weight"])
        self.vic_batch = int(stamp.spec["particle_vic_batch"])
        betas = tuple(float(x) for x in stamp.spec["betas"])

        self.bridge = (bridge if bridge is not None else stamp.bridge()).to(self.device)
        if particles is None:
            parts = int(stamp.spec["parts"])
            particle_dim = int(stamp.spec["particle_dim"])
            self.particles = nn.Parameter(
                torch.randn(parts, particle_dim, device=self.device) * 0.02
            )
        else:
            self.particles = particles
            if self.particles.device != self.device:
                raise ValueError("particles must already live on the game device")

        if critic is None:
            bank = _paired_bank(self.rank, seed)
            targets = critic_targets if critic_targets is not None else bank["targets"]
            neutrals = critic_neutrals if critic_neutrals is not None else bank["neutrals"]
            self.critic = stamp.critic(
                targets.to(self.device),
                neutrals=neutrals.to(self.device) if neutrals is not None else None,
            )
        else:
            self.critic = critic
        self.critic = self.critic.to(self.device)

        generator_params = list(self.bridge.parameters()) + list(extra_generator or [])
        # When the product owns the bridge, generator_params may be empty except particles.
        param_groups = []
        if generator_params:
            param_groups.append({"params": generator_params, "lr": float(declared["g_lr"])})
        param_groups.append({"params": [self.particles], "lr": float(declared["particle_lr"])})
        self.opt_g = torch.optim.Adam(param_groups, betas=betas)
        self.opt_d = torch.optim.Adam(self.critic.parameters(), lr=float(declared["d_lr"]), betas=betas)
        self.edit_rms = float(getattr(self.critic, "edit_rms", 1.0))

    def step(self, index: int, features: torch.Tensor) -> dict[str, float]:
        if index < 1:
            raise ValueError("steps are numbered starting at 1")
        if features.ndim != 2 or features.shape[1] != self.rank:
            raise ValueError(f"features must be [batch, {self.rank}]")
        features = features.to(self.device)
        sigma = float(self.stamp.noise_std_at(index - 1, self.edit_rms))
        noise = torch.randn(features.shape[0], self.rank, device=self.device) * sigma

        self.critic.requires_grad_(True)
        self.opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_detached = noise + self.bridge(features, self.particles)
        adv_d = self.d_loss(self.critic(noise), self.critic(fake_detached))
        penalty = _regularizer_penalty(self.regularizer, self.critic, noise, fake_detached, step=index)
        loss_d = adv_d + penalty
        if not torch.isfinite(loss_d):
            raise FloatingPointError(f"non-finite critic loss at step {index}")
        loss_d.backward()
        self.opt_d.step()

        self.critic.requires_grad_(False)
        self.opt_g.zero_grad(set_to_none=True)
        fake = noise.detach() + self.bridge(features, self.particles)
        with torch.no_grad():
            real_score = self.critic(noise.detach())
        adv_g = self.g_loss(real_score, self.critic(fake))
        choice = torch.randperm(self.particles.shape[0], device=self.device)[: self.vic_batch]
        vic = self.vic(self.particles[choice])
        loss_g = adv_g + self.vic_weight * vic
        if not torch.isfinite(loss_g):
            raise FloatingPointError(f"non-finite generator loss at step {index}")
        loss_g.backward()
        self.opt_g.step()
        return {
            "step": float(index),
            "d_loss": float(loss_d.detach()),
            "g_loss": float(adv_g.detach()),
            "vic": float(vic.detach()),
            "sigma": sigma,
        }


def run_formulation_game(
    game: FormulationGame,
    steps: int,
    features_for_step: Callable[[int], torch.Tensor],
) -> list[dict[str, float]]:
    history = []
    for step in range(1, int(steps) + 1):
        history.append(game.step(step, features_for_step(step)))
    if not history or not torch.isfinite(torch.tensor(history[-1]["g_loss"])):
        raise RuntimeError("formulation game produced no finite generator loss")
    return history


def dummy_features(rank: int, batch: int, seed: int) -> Callable[[int], torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))

    def draw(step: int) -> torch.Tensor:
        del step
        return torch.randn(batch, rank, generator=generator)

    return draw
