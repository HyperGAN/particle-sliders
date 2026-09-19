"""The explicit D/G update, independent of model loading and GPU orchestration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import torch
import torch.nn.functional as F

from ..lm_adv import rp_d_loss, rp_g_loss
from .critic import conditional_cap, matching_loss, full_width_guard, bounded_policy_kl, pad_sequences
from .optimization import (gradient_vector, current_gradient, install_gradient,
    add_bounded_fm, gradient_statistics, adapter_pairs, snapshot_pairs,
    effective_norm, effective_distance, limit_effective_step, cosine_rate)


@dataclass(frozen=True)
class Recipe:
    name: str = "repaired"
    g_lr: float = .0005
    d_lr: float = .00075
    adversarial_weight: float = 1.
    fm_weight: float = 1.
    end_weight: float = 1.
    guard_weight: float = .1
    guard_radius: float = 1.
    policy_weight: float = .1
    policy_tolerance: float = .05
    paired_fraction: float = .1
    fm_gradient_limit: float = 10.
    gradient_norm_limit: float = 50.
    parameter_step_limit: float = 2.
    effective_relative_limit: float = .02
    effective_norm_floor: float = 1.
    schedule: str = "delayed_cosine"
    schedule_origin: int = 600
    schedule_horizon: int = 900
    schedule_floor: float = .05
    diagnostics_every: int = 25
    # Baseline arms disable each new mechanism explicitly, never by infinity.
    control_fm: bool = True
    control_effective: bool = True
    clip_mode: str = "norm"

    def validate(self):
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, float) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid recipe value {name}")
        for key in ("g_lr", "d_lr", "fm_gradient_limit", "gradient_norm_limit",
                    "parameter_step_limit", "effective_relative_limit", "effective_norm_floor"):
            if values[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if self.clip_mode not in ("norm", "value") or self.schedule not in ("constant", "delayed_cosine"):
            raise ValueError("Unknown update rule")
        if not 0 <= self.paired_fraction <= 1 or self.diagnostics_every < 0:
            raise ValueError("Invalid matching/accounting recipe")
        cosine_rate(0, origin=self.schedule_origin, horizon=self.schedule_horizon, floor=self.schedule_floor)


class GANEngine:
    """Each row is encoded once per phase; only one LM graph is retained.

    ``forward(row, with_policy)`` returns attached fake, optional end margins
    and policy logits. Row fields: real, condition, end_teacher, policy_teacher.
    """
    def __init__(self, network, critic, forward, rows, recipe, *, completed=0):
        recipe.validate()
        self.network, self.critic, self.forward, self.rows, self.recipe = network, critic, forward, rows, recipe
        self.parameters = [p for p in network.parameters() if p.requires_grad]
        self.g_optimizer = torch.optim.AdamW(self.parameters, lr=recipe.g_lr, betas=(0., .999), weight_decay=1e-6)
        self.d_optimizer = torch.optim.Adam(critic.parameters(), lr=recipe.d_lr, betas=(0., .999))
        self.completed = completed

    def update(self, indices):
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Minibatches need distinct rows")
        recipe, critic = self.recipe, self.critic
        # Rates are labeled by the update being applied: the declared horizon
        # itself receives the floor, including short diagnostic continuations.
        lr_scale = (1. if recipe.schedule == "constant" else cosine_rate(self.completed + 1,
            origin=recipe.schedule_origin, horizon=recipe.schedule_horizon, floor=recipe.schedule_floor))
        for optimizer, rate in ((self.g_optimizer, recipe.g_lr), (self.d_optimizer, recipe.d_lr)):
            for group in optimizer.param_groups:
                group["lr"] = rate * lr_scale
        selected = [self.rows[i] for i in indices]
        critic.requires_grad_(True)
        self.d_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake, mask = pad_sequences([self.forward(row, False)["fake"] for row in selected])
            real, real_mask = pad_sequences([row["real"].to(fake) for row in selected])
            condition, condition_mask = pad_sequences([row["condition"].to(fake) for row in selected])
        if not torch.equal(mask, real_mask) or not torch.equal(mask, condition_mask):
            raise ValueError("Student, teacher and neutral condition spans must align")
        penalty, cap_stats = conditional_cap(critic, real, fake, mask, condition)
        d_adv = rp_d_loss(critic(real, mask, condition), critic(fake, mask, condition))
        d_loss = d_adv + penalty
        if not torch.isfinite(d_loss):
            raise FloatingPointError("Nonfinite discriminator loss")
        d_loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in critic.parameters() if p.grad is not None):
            raise FloatingPointError("Nonfinite discriminator gradient")
        self.d_optimizer.step()
        critic.requires_grad_(False)
        with torch.no_grad():
            real_features = critic.features(real, mask, condition)
            fake_features = critic.features(fake, mask, condition)
            real_mean, fake_mean = real_features.mean(0), fake_features.mean(0)
            real_scores = critic(real, mask, condition)

        self.g_optimizer.zero_grad(set_to_none=True)
        fm_gradient = torch.zeros_like(current_gradient(self.parameters))
        diagnostic = bool(recipe.diagnostics_every and (self.completed + 1) % recipe.diagnostics_every == 0)
        vectors, totals, policy_values = {}, {}, []
        for local_index, row in enumerate(selected):
            result = self.forward(row, recipe.policy_weight > 0)
            fake_row = result["fake"]
            target, context = row["real"].to(fake_row), row["condition"].to(fake_row)
            features = critic.features(fake_row, condition=context)
            terms = {
                "adversarial": recipe.adversarial_weight * rp_g_loss(critic(fake_row, condition=context), real_scores[local_index:local_index+1]),
                "fm": recipe.fm_weight * matching_loss(features, fake_mean, real_mean,
                    real_features[local_index:local_index+1], paired_fraction=recipe.paired_fraction),
            }
            if recipe.end_weight:
                terms["end"] = recipe.end_weight * F.mse_loss(result["end"], row["end_teacher"].to(result["end"]))
            if recipe.guard_weight:
                terms["full_width"] = recipe.guard_weight * full_width_guard(fake_row, target,
                    critic.input_scale, radius=recipe.guard_radius)
                if 'continuation_teacher' in row:
                    terms['continuation_full_width'] = recipe.guard_weight * full_width_guard(
                        result['continuation'], row['continuation_teacher'].to(fake_row),
                        critic.input_scale, radius=recipe.guard_radius)
            if recipe.policy_weight:
                policy_loss, kl = bounded_policy_kl(result["policy"], row["policy_teacher"].to(result["policy"]),
                    tolerance=recipe.policy_tolerance)
                terms["policy"] = recipe.policy_weight * policy_loss
                policy_values.extend(kl.cpu().flatten().tolist())
            weighted = {k: v / len(selected) for k, v in terms.items()}
            if not all(torch.isfinite(v) for v in weighted.values()):
                raise FloatingPointError("Nonfinite generator loss component")
            if diagnostic:
                for key, value in weighted.items():
                    vector = gradient_vector(value, self.parameters)
                    vectors[key] = vectors.get(key, torch.zeros_like(vector)) + vector
            if recipe.control_fm:
                fm_gradient.add_(gradient_vector(weighted["fm"], self.parameters))
                sum(v for k, v in weighted.items() if k != "fm").backward()
            else:
                sum(weighted.values()).backward()
            for key, value in weighted.items():
                totals[key] = totals.get(key, 0.) + float(value.detach())

        if recipe.control_fm:
            assembled, fm_stats = add_bounded_fm(current_gradient(self.parameters), fm_gradient, recipe.fm_gradient_limit)
            install_gradient(self.parameters, assembled)
        else:
            fm_stats = {"raw_norm": None, "factor": 1., "applied_norm": None, "status": "unlimited baseline"}
        pre_clip = float(current_gradient(self.parameters).double().norm())
        if not math.isfinite(pre_clip):
            raise FloatingPointError("Nonfinite assembled generator gradient")
        if recipe.clip_mode == "norm":
            torch.nn.utils.clip_grad_norm_(self.parameters, recipe.gradient_norm_limit, error_if_nonfinite=True)
        else:
            torch.nn.utils.clip_grad_value_(self.parameters, 1.)
        post_clip = float(current_gradient(self.parameters).double().norm())
        pairs = adapter_pairs(self.network)
        before = snapshot_pairs(pairs)
        before_parameters = [p.detach().clone() for p in self.parameters]
        old_effective_norm = effective_norm(before)
        self.g_optimizer.step()
        # Keep the proven parameter safeguard as well as the invariant norm.
        from analysis.gan_bcap.parameter_step_limit import bound_step
        parameter_limit = bound_step(self.parameters, before_parameters, recipe.parameter_step_limit)
        if recipe.control_effective:
            maximum = recipe.effective_relative_limit * max(old_effective_norm, recipe.effective_norm_floor)
            effective_limit = limit_effective_step(pairs, before, maximum=maximum)
        else:
            effective_limit = {"actual_norm": effective_distance(pairs, before), "factor": 1., "maximum": None}
        actual_parameter_step = math.sqrt(sum(float((p.detach().double() - old.double()).square().sum())
                                              for p, old in zip(self.parameters, before_parameters)))
        self.completed += 1
        return {"step": self.completed, "rows": indices, "losses": totals,
                "d_loss": float(d_loss.detach()), "d_adversarial": float(d_adv.detach()),
                "cap": cap_stats, "lr_scale": lr_scale,
                "g_lr": self.g_optimizer.param_groups[0]["lr"], "d_lr": self.d_optimizer.param_groups[0]["lr"],
                "gradients": gradient_statistics(vectors) if diagnostic else None,
                "gradient_accounting_status": "measured" if diagnostic else "not_scheduled",
                "fm_gradient": fm_stats, "gradient_before_clip": pre_clip, "gradient_after_clip": post_clip,
                "parameter_limit": parameter_limit, "parameter_step": actual_parameter_step,
                "effective_limit": effective_limit, "effective_weight_norm_before": old_effective_norm,
                "policy_kl_mean": sum(policy_values) / len(policy_values) if policy_values else None,
                "policy_kl_max": max(policy_values) if policy_values else None,
                "collapse": None, "collapse_status": "requires_free_running_audio_evaluation"}
