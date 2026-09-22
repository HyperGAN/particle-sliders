"""Model-independent deterministic bipolar game with operative recipe settings.

Consumers supply frozen teachers and differentiable predictions on the same
states, in the same declared coordinates. No hidden model/critic substitution.
"""
import math
import torch
import torch.nn.functional as F

from .reference import rp_d_loss, rp_g_loss
from .teachers import lm_faithful_guard_e


def teacher_poles(recipe, positive, negative, neutral, *, leak_dir=None, slider_dir=None):
    if recipe.teacher == "raw_poles":
        return positive, negative, "raw_poles"
    if recipe.teacher != "faithful_guard_e":
        raise ValueError("Unsupported teacher")
    if leak_dir is None:
        # Same explicit posture as Music: no declared leftover means no removal.
        return positive, negative, "no_declared_leak_raw_poles"
    if slider_dir is None:
        raise ValueError("faithful_guard_e requires an explicit slider direction with its leak direction")
    plus, minus = lm_faithful_guard_e(positive, negative, neutral,
        leak_dir, slider_dir=slider_dir)
    mode = "guard_fallback_raw_poles" if plus is positive else "guard_admitted"
    return plus, minus, mode


def endpoint_terms(pred_plus, pred_minus, target_plus, target_minus):
    # Music's lm_slider_loss is the sum of the two endpoint MSEs.
    # Toy cover is the same sum on shared residual centers. With no stochastic
    # prior, centers equal predictions: both terms coincide, deliberately.
    pole = F.mse_loss(pred_plus, target_plus) + F.mse_loss(pred_minus, target_minus)
    return {"pole": pole, "cover": pole}


def gradient_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


class EndpointGame:
    def __init__(self, adapter, critic, recipe, count, *, seed=7, microbatch=1):
        recipe.validate()
        if count < 1 or microbatch not in (1, 2, 4, 8) or recipe.batch % microbatch:
            raise ValueError("Invalid dataset or microbatch")
        self.adapter, self.critic, self.recipe = adapter, critic, recipe
        self.count, self.microbatch = count, microbatch
        self.generators = {phase: torch.Generator().manual_seed(seed + i * 1009)
                           for i, phase in enumerate(("d", "g"))}
        self.g = torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad],
            lr=recipe.g_lr, betas=(recipe.beta1, recipe.beta2), weight_decay=0)
        self.d = torch.optim.Adam(critic.parameters(), lr=recipe.d_lr,
            betas=(recipe.beta1, recipe.beta2), weight_decay=0)
        self.capper = recipe.regularizer()

    def update(self, predict, targets, step, prefetch=None):
        """Callbacks return (plus, minus) [batch, flattened full field].

        predict holds its +/- adapter scale through backward (including model
        checkpoint recomputation). A context manager per sign avoids retaining
        two transformer graphs at once.
        """
        if step < 1:
            raise ValueError("Updates start at one")
        rows = {p: torch.randint(self.count, (self.recipe.batch,), generator=g).tolist()
                for p, g in self.generators.items()}
        if prefetch:
            prefetch(rows["d"] + rows["g"])
        totals = dict(d_adv=0., d_penalty=0., g_adv=0., pole=0., cover=0.)
        self.critic.requires_grad_(True)
        self.d.zero_grad(set_to_none=True)
        for start in range(0, self.recipe.batch, self.microbatch):
            ids = rows["d"][start:start+self.microbatch]
            real = targets(ids)
            for sign, target in zip((1., -1.), real):
                with torch.no_grad(), predict(ids, sign) as fake:
                    fake = fake.detach()
                adv = rp_d_loss(self.critic(target), self.critic(fake))
                cap, _ = self.capper.penalty(self.critic, target, fake, step=step, collect_stats=False)
                weight = len(ids) / (2 * self.recipe.batch)
                loss = (adv + cap) * weight
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite critic loss")
                loss.backward()
                totals["d_adv"] += float(adv.detach()) * weight
                totals["d_penalty"] += float(cap.detach()) * weight
        d_norm = gradient_norm(self.critic.parameters())
        if not math.isfinite(d_norm):
            raise FloatingPointError("Non-finite critic gradient")
        self.d.step()
        self.d.zero_grad(set_to_none=True)
        self.critic.requires_grad_(False)
        self.g.zero_grad(set_to_none=True)
        for start in range(0, self.recipe.batch, self.microbatch):
            ids = rows["g"][start:start+self.microbatch]
            for sign, target in zip((1., -1.), targets(ids)):
                with predict(ids, sign) as fake:
                    adv = rp_g_loss(self.critic(target).detach(), self.critic(fake))
                    mse = F.mse_loss(fake, target)
                    weight = len(ids) / self.recipe.batch
                    # RpGAN averages both poles; cover and pole sum both MSEs.
                    loss = weight * (adv / 2 + (self.recipe.pole_weight + self.recipe.cover_weight) * mse)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite generator loss")
                    loss.backward()
                    totals["g_adv"] += float(adv.detach()) * weight / 2
                    totals["pole"] += float(mse.detach()) * weight
                    totals["cover"] += float(mse.detach()) * weight
        g_norm = gradient_norm(self.adapter.parameters())
        if not math.isfinite(g_norm):
            raise FloatingPointError("Non-finite generator gradient")
        self.g.step()
        return dict(step=step, **totals, d_grad_norm=d_norm, g_grad_norm=g_norm,
            vic=0., noise_min=0., noise_max=0., d_rows=rows["d"], g_rows=rows["g"])

    def state_dict(self):
        return dict(recipe=self.recipe.to_dict(), count=self.count,
            critic=self.critic.state_dict(), g=self.g.state_dict(), d=self.d.state_dict(),
            generators={k: v.get_state() for k, v in self.generators.items()})

    def load_state_dict(self, state):
        if state["recipe"] != self.recipe.to_dict() or state["count"] != self.count:
            raise ValueError("Game identity changed")
        self.critic.load_state_dict(state["critic"])
        self.g.load_state_dict(state["g"])
        self.d.load_state_dict(state["d"])
        for key, value in state["generators"].items():
            self.generators[key].set_state(value.cpu())
