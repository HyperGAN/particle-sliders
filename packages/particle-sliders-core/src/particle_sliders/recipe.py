"""Serializable formulation knobs. Unsupported combinations fail before allocation."""
from dataclasses import asdict, dataclass, fields
import math

from .grad_regularizers import GradientPenalty


@dataclass(frozen=True)
class SliderRecipe:
    loss: str = "rpgan_logistic"
    teacher: str = "faithful_guard_e"
    grad_arm: str = "b_cap"
    grad_coeff: float = 1.0
    grad_kappa: float = 1.0
    grad_norm: str = "l2"
    grad_every: int = 1
    fm_weight: float = 0.0
    cover_weight: float = 1.0
    pole_weight: float = 1.0
    particles: int = 1
    vicreg_weight: float = 0.0
    noise_std: float = 0.0
    noise_hold: float = 0.0
    batch: int = 8
    g_lr: float = 2e-5
    d_lr: float = 9e-4
    beta1: float = 0.0
    beta2: float = 0.999
    ema: float = 0.995

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown recipe settings: {sorted(unknown)}")
        result = cls(**value)
        result.validate()
        return result

    def to_dict(self):
        self.validate()
        return asdict(self)

    def validate(self):
        if self.loss != "rpgan_logistic" or self.teacher not in ("faithful_guard_e", "raw_poles"):
            raise ValueError("Unsupported loss/teacher")
        # This endpoint engine has no feature interface or stochastic prior.
        # Reject unused switches instead of advertising metadata-only options.
        if self.fm_weight != 0 or self.vicreg_weight != 0 or self.noise_std != 0 or self.noise_hold != 0:
            raise ValueError("The deterministic endpoint engine requires FM/VIC/noise/hold off")
        if isinstance(self.particles, bool) or self.particles not in (0, 1):
            raise ValueError("The deterministic endpoint engine supports zero or one particle")
        if isinstance(self.batch, bool) or not isinstance(self.batch, int) or self.batch < 1:
            raise ValueError("batch must be a positive integer")
        if isinstance(self.grad_every, bool) or not isinstance(self.grad_every, int) or self.grad_every < 1:
            raise ValueError("grad_every must be a positive integer")
        for name in ("cover_weight", "pole_weight", "grad_coeff", "grad_kappa", "g_lr", "d_lr", "beta1", "beta2", "ema"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1 or not 0 <= self.ema < 1:
            raise ValueError("Invalid optimizer/EMA coefficients")
        if not self.g_lr or not self.d_lr:
            raise ValueError("Learning rates must be positive")
        self.regularizer()

    def regularizer(self):
        """ParticleGAN ``GradientPenalty`` for this endpoint recipe."""
        return GradientPenalty(arm=self.grad_arm, coeff=self.grad_coeff,
            kappa=self.grad_kappa, norm=self.grad_norm, lazy_k=self.grad_every,
            method="autograd")

    def require_locked_shared(self):
        self.validate()
        expected = SliderRecipe()
        keys = ("loss", "teacher", "grad_arm", "grad_coeff", "grad_kappa", "grad_norm",
                "grad_every", "fm_weight", "cover_weight", "pole_weight", "vicreg_weight",
                "noise_std", "noise_hold")
        differences = {k: {"expected": getattr(expected, k), "actual": getattr(self, k)}
                       for k in keys if getattr(self, k) != getattr(expected, k)}
        if differences:
            raise ValueError(f"locked_shared recipe drift: {differences}")


def require_same_critic(before, after):
    """Identity includes architecture kwargs and source hash, not learned weights."""
    if before != after:
        raise ValueError(f"Critic changed: before={before!r}, after={after!r}")
