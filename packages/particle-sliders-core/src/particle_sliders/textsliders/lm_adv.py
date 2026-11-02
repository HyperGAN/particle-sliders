"""Training-only adversarial loss signal for LM sliders (ParticleGAN-style).

Inference stays pure LoRA: the discriminator below is built, stepped, and
discarded inside ``train_lm_slider_music3.train``. It is never saved into the
checkpoint — ``LoRANetwork.save_weights`` only serializes LoRA params — and
the sidecar records only the hyperparameters, so a shipped file renders
exactly as before with ``adv_weight=0``.

Real = teacher hidden deltas (``tgt - neu``), fake = student hidden deltas
    (``pred - neu``), scored by a small head: ``LMDiscriminator`` (MLP on the
    last-token delta) or ``SpanTransformerD`` (set-transformer over the
    lyric-span delta sequence — same game, sees prefix behavior the MLP is
    blind to). Objective is RpGAN logistic pairing
with the one-sided cap gradient penalty (``b_cap``: ``relu(||grad|| - 1)^2``
on reals + fakes, coeff 1.0) from 255BITS/ParticleGAN, whose 420-run study
picked it over zero-centered R1/R2 for keeping a sharp discriminator usable:

- ``relu`` cap is free below the margin, so D keeps usable slope instead of
  being driven toward flatness (what R1/R2's ``||g||^2`` minimum does);
- the sample-point penalty still damps the game, which is what stops a
  sharp D from stranding modes.

Transferred optimizer settings: Adam ``beta1=0``, D LR 1.5x the LoRA LR.
Only the optional row-miner table receives sparse row updates; LoRA
parameters receive gradients from each prompt.

The fixed ``scaled`` stem calibrates hidden coordinates once from teacher
component RMS, so zero student deltas have usable gradients and the cap
has explicit units. Unit/lognorm stems remain for legacy experiment cards.
Adaptive prompt-row mining lives in ``lm_particles``; it reweights fixed
caption queries and is not the source algorithm's movable latent prior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Default D lr multiplier over the LoRA lr (ParticleGAN ``d_lr_mult``).
D_LR_MULT = 1.5

#: Default one-sided cap coefficient (ParticleGAN ``b_cap`` default).
CAP_COEFF = 1.0

#: Cap margin kappa: penalty is free while ``||grad|| <= kappa``.
CAP_KAPPA = 1.0

#: Rows with raw norm at or below this are degenerate (a fresh LoRA's delta
#: is exactly zero) and carry no discriminative signal. They are masked out
#: of the cap penalty: unit-normalizing them divides by the eps clamp, which
#: would otherwise manufacture ~1e8 phantom gradients.
DEGENERATE_NORM = 1e-6

#: Scale for the log-norm channel of ``unit_lognorm`` input mode: raw hidden
#: norms sit in the tens (log ~ 2-4), so /4 keeps the channel O(1) next to
#: the unit-vector entries and keeps input grad norms near the b_cap margin.
LOGNORM_SCALE = 4.0

#: Accepted ``LMDiscriminator`` input modes.
IN_MODES = ("unit", "unit_lognorm", "scaled")

#: Accepted discriminator architectures (``--adv_arch``).
ADV_ARCHES = ("mlp", "tx")


def _stem_embed(
    delta: torch.Tensor, in_mode: str, input_scale: torch.Tensor | float = 1.0
) -> torch.Tensor:
    """Float32 legacy unit/lognorm or fixed calibrated-coordinate stem.

    Applied per position for sequence inputs. The fixed scale comes from
    teacher statistics; it never depends on the current student vector.
    """
    raw = delta.float()
    if in_mode == "scaled":
        # A fixed teacher scale preserves both direction and magnitude, with
        # a finite, constant Jacobian at a fresh LoRA's zero delta.
        return raw / input_scale
    norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    unit = raw / norm
    if in_mode == "unit_lognorm":
        logn = norm.clamp_min(DEGENERATE_NORM).log() / LOGNORM_SCALE
        unit = torch.cat([unit, logn], dim=-1)
    return unit


class _ScaledInputD(nn.Module):
    """Shared fixed-coordinate calibration for training-only critics.

    ``scaled`` uses teacher per-component RMS. The cap is measured in these
    calibrated coordinates, so its margin does not change with hidden units.
    Calibration is explicit and happens once before either optimizer runs.
    """

    def _init_input_scale(self, input_scale: float) -> None:
        if not 0.0 < float(input_scale) < float("inf"):
            raise ValueError("input_scale must be positive and finite")
        self.register_buffer("input_scale", torch.tensor(float(input_scale)))

    @torch.no_grad()
    def calibrate_input_scale(
        self, real: torch.Tensor, mask: torch.Tensor | None = None
    ) -> float:
        values = real.detach().float()
        if mask is not None:
            if tuple(mask.shape) != tuple(values.shape[:-1]):
                raise ValueError("calibration mask must match all non-feature dimensions")
            values = values[mask.bool()]
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("calibration requires nonempty finite teacher deltas")
        scale = values.square().mean().sqrt()
        if not bool(scale > 0):
            raise ValueError("calibration requires a nonzero teacher delta")
        self.input_scale.copy_(scale.to(self.input_scale))
        return float(self.input_scale)

    def _embed(self, delta: torch.Tensor) -> torch.Tensor:
        return _stem_embed(delta, self.in_mode, self.input_scale).to(
            next(self.parameters()).dtype
        )


class LMDiscriminator(_ScaledInputD):
    """Small MLP: hidden delta (B, H) -> scalar score (B,).

    Legacy ``unit`` normalizes each delta and is blind to its magnitude;
    ``unit_lognorm`` adds ``log||delta|| / LOGNORM_SCALE``. Both have a
    singular direction derivative at the zero delta of a fresh LoRA, so
    neither is a stable coordinate system for the repaired GAN.

    ``scaled`` is the repaired GAN coordinate system: divide every vector
    by one fixed, teacher-calibrated component RMS. It retains magnitude
    and has a finite constant derivative at zero; use ``calibrate_input_scale``
    before training. Unit modes remain available for old experiment cards.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        in_mode: str = "unit",
        input_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if in_mode not in IN_MODES:
            raise ValueError(f"in_mode must be one of {IN_MODES}, got {in_mode!r}")
        self.in_mode = in_mode
        self._init_input_scale(input_scale)
        layers: list[nn.Module] = []
        dim = in_dim + (1 if in_mode == "unit_lognorm" else 0)
        for _ in range(max(1, n_hidden)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, delta: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Score deltas ((B,) out) in the configured input coordinates.

        ``mask`` must be None: this head is last-token only — use
        ``SpanTransformerD`` with a padding mask for sequences.
        """
        if mask is not None:
            raise ValueError("LMDiscriminator takes single vectors; got a mask")
        return self.net(self._embed(delta)).squeeze(-1)

    def features(self, delta: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Post-activation features after the first hidden layer ((B, H)).

        Used by feature-matching: matching batch-mean features grounds the
        student to the teacher distribution (including scale, via the
        log-norm channel) instead of merely outranking it — the fix for
        ranking-only overshoot, where the student grows unboundedly along
        D's weight vector past the teacher point.
        """
        if mask is not None:
            raise ValueError("LMDiscriminator takes single vectors; got a mask")
        return self.net[1](self.net[0](self._embed(delta)))


class _SetBlock(nn.Module):
    """Pre-norm self-attention + SiLU FFN residual block (no posemb).

    Attention is handwritten matmul/softmax, not ``nn.MultiheadAttention``:
    the b_cap penalty backprops through D with ``create_graph=True`` and the
    memory-efficient SDPA backend has no double-backward formula
    (``_scaled_dot_product_efficient_attention_backward is not implemented``).
    Plain ops differentiate twice everywhere. At width 128 the sequence is
    tiny; flash buys nothing.
    """

    def __init__(self, width: int, n_heads: int, ffn_mult: float = 2.0) -> None:
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width {width} must split over {n_heads} heads")
        self.n_heads = int(n_heads)
        # The default float32 eps amplifies near-zero fresh-LoRA features by
        # thousands. A fixed eps also keeps cap double backward well scaled.
        self.norm1 = nn.RMSNorm(width, eps=1e-3)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.RMSNorm(width, eps=1e-3)
        hidden = max(1, int(round(width * ffn_mult)))
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, width),
        )

    def forward(
        self, h: torch.Tensor, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        bsz, seq, width = h.shape
        heads = self.n_heads
        dh = width // heads
        q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
        q = q.view(bsz, seq, heads, dh).transpose(1, 2)
        k = k.view(bsz, seq, heads, dh).transpose(1, 2)
        v = v.view(bsz, seq, heads, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (dh**0.5)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :].bool(),
                torch.finfo(scores.dtype).min,
            )
        a = torch.softmax(scores.float(), dim=-1).to(dtype=h.dtype) @ v
        a = a.transpose(1, 2).reshape(bsz, seq, width)
        h = h + self.out(a)
        return h + self.ffn(self.norm2(h))


class SpanTransformerD(_ScaledInputD):
    """Set-transformer D over the lyric-span delta sequence.

    Real = teacher span deltas (``pos_full - neu_full`` on the span),
    fake = student span deltas (``pred_full - neu_full`` on the span),
    scored as one scalar per row. Same RpGAN + b_cap game as the MLP head,
    different eye: the MLP judges the shifted last token, this judges how
    the caption contextualizes the whole lyric sheet — prefix behavior the
    last-token head is structurally blind to.

    Soundness constraints (all load-bearing, all tested):

    - **Span only, never the full prefix.** Student encodes the neutral
      caption, teacher the pos caption: different token strings, different
      lengths. A full-sequence D would separate by content/length in one
      step and G could never answer (it must keep neutral tokens). The
      lyric span is token-ID-identical across captions (asserted at setup),
      so per-position deltas are meaningful there.
    - **No positional embeddings.** Span offsets differ per caption; an
      absolute position channel would be a spurious cue G cannot move
      (token positions are fixed per caption). Permutation invariance is
      the point — attention still contextualizes by content before pooling.
    - **Same per-token input stem as the MLP**. Padded positions are zeroed before the
      projection and excluded from attention and pooling, so their input
      grads are exactly 0 and neither the score nor the penalty sees padding.
    - **Capacity-matched to the MLP** (~0.8M params at width 128 / 2 layers
      vs ~1.1M): down-projection dominates. A data-starved game (<= 4 rows)
      must not hand D a bigger gun.

    Readout is masked mean-pool, optionally concatenated with the final
    valid token (``readout="mean_last"``). The latter gives the appended
    audio-start token its own channel instead of diluting it across the
    lyric sheet. There are no batch-dependent features: single-row G
    accumulation and batched D evaluation score exactly the same function.
    ``features`` returns the pooled pre-logit vector for FM.
    """

    def __init__(
        self,
        in_dim: int,
        width: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        in_mode: str = "unit_lognorm",
        ffn_mult: float = 2.0,
        input_scale: float = 1.0,
        readout: str = "mean",
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if in_mode not in IN_MODES:
            raise ValueError(f"in_mode must be one of {IN_MODES}, got {in_mode!r}")
        self.in_mode = in_mode
        self._init_input_scale(input_scale)
        if readout not in ("mean", "mean_last"):
            raise ValueError("readout must be 'mean' or 'mean_last'")
        self.readout = readout
        self.width = int(width)
        stem_dim = in_dim + (1 if in_mode == "unit_lognorm" else 0)
        self.proj = nn.Linear(stem_dim, width)
        self.blocks = nn.ModuleList(
            [_SetBlock(width, n_heads, ffn_mult) for _ in range(n_layers)]
        )
        feature_dim = width * (2 if readout == "mean_last" else 1)
        self.out_norm = nn.RMSNorm(feature_dim, eps=1e-3)
        self.head = nn.Linear(feature_dim, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if in_mode == "scaled":
            # A fresh LoRA has zero deltas. Mapping this origin to a smooth
            # O(1) state prevents stacked RMSNorm Jacobians from multiplying
            # epsilon inverses before D has taken its first update.
            nn.init.normal_(self.proj.bias, std=0.5)

    def _pooled(
        self, seq: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        h = self._embed(seq)
        if mask is not None:
            m = mask.to(dtype=dtype).unsqueeze(-1)
            h = h * m
            key_mask = ~mask.bool()
        else:
            m = None
            key_mask = None
        h = self.proj(h.to(dtype))
        for blk in self.blocks:
            h = blk(h, key_mask)
        if m is None:
            pooled = h.mean(dim=1)
            last = h[:, -1]
        else:
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            positions = torch.arange(h.shape[1], device=h.device).expand(h.shape[:2])
            last_idx = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
            last = h[torch.arange(h.shape[0], device=h.device), last_idx.clamp_min(0)]
            last = last * (last_idx >= 0).unsqueeze(-1)
        if self.readout == "mean_last":
            return torch.cat([pooled, last], dim=-1)
        return pooled

    def forward(
        self, seq: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score span-delta sequences ((B,) out). ``mask`` (B, S) bool,
        True = valid span position (padding mask for batched rows of
        different span lengths; None = no padding)."""
        pooled = self._pooled(seq, mask)
        # D is trained on batches, G uses one LM row graph at a time. A
        # minibatch-stddev feature would change the game between those phases
        # and invalidate per-sample cap gradients (scores couple across rows).
        return self.head(self.out_norm(pooled)).squeeze(-1)

    def features(
        self, seq: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pooled pre-logit vectors ((B, W)) for feature matching."""
        return self._pooled(seq, mask)


class RowConditionalD(nn.Module):
    """Projection critic conditioned on the prompt row being matched.

    Unconditional hidden-delta matching admits a permutation of teacher
    deltas across captions. The projection term ties each score to its
    caption row while keeping the condition fixed for a real/fake pair.
    Row IDs are discrete context; b_cap differentiates only hidden deltas.
    """

    def __init__(self, base: nn.Module, n_conditions: int, feature_dim: int) -> None:
        super().__init__()
        if n_conditions <= 0 or feature_dim <= 0:
            raise ValueError("n_conditions and feature_dim must be positive")
        self.base = base
        self.embedding = nn.Embedding(n_conditions, feature_dim)
        self.feature_dim = int(feature_dim)
        nn.init.normal_(self.embedding.weight, std=1.0)

    @property
    def in_mode(self):
        return self.base.in_mode

    @property
    def input_scale(self):
        return self.base.input_scale

    def calibrate_input_scale(
        self, real: torch.Tensor, mask: torch.Tensor | None = None
    ) -> float:
        return self.base.calibrate_input_scale(real, mask)

    def features(
        self, delta: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.base.features(delta, mask)

    def forward(
        self,
        delta: torch.Tensor,
        mask: torch.Tensor | None = None,
        row_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if row_ids is None:
            raise ValueError("RowConditionalD requires row_ids for every sample")
        if row_ids.ndim != 1 or row_ids.shape[0] != delta.shape[0]:
            raise ValueError("row_ids must have shape (batch,)")
        if row_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("row_ids must be integer indices")
        row_ids = row_ids.to(device=self.embedding.weight.device)
        if bool((row_ids < 0).any()) or bool((row_ids >= self.embedding.num_embeddings).any()):
            raise ValueError("row_ids are outside the configured condition range")
        features = self.features(delta, mask)
        if features.shape[-1] != self.feature_dim:
            raise ValueError("feature_dim does not match the base discriminator features")
        projection = (features * self.embedding(row_ids)).sum(dim=-1) / self.feature_dim**0.5
        return self.base(delta, mask) + projection


def feature_matching_loss(
    discriminator: nn.Module,
    fake: torch.Tensor,
    real: torch.Tensor,
) -> torch.Tensor:
    """``MSE(mean(feat_fake), mean(feat_real))`` over the batch.

    D params must be frozen by the caller (grad flows to the student only);
    reals are detached inside. Needs batch > 1 to mean anything — pair with
    ``--adv_batch`` >= 4. Works for either head (``features`` returns (B, F)
    in both); padded sequence batches should call ``features(seq, mask)``
    directly instead of this helper.
    """
    with torch.no_grad():
        target = discriminator.features(real.detach()).float().mean(dim=0)
    return F.mse_loss(discriminator.features(fake).float().mean(dim=0), target)


def _grad_norm_per_sample(logits_sum, x: torch.Tensor) -> torch.Tensor:
    """``||grad_x D(x)||`` per sample (L2), differentiable w.r.t. D params."""
    grad = torch.autograd.grad(logits_sum, x, create_graph=True)[0]
    return torch.sqrt(grad.pow(2).flatten(1).sum(dim=1) + 1e-12)


def cap_penalty(
    discriminator: nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    coeff: float = CAP_COEFF,
    kappa: float = CAP_KAPPA,
    mask_real: torch.Tensor | None = None,
    mask_fake: torch.Tensor | None = None,
    condition_real: torch.Tensor | None = None,
    condition_fake: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """One-sided cap penalty at the samples (``b_cap``).

    ``(coeff / 2) * (E[relu(n_r - kappa)^2] + E[relu(n_f - kappa)^2])`` with
    ``n = ||grad_x D(x)||``. Recomputes its own graph internally (detached
    clones with ``requires_grad``), so callers need no ``requires_grad``
    on their batches. Differentiable w.r.t. D params via ``create_graph``.
    Legacy unit stems mask rows at or below ``DEGENERATE_NORM`` because of
    their singular Jacobian. Fixed ``scaled`` coordinates include zero
    fakes and measure norms in calibrated units (raw Jacobian times the
    fixed input scale). Optional bool ``mask_*`` ((B, S),
    True = valid) are forwarded to sequence heads; padded positions are
    zeroed by the head so their input grads are exactly 0 and the norm is
    unaffected by padding. ``condition_*`` supplies fixed row IDs to a
    conditional head; IDs are never concatenated to differentiated inputs.
    """
    if coeff <= 0.0:
        zero = torch.zeros((), device=x_real.device, dtype=torch.float32)
        return zero, {"applied": False, "pen": 0.0}
    side_means = []
    stats = {"applied": False, "pen": 0.0}
    scaled = getattr(discriminator, "in_mode", None) == "scaled"
    for side, batch, mask, condition in (
        ("real", x_real, mask_real, condition_real),
        ("fake", x_fake, mask_fake, condition_fake),
    ):
        det = batch.detach().float()
        keep = (
            torch.ones(det.shape[0], device=det.device, dtype=torch.bool)
            if scaled else det.flatten(1).norm(dim=-1) > DEGENERATE_NORM
        )
        stats[f"{side}_kept"] = int(keep.sum())
        if not bool(keep.any()):
            continue
        kept = det[keep].clone().requires_grad_(True)
        kept_mask = None if mask is None else mask[keep]
        if condition is None:
            logits = discriminator(kept) if kept_mask is None else discriminator(kept, kept_mask)
        else:
            if condition.ndim != 1 or condition.shape[0] != det.shape[0]:
                raise ValueError("cap conditions must have shape (batch,)")
            kept_condition = condition.to(device=det.device)[keep]
            logits = discriminator(kept, kept_mask, row_ids=kept_condition)
        norms = _grad_norm_per_sample(logits.sum(), kept)
        if scaled:
            # x_calibrated = x_raw / s, hence dD/dx_calibrated = s*dD/dx_raw.
            norms = norms * discriminator.input_scale.detach()
        stats[f"{side}_grad_mean"] = float(norms.detach().mean())
        stats[f"{side}_grad_max"] = float(norms.detach().max())
        side_means.append(F.relu(norms - kappa).pow(2).mean())
    if not side_means:
        zero = torch.zeros((), device=x_real.device, dtype=torch.float32)
        return zero, stats
    # coeff * mean(side means): with both sides present this is exactly
    # (coeff/2) * (mean_r + mean_f), the b_cap units. A lone side keeps the
    # same per-sample weight instead of going quiet.
    pen = float(coeff) * sum(side_means) / len(side_means)
    stats.update(applied=True, pen=float(pen.detach()))
    return pen, stats


def rp_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Relativistic-pairing discriminator loss: ``softplus(-(real - fake))``."""
    return F.softplus(-(real_logits - fake_logits)).mean()


def rp_g_loss(fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
    """Relativistic-pairing generator loss: ``softplus(-(fake - real))``.

    D params must be frozen (``requires_grad_(False)``) by the caller so grad
    flows only to the student deltas, not into D.
    """
    return F.softplus(-(fake_logits - real_logits)).mean()


def param_grad_norm(params) -> float:
    """Global L2 norm of current ``.grad``s (missing grad counts as 0)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return total ** 0.5


def flat_param_grad(params) -> torch.Tensor:
    """Detached flattened concat of current ``.grad``s (zeros if missing)."""
    params = list(params)
    parts = []
    for p in params:
        if p.grad is not None:
            parts.append(p.grad.detach().float().flatten())
        else:
            parts.append(torch.zeros(p.numel()))
    device = params[0].device if params else torch.device("cpu")
    return torch.cat([part.to(device) for part in parts])
