"""Pinned particle-game reference excerpts; see provenance.json."""

import math

import torch

from torch import nn

from torch.nn import functional as F

def rp_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Relativistic-pair logistic critic loss: prefer D(real) > D(fake)."""
    return F.softplus(-(d_real - d_fake)).mean()

def rp_g_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Relativistic-pair logistic generator loss: prefer D(fake) > D(real)."""
    return F.softplus(-(d_fake - d_real)).mean()

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

REFERENCE = dict(particles=128, z_dim=4, width=48, router_width=16,
    batch_size=64, g_lr=.0006, d_lr=.0009, particle_lr=.006,
    betas=(0., .999), vic_coeff=1., cap_coeff=1., cap_kappa=1., cap_every=4,
    ema=.995, noise_start=1., noise_decay_steps=8000, noise_floor=.03,
    target_std_floor=1e-4,
    # Paired-edit whitening: std of (target−neutral), then a scalar gain so
    # median per-row normalized edit RMS equals edit_rms_target. Noise starts
    # at edit_rms / edit_noise_ratio so initial edit/noise matches the metal
    # pilot (~0.28). Absolute-state whitening buried voice edits (~0.05).
    edit_rms_target=1., edit_noise_ratio=.28,
    # Sigma hold: the anneal stops at edit_rms * noise_hold_ratio. Per run, not a
    # constant: the paired critic stays balanced only while sigma is ~3-4x the
    # reachable residual (pop at 1.0 hold: ratio 3.7, flat d_loss 0.45; gender v2
    # at 1.0: ratio 1.9, d_loss eroding 0.45->0.18 with G grad norm 5->15).
    noise_hold_ratio=1.3,
    # Critic: 'sn_mlp' is the near-term default (spectral-norm residual MLP).
    # 'gmix' is the global-mix hybrid ablation (H→T·W then attend).
    # Legacy: 'mlp' thin head; 'mix' fat first layer; patch/bottleneck/…
    critic='mlp', critic_patch=64, critic_width=48, critic_layers=1, critic_heads=4,
    critic_tokens=8, critic_queries=4, critic_rank=64, critic_score_bound=8.0)

def mlp(inputs, outputs, width):
    layers = []
    for n in (inputs, width, width):
        layers.extend((nn.Linear(n, width), nn.LeakyReLU(.2)))
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)

def bound_score(score, bound):
    """Keep relativistic logits in the softplus-sensitive band."""
    bound = float(bound)
    if bound <= 0:
        return score
    return bound * torch.tanh(score / bound)

def register_paired_error_norm(module, targets, neutrals=None):
    """Whiten by paired edit, not absolute target-state distribution.

    When ``neutrals`` is provided, per-coordinate scale is ``std(target−neutral)``,
    then a scalar gain pins median row edit RMS to ``edit_rms_target``. Sets
    ``module.noise_start = edit_rms / edit_noise_ratio``. Without neutrals,
    keeps legacy absolute-state whitening and ``noise_start = REFERENCE['noise_start']``.
    """
    if targets.ndim != 2 or len(targets) < 2 or not torch.isfinite(targets).all():
        raise ValueError('Expected finite training targets [rows, width]')
    floor = float(REFERENCE['target_std_floor'])
    if neutrals is None:
        scale = targets.std(0).clamp_min(floor)
        module.register_buffer('target_mean', targets.mean(0))
        module.register_buffer('target_std', scale)
        module.register_buffer('input_scale', scale.square().mean().sqrt())
        module.register_buffer('edit_rms', module.input_scale.detach().clone())
        module.noise_start = float(REFERENCE['noise_start'])
        module.normalization = 'absolute_target_per_coordinate_sample_std'
        return module
    if neutrals.shape != targets.shape or not torch.isfinite(neutrals).all():
        raise ValueError('Expected finite neutrals matching targets')
    edits = targets - neutrals
    scale = edits.std(0).clamp_min(floor)
    normed = edits / scale
    row_rms = normed.pow(2).mean(-1).sqrt()
    median_rms = row_rms.median().clamp_min(floor)
    gain = float(REFERENCE['edit_rms_target']) / float(median_rms)
    scale = scale / gain
    normed = edits / scale
    edit_rms = normed.pow(2).mean().sqrt().detach()
    module.register_buffer('target_mean', targets.mean(0))
    module.register_buffer('target_std', scale)
    module.register_buffer('edit_rms', edit_rms.to(dtype=torch.float32))
    module.register_buffer('input_scale', module.edit_rms.clone())
    ratio = float(REFERENCE['edit_noise_ratio'])
    module.noise_start = max(float(edit_rms) / ratio, float(REFERENCE['noise_floor']))
    module.normalization = 'paired_edit_per_coordinate_std_median_rms_gain'
    return module

class RoutedMLP(nn.Module):
    """Every input uses the same softmax route at train and inference time."""
    def __init__(self, inputs, outputs, z_dim=4, width=48, router_width=16):
        super().__init__()
        self.router = mlp(inputs, z_dim, router_width)
        self.net = mlp(inputs + z_dim, outputs, width)

    def forward(self, x, particles):
        q = self.router(x)
        weights = (q @ particles.T / math.sqrt(particles.shape[1])).softmax(-1)
        z = weights @ particles
        return self.net(torch.cat((x, z), dim=-1))

def particle_vic(z):
    """Pinned upstream VICRegLikeLoss (target_std=1, eps=1e-4).

    Sample standard deviation and off-diagonal covariance, on particles only.
    Source SHA256 d2563bcab93d443cbf2cf2260946a3ba470f1fc584c231b704ec5ac154f668c4.
    """
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    variance = F.relu(1. - std).mean()
    centered = z - z.mean(dim=0)
    covariance = centered.T @ centered / (len(z) - 1)
    dim = z.shape[1]
    if dim == 1:
        return variance
    off = covariance.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
    return variance + off.square().sum() / dim

def noise_std(step, start=None, decay_steps=None, hold=None):
    """Geometric anneal from ``start`` to the floor, done at ``decay_steps``.

    The reference horizon is 8000. A training run passes its own step budget
    so the floor is reached when that run ends, not thousands of steps later.

    ``hold`` is the normalized residual scale (about 1). Once the curve would
    pass under it, sigma stays there. A start already at or below the hold
    is left on the geometric curve, so the hold never raises noise.
    """
    if step < 0:
        raise ValueError('Step must be nonnegative')
    start = float(REFERENCE['noise_start'] if start is None else start)
    floor = float(REFERENCE['noise_floor'])
    horizon = float(REFERENCE['noise_decay_steps'] if decay_steps is None else decay_steps)
    if start <= 0 or horizon <= 0 or floor <= 0:
        raise ValueError('noise start, floor, and decay horizon must be positive')
    t = min(step / horizon, 1.)
    sigma = start * (floor / start) ** t
    if hold is None:
        return sigma
    hold = float(hold)
    if hold <= 0:
        raise ValueError('noise hold must be positive')
    if start > hold:
        sigma = max(sigma, hold)
    return sigma

class GlobalMixErrorCritic(nn.Module):
    """Learned global mix before tokenizing, then attention (recommended hybrid).

    ``2048 → Linear(H, T·W) → T tokens → 1–2 attn blocks → mean+max pool →
    bounded score``. Every token is a mixture of the full hidden state, so
    token adjacency is learned rather than invented by patching.
    """

    def __init__(self, targets, *, tokens=None, width=None, layers=None, heads=None,
                 neutrals=None, score_bound=None):
        super().__init__()
        register_paired_error_norm(self, targets, neutrals)
        dim = int(targets.shape[1])
        tokens = int(8 if tokens is None else tokens)
        width = int(48 if width is None else width)
        layers = int(REFERENCE['critic_layers'] if layers is None else layers)
        heads = int(REFERENCE['critic_heads'] if heads is None else heads)
        if tokens < 1 or width < 1 or layers < 1:
            raise ValueError('tokens/width/layers must be positive')
        if width % heads != 0:
            raise ValueError(f'critic_width {width} must split over {heads} heads')
        self.tokens = tokens
        self.width = width
        self.mix = nn.Linear(dim, tokens * width)
        self.pos = nn.Parameter(torch.zeros(1, tokens, width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_SetBlock(width, heads) for _ in range(layers)])
        self.out_norm = nn.RMSNorm(width, eps=1e-3)
        self.head = nn.Linear(2 * width, 1)
        self.score_bound = float(
            REFERENCE.get('critic_score_bound') if score_bound is None else score_bound
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.mix.bias, std=0.02)

    def normalize(self, value):
        return (value - self.target_mean) / self.target_std

    def forward(self, coordinates):
        bsz = coordinates.shape[0]
        tokens = self.mix(coordinates).view(bsz, self.tokens, self.width) + self.pos
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=None)
        mean = self.out_norm(tokens.mean(dim=1))
        peak = self.out_norm(tokens.amax(dim=1))
        return bound_score(self.head(torch.cat((mean, peak), dim=-1)).squeeze(-1), self.score_bound)
