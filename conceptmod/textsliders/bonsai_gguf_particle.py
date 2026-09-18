"""Opt-in routed particle head on the frozen Bonsai GGUF readout.

Runs YuE2's working recipe ``anneal-routed-particle-error-yue2-v1``
(``--recipe particle_bridge``) on Ternary-Bonsai-2-27B last-token hidden
states served by the fork's ``llama-server --embedding --pooling last``.
The game itself is NOT reimplemented here: ``shared`` is the exact YuE2
module ``conceptmod.textsliders.particle_bridge_gan`` (same Rp logistic
on the paired error, same noise anneal, same lazy b_cap, same particle
VIC, same EMA decay).

DRIFT (honest, documented in docs/bonsai-gguf-slider.md): a GGUF is
inference-only frozen ternary — there is no gradient path through the
fork, so full LoRA-in-GGUF training is impossible and no in-attention
LoRA is claimed. The trainable slider is a torch-side residual head on
the frozen 5120-dim readout: ``down`` is kaiming, ``up`` starts at zero
(base preserved at init), one shared 128x4 cloud, router + MLP bridge,
multiplier 0 bypasses the readout exactly. Same game, different (and
explicitly weaker) adapter placement than the Qwen3-0.6B path, which
wraps live attention q/k/v/o.

Unipolar: + caption vs raw positive teacher; scale 0 bypasses exactly.
No output MSE, FM, ending, hold, or anchor — on G or anywhere.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from conceptmod.textsliders import particle_bridge_gan as shared
from conceptmod.textsliders.bonsai_gguf_backend import HIDDEN, EncodedText

FORMAT = "conceptmod-bonsai-gguf-particle-v1"
RANK = 8
ALPHA = 8.0
RECIPE = dict(
    name="anneal-routed-particle-error-bonsai-gguf-v1",
    source_recipe="anneal-routed-particle-error-yue2-v1",
    generator_objective="paired_error_rpgan_plus_particle_vic",
    parts=128,
    particle_dim=4,
    vicreg_weight=1.0,
    g_lr=shared.REFERENCE["g_lr"],
    d_lr=shared.REFERENCE["d_lr"],
    particle_lr=shared.REFERENCE["particle_lr"],
    g_optimizer="Adam",
    g_weight_decay=0.0,
    optimizer_betas=shared.REFERENCE["betas"],
    critic_hidden=shared.REFERENCE["width"],
    critic_layers=3,
    penalty_lazy_k=shared.REFERENCE["cap_every"],
    cap_coordinates="normalized_paired_error_plus_shared_gaussian",
    target_normalization="training_target_per_coordinate_sample_std_floor_1e-4",
    noise_start=1.0,
    noise_floor=shared.REFERENCE["noise_floor"],
    noise_decay_steps=shared.REFERENCE["noise_decay_steps"],
    ema=shared.REFERENCE["ema"],
    adv_batch=shared.REFERENCE["batch_size"],
    schedule="constant",
    prompt_policy="independent_D_G_with_replacement",
    routing="all_examples",
    particle_vic_batch=64,
    particle_vic_target_std=1.0,
    particle_vic_eps=1e-4,
    adapter_format=FORMAT,
    adapter_rank=RANK,
    adapter_alpha=ALPHA,
    adapter_width=shared.REFERENCE["width"],
    router_width=shared.REFERENCE["router_width"],
    generator_initialization="zero_output_low_rank; standard_normal_particles",
    readout="fork_server_last_token_embedding",
    adapter_placement="readout_residual_zero_init",
    propose_only=True,
    merge_to_trainer=False,
)


class BonsaiParticleHead(nn.Module):
    """Routed low-rank residual on the frozen 5120-dim readout.

    ``up`` starts at zero so a fresh head is the base behavior;
    ``multiplier`` 0 bypasses it exactly. The 128x4 cloud belongs to
    this head exactly once.
    """

    def __init__(self, hidden: int = HIDDEN, rank: int = RANK,
                 alpha: float = ALPHA) -> None:
        super().__init__()
        if int(rank) != RANK or float(alpha) != ALPHA:
            raise ValueError("The Bonsai particle format pins rank/alpha 8")
        if int(hidden) != HIDDEN:
            raise ValueError(
                f"Bonsai readout width is {HIDDEN}, got {hidden}"
            )
        self.rank, self.alpha = int(rank), float(alpha)
        self.scale = float(alpha) / float(rank)
        self.multiplier = 0.0
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        self.bridge = shared.RoutedMLP(rank, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0:
            return x
        host_device, host_dtype = x.device, x.dtype
        features = self.down(x.to(device=self.down.weight.device,
                                   dtype=self.down.weight.dtype))
        routed = self.bridge(features, self.particles)
        delta = self.up(routed).to(device=host_device, dtype=host_dtype)
        return x + delta * (self.multiplier * self.scale)

    @contextmanager
    def scaled(self, scale: float):
        previous = self.multiplier
        self.multiplier = float(scale)
        try:
            yield
        finally:
            self.multiplier = previous

    def save(self, path, metadata):
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(
            metadata, format=FORMAT, rank=self.rank, alpha=self.alpha,
            hidden=HIDDEN, particles=128, particle_dim=4,
            bridge_width=48, router_width=16,
        )
        state = self.state_dict()
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Invalid Bonsai particle-head export")
        save_file(
            {k: v.detach().float().cpu().contiguous() for k, v in state.items()},
            str(path), metadata={"conceptmod": json.dumps(record, sort_keys=True)},
        )
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            record = json.loads((handle.metadata() or {}).get("conceptmod", "{}"))
        required = dict(format=FORMAT, rank=8, alpha=8.0, hidden=HIDDEN,
                        particles=128, particle_dim=4,
                        bridge_width=48, router_width=16)
        if any(record.get(k) != v for k, v in required.items()):
            raise ValueError("Incompatible Bonsai routed-particle checkpoint")
        state = load_file(str(path), device="cpu")
        probe = cls()
        expected = set(probe.state_dict())
        if set(state) != expected or any(
            tuple(state[k].shape) != tuple(probe.state_dict()[k].shape)
            for k in expected
        ):
            raise ValueError("Incomplete Bonsai routed-particle tensors")
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Non-finite Bonsai routed-particle tensors")
        network = cls()
        network.load_state_dict(state, strict=True)
        return network, record


def prepare_rows(backend, rows: list[dict]) -> list[dict]:
    """Freeze per-row teachers: raw + caption and neutral readout states."""
    fixed = []
    with torch.no_grad():
        for row in rows:
            plus = backend.encode(row["positive"])
            neu = backend.encode(row["neutral"])
            plus_h = backend.teacher_hidden(plus).float()
            neutral = backend.teacher_hidden(neu).float()
            fixed.append({
                "enc": plus,
                "targets": plus_h.cpu(),
                "neutral": neutral.cpu(),
                "positive": row["positive"],
                "neutral_text": row["neutral"],
            })
    return fixed


def build_game(backend, network, fixed):
    targets = torch.cat([r["targets"] for r in fixed]).to(
        next(network.parameters()).device
    )
    return shared.build_game(network, targets)


def update(backend, network, critic, g, d, rows, *, sampler, step):
    device = critic.target_mean.device
    targets = torch.cat([r["targets"] for r in rows]).to(device)
    predictions = {}
    # Frozen readout: fetch once per row per update call (the GGUF never
    # changes; D and G phases share the same frozen vectors).
    frozen: dict[int, torch.Tensor] = {}

    def predict(i, phase):
        row = rows[i]
        enc = row["enc"]
        assert isinstance(enc, EncodedText)
        if i not in frozen:
            frozen[i] = backend.hidden(enc)
        with network.scaled(1.0):
            pred = network(frozen[i]).float()
        if phase == "g":
            predictions[i] = pred.detach()
        return critic.normalize(pred)

    result = shared.update(
        network, critic, g, d, targets, predict, sampler=sampler, step=step
    )
    result["cos_pos"] = sum(
        float(
            torch.nn.functional.cosine_similarity(
                predictions[i] - rows[i]["neutral"].to(device),
                targets[i:i + 1] - rows[i]["neutral"].to(device),
                dim=-1,
            ).mean()
        )
        for i in result["g_rows"]
    ) / len(result["g_rows"])
    return result


def resolve_head_path(path: str):
    path = Path(path)
    if path.is_file():
        return path
    candidates = []
    if path.suffix == ".safetensors":
        candidates.append(path)
    candidates.append(Path(str(path) + "_last.safetensors"))
    candidates.append(Path(str(path) + ".safetensors"))
    if path.is_dir():
        candidates.extend(sorted(path.glob("*_last.safetensors")))
        candidates.extend(sorted(path.glob("*.safetensors")))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no Bonsai particle-head safetensors under {path} "
        "(expected {name}_last.safetensors from the particle trainer)"
    )
