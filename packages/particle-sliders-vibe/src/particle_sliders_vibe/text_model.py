"""Text-side binding for Anima's routed attention adapter and stock EMPF game."""

from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import math
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from lumen_studio.particles import Branch, Mixer
from particle_sliders.weight_policy import require_no_weight_ema
from particle_sliders.backends import PreparedResponse, ResponseRow
from particle_sliders.engine import ConditionalParticleGAN
from particlegan import init as pg_init

MODELS = {
    "glimmer": ("muse_glimmer", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"),
    "tiny-aya": ("cohere2", "af89d219b53ed9b13b8a4645f9c8028973510324"),
}
FORMAT = "text-vibe-routed-empf-v1"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def require_formulation():
    if (
        version("particlegan") != "0.12.0"
        or version("particle-sliders-core") != "0.15.3"
    ):
        raise RuntimeError(
            "Text vibes require ParticleGAN 0.12.0 and particle-sliders-core 0.15.3"
        )


def token_gain(schedule, position):
    if schedule == "constant":
        return 1.0
    if schedule not in ("early", "late"):
        raise ValueError("Unknown text strength schedule")
    early = 1.5 if position < 4 else 0.5 if position < 8 else 1.0
    return early if schedule == "early" else 2.0 - early


class TextAdapter(nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.names = [
            f"layers.{i}.self_attn.{name}"
            for i in range(len(tower.layers))
            for name in PROJECTIONS
        ]
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.branches = nn.ModuleList()
        for name in self.names:
            module = tower.get_submodule(name)
            self.branches.append(Branch(module.in_features, module.out_features))


class FreshWeights(nn.Module):
    """Only the new bank and DOWN matrices participate in the scoped initializer."""

    def __init__(self, adapter):
        super().__init__()
        self.bank = adapter.particles
        self.laws = {"bank": pg_init.R2Normal(0.0, 1.0)}
        for i, branch in enumerate(adapter.branches):
            name = f"down_{i}"
            self.register_parameter(name, branch.down.weight)
            bound = 1.0 / math.sqrt(branch.down.in_features)
            self.laws[name] = pg_init.Uniform(-bound, bound)


pg_init.register(FreshWeights, lambda weights: weights.laws)


class ResponseCritic(nn.Module):
    """Width-32 conditional critic on the native last-token vector."""

    def __init__(self, width):
        super().__init__()
        self.field = nn.Sequential(
            nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU()
        )
        self.context = nn.Sequential(nn.Linear(width, 32), nn.SiLU())
        self.logits = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, response, condition):
        return self.logits(
            torch.cat((self.field(response), self.context(condition)), dim=1)
        ).flatten()


class TextModel:
    def __init__(self, model, tokenizer, tower, identity):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer, self.tower, self.identity = tokenizer, tower, identity
        self.device = next(tower.parameters()).device
        self.mixer = Mixer(tower)
        self.profiles = {}

    @classmethod
    def load(cls, name, snapshot):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        require_formulation()
        torch.backends.cuda.matmul.allow_tf32 = False
        snapshot = Path(snapshot).resolve(strict=True)
        kind, revision = MODELS[name]
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        if config.model_type != kind or snapshot.name != revision:
            raise ValueError(f"Expected the pinned {name} snapshot {revision}")
        options = dict(
            local_files_only=True,
            device_map={"": 0},
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        if name == "glimmer":
            from transformers import (
                BitsAndBytesConfig,
                MuseGlimmerForConditionalGeneration,
            )

            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
            loader = MuseGlimmerForConditionalGeneration
        else:
            loader = AutoModelForCausalLM
        model = loader.from_pretrained(snapshot, **options)
        tower = model.model.language_model if name == "glimmer" else model.model
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise ValueError("Text model must fit entirely on the admitted GPU")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        identity = dict(
            model=name,
            revision=revision,
            config_sha256=sha256((snapshot / "config.json").read_bytes()).hexdigest(),
            precision="nf4-bf16" if name == "glimmer" else "bf16",
        )
        return cls(model, tokenizer, tower, identity)

    def encode(self, prompt, cue=""):
        # The Tiny Aya template accepts user content on every supported runtime.
        content = f"{cue}\n\n{prompt}" if cue else prompt
        messages = [dict(role="user", content=content)]
        prefix = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            reasoning_strength="low",
        )
        if self.identity["model"] == "glimmer":
            # Complete the model's ATEM user channel, before generating any answer tokens.
            prefix += " to=user<|message|>"
        ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(self.device)
        if ids.shape[1] > 2048:
            raise ValueError("Text prompt exceeds the 2048-token input budget")
        return ids

    def response(self, ids):
        hidden = self.tower(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return hidden[:, -1, :].float()

    @torch.no_grad()
    def reply(self, prompt, *, cue="", strengths=None, max_tokens=96, profiles=None):
        ids = self.encode(prompt, cue)
        strengths = strengths or {}
        schedules = self.profiles if profiles is None else profiles
        with ExitStack() as stack:
            stack.enter_context(self.mixer.scales(strengths))
            if any(schedules.get(key, "constant") != "constant" for key in strengths):
                position = 0

                def apply_schedule(module, args):
                    nonlocal position
                    for key, strength in strengths.items():
                        self.mixer.strengths[key] = strength * token_gain(
                            schedules.get(key, "constant"), position
                        )
                    position += 1

                handle = self.model.register_forward_pre_hook(apply_schedule)
                stack.callback(handle.remove)
            out = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=max_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        suffix = out[0, ids.shape[1] :].tolist()
        return dict(
            text=self.tokenizer.decode(suffix, skip_special_tokens=True).strip(),
            tokens=suffix,
        )

    def new_adapter(self, key):
        adapter = TextAdapter(self.tower).to(self.device)
        self.mixer.add(key, adapter)
        return adapter

    def load_adapter(self, key, path):
        with safe_open(path, framework="pt", device="cpu") as file:
            meta = json.loads((file.metadata() or {}).get("text_vibe", "{}"))
        if (
            meta.get("format") != FORMAT
            or meta.get("identity") != self.identity
            or meta.get("particlegan") != "0.12.0"
            or meta.get("weights") != "raw"
            or "ema_decay" not in meta
            or meta["ema_decay"] is not None
        ):
            raise ValueError("Incompatible text slider weights or base model")
        adapter = self.new_adapter(key)
        if meta.get("targets") != adapter.names:
            raise ValueError("Text slider attention targets differ")
        state = load_file(path)
        if any(not torch.isfinite(t.to(self.device)).all() for t in state.values()):
            raise ValueError("Text slider contains nonfinite weights")
        adapter.load_state_dict(state, strict=True)
        self.profiles[key] = meta.get("schedule", "constant")
        token_gain(self.profiles[key], 0)
        return meta


class TextResponse:
    def __init__(self, host, network):
        self.host, self.network = host, network

    def metadata(self):
        return dict(
            base=self.host.identity,
            targets=self.network.names,
            response="last-hidden/shared-token-history",
        )

    def generated(self, prepared):
        return self.host.response(prepared.row.neutral.to(self.host.device))


def freeze_rows(host, pairs):
    rows = []
    with torch.no_grad(), host.mixer.scales({}):
        for pair in pairs:
            neutral = host.encode(pair["prompt"])
            positive = host.encode(pair["prompt"], pair["cue"])
            # Both conditions see exactly the same continuation tokens at each observation.
            tokens = pair["neutral"]["tokens"]
            for length in sorted({0, min(4, len(tokens)), min(12, len(tokens))}):
                suffix = neutral.new_tensor([tokens[:length]])
                n, p = [
                    torch.cat((prefix, suffix), dim=1) for prefix in (neutral, positive)
                ]
                row = ResponseRow(
                    neutral=n.cpu(),
                    positive=p.cpu(),
                    negative=None,
                    latent=suffix.cpu(),
                )
                rows.append(
                    PreparedResponse(
                        row, host.response(p).cpu(), host.response(n).cpu()
                    )
                )
    return rows


def fit(
    host,
    rows,
    destination,
    *,
    steps=448,
    horizon=1024,
    seed=1,
    ema_decay=None,
    progress=lambda *_: None,
    check_output=None,
):
    """Anima's stock batch-two EMPF update, schedule and three optimizer rates."""
    require_formulation()
    require_no_weight_ema(ema_decay)
    if not 1 <= steps <= horizon:
        raise ValueError("The update budget must fit the optimizer schedule")
    torch.manual_seed(seed)
    adapter = host.new_adapter("training")
    pg_init.deterministic_orthogonal_(FreshWeights(adapter), seed=seed, strict=True)
    if any(torch.count_nonzero(b.up.weight) for b in adapter.branches):
        raise RuntimeError("Fresh text adapter must start at zero output")
    conditions = torch.cat([row.neutral for row in rows])
    critic = ResponseCritic(conditions.shape[1]).to(host.device)
    game = ConditionalParticleGAN(
        TextResponse(host, adapter),
        rows,
        critic,
        conditions=conditions,
        particles=adapter.particles,
        preset="empf",
        batch_size=2,
        total_steps=horizon,
        seed=seed,
        response_domain="continuous_vector",
        penalty_input_domain="continuous",
        optimizer_dense_profile="recipe",
        recipe_overrides=dict(lr=0.00075, d_lr_mult=12.0, prior_lr_mult=40.0),
    )
    if (
        game.recipe.effective_critic_formulation != "empf"
        or game.recipe.reg_every != 1
        or game.recipe.ema_decay != 0.0
    ):
        raise RuntimeError("The stock EMPF formulation or RAW weight policy changed")

    def save_raw(path):
        meta = dict(
            format=FORMAT,
            identity=host.identity,
            targets=adapter.names,
            weights="raw",
            ema_decay=None,
            particlegan="0.12.0",
            preset="empf",
            updates=game.completed_steps,
            schedule_horizon=horizon,
            seed=seed,
            rates=dict(g=0.00075, d=0.009, particle=0.03),
        )
        temp = path.with_suffix(".partial")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in adapter.state_dict().items()},
            temp,
            metadata={"text_vibe": json.dumps(meta)},
        )
        temp.replace(path)
        return meta

    with ExitStack() as stack:
        host.tower.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        stack.callback(host.tower.gradient_checkpointing_disable)
        for block in host.tower.layers:
            # Enable the HF checkpoint wrapper without changing frozen submodule/dropout modes.
            stack.callback(setattr, block, "training", block.training)
            block.training = True
        stack.enter_context(host.mixer.scales({"training": 1.0}))
        for step in range(1, steps + 1):
            game.step()
            if step % max(1, steps // 4) == 0 or step == steps:
                path = Path(destination)
                checkpoint = (
                    path if step == steps else path.with_name(f"raw-{step}.safetensors")
                )
                meta = save_raw(checkpoint)
                progress(step, steps)
                if check_output is not None:
                    with ExitStack() as inference:
                        for block in host.tower.layers:
                            inference.callback(
                                setattr, block, "training", block.training
                            )
                            block.training = False
                        check_output(step)
    if game.completed_steps != steps:
        raise RuntimeError("Optimizer update count differs from the budget")
    return meta
