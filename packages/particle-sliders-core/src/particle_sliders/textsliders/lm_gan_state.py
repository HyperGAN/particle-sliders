"""Local training-state snapshots for continuing a constant-LR GAN game.

Inference adapters remain separate LoRA-only safetensors. These snapshots also
retain critics, optimizers, history and RNG state, and never go in the catalog.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import torch


def signature(args, rows, prompts_meta):
    ignored = {"name", "save_dir", "steps", "save_every", "device", "resume_state",
               "save_training_state", "endreg_cache", "prompts_file"}
    settings = {key: value for key, value in vars(args).items() if key not in ignored}
    sources = {}
    for name in ("train_lm_slider_music3.py", "lm_gan_state.py", "lm_adv.py",
                 "lm_gan.py", "lm_particles.py", "lora.py", "slider_targets.py"):
        path = Path(__file__).with_name(name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(settings=settings, sources=sources,
                prompts_sha256=hashlib.sha256(json.dumps([rows, prompts_meta], sort_keys=True).encode()).hexdigest())


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def save(path, *, run_signature, modules, optimizers, history):
    path = Path(path)
    state = dict(schema=1, signature=run_signature, completed_updates=len(history),
                 modules={name: _cpu(module.state_dict()) for name, module in modules.items() if module is not None},
                 optimizers={name: _cpu(opt.state_dict()) for name, opt in optimizers.items() if opt is not None},
                 history=history, python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                 cuda_rng=[rng.cpu() for rng in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore(path, *, run_signature, modules, optimizers):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != 1 or state["signature"] != run_signature:
        raise ValueError("Resume state has different training settings, prompts or source code")
    expected_modules = {name for name, module in modules.items() if module is not None}
    expected_optimizers = {name for name, opt in optimizers.items() if opt is not None}
    if set(state["modules"]) != expected_modules or set(state["optimizers"]) != expected_optimizers:
        raise ValueError("Resume state has different GAN components")
    history = state["history"]
    if [row["step"] for row in history] != list(range(1, state["completed_updates"] + 1)):
        raise ValueError("Resume history is not a continuous sequence of completed updates")
    for name in expected_modules:
        modules[name].load_state_dict(state["modules"][name], strict=True)
    for name in expected_optimizers:
        optimizers[name].load_state_dict(state["optimizers"][name])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return history
