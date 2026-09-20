"""Pinned official conversion; does not use or mutate Music Studio environments."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from .contracts import atomic_json, file_hash

ROOT = Path(__file__).resolve().parents[1]


def convert(output):
    from huggingface_hub import hf_hub_download, snapshot_download
    lock = json.loads((ROOT / "configs/anima/model.lock.json").read_text())
    output = Path(output).resolve()
    if (output / "anima-lock.json").exists():
        return output
    converter = output.parent / "converter"
    converter.mkdir(parents=True, exist_ok=True)
    for name, expected in lock["converter"].items():
        path = converter / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/huggingface/diffusers/{lock['diffusers_revision']}/scripts/{name}"
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        if file_hash(path) != expected:
            raise ValueError(f"Converter checksum mismatch: {name}")
    weights = {}
    for name, expected in lock["source_files"].items():
        path = hf_hub_download(lock["repo"], name, revision=lock["revision"])
        if file_hash(path) != expected:
            raise ValueError(f"Weight checksum mismatch: {name}")
        weights[Path(name).name] = path
    tokenizers = Path(snapshot_download(lock["tokenizer_repo"], revision=lock["tokenizer_revision"],
                                        allow_patterns=["tokenizer/*", "t5_tokenizer/*"]))
    # The official Turbo file has Comfy's wrapper prefix, unlike the Base file
    # used by the upstream converter. Strip exactly that prefix, with collision
    # and mixed-layout checks; the upstream tensor conversion remains unmodified.
    from safetensors.torch import load_file, save_file
    state = load_file(weights["anima-turbo-v1.1.safetensors"], device="cpu")
    prefix = "model.diffusion_model."
    if any(k.startswith(prefix) for k in state):
        if not all(k.startswith(prefix) for k in state):
            raise ValueError("Unexpected mixed transformer key prefixes")
        normalized = converter / "turbo-unprefixed.safetensors"
        save_file({"net." + k[len(prefix):]: v for k, v in state.items()}, str(normalized))
        weights["anima-turbo-v1.1.safetensors"] = str(normalized)
        lock["prefix_normalization"] = dict(removed=prefix, replacement="net.", sha256=file_hash(normalized))
    del state
    command = [sys.executable, str(converter / "convert_anima_to_diffusers.py"),
        "--transformer_ckpt_path", weights["anima-turbo-v1.1.safetensors"],
        "--text_encoder_ckpt_path", weights["qwen_3_06b_base.safetensors"],
        "--vae_ckpt_path", weights["qwen_image_vae.safetensors"],
        "--qwen_tokenizer_path", str(tokenizers / "tokenizer"),
        "--t5_tokenizer_path", str(tokenizers / "t5_tokenizer"),
        "--output_path", str(output), "--save_pipeline", "--dtype", "bf16"]
    subprocess.run(command, check=True)
    lock["environment_sha256"] = file_hash(ROOT / "configs/anima/requirements.lock")
    lock["files"] = {str(p.relative_to(output)): file_hash(p)
                     for p in sorted(output.rglob("*")) if p.is_file() and p.name != "anima-lock.json"}
    atomic_json(output / "anima-lock.json", lock)
    return output
