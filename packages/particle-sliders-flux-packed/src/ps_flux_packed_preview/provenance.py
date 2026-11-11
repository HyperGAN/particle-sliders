"""Require the exact installed cohort before touching a staged native model."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

ROOT = Path(__file__).parent
CORE_COMMIT = "7a8a321d04323d49319d0b08e6093a0ddce36b27"


def checked_builder():
    import particle_sliders
    from particle_sliders.fibo_hndl import verified_builder
    expected = json.loads((ROOT / "sources.json").read_text())
    if version("particle-sliders-core") != "0.12.0":
        raise ValueError("Flux prerequisite requires exact core0.12 source cohort")
    core_root = Path(particle_sliders.__file__).parent
    actual = {str(p.relative_to(core_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in core_root.rglob("*") if p.suffix in (".py", ".json", ".hndl")}
    if actual != expected["core_sources"] or expected["core_commit"] != CORE_COMMIT:
        raise ValueError("Flux prerequisite shared-core source closure differs")
    if hashlib.sha256((ROOT / "lora.hndl").read_bytes()).hexdigest() != expected["lora_sha256"]:
        raise ValueError("Flux prerequisite LoRA graph differs")
    builder, identity = verified_builder()
    return builder, dict(identity, flux_core_commit=CORE_COMMIT,
                         flux_lora_sha256=expected["lora_sha256"])


def check_native(model):
    import diffusers
    from diffusers import FluxTransformer2DModel
    expected = json.loads((ROOT / "sources.json").read_text())
    if type(model) is not FluxTransformer2DModel or version("diffusers") != "0.38.0":
        raise ValueError("exact native Diffusers0.38 FluxTransformer2DModel required")
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    if (any(type(p) is not FluxAttnProcessor or p._attention_backend is not None
            for p in model.attn_processors.values())
            or any(getattr(m, "fused_projections", False) or hasattr(m, "_hf_hook") for m in model.modules())):
        raise ValueError("native unfused Flux attention processors required")
    root = Path(diffusers.__file__).parent
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
           for name, digest in expected["native_sources"].items()):
        raise ValueError("native Flux forward source closure differs")
