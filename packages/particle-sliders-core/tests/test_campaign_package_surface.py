"""Campaign and native architecture APIs work from the installable package."""
import ast
import importlib
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap

import pytest
import torch


SURFACES = {
    "pg09_moss_campaign": (
        "CampaignAddress", "MossCampaignRows", "CampaignFieldCritic",
        "MossCampaignBackend", "MossCampaignRun", "build_campaign_run",
        "canonical_digest", "file_digest", "coverage_budget",
    ),
    "pg09_remaining_campaigns": (
        "SupraCampaignRows", "SupraCampaignRun", "build_supra_campaign",
        "build_final_form_campaign", "export_admitted_supra_contexts",
        "export_live_supra_contexts", "load_supra_contexts", "teacher_parity",
    ),
    "supra_model": (
        "SupraDiT", "LoRALinear", "attach_supra_lora", "load_supra_weights",
        "supra_weights_from_checkpoint", "count_parameters",
    ),
}


@pytest.mark.parametrize("module_name,names", SURFACES.items())
def test_campaign_compatibility_imports_are_exact_module_aliases(module_name, names):
    canonical = importlib.import_module(f"particle_sliders.{module_name}")
    if importlib.util.find_spec("conceptmod") is None:
        pytest.skip("legacy compatibility aliases belong to the repository")
    legacy = importlib.import_module(f"conceptmod.textsliders.{module_name}")
    assert canonical is legacy
    for name in names:
        assert getattr(legacy, name) is getattr(canonical, name)


@pytest.mark.parametrize("module_name", SURFACES)
def test_campaign_modules_do_not_depend_on_repository_namespace(module_name):
    module = importlib.import_module(f"particle_sliders.{module_name}")
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("conceptmod") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("conceptmod")


def test_campaign_imports_do_not_load_optional_runtimes_or_construct_models():
    import particle_sliders
    import particlegan

    # Derive the installed package location so this also runs against a wheel.
    # -I removes checkout cwd/PYTHONPATH from the child interpreter's search path.
    package_root = str(Path(particle_sliders.__file__).resolve().parent.parent)
    dependency_root = str(Path(particlegan.__file__).resolve().parent.parent)
    script = textwrap.dedent("""
        import builtins
        import importlib
        import sys
        sys.path.insert(0, sys.argv[1])
        sys.path.insert(0, sys.argv[2])
        import particle_sliders
        import torch

        forbidden_roots = {
            'conceptmod', 'safetensors', 'moss_sliders', 'PIL',
            'transformers', 'diffusers', 'huggingface_hub',
        }
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in forbidden_roots:
                raise AssertionError('eager optional runtime import: ' + name)
            return original_import(name, *args, **kwargs)
        def forbidden(*args, **kwargs):
            raise AssertionError('model construction, deserialization or CUDA on import')
        builtins.__import__ = guarded_import
        torch.load = forbidden
        torch.nn.Module.__init__ = forbidden
        torch.cuda._lazy_init = forbidden
        torch.cuda.init = forbidden
        for name in ('pg09_moss_campaign', 'pg09_remaining_campaigns', 'supra_model'):
            importlib.import_module('particle_sliders.' + name)
        assert not torch.cuda.is_initialized()
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, package_root, dependency_root],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_campaign_types_serialize_with_installable_class_identity():
    from particle_sliders.pg09_moss_campaign import CampaignAddress, CampaignFieldCritic
    from particle_sliders.pg09_remaining_campaigns import SupraCampaignRun

    address = CampaignAddress(1, 0, 2, -1)
    assert type(pickle.loads(pickle.dumps(address))) is CampaignAddress
    assert CampaignAddress.__module__ == "particle_sliders.pg09_moss_campaign"
    assert CampaignFieldCritic.__module__ == "particle_sliders.pg09_moss_campaign"
    assert SupraCampaignRun.__module__ == "particle_sliders.pg09_remaining_campaigns"


def test_installed_supra_architecture_uses_only_owned_lora_parameters():
    from particle_sliders.supra_model import SupraDiT, attach_supra_lora

    torch.manual_seed(71)
    model = SupraDiT(d_model=16, depth=1, n_heads=4, ctx_dim=8,
                     mlp_ratio=2, num_tokens=4, patch=2)
    latent, context = torch.randn(1, 4, 4, 4), torch.randn(1, 3, 8)
    time, mask = torch.tensor([0.5]), torch.ones(1, 3, dtype=torch.bool)
    before = model(latent, time, context, mask).detach()
    loras = attach_supra_lora(model, rank=2, alpha=2,
                             targets=("ctx_proj", "cross_attn.q"))
    torch.testing.assert_close(model(latent, time, context, mask), before, rtol=0, atol=0)
    expected = {id(p) for lora in loras for p in (lora.down.weight, lora.up.weight)}
    assert {id(p) for p in model.parameters() if p.requires_grad} == expected
    model(latent, time, context, mask).square().mean().backward()
    assert any(torch.count_nonzero(lora.up.weight.grad) for lora in loras)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert not torch.cuda.is_initialized()
