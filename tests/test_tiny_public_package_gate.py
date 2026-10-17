"""Public legacy CLI retirement and immutable packaged inference, CPU only."""
import builtins
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from conceptmod.textsliders import train_lora_tiny_llm as cli


def forbidden(*args, **kwargs):
    raise AssertionError("training/data/output must not be reached")


@pytest.mark.parametrize("steps", [1, 20, -1])
def test_legacy_budget_rejected_before_model_rows_or_output(tmp_path, monkeypatch, steps):
    args = cli.parse_args(["--steps", str(steps), "--prompts_file", str(tmp_path / "absent.yaml"),
                           "--save_dir", str(tmp_path / "output")])
    monkeypatch.setattr(cli, "build_backend", forbidden)
    monkeypatch.setattr(cli, "load_slider_rows", forbidden)
    with pytest.raises(ValueError, match="retired"):
        cli.train(args)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("observer", [None, object()])
def test_zero_requires_export_and_rejects_observer_before_data(tmp_path, monkeypatch, observer):
    args = cli.parse_args(["--steps", "0", "--save_dir", str(tmp_path / "output")])
    monkeypatch.setattr(cli, "build_backend", forbidden)
    monkeypatch.setattr(cli, "load_slider_rows", forbidden)
    with pytest.raises(ValueError, match="requires --load_tiny_lora" if observer is None else "Observers/resume"):
        cli.train(args, observer=observer)
    assert not (tmp_path / "output").exists()


def test_public_main_retires_training_without_package_or_local_game_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    program = '''
import sys
class BlockTrainingImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'particle_sliders' or fullname in (
            'conceptmod.textsliders.particle_bridge_gan',
            'conceptmod.textsliders.tiny_llm_particle',
            'conceptmod.textsliders.tiny_llm_backend'):
            raise AssertionError('training/package import reached before retirement gate')
sys.meta_path.insert(0, BlockTrainingImports())
from conceptmod.textsliders.train_lora_tiny_llm import main
try:
    main(['--steps', '1', '--prompts_file', 'absent.yaml', '--save_dir', sys.argv[1]])
except ValueError as exc:
    assert 'retired' in str(exc)
else:
    raise AssertionError('legacy training accepted')
'''
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path / "output")],
                            cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "output").exists()


def make_export(tmp_path):
    from particle_sliders.tiny_llm_backend import TinyLLMBackend
    from particle_sliders.tiny_native import TinyParticleSlider

    torch.manual_seed(91)
    backend = TinyLLMBackend(dummy=True)
    network = TinyParticleSlider(backend.model).float()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.copy_(torch.rand_like(parameter) * .02)
    source = tmp_path / "archived_last.safetensors"
    network.save(source, dict(weights_kind="EMA", completed_steps=1200,
                              quality_qualified=False, readout_transfer_gate_passed=False))
    return source


@pytest.mark.parametrize("no_report", [False, True])
def test_load_only_uses_packaged_weights_without_training_or_export(tmp_path, monkeypatch, no_report):
    from safetensors import safe_open
    from safetensors.torch import load_file
    from particle_sliders.tiny_llm_backend import TinyLLMBackend
    from particle_sliders.tiny_native import TinyParticleSlider

    torch.set_num_threads(1)
    source = make_export(tmp_path)
    original_bytes, original_sidecar = source.read_bytes(), source.with_suffix(".json").read_bytes()
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        original_metadata = json.loads(handle.metadata()["conceptmod"])
    prompts = tmp_path / "pairs.yaml"
    prompts.write_text("- neutral: a garden\n  positive: a rose garden\n"
                       "- neutral: a sea\n  positive: a turquoise sea\n")
    flags = ["--no_report"] if no_report else []
    args = cli.parse_args(["--dummy", "--steps", "0", "--load_tiny_lora", str(source),
                           "--save_dir", str(tmp_path), "--name", "archived",
                           "--prompts_file", str(prompts), "--report_scales", "0,.5,1", *flags])
    torch.manual_seed(91)
    backend = TinyLLMBackend(dummy=True)
    backend.model.bfloat16()
    base = {key: value.clone() for key, value in backend.model.state_dict().items()}
    captured = {}
    original_load = TinyParticleSlider.load.__func__

    def load(cls, model, path, *, adapter_dtype=None):
        assert adapter_dtype == torch.float32
        result = original_load(cls, model, path, adapter_dtype=adapter_dtype)
        captured["network"] = result[0]
        return result

    monkeypatch.setattr(TinyParticleSlider, "load", classmethod(load))
    monkeypatch.setattr(TinyParticleSlider, "save", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    if no_report:
        monkeypatch.setattr(cli, "load_slider_rows", forbidden)
    actual_import = builtins.__import__

    def no_legacy_game(name, *args, **kwargs):
        if name in ("conceptmod.textsliders.particle_bridge_gan",
                    "conceptmod.textsliders.tiny_llm_particle"):
            raise AssertionError("local native adapter/game import is forbidden")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_legacy_game)
    rng = torch.get_rng_state().clone()
    receipt = cli.train(args, backend=backend)
    assert torch.equal(rng, torch.get_rng_state())
    assert type(captured["network"]) is TinyParticleSlider
    expected_weights = load_file(str(source))
    for key, value in captured["network"].state_dict().items():
        assert value.dtype == torch.float32
        assert torch.equal(value, expected_weights[key])
    for key, value in base.items():
        assert torch.equal(value, backend.model.state_dict()[key])
    assert source.read_bytes() == original_bytes
    assert source.with_suffix(".json").read_bytes() == original_sidecar
    assert sorted(tmp_path.glob("*.safetensors")) == [source]
    assert receipt["source_metadata"] == original_metadata
    assert receipt["source_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert receipt["source_weights_kind"] == "EMA"
    assert receipt["source_completed_steps"] == 1200
    assert receipt["base_identity_verified"] is False
    assert receipt["protocol_validated"] is False
    assert receipt["training_performed"] is False and receipt["optimizer_constructed"] is False
    assert receipt["quality_qualified"] is False
    assert (tmp_path / "archived_inference.json").is_file()
    assert (tmp_path / "report/hidden_delta.json").exists() == (not no_report)
    assert not torch.cuda.is_initialized()


def test_ambiguous_weights_and_source_sidecar_collision_rejected_before_model(tmp_path, monkeypatch):
    source = make_export(tmp_path)
    monkeypatch.setattr(cli, "build_backend", forbidden)
    for path, name, message in [
        (tmp_path, "report", "unambiguous"),
        (source, "archived_last", "overwrite source metadata"),
    ]:
        if path.is_dir():
            (tmp_path / "second.safetensors").write_bytes(source.read_bytes())
        else:
            # The helper uses <name>_inference.json for receipts.
            name = "archived"
            collision = tmp_path / "archived_inference.safetensors"
            collision.write_bytes(source.read_bytes())
            path = collision
        args = cli.parse_args(["--dummy", "--steps", "0", "--load_tiny_lora", str(path),
                               "--name", name, "--save_dir", str(tmp_path)])
        with pytest.raises(ValueError, match=message):
            cli.train(args)


def test_existing_receipt_rejected_before_model(tmp_path, monkeypatch):
    source = make_export(tmp_path)
    (tmp_path / "example_inference.json").write_text("original\n")
    monkeypatch.setattr(cli, "build_backend", forbidden)
    args = cli.parse_args(["--dummy", "--steps", "0", "--load_tiny_lora", str(source),
                           "--name", "example", "--save_dir", str(tmp_path)])
    with pytest.raises(FileExistsError):
        cli.train(args)
    assert (tmp_path / "example_inference.json").read_text() == "original\n"


def test_nonfinite_report_scale_rejected_before_model_or_rows(tmp_path, monkeypatch):
    source = make_export(tmp_path)
    monkeypatch.setattr(cli, "build_backend", forbidden)
    monkeypatch.setattr(cli, "load_slider_rows", forbidden)
    args = cli.parse_args(["--dummy", "--steps", "0", "--load_tiny_lora", str(source),
                           "--save_dir", str(tmp_path / "output"), "--report_scales", "nan,1"])
    with pytest.raises(ValueError, match="finite scales"):
        cli.train(args)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("corruption", ["fingerprint", "unsupported_protocol"])
def test_unverified_export_rejected_before_report_or_attachment(tmp_path, monkeypatch, corruption):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from particle_sliders.tiny_llm_backend import TinyLLMBackend
    from particle_sliders.tiny_native import _attention_targets

    source = make_export(tmp_path)
    tensors = load_file(str(source))
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["conceptmod"])
    metadata.update({"model_fingerprint": "wrong"} if corruption == "fingerprint"
                    else {"protocol": "tiny-qwen-shared-pg090-neutral-v2"})
    save_file(tensors, str(source), metadata={"conceptmod": json.dumps(metadata)})
    torch.manual_seed(91)
    backend = TinyLLMBackend(dummy=True)
    bindings = {key: module.forward for key, module in _attention_targets(backend.model).items()}
    args = cli.parse_args(["--dummy", "--steps", "0", "--load_tiny_lora", str(source),
                           "--save_dir", str(tmp_path / "output")])
    monkeypatch.setattr(cli, "load_slider_rows", forbidden)
    monkeypatch.setattr(cli, "emit_report", forbidden)
    with pytest.raises(ValueError, match="fingerprint" if corruption == "fingerprint" else "Unsupported"):
        cli.train(args, backend=backend)
    for key, module in _attention_targets(backend.model).items():
        assert module.forward == bindings[key]
    assert not (tmp_path / "output").exists()


def test_current_protocol_restores_packaged_dummy_vocabulary_without_game(tmp_path, monkeypatch):
    from particlegan import get_recipe
    from particle_sliders.pg09 import PG_VERSION
    from particle_sliders.tiny_llm_backend import TinyLLMBackend
    from particle_sliders.tiny_native import TinyParticleSlider
    from conceptmod.textsliders import tiny_llm_pg09 as facade

    torch.manual_seed(91)
    backend = TinyLLMBackend(dummy=True)
    backend.encode("retained heldout orchid vocabulary")
    fingerprint = facade.tensor_digest(backend.model.state_dict())
    network = TinyParticleSlider(backend.model)
    source = tmp_path / "qualified.safetensors"
    network.save(source, dict(protocol=facade.PROTOCOL, pg_version=PG_VERSION, core_version="0.6.0",
        package_pin=facade._portable_package_pin(facade.require_text_package()),pg_source_profile='release',
        adapter_dtype="float32", student_conditioning="neutral", teacher_conditioning="positive",
        critic=facade.CRITIC, source_sha256=hashlib.sha256(Path(facade.__file__).read_bytes()).hexdigest(),
        model_fingerprint=fingerprint, inference_sources=facade._inference_sources(backend),
        resolved_recipe=asdict(get_recipe("bispf_optimized", num_particles=128, z_dim=4,
                                        batch_size=4, total_steps=6, training_precision='float32')),
        dummy=True, dummy_tokenizer_vocab=dict(backend.tokenizer._vocab), completed_steps=0,
        quality_qualified=False))
    torch.manual_seed(91)
    fresh = TinyLLMBackend(dummy=True)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(cli, "load_slider_rows", forbidden)
    args = cli.parse_args(["--dummy", "--steps", "0", "--load_tiny_lora", str(source),
                           "--save_dir", str(tmp_path / "output"), "--no_report"])
    receipt = cli.train(args, backend=fresh)
    assert fresh.tokenizer._vocab == backend.tokenizer._vocab
    assert receipt["base_identity_verified"] is True
    assert receipt["protocol_validated"] is True
    assert not torch.cuda.is_initialized()


def test_finite_export_with_overflowing_response_does_not_write_report_json(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from particle_sliders.tiny_llm_backend import TinyLLMBackend

    source = make_export(tmp_path)
    tensors = load_file(str(source))
    for value in tensors.values():
        value.fill_(torch.finfo(torch.float32).max)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    save_file(tensors, str(source), metadata=metadata)
    torch.manual_seed(91)
    backend = TinyLLMBackend(dummy=True)
    output = tmp_path / "overflow-report"
    args = cli.parse_args(["--dummy", "--steps", "0", "--load_tiny_lora", str(source),
                           "--save_dir", str(output), "--report_scales", "0,1"])
    with pytest.raises(FloatingPointError, match="Non-finite Tiny"):
        cli.train(args, backend=backend)
    assert not (output / "report/hidden_delta.json").exists()
    assert not torch.cuda.is_initialized()
