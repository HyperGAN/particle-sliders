"""Exact native games cover lazy cap4/8, EMA, targets and all RNG streams."""

import random
import numpy as np
import pytest
import torch
from conceptmod.textsliders.caption_recovery import CaptionState, FrozenReadouts
from conceptmod.textsliders import (
    train_lora_tiny_llm as tiny,
    train_lora_bonsai_gguf as bonsai,
)
from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend, DummyTokenizer
from conceptmod.textsliders.bonsai_gguf_backend import BonsaiGGUFBackend


def same(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


class Observer:
    def __init__(self, root, until):
        self.state = CaptionState(root, "test")
        self.until = until
        self.metrics = []

    def ready(self, c):
        self.c = c
        return self.state.restore(c)

    def before_step(self, step):
        pass

    def after_step(self, step, metrics):
        self.metrics.append(metrics)
        self.state.save(step, self.c)
        return step < self.until

    def finished(self, sidecar):
        self.sidecar = sidecar


@pytest.mark.parametrize("kind", ["tiny", "real-tiny-qwen", "bonsai"])
def test_native8_equals4_plus4_and_unobserved(tmp_path, monkeypatch, kind):
    torch.set_num_threads(1)
    if kind == "real-tiny-qwen":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from conceptmod.textsliders import tiny_llm_backend as backend_module

        def load(*args, **kwargs):
            config = Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
                attention_dropout=0.0,
            )
            return DummyTokenizer(), Qwen3ForCausalLM(config)

        monkeypatch.setattr(backend_module, "_load_live_model", load)
    trainer = bonsai if kind == "bonsai" else tiny

    def run(name, until, observe=True):
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        args = trainer.parse_args(
            [
                "--dummy",
                "--steps",
                "8",
                "--name",
                "test",
                "--no_report",
                "--save_dir",
                str(tmp_path / name / "export"),
            ]
        )
        backend = (
            BonsaiGGUFBackend(dummy=True)
            if kind == "bonsai"
            else TinyLLMBackend(dummy=kind != "real-tiny-qwen")
        )
        if kind == "bonsai" and name != "normal":
            backend = FrozenReadouts(
                backend, tmp_path / "cache", "test-official-identity"
            )
        obs = Observer(tmp_path / name / "state", until) if observe else None
        record = trainer.train(args, backend, observer=obs)
        return obs, record

    full, _ = run("full", 8)
    first, first_record = run("split", 4)
    resumed, resumed_record = run("split", 8)
    assert (
        first_record["steps"] == 4
        and first_record["planned_steps"] == 8
        and resumed_record["steps"] == 8
    )
    assert full.metrics == first.metrics + resumed.metrics
    a = torch.load(tmp_path / "full/state/step-000008.pt", weights_only=True)
    b = torch.load(tmp_path / "split/state/step-000008.pt", weights_only=True)
    same(a, b)
    run("normal", 8, observe=False)
    from safetensors.torch import load_file

    same(
        a["network"],
        load_file(str(tmp_path / "normal/export/test_live_last.safetensors")),
    )


def test_frozen_cache_exact_requests_and_offline_replay(tmp_path):
    backend = BonsaiGGUFBackend(dummy=True)
    calls = []
    read = backend.hidden

    def track(enc):
        calls.append(enc.text)
        return read(enc)

    backend.hidden = track
    cache = FrozenReadouts(
        backend, tmp_path, "pinned-model-runtime-pooling-normalization"
    )
    enc = cache.encode("a garden")
    first = cache.hidden(enc)
    second = cache.hidden(enc)
    assert len(calls) == 2 and torch.equal(first, second)
    offline = FrozenReadouts(
        None, tmp_path, "pinned-model-runtime-pooling-normalization"
    )
    assert torch.equal(first, offline.hidden(offline.encode("a garden")))
    with pytest.raises(FileNotFoundError):
        offline.encode("uncached")
    path = next(tmp_path.glob("*.pt"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cache.hidden(enc)


def test_nonrepeatable_readout_rejects_cache(tmp_path):
    backend = BonsaiGGUFBackend(dummy=True)
    backend.hidden = lambda enc: torch.randn(1, 5120)
    cache = FrozenReadouts(backend, tmp_path, "test")
    with pytest.raises(ValueError, match="not bitwise repeatable"):
        cache.hidden(cache.encode("caption"))
    assert not list(tmp_path.glob("*.pt"))


def test_quality_gate_requires_directional_transfer_and_preservation():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from caption_quality import quality_gate

    records = [
        dict(scale=1.0, relative_edit_mse=0.8, edit_cosine=0.5, edit_rms_ratio=0.8)
        for _ in range(3)
    ]
    assert quality_gate(records, 0.1, 0.001)
    assert not quality_gate(records, 0.3, 0.001)
    assert not quality_gate(records, 0.1, 0.02)
    assert not quality_gate(
        [dict(r, relative_edit_mse=1.1) for r in records], 0.1, 0.001
    )


@pytest.mark.parametrize("model", ["tiny", "bonsai"])
def test_real_qualification_observer_preserves_native_optimizer_replay(
    tmp_path, monkeypatch, model
):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from qualify_captions import Observer as Qualification

    torch.set_num_threads(1)
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    trainer = tiny if model == "tiny" else bonsai
    backend = (
        TinyLLMBackend(dummy=True)
        if model == "tiny"
        else FrozenReadouts(BonsaiGGUFBackend(dummy=True), tmp_path / "cache", "fixed")
    )
    recipe = dict(
        model=model,
        base_identity="test",
        code_commit="test",
        model_id="test",
        generate_controls=False,
    )
    args = trainer.parse_args(
        [
            "--dummy",
            "--steps",
            "4",
            "--name",
            "test",
            "--no_report",
            "--save_dir",
            str(tmp_path / "native"),
        ]
    )
    run = tmp_path / "run"
    run.mkdir()
    observer = Qualification(recipe, run, 2)
    result = trainer.train(args, backend, observer=observer)
    assert result["steps"] == 2 and result["planned_steps"] == 4
    observer.eval(2)
    assert (
        observer.progress["zero_exact"] and observer.progress["ema_export_reload_exact"]
    )
    assert observer.progress["committed_step"] == 2
