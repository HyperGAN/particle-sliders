"""Archived cache diagnostics; live local-game recovery entry points are retired."""

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
