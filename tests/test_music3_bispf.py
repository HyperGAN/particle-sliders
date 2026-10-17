"""Weights-free Music 3 CLI integration, exports and split-run continuation."""
import json

import pytest
import torch
import yaml

from conceptmod.textsliders import music3_particle_bridge as bridge
from conceptmod.textsliders import train_lora_music3_particle as cli


def _same(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    else:
        assert a == b


def test_dummy_cli_exports_and_exact_split_resume(tmp_path, monkeypatch):
    # The product fixture runs its real routing, updater and serialization,
    # with a 32-wide fake LM. It neither downloads nor loads Music 3 weights.
    monkeypatch.setattr(bridge, "RECIPE", dict(bridge.RECIPE))
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(yaml.safe_dump(dict(rows=[
        dict(neutral="soft piano", positive="bright piano", lyrics="morning light"),
        dict(neutral="soft guitar", positive="bright guitar", lyrics="evening rain"),
        dict(neutral="warm strings", positive="bright strings", lyrics="carry us home"),
    ])))
    def run(name, until):
        args = cli.parse_args(["--dummy", "--device", "cpu", "--prompts_file", str(prompts),
            "--save_dir", str(tmp_path/name), "--steps", "3", "--until", str(until),
            "--batch_size", "4", "--max_seq_len", "512", "--save_every", "1", "--seed", "0"])
        cli.train(args)
        return torch.load(tmp_path/name/"state.pt", weights_only=True)
    first = run("split", 1)
    assert first["completed"] == first["bispf"]["completed_steps"] == 1
    split = run("split", 3)
    full = run("full", 3)
    _same(split["bispf"], full["bispf"])
    for a, b in zip(split["history"], full["history"]):
        _same({k: v for k, v in a.items() if k != "step_seconds"},
              {k: v for k, v in b.items() if k != "step_seconds"})
    _same(split["ema"], split["network"])
    assert split["history"][0]["cos_pos"] == pytest.approx(0.)
    assert split["history"][-1]["cos_pos"] > 0.
    for row in split["history"]:
        assert (row["g_lr"], row["d_lr"], row["particle_lr"], row["noise_std"]) == (.0006, .0009, .006, 1.)
    from safetensors.torch import load_file
    exported = load_file(str(tmp_path/"split/music3-particle_last.safetensors"))
    live = load_file(str(tmp_path/"split/music3-particle_live_last.safetensors"))
    _same(exported, live)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        backend = bridge.Music3Backend(dummy=True, device="cpu")
    ids, mask = backend.encode_text("soft piano", "morning light")
    baseline = backend.hidden(ids, mask).detach().clone()
    adapter, metadata = bridge.ParticleNetwork.load(backend.model,
        tmp_path/"split/music3-particle_last.safetensors")
    with adapter.scaled(0.):
        torch.testing.assert_close(backend.hidden(ids, mask), baseline, rtol=0, atol=0)
    with adapter.scaled(1.):
        assert not torch.equal(backend.hidden(ids, mask), baseline)
    assert metadata["weights_kind"] == "live"
    status = json.loads((tmp_path/"split/status.json").read_text())
    assert status["status"] == "complete" and status["completed"] == 3


def test_bispf_predict_uses_neutral_caption():
    calls = []
    class Backend:
        dummy = True
        def hidden(self, ids, mask, train):
            calls.append(ids.clone())
            return torch.ones(1, 3)
    class Network:
        from contextlib import nullcontext
        def scaled(self, scale):
            return self.nullcontext()
    class Critic:
        target_mean = torch.zeros(3)
        def normalize(self, value):
            return value
    rows = [dict(neutral_ids=torch.tensor([[1]]), neutral_mask=torch.ones(1, 1),
                 positive_ids=torch.tensor([[2]]), positive_mask=torch.ones(1, 1),
                 targets=torch.ones(1, 3), neutral=torch.zeros(1, 3))]
    class Update:
        def step(self, index, predict):
            predict(0, "g")
            return dict(g_rows=[0])
    bridge.update(Backend(), Network(), Critic(), None, None, rows,
                  sampler=None, step=1, bispf=Update())
    assert calls[0].item() == 1
