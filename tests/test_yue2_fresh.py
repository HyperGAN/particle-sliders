"""Fresh-from-start sampling, frozen teachers, and uninterrupted-state parity."""
import json

import pytest
import torch
import yaml

pytest.importorskip("yue2")
from conceptmod.textsliders.train_lora_yue2_fresh import (
    FRESH_RECIPE, RowSampler, fresh_history, prepare, parse_args, train,
)
from conceptmod.textsliders.yue2_backend import YuE2Backend, YuE2Slider


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def rows():
    return [dict(neutral="Piano, solo lead", positive="Piano, female lead", lyrics="[verse]\nWe leave a little room"),
            dict(neutral="Guitar, solo lead", positive="Guitar, female lead", lyrics="[verse]\nThe rain is falling softly")]


def same(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b


def test_balanced_prompt_sampler_and_exact_resume():
    sampler = RowSampler(4)
    first = [sampler.next() for _ in range(7)]
    saved = sampler.state_dict()
    rest = [sampler.next() for _ in range(9)]
    restored = RowSampler(4, seed=99)
    restored.load_state_dict(saved)
    assert rest == [restored.next() for _ in range(9)]
    all_rows = first + rest
    assert all(sorted(all_rows[i:i+4]) == list(range(4)) for i in range(0,16,4))


def test_no_special_600_boundary_or_step_cap(tmp_path):
    args = parse_args(["--save_dir", str(tmp_path)])
    assert args.steps == args.until == 3400
    assert FRESH_RECIPE["phase_changes"] == [] and FRESH_RECIPE["batch"] == 1
    assert FRESH_RECIPE["parameter_step_limit"] is None
    assert FRESH_RECIPE["adversarial_weight"] == FRESH_RECIPE["fm_weight"] == FRESH_RECIPE["end_weight"] == 1
    assert FRESH_RECIPE["lyrichold_weight"] == FRESH_RECIPE["pole_weight"] == 0


def test_histories_vary_but_frozen_targets_and_base_remain_exact():
    backend = YuE2Backend(dummy=True)
    fixed = prepare(backend, rows(), 4, 512)
    network = YuE2Slider(backend.model, rank=2, alpha=2)
    a, tokens_a, audit_a = fresh_history(backend, network, rows()[0], fixed[0], 4, 3000001, False)
    for adapter in network.adapters.values():
        torch.nn.init.normal_(adapter.lora_up.weight, std=.5)
    with network.scaled(1):
        b, tokens_b, audit_b = fresh_history(backend, network, rows()[0], fixed[0], 4, 3000001, False)
        assert all(adapter.multiplier == 1 for adapter in network.adapters.values())
    c, tokens_c, audit_c = fresh_history(backend, network, rows()[0], fixed[0], 4, 3000002, False)
    assert tokens_a == tokens_b and audit_a["tokens_sha256"] == audit_b["tokens_sha256"]
    assert torch.equal(a["end_teacher"], b["end_teacher"])
    assert tokens_a != tokens_c and audit_a["tokens_sha256"] != audit_c["tokens_sha256"]
    assert torch.equal(a["real"], c["real"]) and torch.equal(a["neutral"], c["neutral"])
    assert all(not p.requires_grad for p in backend.model.parameters())


def test_full_game_resume_is_exact_from_step_one(tmp_path):
    prompts = tmp_path / "rows.yaml"
    prompts.write_text(yaml.safe_dump({"rows": rows()}))
    common = ["--dummy", "--rank", "2", "--alpha", "2", "--train_tokens", "4",
              "--steps", "4", "--prompts_file", str(prompts), "--history_backend", "eager"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(parse_args(common + ["--save_dir", str(full)]))
    train(parse_args(common + ["--save_dir", str(resumed), "--until", "2"]))
    train(parse_args(common + ["--save_dir", str(resumed)]))
    a = torch.load(full / "state.pt", weights_only=True)
    b = torch.load(resumed / "state.pt", weights_only=True)
    for key in ("network","critic","g_optimizer","d_optimizer","sampler","rng"):
        assert same(a[key], b[key]), key
    assert [r["history"]["seed"] for r in b["history"]] == list(range(3000001,3000005))
    assert len({r["history"]["tokens_sha256"] for r in b["history"]}) == 4
    assert sorted(r["row"] for r in b["history"]) == [0,0,1,1]
    assert b["history"][0]["end"] == 0
    assert json.loads((resumed/"status.json").read_text())["status"] == "complete"
    with pytest.raises(ValueError, match="Resume"):
        train(parse_args(common + ["--save_dir", str(resumed), "--seed_start", "4000001"]))
