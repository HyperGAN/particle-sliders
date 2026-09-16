"""The YuE2 port must use the release's game, span and stopping objective."""
import json

import pytest
import torch
import yaml
from safetensors.torch import load_file

pytest.importorskip("yue2")
from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END
from conceptmod.textsliders.yue2_backend import YuE2Backend, YuE2Slider
from conceptmod.textsliders.yue2_uni import RECIPE, end_margins, lyric_positions, prepare_rows, update
from conceptmod.textsliders.train_lora_yue2 import parse_args, train
from conceptmod.textsliders.lm_adv import SpanTransformerD
from conceptmod.textsliders.gan_v2.critic import pad_sequences


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def rows():
    return [{"neutral": "Warm piano, adult lead vocal", "positive": "Warm piano, adult female lead vocal",
             "lyrics": "[Verse]\nA café in the rain"},
            {"neutral": "Dry guitar, solo voice", "positive": "Dry guitar, solo female voice",
             "lyrics": "[Verse]\nWe leave the door ajar\n[Chorus]\nCome home"}]


def test_winning_defaults_and_no_hidden_hold():
    args = parse_args([])
    assert (args.recipe, args.rank, args.alpha, args.steps, args.lr, args.adv_batch,
            args.train_tokens, args.hold_weight) == ("uni16", 8, 8, 600, .0005, 4, 250, 0)
    assert RECIPE["pole_weight"] == RECIPE["lyrichold_weight"] == 0
    assert RECIPE["critic"]["readout"] == "mean_last"
    assert RECIPE["critic"]["in_mode"] == "scaled"
    for options in (["--hold_weight", ".1"], ["--steps", "601"]):
        with pytest.raises(SystemExit):
            parse_args(options)


def test_lyric_alignment_excludes_caption_and_abc_markers():
    backend = YuE2Backend(dummy=True)
    row = rows()[0]
    neu, ni = lyric_positions(backend, row["neutral"], row["lyrics"])
    pos, pi = lyric_positions(backend, row["positive"], row["lyrics"])
    assert ni[0] != pi[0]
    assert [neu[i] for i in ni] == [pos[i] for i in pi]
    assert bytes([neu[i] for i in ni[:-1]]).decode() == row["lyrics"]
    assert ni[-1] == len(neu) - 1
    assert len(neu) - 2 not in ni and len(neu) - 3 not in ni


def test_end_readout_matches_full_native_logits_and_gradients():
    backend = YuE2Backend(dummy=True)
    hidden = torch.randn(1, 3, 32, requires_grad=True)
    full = backend.model.lm_head(hidden)
    expected = full[..., MUSIC_END].float() - full[..., CODEC_OFFSET:CODEC_OFFSET+CODEC_SIZE].float().logsumexp(-1)
    actual = end_margins(backend.model, hidden)
    assert torch.allclose(actual, expected, atol=1e-6)
    g1, = torch.autograd.grad(expected.sum(), hidden)
    g2, = torch.autograd.grad(actual.sum(), hidden)
    assert torch.allclose(g1, g2, atol=1e-6)


def test_native_game_uses_fixed_prompt_deltas_and_updates_only_adapter():
    torch.manual_seed(7)
    backend = YuE2Backend(dummy=True)
    prepared = prepare_rows(backend, rows(), 3, 7, 512)
    for source, row in zip(rows(), prepared):
        prefix, indices = lyric_positions(backend, source["positive"], source["lyrics"])
        expected = backend.hidden(prefix)[:, indices].float() - row["neutral"]
        assert torch.equal(expected, row["real"])
        assert row["end_teacher"].shape == (1, 4)
    network = YuE2Slider(backend.model, rank=2, alpha=2)
    critic = SpanTransformerD(32, **RECIPE["critic"])
    real, mask = pad_sequences([r["real"] for r in prepared])
    critic.calibrate_input_scale(real, mask)
    g = torch.optim.AdamW(network.parameters(), lr=.0005, betas=(0.,.999), weight_decay=1e-6)
    d = torch.optim.Adam(critic.parameters(), lr=.00075, betas=(0.,.999))
    before = backend.hidden(prepared[0]["ids"]).detach().clone()
    metrics = update(backend, network, critic, g, d, prepared)
    assert metrics["end"] == 0
    assert metrics["grad_norm"] > 0 and all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert metrics["loss"] == pytest.approx(metrics["g_adv"]+metrics["fm"]+metrics["end"])
    assert any(a.lora_up.weight.abs().sum() > 0 for a in network.adapters.values())
    assert torch.equal(before, backend.hidden(prepared[0]["ids"]))
    assert all(p.grad is None for p in backend.model.parameters())


def test_full_state_resume_matches_uninterrupted(tmp_path):
    prompts = tmp_path / "pairs.yaml"
    prompts.write_text(yaml.safe_dump({"rows": rows()}))
    common = ["--dummy", "--recipe", "uni16", "--adv_batch", "2", "--rank", "2", "--alpha", "2",
              "--train_tokens", "2", "--prompts_file", str(prompts), "--name", "female-test"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(parse_args(common + ["--steps", "3", "--save_dir", str(full)]))
    train(parse_args(common + ["--steps", "2", "--save_dir", str(resumed)]))
    train(parse_args(common + ["--steps", "3", "--save_dir", str(resumed), "--resume_state",
                              str(resumed / "female-test_state.pt")]))
    a = load_file(str(full / "female-test_last.safetensors"))
    b = load_file(str(resumed / "female-test_last.safetensors"))
    assert all(torch.equal(a[k], b[k]) for k in a)
    state = torch.load(resumed / "female-test_state.pt", weights_only=True)
    assert state["completed"] == 3 and state["critic"] and state["g_optimizer"] and state["d_optimizer"]
    assert len((resumed / "female-test_train.jsonl").read_text().splitlines()) == 3
    meta = json.loads((resumed / "female-test_last.json").read_text())
    assert meta["recipe_settings"]["recipe"]["lyrichold_weight"] == 0
