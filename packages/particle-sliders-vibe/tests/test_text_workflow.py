"""The teacher qualification policy can be tested without model dependencies."""

from types import SimpleNamespace
import pytest

# With the optional model extra installed this suite exercises the real collection logic.
pytest.importorskip("lumen_studio.particles")
from particle_sliders_vibe import text_workflow as flow
from particle_sliders_vibe.text_model import token_gain


def test_qualification_retries_missing_slots_and_requires_intact_replies(
    tmp_path, monkeypatch
):
    calls = []

    def reply(prompt, *, cue=""):
        calls.append((prompt, cue))
        return {"text": cue or "ordinary", "tokens": [1, 2]}

    def measure(prompt, reply):
        broken = (
            "a pottery workshop" in prompt
            and prompt.startswith("Write")
            and reply["text"] != "ordinary"
        )
        return {"effect": float(reply["text"] != "ordinary"), "broken": float(broken)}

    monkeypatch.setattr(flow, "TOPICS", flow.TOPICS[:16])
    pairs = flow.qualify(
        SimpleNamespace(reply=reply),
        SimpleNamespace(measure=measure),
        tmp_path,
        "warm",
        lambda *_: None,
    )
    assert len(pairs) == 16
    assert len([p for p, c in calls if not c]) == 17
    assert any(
        p["prompt"].startswith("Tell") and "pottery" in p["prompt"] for p in pairs
    )


def test_token_profiles_have_equal_dose_across_the_first_eight_tokens():
    for profile in ("constant", "early", "late"):
        assert sum(token_gain(profile, i) for i in range(8)) == 8.0
        assert token_gain(profile, 8) == 1.0
    assert token_gain("early", 0) > token_gain("late", 0)
    with pytest.raises(ValueError):
        token_gain("typo", 0)


def test_export_uses_intact_winner_and_replays_saved_schedule(monkeypatch, tmp_path):
    import json
    from dataclasses import dataclass
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    @dataclass
    class Row:
        target: int

    class Host:
        def __init__(self):
            self.identity = {"model": "tiny-aya"}
            self.mixer = SimpleNamespace(close=lambda: None)
            self.replay_profile = None

        def reply(self, prompt, *, strengths=None, profiles=None):
            if not strengths:
                gain = 0
            else:
                alpha = next(iter(strengths.values()))
                profile = (
                    next(iter(profiles.values())) if profiles else self.replay_profile
                )
                gain = (
                    10
                    if alpha == 2
                    else int(4 * alpha) + (2 if profile == "late" else 0)
                )
            return dict(text=str(gain), tokens=[gain])

        def load_adapter(self, key, path):
            with safe_open(path, framework="pt") as f:
                self.replay_profile = json.loads(f.metadata()["text_vibe"])["schedule"]

    judge = SimpleNamespace(
        measure=lambda p, r: dict(
            effect=int(r["text"]) / 10, broken=float(r["text"] == "10")
        )
    )

    def qualified(host, judge, root, vibe, progress):
        pairs = [dict(prompt="training prompt")]
        flow.write_json(root / "teachers.json", dict(qualified=pairs))
        return pairs

    monkeypatch.setattr(flow, "qualify", qualified)
    monkeypatch.setattr(flow, "freeze_rows", lambda *args: [Row(1)])
    monkeypatch.setattr(flow, "TextJudge", lambda *args: judge)

    def fit(
        host,
        rows,
        path,
        *,
        steps,
        generator_lr,
        critic_lr,
        horizon,
        seed,
        check_output,
        progress,
    ):
        save_file(dict(a=torch.ones(1)), path)
        check_output(steps)
        return dict(updates=steps, weights="raw")

    monkeypatch.setattr(flow, "fit", fit)
    out, report = flow.train_vibe(
        Host(), "warmth", tmp_path / "run", steps=1, progress=lambda *_: None
    )
    assert out.is_file()
    assert (report["alpha"], report["schedule"]) == (1.5, "late")
    assert report["confirmation"]["gain"] == 0.8
    assert report["replay_tokens"] == [8]
    assert len(json.loads((out.parent / "strengths.json").read_text())) == 12
    assert json.loads((out.parent / "fit-checks.json").read_text())[0]["updates"] == 1


def test_reusing_frozen_targets_binds_the_vibe_model_and_hashes(tmp_path):
    from dataclasses import asdict
    from hashlib import sha256
    import torch

    src = tmp_path / "source"
    src.mkdir()
    identity = {"model": "glimmer", "revision": "pinned"}
    pairs = [{"prompt": f"training question {i}"} for i in range(16)]
    flow.write_json(src / "teachers.json", {"qualified": pairs})
    tensor = torch.ones(1, 3)
    row = flow.PreparedResponse(
        flow.ResponseRow(tensor, tensor, None, tensor), tensor, tensor
    )
    torch.save([asdict(row)] * len(pairs), src / "frozen-targets.pt")
    flow.write_json(
        src / "frozen-source.json",
        dict(
            format="text-vibe-frozen-v1",
            identity=identity,
            vibe="warm",
            teachers_sha256=sha256((src / "teachers.json").read_bytes()).hexdigest(),
            targets_sha256=sha256((src / "frozen-targets.pt").read_bytes()).hexdigest(),
        ),
    )
    destination = tmp_path / "fresh"
    destination.mkdir()
    kept, rows = flow.load_frozen(src, destination, identity, "warm")
    assert kept == pairs
    assert torch.equal(rows[0].target, row.target)
    with pytest.raises(ValueError, match="different base model or vibe"):
        flow.load_frozen(src, destination, {"model": "tiny-aya"}, "warm")
    with pytest.raises(ValueError, match="different base model or vibe"):
        flow.load_frozen(src, destination, identity, "rhyme")
    (src / "frozen-targets.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        flow.load_frozen(src, destination, identity, "warm")


def test_product_defaults_match_the_qualified_training_budgets():
    from particle_sliders_vibe.text_model import TEXT_RECIPES, TextRecipe

    assert TEXT_RECIPES["tiny-aya"] == TextRecipe(448, 1024, 1, 0.00075, 0.009)
    assert TEXT_RECIPES["glimmer"] == TextRecipe(32, 128, 7, 0.001, 0.018)


def test_constant_text_selects_raw_before_alpha_and_reserves_before_confirmation(
    tmp_path, monkeypatch
):
    host = SimpleNamespace(mixer=SimpleNamespace(close=lambda: None))

    def load(key, path):
        host.raw = path.name
        return {"updates": 16 if host.raw.startswith("raw-") else 32}

    host.load_adapter = load
    raw = tmp_path / "raw-16.safetensors"
    last = tmp_path / "trained.safetensors"
    raw.write_bytes(b"first RAW")
    last.write_bytes(b"second RAW")

    def compare(host, judge, prompts, off, alpha, profile):
        assert prompts == flow.RANK and profile == "constant"
        # Only the first RAW has an intact unit-strength effect. After choosing
        # it, the scalar search must expand past two and refine that same RAW.
        gain = 1 - (alpha - 2.375) ** 2 / 3 if host.raw == raw.name else -1
        return dict(alpha=alpha, schedule=profile, gain=gain, intact=True)

    monkeypatch.setattr(flow, "compare", compare)
    chosen, meta, winner = flow.select_constant_text(
        host,
        None,
        tmp_path,
        last,
        [{"updates": 16}, {"updates": 32}],
        [],
        lambda *args: None,
    )
    assert chosen == raw and meta["updates"] == 16
    assert abs(winner["alpha"] - 2.375) <= 1 / 32
    selection = flow.read_json(tmp_path / "confirmation-selection.json")
    assert selection["raw_sha256"] == flow.sha256(raw.read_bytes()).hexdigest()
    assert selection["prompts"] == list(flow.CONFIRM)
    assert selection["alpha"] == winner["alpha"]


def test_authored_instructions_render_then_freeze_the_winning_explicit_cue(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "TOPICS", flow.TOPICS[:16])
    plan = {
        str(i): {"neutral": f"neutral {i}", "treatments": [
            {"positive": f"render {j}", "explicit": f"strong {j}"} for j in range(3)
        ]} for i in range(16)
    }
    calls = []
    def reply(prompt, *, cue=""):
        assert prompt.startswith("neutral ")
        calls.append(cue)
        return {"text": cue or "plain", "tokens": [1, 2]}
    judge = SimpleNamespace(measure=lambda prompt, r: {
        "broken": 0, "effect": 1.0 if r["text"] == "render 2" else 0,
    })
    pairs = flow.qualify(SimpleNamespace(reply=reply), judge, tmp_path, "warmth", lambda *_: None, plan)
    assert calls == ["", "render 0", "render 1", "render 2"] * 16
    assert len(pairs) == 16 and all(p["cue"] == "strong 2" for p in pairs)
    assert all(p["sampling_cue"] == "render 2" for p in pairs)
