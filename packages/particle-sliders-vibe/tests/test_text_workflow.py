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
    monkeypatch.setattr(flow, "qualify", lambda *args: [dict(prompt="training prompt")])
    monkeypatch.setattr(flow, "freeze_rows", lambda *args: [Row(1)])
    monkeypatch.setattr(flow, "TextJudge", lambda *args: judge)

    def fit(host, rows, path, *, steps, check_output, progress):
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
