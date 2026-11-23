from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from particle_sliders import alpha_tune
from particle_sliders.alpha_search import AlphaObservation
from particle_sliders.candidate_failure import FileRecord
from particle_sliders.execution.cli import should_dispatch


def definition(tmp_path):
    source = tmp_path / "slider.json"
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"saved adapter")
    off = tmp_path / "off.png"
    off.write_bytes(b"neutral")
    rows = [
        {"id": split, "split": split, "request": {"seed": i}, "preserve": split,
         "off": asdict(FileRecord.from_path(off))}
        for i, split in enumerate(("search", "reserved"))
    ]
    payload = dict(schema=alpha_tune.SCHEMA, selection="raw", ema_decay=None, judge="jev",
                   renderer="fixture", checkpoint=asdict(FileRecord.from_path(checkpoint)),
                   concept="boss costume", profiles={"constant": [.3, .5, .75, 1.]},
                   refinement_rounds=0, cases=rows)
    source.write_text(json.dumps(payload))
    return source, payload


def test_uses_jev_to_tune_saved_raw_and_reuses_committed_result(tmp_path, monkeypatch):
    source, payload = definition(tmp_path)
    renders = []

    class Judge:
        def __init__(self, receipts, concept):
            assert concept == payload["concept"]

        def observe(self, off, on, *, preserve):
            on.verify()
            alpha = float(Path(on.path).read_text())
            limit = .8 if preserve == "search" else .6
            status = "fail" if alpha >= limit else "pass"
            return SimpleNamespace(observation=AlphaObservation(0., alpha, status),
                                   checks={"defect": float(status == "fail")})

    class Renderer:
        interval_widths = (1.,)
        sampler = {"steps": 1, "model": "fixture"}

        def render(self, request, gains, output):
            renders.append((request, gains))
            output.write_text(str(gains[0]))
            return FileRecord.from_path(output)

    @contextmanager
    def open_renderer(checkpoint, options):
        assert asdict(checkpoint) == payload["checkpoint"]
        yield Renderer()

    monkeypatch.setattr(alpha_tune, "JevJudge", Judge)
    monkeypatch.setattr(alpha_tune, "_renderer", lambda _: open_renderer)
    receipt = alpha_tune.tune(source)
    saved = json.loads(receipt.read_text())
    assert saved["checkpoint"] == payload["checkpoint"]
    assert saved["judge"] == "jev"
    assert saved["selected"] == [
        {"alpha": .5, "profile": "constant", "effective_strengths": [.5],
         "step_gains": [1.], "search_gain": .5, "reserved_gain": .5},
    ]
    assert len(renders) == 6
    assert json.loads((receipt.parent / "validation.json").read_text()) == [
        {"profile": "constant", "alpha": .75, "status": "fail"},
        {"profile": "constant", "alpha": .5, "status": "pass"},
    ]
    monkeypatch.setattr(alpha_tune, "JevJudge", lambda *_: pytest.fail("repeated paid setup"))
    monkeypatch.setattr(alpha_tune, "_renderer", lambda *_: pytest.fail("reopened saved model"))
    assert alpha_tune.tune(source) == receipt
    payload["concept"] = "different concept"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="definition changed"):
        alpha_tune.tune(source)


def test_refuses_alternate_judge_before_creating_run(tmp_path):
    source, payload = definition(tmp_path)
    payload["judge"] = "other"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="uses Jev"):
        alpha_tune.tune(source)
    assert not (tmp_path / "slider.alpha-tune").exists()


def test_shared_command_accepts_alpha_tune_subcommand():
    assert should_dispatch(["alpha-tune", "slider.json"])
