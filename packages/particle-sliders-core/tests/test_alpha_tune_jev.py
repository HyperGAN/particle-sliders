"""The Jev observer is exercised through a hermetic transport."""

import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from particle_sliders import alpha_tune_jev
from particle_sliders.alpha_search import AlphaObservation
from particle_sliders.candidate_failure import FileRecord


def _image(path, content):
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + content)
    return FileRecord.from_path(path)


def _inputs(payload):
    return [base64.b64decode(part["image_url"]["url"].partition(",")[2])[8:]
            for part in payload["state"] if isinstance(part, dict)]


def _response(payload, *, scores=None, checks=None):
    scores = scores or {}
    checks = checks or {}
    images = _inputs(payload)
    answers = {}
    for name, question in payload["questions"].items():
        kind = question["type"]
        default = .5 if name.startswith("clear_") else .01
        value = scores.get(images[0], 1.5) if kind == "score" else checks.get(name, default)
        answers[name] = {"type": kind, kind: value}
    return {"model": "typesafe/jev-1.14-20261014", "answers": answers}


def _transport(monkeypatch, respond):
    requests = []

    def send(request, *, timeout):
        assert request.full_url == "https://api.nano-gpt.com/api/v1/decisions"
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") == "Bearer test-jev-token"
        assert timeout > 0
        payload = json.loads(request.data)
        assert payload["model"] == "typesafe/jev-1.14"
        requests.append(payload)
        return BytesIO(json.dumps(respond(payload)).encode())

    monkeypatch.setenv("NANOGPT_API_KEY", "test-jev-token")
    monkeypatch.setattr(alpha_tune_jev, "urlopen", send)
    return requests


@pytest.fixture
def setup(tmp_path):
    receipts = tmp_path / "receipts"
    judge = alpha_tune_jev.JevJudge(receipts, "ornate regal armor")
    return judge, receipts, _image(tmp_path / "off.png", b"off"), _image(tmp_path / "on.png", b"on")


def test_independent_scores_and_off_cache_for_more_than_one_alpha(setup, monkeypatch, tmp_path):
    judge, receipts, off, on = setup
    requests = _transport(monkeypatch, lambda payload: _response(payload, scores={b"off": .75, b"on": 2.25}))
    first = judge.observe(off, on, preserve="original garden")
    second = judge.observe(_image(tmp_path / "elsewhere.png", b"off"),
                           _image(tmp_path / "other_alpha.png", b"other_alpha"), preserve="original garden")

    assert first.observation == AlphaObservation(.25, .75, "pass")
    assert second.observation == AlphaObservation(.25, .5, "pass")
    assert [_inputs(item) for item in requests] == [
        [b"off"], [b"on"], [b"off", b"on"], [b"other_alpha"], [b"off", b"other_alpha"],
    ]
    assert requests[0]["questions"] == requests[1]["questions"] == requests[3]["questions"]
    assert requests[0]["state"][0] == requests[1]["state"][0] == requests[3]["state"][0]
    assert set(requests[2]["questions"]) == {"identity_replaced", "scene_lost"}
    assert "original garden" in requests[2]["state"][0]
    assert "concept" not in first.checks and first.checks["broken_anatomy"] == .01
    assert all("test-jev-token" not in path.read_text() for path in receipts.glob("*.json"))

    monkeypatch.delenv("NANOGPT_API_KEY")
    assert judge.observe(off, on, preserve="original garden") == first
    assert len(requests) == 5


@pytest.mark.parametrize(("value", "status"), [(.49, "pass"), (.5, "unmeasured"), (.8, "fail")])
def test_explicit_gate_thresholds_keep_score_independent(setup, monkeypatch, value, status):
    judge, _, off, on = setup
    _transport(monkeypatch, lambda payload: _response(payload, checks={"scene_lost": value}))
    assessment = judge.observe(off, on, preserve="original garden")
    assert assessment.checks["scene_lost"] == value
    assert assessment.observation == AlphaObservation(.5, .5, status)


def test_jev_clarifies_uncertain_checks_without_rescoring_intensity(setup, monkeypatch):
    judge, _, off, on = setup
    requests = _transport(monkeypatch, lambda payload: _response(payload, checks={
        "broken_anatomy": .58, "scene_lost": .57, "clear_broken_anatomy": .92, "clear_scene_lost": .9,
    }))
    result = judge.observe(off, on, preserve="original garden")
    assert result.observation == AlphaObservation(.5, .5, "pass")
    assert result.checks["broken_anatomy"] == pytest.approx(.08)
    assert result.checks["scene_lost"] == pytest.approx(.1)
    assert [_inputs(item) for item in requests[3:]] == [[b"on"], [b"off", b"on"]]
    assert set(requests[3]["questions"]) == {"clear_broken_anatomy"}
    assert set(requests[4]["questions"]) == {"clear_scene_lost"}


def test_uncertain_paid_transport_keeps_existing_off_and_never_retries(setup, monkeypatch):
    judge, receipts, off, on = setup

    def respond(payload):
        if _inputs(payload) == [b"on"]:
            raise TimeoutError("unknown server state")
        return _response(payload)

    requests = _transport(monkeypatch, respond)
    first = judge.observe(off, on, preserve="reference condition")
    assert first.observation == AlphaObservation(.5, None, "unmeasured")
    assert judge.observe(off, on, preserve="reference condition") == first
    pending, = receipts.glob("*.pending")
    assert json.loads(pending.read_text())["images"][0]["sha256"] == on.sha256
    assert len(requests) == 2


@pytest.mark.parametrize("malformed", [True, False])
def test_invalid_response_or_http_error_is_kept_as_unmeasured(setup, monkeypatch, malformed):
    judge, receipts, off, on = setup

    def respond(payload):
        if _inputs(payload) == [b"on"]:
            if not malformed:
                raise HTTPError(alpha_tune_jev.JEV_URL, 429, "rate limit", {}, BytesIO())
            body = _response(payload)
            body["answers"]["concept"] = {"type": "score", "score": True}
            return body
        return _response(payload)

    requests = _transport(monkeypatch, respond)
    first = judge.observe(off, on, preserve="reference condition")
    assert first.observation == AlphaObservation(.5, None, "unmeasured")
    assert judge.observe(off, on, preserve="reference condition") == first
    assert not list(receipts.glob("*.pending"))
    statuses = {json.loads(path.read_text())["status"] for path in receipts.glob("*.json")}
    assert statuses == {"ok", "invalid_result" if malformed else "http_error"}
    assert len(requests) == 2


def test_corrupted_images_and_cached_response_fail_before_new_paid_call(setup, monkeypatch):
    judge, receipts, off, on = setup
    requests = _transport(monkeypatch, _response)
    judge.observe(off, on, preserve="reference condition")
    for path in receipts.glob("*.json"):
        record = json.loads(path.read_text())
        if record["request"]["images"] == [
            {"sha256": off.sha256, "bytes": off.bytes, "media_type": "image/png"},
        ]:
            record["response"]["answers"]["concept"]["score"] = 3
            path.write_text(json.dumps(record))
            break
    with pytest.raises(ValueError, match="does not bind"):
        judge.observe(off, on, preserve="reference condition")
    Path(on.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="finished-image bytes differ"):
        judge.observe(off, on, preserve="reference condition")
    assert len(requests) == 3


def test_concurrent_observations_share_first_request_receipt(setup, monkeypatch):
    judge, _, off, on = setup
    requests = _transport(monkeypatch, _response)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(judge.observe, off, on, preserve="reference condition") for _ in range(2)]
        results = [future.result() for future in futures]
    assert results[0] == results[1]
    assert len(requests) == 3
