"""Exercise blinding, durable ratings, phase locks and HTTP audio seeking."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study
import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    source = study.read(study.WORK / "data/state.json")
    monkeypatch.setattr(study, "DATA", tmp_path)
    monkeypatch.setenv("LOFI_TEST_NO_WORKER", "1")
    study.write(tmp_path / "state.json", source)
    mark_ready(source)
    with TestClient(server.app) as c:
        yield c


def mark_ready(state):
    for clip_id, clip in state["clips"].items():
        for ext in ("mp3", "wav"):
            p = study.DATA / "audio" / f"{clip_id}.{ext}"
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(b"0123456789" * 100)
        study.write(study.DATA / "audio" / f"{clip_id}.json", dict(duration=20., identity=clip["identity"]))


def finish(c, stage):
    state = c.get("/api/study").json()
    for case in state["stages"][stage]["cases"]:
        for take in case["takes"]:
            assert c.post("/api/rate/"+take["id"], json=dict(style=4,enjoyment=3,words=5,issues=[])).status_code == 200
    assert c.post("/api/reveal/"+stage, json={}).status_code == 200


def test_blind_api_audio_ranges_and_private_paths(client):
    body = client.get("/api/study").json()
    s = json.dumps(body)
    assert "step600" not in s and "safetensors" not in s and "checkpoint_sha256" not in s
    assert body["stages"]["shortlist"]["summary"] == []
    assert client.get("/api/export").json()["stages"] == {}
    take = body["stages"]["shortlist"]["cases"][0]["takes"][0]
    audio = client.get(take["preview"], headers={"Range":"bytes=10-29"})
    assert audio.status_code == 206 and audio.content == b"0123456789"*2
    assert client.head(take["original"]).headers["content-length"] == "1000"
    for url in ("/data/state.json", "/api/protocol", "/audio/state.json", "/audio/../../study.py"):
        assert client.get(url).status_code == 404
    assert client.post("/api/reveal/shortlist", json={}).status_code == 400
    assert client.post("/api/choose/shortlist", json={"choices":["step600-s1","step2000-s1"]}).status_code == 400
    assert client.post("/api/rate/"+take["id"],json={"style":99}).status_code == 400
    assert client.post("/api/rate/"+take["id"],json={"style":4},headers={"Origin":"https://another.example"}).status_code == 403


def test_saved_ratings_survive_reload_and_freeze_before_new_songs(client):
    finish(client, "shortlist")
    stage = client.get("/api/study").json()["stages"]["shortlist"]
    take = stage["cases"][0]["takes"][0]
    assert study.load()["ratings"][take["id"]]["words"] == 5
    assert take["label"] and len(stage["summary"]) == 6
    assert client.post("/api/rate/"+take["id"],json={"style":1}).status_code == 400
    assert client.post("/api/choose/shortlist",json={"choices":["step600-s1","step2000-s1"]}).status_code == 200
    assert client.post("/api/choose/shortlist",json={"choices":["step3000-s1","step3400-s1"]}).status_code == 400
    state = study.load()
    assert "confirmation" not in state["stages"]
    assert len(state["stages"]["strength"]["cases"]) == 4
    assert len(state["clips"]) == 40  # 24 old + 16 new strengths; no re-rendered controls.
    mark_ready(state)
    finish(client, "strength")
    assert client.post("/api/choose/strength",json={"choices":["step600-s0.5","step600-s0.75"]}).status_code == 400
    assert client.post("/api/choose/strength",json={"choices":["step600-s0.5","step2000-s0.75"]}).status_code == 200
    state = study.load()
    assert len(state["stages"]["confirmation"]["cases"]) == 36
    assert len(state["stages"]["full"]["cases"]) == 6
    assert state["selections"]["strength"] == ["step600-s0.5","step2000-s0.75"]
    fresh = client.get("/api/study").json()["stages"]["confirmation"]
    assert fresh["total"] == 144 and fresh["ready"] == 0
    assert all("config" not in t and "label" not in t for c in fresh["cases"] for t in c["takes"])
    assert client.post("/api/reveal/confirmation",json={}).status_code == 400
    # Resubmission cannot change the frozen candidates after confirmation begins.
    assert client.post("/api/choose/strength",json={"choices":["step600-s1","step2000-s1"]}).status_code == 400
    assert set(client.get("/api/export").json()["stages"]) == {"shortlist","strength"}


def test_protocol_keeps_confirmation_and_full_songs_disjoint(client):
    state = study.load()
    monitor = {f["row"]["lyrics"] for f in state["monitoring"]}
    confirm = {f["row"]["lyrics"] for f in state["confirmation"]}
    full = {f["row"]["lyrics"] for f in state["full"]}
    assert len(confirm) == 12 and len(full) == 3
    assert not monitor & confirm and not monitor & full and not confirm & full
    assert {f["seed"] for f in state["confirmation"]} == {1709,2903,4517}
    assert {f["duration"] for f in state["full"]} == {90.}
