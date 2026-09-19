"""Critical scoring semantics, leakage prevention, and response persistence."""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slider_selection import model as M
from slider_selection.dataset import RATINGS, build_session, read_labels, validate_response
from slider_selection.features import FEATURE_NAMES, digest, lyric_features, windows, write_json, fullband_features
from slider_selection.server import make_server


def test_phrase_alignment_catches_garble_without_penalizing_whole_line_repeats():
    sheet = "[verse] / we carry morning home / [chorus] / the little lights are on"
    good = lyric_features(sheet, "we carry morning home the little lights are on the little lights are on")
    wrong_order = lyric_features(sheet, "home morning carry we on are lights little the")
    extra = lyric_features(sheet, "we carry morning home the little lights are on random invented addition")
    loop = lyric_features(sheet, "we we we we we we we")
    assert good["phrase_accuracy"] == 1
    assert wrong_order["recall"] == 1 and wrong_order["phrase_accuracy"] < .7
    assert extra["recall"] == 1 and extra["precision"] < .8
    assert loop["phrase_accuracy"] < .5 and loop["recall"] < .5


@pytest.mark.parametrize("length", [1, 5000, 160000, 320251, 1440000])
def test_audio_windows_cover_full_file(length):
    sections = windows(length, 16000)
    covered = np.zeros(length, dtype=bool)
    for start, end in sections:
        assert 0 <= start < end <= length
        assert end - start <= 160000
        covered[start:end] = True
    assert covered.all()


def test_fullband_detects_high_frequency_noise_even_when_stereo_downmix_cancels(tmp_path):
    import soundfile as sf
    rate = 44100
    time = np.arange(rate) / rate
    low = .1*np.sin(2*np.pi*440*time)
    high = .1*np.sin(2*np.pi*17000*time)
    base, attacked = tmp_path/"base.wav", tmp_path/"attacked.wav"
    sf.write(base, np.column_stack([low, low]), rate)
    sf.write(attacked, np.column_stack([low+high, low-high]), rate)
    assert fullband_features(base)["hf14k_fraction"] < .001
    assert fullband_features(attacked)["hf14k_fraction"] > .4


def data(count=18, prefix="train"):
    records = []
    for i in range(count):
        good = i % 2
        records.append({"id": f"{prefix}{i}", "candidate": f"{prefix}-candidate",
                        "recipe_family": f"{prefix}-recipe{i % 3}",
                        "prompt_family": f"{prefix}-prompt{i // 6}",
                        "lyric_family": f"{prefix}-lyric{i // 6}", "seed": i % 6,
                        "audio_hashes": [f"{prefix}-audio{i}"], "historical_folder_label": "FAIL" if good else "PASS",
                        "features": {key: float(good + .001*i) for key in FEATURE_NAMES},
                        "frozen_gates": [], "frozen_verdict": "PASS"})
    return {"manifest_id": prefix, "measurement_id": "v1", "records": records}


def trained():
    dataset = data()
    labels = {record["id"]: i % 2 for i, record in enumerate(dataset["records"])}
    audit = {"manifest_id": "train", "listener_checks_pass": True, "preferences": [], "labels_source_sha256": "test-only"}
    return M.fit(dataset, labels, audit, draws=12)


def test_fit_uses_human_labels_and_never_historical_verdicts():
    model = trained()
    p = M.predictions(model, data()["records"])[:, 0]
    assert p[1::2].min() > p[::2].max()
    assert model["status"] == "exploratory"
    with pytest.raises(ValueError, match="controls"):
        M.fit(data(), {}, {"manifest_id": "train", "listener_checks_pass": False})


def test_heldout_validation_rejects_shared_lyrics_even_when_ids_are_new():
    model = trained()
    new = data(prefix="fresh")
    new["records"][0]["lyric_family"] = "train-lyric0"
    result = M.validate(model, new, {r["id"]: i % 2 for i, r in enumerate(new["records"])},
                        {"manifest_id": "fresh"})
    assert result["status"] == "rejected_overlap"
    assert result["overlap"]["fresh0"] == ["lyric_family"]


def test_legacy_gan_family_alias_cannot_make_smoke_look_unseen():
    model = trained()
    model["seen"]["recipe_family"] = ["gan-bcap-repair/paired"]
    record = data(1, prefix="new")["records"][0]
    record["recipe_family"] = "gan-bcap-repair/smoke"
    assert M.overlap(record, model["seen"]) == ["recipe_family"]


def test_unvalidated_or_incomplete_data_never_returns_optimizer_score():
    model = trained()
    result = M.score(model, data(prefix="new"), draws=30)
    assert result["candidates"][0]["optimizer_score"] is None
    assert "model_not_validated" in result["candidates"][0]["reasons"]
    model["status"] = "validated"
    short = data(2, prefix="new")
    result = M.score(model, short, draws=30)["candidates"][0]
    assert result["optimizer_score"] is None
    assert "fewer_than_three_seeds_per_prompt" in result["reasons"]


def test_gate_failure_cannot_be_bought_off_with_probability():
    model = trained()
    model["status"] = "validated"  # Test fixture only; real CLI requires validation.
    dataset = data(prefix="new")
    # Keep features in the calibration domain and full, distinct seed coverage.
    before = M.score(model, dataset, draws=30)["candidates"][0]
    assert before["optimizer_score"] is not None and before["optimizer_score"] >= 0
    dataset["records"][0]["frozen_gates"] = ["U2_lyric_hold"]
    after = M.score(model, dataset, draws=30)["candidates"][0]
    assert after["optimizer_score"] < 0


def test_validation_cannot_certify_classifier_when_gates_reject_usable_audio():
    model = trained()
    dataset = data(36, prefix="fresh")
    labels = {r["id"]: i % 2 for i, r in enumerate(dataset["records"])}
    audit = {"manifest_id": "fresh", "listener_checks_pass": True, "labels_source_sha256": "fixture",
             "preferences": [{"a": f"fresh{2*i+1}", "b": f"fresh{2*i}", "choice": "a"} for i in range(10)]}
    assert M.validate(model, dataset, labels, audit)["passed"]
    for i, record in enumerate(dataset["records"]):
        if i % 2:
            record["frozen_gates"] = ["U2_lyric_hold"]
            record["frozen_verdict"] = "FAIL"
    report = M.validate(model, dataset, labels, audit)
    assert not report["passed"]
    assert not report["checks"]["usable_recall"]
    assert report["usable_accepted"] == 0
    assert report["raw_probability_brier"] < report["brier"]


def test_schema_and_measurement_version_fail_closed():
    model = trained()
    dataset = data(prefix="new")
    dataset["measurement_id"] = "changed"
    with pytest.raises(ValueError, match="protocol"):
        M.score(model, dataset)
    model["feature_names"] = list(reversed(FEATURE_NAMES))
    with pytest.raises(ValueError, match="schema"):
        M.predictions(model, data()["records"])


@pytest.fixture
def session(tmp_path):
    import soundfile as sf
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros((8000, 2)), 16000)
    record = {"id": "fixture", "folder": "test", "baseline": str(audio), "seed": 7,
              "lyrics": "[verse] / the little lights are on", "clips": [
                  {"path": str(audio), "scale": 1, "concept": "Female"}]}
    # Two distinct record IDs provide the two repeat checks in the real design.
    manifest = {"id": "test-manifest", "records": [record, {**record, "id": "fixture2"}]}
    build_session(manifest, tmp_path / "listening")
    return tmp_path / "listening"


def response_for(trial, session_id, value="yes"):
    if trial["kind"] == "preference":
        return {"session_id": session_id, "trial_id": trial["id"],
                "played": list({trial["original"], trial["a"], trial["b"]}), "preference": "a"}
    return {"session_id": session_id, "trial_id": trial["id"],
            "played": list({trial["original"], *[clip["url"] for clip in trial["settings"]]}),
            "ratings": {key: value for key in RATINGS}}


def test_blinding_and_response_audit(session):
    public_text = (session / "public/session.json").read_text()
    assert "record_id" not in public_text and "noop_control" not in public_text and "fixture" not in public_text
    public = json.loads(public_text)
    key = json.loads((session / "session-key.json").read_text())
    rows = []
    for trial in public["trials"]:
        value = "no" if key["trials"][trial["id"]]["kind"].endswith("control") else "yes"
        rows.append(response_for(trial, public["id"], value))
    path = session / "test-responses.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    labels, audit = read_labels(session, path)
    assert labels == {"fixture": 1, "fixture2": 1}
    assert audit["listener_checks_pass"]
    rows[0]["session_id"] = "another-session"
    with pytest.raises(ValueError, match="different session"):
        validate_response(rows[0], key)


def test_server_persists_answers_and_does_not_serve_private_key(session):
    server = make_server(session, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        for suffix in ["/session-key.json", "/../session-key.json", "/%2e%2e/session-key.json"]:
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(root + suffix)
            assert exc.value.code == 404
        public = json.load(urllib.request.urlopen(root + "/session.json"))
        response = response_for(public["trials"][0], public["id"])
        request = urllib.request.Request(root + "/api/response", json.dumps(response).encode(),
                                         {"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(request))["saved"]
        assert json.load(urllib.request.urlopen(root + "/api/responses"))[0] == response
        assert (session / "responses.jsonl").read_text().strip() == json.dumps(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_completed_session_automatically_fits_only_an_exploratory_model(tmp_path):
    import soundfile as sf
    path = tmp_path / "synthetic-test.wav"
    sf.write(path, np.zeros((1000, 2)), 16000)
    dataset = data(12)
    records = [{**record, "folder": "test-only", "baseline": str(path),
                "lyrics": "a small light", "clips": [{"path": str(path), "scale": 1, "concept": "Female"}]}
               for record in dataset["records"]]
    # Audio identities in the numerical fixture are deliberately synthetic.
    # They are omitted from the synthetic session, not used as real provenance.
    for record in records:
        record.pop("audio_hashes")
    session_dir = tmp_path / "session"
    public = build_session({"id": "train", "records": records}, session_dir, max_preferences=0)
    features = tmp_path / "features.json"
    write_json(features, dataset)
    key = json.loads((session_dir / "session-key.json").read_text())
    server = make_server(session_dir, "127.0.0.1", 0, features)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for trial in public["trials"]:
            info = key["trials"][trial["id"]]
            value = "yes" if int(info["record_id"].removeprefix("train")) % 2 else "no"
            if info["kind"].endswith("control"):
                value = "no"
            response = response_for(trial, public["id"], value)
            request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/response",
                                             json.dumps(response).encode(), {"Content-Type": "application/json"})
            assert json.load(urllib.request.urlopen(request))["saved"]
        fitted = json.loads((session_dir / "model.json").read_text())
        assert fitted["status"] == "exploratory"
        assert fitted["coverage"]["ladders"] == 12
        preview = json.loads((session_dir / "score-preview.json").read_text())
        assert all(item["optimizer_score"] is None for item in preview["candidates"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
