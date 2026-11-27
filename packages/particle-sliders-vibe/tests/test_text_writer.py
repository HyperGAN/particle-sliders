import pytest

from particle_sliders_vibe import read_json, text_writer as writer, write_json


def reply(settings, inputs):
    return "served-test-model", {
        "scenes": [
            {
                "id": anchor["id"],
                "neutral": "Visit " + anchor["topic"],
                "treatments": [
                    {"positive": f"Use style {i}", "explicit": f"Intensify style {i}"}
                    for i in range(3)
                ],
            }
            for anchor in reversed(inputs["scenes"])
        ]
    }


def test_partial_setup_reuses_exact_batches_and_requested_model(tmp_path, monkeypatch):
    anchors = [{"id": str(i), "topic": f"topic {i}"} for i in range(4)]

    def interrupted(settings, inputs):
        if inputs["scenes"] == anchors[3:]:
            raise RuntimeError("interrupted")
        return reply(settings, inputs)

    monkeypatch.setenv("INKLING_PROMPT_MODEL", "first-model")
    args = tmp_path, "couplets", "Write an invitation", anchors
    with pytest.raises(RuntimeError, match="interrupted"):
        writer.write_plan(*args, progress=lambda *_: None, ask=interrupted)
    monkeypatch.setenv("INKLING_PROMPT_MODEL", "changed")

    def resumed(settings, inputs):
        assert settings["model"] == "first-model"
        assert inputs["scenes"] == anchors[3:]
        return reply(settings, inputs)

    plan = writer.write_plan(*args, progress=lambda *_: None, ask=resumed)
    assert list(plan) == ["0", "1", "2", "3"]
    assert plan["3"]["neutral"] == "Visit topic 3"

    def offline(*args):
        pytest.fail("unexpected writer request")

    assert writer.write_plan(*args, progress=lambda *_: None, ask=offline) == plan
    with pytest.raises(ValueError, match="different vibe"):
        writer.write_plan(
            tmp_path, "different", *args[2:], progress=lambda *_: None, ask=offline
        )
    folder = tmp_path / "prompt-rewrites"
    receipt = next(p for p in folder.glob("*.json") if p.name != "writer.json")
    batch = read_json(receipt)
    batch["scenes"][0]["neutral"] = "unexpected edit"
    write_json(receipt, batch)
    with pytest.raises(ValueError, match="changed"):
        writer.write_plan(*args, progress=lambda *_: None, ask=offline)


def test_invalid_writer_output_cannot_fall_back_to_templates(tmp_path, monkeypatch):
    calls = []

    def duplicate(settings, inputs):
        calls.append(inputs)
        served, result = reply(settings, inputs)
        candidates = result["scenes"][0]["treatments"]
        candidates[1]["positive"] = candidates[0]["positive"]
        return served, result

    monkeypatch.setattr(writer.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="Duplicate"):
        writer.write_plan(
            tmp_path,
            "short rhymes",
            "Welcome a visitor",
            [{"id": "0", "topic": "a library"}],
            progress=lambda *_: None,
            ask=duplicate,
        )
    assert len(calls) == 3
    assert list((tmp_path / "prompt-rewrites").glob("*.json")) == [
        tmp_path / "prompt-rewrites/writer.json"
    ]


def test_gemini_request_field_is_saved_as_the_neutral_prompt():
    anchors = [{"id": "0", "topic": "a library"}]
    _, result = reply({}, {"scenes": anchors})
    scene = result["scenes"][0]
    scene["request"] = scene.pop("neutral")
    assert writer.validate(result, anchors)[0]["neutral"] == scene["request"]
    scene["neutral"] = ""
    with pytest.raises(ValueError, match="Invalid neutral"):
        writer.validate(result, anchors)


def test_provider_json_with_markdown_fence_is_validated_normally(monkeypatch):
    import io
    import json

    inputs = {"scenes": [{"id": "0", "topic": "a library"}]}
    _, result = reply({}, inputs)
    completion = {
        "model": "served",
        "choices": [
            {
                "message": {
                    "content": "```json\n" + json.dumps(result) + "\n```",
                }
            }
        ],
    }
    monkeypatch.setenv("NANOGPT_API_KEY", "fixture")
    monkeypatch.setattr(
        writer, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(completion).encode())
    )
    served, parsed = writer.request_batch(
        {"instructions": writer.BRIEF, "model": "requested"}, inputs
    )
    assert served == "served"
    assert writer.validate(parsed, inputs["scenes"]) == result["scenes"]
