from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from particle_sliders_vibe import digest, write_json
from particle_sliders_vibe import image_workflow as flow


def test_continues_after_two_rounds_and_skips_a_broken_first_choice(tmp_path, monkeypatch):
    model = tmp_path / "u2net"
    model.write_bytes(b"model")
    monkeypatch.setenv("U2NET_MODEL",str(model))
    monkeypatch.setattr(flow,"MATTE_SHA",flow.file_digest(model))
    qualified = {str(i):{"seed":i} for i in range(15)}
    write_json(tmp_path/"teacher-progress.json",{"round":2,"kept":qualified,"attempts":[]})
    scene = {"id":"15","neutral":"street","framing":"portrait"}
    calls = []
    def candidates(s, attempt):
        calls.append((s["id"],attempt))
        return {"scene":s["id"], "seed":42+104729*attempt, "attempt":attempt,
                "neutral":{"sha256":"off"},"options":[]}
    def option(image,score):
        return dict(positive=image,explicit=image+" outfit",image={"sha256":image},
                    reading=dict(case=score,separate=0,quality=1,broken=0,off_broken=0))
    options = [option("broken",.9),option("usable",.8),option("weak",.1)]
    for item in options:
        item["image"]["path"] = str(tmp_path / (item["positive"] + ".png"))
    monkeypatch.setattr(flow,"read_candidates",lambda *args:{"options":options})
    monkeypatch.setattr(flow,"raw_image",lambda ref:ref["sha256"])
    monkeypatch.setattr(flow,"check",lambda reader,image:{"status":"fail" if image=="broken" else "pass"})
    def matte(cmd,**kwargs):
        Path(cmd[-1]).write_bytes(b"matte")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(flow.subprocess,"run",matte)
    def render_candidates(s,attempt):
        data = candidates(s,attempt)
        data["neutral"]["path"] = str(tmp_path / "starting.png")
        data["options"] = options
        return data
    run = SimpleNamespace(root=tmp_path,vibe="pirate",grid={"teaching":[*({"id":k} for k in qualified),scene]},
                          candidates=render_candidates)
    result = flow.collect(run)
    assert calls == [("15",2)]
    assert len(result["kept"]) == 16 and result["kept"]["15"]["positive"] == "usable"
    assert result["attempts"][0]["structure"]["broken"] == "fail"
    assert result["attempts"][0]["structure"]["weak"] == "pass"
    gallery = (tmp_path / "teachers.html").read_text()
    for label in ("Starting image", "Candidate 1 · Broken image", "Candidate 2 · Selected", "Candidate 3 · Skipped"):
        assert label in gallery
    assert digest(flow.collect(run)) == digest(result)
    assert len(calls) == 1
