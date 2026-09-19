"""Audit the 600-step export against the full saved game state and render files."""
from pathlib import Path
import hashlib
import json
import math
import sys

import torch
from safetensors.torch import load_file

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0, str(ROOT.parent))
from app.rewriter import _artist_name_hit


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    run = ROOT / "models/female-yue2-uni16-600-20260916"
    weights = run / "female-yue2-uni16-600_600.safetensors"
    state_path = run / "female-yue2-uni16-600_state.pt"
    exported = load_file(str(weights))
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    assert state["completed"] == 600
    assert exported.keys() == state["network"].keys()
    assert all(torch.isfinite(v).all() and torch.equal(v, state["network"][k]) for k, v in exported.items())
    metadata = json.loads(weights.with_suffix(".json").read_text())
    assert metadata["step"] == 600 and metadata["rank"] == 8 and metadata["alpha"] == 8
    assert metadata["recipe"] == "uni16-rpgan-bcap-warmup"
    assert not _artist_name_hit("", json.dumps(metadata))
    logs = [json.loads(line) for line in (run / "female-yue2-uni16-600_train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in logs] == list(range(1, 601))
    assert all(math.isfinite(v) for r in logs for v in r.values() if isinstance(v, (float, int)))
    pinned = json.loads((WORK / "source-sha256.json").read_text())
    assert all(sha(ROOT / name) == expected for name, expected in pinned.items())
    report = dict(completed_steps=600, checkpoint=str(weights), checkpoint_sha256=sha(weights),
        full_state_sha256=sha(state_path), tensor_count=len(exported),
        tensors_finite=True, export_matches_full_state=True, source_hashes_match=True,
        artist_name_validation_passed=True, all_training_metrics_finite=True,
        final_metrics=logs[-1], mean_last50={k: sum(r[k] for r in logs[-50:])/50
            for k in ("loss", "g_adv", "fm", "end", "cos_pos", "mag_ratio")})
    listen = ROOT / "eval/listen/yue2-female-uni16-600-20260916"
    if (listen / "evaluation.json").exists():
        evaluation = json.loads((listen / "evaluation.json").read_text())
        assert evaluation["checkpoint_sha256"] == report["checkpoint_sha256"]
        assert len(evaluation["records"]) == 10 and len(evaluation["comparisons"]) == 4
        for row in evaluation["records"]:
            folder = listen / row["path"]
            assert sha(folder / "song.wav") == row["wav_sha256"]
            result = json.loads((folder / "result.json").read_text())
            for name, expected in result["artifacts"].items():
                assert sha(folder / name) == expected["sha256"]
                assert (folder / name).stat().st_size == expected["bytes"]
            assert row["sample_rate"] == 48000 and row["rms"] > 1e-8
        report["rendered_clips"] = 10
        report["render_manifests_verified"] = True
        report["comparisons"] = evaluation["comparisons"]
    (WORK / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
