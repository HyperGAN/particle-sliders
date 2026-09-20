"""Compare streaming paired normalization to pinned ParticleGAN on real cached fields."""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/anima/correctness/normalization-reverification.json")
    args = parser.parse_args()
    import torch
    from torch import nn
    from lumen_studio.cache import fit_normalization
    from lumen_studio.contracts import atomic_json, digest, file_hash
    from lumen_studio.vendor.reference import REFERENCE, noise_std
    torch.set_num_threads(2)
    source_name = "conceptmod/textsliders/particle_bridge_gan.py"
    pinned = json.loads((ROOT / "lumen_studio/vendor/provenance.json").read_text())
    source = args.reference_root / source_name
    assert file_hash(source) == pinned["files"][source_name]
    names = {"register_paired_error_norm", "noise_std"}
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    scope = dict(torch=torch, REFERENCE=REFERENCE.copy())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    report = dict(passed=False, source_sha256=file_hash(source), caches=[])
    for path in args.cache:
        index = json.loads((path / "index.json").read_text())
        assert index["identity"]["split"] == "train"
        # Even coverage spans every character/definition row, both paths and all positions.
        selected = sorted({round(i * (len(index["shards"]) - 1) / 31) for i in range(32)})
        records = []
        for i in selected:
            shard = index["shards"][i]
            assert file_hash(path / shard["path"]) == shard["sha256"]
            data = torch.load(path / shard["path"], map_location="cpu", weights_only=True)
            records.extend({k: r[k] for k in ("position", "neutral", "positive")} for r in data["records"])

        class Fixture:
            def __init__(self):
                self.index = dict(identity=dict(split="train"), fingerprint=digest(selected))
            def __len__(self):
                return len(records)
            def __getitem__(self, i):
                return records[i]

        actual = fit_normalization(Fixture())
        checks = []
        for position in range(10):
            rows = [r for r in records if r["position"] == position]
            neutral = torch.stack([r["neutral"].float().flatten() for r in rows])
            positive = torch.stack([r["positive"].float().flatten() for r in rows])
            expected = scope["register_paired_error_norm"](nn.Module(), positive, neutral)
            scale, rms = actual["scale"][position], actual["edit_rms"][position]
            torch.testing.assert_close(scale, expected.target_std, rtol=2e-6, atol=1e-7)
            torch.testing.assert_close(rms, expected.edit_rms, rtol=2e-6, atol=1e-7)
            for step in (0, 199, 399, 799, 1199, 1599):
                start = float(rms) / .28
                assert noise_std(step, start=start, decay_steps=1600, hold=1.) == scope["noise_std"](
                    step, start=start, decay_steps=1600, hold=1.)
            checks.append(dict(position=position, rows=len(rows),
                scale_max_abs=float((scale - expected.target_std).abs().max()),
                scale_max_relative=float(((scale - expected.target_std).abs() / expected.target_std).max()),
                edit_rms_abs=float((rms - expected.edit_rms).abs())))
        report["caches"].append(dict(cache=index["fingerprint"], shards=selected, checks=checks))
        del records, actual, neutral, positive
    report["passed"] = True
    atomic_json(args.output, report)
    print(json.dumps(dict(passed=True, caches=len(report["caches"]),
        max_scale_relative=max(c["scale_max_relative"] for v in report["caches"] for c in v["checks"]))))


if __name__ == "__main__":
    main()
