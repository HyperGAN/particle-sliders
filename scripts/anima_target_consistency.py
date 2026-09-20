"""Describe full-field teacher-edit directions on training data without updating a model."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from lumen_studio.cache import TargetCache
    from lumen_studio.contracts import VARIATIONS, atomic_json, file_hash
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/anima/remote")
    parser.add_argument("--variation", action="append", choices=VARIATIONS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/anima/diagnostics/target-consistency.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = dict(diagnostic_only=True, split="train", spatial_averaging=False,
        interpretation="Cosines compare average full-field edit vectors across definitions. "
                       "Different characters and states limit causal interpretation; this is not a quality gate.",
        variations={})
    started = time.monotonic()
    for variation in (args.variation or VARIATIONS):
        cache = TargetCache(args.root / "targets" / variation / "train")
        norm_path = args.root / "runs" / variation / "normalization.pt"
        norm = torch.load(norm_path, map_location="cpu", weights_only=True)
        assert norm["provenance"]["cache_fingerprint"] == cache.index["fingerprint"]
        buckets = {}
        for index in range(len(cache)):
            rec = cache[index]
            value = ((rec["positive"].float() - rec["neutral"].float()).flatten()
                     / norm["scale"][rec["position"]]).double()
            key = (rec["row"]["definition"], rec["position"])
            if key not in buckets:
                buckets[key] = [0, torch.zeros_like(value), 0.]
            state = buckets[key]
            state[0] += 1
            state[1] += value
            state[2] += float(value.square().sum())
        names = sorted({key[0] for key in buckets})
        means = torch.stack([torch.cat([buckets[(name, t)][1] / buckets[(name, t)][0]
                                       for t in range(10)]) for name in names])
        lengths = means.norm(dim=1)
        cosine = (means @ means.T) / (lengths[:, None] * lengths[None, :]).clamp_min(1e-30)
        records = []
        for name in names:
            for position in range(10):
                count, total, squares = buckets[(name, position)]
                rms = (squares / (count * total.numel())) ** .5
                mean_rms = float((total / count).square().mean().sqrt())
                records.append(dict(definition=name, position=position, count=count,
                    normalized_edit_rms=rms, mean_edit_rms=mean_rms,
                    mean_direction_coherence=mean_rms / rms if rms else 0.))
        value = dict(cache=cache.index["fingerprint"], normalization_sha256=file_hash(norm_path),
                     definitions=names, mean_edit_cosines=cosine.tolist(), by_position=records)
        report["variations"][variation] = value
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, report)
        off_diagonal = cosine[~torch.eye(len(names), dtype=torch.bool)]
        print(json.dumps(dict(variation=variation, minimum_mean_edit_cosine=float(off_diagonal.min()),
                              maximum_mean_edit_cosine=float(off_diagonal.max()))), flush=True)


if __name__ == "__main__":
    main()
