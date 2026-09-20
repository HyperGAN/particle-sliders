"""Resumable CPU shards: both teachers at every state on both frozen paths."""
from collections import OrderedDict
from pathlib import Path
import json
import time

import torch

from .contracts import atomic_json, digest, file_hash
from .dataset import validate_manifest


def save_tensor_file(path, values):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(values, temp)
    temp.replace(path)


@torch.no_grad()
def prepare_targets(runtime, manifest, variation, directory, *, resolution=512, progress=None):
    validate_manifest(manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(model=runtime.identity, manifest_sha256=manifest["sha256"], split=manifest["split"],
                    variation=variation, resolution=resolution, steps=10, cfg=1,
                    layout="both-trajectories/all-positions/full-velocity/shared-state-v1")
    fingerprint = digest(identity)
    identity_path = directory / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Cache settings or provenance changed")
    atomic_json(identity_path, identity)
    rows = [r for r in manifest["rows"] if r["variation"] == variation]
    neutral_dir = directory.parent.parent / "neutral-cache"
    neutral_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    start = time.monotonic()
    with runtime.mixer.scales({}):
        for row in rows:
            neutral, positive = runtime.encode(row["neutral"]), runtime.encode(row["positive"])
            for seed in row["seeds"]:
                name = f"{row['id']}-{seed}.pt"
                path = directory / name
                if path.exists():
                    data = torch.load(path, map_location="cpu", weights_only=True)
                    if data["fingerprint"] != fingerprint or data["row"] != row or len(data["records"]) != 20:
                        raise ValueError(f"Invalid cache shard {name}")
                else:
                    records = []
                    initial = runtime.noise(seed, resolution, resolution)
                    neutral_key = digest(dict(model=runtime.identity, prompt=row["neutral"], seed=seed,
                                              resolution=resolution, steps=10))
                    neutral_path = neutral_dir / f"{neutral_key}.pt"
                    cached_neutral = (torch.load(neutral_path, map_location="cpu", weights_only=True)
                                      if neutral_path.exists() else None)
                    neutral_records = []
                    first_pair = None
                    for trajectory in ("neutral", "positive"):
                        latent = initial.clone()
                        sched = runtime.scheduler(10)
                        for index, timestep in enumerate(sched.timesteps):
                            if trajectory == "neutral" and cached_neutral is not None:
                                state = cached_neutral[index]
                                if state["timestep"] != float(timestep):
                                    raise ValueError("Neutral cache scheduler changed")
                                latent = state["latent"].to(runtime.device)
                                vn = state["neutral"].to(runtime.device)
                                vp = runtime.predict(latent, timestep, positive)
                            elif trajectory == "positive" and index == 0:
                                vn, vp = first_pair
                            else:
                                vn = runtime.predict(latent, timestep, neutral)
                                vp = runtime.predict(latent, timestep, positive)
                            if trajectory == "neutral":
                                neutral_records.append(dict(timestep=float(timestep), latent=latent.cpu().clone(), neutral=vn.cpu()))
                                if index == 0:
                                    first_pair = vn, vp
                            records.append(dict(position=index, timestep=float(timestep), trajectory=trajectory,
                                latent=latent.cpu().clone(), neutral=vn.cpu(), positive=vp.cpu()))
                            latent = runtime.step(sched, vn if trajectory == "neutral" else vp, timestep, latent)
                    if cached_neutral is None:
                        save_tensor_file(neutral_path, neutral_records)
                    save_tensor_file(path, dict(fingerprint=fingerprint, row=row, seed=seed,
                        embedding=neutral.cpu(), positive_embedding=positive.cpu(), records=records))
                shards.append(dict(path=name, sha256=file_hash(path), count=20, row=row["id"], seed=seed))
                if progress:
                    progress(len(shards), sum(len(r["seeds"]) for r in rows))
    index = dict(identity=identity, input_fingerprint=fingerprint,
                 fingerprint=digest(dict(input=fingerprint, shards=shards)), shards=shards,
                 elapsed_seconds=time.monotonic() - start,
                 storage_bytes=sum((directory / s["path"]).stat().st_size for s in shards))
    atomic_json(directory / "index.json", index)
    return index


class TargetCache:
    def __init__(self, directory, *, verify=True, max_shards=4, pin_memory=False):
        self.directory = Path(directory)
        self.index = json.loads((self.directory / "index.json").read_text())
        self.shards = OrderedDict()
        self.max_shards = max_shards
        self.pin_memory = pin_memory
        self.offsets = [(s, i) for s in self.index["shards"] for i in range(s["count"])]
        if verify:
            for s in self.index["shards"]:
                if file_hash(self.directory / s["path"]) != s["sha256"]:
                    raise ValueError(f"Corrupt target shard: {s['path']}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        shard, position = self.offsets[index]
        key = shard["path"]
        if key not in self.shards:
            data = torch.load(self.directory / key, map_location="cpu", weights_only=True)
            if data["fingerprint"] != self.index.get("input_fingerprint", self.index["fingerprint"]):
                raise ValueError("Shard identity differs from cache")
            if self.pin_memory:
                data["embedding"] = data["embedding"].pin_memory()
                for rec in data["records"]:
                    for name in ("latent", "neutral", "positive"):
                        rec[name] = rec[name].pin_memory()
            self.shards[key] = data
            if len(self.shards) > self.max_shards:
                self.shards.popitem(last=False)
        self.shards.move_to_end(key)
        data = self.shards[key]
        return {**data["records"][position], "embedding": data["embedding"], "row": data["row"], "seed": data["seed"]}


def fit_normalization(cache):
    if cache.index["identity"]["split"] != "train":
        raise ValueError("Normalization may only use training examples")
    scales, rms = [], []
    # Two sequential shard passes, rather than rereading all shards twenty times.
    moments = {}
    for index in range(len(cache)):
        rec = cache[index]
        position = rec["position"]
        if position != index % 10:
            raise ValueError("Cache timesteps were reordered")
        edit = (rec["positive"].float() - rec["neutral"].float()).flatten()
        if position not in moments:
            moments[position] = [0, torch.zeros_like(edit), torch.zeros_like(edit)]
        state = moments[position]
        state[0] += 1
        delta = edit - state[1]
        state[1] += delta / state[0]
        state[2] += delta * (edit - state[1])
    raw_scales = []
    for position in range(10):
        count, mean, m2 = moments[position]
        if count < 2:
            raise ValueError("Each timestep needs at least two training samples")
        raw_scales.append((m2 / (count - 1)).clamp_min(0).sqrt().clamp_min(1e-4))
    row_values = [[] for _ in range(10)]
    for index in range(len(cache)):
        rec = cache[index]
        position = rec["position"]
        normalized = (rec["positive"].float() - rec["neutral"].float()).flatten() / raw_scales[position]
        row_values[position].append(normalized.square().mean().sqrt())
    for position in range(10):
        scale = raw_scales[position]
        row_rms = torch.stack(row_values[position])
        median = row_rms.median().clamp_min(1e-4)
        scales.append(scale * median)
        rms.append((row_rms / median).square().mean().sqrt())
    result = dict(scale=torch.stack(scales), edit_rms=torch.stack(rms),
        provenance=dict(method="paired_edit_per_coordinate_sample_std_median_rms_gain_per_timestep",
                        cache_fingerprint=cache.index["fingerprint"], split="train", positions=10))
    result["provenance"]["scale_sha256"] = __import__('hashlib').sha256(result["scale"].numpy().tobytes()).hexdigest()
    return result
