"""Idempotent matched YuE2 checkpoint comparisons on the designated GPU."""
import argparse
import json

from common import *
import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from conceptmod.textsliders.infer_yue2 import render
from conceptmod.textsliders.yue2_backend import YuE2Slider
from yue2 import YuE2Pipeline
from yue2.storage import verify_result


def integrity(label):
    path=weights(label)
    values=load_file(str(path))
    state=torch.load(state_path(label),map_location="cpu",weights_only=True,mmap=True)
    assert values.keys()==state["network"].keys()
    assert all(torch.isfinite(v).all() and torch.equal(v,state["network"][k]) for k,v in values.items())
    from app.rewriter import _artist_name_hit
    metadata=read(path.with_suffix(".json"))
    assert not _artist_name_hit("",json.dumps(metadata))
    assert metadata["rank"]==8 and metadata["alpha"]==8 and metadata["step"]==state["completed"]
    result=dict(weights=str(path),weights_sha256=sha(path),state_sha256=sha(state_path(label)),
        export_matches_full_state=True,finite=True,step=state["completed"],model_identity=metadata["model_identity"])
    write(WORK/"integrity"/f"{label}.json",result)
    return result


def main(labels):
    spec=verify_protocol()
    sys.path.insert(0,str(ROOT.parent))
    checks={label:integrity(label) for label in labels}
    rows=yaml.safe_load(EVAL.read_text())["rows"]
    takes=["off","caption",*labels]
    with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B",vae="m-a-p/YuE2-Vae",local_files_only=True,
            device="cuda:0",backend="torch-eager",memory_budget_gib=18,quantization="none",offload_ar=False) as pipe:
        network,record=YuE2Slider.load(pipe._load_model(),weights(labels[0]))
        if record["model_identity"]!=pipe.weights["mot"]: raise ValueError("Base model differs from training")
        if any(check["model_identity"]!=pipe.weights["mot"] for check in checks.values()):
            raise ValueError("Candidates were trained on different base weights")
        for row,seed in CASES:
            for label in takes:
                dest=PAGE/f"row-{row}-seed-{seed}"/label
                request=dict(protocol_sha256=sha(WORK/"protocol.json"),row=row,seed=seed,label=label,
                    checkpoint_sha256=checks[label]["weights_sha256"] if label in checks else None,
                    scale=1 if label in checks else 0,max_tokens=spec["max_tokens"],
                    style=rows[row]["positive" if label=="caption" else "neutral"],lyrics=rows[row]["lyrics"])
                existing=read(dest/"render.json")
                if existing:
                    if existing["request"]!=request or sha(dest/"song.wav")!=existing["wav_sha256"]:
                        raise ValueError("Existing render differs from pinned request")
                    verify_result(dest)
                    continue
                if dest.exists(): raise FileExistsError(f"Incomplete final render directory: {dest}")
                staging=dest.with_name(dest.name+".incomplete")
                if staging.exists():
                    staging.rename(staging.with_name(staging.name+f".{time.time_ns()}"))
                if label in checks:
                    network.load_state_dict(load_file(str(weights(label))),strict=True)
                result=render(pipe,network,style=request["style"],lyrics=request["lyrics"],
                    scale=request["scale"],seed=seed,adapter_identity=request["checkpoint_sha256"],
                    semantic_sampling={"min_tokens":200,"max_tokens":spec["max_tokens"]})
                audio=result.audio
                if audio.ndim!=2 or audio.shape[1]!=2 or not audio.size or not np.isfinite(audio).all():
                    raise ValueError("Invalid stereo audio")
                staging.mkdir(parents=True)
                result.save(staging/"song.wav")
                result.save_artifacts(staging)
                meta=dict(request=request,wav_sha256=sha(staging/"song.wav"),duration=len(audio)/48000,
                    rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
                    clipped_fraction=float(np.mean(np.abs(audio)>=.999)),truncated=result.truncated)
                write(staging/"render.json",meta)
                staging.rename(dest)
                print(json.dumps(dict(row=row,seed=seed,label=label,duration=meta["duration"],rms=meta["rms"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--labels",nargs="+",required=True)
    main(p.parse_args().labels)
