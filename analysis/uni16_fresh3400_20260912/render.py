"""First-draw milestone audio, fixed controls and basic health diagnostics."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from common import WORK, ROOT, RUNTIME, PAGE, read, write, sha, digest, verify
sys.path[:0] = [str(RUNTIME), str(ROOT.parent)]


def inspect(path, requested):
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("Empty or non-finite rendered audio")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    duration = len(audio)/sr
    clipped = float(np.mean(np.abs(audio) >= .999))
    flags = []
    if duration < requested*.95:
        flags.append("short")
    if rms < .0001:
        flags.append("near_silent")
    if clipped >= .01:
        flags.append("clipping")
    return dict(duration=duration, rms=rms, peak=float(np.max(np.abs(audio))),
                clipped_fraction=clipped, flags=flags, sha256=sha(path))


def catastrophic(records):
    return any(set(r["flags"]) & {"near_silent", "clipping"} for r in records) or (
        bool(records) and all(r["duration"] < 4. for r in records))


def main(args):
    manifest = verify()
    item = next(i for i in read(WORK / "catalog.json")["sliders"] if i["id"]==args.id)
    import torch
    import yaml
    from safetensors.torch import load_file
    from conceptmod.textsliders import generate_listen as G
    from conceptmod.textsliders.lora import LoRANetwork
    torch.set_num_threads(4)
    fixtures = yaml.safe_load(Path(item["eval_prompts"]).read_text())["rows"]
    configs = [("off",None,0.),("caption",None,0.),("published660",Path(item["published_weights"]),1.),
               ("step600",args.warmup_weights,1.),(f"step{args.step}",args.weights,1.)]
    assert sha(item["published_weights"]) == item["published_sha256"]
    jobs = []
    for row in manifest["evaluation"]["rows"]:
        for seed in manifest["evaluation"]["seeds"]:
            for label,weights,scale in configs:
                dest = PAGE / item["id"] / f"row-{row}-seed-{seed}" / f"{label}.wav"
                prompt = fixtures[row]["positive" if label=="caption" else "neutral"]
                spec = dict(prompt_sha256=digest(prompt),lyrics_sha256=digest(fixtures[row]["lyrics"]),
                    seed=seed,scale=scale,weights_sha256=sha(weights) if weights else None,
                    duration=manifest["evaluation"]["duration"],campaign_sha256=sha(WORK / "manifest.json"))
                jobs.append(dict(row=row,seed=seed,label=label,weights=weights,scale=scale,dest=dest,
                                 prompt=prompt,spec=spec))
    pipe=network=None
    loaded=None
    records=[]
    for job in jobs:
        dest=job["dest"];dest.parent.mkdir(parents=True,exist_ok=True)
        meta=read(dest.with_suffix(".json"))
        if meta:
            assert meta["spec"]==job["spec"] and sha(dest)==meta["sha256"], "Rendered fixture changed"
        else:
            if pipe is None:
                pipe=G._load_pipeline(G.DEFAULT_MODEL_DIR,"cuda:0")
                network=LoRANetwork(pipe.language_model,multiplier=0.,rank=8,alpha=8,delimiter="-",
                    target_replace=["Qwen3Attention"],prefix="lora_te",train_method="full").to("cuda:0").eval()
                assert len(network.unet_loras)==144
            if job["weights"] and loaded!=str(job["weights"]):
                tensors=load_file(str(job["weights"]))
                assert all(torch.isfinite(t).all() for t in tensors.values())
                network.load_state_dict(tensors,strict=True)
                loaded=str(job["weights"])
            network.set_lora_slider(job["scale"])
            print(f"{item['id']} row={job['row']} seed={job['seed']} {job['label']}",flush=True)
            with torch.inference_mode(),network:
                audio=pipe(prompt=job["prompt"],lyrics=fixtures[job["row"]]["lyrics"],
                    audio_duration=job["spec"]["duration"],generator=torch.Generator("cuda:0").manual_seed(job["seed"]),
                    output="audios")[0]
            G._write_wav(dest,audio,int(pipe.sampling_rate),job["spec"]["duration"],accept_short=True,accept_silent=True)
            meta=dict(spec=job["spec"],**inspect(dest,job["spec"]["duration"]),physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]),
                      row=job["row"],seed=job["seed"],label=job["label"],rendered=time.time(),seed_retries=0)
            write(dest.with_suffix(".json"),meta)
        mp3=dest.with_suffix(".mp3")
        if not mp3.exists():
            temp=mp3.with_name(mp3.stem+".tmp.mp3")
            subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-i",str(dest),"-map_metadata","-1",
                            "-c:a","libmp3lame","-b:a","192k","-threads","1",str(temp)],check=True)
            temp.replace(mp3)
        records.append(dict(**meta,wav=str(dest.relative_to(PAGE)),mp3=str(mp3.relative_to(PAGE))))
    candidates=[r for r in records if r["label"]==f"step{args.step}"]
    assert len(candidates)==4
    report=dict(step=args.step,records=records,candidates=candidates,complete=True,
        catastrophic=catastrophic(candidates),flagged=sum(bool(r["flags"]) for r in candidates),
        description="Basic audio health only; musical quality, repetition and style require listening.")
    write(WORK / "samples" / f"{item['id']}-step{args.step}.json",report)
    write(PAGE / item["id"] / f"health-step{args.step}.json",report)
    print(f"Milestone {args.step}: {len(candidates)} clips; {report['flagged']} flagged; catastrophic={report['catastrophic']}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--id",required=True)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--warmup-weights",type=Path,required=True)
    main(p.parse_args())
