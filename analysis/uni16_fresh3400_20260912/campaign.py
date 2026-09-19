"""Two persistent GPU queues, with samples and health checks every 1000 updates."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from common import WORK,ROOT,RUNTIME,MODELS,PAGE,PY,read,write,sha,verify,env,warmup_args

STOP=threading.Event()
CHILDREN={}
LOCK=threading.RLock()


def job_path(item):
    return WORK / "jobs" / f"{item['id']}.json"


def update(item, **values):
    with LOCK:
        status=read(job_path(item),dict(id=item["id"],gpu=item["gpu"],status="queued",completed=0))
        status.update(values,updated=time.time())
        write(job_path(item),status)
        return status


def command(item,argv,log,phase):
    verify()
    if shutil.disk_usage(ROOT).free < 15*1024**3:
        raise RuntimeError("Disk space below 15 GiB; completed files preserved")
    gpu=item["gpu"]
    gpu_lock=ROOT.parent / f".music-gpu-{gpu}.lock"
    with gpu_lock.open("a") as lease:
        while not STOP.is_set():
            try:
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                update(item,status="waiting_gpu",phase=phase)
                STOP.wait(5)
        if STOP.is_set():
            raise InterruptedError("Campaign stopping")
        while not STOP.is_set():
            memory=int(subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.used",
                                               "--format=csv,noheader,nounits"],text=True).strip())
            if memory<2048:
                break
            update(item,status="waiting_gpu",phase=phase,gpu_memory_mib=memory)
            STOP.wait(10)
        if STOP.is_set():
            raise InterruptedError("Campaign stopping")
        log.parent.mkdir(parents=True,exist_ok=True)
        write(log.with_suffix(".command.json"),dict(argv=list(map(str,argv)),gpu=gpu,started=time.time()))
        with log.open("a") as handle:
            child=subprocess.Popen(list(map(str,argv)),cwd=ROOT,env=env(gpu),stdout=handle,stderr=subprocess.STDOUT)
            with LOCK:
                CHILDREN[gpu]=child
            update(item,status="running",phase=phase,child_pid=child.pid,current_log=str(log))
            while child.poll() is None:
                STOP.wait(5)
                if STOP.is_set():
                    child.terminate()
                    try:child.wait(timeout=90)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
            with LOCK:
                CHILDREN.pop(gpu,None)
        update(item,child_pid=None)
        if STOP.is_set():
            raise InterruptedError("Campaign stopping; restart resumes saved states")
        if child.returncode:
            raise RuntimeError(f"{phase} exited {child.returncode}; retained log: {log.name}")


def audit_warmup(item,run):
    import torch
    from safetensors.torch import load_file
    state=run / f"{run.name}_state.pt"
    weights=run / f"{run.name}_last.safetensors"
    blob=torch.load(state,map_location="cpu",weights_only=True)
    assert blob["completed_updates"]==600 and blob["signature"]==item["warmup_signature"]
    tensors=load_file(str(weights))
    assert tensors.keys()==blob["modules"]["lora"].keys()
    assert all(torch.isfinite(t).all() and torch.equal(t,blob["modules"]["lora"][k]) for k,t in tensors.items())
    for group in blob["modules"].values():
        assert all(torch.isfinite(t).all() for t in group.values())
    canonical=MODELS / item["id"] / f"{item['id']}-warmup600.safetensors"
    if canonical.exists():
        assert sha(canonical)==sha(weights)
    else:
        os.link(weights,canonical)
    metadata=read(weights.with_suffix(".json"),{})
    metadata.update(steps=600,rank=8,alpha=8.,kind="language_model",target_replace=["Qwen3Attention"],
        prefix="lora_te",delimiter="-",train_method="full",unit_scale=1.,plus_label=item["label"],minus_label="Off",
        prompts_file=item["train_prompts"],recommended_range=[0.,1.])
    write(canonical.with_suffix(".json"),metadata)
    audit=dict(completed=600,finite=True,export_matches_state=True,weights_sha256=sha(canonical),state_sha256=sha(state))
    update(item,completed=600,warmup=dict(status="complete",directory=str(run),state=str(state),weights=str(canonical),audit=audit))
    return state,canonical


def warmup(item):
    status=read(job_path(item),{})
    stage=status.get("warmup",{})
    if stage.get("status")=="complete":
        assert sha(stage["weights"])==stage["audit"]["weights_sha256"]
        assert sha(stage["state"])==stage["audit"]["state_sha256"]
        return Path(stage["state"]),Path(stage["weights"])
    resume=None
    if stage.get("directory"):
        old=Path(stage["directory"])
        candidate=old / f"{old.name}_state.pt"
        if candidate.exists():
            import torch
            blob=torch.load(candidate,map_location="cpu",weights_only=True,mmap=True)
            if blob["completed_updates"]==600 and (old / f"{old.name}_last.safetensors").exists():
                return audit_warmup(item,old)
            resume=candidate
    attempt=stage.get("attempt",0)+1
    run=MODELS / item["id"] / f"{item['id']}-warmup-a{attempt:03d}"
    if run.exists():
        raise ValueError("Warmup attempt directory already exists")
    run.mkdir(parents=True)
    update(item,warmup=dict(status="running",directory=str(run),attempt=attempt))
    command(item,[PY,"-u",RUNTIME / "conceptmod/textsliders/train_lm_slider_music3.py",*warmup_args(item,run,resume)],
            WORK / "logs" / f"{run.name}.log","Warm-up to 600")
    return audit_warmup(item,run)


def execute(item):
    status=read(job_path(item),{})
    if status.get("status")=="complete":
        assert sha(status["weights"])==status["weights_sha256"]
        return
    source,initial=warmup(item)
    run=MODELS / item["id"] / f"{item['id']}-fresh3400"
    for step in (1000,2000,3000,3400):
        if STOP.is_set():
            raise InterruptedError("Campaign stopping")
        weights=run / f"{run.name}_step{step}.safetensors"
        audit=read(run / f"audit-step{step}.json")
        if not audit:
            command(item,[PY,"-u",WORK / "train.py","--id",item["id"],"--source-state",source,
                "--source-weights",initial,"--until",str(step)],WORK / "logs" / f"{item['id']}-to-{step}.log",
                f"Fresh continuations to {step}")
            audit=read(run / f"audit-step{step}.json")
        if not audit or audit["completed"]!=step or not audit["finite"] or not audit["export_matches_state"]:
            raise ValueError(f"Step {step} checkpoint audit did not pass")
        assert sha(weights)==audit["weights_sha256"]
        update(item,completed=step,checkpoint_audit=audit)
        sampled=read(WORK / "samples" / f"{item['id']}-step{step}.json")
        if not sampled or not sampled.get("complete"):
            command(item,[PY,"-u",WORK / "render.py","--id",item["id"],"--step",str(step),
                          "--weights",weights,"--warmup-weights",initial],
                    WORK / "logs" / f"{item['id']}-samples-{step}.log",f"Sampling and health check at {step}")
            sampled=read(WORK / "samples" / f"{item['id']}-step{step}.json")
        if sampled["catastrophic"]:
            update(item,status="needs_listening",phase=f"Audio health flag at {step}",
                   error="Near-silence, severe clipping or every clip under 4 seconds. Samples retained; review before continuing.")
            return
    update(item,status="complete",phase="3400 updates; all milestone samples retained",completed=3400,
           weights=str(weights),weights_sha256=sha(weights),error=None)


def lane(gpu,catalog):
    for item in catalog:
        if item["gpu"]!=gpu or STOP.is_set():
            continue
        try:
            execute(item)
        except InterruptedError as exc:
            update(item,status="paused",phase=str(exc),child_pid=None)
            return
        except Exception as exc:
            update(item,status="failed",phase="Stopped; state and logs retained",error=str(exc),child_pid=None)
            print(f"{item['id']}: {type(exc).__name__}: {exc}",flush=True)


def main():
    verify()
    catalog=read(WORK / "catalog.json")["sliders"]
    for item in catalog:
        if not job_path(item).exists():
            update(item,status="queued",phase="Queued",completed=0)
    def stop(*_):
        STOP.set()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    (WORK / "campaign.lock").touch(exist_ok=True)
    with (WORK / "campaign.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,gpu,catalog) for gpu in (0,1)]
            for future in futures:
                future.result()
    jobs=[read(job_path(i)) for i in catalog]
    write(WORK / "completion.json",dict(updated=time.time(),complete=all(j["status"]=="complete" for j in jobs),
        completed=[j["id"] for j in jobs if j["status"]=="complete"],
        needs_attention=[j["id"] for j in jobs if j["status"] in ("failed","needs_listening","paused")]))


if __name__=="__main__":
    main()
