"""Durable train/render/rank orchestration; restart resumes the exact saved game."""
import fcntl
import signal
import subprocess

from common import *
import report

STOP=False
CHILD=None


def initial_alive():
    initial=read(WORK/"initial-process.json",{})
    pid=initial.get("pid")
    if not pid: return False
    try:
        command=Path(f"/proc/{pid}/cmdline").read_bytes()
        return b"train_lora_yue2_fresh.py" in command and str(RUN).encode() in command
    except FileNotFoundError:
        return False


def status(stage,**extra):
    write(WORK/"campaign.json",dict(stage=stage,updated=time.time(),pid=os.getpid(),**extra))
    report.main()


def command(argv,log_path,stage,gpu="1"):
    global CHILD
    if STOP: raise InterruptedError("Campaign paused")
    status(stage,command=[str(v) for v in argv])
    with Path(log_path).open("a") as log:
        CHILD=subprocess.Popen([str(v) for v in argv],cwd=ROOT,env=environment(gpu),
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        while CHILD.poll() is None:
            if STOP: CHILD.terminate()
            report.main()
            time.sleep(5)
        code=CHILD.returncode
        CHILD=None
    if STOP: raise InterruptedError("Campaign paused")
    if code: raise RuntimeError(f"{stage} exited {code}; see {log_path}")


def check_health(label):
    ranking=read(PAGE/"ranking.json")
    candidate=next(r for r in ranking["ranking"] if r["label"]==label)
    if {"near_silence","excess_clipping"}&set(candidate["flags"]):
        raise RuntimeError(f"Technical health stop at {label}: {candidate['flags']}")
    if all(c["candidate"]["duration"]<4 for c in candidate["clips"]):
        raise RuntimeError(f"All four {label} clips ended before four seconds")


def final_audit():
    import torch
    from safetensors.torch import load_file
    state=torch.load(RUN/"state.pt",map_location="cpu",weights_only=True,mmap=True)
    assert state["completed"]==3400
    histories=state["history"]
    assert [h["step"] for h in histories]==list(range(1,3401))
    assert [h["history"]["seed"] for h in histories]==list(range(3000001,3003401))
    assert len({h["history"]["tokens_sha256"] for h in histories})==3400
    counts={row:sum(h["row"]==row for h in histories) for row in range(4)}
    assert list(counts.values())==[850]*4
    for record in histories:
        blob=torch.load(RUN/"histories"/f"step{record['step']}.pt",map_location="cpu",weights_only=True)
        token_hash=hashlib.sha256(torch.tensor(blob["tokens"],dtype=torch.int64).numpy().tobytes()).hexdigest()
        assert token_hash==record["history"]["tokens_sha256"] and blob["row"]==record["row"]
        assert torch.isfinite(blob["end_teacher"]).all()
    for step in MILESTONES:
        pinned=torch.load(state_path(f"step{step}"),map_location="cpu",weights_only=True,mmap=True)
        tensors=load_file(str(weights(f"step{step}")))
        assert tensors.keys()==pinned["network"].keys()
        assert all(torch.isfinite(v).all() and torch.equal(v,pinned["network"][k]) for k,v in tensors.items())
    verify_protocol()
    ranking=read(PAGE/"ranking.json")
    assert ranking["complete"] and len(ranking["ranking"])==6
    result=dict(completed_steps=3400,unique_seeds=3400,unique_histories=3400,row_counts=counts,
        all_milestone_exports_match_full_state=True,history_files_verified=True,
        phase_changes=[],ranking_complete=True,ranking_sha256=sha(PAGE/"ranking.json"),
        final_state_sha256=sha(RUN/"state.pt"),selected=ranking["recommendation"])
    write(WORK/"completion-audit.json",result)
    write(PAGE/"completion-audit.json",result)
    return result


def main():
    global STOP
    WORK.mkdir(exist_ok=True)
    lock=(WORK/"campaign.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    prepare_protocol()
    try:
        while initial_alive():
            status("Training initial segment",initial_pid=read(WORK/"initial-process.json")["pid"])
            if STOP:
                os.kill(read(WORK/"initial-process.json")["pid"],signal.SIGTERM)
                raise InterruptedError("Campaign paused")
            time.sleep(5)
        if read(RUN/"status.json",{}).get("status")=="failed":
            raise RuntimeError("Initial training failed; inspect its log before resuming")
        # Validate the reference render/measurement path before later milestones.
        command([PY,"-u",WORK/"render.py","--labels","reference600"],WORK/"render-reference.log","Rendering original reference")
        command([PY,"-u",WORK/"measure.py"],WORK/"measure-reference.log","Scoring original reference",gpu="")
        for step in MILESTONES:
            verify_protocol()
            command([PY,"-u",ROOT/"conceptmod/textsliders/train_lora_yue2_fresh.py",
                "--save_dir",RUN,"--steps","3400","--until",str(step)],
                WORK/f"train-to-{step}.log",f"Training toward {step}")
            command([PY,"-u",WORK/"render.py","--labels",f"step{step}"],
                WORK/f"render-{step}.log",f"Rendering step {step}")
            command([PY,"-u",WORK/"measure.py"],WORK/f"measure-{step}.log",f"Ranking through step {step}",gpu="")
            check_health(f"step{step}")
        audit=final_audit()
        status("Complete",completion=audit)
    except InterruptedError as exc:
        status("Paused",message=str(exc))
    except BaseException as exc:
        status("Failed",error=f"{type(exc).__name__}: {exc}")
        raise


if __name__=="__main__": main()
