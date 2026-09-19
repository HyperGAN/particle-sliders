"""Presentation-only progress publisher, independent of GPU workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from common import WORK,ROOT,MODELS,PAGE,read,write
from budget import load_budget, milestones


def last_metric(path):
    if not path.exists():
        return {}
    with path.open("rb") as f:
        f.seek(max(0,path.stat().st_size-16384))
        lines=f.read().decode("utf-8",errors="replace").splitlines()
    for line in reversed(lines):
        try:
            value=json.loads(line)
            if isinstance(value,dict) and "step" in value:
                return value
        except ValueError:
            continue
    return {}


def publish():
    PAGE.mkdir(parents=True,exist_ok=True)
    page=(WORK / "page.html").read_text()
    if not (PAGE / "index.html").exists() or (PAGE / "index.html").read_text()!=page:
        (PAGE / "index.html").write_text(page)
    manifest=read(WORK / "manifest.json")
    catalog=read(WORK / "catalog.json")["sliders"]
    budget=load_budget()
    jobs=[]
    for item in catalog:
        job=read(WORK / "jobs" / f"{item['id']}.json",dict(id=item["id"],gpu=item["gpu"],status="queued",completed=0,phase="Queued"))
        job.update(label=item["label"],description=item["description"],target=budget["targets"][item["id"]])
        job["milestones"]=milestones(job["target"])
        run=MODELS / item["id"] / f"{item['id']}-fresh3400"
        current=read(run / "status.json",{})
        progress=read(run / "progress.json",{})
        warm=job.get("warmup",{})
        if warm.get("directory"):
            w=Path(warm["directory"])
            metric=last_metric(w / f"{w.name}_train.jsonl")
            job["completed"]=max(job.get("completed",0),metric.get("step",0))
            job["warmup_loss"]=metric.get("loss")
        job["completed"]=max(job.get("completed",0),current.get("completed",0),progress.get("step",0))
        job["last_update_time"]=max((p.stat().st_mtime for p in (run / "progress.json",run / "status.json") if p.exists()),default=job.get("updated"))
        job["losses"]=progress.get("losses")
        job["effective_weight_norm"]=progress.get("effective_weight_norm_before")
        job["active_row"]=current.get("row",progress.get("prompt_row"))
        job["row_counts"]=read(run / "checkpoint-audit.json",{}).get("row_counts",{})
        job["samples"]={}
        for step in manifest["evaluation"]["steps"]:
            report=read(WORK / "samples" / f"{item['id']}-step{step}.json")
            if report:
                job["samples"][str(step)]=dict(flagged=report["flagged"],catastrophic=report["catastrophic"],
                                                count=len(report["candidates"]))
        cases=[]
        for row in manifest["evaluation"]["rows"]:
            for seed in manifest["evaluation"]["seeds"]:
                takes=[]
                for key,label in [("off","Off"),("caption","Style caption"),("published660","Released 660"),
                    ("step600","New 600"),("step1000","New 1000"),("step2000","New 2000"),
                    ("step3000","New 3000"),("step3400","New 3400")]:
                    base=PAGE / item["id"] / f"row-{row}-seed-{seed}" / key
                    meta=read(base.with_suffix(".json"))
                    if meta and base.with_suffix(".mp3").exists():
                        takes.append(dict(label=label,wav=str(base.with_suffix(".wav").relative_to(PAGE)),
                            mp3=str(base.with_suffix(".mp3").relative_to(PAGE)),duration=meta["duration"],
                            rms=meta["rms"],flags=meta["flags"]))
                if takes:
                    cases.append(dict(row=row,seed=seed,takes=takes))
        job["cases"]=cases
        # The public page needs progress, not process arguments or local source paths.
        for key in ("current_log","warmup","checkpoint_audit"):
            job.pop(key,None)
        jobs.append(job)
    write(PAGE / "status.json",dict(updated=time.time(),started=manifest["created"],sliders=jobs,
        complete=sum(j["status"]=="complete" for j in jobs),total=16,
        completed_updates=sum(j["completed"] for j in jobs),total_updates=sum(j["target"] for j in jobs),
        budget_updated=budget["updated"],
        mode="Eight sliders completed at 3400; the remaining eight stop at 2000 for listening. All four prompt pairs, both GPUs, full states and milestone samples retained."))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--once",action="store_true");args=p.parse_args()
    while True:
        try:publish()
        except Exception as e:print(f"Page refresh failed: {e}",flush=True)
        if args.once:break
        time.sleep(5)
