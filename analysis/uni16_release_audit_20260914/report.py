"""Continuously publish ranked evidence without exposing unmeasured candidates."""
import csv
import io
import time
from common import *
from waveforms import audit as waveform_audit
from selection import select


def publish():
    PAGE.mkdir(parents=True, exist_ok=True)
    html=(WORK/"page.html").read_text()
    if not (PAGE/"index.html").exists() or (PAGE/"index.html").read_text()!=html:
        (PAGE/"index.html").write_text(html)
    protocol=read(WORK/"protocol.json")
    policy=read(WORK/"selection-policy.json")
    policy=dict(policy,source_sha256=sha(WORK/"selection.py"),policy_sha256=sha(WORK/"selection-policy.json"))
    budgets=read(CAMPAIGN/"budget.json")["targets"]
    sliders=[]
    for item in catalog():
        full=read(WORK/"sliders"/f"{item['id']}.json",{})
        job=read(CAMPAIGN/"jobs"/f"{item['id']}.json",{})
        available=[]
        for label in protocol["candidates"]:
            if all((AUDIO/item["id"]/f"row-{row}-seed-{seed}"/f"{label}.json").exists()
                   for row in protocol["rows"] for seed in protocol["seeds"]):available.append(label)
        measured={c["label"] for c in full.get("candidates",[])}
        ready=set(available)<=measured and bool(available)
        final=ready and job.get("status")=="complete" and f"step{budgets[item['id']]}" in measured
        waveform=waveform_audit(item,available) if available else None
        decision=select(full.get("candidates",[]),protocol,policy,waveform["flags"] if waveform else [])
        ranked=decision["ranking"]
        for entry in ranked:
            for clip in entry["clips"]:
                # Original waveforms stay on the campaign page; only measured
                # scalars, diagnostics and an explicit preview link are public.
                for key in ("candidate","baseline","positive_reference"):
                    clip[key].pop("audio",None)
        sliders.append(dict(id=item["id"],label=item["label"],target=budgets[item["id"]],
            training_status=job.get("status"),training_phase=job.get("phase"),available=available,
            measured=len(measured),all_current_scored=ready,all_budget_samples_scored=final,
            **decision,waveform_audit=waveform,
            updated=full.get("updated")))
    result=dict(updated=time.time(),workers=[read(WORK/f"worker-{i}.json",dict(status="starting")) for i in (0,1)],
        sliders=sliders,ranked=sum(bool(s["ranking"]) for s in sliders),complete=sum(s["all_budget_samples_scored"] for s in sliders),
        candidates_scored=sum(s["measured"] for s in sliders),measurement_protocol=protocol,
        selection_policy=policy,
        interpretation="Current recommendations use selection_policy. The frozen measurement_protocol retains the historical blend for provenance only. Quality and intended style still require listening; no checkpoint is certified free of subtle defects.")
    write(PAGE/"summary.json",result)
    write(WORK/"summary.json",result)
    output=io.StringIO();writer=csv.writer(output)
    writer.writerow(["slider","suggested_checkpoint","weights","sha256","close_call","all_budget_samples_scored","status",
        "quality_tolerance","enjoyment_shortfall","production_shortfall","tolerance_sensitive","selection_policy"])
    for s in sliders:
        q=s["recommendation"] or {}
        status="incomplete_scoring" if not s["all_current_scored"] else "provisional" if q else "review_required"
        deficits=q.get("quality_shortfall",{})
        writer.writerow([s["id"],q.get("label",""),q.get("weights",""),q.get("weights_sha256",""),q.get("close_call",""),s["all_budget_samples_scored"],status,
            policy["tolerance"],deficits.get("enjoyment",""),deficits.get("production",""),q.get("tolerance_sensitive",""),policy["version"]])
    tmp=PAGE/"recommendations.csv.tmp";tmp.write_text(output.getvalue());tmp.replace(PAGE/"recommendations.csv")


if __name__=="__main__":
    while True:
        try:publish()
        except Exception as exc:print(f"Report refresh failed: {exc}",flush=True)
        time.sleep(5)
