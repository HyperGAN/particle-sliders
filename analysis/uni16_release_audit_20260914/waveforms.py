"""Conservative near-identical waveform screen; not a musical-diversity test."""
import itertools
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from common import WORK, AUDIO, read, write, digest, sha


def correlation(a, b):
    if abs(len(a)-len(b)) > .02*max(len(a),len(b)):
        return None
    n=min(len(a),len(b));a=np.asarray(a[:n],dtype=np.float64);b=np.asarray(b[:n],dtype=np.float64)
    a=a-a.mean();b=b-b.mean()
    scale=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return None if scale < 1e-12 else float(min(1.,abs(np.sum(a*b))/scale))


def audit(item, labels):
    files=[]
    for label in ["off","caption",*labels]:
        for row in (2,3):
            for seed in (1709,2903):
                path=AUDIO/item["id"]/f"row-{row}-seed-{seed}"/f"{label}.wav"
                meta=read(path.with_suffix(".json"))
                if meta:files.append((label,row,seed,path,meta))
    key=digest([sha(__file__),[(a,b,c,m["sha256"]) for a,b,c,p,m in files]])
    target=WORK/"waveforms"/f"{item['id']}-{key}.json"
    cached=read(target)
    if cached:return cached
    waves={}
    for label,row,seed,path,meta in files:
        if sha(path)!=meta["sha256"]:raise ValueError("Audio hash changed during waveform comparison")
        data,rate=sf.read(path,dtype="float32",always_2d=True)
        mono=data.mean(1);width=max(1,round(rate/4000))
        filtered=np.convolve(mono,np.ones(width)/width,mode="same")
        waves[label,row,seed]=np.interp(np.arange(0,len(mono),rate/4000),np.arange(len(mono)),filtered)
    cases=[(row,seed) for row in (2,3) for seed in (1709,2903)]
    results=[]
    for label in labels:
        for a,b in itertools.combinations(cases,2):
            if (label,*a) not in waves or (label,*b) not in waves:continue
            value=correlation(waves[label,*a],waves[label,*b])
            controls={k:correlation(waves[k,*a],waves[k,*b]) for k in ("off","caption")}
            reference=max((v for v in controls.values() if v is not None),default=0.)
            results.append(dict(checkpoint=label,first=list(a),second=list(b),correlation=value,
                controls=controls,flagged=value is not None and value>.995 and reference<.98))
    result=dict(slider=item["id"],source_sha256=sha(__file__),comparisons=len(results),
        flags=[r for r in results if r["flagged"]],
        maximum_correlation=max((r["correlation"] for r in results if r["correlation"] is not None),default=None),
        method="Absolute normalized waveform correlation at identical timing, mono box-filtered/interpolated 4 kHz; duration difference at most 2%. Flags require >0.995 candidate correlation and <0.98 for both controls.",
        limitation="Detects near-identical waveforms, including gain or polarity changes. Does not measure shared melodies, timbres or musical diversity.")
    write(target,result)
    return result
