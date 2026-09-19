"""Cached CPU measurements and the existing quality-later-v2 selection rule."""
import itertools
import shutil

from common import *
import numpy as np
import soundfile as sf
sys.path.insert(0,str(ROOT.parent/".cache/slider-quality/python"))
sys.path.insert(0,str(AUDIT))
from selection import select
from analysis.uni16_release_audit_20260914 import dsp
from analysis.gan_bcap.autonomous_audio import coverage_weights
from slider_selection import features as F


def correlation(a,b):
    # Same waveform screen as the existing release audit.
    if abs(len(a)-len(b))>.02*max(len(a),len(b)): return None
    n=min(len(a),len(b));a=np.asarray(a[:n],dtype=np.float64);b=np.asarray(b[:n],dtype=np.float64)
    a=a-a.mean();b=b-b.mean();scale=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return None if scale<1e-12 else float(min(1.,abs(np.sum(a*b))/scale))


class Measurer:
    def __init__(self):
        F.CONCEPTS={"female":F.CONCEPTS["female"]}
        self.judge=F.AudioMeasurer(WORK/"audio-cache","cpu")

    def measure(self,folder):
        meta=read(folder/"render.json")
        path=folder/"song.wav"
        if sha(path)!=meta["wav_sha256"]: raise ValueError("Audio hash changed")
        cache=WORK/"measurements"/(meta["wav_sha256"]+".json")
        cached=read(cache)
        if cached: return cached
        data,rate=sf.read(path,dtype="float32",always_2d=True)
        if not data.size or not np.isfinite(data).all(): raise ValueError("Invalid audio")
        rms=float(np.sqrt(np.mean(data.astype(np.float64)**2)))
        norm=WORK/"normalized"/(meta["wav_sha256"]+".wav")
        norm.parent.mkdir(exist_ok=True)
        sf.write(norm,data*(.1/max(rms,1e-12)),rate,subtype="FLOAT")
        measured=self.judge.measure(norm)
        shares=coverage_weights(measured["windows"])
        average=lambda fn:sum(w*fn(m) for w,m in zip(shares,measured["windows"]))
        result=dict(audio=str(path),sha256=meta["wav_sha256"],rms=rms,duration=len(data)/rate,
            clipped_fraction=float(np.mean(np.abs(data)>=.999)),
            concept=average(lambda m:m["concept"]["female"]),
            enjoyment=average(lambda m:m["aesthetics"]["CE"]),production=average(lambda m:m["aesthetics"]["PQ"]),
            hf14k_fraction=F.fullband_features(path)["hf14k_fraction"],dsp=dsp.measure(path),
            lyrics=None,transcript=None,lyric_measurement="not_rescored_informational_only",
            provenance=measured["provenance"],windows=measured["windows"])
        write(cache,result)
        return result


def waveforms(labels):
    waves={}
    for label in ["off","caption",*labels]:
        for row,seed in CASES:
            path=PAGE/f"row-{row}-seed-{seed}"/label/"song.wav"
            data,rate=sf.read(path,dtype="float32",always_2d=True)
            mono=data.mean(1);width=max(1,round(rate/4000))
            filtered=np.convolve(mono,np.ones(width)/width,mode="same")
            waves[label,row,seed]=np.interp(np.arange(0,len(mono),rate/4000),np.arange(len(mono)),filtered)
    results=[]
    for label in labels:
        for a,b in itertools.combinations(CASES,2):
            value=correlation(waves[label,*a],waves[label,*b])
            controls={k:correlation(waves[k,*a],waves[k,*b]) for k in ("off","caption")}
            reference=max((v for v in controls.values() if v is not None),default=0.)
            results.append(dict(checkpoint=label,first=list(a),second=list(b),correlation=value,controls=controls,
                flagged=value is not None and value>.995 and reference<.98))
    return dict(comparisons=results,flags=[r for r in results if r["flagged"]])


def main():
    spec=verify_protocol()
    labels=[label for label in ["reference600",*[f"step{s}" for s in MILESTONES]]
        if all((PAGE/f"row-{row}-seed-{seed}"/label/"render.json").exists() for row,seed in CASES)]
    if not labels: raise ValueError("No complete four-case candidate")
    judge=Measurer();candidates=[]
    for label in labels:
        integrity=read(WORK/"integrity"/f"{label}.json")
        if sha(weights(label))!=integrity["weights_sha256"]: raise ValueError("Checkpoint hash changed")
        clips=[]
        for row,seed in CASES:
            base=PAGE/f"row-{row}-seed-{seed}"
            values={k:judge.measure(base/take) for k,take in
                (("candidate",label),("baseline","off"),("positive_reference","caption"))}
            c,b,p=(values[k] for k in ("candidate","baseline","positive_reference"))
            diag=dsp.diagnostics(c,b,p)
            flags=[f for f in diag["failures"] if f!="lyric_proxy_regression"]
            flags+=dsp.warnings(c["dsp"],b["dsp"],p["dsp"])
            if c["sha256"]==b["sha256"]: flags.append("identical_to_off")
            clips.append(dict(row=row,seed=seed,fixture=f"row-{row}-seed-{seed}",**values,
                diagnostics=diag,technical_flags=flags,lyric_flags=[],score=0.,components={}))
        candidates.append(dict(label=label,step=int(label[4:]) if label.startswith("step") else 600,
            weights=str(weights(label)),integrity=integrity,clips=clips))
    waveform_audit=waveforms(labels)
    write(PAGE/"waveform-audit.json",waveform_audit)
    decision=select(candidates,{"close_call_gap":.1},spec["ranking_policy"],waveform_audit["flags"])
    result=dict(decision,complete=all(f"step{s}" in labels for s in MILESTONES),
        protocol_sha256=sha(WORK/"protocol.json"),coverage=dict(rows=[2,3],seeds=[1709,2903],seconds=20,strength=1),
        interpretation="Same quality-later-v2 rule as the Music 3 audit. Style is diagnostic only; no new ASR scoring.")
    write(PAGE/"ranking.json",result)
    write(WORK/"ranking.json",result)
    chosen=result["recommendation"]
    if chosen:
        target=PAGE/"selected";target.mkdir(exist_ok=True)
        for suffix in (".safetensors",".json"):
            source=Path(chosen["weights"]).with_suffix(suffix)
            temporary=target/f"female-yue2{suffix}.tmp"
            shutil.copyfile(source,temporary);temporary.replace(target/f"female-yue2{suffix}")
        assert sha(target/"female-yue2.safetensors")==chosen["weights_sha256"]
        write(target/"selection.json",chosen)
    print(json.dumps(dict(complete=result["complete"],recommendation=chosen,
        eligible=result["eligible_checkpoints"])),flush=True)


if __name__=="__main__": main()
