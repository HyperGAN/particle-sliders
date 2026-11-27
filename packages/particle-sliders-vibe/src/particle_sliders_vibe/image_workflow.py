"""Anima's phrase-to-slider process on a product's bound native image sampler."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys

from PIL import Image
import torch

from . import digest, file_digest, read_json, write_json
from .image_integrity import check
from .image_judge import Reader, raw_image, read_candidates
from .image_run import pin
from .image_targets import capture
from .image_training import train, UPDATES

MATTE_SHA = "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"


def collect(run):
    saved = run.root/"teachers.json"
    if saved.exists():
        return read_json(saved)
    model = Path(os.environ.get("U2NET_MODEL", Path.home()/".u2net/u2net.onnx"))
    if not model.exists() or file_digest(model) != MATTE_SHA:
        raise ValueError(f"expected the pinned U²-Net model at {model}")
    progress = run.root/"teacher-progress.json"
    prior = read_json(progress) if progress.exists() else {}
    kept, record, attempt = prior.get("kept",{}),prior.get("attempts",[]),prior.get("round",0)
    with ThreadPoolExecutor(4) as pool:
        while len(kept) < 16:
            pending = []
            for scene in run.grid["teaching"]:
                if scene["id"] in kept:
                    continue
                candidates = run.candidates(scene, attempt)
                write_json(run.root/"candidates"/f"{scene['id']}-{attempt}.json", candidates)
                pending.append((scene,candidates,pool.submit(read_candidates,run.root,run.vibe,candidates)))
            for scene,candidates,job in pending:
                options = job.result()["options"]
                viable = [o for o in options if o["reading"]["case"] >= .35 and o["reading"]["separate"] < .5
                          and o["reading"]["quality"] >= .5 and max(o["reading"]["off_broken"], o["reading"]["broken"]) < .5]
                chosen = None
                structure = {}
                reader = Reader(run.root,run.vibe)
                for candidate in sorted(viable,key=lambda o:o["reading"]["case"],reverse=True):
                    for ref in (candidates["neutral"],candidate["image"]):
                        if ref["sha256"] not in structure:
                            structure[ref["sha256"]] = check(reader,raw_image(ref))["status"]
                    if all(structure[ref["sha256"]] == "pass" for ref in (candidates["neutral"],candidate["image"])):
                        chosen = candidate
                        break
                key = scene["id"]
                entry = dict(scene=key,attempt=attempt,seed=candidates["seed"],chosen=None,structure=structure)
                record.append(entry)
                if chosen is None:
                    continue
                matte = run.root/"mattes"/f"{key}-{attempt}.png"
                matte.parent.mkdir(exist_ok=True)
                cmd = [os.environ.get("PARTICLE_VIBE_MATTE_PYTHON",sys.executable),
                       str(Path(__file__).with_name("image_matte.py")), str(model), candidates["neutral"]["path"], str(matte)]
                process = subprocess.run(cmd,capture_output=True,text=True)
                if process.returncode:
                    if "Unusable foreground region" in process.stderr or "No character region" in process.stderr:
                        entry["matte_error"] = process.stderr.splitlines()[-1]
                        continue
                    raise RuntimeError(process.stderr)
                entry["chosen"] = chosen["positive"]
                kept[key] = dict(neutral=scene["neutral"],positive=chosen["positive"],explicit=chosen["explicit"],
                                 seed=candidates["seed"],framing=scene["framing"],case=chosen["reading"]["case"],
                                 neutral_image=candidates["neutral"],positive_image=chosen["image"],matte=pin(matte))
            attempt += 1
            write_json(run.root/"teacher-progress.json",dict(kept=kept,round=attempt,attempts=record))
    result = dict(vibe=run.vibe, kept=kept, attempts=record, matte_model_sha256=MATTE_SHA)
    write_json(saved,result)
    return result


def named_images(run, cases, alpha=0., raw=None):
    return {c:run.image(v["prompt"],v["seed"],strength=alpha,raw=raw) for c,v in cases.items()}


def sheet(pairs, path):
    width, cols = 256, 4
    sheet = Image.new("RGB",(width*cols,2*width*((len(pairs)+cols-1)//cols)),(24,25,28))
    for i,pair in enumerate(pairs):
        for j,ref in enumerate(pair):
            with Image.open(ref["path"]) as image:
                sheet.paste(image.convert("RGB").resize((width,width),Image.Resampling.LANCZOS),
                            (i%cols*width,(2*(i//cols)+j)*width))
    sheet.save(path,quality=88)


def search(run, reader, raw, cases):
    off = named_images(run,cases)
    def trial(alpha):
        path = run.root/"search"/(float(alpha).hex()+".json")
        binding = dict(raw=raw["sha256"],alpha=alpha,cases=cases)
        if path.exists():
            cached = read_json(path)
            if cached["binding"] != binding:
                raise ValueError("alpha trial belongs to a different RAW or panel")
            return cached["score"]
        on = named_images(run,cases,alpha,raw["sha256"])
        result = reader.panel(off,on)
        write_json(path,dict(binding=binding,**result))
        print(f"alpha {alpha:g} scored {result['score']:.4f}",flush=True)
        return result["score"]
    scores = {a:trial(a) for a in (.75,1.,1.25,1.5,1.75,2.)}
    for _ in range(2):
        ordered = sorted(scores)
        best = max(scores,key=lambda a:(scores[a],-a))
        i = ordered.index(best)
        neighbors = ordered[max(0,i-1):i] + ordered[i+1:i+2]
        for other in neighbors:
            if abs(other-best) > 1/16:
                mid = (best+other)/2
                scores[mid] = trial(mid)
    alpha = max(scores,key=lambda a:(scores[a],-a))
    return alpha, sorted(scores.items())


def finish(run, *, measure=False):
    teachers = collect(run)
    print(f"qualified teachers {len(teachers['kept'])}/24",flush=True)
    shards = [capture(run,s) for _,s in sorted(teachers["kept"].items())]
    write_json(run.root/"targets"/"index.json",dict(teachers=digest(teachers),shards=shards))
    raw = train(run,shards)
    torch.cuda.empty_cache()
    run.host.attach(raw["path"])
    try:
        return evaluate(run,raw,teachers,measure)
    finally:
        run.host.runtime.mixer.close()


def evaluate(run, raw, teachers, measure):
    reader = Reader(run.root,run.vibe)
    split, panel = run.panel["splits"],run.panel["cases"]
    cases = {c:panel[c] for c in split["search"]}
    alpha, trials = search(run,reader,raw,cases)
    shown = {c:panel[c] for c in split["search"]+split["ranking"]}
    off, on = named_images(run,shown), named_images(run,shown,alpha,raw["sha256"])
    dest = run.root/"slider"
    dest.mkdir(exist_ok=True)
    slug = re.sub(r"[^\w-]+","-",run.vibe).strip("-") or "slider"
    target = dest/(slug+".safetensors")
    for suffix in (".safetensors",".engine.json",".scoped.json"):
        shutil.copyfile(Path(raw["path"]).with_suffix(suffix),target.with_suffix(suffix))
    sheet([(off[c],on[c]) for c in shown],dest/"samples.jpg")
    write_json(dest/"use.json",dict(vibe=run.vibe,updates=UPDATES,alpha=alpha,raw_sha256=raw["sha256"],
                                     sampler=run.host.profile,step_strengths=[alpha]*run.host.profile["steps"]))
    result = dict(vibe=run.vibe,teachers=len(teachers["kept"]),updates=UPDATES,raw=raw,alpha=alpha,
                  search=dict(trials=trials,score=dict(trials)[alpha]),slider=pin(target),
                  samples=dict(sheet=pin(dest/"samples.jpg"),off=off,on=on))
    if measure:
        held = {s["id"]:dict(prompt=s["neutral"],seed=s["seed"]) for s in run.grid["confirmation"]}
        for name,original,edited in [
            ("ranking",{c:off[c] for c in split["ranking"]},{c:on[c] for c in split["ranking"]}),
            ("confirmation",named_images(run,held),named_images(run,held,alpha,raw["sha256"]))]:
            metrics = reader.panel(original,edited)
            with ThreadPoolExecutor(8) as pool:
                integrity = dict(zip(edited,pool.map(lambda ref:check(reader,raw_image(ref)),edited.values())))
            result[name] = dict(**metrics,structure=integrity)
            write_json(run.root/(name+".json"),result[name])
        statuses = [row["status"] for name in ("ranking","confirmation") for row in result[name]["structure"].values()]
        result["structure_summary"] = {state:statuses.count(state) for state in ("pass","fail","unmeasured")}
    write_json(run.root/("result-measured.json" if measure else "result.json"),result)
    return result
