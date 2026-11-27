"""Anima's typed image questions, two-order diversity score and durable readings."""
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import sha256
from http.client import HTTPException
from itertools import combinations
import json
import math
import os
from pathlib import Path
from statistics import fmean
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from . import canonical, digest, read_json, write_json
from . import image_questions, image_score as ws

JEV = "typesafe/jev-1.14"


class Unmeasured(RuntimeError):
    """The provider has not supplied the required measurement."""


def raw_image(ref):
    raw = Path(ref["path"]).read_bytes()
    if len(raw) != ref["bytes"] or sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("image differs from its render receipt")
    return raw


class Reader:
    def __init__(self, root, phrase):
        self.root = Path(root) / "readings"
        self.root.mkdir(parents=True, exist_ok=True)
        self.asked = image_questions.build(phrase)

    def decide(self, images, model, state, questions):
        binding = dict(model=model, state=state, questions=questions, endpoint=ws.URL,
                       images=[sha256(raw).hexdigest() for raw in images])
        key = digest(binding)
        with (self.root / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for i in range(12):
                receipt = self.root / f"{key}.{i}.json"
                if receipt.exists():
                    result = read_json(receipt)
                    if result["binding"] != binding or result["body_sha256"] != digest(result["body"]):
                        raise ValueError("judge receipt identity changed")
                else:
                    pending = receipt.with_suffix(".pending")
                    if pending.exists():
                        # A prior worker exited without receiving a result. Its immutable
                        # attempt is unmeasured. Request a fresh independent reading.
                        interrupted = {"delivery":"unknown"}
                        write_json(receipt, dict(binding=binding, status=0, body=interrupted,
                                                 body_sha256=digest(interrupted)))
                        pending.unlink()
                        continue
                    secret = os.environ.get("NANOGPT_API_KEY")
                    if not secret:
                        raise Unmeasured("NANOGPT_API_KEY is required to score new images")
                    parts = [state]
                    for n, raw in enumerate(images, 1):
                        uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
                        parts.extend([f"image_{n}:", {"type":"image_url","image_url":{"url":uri}}])
                    request = Request(ws.URL, data=canonical(dict(model=model, state=parts, questions=questions)).encode(),
                                      headers={"Content-Type":"application/json"})
                    request.add_unredirected_header("Authorization", "Bearer " + secret)
                    write_json(pending, binding)
                    try:
                        with urlopen(request, timeout=120) as response:
                            body = json.load(response)
                        status = 200
                    except HTTPError as error:
                        status = error.code
                        body = {"http_status":status}
                        error.close()
                    except (OSError, HTTPException, ValueError) as error:
                        status, body = 0, {"delivery":"unknown","error_type":type(error).__name__}
                    result = dict(binding=binding, status=status, body=body, body_sha256=digest(body))
                    write_json(receipt, result)
                    pending.unlink()
                if result["status"] == 200:
                    body = result["body"]
                    if set(body.get("answers", {})) == set(questions):
                        values = {}
                        for name, question in questions.items():
                            kind = question["type"]
                            answer = body["answers"][name]
                            value = answer.get(kind)
                            bound = len(question["criteria"])-1 if kind == "score" else 1
                            if answer.get("type",kind) != kind or type(value) not in (int,float) or not 0 <= value <= bound:
                                break
                            values[name] = value / bound
                        if len(values) == len(questions):
                            return values
                elif result["status"] not in (0, 408, 429, 500, 502, 503, 504):
                    raise Unmeasured(f"judge returned HTTP {result['status']} in {receipt}")
                if i < 11 and not (self.root/f"{key}.{i+1}.json").exists():
                    time.sleep(min(30, 2**i))
        raise Unmeasured(f"no valid response for {key}")

    def ask(self, images, which):
        question = self.asked[which]
        return self.decide(images, JEV, question["state"], question["questions"])

    def broken(self, ref):
        return fmean(fmean(self.decide([piece], ws.MODEL, ws.STATE, ws.QUESTIONS)["broken_drawing"]
                           for piece in pieces) for pieces in ws.tiles(ref["path"]))

    def measure(self, off, on, *, teacher=False):
        a,b = raw_image(off),raw_image(on)
        paired = self.ask([a,b],"pair")
        original = self.ask([a,a],"effect")["effect"]
        effect = self.ask([a,b],"effect")["effect"]
        broken,off_broken = self.broken(on),self.broken(off)
        figure = self.ask([b],"figure")["figure"]
        recolored = self.ask([a,b],"recolor")["recolored"]
        quality = ws.quality(broken, off_broken, paired["one_figure"], figure)
        row = dict(off=off,on=on,original=original,effect=effect,broken=broken,off_broken=off_broken,
                   figure=figure,recolored=recolored,quality=quality,case=ws.score_case(effect,original,quality,recolored),**paired)
        if teacher:
            row["separate"] = self.ask([a,b],"separate")["separate"]
        return row

    def repeat(self, pair, on):
        left,right = [raw_image(on[c]) for c in pair]
        return dict(cases=list(pair),repeat=fmean(self.ask(order,"repeat")["repeat"]
                                                  for order in ([left,right],[right,left])))

    def panel(self, off, on):
        with ThreadPoolExecutor(12) as pool:
            futures = {c:pool.submit(self.measure,off[c],on[c]) for c in off}
            repeat = [pool.submit(self.repeat,p,on) for p in combinations(off,2)]
            rows = {c:f.result() for c,f in futures.items()}
            pairs = [f.result() for f in repeat]
        summary = ws.summarize({c:r["case"] for c,r in rows.items()},pairs)
        working = sum(r["quality"] >= .5 and r["recolored"] < .5 and ws.effect(r["effect"],r["original"]) >= .5 for r in rows.values())
        return dict(**summary,working=working,rows=rows,pairs=pairs)


def read_candidates(root, phrase, candidates):
    path = Path(root)/"judgments"/f"{candidates['scene']}-{candidates['attempt']}.json"
    binding = digest(candidates)
    if path.exists():
        saved = read_json(path)
        if saved["candidates"] != binding:
            raise ValueError("candidate images changed after judgment")
        return saved
    reader = Reader(root, phrase)
    with ThreadPoolExecutor(3) as pool:
        jobs = [pool.submit(reader.measure,candidates["neutral"],t["image"],teacher=True)
                for t in candidates["options"]]
        options = [dict(**t,reading=j.result()) for t,j in zip(candidates["options"],jobs)]
    eligible = [o for o in options if o["reading"]["case"] >= .35 and o["reading"]["separate"] < .5]
    selected = max(eligible,key=lambda o:o["reading"]["case"]) if eligible else None
    result = dict(candidates=binding, options=options, chosen=selected)
    write_json(path,result)
    return result
