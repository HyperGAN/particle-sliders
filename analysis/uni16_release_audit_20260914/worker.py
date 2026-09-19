"""Cached CPU measurements of every available matched candidate, continuously."""
import argparse
import fcntl
import os
import signal
import time

from common import *
from dsp import measure as dsp_measure, warnings as dsp_warnings
from rank import summarize

STOP = False


def integrity(item, label, expected):
    import torch
    from safetensors.torch import load_file
    path = weights(item, label)
    if sha(path) != expected: raise ValueError("Weights differ from rendered checkpoint")
    target = WORK / "integrity" / f"{expected}.json"
    previous = read(target)
    if previous and previous["path"] == str(path) and previous["mtime_ns"] == path.stat().st_mtime_ns:
        return previous
    tensors = load_file(str(path))
    if not tensors or not all(torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("Non-finite or empty LoRA weights")
    state_path = None
    if label == "step600":
        state_path = Path(read(CAMPAIGN / "jobs" / f"{item['id']}.json")["warmup"]["state"])
    elif label.startswith("step"):
        state_path = path.parent / f"state-{label}.pt"
    match = None
    if state_path:
        blob = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
        full = blob["modules"]["lora"] if label == "step600" else blob["network"]
        match = tensors.keys() == full.keys() and all(torch.equal(t, full[k]) for k,t in tensors.items())
        if not match: raise ValueError("Export does not match pinned full training state")
    meta = read(path.with_suffix(".json"), {})
    if meta.get("rank") != 8 or float(meta.get("alpha", 0)) != 8.:
        raise ValueError("Unexpected rank or alpha in checkpoint metadata")
    from app.rewriter import _artist_name_hit
    if _artist_name_hit("", str(meta)): raise ValueError("Checkpoint metadata requires prompt-name review")
    result = dict(path=str(path), weights_sha256=expected, mtime_ns=path.stat().st_mtime_ns,
        finite=True, export_matches_full_state=match, full_state_sha256=sha(state_path) if state_path else None,
        state_scope="Published export only" if state_path is None else "Compared every LoRA tensor with full state")
    write(target, result)
    return result


class Measurer:
    def __init__(self, protocol):
        from analysis.uni16_20260906.score import DESCRIPTIONS
        from analysis.gan_bcap import autonomous_audio as audio
        audio.CONCEPTS = DESCRIPTIONS
        self.audio, self.judge, self.protocol = audio, audio.Judge(), protocol

    def measure(self, path, sheet, concept, meta):
        identity = digest([meta["sha256"], sheet, concept, sha(WORK / "protocol.json")])
        target = WORK / "measurements" / f"{identity}.json"
        (WORK / "locks").mkdir(exist_ok=True)
        with (WORK / "locks" / f"{meta['sha256']}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if sha(path) != meta["sha256"]: raise ValueError("Rendered audio hash changed")
            cached = read(target)
            if cached: return cached
            result = self.perceptual_measure(path, sheet, concept)
            result["dsp"] = dsp_measure(path)
            from app.rewriter import _artist_name_hit
            if result["transcript"] and _artist_name_hit("", result["transcript"]):
                result["transcript"] = "ASR text withheld by prompt-name validation."
            write(target, result)
            return result

    def perceptual_measure(self, path, sheet, concept):
        """Same perceptual windows as the original Judge; ASR cache reads only."""
        import numpy as np
        import soundfile as sf
        from scripts.lm_score import sha1_file, MEASURE_VERSION
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        if not data.size or not np.isfinite(data).all(): raise ValueError("Empty or non-finite audio")
        original_sha = sha(path)
        rms = float(np.sqrt(np.mean(data.astype("float64")**2)))
        normalized = ROOT / "analysis/gan_bcap/normalized_measurement_audio" / f"{original_sha}-rms010-f32.wav"
        if not normalized.exists():
            normalized.parent.mkdir(parents=True, exist_ok=True)
            temp = normalized.with_suffix(f".{os.getpid()}.tmp.wav")
            sf.write(temp, data*(.1/max(rms,1e-12)), rate, subtype="FLOAT"); temp.replace(normalized)
        measured = self.judge.audio.measure(normalized)
        shares = self.audio.coverage_weights(measured["windows"])
        def average(fn): return sum(a*fn(w) for a,w in zip(shares,measured["windows"]))
        cached_asr = read(self.judge.asr.cache_dir / f"{sha1_file(path)}-v{MEASURE_VERSION}.json")
        lyric = self.audio.F.lyric_features(sheet, cached_asr["text"]) if cached_asr else None
        return dict(audio=str(path.resolve()), sha256=original_sha,
            perceptual_measurement_sha256=measured["sha256"],
            concept=average(lambda w:w["concept"][concept]),
            enjoyment=average(lambda w:w["aesthetics"]["CE"]),
            production=average(lambda w:w["aesthetics"]["PQ"]),
            lyrics=(lyric["phrase_accuracy"]+lyric["recall"])/2 if lyric else None,
            lyric_diagnostics=lyric, transcript=cached_asr["text"] if cached_asr else None,
            lyric_measurement="cached_asr" if cached_asr else "not_rescored_user_reports_consistently_fine",
            rms=rms, duration=len(data)/rate, clipped_fraction=float(np.mean(np.abs(data)>=.999)),
            hf14k_fraction=self.audio.F.fullband_features(path)["hf14k_fraction"])

    def candidate(self, item, label, rows, metas):
        from dsp import diagnostics, components
        expected = {meta["spec"]["weights_sha256"] for meta in metas.values()}
        if len(expected) != 1: raise ValueError("Candidate clips came from different checkpoints")
        check = integrity(item, label, expected.pop())
        clips = []
        for row in self.protocol["rows"]:
            for seed in self.protocol["seeds"]:
                values = {}
                for key, take in (("baseline", "off"), ("positive_reference", "caption"), ("candidate", label)):
                    path = AUDIO / item["id"] / f"row-{row}-seed-{seed}" / f"{take}.wav"
                    meta = read(path.with_suffix(".json"))
                    expected_prompt = rows[row]["positive" if take == "caption" else "neutral"]
                    if not meta or meta["spec"]["prompt_sha256"] != digest(expected_prompt) or meta["spec"]["lyrics_sha256"] != digest(rows[row]["lyrics"]):
                        raise ValueError("Prompt or lyric identity mismatch")
                    if meta["spec"]["campaign_sha256"] != self.protocol["training_manifest_sha256"] or meta["seed"] != seed or meta["row"] != row:
                        raise ValueError("Render protocol, seed or row mismatch")
                    if meta["spec"]["duration"] != 20. or meta["spec"]["scale"] != (1. if take == label else 0.):
                        raise ValueError("Unmatched duration or strength")
                    values[key] = self.measure(path, rows[row]["lyrics"], item["id"], meta)
                c, b, p = (values[k] for k in ("candidate", "baseline", "positive_reference"))
                parts = components(c, b, p, self.audio.RULE)
                score = sum(self.protocol["ranking_weights"][k]*parts[k] for k in self.protocol["ranking_weights"])
                diag = diagnostics(c, b, p)
                technical = [f for f in diag["failures"] if f != "lyric_proxy_regression"]
                technical += dsp_warnings(c["dsp"], b["dsp"], p["dsp"])
                if c["sha256"] == b["sha256"]: technical.append("identical_to_off")
                clips.append(dict(row=row, seed=seed, fixture=f"row-{row}-seed-{seed}", **values,
                    score=score, components=parts, diagnostics=diag, technical_flags=technical,
                    lyric_flags=[f for f in diag["failures"] if f == "lyric_proxy_regression"],
                    mp3=f"../uni16-fresh3400-20260912/{item['id']}/row-{row}-seed-{seed}/{label}.mp3"))
        return dict(label=label, step=660 if label == "published660" else int(label[4:]),
            weights=str(weights(item, label)), integrity=check, clips=clips)


def main(lane, once=False):
    global STOP
    import torch
    import yaml
    torch.set_num_threads(4)
    protocol = verify()
    items = [item for item in catalog() if item["index"] % 2 == lane]
    judge = Measurer(protocol)
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    (WORK / "locks").mkdir(exist_ok=True)
    with (WORK / "locks" / f"worker-{lane}.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not STOP:
            verify()
            work_done = False
            for item in items:
                if STOP: break
                rows = yaml.safe_load(Path(item["eval_prompts"]).read_text())["rows"]
                report_path = WORK / "sliders" / f"{item['id']}.json"
                report = read(report_path, dict(id=item["id"], label=item["label"], candidates=[]))
                known = {r["label"] for r in report["candidates"]}
                for label in protocol["candidates"]:
                    if STOP or label in known: continue
                    metas = {(row,seed):read(AUDIO / item["id"] / f"row-{row}-seed-{seed}" / f"{label}.json")
                             for row in protocol["rows"] for seed in protocol["seeds"]}
                    if not all(metas.values()): continue
                    write(WORK / f"worker-{lane}.json", dict(status="measuring", slider=item["id"], checkpoint=label, updated=time.time()))
                    started = time.monotonic()
                    try:
                        candidate = judge.candidate(item, label, rows, metas)
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                        candidate = dict(label=label, step=660 if label == "published660" else int(label[4:]),
                            error=f"{type(exc).__name__}: {exc}", weights=str(weights(item, label)))
                    report["candidates"].append(candidate)
                    report.update(summarize(report["candidates"], protocol), updated=time.time(), protocol_sha256=sha(WORK / "protocol.json"))
                    write(report_path, report)
                    print(f"{item['id']} {label}: {'ERROR' if candidate.get('error') else 'measured'} in {time.monotonic()-started:.1f}s", flush=True)
                    work_done = True
            if once: break
            budgets = read(CAMPAIGN / "budget.json")["targets"]
            finished = all(read(CAMPAIGN / "jobs" / f"{i['id']}.json", {}).get("status") == "complete" and
                any(c["label"] == f"step{budgets[i['id']]}" for c in read(WORK / "sliders" / f"{i['id']}.json", {}).get("candidates", [])) for i in items)
            if finished: break
            write(WORK / f"worker-{lane}.json", dict(status="waiting_for_new_samples", updated=time.time()))
            if not work_done:
                for _ in range(30):
                    if STOP: break
                    time.sleep(1)
        write(WORK / f"worker-{lane}.json", dict(status="stopped" if STOP else "complete", updated=time.time()))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--lane", type=int, choices=(0,1), required=True); p.add_argument("--once", action="store_true")
    args=p.parse_args(); main(args.lane, args.once)
