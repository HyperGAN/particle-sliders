"""Render and measure each declared longer checkpoint while training continues."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
PARENT = ROOT / "analysis/fresh_seed_20260910"
PAGE = ROOT / "eval/listen/fresh-seed-long-lofi-20260910"
RUN = WORK / "lofi-fresh-long"
PY = "/home/mikkel/anaconda3/envs/minimax-music3/bin/python"
UNIT = "music-fresh-long-train-20260910"
PROTOCOL = json.loads((WORK / "protocol.json").read_text())


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weight(step):
    if step == 660:
        return PARENT / "lofi-fresh-history/lofi-fresh-history_step660.safetensors"
    return RUN / f"lofi-fresh-long_step{step}.safetensors"


def folder(step, row, seed):
    if step == 660:
        return Path(PROTOCOL["evaluation"]["reference_page"]) / f"row-{row}" / f"{weight(step).stem}-s{seed}"
    return PAGE / f"step-{step}/row-{row}" / f"{weight(step).stem}-s{seed}"


def records():
    reference = read(PARENT / "scores.json")["records"]
    result = {660: [r for r in reference if r["checkpoint"]["path"] == str(weight(660))]}
    for step in PROTOCOL["evaluation"]["steps"]:
        measured = read(WORK / f"scores-{step}.json", {})
        if measured.get("status") == "complete":
            assert len(measured["records"]) == 4
            result[step] = measured["records"]
    return result


def case(record):
    return (int(Path(record["candidate"]["audio"]).parent.parent.name.split("-")[-1]), record["seed"])


def publish(phase):
    measured = records()
    lookup = {(step, *case(r)): r for step, group in measured.items() for r in group}
    pairs = []
    for row in PROTOCOL["evaluation"]["rows"]:
        for seed in PROTOCOL["evaluation"]["seeds"]:
            takes = []
            for step in [660, *PROTOCOL["evaluation"]["steps"]]:
                source = folder(step, row, seed)
                if not (source / "checkpoint.json").exists():
                    continue
                wav = next(source.glob("02_slider_*_plus1.wav"))
                dest = PAGE / "audio" / f"step{step}-r{row}-s{seed}.wav"
                mp3 = dest.with_suffix(".mp3")
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():
                    os.link(wav, dest)
                assert sha(dest) == sha(wav)
                if not mp3.exists():
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(dest), "-map_metadata", "-1",
                        "-c:a", "libmp3lame", "-b:a", "192k", "-threads", "1", str(mp3)], check=True)
                score = lookup.get((step, row, seed))
                takes.append(dict(step=step, label=f"Fresh seed · step {step}", wav=f"audio/{dest.name}",
                    mp3=f"audio/{mp3.name}", sha256=sha(dest), mp3_sha256=sha(mp3),
                    ce=score["candidate"]["enjoyment"] if score else None))
            pairs.append(dict(row=row, seed=seed, takes=takes))
    summaries = []
    baseline = {case(r): r for r in measured[660]}
    for step, group in measured.items():
        differences = [r["candidate"]["enjoyment"] - baseline[case(r)]["candidate"]["enjoyment"] for r in group]
        summaries.append(dict(step=step, mean_ce=statistics.mean(r["candidate"]["enjoyment"] for r in group),
            mean_pq=statistics.mean(r["candidate"]["production"] for r in group),
            mean_lyric_agreement=statistics.mean(r["candidate"]["lyrics"] for r in group),
            delta_vs660=statistics.mean(differences), wins=sum(d>0 for d in differences), total=len(group),
            paired_deltas=[dict(row=case(r)[0], seed=r["seed"], ce_delta=d) for r, d in zip(group, differences)]))
    training = read(RUN / "status.json", {"status": "starting", "completed": 660, "total": PROTOCOL["end_steps"]})
    status = dict(phase=phase, training=training, updated=time.time(), rendered=sum(len(p["takes"]) for p in pairs),
        total_previews=4*(1+len(PROTOCOL["evaluation"]["steps"])))
    write(WORK / "status.json", status); write(PAGE / "status.json", status)
    write(WORK / "summary.json", summaries)
    write(PAGE / "manifest.json", dict(pairs=pairs, summaries=summaries))
    write(PAGE / "measurements.json", {str(k): v for k, v in measured.items()})
    lines = ["# Longer fresh-seed Lo-fi results", "", f"Status: {phase}.", "",
        "All four monitoring cases receive equal weight. The reference is fresh step 660.", "",
        "| Step | Mean CE | CE change vs 660 | Wins | Production quality | Lyric agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for value in summaries:
        lines.append(f"| {value['step']} | {value['mean_ce']:.6f} | {value['delta_vs660']:+.6f} | "
            f"{value['wins']}/{value['total']} | {value['mean_pq']:.6f} | {value['mean_lyric_agreement']:.6f} |")
    for value in summaries:
        if value["step"] == 660:
            continue
        lines.extend(["", f"## Step {value['step']}: every paired CE difference", ""])
        for delta in value["paired_deltas"]:
            lines.append(f"- Arrangement {delta['row']-1}, seed {delta['seed']}: {delta['ce_delta']:+.6f}")
    lines.extend(["", "These reused monitoring cases do not independently confirm checkpoint selection.",
        "One training prompt and one continued trajectory; style teachers remain fixed.", "",
        "[Listen](http://100.90.104.57:8888/fresh-seed-long-lofi-20260910/)", ""])
    (WORK / "result.md").write_text("\n".join(lines))


def wait_process(process, phase):
    while process.poll() is None:
        publish(phase)
        time.sleep(10)
    if process.returncode:
        raise RuntimeError(f"{phase} failed with exit code {process.returncode}; see retained log")


def audit_step(step):
    measured = records()
    references = {case(r): r for r in measured[660]}
    checks = []
    for record in measured[step]:
        row, seed = case(record)
        reference = references[(row, seed)]
        spec = read(PAGE / f"step-{step}/row-{row}/render_spec.json")
        assert spec["row"] == row and spec["seeds"] == PROTOCOL["evaluation"]["seeds"]
        assert spec["duration"] == 20 and spec["scales"] == [0., 1.] and spec["seed_retries"] == 0
        assert spec["prompts_sha256"] == sha(Path(PROTOCOL["evaluation"]["prompts"]))
        assert record["checkpoint"]["sha256"] == sha(weight(step))
        assert record["candidate"]["sha256"] == sha(Path(record["candidate"]["audio"]))
        for key in ("baseline", "positive_reference"):
            assert record[key]["sha256"] == reference[key]["sha256"]
        assert record["fixture"] == reference["fixture"]
        checks.append(dict(row=row, seed=seed, candidate_sha256=record["candidate"]["sha256"],
            matched_reference_hashes=True, locked_render_settings=True))
    assert len(checks) == 4
    write(WORK / f"evaluation-audit-{step}.json", dict(step=step, cases=checks, all_checks_passed=True))


def main():
    PAGE.mkdir(parents=True, exist_ok=True)
    (PAGE / "index.html").write_text((WORK / "page.html").read_text())
    for step in PROTOCOL["evaluation"]["steps"]:
        while not (RUN / f"state-step{step}.pt").exists():
            publish(f"Training toward step {step}")
            failure = read(RUN / "failure.json")
            if failure:
                raise RuntimeError(f"Training failed: {failure}")
            active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not active:
                raise RuntimeError("Training service stopped before the declared checkpoint")
            time.sleep(10)
        for row in PROTOCOL["evaluation"]["rows"]:
            reuse = []
            for seed in PROTOCOL["evaluation"]["seeds"]:
                source, target = folder(660, row, seed), folder(step, row, seed)
                target.mkdir(parents=True, exist_ok=True)
                for name in ("01_slider_neutral_base_zero.wav", "03_REF_prompt_Lo-fi_no_slider.wav", "04_REF_prompt_Off_no_slider.wav"):
                    if not (target / name).exists():
                        shutil.copyfile(source / name, target / name)
                    assert sha(target / name) == sha(source / name)
                    reuse.append(dict(source=str(source / name), target=str(target / name), sha256=sha(source / name)))
            write(WORK / f"reference-reuse-{step}-row-{row}.json", reuse)
            if all((folder(step, row, seed) / "checkpoint.json").exists() for seed in PROTOCOL["evaluation"]["seeds"]):
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", HF_HOME="/ml2/music/.cache/huggingface",
                PYTHONPATH=f"{PARENT}/runtime:/ml2/music", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            with (WORK / f"render-{step}-row-{row}.log").open("a") as log:
                process = subprocess.Popen([PY, "-u", str(PARENT / "runtime/analysis/gan_bcap/render_v2.py"),
                    "--weights", str(weight(step)), "--out", str(PAGE / f"step-{step}/row-{row}"),
                    "--prompts", PROTOCOL["evaluation"]["prompts"], "--row", str(row), "--seeds", "101", "303", "--duration", "20"],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                wait_process(process, f"Rendering step {step}")
        if step not in records():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_HOME="/ml2/music/.cache/huggingface",
                PYTHONPATH=str(ROOT), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            with (WORK / f"score-{step}.log").open("a") as log:
                process = subprocess.Popen([PY, "-u", str(ROOT / "analysis/uni16_20260906/score.py"), "--id", "lofi", "--folders",
                    *[str(PAGE / f"step-{step}/row-{r}") for r in PROTOCOL["evaluation"]["rows"]],
                    "--output", str(WORK / f"scores-{step}.json")], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                wait_process(process, f"Measuring step {step}")
        audit_step(step)
        publish(f"Step {step} measured")
    assert read(RUN / "status.json")["status"] == "complete"
    publish("Complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(WORK / "evaluation-failure.json", dict(type=type(error).__name__, message=str(error)))
        publish("Stopped — see experiment log")
        raise
