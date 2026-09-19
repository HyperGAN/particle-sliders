"""Render and measure each declared checkpoint on the way to step 5000."""
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
PAGE = ROOT / "eval/listen/fresh-seed-break-lofi-20260911"
RUN = WORK / "lofi-fresh-break"
PY = "/home/mikkel/anaconda3/envs/minimax-music3/bin/python"
UNIT = "music-fresh-break-train-20260911"
PROTOCOL = json.loads((WORK / "protocol.json").read_text())
# Single-GPU run: rendering waits for training to finish and then reuses the same GPU.
RENDER_GPU = os.environ.get("RENDER_GPU", "1")
REF = PROTOCOL["evaluation"]["reference_step"]


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
    if step == REF:
        return Path(PROTOCOL["source_weights"])
    return RUN / f"lofi-fresh-break_step{step}.safetensors"


def folder(step, row, seed):
    if step == REF:
        return Path(PROTOCOL["evaluation"]["reference_page"]) / f"step-{REF}/row-{row}" / f"{weight(step).stem}-s{seed}"
    return PAGE / f"step-{step}/row-{row}" / f"{weight(step).stem}-s{seed}"


def records():
    reference = read(Path(PROTOCOL["evaluation"]["reference_scores"]))["records"]
    result = {REF: [r for r in reference if r["checkpoint"]["path"] == str(weight(REF))]}
    assert len(result[REF]) == 4, "The parent run must have measured all four reference cases"
    for step in PROTOCOL["evaluation"]["steps"]:
        measured = read(WORK / f"scores-{step}.json", {})
        if measured.get("status") == "complete":
            assert len(measured["records"]) == 4
            result[step] = measured["records"]
    return result


def case(record):
    return (int(Path(record["candidate"]["audio"]).parent.parent.name.split("-")[-1]), record["seed"])


def diagnostic_flags(candidate):
    flags = []
    if candidate["duration"] < 19.:
        flags.append("Short clip: under 19 seconds")
    if candidate["rms"] < 1e-4:
        flags.append("Near-silent audio: RMS under 0.0001")
    if candidate["clipped_fraction"] >= .01:
        flags.append("At least 1% of samples near clipping")
    return flags


def publish(phase):
    measured = records()
    lookup = {(step, *case(r)): r for step, group in measured.items() for r in group}
    pairs = []
    for row in PROTOCOL["evaluation"]["rows"]:
        for seed in PROTOCOL["evaluation"]["seeds"]:
            takes = []
            for step in [REF, *PROTOCOL["evaluation"]["steps"]]:
                source = folder(step, row, seed)
                if not (source / "checkpoint.json").exists():
                    continue
                wav = next(source.glob("02_slider_*_plus1.wav"))
                dest = PAGE / "audio" / f"step{step}-r{row}-s{seed}.wav"
                mp3 = dest.with_suffix(".mp3")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    os.link(wav, dest)
                assert sha(dest) == sha(wav)
                if not mp3.exists():
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(dest), "-map_metadata", "-1",
                        "-c:a", "libmp3lame", "-b:a", "192k", "-threads", "1", str(mp3)], check=True)
                score = lookup.get((step, row, seed))
                takes.append(dict(step=step, label=f"Step {step:,}" + (" · reference" if step == REF else ""),
                    wav=f"audio/{dest.name}", mp3=f"audio/{mp3.name}", sha256=sha(dest), mp3_sha256=sha(mp3),
                    ce=score["candidate"]["enjoyment"] if score else None,
                    pq=score["candidate"]["production"] if score else None,
                    lyrics=score["candidate"]["lyrics"] if score else None,
                    flags=diagnostic_flags(score["candidate"]) if score else [],
                    measured=score is not None))
            pairs.append(dict(row=row, seed=seed, takes=takes))
    summaries = []
    baseline = {case(r): r for r in measured[REF]}
    for step, group in sorted(measured.items()):
        differences = [r["candidate"]["enjoyment"] - baseline[case(r)]["candidate"]["enjoyment"] for r in group]
        summaries.append(dict(step=step, mean_ce=statistics.mean(r["candidate"]["enjoyment"] for r in group),
            mean_pq=statistics.mean(r["candidate"]["production"] for r in group),
            mean_lyric_agreement=statistics.mean(r["candidate"]["lyrics"] for r in group),
            min_duration_seconds=min(r["candidate"]["duration"] for r in group),
            mean_rms=statistics.mean(r["candidate"]["rms"] for r in group),
            max_clipped_fraction=max(r["candidate"]["clipped_fraction"] for r in group),
            flagged_cases=[dict(row=case(r)[0], seed=r["seed"], flags=diagnostic_flags(r["candidate"]))
                for r in group if diagnostic_flags(r["candidate"])],
            delta_vs_reference=statistics.mean(differences), wins=sum(d > 0 for d in differences), total=len(group),
            paired_deltas=[dict(row=case(r)[0], seed=r["seed"], ce_delta=d) for r, d in zip(group, differences)]))
    training = read(RUN / "status.json", {"status": "queued", "completed": REF, "total": PROTOCOL["end_steps"]})
    status = dict(phase=phase, training=training, updated=time.time(), reference_step=REF,
        rendered=sum(len(p["takes"]) for p in pairs), total_previews=4*(1+len(PROTOCOL["evaluation"]["steps"])))
    write(WORK / "status.json", status); write(PAGE / "status.json", status)
    write(WORK / "summary.json", summaries)
    write(PAGE / "manifest.json", dict(pairs=pairs, summaries=summaries, reference_step=REF,
        start_step=REF, end_step=PROTOCOL["end_steps"]))
    write(PAGE / "measurements.json", {str(k): v for k, v in measured.items()})
    lines = [f"# Fresh-seed Lo-fi: continued listening from step {REF} to {PROTOCOL['end_steps']}", "",
        f"Status: {phase}.", "",
        f"All four monitoring cases receive equal weight. The reference is fresh step {REF}.", "",
        f"| Step | Mean CE | CE change vs {REF} | Wins | Production quality | Lyric agreement | Flagged |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for value in summaries:
        lines.append(f"| {value['step']} | {value['mean_ce']:.6f} | {value['delta_vs_reference']:+.6f} | "
            f"{value['wins']}/{value['total']} | {value['mean_pq']:.6f} | {value['mean_lyric_agreement']:.6f} | "
            f"{len(value['flagged_cases'])} |")
    for value in summaries:
        if value["step"] == REF:
            continue
        lines.extend(["", f"## Step {value['step']}: every paired CE difference", ""])
        for delta in value["paired_deltas"]:
            lines.append(f"- Arrangement {delta['row']-1}, seed {delta['seed']}: {delta['ce_delta']:+.6f}")
        for flagged in value["flagged_cases"]:
            lines.append(f"- Diagnostic flag, arrangement {flagged['row']-1}, seed {flagged['seed']}: "
                + "; ".join(flagged["flags"]))
    lines.extend(["", "These reused monitoring cases do not independently confirm checkpoint selection.",
        "Listening is the main assessment. CE and preservation metrics are diagnostics; a metric dip does not stop training.",
        "Degradation is the object of this run, not a failure of it.",
        "One training prompt and one continued trajectory; style teachers remain fixed.", "",
        "[Listen](http://100.90.104.57:8888/fresh-seed-break-lofi-20260911/)", ""])
    (WORK / "result.md").write_text("\n".join(lines))


def wait_process(process, phase):
    while process.poll() is None:
        publish(phase)
        time.sleep(10)
    if process.returncode:
        raise RuntimeError(f"{phase} failed with exit code {process.returncode}; see retained log")


def audit_step(step):
    measured = records()
    references = {case(r): r for r in measured[REF]}
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


def training_active():
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def await_training():
    """Hold the GPU free until training stops, however it stops."""
    while True:
        status = read(RUN / "status.json", {})
        done, total = status.get("completed", REF), status.get("total", PROTOCOL["end_steps"])
        if status.get("status") == "complete":
            return dict(outcome="complete", completed=done)
        failure = read(RUN / "failure.json")
        if failure:
            return dict(outcome="failed", completed=done, failure=failure)
        if not training_active():
            return dict(outcome="stopped", completed=done,
                note="The training service is no longer running; scoring what exists.")
        publish(f"Training {done:,} / {total:,} — scoring starts when training stops")
        time.sleep(30)


def main():
    PAGE.mkdir(parents=True, exist_ok=True)
    (PAGE / "index.html").write_text((WORK / "page.html").read_text())
    publish("Waiting for training")
    training = await_training()
    available = [s for s in PROTOCOL["evaluation"]["steps"] if weight(s).exists()]
    missing = [s for s in PROTOCOL["evaluation"]["steps"] if s not in available]
    write(WORK / "scoring-scope.json", dict(training=training, render_gpu=RENDER_GPU,
        scoring=available, not_reached=missing,
        note="A training break is the object of this run; every checkpoint that exists is still scored."))
    for step in available:
        for row in PROTOCOL["evaluation"]["rows"]:
            reuse = []
            for seed in PROTOCOL["evaluation"]["seeds"]:
                source, target = folder(REF, row, seed), folder(step, row, seed)
                target.mkdir(parents=True, exist_ok=True)
                for name in ("01_slider_neutral_base_zero.wav", "03_REF_prompt_Lo-fi_no_slider.wav", "04_REF_prompt_Off_no_slider.wav"):
                    if not (target / name).exists():
                        shutil.copyfile(source / name, target / name)
                    assert sha(target / name) == sha(source / name)
                    reuse.append(dict(source=str(source / name), target=str(target / name), sha256=sha(source / name)))
            write(WORK / f"reference-reuse-{step}-row-{row}.json", reuse)
            if all((folder(step, row, seed) / "checkpoint.json").exists() for seed in PROTOCOL["evaluation"]["seeds"]):
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=RENDER_GPU, HF_HUB_OFFLINE="1", HF_HOME="/ml2/music/.cache/huggingface",
                PYTHONPATH=f"{PARENT}/runtime:/ml2/music", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            with (WORK / f"render-{step}-row-{row}.log").open("a") as log:
                process = subprocess.Popen([PY, "-u", str(PARENT / "runtime/analysis/gan_bcap/render_v2.py"),
                    "--weights", str(weight(step)), "--out", str(PAGE / f"step-{step}/row-{row}"),
                    "--prompts", PROTOCOL["evaluation"]["prompts"], "--row", str(row), "--seeds", "101", "303", "--duration", "20"],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                wait_process(process, f"Rendering step {step:,}")
        if step not in records():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_HOME="/ml2/music/.cache/huggingface",
                PYTHONPATH=str(ROOT), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            with (WORK / f"score-{step}.log").open("a") as log:
                process = subprocess.Popen([PY, "-u", str(ROOT / "analysis/uni16_20260906/score.py"), "--id", "lofi", "--folders",
                    *[str(PAGE / f"step-{step}/row-{r}") for r in PROTOCOL["evaluation"]["rows"]],
                    "--output", str(WORK / f"scores-{step}.json")], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                wait_process(process, f"Measuring step {step:,}")
        audit_step(step)
        publish(f"Step {step:,} measured")
    if training["outcome"] == "complete":
        publish(f"Complete — {len(available)} checkpoints scored")
    else:
        publish(f"Training {training['outcome']} at step {training['completed']:,} — "
                f"{len(available)} checkpoints scored, {len(missing)} never reached")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(WORK / "evaluation-failure.json", dict(type=type(error).__name__, message=str(error)))
        publish("Stopped — see experiment log")
        raise
