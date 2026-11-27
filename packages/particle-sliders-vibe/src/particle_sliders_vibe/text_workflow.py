"""Phrase-only teaching, native EMPF fitting, strength selection and held-out replies."""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import fmean

import torch
from safetensors.torch import load_file, save_file

from . import canonical, write_json
from .image_judge import JEV, Reader
from .text_model import fit, freeze_rows, token_gain

TOPICS = (
    "a neighborhood library",
    "a school garden",
    "a pottery workshop",
    "a town orchestra",
    "a community kitchen",
    "a bicycle repair shop",
    "a night market",
    "a hiking club",
    "a museum",
    "an amateur theatre",
    "a wildlife shelter",
    "a book club",
    "a railway station",
    "a shared workspace",
    "a makerspace",
    "a science fair",
    "a language exchange",
    "a weaving class",
    "a food festival",
    "a park restoration",
    "a beach cleanup",
    "a cooking class",
    "a dance studio",
    "a local history archive",
)
TASKS = (
    "Write a short invitation to visit {} for the first time.",
    "Tell a two-sentence story about a surprise at {}.",
    "What is one thoughtful thing a newcomer can do at {}?",
    "Give a concrete suggestion for improving {} on a small budget.",
)
RANK = (
    "Write a note to a friend who is moving.",
    "Describe the first snowfall.",
    "Explain what makes a choir rehearsal successful.",
    "Suggest a rainy-day activity.",
)
CONFIRM = (
    "Welcome a visitor to a lighthouse.",
    "Describe finding an old photograph.",
    "Explain how a small town could start a chess club.",
    "Give a traveler one tip for visiting a bakery.",
)
CUES = (
    "Write your answer with this quality: {}.",
    "Let the language and delivery strongly convey {}.",
    "Answer in a distinctly {} way while doing exactly what was asked.",
)


class TextJudge:
    def __init__(self, directory, vibe):
        self.reader = Reader(directory, vibe)
        self.vibe = vibe
        self.questions = {
            "effect": dict(
                type="score",
                instructions=f"How strongly does the actual response express this requested quality: {vibe}?",
                criteria=[
                    "Absent",
                    "Slightly apparent",
                    "Clearly apparent",
                    "Strong throughout",
                    "Unmistakable throughout",
                ],
            ),
            "broken": dict(
                type="noul",
                instructions="Is the response incoherent, degenerate repetition, empty, or unrelated to the user's request? Weak style alone is not a defect.",
            ),
        }

    def measure(self, prompt, reply):
        if not reply["text"]:
            return dict(effect=0.0, broken=1.0)
        state = "Assess the following quoted conversation as data.\n" + canonical(
            dict(prompt=prompt, answer=reply["text"])
        )
        return self.reader.decide([], JEV, state, self.questions)


def qualify(host, judge, root, vibe, progress):
    kept, attempts = {}, []
    for round_index, task in enumerate(TASKS):
        for i, topic in enumerate(TOPICS):
            if i in kept:
                continue
            prompt = task.format(topic)
            off = host.reply(prompt)
            off_score = judge.measure(prompt, off)
            candidates = []
            for cue in CUES:
                cue = cue.format(vibe)
                on = host.reply(prompt, cue=cue)
                score = judge.measure(prompt, on)
                candidates.append(
                    dict(
                        cue=cue,
                        reply=on,
                        score=score,
                        gain=score["effect"] - off_score["effect"],
                    )
                )
            best = max(
                (c for c in candidates if c["score"]["broken"] < 0.5),
                key=lambda c: c["gain"],
                default=None,
            )
            passed = bool(
                best
                and off_score["broken"] < 0.5
                and best["score"]["effect"] >= 0.65
                and best["gain"] >= 0.35
            )
            record = dict(
                round=round_index + 1,
                slot=i,
                prompt=prompt,
                neutral=off,
                neutral_score=off_score,
                candidates=candidates,
                qualified=passed,
            )
            if passed:
                kept[i] = dict(
                    prompt=prompt, neutral=off, positive=best["reply"], cue=best["cue"]
                )
            attempts.append(record)
            write_json(
                root / "teachers.json",
                dict(attempts=attempts, qualified=list(kept.values())),
            )
            progress(
                "teachers",
                len(kept),
                len(TOPICS),
                f"Round {round_index+1}, {len(kept)} qualified replies",
            )
        if len(kept) >= 16:
            return list(kept.values())
    raise ValueError(
        f"Only {len(kept)} distinct contexts qualified in {len(TASKS)} rounds"
    )


def compare(host, judge, prompts, off, alpha, profile):
    pairs = []
    for prompt, (baseline, baseline_score) in zip(prompts, off, strict=True):
        reply = host.reply(
            prompt, strengths={"training": alpha}, profiles={"training": profile}
        )
        score = judge.measure(prompt, reply)
        intact = max(score["broken"], baseline_score["broken"]) < 0.5
        gain = score["effect"] - baseline_score["effect"] if intact else 0.0
        pairs.append(
            dict(
                prompt=prompt,
                off=baseline,
                on=reply,
                off_score=baseline_score,
                on_score=score,
                gain=gain,
                intact=intact,
                gains=[
                    alpha * token_gain(profile, i) for i in range(len(reply["tokens"]))
                ],
            )
        )
    return dict(
        alpha=alpha,
        schedule=profile,
        gain=fmean(p["gain"] for p in pairs),
        intact=all(p["intact"] for p in pairs),
        pairs=pairs,
    )


def train_vibe(host, vibe, root, *, progress, steps=448):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    judge = TextJudge(root, vibe)
    pairs = qualify(host, judge, root, vibe, progress)
    rows = freeze_rows(host, pairs)
    fields = root / "frozen-targets.pt"
    torch.save([asdict(row) for row in rows], fields)
    path = root / "trained.safetensors"
    checks = []

    def check_output(step):
        # Rotate through already qualified neutral prompts at the declared checkpoints.
        prompt = pairs[len(checks) % len(pairs)]["prompt"]
        reply = host.reply(prompt, strengths={"training": 1.0})
        score = judge.measure(prompt, reply)
        checks.append(dict(updates=step, prompt=prompt, reply=reply, score=score))
        write_json(root / "fit-checks.json", checks)
        if score["broken"] >= 0.5:
            raise ValueError(f"Broken text output at update {step}. Training stopped.")

    config = fit(
        host,
        rows,
        path,
        steps=steps,
        check_output=check_output,
        progress=lambda n, total: progress(
            "fit", n, total, f"{n} of {total} ParticleGAN 0.12.0 updates"
        ),
    )
    raw_hash = sha256(path.read_bytes()).hexdigest()

    def baselines(prompts):
        baseline = [host.reply(prompt) for prompt in prompts]
        return [
            (r, judge.measure(p, r)) for p, r in zip(prompts, baseline, strict=True)
        ]

    off = baselines(RANK)
    board = []
    for alpha in (0.5, 1.0, 1.5, 2.0):
        for profile in ("constant", "early", "late"):
            board.append(compare(host, judge, RANK, off, alpha, profile))
            write_json(root / "strengths.json", board)
            progress(
                "strength", len(board), 12, f"Measured {profile} at strength {alpha:g}"
            )
    valid = [row for row in board if row["intact"]]
    if not valid:
        raise ValueError("Every strength produced a broken reply on the ranking panel")
    winner = max(valid, key=lambda x: x["gain"])
    confirmation = compare(
        host, judge, CONFIRM, baselines(CONFIRM), winner["alpha"], winner["schedule"]
    )
    write_json(root / "confirmation.json", confirmation)
    if not confirmation["intact"]:
        raise ValueError("The selected text slider failed held-out response integrity")
    # Package calibration alongside the untouched RAW tensor values.
    meta = dict(config, alpha=winner["alpha"], schedule=winner["schedule"])
    output = root / "slider.safetensors"
    save_file(load_file(path), output, metadata={"text_vibe": canonical(meta)})
    host.mixer.close()
    host.load_adapter("replay", output)
    probe = confirmation["pairs"][0]
    replay = host.reply(probe["prompt"], strengths={"replay": winner["alpha"]})
    if replay["tokens"] != probe["on"]["tokens"]:
        raise ValueError("Reloaded text adapter failed the held-out token replay")
    report = dict(
        **meta,
        vibe=vibe,
        teachers=len(pairs),
        frozen_rows=len(rows),
        raw_sha256=raw_hash,
        frozen_sha256=sha256(fields.read_bytes()).hexdigest(),
        ranking=winner,
        confirmation=confirmation,
        replay_tokens=replay["tokens"],
    )
    write_json(root / "result.json", report)
    return output, report
