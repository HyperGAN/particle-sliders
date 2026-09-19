#!/usr/bin/env python3
"""Reload final LM LoRA files and compare them against pristine caption teachers.

Example (run only on an assigned free GPU):
  CUDA_VISIBLE_DEVICES=0 /home/mikkel/anaconda3/envs/minimax-music3/bin/python \
    analysis/gan_bcap/lm_evaluate.py --weights models/run/run_last.safetensors \
    --output analysis/gan_bcap/lm_evaluation.json

Uses one bf16 model load for all checkpoints, which must have matching LoRA
topology. Caches neutral and positive teachers before attaching any adapter.
Training rows come from the checkpoint's prompt YAML unless explicitly supplied.
Four heldout arrangements and original lyric sheets retain the training concept's
Vocal Details. These hidden-state checks do not evaluate generated audio quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conceptmod.textsliders.train_lm_slider_music3 import (
    DEFAULT_MODEL, _assemble, _assert_last_token_is_audio_start,
    _assert_lyric_span, _encode_full, _gather_last_hidden, _load_rows, _set_scale,
    _tokenize,
)


def resolve_project_file(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute() and (ROOT / path).is_file():
        return (ROOT / path).resolve()
    raise FileNotFoundError(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(weights: Path) -> tuple[dict, Path, int]:
    """Resolve topology and update count without labeling a step file as final.

    The trainer writes metadata only for the final checkpoint. Intermediate
    weights from that completed run share its topology and training card, but
    their completed-update count comes from the checkpoint filename.
    """
    match = re.search(r"_step(\d+)$", weights.stem)
    metadata_path = weights.with_suffix(".json")
    if not metadata_path.exists() and match:
        metadata_path = weights.with_name(weights.stem[:match.start()] + "_last.json")
    metadata = json.loads(metadata_path.read_text())
    steps = int(match.group(1)) if match else int(metadata["steps"])
    if match and not 0 < steps <= int(metadata["steps"]):
        raise ValueError(f"Step checkpoint exceeds completed run in {metadata_path}: {weights}")
    return metadata, metadata_path, steps


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def vocal_details(caption: str) -> str:
    """Retain the actual caption concept while replacing arrangement and lyrics."""
    caption = caption.replace("\\n", "\n")
    marker, end = "Vocal Details:", "Arrangement:"
    if marker not in caption or end not in caption:
        raise ValueError("Built-in heldouts need Vocal Details: and Arrangement: sections; "
                         "supply --heldout-prompts for a different caption format")
    return caption.split(marker, 1)[1].split(end, 1)[0].strip()


def make_heldout_rows(training_row: dict, count: int) -> list[dict]:
    neutral_vocal = vocal_details(str(training_row.get("neutral") or training_row["target"]))
    positive_vocal = vocal_details(str(training_row["positive"]))
    contexts = [
        ("BPM 88. A patient groove in a small rehearsal room. Dry, close recording.",
         "Muted electric guitar chords, rounded bass, brushed snare and a soft shaker.",
         "[verse]\nThe kettle clicks beside the window\nA paper boat rests by the door\n"
         "[chorus]\nWe count the stairs and carry daylight\nAcross the quiet kitchen floor"),
        ("BPM 136. Bright motion and clipped syncopation. A clean, spacious studio mix.",
         "Short synth plucks, a rubbery bass line, tight kick and crisp handclaps.",
         "[verse]\nThe tram rolls past the empty fountain\nMy ticket folds into a square\n"
         "[chorus]\nA silver bell above the crossing\nKeeps little circles in the air"),
        ("BPM 64. Slow, measured phrasing with a warm room sound and a gentle pulse.",
         "Sparse upright piano, bowed low strings and a soft mallet drum.",
         "[verse]\nI leave a spoon beside the teacup\nAnd turn the hallway lamp down low\n"
         "[chorus]\nA little crease inside the curtain\nShows where the evening breezes go"),
        ("BPM 112. Relaxed offbeat motion, a clear center image and short room decay.",
         "Clipped organ chords, damped guitar, fingered bass and a close drum kit.",
         "[verse]\nWe paint a stripe along the doorway\nAnd wipe the color from our hands\n"
         "[chorus]\nThe garden gate swings toward the morning\nA blue wheel settles in the sand"),
    ]
    rows = []
    for metadata, arrangement, lyrics in contexts[:count]:
        def caption(vocal):
            return (f"Global Metadata:\n{metadata}\nVocal Details:\n{vocal}\n"
                    f"Arrangement:\n{arrangement}")
        neutral, positive = caption(neutral_vocal), caption(positive_vocal)
        rows.append(dict(target=neutral, neutral=neutral, positive=positive,
                         negative=neutral, lyrics=lyrics))
    return rows


def delta_metrics(prediction: torch.Tensor, teacher: torch.Tensor,
                  baseline: torch.Tensor, prefix: str) -> dict[str, float]:
    """Measure the adapter-induced change relative to the caption-induced change."""
    predicted = (prediction - baseline).float().reshape(-1)
    target = (teacher - baseline).float().reshape(-1)
    target_norm = target.norm().clamp_min(1e-12)
    return {
        f"{prefix}_delta_cosine": float(F.cosine_similarity(predicted, target, dim=0)),
        f"{prefix}_magnitude_ratio": float(predicted.norm() / target_norm),
        f"{prefix}_normalized_error": float((predicted - target).norm() / target_norm),
        f"{prefix}_parallel_gain": float(predicted.dot(target) / target_norm.square()),
        f"{prefix}_teacher_delta_norm": float(target_norm),
        f"{prefix}_prediction_delta_norm": float(predicted.norm()),
    }


def topology(sidecar: dict) -> dict:
    return dict(rank=int(sidecar["rank"]), alpha=float(sidecar["alpha"]),
                delimiter=str(sidecar.get("delimiter", "-")),
                target_replace=sidecar.get("target_replace", ["Qwen3Attention"]),
                prefix=str(sidecar.get("prefix", "lora_te")),
                train_method=str(sidecar.get("train_method", "full")))


@torch.no_grad()
def cache_teachers(lm, tokenizer, examples: list[dict], device: torch.device):
    cached = []
    for example in examples:
        row = example["row"]
        neutral = _assemble(str(row.get("neutral") or row["target"]), str(row["lyrics"]))
        positive = _assemble(str(row["positive"]), str(row["lyrics"]))
        ni, nm = _tokenize(tokenizer, neutral, device)
        pi, pm = _tokenize(tokenizer, positive, device)
        for label, ids, mask in (("neutral", ni, nm), ("positive", pi, pm)):
            _assert_last_token_is_audio_start(ids, mask, tokenizer,
                                             where=f"{example['id']} {label}")
        ns, ps = _assert_lyric_span(ni, nm, pi, pm, tokenizer, str(row["lyrics"]),
                                   where=example["id"])
        cached.append(dict(
            id=example["id"], group=example["group"],
            neutral_sha256=text_hash(neutral), positive_sha256=text_hash(positive),
            input_ids=ni, attention_mask=nm, positive_mask=pm.cpu(),
            neutral_span=ns.cpu().bool(), positive_span=ps.cpu().bool(),
            neutral_hidden=_encode_full(lm, ni, nm).cpu(),
            positive_hidden=_encode_full(lm, pi, pm).cpu(),
        ))
        print(f"Cached pristine teachers: {example['id']}", flush=True)
    return cached


@torch.no_grad()
def evaluate_checkpoint(lm, network, cached: list[dict], scale: float) -> list[dict]:
    evaluated = []
    for example in cached:
        ni, nm = example["input_ids"], example["attention_mask"]
        neutral, positive = example["neutral_hidden"], example["positive_hidden"]
        _set_scale(network, 0.)
        zero = _encode_full(lm, ni, nm).cpu()
        _set_scale(network, scale)
        prediction = _encode_full(lm, ni, nm).cpu()
        last_neutral = _gather_last_hidden(neutral, nm.cpu())
        last_positive = _gather_last_hidden(positive, example["positive_mask"])
        last_prediction = _gather_last_hidden(prediction, nm.cpu())
        ns, ps = example["neutral_span"], example["positive_span"]
        lyric_neutral, lyric_positive, lyric_prediction = neutral[ns], positive[ps], prediction[ns]
        lyric_difference = lyric_prediction - lyric_neutral
        result = dict(
            id=example["id"], group=example["group"],
            neutral_sha256=example["neutral_sha256"], positive_sha256=example["positive_sha256"],
            lyric_tokens=int(ns.sum()), lyric_tokens_identical=True,
            zero_scale_exact_identity=bool(torch.equal(zero, neutral)),
            zero_scale_max_abs=float((zero - neutral).abs().max()),
            last_hidden_cosine_to_teacher=float(F.cosine_similarity(
                last_prediction, last_positive, dim=-1).mean()),
            lyric_hidden_rms_drift=float(lyric_difference.square().mean().sqrt()),
            lyric_hidden_relative_rms_drift=float(
                lyric_difference.square().mean().sqrt() /
                lyric_neutral.square().mean().sqrt().clamp_min(1e-12)),
            **delta_metrics(last_prediction, last_positive, last_neutral, "last"),
            **delta_metrics(lyric_prediction, lyric_positive, lyric_neutral, "lyric"),
        )
        if not all(torch.isfinite(torch.tensor(value)) for value in result.values()
                   if isinstance(value, float)):
            raise FloatingPointError(f"Non-finite evaluation metric: {example['id']}")
        evaluated.append(result)
        print(json.dumps(result), flush=True)
    _set_scale(network, 0.)
    return evaluated


def summarize(rows: list[dict]) -> dict:
    summaries = {}
    for group in ("train", "heldout"):
        members = [row for row in rows if row["group"] == group]
        if not members:
            continue
        summary = {key: sum(row[key] for row in members) / len(members)
                   for key, value in members[0].items()
                   if isinstance(value, float) and key != "zero_scale_max_abs"}
        summary.update(rows=len(members),
                       zero_scale_exact_identity=all(row["zero_scale_exact_identity"] for row in members),
                       zero_scale_max_abs=max(row["zero_scale_max_abs"] for row in members))
        summaries[group] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--heldout-prompts", type=Path)
    parser.add_argument("--heldout-count", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale", type=float, default=1.)
    args = parser.parse_args()
    began = time.monotonic()
    weights = list(dict.fromkeys(resolve_project_file(path) for path in args.weights))
    resolved_metadata = [checkpoint_metadata(path) for path in weights]
    sidecars = [item[0] for item in resolved_metadata]
    shape = topology(sidecars[0])
    if any(topology(sidecar) != shape for sidecar in sidecars):
        raise ValueError("Checkpoints must share LoRA topology for a single model load")
    if any(sidecar.get("kind") != "language_model" for sidecar in sidecars):
        raise ValueError("Only language-model LoRA checkpoints are supported")
    if args.prompts is None and len({sidecar["prompts_file"] for sidecar in sidecars}) != 1:
        raise ValueError("Checkpoint prompt files differ; supply --prompts explicitly")
    prompts_path = resolve_project_file(args.prompts or sidecars[0]["prompts_file"])
    training_rows, _ = _load_rows(prompts_path)
    if args.heldout_prompts:
        heldout_rows, _ = _load_rows(resolve_project_file(args.heldout_prompts))
        heldout_rows = heldout_rows[:args.heldout_count]
    else:
        heldout_rows = make_heldout_rows(training_rows[0], args.heldout_count)
    examples = [dict(id=f"{group}_{index}", group=group, row=row)
                for group, group_rows in (("train", training_rows), ("heldout", heldout_rows))
                for index, row in enumerate(group_rows)]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from conceptmod.textsliders.lora import LoRANetwork
    device = torch.device(args.device)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir / "tokenizer"), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(str(args.model_dir / "language_model"),
                                           torch_dtype=torch.bfloat16, local_files_only=True)
    lm.to(device).eval().requires_grad_(False)
    lm.config.use_cache = False
    cached = cache_teachers(lm, tokenizer, examples, device)
    network = LoRANetwork(lm, multiplier=0., **shape).to(device)
    network.eval().requires_grad_(False)
    if not network.unet_loras:
        raise RuntimeError("LoRA wrapped no modules")
    report = dict(
        schema=1, model_dir=str(args.model_dir.resolve()),
        model_dtype="bfloat16", adapter_dtype="float32", scale=args.scale,
        prompts_file=str(prompts_path), prompts_sha256=file_hash(prompts_path),
        topology=shape, modules=len(network.unet_loras),
        teacher="Pristine model encoded before attaching any LoRA",
        normalized_error="L2(student - positive) / L2(positive - neutral); neutral baseline is 1",
        lyric_metrics="Aligned identical lyric tokens; teacher matching and neutral-hold drift reported separately",
        limitation="Hidden-state verification only; generated audio and transcription are not evaluated",
        heldout_examples=heldout_rows, results=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, (sidecar, metadata_path, completed_steps) in zip(weights, resolved_metadata):
        if int(sidecar["modules"]) != len(network.unet_loras):
            raise ValueError(f"Sidecar module count mismatch: {path}")
        network.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        rows = evaluate_checkpoint(lm, network, cached, args.scale)
        report["results"].append(dict(
            checkpoint=str(path), checkpoint_sha256=file_hash(path),
            sidecar_sha256=file_hash(metadata_path), sidecar_path=str(metadata_path),
            training_steps=completed_steps, rows=rows, summary=summarize(rows),
        ))
        report["elapsed_seconds"] = time.monotonic() - began
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved checkpoint evaluation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
