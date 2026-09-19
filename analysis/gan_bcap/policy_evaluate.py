#!/usr/bin/env python3
"""Measure predictive-policy KL on fixed audio continuations, without training.

The policies are raw conditional next-token distributions at temperature 1
over legal semantic audio codes plus EOS. They precede CFG/top-k and do not
include depth-code predictions or the flow transformer. This is a diagnostic
of teacher agreement, not an audio-quality score or full sequence KL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from lm_evaluate import checkpoint_metadata, file_hash, make_heldout_rows, topology
from conceptmod.textsliders.train_lm_slider_music3 import (
    DEFAULT_MODEL, _assemble, _assert_last_token_is_audio_start, _load_rows,
    _preroll_frames, _set_scale, _tokenize,
)


def policy_distances(student_logits, teacher_logits, neutral_logits):
    """Per-position nats. The final logit column is always audio EOS."""
    s, t, n = [F.log_softmax(x.double(), dim=-1) for x in
               (student_logits, teacher_logits, neutral_logits)]
    ps, pt, pn = s.exp(), t.exp(), n.exp()
    mixture = torch.logaddexp(s, t) - math.log(2.)
    forward = (pt * (t-s)).sum(-1).clamp_min(0)
    reverse = (ps * (s-t)).sum(-1).clamp_min(0)
    base = (pt * (t-n)).sum(-1).clamp_min(0)
    return dict(teacher_to_student_kl=forward, student_to_teacher_kl=reverse,
                teacher_to_neutral_kl=base,
                teacher_student_js=(.5*((pt*(t-mixture)).sum(-1)+(ps*(s-mixture)).sum(-1))).clamp_min(0),
                neutral_to_student_kl=(pn*(n-s)).sum(-1).clamp_min(0),
                student_entropy=-(ps*s).sum(-1), teacher_entropy=-(pt*t).sum(-1),
                student_eos_probability=ps[:, -1], teacher_eos_probability=pt[:, -1],
                neutral_eos_probability=pn[:, -1],
                teacher_kl_improvement=base-forward)


def summarize_positions(values):
    output = {}
    for segment, selection in (("audio_start", slice(0, 1)), ("continuation", slice(1, None))):
        chunk = {key: value[selection] for key, value in values.items()}
        if not next(iter(chunk.values())).numel():
            continue
        output[segment] = {key: float(value.mean()) for key, value in chunk.items()}
        output[segment]["positions"] = next(iter(chunk.values())).numel()
        output[segment]["teacher_to_student_kl_p95"] = float(torch.quantile(chunk["teacher_to_student_kl"], .95))
    return output


@torch.no_grad()
def policy_logits(lm, ids, frames, legal_weight, legal_bias):
    prompt = lm.model.embed_tokens(ids)
    embeddings = prompt if frames is None else torch.cat((prompt, frames), dim=1)
    hidden = lm.model(inputs_embeds=embeddings,
                      attention_mask=torch.ones(embeddings.shape[:2], dtype=torch.long, device=ids.device),
                      use_cache=False).last_hidden_state
    tail = hidden[0, prompt.shape[1]-1:]
    logits = F.linear(tail, legal_weight, legal_bias).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite policy logits")
    return logits


def grouped_summary(records):
    output = {}
    for group in sorted({record["group"] for record in records}):
        output[group] = {}
        for origin in ("neutral", "positive"):
            members = [r for r in records if r["group"] == group and r["history_origin"] == origin]
            output[group][origin] = {}
            for segment in ("audio_start", "continuation"):
                values = [r["metrics"][segment] for r in members if segment in r["metrics"]]
                if values:
                    output[group][origin][segment] = {
                        key: statistics.mean(row[key] for row in values) for key in values[0] if key != "positions"}
                    output[group][origin][segment]["examples"] = len(values)
    return output


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=ROOT / "conceptmod/textsliders/data/prompts-gender-uni-v2.yaml")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "policy_cache")
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float,
                        help="optional per-process CUDA allocator limit when sharing the assigned GPU")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from diffusers import MiniMaxMusic3RVQDepthDecoder
    from diffusers.modular_pipelines.minimax_music3 import encoders
    from safetensors.torch import load_file
    from conceptmod.textsliders.lora import LoRANetwork
    started = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.memory_fraction is not None:
        if not 0 < args.memory_fraction <= 1:
            raise ValueError("--memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    weights = list(dict.fromkeys(path.resolve() for path in args.weights))
    metadata = [checkpoint_metadata(path) for path in weights]
    shape = topology(metadata[0][0])
    if any(topology(meta[0]) != shape for meta in metadata):
        raise ValueError("Checkpoint topologies differ")
    training_rows, _ = _load_rows(args.prompts)
    heldout_rows = make_heldout_rows(training_rows[0], 4)
    examples = [("train", i, row) for i, row in enumerate(training_rows)] if args.include_train else []
    examples += [("heldout", i, row) for i, row in enumerate(heldout_rows)]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir / "tokenizer"), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(str(args.model_dir / "language_model"),
                                           torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    lm.requires_grad_(False)
    lm.config.use_cache = False
    ids = list(range(encoders._AUDIO_CODE_OFFSET,
                     encoders._AUDIO_CODE_OFFSET + encoders._SEMANTIC_VOCAB_SIZE)) + [encoders._AUDIO_END_TOKEN_ID]
    legal_ids = torch.tensor(ids, device=device)
    legal_weight = lm.lm_head.weight[legal_ids]
    legal_bias = lm.lm_head.bias[legal_ids] if lm.lm_head.bias is not None else None
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    decoder = None
    cached = []
    for group, index, row in examples:
        neutral = _assemble(str(row.get("neutral") or row["target"]), row["lyrics"])
        positive = _assemble(row["positive"], row["lyrics"])
        ni, nm = _tokenize(tokenizer, neutral, device)
        pi, pm = _tokenize(tokenizer, positive, device)
        _assert_last_token_is_audio_start(ni, nm, tokenizer, where=f"{group}_{index} neutral")
        _assert_last_token_is_audio_start(pi, pm, tokenizer, where=f"{group}_{index} positive")
        ni, pi = ni[:, nm[0].bool()], pi[:, pm[0].bool()]
        for seed in args.seeds:
            for origin, text, prompt_ids in (("neutral", neutral, ni), ("positive", positive, pi)):
                spec = dict(model=str(args.model_dir.resolve()), prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            frames=args.frames, seed=seed, encoders_sha256=file_hash(Path(encoders.__file__)),
                            trainer_sha256=file_hash(ROOT / "conceptmod/textsliders/train_lm_slider_music3.py"))
                digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
                cache_path = args.cache_dir / f"{digest}.pt"
                if cache_path.exists():
                    blob = torch.load(cache_path, map_location="cpu", weights_only=True)
                    if blob["spec"] != spec:
                        raise ValueError("Trajectory cache provenance mismatch")
                else:
                    if decoder is None:
                        decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
                            str(args.model_dir / "rvq_depth_decoder"), torch_dtype=torch.bfloat16,
                            local_files_only=True).to(device).eval().requires_grad_(False)
                    print(f"Compose pristine {group}_{index}, {origin}, seed {seed}", flush=True)
                    frames, ended = _preroll_frames(lm, decoder, prompt_ids, args.frames, seed, device)
                    blob = dict(spec=spec, frames=None if frames is None else frames.cpu(), ended=ended)
                    temporary = cache_path.with_suffix(".tmp")
                    torch.save(blob, temporary)
                    temporary.replace(cache_path)
                frames = blob["frames"].to(device) if blob["frames"] is not None else None
                teacher = policy_logits(lm, pi, frames, legal_weight, legal_bias).cpu()
                baseline = policy_logits(lm, ni, frames, legal_weight, legal_bias).cpu()
                cached.append(dict(id=f"{group}_{index}_{origin}_s{seed}", group=group, row=index,
                                   history_origin=origin, seed=seed, ids=ni.cpu(), frames=blob["frames"],
                                   teacher=teacher, baseline=baseline, ended_naturally=blob["ended"],
                                   cache_path=str(cache_path), cache_sha256=file_hash(cache_path)))
    if decoder is not None:
        del decoder
    torch.cuda.empty_cache()
    network = LoRANetwork(lm, multiplier=0., **shape).to(device).eval().requires_grad_(False)
    if not network.unet_loras:
        raise ValueError("No LoRA modules attached")
    report = dict(schema=1, definition="KL(positive-caption teacher || neutral-caption adapter), nats per prediction",
                  policy="Conditional semantic codes + EOS at T=1, before CFG and top-k; depth/flow predictions excluded",
                  history="Fixed pristine neutral and positive rollouts, shared across all checkpoints; neither is a student rollout",
                  aggregation="Mean per position within an example, then equal weight per row/seed within each history origin",
                  limitations=["Not full sequence KL or audio quality", "Only the semantic head is scored; vocal timbre can also live in depth/flow pathways",
                               "Previously used heldouts are development validation", "No candidate selected by these numbers alone"],
                  model_dir=str(args.model_dir), prompts_sha256=file_hash(args.prompts),
                  frames_cap=args.frames, seeds=args.seeds, heldout_examples=heldout_rows,
                  evaluator_sha256=file_hash(Path(__file__)), results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, (meta, metadata_path, steps) in zip(weights, metadata):
        network.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        records = []
        for example in cached:
            ni = example["ids"].to(device)
            frames = example["frames"].to(device) if example["frames"] is not None else None
            _set_scale(network, 0.)
            zero = policy_logits(lm, ni, frames, legal_weight, legal_bias).cpu()
            exact = torch.equal(zero, example["baseline"])
            if not exact:
                raise AssertionError(f"Slider-off policy changed: {example['id']}")
            _set_scale(network, 1.)
            predicted = policy_logits(lm, ni, frames, legal_weight, legal_bias).cpu()
            values = policy_distances(predicted, example["teacher"], example["baseline"])
            record = {key: value for key, value in example.items() if key not in {"ids", "frames", "teacher", "baseline"}}
            record.update(zero_scale_exact_identity=exact, metrics=summarize_positions(values),
                          per_position={key: value.tolist() for key, value in values.items()})
            records.append(record)
        result = dict(checkpoint=str(path), checkpoint_sha256=file_hash(path), steps=steps,
                      sidecar_path=str(metadata_path), sidecar_sha256=file_hash(metadata_path),
                      rows=records, summary=grouped_summary(records))
        report["results"].append(result)
        report["elapsed_seconds"] = time.monotonic()-started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(path.name, json.dumps(result["summary"]), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
