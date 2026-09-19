"""One-prompt, fixed-versus-fresh history continuation of the exact UNI16 state."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
RUNTIME = WORK / "runtime"
sys.path[:0] = [str(RUNTIME), str(ROOT.parent)]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from conceptmod.textsliders import train_lm_slider_music3 as legacy
from conceptmod.textsliders.lora import LoRANetwork
from conceptmod.textsliders.gan_v2 import state
from conceptmod.textsliders.gan_v2.critic import SpanCritic
from conceptmod.textsliders.gan_v2.data import (
    allowed_logits, check_disjoint, digest, fixture_manifest, gather, prepare_rows, sha,
)
from conceptmod.textsliders.gan_v2.data import StudentForward
from conceptmod.textsliders.gan_v2.engine import GANEngine
from conceptmod.textsliders.gan_v2.train import arm_settings, export, write_json


def history_seed(mode, update):
    if not 601 <= update <= 660:
        raise ValueError("Outside the locked 60-update continuation")
    return 7 if mode == "fixed" else 1_000_001 + update - 601


class HistoryFactory:
    def __init__(self, lm, tokenizer, network, row, initial, cache, model_dir, device):
        self.lm, self.network, self.row = lm, network, row
        self.initial, self.cache, self.model_dir, self.device = initial, cache, model_dir, device
        self.cache.mkdir(exist_ok=True)
        self.neutral = legacy._assemble(row.get("neutral") or row["target"], row["lyrics"])
        positive = legacy._assemble(row["positive"], row["lyrics"])
        self.nt, nm = legacy._tokenize(tokenizer, self.neutral, device)
        pt, pm = legacy._tokenize(tokenizer, positive, device)
        self.ns, ps = legacy._assert_lyric_span(self.nt, nm, pt, pm, tokenizer, row["lyrics"], where="fresh-history")
        self.ps = ps
        with torch.no_grad():
            self.ne = lm.model.embed_tokens(self.nt)
            self.pe = lm.model.embed_tokens(pt)
        self.depth = None

    @torch.no_grad()
    def get(self, seed):
        legacy._set_scale(self.network, 0.)
        path = legacy._endreg_cache_path(self.cache, str(self.model_dir), self.neutral, 250, seed)
        started = time.monotonic()
        if path.exists():
            blob = torch.load(path, map_location="cpu", weights_only=True)
        else:
            if self.depth is None:
                from diffusers import MiniMaxMusic3RVQDepthDecoder
                self.depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
                    str(self.model_dir / "rvq_depth_decoder"), torch_dtype=torch.bfloat16,
                    local_files_only=True,
                ).to(self.device).eval().requires_grad_(False)
            frames, ended = legacy._preroll_frames(self.lm, self.depth, self.nt, 250, seed, self.device)
            if frames is None:
                raise RuntimeError("Fresh history ended before its first frame; retain seed and investigate")
            blob = dict(frame_embeds=frames.cpu(), ended=bool(ended))
            temporary = path.with_suffix(".tmp")
            torch.save(blob, temporary)
            temporary.replace(path)
        frames = blob["frame_embeds"].to(self.device)
        _, end_teacher, nh = legacy._forward_teacher_forced(self.lm, self.ne, frames)
        _, _, ph = legacy._forward_teacher_forced(self.lm, self.pe, frames)
        neutral_span = gather(nh[:, :self.ne.shape[1]], self.ns)
        positive_span = gather(ph[:, :self.pe.shape[1]], self.ps)
        real = positive_span - neutral_span
        item = dict(
            prompt_index=0, history_seed=seed, prompt_hash=digest(self.row),
            lyric_hash=digest(self.row["lyrics"]), fixture=digest([self.row, seed, 250, 4]),
            prompt_embeds=self.ne.cpu(), frame_embeds=frames.cpu(),
            neutral_span=neutral_span.cpu(), condition=neutral_span.cpu(), real=real.cpu(),
            span_mask=self.ns.cpu(), end_teacher=end_teacher.cpu(),
            policy_teacher=allowed_logits(self.lm, ph[:, self.pe.shape[1]-1:])[0][::4].cpu(),
            continuation_teacher=ph[:, self.pe.shape[1]-1::4].float().cpu(), policy_stride=4,
            ended=bool(blob["ended"]), history_cache=str(path), history_cache_sha256=sha(path),
        )
        frame_hash = sha256(frames.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        audit = dict(seed=seed, frames=frames.shape[1], frame_sha256=frame_hash,
            cache_sha256=item["history_cache_sha256"], sampling_and_teacher_seconds=time.monotonic()-started,
            real_target_max_abs_change=float((item["real"]-self.initial["real"]).abs().max()),
            neutral_span_max_abs_change=float((item["neutral_span"]-self.initial["neutral_span"]).abs().max()))
        if item["end_teacher"].shape == self.initial["end_teacher"].shape:
            audit["ending_target_rms_change"] = float((item["end_teacher"]-self.initial["end_teacher"]).square().mean().sqrt())
        legacy._set_scale(self.network, 1.)
        return item, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["fixed", "fresh"], required=True)
    args = parser.parse_args()
    protocol = json.loads((WORK / "protocol.json").read_text())
    run = WORK / f"lofi-{args.arm}-history"
    run.mkdir(exist_ok=True)
    if (run / "status.json").exists() and json.loads((run / "status.json").read_text()).get("status") == "complete":
        print("Already complete", run, flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(7)
    random.seed(7)
    device = torch.device("cuda:0")
    source = Path(protocol["source_state"])
    assert sha(source) == protocol["source_state_sha256"]
    for rel, expected in json.loads((WORK / "runtime-snapshot.json").read_text()).items():
        assert sha(RUNTIME / rel) == expected, f"Frozen source changed: {rel}"
    rows, metadata = legacy._load_rows(Path(protocol["training_prompt"]))
    evaluation, _ = legacy._load_rows(Path(protocol["evaluation"]["prompts"]))
    assert len(rows) == 1
    check_disjoint(rows, evaluation)
    model_dir = Path("/ml2/music/models/MiniMax-Music3")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), local_files_only=True)
    write_json(run / "status.json", dict(status="loading", arm=args.arm, completed=600, total=660))
    lm = AutoModelForCausalLM.from_pretrained(str(model_dir / "language_model"), torch_dtype=torch.bfloat16,
        local_files_only=True).to(device).eval().requires_grad_(False)
    lm.config.use_cache = False
    prepared = prepare_rows(lm, tokenizer, rows, model_dir=model_dir, cache_dir=ROOT / "cache/endreg",
        device=device, frames=250, seeds=[7], policy_stride=4)
    network = LoRANetwork(lm, multiplier=1., rank=8, alpha=8, delimiter="-",
        target_replace=["Qwen3Attention"], prefix="lora_te", train_method="full").to(device)
    network.requires_grad_(True)
    recipe, critic_config = arm_settings("baseline", origin=600, horizon=660, diagnostics_every=15)
    critic = SpanCritic(prepared[0]["real"].shape[-1], **critic_config).to(device)
    engine = GANEngine(network, critic, StudentForward(lm, network, device), prepared, recipe)
    migration = state.initialize_from_legacy(source, engine, reuse_critic=True)
    initial_network = state.cpu(network.state_dict())
    from safetensors.torch import load_file
    expected_weights = load_file(str(source.parent / "lofi-warmup600-a02_last.safetensors"))
    assert set(expected_weights) == set(initial_network)
    assert all(torch.equal(initial_network[k], expected_weights[k]) for k in expected_weights)
    signature = state.signature(recipe, critic_config, fixture_manifest(prepared), rank=8, alpha=8.,
        ema_config=None, model_identity={"path": str(model_dir)}, source=migration)
    signature.update(experiment=protocol, experiment_arm=args.arm, experiment_code_sha256=sha(__file__),
        runtime_manifest_sha256=sha(WORK / "runtime-snapshot.json"))
    sampler = state.RowSampler(1, 1, seed=7)
    history = []
    if (run / "state.pt").exists():
        history = state.restore(run / "state.pt", engine, sampler, None, signature)
    write_json(run / "manifest.json", signature)
    factory = HistoryFactory(lm, tokenizer, network, rows[0], prepared[0], run / "histories", model_dir, device)
    # Replay the original frozen history through the new builder before training.
    original_cache = Path(prepared[0]["history_cache"])
    replay_path = legacy._endreg_cache_path(factory.cache, str(model_dir), factory.neutral, 250, 7)
    if not replay_path.exists():
        os.link(original_cache, replay_path)
    replay, replay_audit = factory.get(7)
    for field in ("real", "neutral_span", "end_teacher", "frame_embeds"):
        assert torch.equal(replay[field], prepared[0][field]), f"Fixed-history replay changed {field}"
    restored_weights = network.state_dict()
    if not history:
        assert all(torch.equal(restored_weights[k].cpu(), expected_weights[k]) for k in expected_weights)
    write_json(run / "preflight.json", dict(source_weights_exact=True, fixed_history_replay_exact=True,
        original_training_sources_match=True, adapter_disabled_for_teacher_sampling=True, replay=replay_audit))
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    started = time.monotonic()
    seen = {r["history"]["frame_sha256"] for r in history if args.arm == "fresh"}
    def save():
        outputs = export(run, engine, None, rank=8, alpha=8., metadata=metadata,
            prompts=protocol["training_prompt"], run_signature=signature)
        state.save(run / "state.pt", engine, sampler, None, signature, history)
        write_json(run / "status.json", dict(status="complete" if engine.completed == 660 else "paused",
            arm=args.arm, completed=engine.completed, total=660, outputs=outputs,
            elapsed_this_process=time.monotonic()-started))
    try:
        with (run / f"updates-from-{engine.completed}-{time.time_ns()}.jsonl").open("w") as log:
            while engine.completed < 660 and not stop:
                update = engine.completed + 1
                seed = history_seed(args.arm, update)
                write_json(run / "status.json", dict(status="sampling" if args.arm == "fresh" else "training",
                    arm=args.arm, completed=engine.completed, total=660, next_history_seed=seed))
                if args.arm == "fixed":
                    current, audit = replay, replay_audit
                else:
                    print(f"Sampling update {update}, unchanged prompt, fresh seed {seed}", flush=True)
                    current, audit = factory.get(seed)
                    if audit["frame_sha256"] in seen:
                        raise RuntimeError("A fresh seed produced a duplicate continuation; do not silently resample")
                    seen.add(audit["frame_sha256"])
                engine.rows = [current]
                row = engine.update([0])
                row["history"] = audit
                history.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(f"{args.arm}: update {engine.completed}/660, seed={seed}, losses={row['losses']}", flush=True)
                write_json(run / "progress.json", row)
                if engine.completed % 10 == 0:
                    save()
        save()
    except BaseException as error:
        # Record failures without converting an unsuccessful update into a checkpoint.
        write_json(run / "failure.json", dict(type=type(error).__name__, message=str(error), completed=engine.completed))
        raise


if __name__ == "__main__":
    main()
