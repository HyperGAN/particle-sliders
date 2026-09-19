"""Continue fresh histories from step 1000 to probe audible degradation."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
PARENT = ROOT / "analysis/fresh_seed_20260910"
spec = importlib.util.spec_from_file_location("fresh_history_parent", PARENT / "train.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
torch, state = base.torch, base.state


def history_seed(update):
    if update < 1001:
        raise ValueError("This continuation starts at update 1001")
    return 1_000_001 + update - 601


def same(left, right):
    if torch.is_tensor(left):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def main():
    protocol = json.loads((WORK / "protocol.json").read_text())
    start, end = protocol["source_steps"], protocol["end_steps"]
    run = WORK / "lofi-fresh-stress"
    run.mkdir(exist_ok=True)
    source = Path(protocol["source_state"])
    assert base.sha(source) == protocol["source_state_sha256"]
    parent_signature = json.loads((source.parent / "manifest.json").read_text())
    for rel, expected in json.loads((PARENT / "runtime-snapshot.json").read_text()).items():
        assert base.sha(base.RUNTIME / rel) == expected, rel
    assert state.code_fingerprints() == parent_signature["sources"]
    torch.set_num_threads(4)
    torch.manual_seed(7)
    random.seed(7)
    device = torch.device("cuda:0")
    model_dir = Path("/ml2/music/models/MiniMax-Music3")
    rows, metadata = base.legacy._load_rows(Path(protocol["training_prompt"]))
    evaluation, _ = base.legacy._load_rows(Path(protocol["evaluation"]["prompts"]))
    assert len(rows) == 1
    base.check_disjoint(rows, evaluation)
    base.write_json(run / "status.json", dict(status="loading", completed=start, total=end))
    tokenizer = base.AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), local_files_only=True)
    lm = base.AutoModelForCausalLM.from_pretrained(str(model_dir / "language_model"),
        torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    lm.config.use_cache = False
    prepared = base.prepare_rows(lm, tokenizer, rows, model_dir=model_dir, cache_dir=ROOT / "cache/endreg",
        device=device, frames=250, seeds=[7], policy_stride=4)
    network = base.LoRANetwork(lm, multiplier=1., rank=8, alpha=8, delimiter="-",
        target_replace=["Qwen3Attention"], prefix="lora_te", train_method="full").to(device)
    network.requires_grad_(True)
    # Keep the original schedule parameters: its constant rate extends unchanged.
    recipe, critic_config = base.arm_settings("baseline", origin=600, horizon=660, diagnostics_every=15)
    assert asdict(recipe) == parent_signature["recipe"] and recipe.schedule == "constant"
    assert critic_config == parent_signature["critic"]
    assert base.fixture_manifest(prepared) == parent_signature["fixtures"]
    critic = base.SpanCritic(prepared[0]["real"].shape[-1], **critic_config).to(device)
    engine = base.GANEngine(network, critic, base.StudentForward(lm, network, device), prepared, recipe)
    sampler = state.RowSampler(1, 1, seed=7)
    factory = base.HistoryFactory(lm, tokenizer, network, rows[0], prepared[0], run / "histories", model_dir, device)
    replay_path = base.legacy._endreg_cache_path(factory.cache, str(model_dir), factory.neutral, 250, 7)
    if not replay_path.exists():
        os.link(Path(prepared[0]["history_cache"]), replay_path)
    replay, replay_audit = factory.get(7)
    for field in ("real", "neutral_span", "end_teacher", "frame_embeds"):
        assert torch.equal(replay[field], prepared[0][field]), field
    # Construct every module before restoring RNG, including the depth decoder.
    from diffusers import MiniMaxMusic3RVQDepthDecoder
    factory.depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        str(model_dir / "rvq_depth_decoder"), torch_dtype=torch.bfloat16,
        local_files_only=True).to(device).eval().requires_grad_(False)
    history = state.restore(source, engine, sampler, None, parent_signature)
    assert engine.completed == start and len(history) == start - 600
    blob = torch.load(source, map_location="cpu", weights_only=True)
    from safetensors.torch import load_file
    expected_weights = load_file(protocol["source_weights"])
    checks = dict(
        source_network_exact=same(state.cpu(network.state_dict()), expected_weights),
        source_critic_exact=same(state.cpu(critic.state_dict()), blob["critic"]),
        generator_optimizer_exact=same(state.cpu(engine.g_optimizer.state_dict()), blob["g_optimizer"]),
        critic_optimizer_exact=same(state.cpu(engine.d_optimizer.state_dict()), blob["d_optimizer"]),
        python_rng_exact=same(random.getstate(), blob["python_rng"]),
        torch_rng_exact=torch.equal(torch.get_rng_state(), blob["torch_rng"]),
        cuda_rng_exact=same(torch.cuda.get_rng_state_all(), blob["cuda_rng"]),
        fixed_history_replay_exact=True, original_runtime_unchanged=True,
    )
    assert all(checks.values()), checks
    del blob, expected_weights
    signature = dict(parent_signature, experiment=protocol, experiment_code_sha256=base.sha(__file__),
        continuation_parent=dict(state=str(source), sha256=base.sha(source),
            manifest_sha256=base.sha(source.parent / "manifest.json"), completed=start))
    if (run / "state.pt").exists():
        history = state.restore(run / "state.pt", engine, sampler, None, signature)
    base.write_json(run / "manifest.json", signature)
    base.write_json(run / "preflight.json", dict(**checks, resume_step=engine.completed,
        next_seed=history_seed(engine.completed + 1), replay=replay_audit))
    if engine.completed >= end:
        base.write_json(run / "status.json", dict(status="complete", completed=engine.completed, total=end))
        return
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    started, start_step = time.monotonic(), engine.completed
    seen = {r["history"]["frame_sha256"] for r in history}
    assert len(seen) == len(history)

    def save():
        tensors = state.cpu(network.state_dict())
        assert all(torch.isfinite(value).all() for value in tensors.values())
        outputs = base.export(run, engine, None, rank=8, alpha=8., metadata=metadata,
            prompts=protocol["training_prompt"], run_signature=signature)
        state.save(run / "state.pt", engine, sampler, None, signature, history)
        if engine.completed in protocol["evaluation"]["steps"]:
            pinned = run / f"state-step{engine.completed}.pt"
            if not pinned.exists():
                os.link(run / "state.pt", pinned)
        base.write_json(run / "status.json", dict(status="complete" if engine.completed == end else "training",
            completed=engine.completed, total=end, outputs=outputs,
            elapsed_this_process=time.monotonic()-started, process_start_step=start_step))
        if engine.completed == end:
            loaded = load_file(outputs["live"])
            assert same(tensors, loaded)
            base.write_json(run / "completion-audit.json", dict(completed=engine.completed,
                additional_updates=end-start, total_fresh_updates=len(history), unique_seeds=len({r["history"]["seed"] for r in history}),
                unique_continuations=len(seen), all_tensors_finite=True, export_matches_full_state=True,
                max_style_target_change=max(r["history"]["real_target_max_abs_change"] for r in history)))

    try:
        with (run / f"updates-from-{engine.completed}-{time.time_ns()}.jsonl").open("w") as log:
            while engine.completed < end and not stop:
                update, seed = engine.completed + 1, history_seed(engine.completed + 1)
                base.write_json(run / "status.json", dict(status="sampling", completed=engine.completed, total=end,
                    next_history_seed=seed, elapsed_this_process=time.monotonic()-started, process_start_step=start_step))
                current, audit = factory.get(seed)
                if audit["frame_sha256"] in seen:
                    raise RuntimeError("Duplicate continuation; do not silently substitute another seed")
                engine.rows = [current]
                row = engine.update([0])
                row["history"] = audit
                history.append(row)
                seen.add(audit["frame_sha256"])
                log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
                base.write_json(run / "progress.json", row)
                print(f"fresh: {engine.completed}/{end}, seed={seed}, losses={row['losses']}", flush=True)
                if engine.completed % 20 == 0 or engine.completed in protocol["evaluation"]["steps"]:
                    save()
            save()
            if stop and engine.completed < end:
                base.write_json(run / "status.json", dict(status="paused", completed=engine.completed, total=end))
    except BaseException as error:
        base.write_json(run / "failure.json", dict(type=type(error).__name__, message=str(error), completed=engine.completed))
        raise


if __name__ == "__main__":
    main()
