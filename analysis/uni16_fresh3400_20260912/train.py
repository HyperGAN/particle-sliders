"""Fresh continuations across four prompt pairs, with exact full-state resume."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import os
from pathlib import Path
import random
import signal
import sys
import time

from common import WORK, ROOT, RUNTIME, MODELS, MILESTONES, read, write, sha, digest, seed_for, verify
sys.path[:0] = [str(RUNTIME), str(ROOT.parent)]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from conceptmod.textsliders import train_lm_slider_music3 as legacy
from conceptmod.textsliders.lora import LoRANetwork
from conceptmod.textsliders.gan_v2 import state
from conceptmod.textsliders.gan_v2.critic import SpanCritic
from conceptmod.textsliders.gan_v2.data import prepare_rows, StudentForward, fixture_manifest, check_disjoint
from conceptmod.textsliders.gan_v2.engine import GANEngine
from conceptmod.textsliders.gan_v2.train import arm_settings, export
from histories import HistoryFactory


def same(left, right):
    if torch.is_tensor(left):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(same(a,b) for a,b in zip(left,right))
    return left == right


def main(args):
    manifest = verify()
    item = next(i for i in read(WORK / "catalog.json")["sliders"] if i["id"] == args.id)
    run = args.run_dir or MODELS / item["id"] / f"{item['id']}-fresh3400"
    run.mkdir(parents=True, exist_ok=True)
    source, source_weights = args.source_state.resolve(), args.source_weights.resolve()
    assert 600 < args.until <= 3400
    torch.set_num_threads(4)
    torch.manual_seed(7)
    random.seed(7)
    device = torch.device("cuda:0")
    model_dir = ROOT.parent / "models/MiniMax-Music3"
    rows, metadata = legacy._load_rows(Path(item["train_prompts"]))
    evaluation, _ = legacy._load_rows(Path(item["eval_prompts"]))
    check_disjoint(rows, evaluation)
    selected_rows = item["fresh_rows"]
    write(run / "status.json", dict(status="loading", completed=read(run / "latest.json", {}).get("step", 600),
                                    total=3400, until=args.until, gpu=os.environ["CUDA_VISIBLE_DEVICES"]))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(str(model_dir / "language_model"), torch_dtype=torch.bfloat16,
        local_files_only=True).to(device).eval().requires_grad_(False)
    lm.config.use_cache = False
    prepared = prepare_rows(lm, tokenizer, rows, model_dir=model_dir, cache_dir=Path(item["fixed_cache"]),
                            device=device, frames=250, seeds=[7], policy_stride=4)
    network = LoRANetwork(lm, multiplier=1., rank=8, alpha=8, delimiter="-", target_replace=["Qwen3Attention"],
                          prefix="lora_te", train_method="full").to(device).requires_grad_(True)
    assert len(network.unet_loras) == 144
    recipe, critic_config = arm_settings("baseline", origin=600, horizon=660, diagnostics_every=15)
    assert asdict(recipe) == manifest["continuation"]["recipe"] and critic_config == manifest["continuation"]["critic"]
    critic = SpanCritic(prepared[0]["real"].shape[-1], **critic_config).to(device)
    engine = GANEngine(network, critic, StudentForward(lm, network, device), prepared, recipe)
    sampler = state.RowSampler(len(selected_rows), 1, seed=7)
    (run / "histories").mkdir(parents=True, exist_ok=True)
    factories = {index:HistoryFactory(lm,tokenizer,network,rows[index],prepared[index],
                 run / "histories" / f"row-{index}",model_dir,device) for index in selected_rows}
    # Load all randomly initialized components before restoring RNG. Both first
    # launch and resumed launches reach the saved RNG at the same boundary.
    from diffusers import MiniMaxMusic3RVQDepthDecoder
    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(str(model_dir / "rvq_depth_decoder"),
        torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval().requires_grad_(False)
    for factory in factories.values():
        factory.depth = depth
    replays = []
    for index, factory in factories.items():
        path = legacy._endreg_cache_path(factory.cache,str(model_dir),factory.neutral,250,7+index)
        if not path.exists():
            os.link(prepared[index]["history_cache"],path)
        replay, audit = factory.get(7+index)
        assert all(torch.equal(replay[k], prepared[index][k]) for k in
                   ("real", "neutral_span", "end_teacher", "frame_embeds")), f"Replay changed for row {index}"
        replays.append(dict(row=index, exact=True))
    migration = state.initialize_from_legacy(source, engine, reuse_critic=True)
    assert engine.completed == 600
    blob = torch.load(source,map_location="cpu",weights_only=True)
    assert blob["signature"] == item["warmup_signature"], "Source belongs to another prompt or recipe"
    expected = load_file(str(source_weights))
    initial_checks = dict(network_exact=same(network.state_dict(),expected),
        critic_exact=same(critic.state_dict(),blob["modules"]["critic"]),
        generator_optimizer_exact=same(engine.g_optimizer.state_dict(),blob["optimizers"]["lora"]),
        critic_optimizer_exact=same(engine.d_optimizer.state_dict(),blob["optimizers"]["critic"]),
        python_rng_exact=same(random.getstate(),blob["python_rng"]),
        torch_rng_exact=torch.equal(torch.get_rng_state(),blob["torch_rng"]),
        cuda_rng_exact=same(torch.cuda.get_rng_state_all(),blob["cuda_rng"]))
    assert all(initial_checks.values()), initial_checks
    del blob, expected
    signature = state.signature(recipe,critic_config,fixture_manifest(prepared),rank=8,alpha=8.,
        ema_config=None,model_identity={"path":str(model_dir)},source=migration)
    signature.update(campaign_sha256=sha(WORK / "manifest.json"), slider=item["id"],
        row_indices=selected_rows, batch=1, sampler_seed=7, endpoint=3400,
        source_weights_sha256=sha(source_weights), seed_rule=manifest["continuation"]["seed_rule"])
    history = state.restore(run / "state.pt",engine,sampler,None,signature) if (run / "state.pt").exists() else []
    assert len(history) == engine.completed-600
    if history:
        assert [r["step"] for r in history] == list(range(601,engine.completed+1))
    write(run / "manifest.json",signature)
    write(run / "preflight.json",dict(**initial_checks,replays=replays,resume_step=engine.completed,
        rows=selected_rows, source_state_sha256=migration["sha256"], next_seed=seed_for(item["index"],engine.completed+1)
        if engine.completed < 3400 else None))
    if engine.completed >= args.until:
        write(run / "status.json",dict(status="complete" if engine.completed==3400 else "milestone_complete",
            completed=engine.completed,total=3400,until=args.until))
        return
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig,request_stop)
    seen = {r["history"]["frame_sha256"] for r in history}
    seeds = {r["history"]["seed"] for r in history}
    assert len(seen) == len(seeds) == len(history)
    started,start_step = time.monotonic(),engine.completed

    def save():
        tensors = state.cpu(network.state_dict())
        assert all(torch.isfinite(t).all() for t in tensors.values())
        assert all(torch.isfinite(t).all() for t in critic.state_dict().values())
        outputs = export(run,engine,None,rank=8,alpha=8.,metadata=metadata,
                         prompts=item["train_prompts"],run_signature=signature)
        assert same(tensors,load_file(outputs["live"]))
        state.save(run / "state.pt",engine,sampler,None,signature,history)
        if engine.completed in MILESTONES:
            pinned = run / f"state-step{engine.completed}.pt"
            if pinned.exists():
                assert sha(pinned) == sha(run / "state.pt"), "Milestone state is immutable"
            else:
                os.link(run / "state.pt",pinned)
        counts = dict(Counter(r["prompt_row"] for r in history))
        audit = dict(completed=engine.completed,finite=True,export_matches_state=True,
            weights_sha256=sha(outputs["live"]),state_sha256=sha(run / "state.pt"),
            fresh_updates=len(history),unique_seeds=len(seeds),unique_continuations=len(seen),
            row_counts=counts,all_rows_seen=set(counts)==set(selected_rows),
            max_style_target_change=max((r["history"]["real_target_max_abs_change"] for r in history),default=0.))
        write(run / "checkpoint-audit.json",audit)
        if engine.completed in MILESTONES:
            write(run / f"audit-step{engine.completed}.json",audit)
        write(run / "status.json",dict(status="complete" if engine.completed==3400 else
            "milestone_complete" if engine.completed==args.until else "paused" if stop else "training",
            completed=engine.completed,total=3400,until=args.until,outputs=outputs,row_counts=counts,
            elapsed_this_process=time.monotonic()-started,process_start_step=start_step))

    try:
        with (run / f"updates-from-{engine.completed}-{time.time_ns()}.jsonl").open("w") as log:
            while engine.completed < args.until and not stop:
                update = engine.completed+1
                index = selected_rows[sampler.next()[0]]
                seed = seed_for(item["index"],update)
                write(run / "status.json",dict(status="sampling",completed=engine.completed,total=3400,
                    until=args.until,row=index,next_seed=seed,elapsed_this_process=time.monotonic()-started,
                    process_start_step=start_step))
                current,audit = factories[index].get(seed)
                if audit["frame_sha256"] in seen or seed in seeds:
                    raise RuntimeError("Duplicate fresh continuation; no replacement seed selected")
                current["prompt_index"] = index
                engine.rows = [current]
                update_result = engine.update([0])
                update_result.update(prompt_row=index,history=audit)
                history.append(update_result)
                seen.add(audit["frame_sha256"]);seeds.add(seed)
                log.write(__import__("json").dumps(update_result,allow_nan=False)+"\n");log.flush()
                write(run / "progress.json",update_result)
                print(f"{item['id']}: {engine.completed}/3400 row={index} seed={seed} losses={update_result['losses']}",flush=True)
                if engine.completed % 20 == 0 and engine.completed != args.until:
                    save()
            save()
    except BaseException as exc:
        write(run / "failure.json",dict(type=type(exc).__name__,message=str(exc),completed=engine.completed))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",required=True)
    parser.add_argument("--source-state",type=Path,required=True)
    parser.add_argument("--source-weights",type=Path,required=True)
    parser.add_argument("--until",type=int,required=True)
    parser.add_argument("--run-dir",type=Path)
    main(parser.parse_args())
