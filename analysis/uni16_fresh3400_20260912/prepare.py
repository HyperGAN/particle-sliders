"""Validate the complete 16-slider campaign before any GPU work."""
from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import time

import yaml

from common import WORK, ROOT, RUNTIME, MODELS, PAGE, read, write, sha, digest, warmup_args, verify


def main(rows_mode):
    if (WORK / "manifest.json").exists():
        manifest = verify()
        if manifest["fresh_rows_mode"] != rows_mode:
            raise ValueError("Prepared row selection differs; preserve the frozen campaign")
        print("Existing campaign verified")
        return
    parent = ROOT / "analysis/fresh_seed_20260910"
    runtime_hashes = read(parent / "runtime-snapshot.json")
    for rel, expected in runtime_hashes.items():
        if sha(parent / "runtime" / rel) != expected:
            raise ValueError(f"Original frozen runtime changed: {rel}")
    if not RUNTIME.exists():
        shutil.copytree(parent / "runtime", RUNTIME, ignore=shutil.ignore_patterns("__pycache__"))
    for rel, expected in runtime_hashes.items():
        assert sha(RUNTIME / rel) == expected
    # Vendor the exact audited history builder, retaining its original algorithm.
    source = (parent / "train.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "HistoryFactory")
    history_source = ('"""Exact history builder copied from the audited fresh-seed experiment."""\n'
        'from hashlib import sha256\nfrom pathlib import Path\nimport time\nimport torch\n'
        'from conceptmod.textsliders import train_lm_slider_music3 as legacy\n'
        'from conceptmod.textsliders.gan_v2.data import gather, digest, allowed_logits, sha\n\n'
        + ast.get_source_segment(source, node) + "\n")
    (WORK / "histories.py").write_text(history_source)
    sys.path[:0] = [str(RUNTIME), str(ROOT.parent)]
    from app.rewriter import _artist_name_hit
    from conceptmod.textsliders import train_lm_slider_music3 as legacy, lm_gan_state
    from conceptmod.textsliders.gan_v2.train import arm_settings
    from conceptmod.textsliders.gan_v2.data import check_disjoint
    import torch
    torch.set_num_threads(2)
    catalog = read(ROOT / "analysis/uni16_20260906/catalog.json")
    registry = read(ROOT.parent / "app/sliders.json")
    active = {s["id"]:s for s in registry["sliders"]}
    assert len(active) == 16 and set(active) == {i["id"] for i in catalog["sliders"]}
    original = torch.load(ROOT / "models/uni16-gan-v1/lofi-warmup600-a02/lofi-warmup600-a02_state.pt",
                          map_location="cpu", weights_only=True, mmap=True)
    score_state = read(ROOT / "analysis/lofi_release_test_20260912/data/state.json")
    (WORK / "evidence").mkdir(exist_ok=True)
    write(WORK / "evidence/listening-state-at-launch.json", score_state)
    sliders = []
    ordered = [next(i for i in catalog["sliders"] if i["id"]==key) for key in
               ["female", "lofi", *[i["id"] for i in catalog["sliders"] if i["id"] not in ("female", "lofi")]]]
    prompt_audits = []
    for index, original_item in enumerate(ordered):
        item = copy.deepcopy(original_item)
        item.update(index=index, gpu=index % 2, fixed_cache=str(MODELS / item["id"] / "fixed-histories"))
        for split in ("train", "eval"):
            src = Path(item[split + "_prompts"])
            document = yaml.safe_load(src.read_text())
            if _artist_name_hit("", str(document)):
                raise ValueError(f"Prompt names must be removed before retraining {item['id']}")
            assert len(document["rows"]) == 4
            for row in document["rows"]:
                assert row["neutral"] == row["target"] == row["negative"]
                assert row["positive"] != row["neutral"] and row["lyrics"].strip()
            dest = WORK / "prompts" / f"{item['id']}-{split}.yaml"
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(src, dest)
            item[split + "_prompts"] = str(dest)
            prompt_audits.append(dict(slider=item["id"], split=split, rows=4, sha256=sha(dest), names_checked=True))
        rows, meta = legacy._load_rows(Path(item["train_prompts"]))
        evaluation, _ = legacy._load_rows(Path(item["eval_prompts"]))
        check_disjoint(rows, evaluation)
        signature = lm_gan_state.signature(legacy.parse_args(warmup_args(item, MODELS / "preflight")), rows, meta)
        assert signature["settings"] == original["signature"]["settings"], "Warmup recipe mismatch"
        assert signature["sources"] == original["signature"]["sources"], "Warmup source mismatch"
        item["warmup_signature"] = signature
        item["fresh_rows"] = list(range(4)) if rows_mode == "all" else [0]
        item["published_weights"] = str(Path(registry["root"]) / active[item["id"]]["components"][0]["weights"])
        item["published_sha256"] = sha(item["published_weights"])
        item["style_description"] = item["description"]
        sliders.append(item)
    recipe, critic = arm_settings("baseline", origin=600, horizon=660, diagnostics_every=15)
    winner = read(ROOT / "analysis/fresh_seed_break_20260911/lofi-fresh-break/manifest.json")
    assert asdict(recipe) == winner["recipe"] and critic == winner["critic"]
    model_dir = ROOT.parent / "models/MiniMax-Music3"
    model_files = {str(p):[p.stat().st_size,p.stat().st_mtime_ns]
                   for p in model_dir.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".json", ".txt", ".model")}
    write(WORK / "catalog.json", dict(sliders=sliders))
    write(WORK / "prompt-audit.json", prompt_audits)
    files = [*RUNTIME.rglob("*.py"), *(WORK / "prompts").glob("*.yaml"), WORK / "catalog.json",
             *[WORK / name for name in ("common.py", "prepare.py", "histories.py", "train.py", "campaign.py", "render.py")]]
    manifest = dict(version="uni16-fresh3400-v1", created=time.time(), fresh_rows_mode=rows_mode,
        gpu_authorization="User explicitly requested training on both graphics cards; physical GPUs 0 and 1.",
        gpus=[0,1], initialization="Fresh zero-output rank-8 alpha-8 LoRA and critic per slider; seed 7.",
        warmup=dict(updates=600, batch=4, histories="Fixed base histories; original winning recipe and source hashes."),
        continuation=dict(start=601, end=3400, batch=1, rows="Shuffled balanced passes over declared training rows",
            histories="One fresh base-model continuation per update, adapter disabled, 250 frames. Seeds do not cycle.",
            seed_rule="2000001 + slider_index * 10000 + update - 601", total_fresh_per_slider=2800,
            style_target="One fixed prompt-state target per training row; fresh histories diversify ending supervision.",
            recipe=asdict(recipe), critic=critic),
        save=dict(recovery_every=20, milestones=[600,1000,2000,3000,3400],
                  full_state="Generator, critic, both optimizers, sampler, RNG and complete continuation history."),
        evaluation=dict(rows=[2,3], seeds=[1709,2903], duration=20,
            steps=[1000,2000,3000,3400], controls=["step600","published660","off","positive_caption"],
            retries=0, accept_short=True, accept_silent=True),
        health_stop="Stop a slider for non-finite state, any near-silent or severely clipped candidate, or all four clips under four seconds. Keep samples; other sliders continue. Shorter natural endings alone do not stop the run.",
        selection="3400 is the fixed requested endpoint; no metric stopping and no automatic studio or Hub promotion.",
        limits="The 3400 preference was on four reused Lo-fi monitoring cases. The four-row continuation is a declared data change; checkpoint optimality does not transfer automatically to other sliders.",
        files={str(p.relative_to(WORK)):sha(p) for p in files}, model_files=model_files,
        original_runtime_hashes=runtime_hashes, original_history_builder_sha256=sha(parent / "train.py"))
    write(WORK / "manifest.json", manifest)
    write(WORK / "preflight.json", dict(all_16_sliders_validated=True, names_checked=True,
        train_eval_lyrics_disjoint=True, warmup_settings_and_sources_match=True,
        continuation_recipe_matches=True, fresh_rows_mode=rows_mode, manifest_sha256=sha(WORK / "manifest.json")))
    print(f"Prepared 16 fresh retrains on GPUs 0 and 1; continuation rows: {rows_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", choices=("all", "one"), default="all")
    main(parser.parse_args().rows)
