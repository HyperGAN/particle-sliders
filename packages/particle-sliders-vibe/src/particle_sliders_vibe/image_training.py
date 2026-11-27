"""Train from immutable fields with the product's stock ParticleGAN implementation."""
import os
from importlib.metadata import version
import torch

from . import digest, read_json, write_json
from .image_run import pin
from .image_targets import TrainingRows, save_tensors

UPDATES, HORIZON, SEED = 448, 1024, 1


def train(run, shards):
    host = run.host
    root = run.root / "fit"
    root.mkdir(exist_ok=True)
    g,d,bank = host.rates
    manifest = dict(model=host.runtime.identity, teacher_shards=[s["shard"]["sha256"] for s in shards],
                    preset="empf", particlegan=version("particlegan"), updates=UPDATES, horizon=HORIZON,
                    seed=SEED, batch_size=2, g=g, d=d, bank=bank, rate_source=host.rate_source, ema_decay=None)
    fingerprint = digest(manifest)
    manifest_path = root/"manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("training inputs or recipe changed")
    write_json(manifest_path,manifest)
    raw_path = root / f"RAW{UPDATES}.safetensors"
    done = root / "finished.json"
    if done.exists():
        saved = read_json(done)
        if saved["manifest"] != fingerprint or saved["raw"] != pin(raw_path):
            raise ValueError("frozen adapter changed")
        return saved["raw"]

    rows = TrainingRows(host, shards)
    ckpt = root/"FULL.pt"
    initial = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt.exists() else None
    if initial and initial["manifest"] != fingerprint:
        raise ValueError("checkpoint belongs to a different training manifest")
    kwargs = dict(lr=g, d_lr_mult=d/g, prior_lr_mult=bank/g)
    with host.game(rows, rows.features(), initial, SEED, HORIZON, kwargs) as (game,adapter,critic,report):
        if game.recipe.reg_every != 1 or game.recipe.ema_decay != 0.:
            raise ValueError("expected stock EMPF regularization and live weights")
        def checkpoint(metrics):
            step = game.completed_steps
            save_tensors(ckpt, dict(manifest=fingerprint,game=game.state_dict(),adapter=adapter.state_dict(),
                                    critic=critic.state_dict(),initialization=report,metrics=metrics))
            record = dict(step=step, budget=UPDATES, metrics=metrics)
            write_json(root/"updates"/f"{step:04}.json",record)
            if step % (UPDATES//4) == 0:
                snapshot = root/f"FULL{step}.pt"
                if not snapshot.exists():
                    os.link(ckpt,snapshot)
                write_json(root/"progress.json",record)
                print(f"training {step}/{UPDATES}",flush=True)
        if initial:
            game.load_state_dict(initial["game"])
            # Recover the receipt if interrupted between saving state and writing its cursor.
            checkpoint(initial["metrics"])
            del initial
        else:
            for i in (0, rows.count-1):
                if not torch.equal(game.evaluate(i).cpu(), rows[i].neutral.cpu()):
                    raise ValueError("zero-output adapter changed native prediction")
            write_json(root/"initialization.json", report)
            checkpoint({})
        while game.completed_steps < UPDATES:
            checkpoint(game.step())
        calls = game.opt_d.state_dict()["regularizer"]["record"]["calls"]
        if calls != UPDATES or game.completed_steps != UPDATES:
            raise RuntimeError("training update/regularizer accounting differs from declared budget")
        adapter.export(raw_path, model_identity=host.runtime.identity,
                       normalization=dict(profile=host.raw_profile,ema_decay=None), manifest_hash=fingerprint, step=UPDATES)
        raw = pin(raw_path)
        write_json(raw_path.with_suffix(".engine.json"),dict(**manifest,regularizer_calls=calls,sha256=raw["sha256"]))
        write_json(raw_path.with_suffix(".scoped.json"),dict(initialization=report,sha256=raw["sha256"]))
        write_json(done,dict(manifest=fingerprint,raw=raw))
    rows.load.cache_clear()
    return raw
