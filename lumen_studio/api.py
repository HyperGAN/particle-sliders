"""Local FastAPI Studio. The HTTP process never allocates a GPU."""
import asyncio
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import secrets
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import VARIATIONS, digest
from .inference_controls import MAX_ENERGY, inference_strengths
from .dataset import compile_manifest, compatible_manifest_hashes, load_definitions
from .store import Store
from .models import ANIMA, MODELS, get_model
from .atmospheres import DRAFT_ATMOSPHERES, MIXER_VARIATIONS, RETIRED

MAX_SEED = 2**53 - 1  # Exactly representable by browsers and JSON clients.

class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    prompt: str = Field(min_length=1, max_length=6000)
    model: str = ANIMA.id
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    width: int = Field(default=768, ge=512, le=1536)
    height: int = Field(default=768, ge=512, le=1536)
    steps: int = Field(default=10, ge=1, le=200)
    batch: int = Field(default=1, ge=1, le=16)
    energy: float = Field(default=1., ge=0, le=MAX_ENERGY)
    mix: dict[str, float] = Field(default_factory=lambda: dict.fromkeys(VARIATIONS, 0.))
    checkpoints: dict[str, str] = Field(default_factory=dict)

    @field_validator("width", "height")
    @classmethod
    def dimension(cls, value):
        if value % 16:
            raise ValueError("Dimensions must be divisible by 16")
        return value

    @field_validator("prompt")
    @classmethod
    def prompt_nonempty(cls, value):
        if not value.strip():
            raise ValueError("Prompt cannot be blank")
        return value  # Effective text is exactly what the user entered.


class Comparison(Generation):
    mode: Literal["off_on", "sweep"] = "off_on"


class Reorder(BaseModel):
    ids: list[str]


class Pause(BaseModel):
    paused: bool


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variation: Literal["candlelit", "moonlit", "theatrical"]
    until: Literal[200, 1600] = 200
    microbatch: Literal[1, 2, 4] = 1
    checkpointing: bool = True
    seed: int = Field(default=7, ge=0, le=2**31-1)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    notes: str = Field(default="", max_length=10000)
    atmosphere: Literal["unrated", "win", "tie", "loss"] = "unrated"
    major_regression: bool = False
    character: bool = True
    outfit: bool = True
    pose: bool = True
    composition: bool = True
    medium: bool = True
    unwanted_objects: bool = False
    recurring_features: bool = False
    quality: bool = True
    atmosphere_strength: float | None = Field(default=None, ge=0, le=5)
    blind: bool = False


def generation_payloads(request, store, model_identity, energies=None):
    model = get_model(request.model)
    if request.steps not in model.supported_steps:
        raise ValueError(f"{model.label} supports {model.supported_steps} steps")
    strengths = inference_strengths(request.mix, request.energy)
    if set(request.checkpoints) - set(VARIATIONS):
        raise ValueError("Unknown checkpoint variation")
    catalog = {c["sha256"]: c for c in store.catalog()}
    for variation, sha in request.checkpoints.items():
        if sha not in catalog or catalog[sha]["variation"] != variation:
            raise ValueError("Checkpoint is unavailable or belongs to another variation")
        if catalog[sha]["metadata"]["model_identity"] != model_identity:
            raise ValueError("Checkpoint uses a different model")
    energies = [request.energy] if energies is None else energies
    if max(energies) > 0 and any(v and name not in request.checkpoints for name, v in request.mix.items()):
        raise ValueError("Select an EMA checkpoint for each active slider")
    base_seed = request.seed if request.seed is not None else secrets.randbelow(MAX_SEED - request.batch + 2)
    if base_seed + request.batch - 1 > MAX_SEED:
        raise ValueError("Batch seeds exceed the maximum exact browser seed")
    rows = []
    for index in range(request.batch):
        for energy in energies:
            values = request.model_dump(exclude={"batch", "mode"})
            values.update(seed=base_seed + index, energy=energy,
                strengths=inference_strengths(request.mix, energy), cfg=model.guidance, model_identity=model_identity)
            rows.append(values)
    return rows


def create_app(directory, model_dir=None, model_identity=None):
    directory = Path(directory).resolve()
    store = Store(directory / "studio.sqlite3")
    if model_identity is None and model_dir is not None:
        from .provenance import model_identity as identify
        model_identity = identify(model_dir)
    app = FastAPI(title="NTC Image Studio")
    app.state.store = store
    app.state.model_identity = model_identity

    @app.exception_handler(ValueError)
    async def value_error(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(KeyError)
    async def missing(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    # This is a local app, with no public-deployment authentication contract.
    # Reject cross-origin browser mutations to protect a localhost GPU queue.
    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=403, content={"detail": "Cross-origin mutation rejected"})
        return await call_next(request)

    @app.get("/api/status")
    def status():
        return dict(model=model_identity, controls=store.controls())

    @app.post("/api/generate", status_code=202)
    def generate(request: Generation):
        if model_identity is None:
            raise HTTPException(503, "The pinned Turbo runtime has not been prepared")
        return store.enqueue(generation_payloads(request, store, model_identity))

    @app.post("/api/comparisons", status_code=202)
    def compare(request: Comparison):
        if model_identity is None:
            raise HTTPException(503, "The pinned Turbo runtime has not been prepared")
        maximum = max(1., request.energy)
        energies = [0., request.energy] if request.mode == "off_on" else [0., .25 * maximum, .5 * maximum, maximum]
        return store.enqueue(generation_payloads(request, store, model_identity, energies))

    @app.get("/api/jobs")
    def jobs():
        return store.jobs()

    @app.get("/api/jobs/{ident}")
    def job(ident: str):
        value = store.job(ident)
        if value is None:
            raise KeyError(ident)
        return value

    @app.post("/api/jobs/{ident}/cancel")
    def cancel(ident: str):
        store.cancel(ident)
        return store.job(ident)

    @app.post("/api/jobs/{ident}/retry")
    def retry(ident: str):
        store.retry(ident)
        return store.job(ident)

    @app.post("/api/queue/reorder")
    def reorder(value: Reorder):
        store.reorder(value.ids)
        return store.jobs()

    @app.post("/api/training/pause")
    def pause(value: Pause):
        store.control(paused=value.paused)
        return store.controls()

    @app.get("/api/history")
    def history(limit: int = 100, offset: int = 0):
        return store.history(max(1, min(500, limit)), max(0, offset))

    @app.get("/api/images/{ident}")
    def image(ident: str, download: bool = False):
        value = store.image(ident)
        if value is None:
            raise KeyError(ident)
        return FileResponse(value["path"], media_type="image/png",
                            filename=f"ntc-{ident}.png" if download else None)

    @app.post("/api/images/{ident}/rerender", status_code=202)
    def rerender(ident: str):
        image = store.image(ident)
        if image is None:
            raise KeyError(ident)
        metadata = image["metadata"]
        request = Generation(**{k: v for k, v in metadata.items() if k in Generation.model_fields})
        if metadata["model_identity"] != model_identity:
            if metadata.get("checkpoints"):
                raise ValueError("Original model no longer available")
            from .provenance import single_image_runtime_compatibility
            single_image_runtime_compatibility(model_dir, metadata["model_identity"], model_identity)
        return store.enqueue(generation_payloads(request, store, model_identity))

    @app.get("/api/images/{ident}/comparison")
    def comparison_image(ident: str):
        current = store.image(ident)
        if current is None:
            raise KeyError(ident)
        m = current["metadata"]
        with store.connect() as db:
            rows = [store.unpack(r) for r in db.execute(
                "SELECT * FROM images WHERE json_extract(metadata,'$.group_id')=? AND id<>? ORDER BY created DESC",
                (m["group_id"], ident))]
        matches = []
        for row in rows:
            other = row["metadata"]
            shared = ("seed", "width", "height", "steps", "model_identity", "case", "family")
            if any(other.get(k) != m.get(k) for k in shared):
                continue
            if m.get("reference_side"):
                if other.get("reference_side") != m["reference_side"]:
                    return row
            elif other.get("prompt") == m.get("prompt") and other.get("checkpoints") == m.get("checkpoints"):
                if (m["energy"] == 0 and other["energy"] > 0) or (m["energy"] > 0 and other["energy"] == 0):
                    matches.append(row)
        return max(matches, key=lambda row: row["metadata"]["energy"], default=None)

    @app.post("/api/images/{ident}/review")
    def review(ident: str, value: Review):
        store.review(ident, value.model_dump())
        return dict(saved=True)

    @app.get("/api/reviews")
    def reviews():
        return store.reviews()

    @app.get("/api/catalog")
    def catalog():
        return dict(variations=VARIATIONS, checkpoints=store.catalog(), model=model_identity,
                    mixer_variations=MIXER_VARIATIONS, retired_variations=RETIRED,
                    draft_atmospheres=DRAFT_ATMOSPHERES,
                    models=[m.public() for m in MODELS.values()], default_model=ANIMA.id)

    @app.get("/api/training/runs")
    def runs():
        result = store.runs()
        for run in result:
            variation = run["config"]["variation"]
            if variation not in VARIATIONS:
                continue
            # Keep archived runs attached to their own evidence. Stored paths
            # may originate on a different host; rebase their runs/ suffix.
            configured = Path(run["config"].get("directory", ""))
            parts = configured.parts
            run_directory = (directory.parent.joinpath(*parts[parts.index("runs"):])
                             if "runs" in parts else directory.parent / "runs" / variation)
            qualification = run_directory / "pilot-qualification.json"
            if qualification.exists():
                run["pilot_qualification"] = json.loads(qualification.read_text())
            path = run_directory / "updates.jsonl"
            run["recent_updates"] = []
            if not path.exists():
                continue
            # The writer may currently be appending the final line. Display
            # complete rows only, bounded by this run's published update.
            lines = path.read_text().splitlines(keepends=True)
            try:
                rows = [json.loads(line) for line in lines if line.endswith("\n")]
            except json.JSONDecodeError:
                run["log_error"] = "Training log contains an invalid completed row"
                continue
            rows = [r for r in rows if r["step"] <= run["step"]]
            fields = ("step", "d_adv", "g_adv", "d_penalty", "vic", "d_grad_norm", "g_grad_norm",
                      "particle_gan_grad_norm", "noise_min", "noise_max", "seconds")
            run["recent_updates"] = [{k: r[k] for k in fields if k in r} for r in rows[-120:]]
            durations = [r["seconds"] for r in rows[-20:] if math.isfinite(r.get("seconds", 0)) and r.get("seconds", 0) > 0]
            if durations:
                mean = sum(durations) / len(durations)
                run["timing"] = dict(seconds_per_update=mean,
                    training_seconds_remaining=max(0, run["config"].get("until", 200) - run["step"]) * mean)
        return result

    @app.get("/api/training/progress")
    def training_progress():
        root = directory.parent
        manifests = {split: compile_manifest(split) for split in ("train", "dev")}
        targets = []
        for variation in VARIATIONS:
            parts = {}
            for split, manifest in manifests.items():
                expected = {f"{r['id']}-{seed}.pt" for r in manifest["rows"]
                            if r["variation"] == variation for seed in r["seeds"]}
                cache = root / "targets" / variation / split
                identity = cache / "identity.json"
                valid = False
                if identity.exists():
                    value = json.loads(identity.read_text())
                    valid = value.get("manifest_sha256") in compatible_manifest_hashes(root, split, variation)
                    if valid and value.get("model") != model_identity:
                        from .provenance import cache_runtime_compatibility
                        try:
                            index = json.loads((cache / "index.json").read_text())
                            cache_runtime_compatibility(model_dir, value["model"], model_identity, index["fingerprint"])
                        except (ValueError, FileNotFoundError):
                            valid = False
                done = len(expected.intersection(p.name for p in cache.glob("*.pt"))) if valid else 0
                parts[split] = dict(done=done, total=len(expected),
                                    complete=done == len(expected) and (cache / "index.json").exists())
            targets.append(dict(variation=variation, **parts))
        preparation = root / "preparation.json"
        saved = json.loads(preparation.read_text()) if preparation.exists() else None
        total = sum(row[split]["total"] for row in targets for split in manifests)
        done = sum(row[split]["done"] for row in targets for split in manifests)
        complete = all(row[split]["complete"] for row in targets for split in manifests)
        benchmark_path = root / "benchmarks/candlelit/report.json"
        benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
        hardware_path = root / "benchmarks/migration-hardware/report.json"
        hardware = json.loads(hardware_path.read_text()) if hardware_path.exists() else None
        campaign_path = root / "campaign.json"
        campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else None
        return dict(targets=targets, done=done, total=total, complete=complete,
                    preparation=saved, updated_at=preparation.stat().st_mtime if saved else None,
                    server_time=time.time(), benchmark=benchmark, hardware_benchmark=hardware, campaign=campaign)

    @app.get("/api/training/convergence")
    def convergence_reports():
        reports = {}
        for variation in VARIATIONS:
            base = directory.parent / "diagnostics/convergence" / variation
            report_path, identity_path = base / "report.json", base / "identity.json"
            current_run = directory.parent / "runs" / variation / "run.json"
            if not report_path.exists():
                reports[variation] = dict(status="not_started")
                continue
            try:
                report = json.loads(report_path.read_text())
                identity = json.loads(identity_path.read_text())
                run = json.loads(current_run.read_text())
                if (identity["run"] != run or report["identity_sha256"] != digest(identity)
                        or report["run_cache"] != run["cache"] or run["model"] != model_identity):
                    raise ValueError("The measurements belong to a different run")
                reports[variation] = dict(report, response_protocol={
                    key: identity[key] for key in ("response_updates", "trial_seeds", "noise_repeats")
                    if key in identity})
            except (OSError, ValueError, KeyError) as exc:
                reports[variation] = dict(status="unavailable", error=str(exc))
        return reports

    @app.get("/api/training/images")
    def training_images():
        # Development samples only: never reveal the untouched final audit here.
        with store.connect() as db:
            return [store.unpack(r) for r in db.execute("""SELECT * FROM images
                WHERE json_extract(metadata,'$.purpose')='development'
                ORDER BY created DESC LIMIT 32""")]

    @app.post("/api/training/runs", status_code=202)
    def start_run(value: TrainingRequest):
        if value.variation in RETIRED:
            raise ValueError(RETIRED[value.variation])
        root = directory.parent
        config = dict(**value.model_dump(), directory=str(root / "runs" / value.variation),
            train_cache=str(root / "targets" / value.variation / "train"),
            dev_cache=str(root / "targets" / value.variation / "dev"))
        if any(not (Path(config[name]) / "index.json").exists() for name in ("train_cache", "dev_cache")):
            raise ValueError("Prepare qualified training and development target caches first")
        if any(r["config"]["variation"] == value.variation and r["status"] in ("queued", "running") for r in store.runs()):
            raise ValueError("This variation already has an active training run")
        if value.until == 1600:
            from .audits import require_pilot_qualification
            require_pilot_qualification(config["directory"])
        ident = f"{value.variation}-{value.until}-{secrets.token_hex(4)}"
        store.add_run(ident, config)
        return dict(id=ident)

    @app.post("/api/training/{variation}/qualify")
    def pilot_qualification(variation: str):
        from .audits import qualify_pilot
        if not (directory.parent / "runs" / variation / "run.json").exists():
            raise KeyError("No pilot run")
        return qualify_pilot(store, directory.parent, variation)

    @app.get("/api/definitions")
    def definitions():
        # Untouched final characters are never exposed through the development UI.
        return dict(definitions=[d.__dict__ for d in load_definitions()],
                    train=compile_manifest("train"), dev=compile_manifest("dev"))

    @app.get("/api/definitions/images")
    def definition_images():
        hashes_by_variation = {v: set().union(*(compatible_manifest_hashes(directory.parent, s, v)
                                                for s in ("train", "dev"))) for v in VARIATIONS}
        hashes = sorted(set().union(*hashes_by_variation.values()))
        definitions = {d.id: d.variation for d in load_definitions()}
        with store.connect() as db:
            rows = db.execute("""SELECT * FROM images WHERE json_extract(metadata,'$.purpose')
                IN ('definition_qualification','definition_candidate')
                AND json_extract(metadata,'$.manifest_sha256') IN (""" + ','.join('?' for _ in hashes)
                + ") ORDER BY created DESC", hashes)
            result, counts = [], {}
            for value in rows:
                image = store.unpack(value)
                meta = image["metadata"]
                definition = meta.get("definition")
                variation = definitions.get(definition)
                if (variation is not None and meta.get("manifest_sha256") in hashes_by_variation[variation]
                        and counts.get(definition, 0) < 4):
                    result.append(image)
                    counts[definition] = counts.get(definition, 0) + 1
            return result

    @app.get("/api/audits/selection")
    def selection():
        from .audits import selection_report
        return {v: selection_report(store, directory.parent, v) for v in VARIATIONS}

    @app.post("/api/audits/final", status_code=202)
    def final_audit():
        from .audits import enqueue_final
        if model_identity is None:
            raise HTTPException(503, "Prepare the model first")
        return enqueue_final(store, model_identity, directory.parent)

    @app.get("/api/audits/report")
    def final_report():
        from .audits import audit_report
        if not (directory.parent / "audits/final.json").exists():
            raise KeyError("No final audit")
        return audit_report(store, directory.parent)

    @app.get("/api/events")
    async def events(request: Request, after: int = 0):
        try:
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            raise HTTPException(400, "Invalid event cursor")

        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                rows = store.events(cursor)
                for row in rows:
                    cursor = row["id"]
                    yield f"id: {cursor}\nevent: update\ndata: {json.dumps(row)}\n\n"
                if not rows:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(.75)
        return StreamingResponse(stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app
