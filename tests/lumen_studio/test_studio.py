from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from lumen_studio.api import create_app
from lumen_studio.coordinator import Coordinator, GpuLease, OwnershipError, choose_work
from lumen_studio.store import Store
from conftest import TinyRuntime


def test_api_queue_persistence_comparisons_and_validation(tmp_path):
    identity = dict(model="tiny-test-only", sha256="a" * 64)
    app = create_app(tmp_path, model_identity=identity)
    client = TestClient(app)
    request = dict(prompt="an adult man in a coat", seed=42, mix={}, energy=.5, mode="sweep", batch=2)
    response = client.post("/api/comparisons", json=request)
    assert response.status_code == 202
    ids = response.json()["ids"]
    assert len(ids) == 8
    assert [j["payload"]["seed"] for j in app.state.store.jobs()] == [42]*4 + [43]*4
    assert [j["payload"]["energy"] for j in app.state.store.jobs()][:4] == [0,.25,.5,1]
    assert len(Store(tmp_path / "studio.sqlite3").jobs()) == 8
    assert client.post("/api/queue/reorder", json={"ids":list(reversed(ids))}).status_code == 200
    assert app.state.store.claim()["id"] == ids[-1]
    client.post(f"/api/jobs/{ids[-1]}/cancel")
    assert app.state.store.job(ids[-1])["status"] == "cancelling"
    app.state.store.recover()
    assert app.state.store.job(ids[-1])["status"] == "cancelled"
    assert client.post(f"/api/jobs/{ids[-1]}/retry").status_code == 200
    assert client.post("/api/queue/reorder", json={"ids":[ids[0],ids[0]]}).status_code == 422
    assert client.post("/api/generate", json={"prompt":"x", "negative_prompt":"bad"}).status_code == 422
    assert client.post("/api/generate", json={"prompt":"x", "width":513}).status_code == 422
    assert client.post("/api/generate", json={"prompt":"x", "mix":{"candlelit":1}}).status_code == 422
    assert client.post("/api/generate", json={"prompt":"x", "mix":{"bad":1}}).status_code == 422
    assert client.post("/api/generate", json={"prompt":"x"}, headers={"Origin":"https://example.com"}).status_code == 403
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.get("/").status_code == 200
    assert client.get("/static/studio.js").status_code == 200
    assert client.post('/api/generate', json=dict(prompt='x', seed=2**53)).status_code == 422
    assert client.post('/api/generate', json=dict(prompt='x', seed=2**53-1, batch=2)).status_code == 422


def test_worker_images_metadata_rerender_failure_and_restart(tmp_path):
    studio = tmp_path / "studio"
    runtime = TinyRuntime()
    app = create_app(studio, model_identity=runtime.identity)
    client = TestClient(app)
    response = client.post("/api/generate", json=dict(prompt="an adult woman",seed=81))
    ident = response.json()["ids"][0]
    worker = Coordinator(studio, tmp_path / "model", gpu="cpu", runtime_factory=TinyRuntime)
    worker.run(once=True)
    assert app.state.store.job(ident)["status"] == "completed"
    image = app.state.store.image(ident)
    assert image["metadata"]["effective_prompt"] == "an adult woman"
    assert image["metadata"]["seed"] == 81
    assert image["metadata"]["model_identity"] == runtime.identity
    assert client.get(f"/api/images/{ident}").headers["content-type"] == "image/png"
    assert client.post(f"/api/images/{ident}/rerender").status_code == 202
    assert client.post(f"/api/images/{ident}/review",json={"notes":"test","atmosphere":"win"}).status_code == 200
    assert len(client.get('/api/reviews').json()) == 1
    unfinished = app.state.store.claim()
    assert unfinished
    Store(studio / "studio.sqlite3").recover()
    assert app.state.store.job(unfinished["id"])["status"] == "queued"
    def failed(*a, **k):
        raise RuntimeError("simulated worker model-load failure")
    worker = Coordinator(studio, tmp_path / "model", gpu="cpu", runtime_factory=failed)
    worker.run(once=True)
    assert app.state.store.job(unfinished["id"])["status"] == "failed"
    assert 'simulated' in app.state.store.job(unfinished['id'])['error']


def test_gpu_ownership_and_fairness(tmp_path):
    with GpuLease(tmp_path, "GPU-test"):
        with pytest.raises(OwnershipError):
            with GpuLease(tmp_path, "GPU-test"):
                pass
    with GpuLease(tmp_path, "GPU-test"):
        pass
    c = dict(paused=False,render_burst=0,training_updates=20)
    for count in range(4):
        c['render_burst'] = count
        assert choose_work(c,True,True) == "render"
    c['render_burst'] = 4
    assert choose_work(c,True,True) == "train"
    c.update(render_burst=0,training_updates=19)
    assert choose_work(c,True,True) == "train"
    c['training_updates'] = 20
    assert choose_work(c,True,True) == "render"
    c.update(paused=True,training_updates=0,render_burst=99)
    assert choose_work(c,True,True) == "render"


def test_test_characters_are_not_in_development_ui(tmp_path):
    app = create_app(tmp_path)
    response = TestClient(app).get('/api/definitions')
    text = response.text
    assert 'original-17' not in text
    assert 'original-24' not in text


def test_retired_atmosphere_is_kept_for_history_but_not_offered_in_mixer(tmp_path):
    app = create_app(tmp_path / 'studio')
    store = app.state.store
    with store.connect() as db:
        db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?)',
                   ('retired-sha', 'theatrical', '/old/theatrical.safetensors', '{}', 0.))
    store.add_run('old-pilot', dict(variation='theatrical', until=200))
    store.update_run('old-pilot', 'pilot_review', 200)
    client = TestClient(app)
    catalog = client.get('/api/catalog').json()
    assert catalog['mixer_variations'] == ['candlelit', 'moonlit']
    assert 'theatrical' in catalog['retired_variations']
    assert catalog['draft_atmospheres']['dusk']['description'] == 'Amber/violet twilight'
    assert 'dusk' not in catalog['mixer_variations']
    assert catalog['checkpoints'][0]['sha256'] == 'retired-sha'
    assert client.get('/api/training/runs').json()[0]['id'] == 'old-pilot'
    response = client.post('/api/training/runs', json=dict(variation='theatrical'))
    assert response.status_code == 422 and 'Retired' in response.json()['detail']


def test_convergence_report_requires_matching_run_identity(tmp_path):
    import json
    from lumen_studio.contracts import digest
    model = dict(model="tiny", sha256="a"*64)
    client = TestClient(create_app(tmp_path / "studio", model_identity=model))
    assert client.get('/api/training/convergence').json()['candlelit']['status'] == 'not_started'
    folder = tmp_path / 'diagnostics/convergence/candlelit'
    run_folder = tmp_path / 'runs/candlelit'
    folder.mkdir(parents=True); run_folder.mkdir(parents=True)
    run = dict(model=model, cache='cache-a', seed=7)
    identity = dict(run=run, steps=[1200,1600], response_updates=16, trial_seeds=[811,947], noise_repeats=2)
    report = dict(status='completed', run_cache=run['cache'], identity_sha256=digest(identity))
    (folder/'identity.json').write_text(json.dumps(identity))
    (folder/'report.json').write_text(json.dumps(report))
    (run_folder/'run.json').write_text(json.dumps(run))
    assert client.get('/api/training/convergence').json()['candlelit']['status'] == 'completed'
    assert client.get('/api/training/convergence').json()['candlelit']['response_protocol'] == dict(
        response_updates=16, trial_seeds=[811,947], noise_repeats=2)
    run['seed'] = 29
    (run_folder/'run.json').write_text(json.dumps(run))
    assert client.get('/api/training/convergence').json()['candlelit']['status'] == 'unavailable'


def test_live_training_log_ignores_partial_and_unpublished_updates(tmp_path):
    import json
    app = create_app(tmp_path / "studio")
    store = app.state.store
    store.add_run("pilot", dict(variation="candlelit", until=200))
    store.update_run("pilot", "running", 2)
    path = tmp_path / "runs/candlelit/updates.jsonl"
    path.parent.mkdir(parents=True)
    rows = [dict(step=i, seconds=8 + i, d_adv=.6, g_adv=.7, vic=.1,
                 d_penalty=0., d_rows=[123], g_rows=[456]) for i in (1, 2, 3)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"step":4,')
    run = TestClient(app).get('/api/training/runs').json()[0]
    assert [r['step'] for r in run['recent_updates']] == [1, 2]
    assert all('d_rows' not in r and 'g_rows' not in r for r in run['recent_updates'])
    assert run['timing'] == dict(seconds_per_update=9.5, training_seconds_remaining=1881.)


def test_archived_run_keeps_its_own_logs_and_qualification_after_host_move(tmp_path):
    import json
    app = create_app(tmp_path / 'studio')
    store = app.state.store
    for ident, suffix, passed, seconds in (
            ('old', 'archive/theatrical-v7', False, 9.), ('new', 'theatrical', True, 6.)):
        store.add_run(ident, dict(variation='theatrical', until=200,
                                directory='/previous/host/artifacts/anima/runs/' + suffix))
        store.update_run(ident, 'pilot_review', 200)
        path = tmp_path / 'runs' / suffix
        path.mkdir(parents=True)
        (path / 'updates.jsonl').write_text(json.dumps(dict(step=200, seconds=seconds)) + '\n')
        (path / 'pilot-qualification.json').write_text(json.dumps(dict(passed=passed)))
    runs = {r['id']: r for r in TestClient(app).get('/api/training/runs').json()}
    assert runs['old']['pilot_qualification']['passed'] is False
    assert runs['new']['pilot_qualification']['passed'] is True
    assert runs['old']['timing']['seconds_per_update'] == 9.
    assert runs['new']['timing']['seconds_per_update'] == 6.


def test_definition_gallery_retains_only_unchanged_archived_variations(tmp_path, monkeypatch):
    from dataclasses import replace
    from lumen_studio import dataset
    app = create_app(tmp_path / 'studio')
    store = app.state.store
    old_hash = dataset.compile_manifest('train')['sha256']
    dataset.archive_catalog(tmp_path)
    original_ids = {}
    def add(variation, sha, purpose):
        meta = dict(purpose=purpose, definition=variation + '-01', manifest_sha256=sha)
        ident = store.enqueue([meta])['ids'][0]
        assert store.claim()['id'] == ident
        store.finish(ident, tmp_path / 'studio/images' / (ident + '.png'), meta)
        return ident
    for variation in ('candlelit', 'moonlit', 'theatrical'):
        original_ids[variation] = add(variation, old_hash, 'definition_qualification')
    definitions = [replace(d, lighting=d.lighting + ', darker shadow edges')
                   if d.variation == 'theatrical' else d for d in dataset.load_definitions()]
    monkeypatch.setattr(dataset, 'load_definitions', lambda: definitions)
    current_id = add('theatrical', dataset.compile_manifest('train')['sha256'], 'definition_candidate')
    shown = {i['id'] for i in TestClient(app).get('/api/definitions/images').json()}
    assert shown == {original_ids['candlelit'], original_ids['moonlit'], current_id}


def test_training_progress_counts_only_current_committed_targets(tmp_path):
    import json
    from lumen_studio.dataset import compile_manifest
    identity = dict(model="tiny-test-only", sha256="a" * 64)
    client = TestClient(create_app(tmp_path / "studio", model_identity=identity))
    empty = client.get('/api/training/progress').json()
    assert empty['done'] == 0 and empty['total'] == 1224 and not empty['complete']
    manifest = compile_manifest('train')
    row = next(r for r in manifest['rows'] if r['variation'] == 'candlelit')
    directory = tmp_path / 'targets/candlelit/train'
    directory.mkdir(parents=True)
    committed = directory / f"{row['id']}-{row['seeds'][0]}.pt"
    committed.touch()
    committed.with_suffix('.tmp').touch()
    (directory / 'unrelated.pt').touch()
    idpath = directory / 'identity.json'
    idpath.write_text(json.dumps(dict(model=identity, manifest_sha256='old')))
    assert client.get('/api/training/progress').json()['done'] == 0
    idpath.write_text(json.dumps(dict(model=identity, manifest_sha256=manifest['sha256'])))
    progress = client.get('/api/training/progress').json()
    assert progress['done'] == 1 and not progress['complete']
    assert progress['targets'][0]['train']['done'] == 1
    assert client.get('/api/training/runs').json() == []
    assert client.get('/api/training/images').json() == []


def test_verified_base_runtime_fix_preserves_queued_requests_and_rerender(tmp_path):
    import json
    runtime = TinyRuntime()
    before = runtime.identity.copy()
    after = dict(before, runtime_sha256='batch-mask-fix')
    studio, model = tmp_path / 'studio', tmp_path / 'model'
    model.mkdir()
    old_app = create_app(studio, model_identity=before)
    old_client = TestClient(old_app)
    first = old_client.post('/api/generate', json=dict(prompt='first', seed=1)).json()['ids'][0]
    worker = Coordinator(studio, model, gpu='cpu', runtime_factory=lambda *a, **k: runtime)
    worker.render_one()
    second = old_client.post('/api/generate', json=dict(prompt='pending', seed=2)).json()['ids'][0]
    proof = dict(before=before, after=after, passed=True, purpose='single-image-teacher-cache-reuse',
        single_image_max_abs=0, positions=list(range(10)), batch_sizes=[1,2,4], cache_fingerprints=[])
    (model / 'runtime-compatibility.json').write_text(json.dumps(proof))
    runtime.identity = after
    worker.render_one()
    saved = worker.store.image(second)['metadata']
    assert saved['model_identity'] == after and saved['requested_model_identity'] == before
    client = TestClient(create_app(studio, model, model_identity=after))
    request = client.post(f'/api/images/{first}/rerender')
    assert request.status_code == 202
    assert worker.store.job(request.json()['ids'][0])['payload']['model_identity'] == after
    worker.release_runtime()


def test_comparison_companion_and_cancellation_during_render(tmp_path):
    runtime = TinyRuntime()
    app = create_app(tmp_path / 'studio', model_identity=runtime.identity)
    client = TestClient(app)
    response = client.post('/api/comparisons', json=dict(prompt='an adult man', seed=17, energy=1))
    first, second = response.json()['ids']
    worker = Coordinator(tmp_path / 'studio', tmp_path / 'model', gpu='cpu', runtime_factory=TinyRuntime)
    worker.render_one(); worker.render_one()
    assert client.get(f'/api/images/{second}/comparison').json()['id'] == first
    assert client.get(f'/api/images/{first}/comparison').json()['id'] == second
    ident = client.post('/api/generate', json=dict(prompt='cancel this render')).json()['ids'][0]
    original = worker.store.progress
    def cancelling(ident, progress):
        original(ident, progress)
        worker.store.cancel(ident)
    worker.store.progress = cancelling
    worker.render_one()
    assert app.state.store.job(ident)['status'] == 'cancelled'
    assert app.state.store.image(ident) is None
    assert not (tmp_path / 'studio/images' / (ident + '.png')).exists()
    worker.release_runtime()
