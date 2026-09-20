from collections import defaultdict

from lumen_studio.particles import ParticleAdapter
from lumen_studio.sampling import enqueue_cadence, selection_rows
from lumen_studio.store import Store


def test_reduced_cadence_retains_pilot_and_selection_coverage(tmp_path, tiny):
    store = Store(tmp_path / "studio.sqlite3")
    for variation in ("candlelit", "moonlit", "theatrical"):
        adapter = ParticleAdapter(tiny.transformer)
        path = tmp_path / f"{variation}.safetensors"
        adapter.export(path, model_identity=tiny.identity, normalization={"test": True},
                       manifest_hash="a" * 64, step=400, ema=adapter.state_dict())
        sha = store.register_checkpoint(path, variation)
        total = 0
        for step, count in ((20, 4), (100, 16), (200, 16),
                            (400, 32), (800, 32), (1200, 32), (1600, 32)):
            result = enqueue_cadence(store, tiny.identity, variation, sha, step)
            payloads = [store.job(i)["payload"] for i in result["ids"]]
            assert len(payloads) == count
            total += len(payloads)
            by_case = defaultdict(list)
            for payload in payloads:
                assert payload["sampling_step"] == step
                assert payload["purpose"] == "development"
                assert payload["checkpoints"] == {variation: sha}
                by_case[payload["case"]].append(payload)
            for group in by_case.values():
                assert [p["energy"] for p in group] == ([0., 1.] if step == 20 else [0., .25, .5, 1.])
                assert len({(p["prompt"], p["seed"], p["width"], p["height"]) for p in group}) == 1
            if step % 400 == 0:
                assert set(by_case) == {r["id"] for r in selection_rows(variation)}
                assert all(p["width"] == p["height"] == 768 for p in payloads)
        assert total == 164
        before = len(store.jobs(limit=1000))
        for step in (0, 19, 21, 300, 500, 600, 700, 900, 1000, 1100, 1300, 1400, 1500):
            assert enqueue_cadence(store, tiny.identity, variation, sha, step) == []
        assert len(store.jobs(limit=1000)) == before
