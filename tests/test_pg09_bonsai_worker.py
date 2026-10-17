import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import verify_bonsai_pg09_cached as worker
from scripts.caption_quality import HELDOUT, CANARIES


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_cached_worker_complete_flow_with_synthetic_cache_only(tmp_path, preset, monkeypatch):
    # This fixture never opens the official cache. It validates worker control,
    # safe serialization, diagnostics and replay using synthetic native readouts.
    identity = 'synthetic-native-5120-readouts-only'
    cache = tmp_path / 'synthetic-cache'
    cache.mkdir()
    data = yaml.safe_load((worker.ROOT / 'conceptmod/textsliders/data/prompts-bonsai-gguf.yaml').read_text())
    texts = set(CANARIES)
    for row in data + HELDOUT:
        texts.update((row['neutral'], row['positive']))
    for text in texts:
        key = hashlib.sha256((identity + '\0' + text).encode()).hexdigest()
        path = cache / (key + '.pt')
        generator = torch.Generator().manual_seed(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
        torch.save(torch.randn(1, 5120, generator=generator), path)
        path.with_suffix('.json').write_text(json.dumps(dict(identity=identity, text=text,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), independent_requests_exact=True)))
        (cache / (key + '-tokens.json')).write_text(json.dumps(dict(identity=identity, text=text, ids=[1, 2])))
    receipt = dict(cache=str(cache), readout_identity=identity, qualification='synthetic cache only',
                   permitted_scope='unit test', model_loaded=False, tensor_payload_loaded=False, gpu_used=False)
    args = argparse.Namespace(core=Path('/mnt/nvme/sliders/text-video-core-3e94b13-20261017'),
                              overlay=Path('/tmp/alz-pg09-domain-proposal'), output=tmp_path / preset,
                              mode='produce', preset=preset)
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', torch.nn.functional.mse_loss)
    worker.execute(args, receipt)
    args.mode = 'restore'
    worker.execute(args, receipt)
    result = json.loads((args.output / 'restore-receipt.json').read_text())
    assert result['exact_full_state'] and result['exact_losses'] and result['exact_export_reload']
    assert result['quality_qualified'] is False and result['gpu_used'] is False
    assert not torch.cuda.is_initialized()
