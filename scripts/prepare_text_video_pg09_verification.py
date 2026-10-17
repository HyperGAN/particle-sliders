"""Prepare an offline metadata receipt and bounded plan; never load a model.

No Torch/model imports, network calls, process dispatch or permission changes.
Large weights are statted only; their prior manifest hash attestations are
retained explicitly, not described as fresh full-file verification.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path('/mnt/nvme/sliders/qualification')
ASSETS = {
    'tiny': dict(directory='tiny-qwen-da87bfb', revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
                 manifest_sha256='4962affb04b7d1d07573428676d516f78dd7969d8a358691058fa35f16e59d4d'),
    'bonsai': dict(directory='bonsai-b072e1d', revision='b072e1d3b35a0a630cece372c2127528e0994386',
                   manifest_sha256='74521990968a62419b87194cc9f14033400c427876ffc3996a1a2783a181f11f'),
    'ltx': dict(directory='ltx25-staged-da0b690', revision='da0b690960026eab794b8ceaf841233cf7064478',
                manifest_sha256='5e2280fdcd6994e31217a14c2a4ce6b73593fac0fa4ff97b31fea028f6be4d35',
                linked_metadata_roots=[str(ROOT / 'models/ltx25-metadata-da0b690')]),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_manifest(root, expected, *, metadata_limit=64 * 1024**2):
    root = Path(root).resolve()
    manifest_path = root / 'qualification-manifest.json'
    manifest_hash = digest(manifest_path)
    if manifest_hash != expected['manifest_sha256']:
        raise ValueError('staged manifest differs from the selected authorized asset pin')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('revision') != expected['revision']:
        raise ValueError('staged model revision differs')
    files = manifest.get('files') or [dict(file=manifest['file'], bytes=manifest['weight_bytes'],
                                          sha256=manifest['weight_sha256'])]
    audited = []
    for entry in files:
        relative = Path(entry['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('manifest file must stay inside its authorized asset directory')
        path = root / relative
        resolved = path.resolve()
        small = entry['bytes'] <= metadata_limit
        known_metadata_link = small and any(resolved.is_relative_to(Path(p).resolve())
                                             for p in expected.get('linked_metadata_roots', []))
        if not resolved.is_relative_to(root) and not known_metadata_link:
            raise ValueError('asset symlink escapes its selected staged directory')
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError(f'missing or wrong-sized staged file: {relative}')
        if small and digest(path) != entry['sha256']:
            raise ValueError(f'staged metadata hash differs: {relative}')
        audited.append(dict(file=str(relative), resolved_path=str(resolved), bytes=entry['bytes'],
                            sha256=entry['sha256'],
                            hash_evidence='fresh metadata hash' if small else 'prior manifest attestation; stat only'))
    return dict(directory=str(root), manifest_sha256=manifest_hash,
                revision=manifest['revision'], files=audited,
                fresh_large_weight_hash_verification=False,
                new_agreements_accepted=False, model_loaded=False)


def prepare(root=ROOT):
    assets = {name: audit_manifest(root / 'models' / spec['directory'], spec)
              for name, spec in ASSETS.items()}
    cached = root / 'models/bonsai-caption-readouts-20261017'
    readouts = []
    for path in sorted(cached.glob('*.json')):
        if path.name.endswith('-tokens.json'):
            continue
        metadata = json.loads(path.read_text())
        tensor = path.with_suffix('.pt')
        if not tensor.is_file():
            raise ValueError(f'missing cached readout: {tensor.name}')
        readouts.append(dict(metadata_path=str(path), metadata_sha256=digest(path),
                             caption=metadata['text'], model_identity=metadata['identity'],
                             declared_tensor_sha256=metadata['sha256'],
                             tensor_path=str(tensor), tensor_bytes=tensor.stat().st_size,
                             tensor_loaded=False))
    comparisons = {}
    for family in ('tiny', 'bonsai'):
        run = root / f'runs/{family}-caption-native-1200-20261017'
        recipe_path, metrics_path = run / 'recipe.json', run / 'evaluation/step-001200/metrics.json'
        recipe, metrics = json.loads(recipe_path.read_text()), json.loads(metrics_path.read_text())
        pairs = {r['case']: dict(neutral=r['neutral'], positive=r['positive']) for r in metrics['records']}
        if family == 'bonsai':
            if {r['model_identity'] for r in readouts} != {recipe['base_identity']}:
                raise ValueError('cached Bonsai identity differs from the archived comparison')
            cached_texts = {r['caption'] for r in readouts}
            if any(text not in cached_texts for pair in pairs.values() for text in pair.values()):
                raise ValueError('cached Bonsai held-out comparison is incomplete')
        comparisons[family] = dict(recipe_path=str(recipe_path), recipe_sha256=digest(recipe_path),
                                   metrics_path=str(metrics_path), metrics_sha256=digest(metrics_path),
                                   prior_source_commit=recipe['code_commit'],
                                   prior_steps=metrics['step'], prior_quality_qualified=metrics['quality_qualified'],
                                   heldout_pairs=list(pairs.values()),
                                   scales=sorted({r['scale'] for r in metrics['records']}),
                                   run_seed=7, autoregressive_controls=dict(do_sample=False, max_new_tokens=32),
                                   prior_metrics_are_diagnostic_only=True)
    return dict(status='metadata_preparation_complete_execution_stopped',
                integrated_source_commits=['e985db9', 'b40d6e6'],
                current_reviewed_core='ec234e184676c46060ecbf6122df0e7b726f45a7',
                domain_candidate_patch_sha256='c6531a21f62fa96682f5aece517a6a862bb849ab971aa2766b3946b9a7543571',
                domain_candidate_status='overlay optimizer QA passed; host profile/domain admission pending',
                assets=assets, bonsai_cached_readout_metadata=readouts,
                archived_comparisons=comparisons,
                strength_grid=[0, .5, 1], seed=7,
                software_pilot=dict(total_steps=4, batch_size=2, checkpoint_after=2,
                                    independent_process_resume=True, presets=['bispf', 'bispf_optimized']),
                hardware_pilot=dict(max_updates=2, batch_size=2, timeout_seconds=600,
                                    automatic_retry=False, budget_extension=False),
                required_evidence_before_model_load=[
                    'QA receipt approving the single explicit continuous-domain source profile',
                    'frozen canonical core binds base wheel, 49 source hashes, patch and both domains',
                    'backend opts into that exact profile; default release-auto behavior stays blocked',
                    'both-preset wide-response gradients, actual-bank updates and exact fresh-process CPU gates',
                    'fresh required weight/config/tokenizer manifest verification',
                    'core owner explicitly admits exact model/device/command/budget/process placement'],
                model_load_allowed=False, gpu_admission_requested=False, model_loaded=False,
                download_performed=False, network_used=False, legacy_entrypoint_executed=False,
                h3=dict(action='native multimodal adapter/model execution stopped',
                        missing=['reviewed native AV geometry/response/critic protocol',
                                 'verified staged exact model assets and applicable community terms',
                                 'measured feasible placement and owner admission']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], model_loaded=False,
                          assets=list(receipt['assets']),
                          cached_bonsai_readouts=len(receipt['bonsai_cached_readout_metadata']),
                          model_load_allowed=False)))


if __name__ == '__main__':
    main()
