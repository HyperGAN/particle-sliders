"""Stdlib-only fault/adoption tests; never import the model or Torch runtime."""
import ast
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('distillation_artifacts', SCRIPTS/'distillation_artifacts.py')
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


def functions(filename, names, namespace):
    """Exercise real wrapper control flow without importing optional runtimes."""
    tree = ast.parse((SCRIPTS/filename).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    for node in selected:
        node.decorator_list = []
    namespace = dict(namespace, __file__=str(SCRIPTS/filename))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SCRIPTS/filename), 'exec'), namespace)
    return namespace


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {'source': 'source-v1', 'model': {'sha256': 'weights-v1'}, 'settings': {'steps': 100}}

    def test_legacy_identity_and_mismatch_refuse_without_mutating_evidence(self):
        marker = self.root/'identity.json'
        old = self.root/'evaluation.json'
        old.write_bytes(b'{"old":"evidence"}')
        with self.assertRaisesRegex(ValueError, 'no identity'):
            artifacts.bind_identity(marker, self.identity, evidence=[old])
        self.assertFalse(marker.exists())
        self.assertEqual(old.read_bytes(), b'{"old":"evidence"}')
        artifacts.bind_identity(marker, self.identity)
        before = marker.read_bytes()
        artifacts.bind_identity(marker, self.identity)
        for key in self.identity:
            changed = deepcopy(self.identity)
            changed[key] = 'changed'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity changed'):
                artifacts.bind_identity(marker, changed)
            self.assertEqual(marker.read_bytes(), before)

    def test_completion_requires_full_output_manifest_and_original_bytes(self):
        output = self.root/'weights.bin'
        sidecar = self.root/'weights.json'
        output.write_bytes(b'checkpoint')
        sidecar.write_text('{}')
        marker = self.root/'evaluation.json'
        artifacts.publish_completion(marker, self.identity, {'complete': True}, [output, sidecar])
        self.assertTrue(artifacts.verify_completion(marker, self.identity, [output, sidecar])['complete'])
        with self.assertRaisesRegex(ValueError, 'output missing or changed'):
            artifacts.verify_completion(marker, self.identity, [output])
        output.write_bytes(b'corruption')
        with self.assertRaisesRegex(ValueError, 'output missing or changed'):
            artifacts.verify_completion(marker, self.identity, [output, sidecar])
        before = marker.read_bytes()
        with self.assertRaises(FileExistsError):
            artifacts.publish_completion(marker, self.identity, {}, [output, sidecar])
        self.assertEqual(marker.read_bytes(), before)
        sidecar.unlink()
        with self.assertRaises(FileNotFoundError):
            artifacts.verify_completion(marker, self.identity, [output, sidecar])

    def test_sealed_payload_detects_changed_tokens_and_missing_identity(self):
        path = self.root/'rollout.json'
        artifacts.write_new_json(path, artifacts.seal({'ids': [1, 2]}, self.identity))
        self.assertEqual(artifacts.verify_record(path, self.identity)['ids'], [1, 2])
        value = json.loads(path.read_text())
        value['ids'].append(3)
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            artifacts.verify_record(path, self.identity)
        path.write_text('{"ids":[1,2]}')
        with self.assertRaisesRegex(ValueError, 'no adoption identity'):
            artifacts.verify_record(path, self.identity)

    def test_publication_failure_retains_existing_files_and_cleans_only_own_temp(self):
        evidence = self.root/'original'
        evidence.write_bytes(b'keep')
        with patch.object(artifacts.os, 'link', side_effect=OSError('injected publication failure')):
            with self.assertRaisesRegex(OSError, 'injected'):
                artifacts.write_new_json(self.root/'receipt.json', {'new': True})
        self.assertEqual(list(self.root.iterdir()), [evidence])
        self.assertEqual(evidence.read_bytes(), b'keep')

    def test_concurrent_attempt_refused_without_a_training_deadline(self):
        code = """import sys
sys.path.insert(0,sys.argv[1])
from distillation_artifacts import attempt_lock
try:
 with attempt_lock(sys.argv[2]): pass
except RuntimeError as error:
 assert 'already owned' in str(error)
else: raise AssertionError('competing writer admitted')
assert 'torch' not in sys.modules and 'yue2' not in sys.modules
"""
        with artifacts.attempt_lock(self.root):
            result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(SCRIPTS), str(self.root)],
                                    capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_rollout_path_reuses_only_exact_sealed_identity(self):
        prefix = [1, 2]
        pipe = NS(calls=0)
        def generate(plan, sampling):
            pipe.calls += 1
            return NS(tokens=[3, 4], truncated=False, timing={})
        pipe.generate_semantic = generate
        namespace = functions('distill_yue2_particles.py',
            {'rollout_identity', 'load_rollout', 'rollout'},
            dict(artifacts=artifacts, make_plan=lambda *args: NS(prefix=list(prefix)),
                 attached=lambda *args: nullcontext()))
        protocol = ModuleType('yue2.protocol')
        protocol.CODEC_OFFSET = 100
        row, cache = {'lyrics': 'words', 'neutral': 'sound'}, self.root/'rollout.json'
        rollout = namespace['rollout']
        with patch.dict(sys.modules, {'yue2': ModuleType('yue2'), 'yue2.protocol': protocol}):
            ids = rollout(pipe, None, None, row, 7, 4, cache, cohort=self.identity)
            self.assertEqual(ids, [1, 2, 103, 104])
            before = cache.read_bytes()
            self.assertEqual(rollout(pipe, None, None, row, 7, 4, cache, cohort=self.identity), ids)
            for field, value in [('source', 'new-source'), ('model', 'new-model'), ('settings', {'steps': 200})]:
                changed = dict(self.identity, **{field: value})
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    rollout(pipe, None, None, row, 7, 4, cache, cohort=changed)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                rollout(pipe, None, None, row, 7, 5, cache, cohort=self.identity)
            prefix.append(9)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                rollout(pipe, None, None, row, 7, 4, cache, cohort=self.identity)
        self.assertEqual(pipe.calls, 1)
        self.assertEqual(cache.read_bytes(), before)

    def test_fit_and_refine_completion_branches_verify_outputs_before_skip(self):
        args, concept = NS(output=self.root, train_tokens=4), 'example'
        out = self.root/concept
        out.mkdir()
        target = out/f'{concept}_distilled_rank8.safetensors'
        sidecar = target.with_suffix('.json')
        for path in (target, sidecar, out/'train-rollout-0.json'):
            path.write_bytes(b'evidence')
        base = dict(Path=Path, artifacts=artifacts, rows=lambda *args: [{}],
                    fit_identity=lambda *args: self.identity, log=lambda *args, **kwargs: None)
        namespace = functions('distill_yue2_particles.py', {'fit_outputs', '_fit_one'}, base)
        artifacts.bind_identity(out/'distillation-identity.json', self.identity)
        artifacts.publish_completion(out/'evaluation.json', self.identity, {},
            namespace['fit_outputs'](out, concept, 1))
        namespace['_fit_one'](args, None, None, self.root, {'id': concept}, [])
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'output missing or changed'):
            namespace['_fit_one'](args, None, None, self.root, {'id': concept}, [])
        refined = out/f'{concept}_distilled_refined_rank8.safetensors'
        refined.write_bytes(b'refined')
        refined.with_suffix('.json').write_bytes(b'sidecar')
        d = NS(artifacts=artifacts)
        namespace = functions('refine_yue2_distillation.py', {'refined_outputs', '_refine'},
            dict(d=d, refinement_identity=lambda *args: self.identity))
        artifacts.bind_identity(out/'refinement-identity.json', self.identity)
        artifacts.publish_completion(out/'refined-evaluation.json', self.identity, {},
                                     namespace['refined_outputs'](out, concept))
        self.assertEqual(namespace['_refine'](args, None, None, {'id': concept}, []), refined)
        refined.unlink()
        with self.assertRaises(FileNotFoundError):
            namespace['_refine'](args, None, None, {'id': concept}, [])

    def test_complete_cohort_builder_binds_source_model_settings_and_inputs(self):
        core = self.root/'core'
        core.mkdir()
        (core/'__init__.py').write_text('# captured core\n')
        (core/'adapter.py').write_text('# adapter\n')
        model_dir = self.root/'model'
        model_dir.mkdir()
        (model_dir/'qwen.tiktoken').write_bytes(b'tokenizer')
        release = self.root/'release'
        release.mkdir()
        teacher = release/'teacher.bin'
        teacher.write_bytes(b'teacher')
        for name in ('catalog.json', 'train.yaml', 'eval.yaml'):
            (release/name).write_text('{}')
        for row in range(2):
            for kind in ('off', 'metal'):
                folder = self.root/'reference'/'example'/f'row-{row}-seed-1709'/kind
                folder.mkdir(parents=True)
                for name in ('prefix.npy', 'semantic.npy'):
                    (folder/name).write_bytes(b'tokens')
        args = NS(gpu=0, reference=self.root/'reference')
        spec = {'id': 'example', 'weights': 'teacher.bin',
                'sha256': artifacts.file_identity(teacher)['sha256'],
                'prompts': {'train': 'train.yaml', 'eval': 'eval.yaml'}}
        torch = NS(__version__='torch-test', get_float32_matmul_precision=lambda: 'highest',
            get_default_dtype=lambda: 'float32', get_num_threads=lambda: 4,
            get_num_interop_threads=lambda: 4, version=NS(cuda='test-cuda'),
            are_deterministic_algorithms_enabled=lambda: False,
            backends=NS(cuda=NS(matmul=NS(allow_tf32=False)),
                        cudnn=NS(allow_tf32=False, benchmark=False, deterministic=True)))
        weights = {name: {'files': {'model.safetensors': {'sha256': artifacts.digest(name), 'bytes': 10}},
                         'config_sha256': artifacts.digest('configuration')} for name in ('mot', 'vae')}
        generation = {'ode_steps': 32}
        pipe = NS(weights=weights, runtime_sha256=artifacts.digest('native-source'), model_dir=model_dir,
            backend='torch', quantization='none', offload_ar=False, vae_core_frames=1024,
            generation_config=NS(to_dict=lambda: dict(generation)))
        model = NS(config=NS(to_dict=lambda: {'architecture': 'test'}), training=False,
                   named_parameters=lambda: [('weight', NS(shape=(2, 3), dtype='float32', device='cpu'))])
        module = NS(__file__=str(core/'__init__.py'))
        def artist(*args): pass
        namespace = functions('distill_yue2_particles.py', {'cohort_identity'},
            dict(Path=Path, sys=sys, os=os, artifacts=artifacts, torch=torch, np=NS(__version__='numpy-test'),
                 yaml=NS(__version__='yaml-test'), _artist_name_hit=artist,
                 importlib=NS(metadata=NS(version=lambda name: 'test-version'))))
        with patch.dict(sys.modules, {'particle_sliders': module}):
            identity = namespace['cohort_identity'](args, pipe, model, release, spec, [{'lyrics': 'calibration'}])
            generation['ode_steps'] = 16
            self.assertNotEqual(identity, namespace['cohort_identity'](args, pipe, model, release, spec, [{'lyrics': 'calibration'}]))
            generation['ode_steps'] = 32
            weights['mot']['files']['model.safetensors']['sha256'] = artifacts.digest('changed-weights')
            self.assertNotEqual(identity, namespace['cohort_identity'](args, pipe, model, release, spec, [{'lyrics': 'calibration'}]))
            weights['mot']['files']['model.safetensors']['sha256'] = artifacts.digest('mot')
            (core/'adapter.py').write_text('# changed adapter\n')
            self.assertNotEqual(identity, namespace['cohort_identity'](args, pipe, model, release, spec, [{'lyrics': 'calibration'}]))
            teacher.write_bytes(b'changed teacher')
            with self.assertRaisesRegex(ValueError, 'Teacher checkpoint content differs'):
                namespace['cohort_identity'](args, pipe, model, release, spec, [])

    def test_real_audio_skip_verifies_every_recording_and_refinement_receipt(self):
        @dataclass
        class Generation:
            ode_steps: int = 32
            def to_dict(self):
                return {'ode_steps': self.ode_steps}
        concept = 'example'
        out = self.root/concept
        out.mkdir()
        target = out/f'{concept}_distilled_refined_rank8.safetensors'
        target.write_bytes(b'refined')
        target.with_suffix('.json').write_bytes(b'sidecar')
        artifacts.bind_identity(out/'refinement-identity.json', self.identity)
        artifacts.publish_completion(out/'refined-evaluation.json', self.identity, {},
                                     [target, target.with_suffix('.json')])
        class NativeReached(Exception): pass
        def native(*args):
            raise NativeReached
        d = NS(artifacts=artifacts, rows=lambda *args: [{}, {}], YuE2Slider=NS(load=native))
        namespace = functions('refine_yue2_distillation.py', {'refined_outputs', '_audio'},
            dict(d=d, replace=replace, refinement_identity=lambda *args: self.identity))
        sf = ModuleType('soundfile')
        sf.__version__, sf.__libsndfile_version__ = 'test-soundfile', 'test-libsndfile'
        vae = ModuleType('yue2.modeling_vae')
        vae.YuE2VAE = object
        args, pipe = NS(output=self.root, release=self.root), NS(generation_config=Generation())
        with patch.dict(sys.modules, {'soundfile': sf, 'yue2': ModuleType('yue2'), 'yue2.modeling_vae': vae}):
            with self.assertRaises(NativeReached):
                namespace['_audio'](args, pipe, None, {'id': concept, 'weights': 'teacher.bin'}, target, [])
            audio = out/'audio'
            identity = artifacts.read_json(audio/'audio-identity.json')['identity']
            outputs = [audio/f'row{index}-{kind}{suffix}'
                for index in range(2) for kind in ('off', 'teacher', 'plain-ar', 'plain-all')
                for suffix in ('.flac', '-semantic.npy', '.json')]
            for path in outputs:
                path.write_bytes(b'held-out evidence')
            artifacts.publish_completion(audio/'summary.json', identity, {}, outputs)
            namespace['_audio'](args, pipe, None, {'id': concept, 'weights': 'teacher.bin'}, target, [])
            outputs[0].write_bytes(b'changed recording')
            with self.assertRaisesRegex(ValueError, 'output missing or changed'):
                namespace['_audio'](args, pipe, None, {'id': concept, 'weights': 'teacher.bin'}, target, [])

    def test_audio_restores_generation_settings_after_failure(self):
        original = object()
        pipe = NS(generation_config=original)
        def failed(args, pipe, *rest):
            pipe.generation_config = object()
            raise RuntimeError('injected audio failure')
        namespace = functions('refine_yue2_distillation.py', {'audio'},
                              dict(d=NS(artifacts=artifacts), _audio=failed))
        with self.assertRaisesRegex(RuntimeError, 'injected audio failure'):
            namespace['audio'](NS(output=self.root), pipe, None, {'id': 'example'}, None, [])
        self.assertIs(pipe.generation_config, original)


if __name__ == '__main__':
    assert 'torch' not in sys.modules and 'yue2' not in sys.modules
    unittest.main()
