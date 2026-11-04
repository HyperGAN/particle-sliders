"""Finite JSON requests, frozen product profiles and explicit input staging."""
from copy import deepcopy
import hashlib
from importlib.metadata import entry_points
import json
import os
from pathlib import Path
import re

PROFILE_SCHEMA = 'particle-slider-profile-v1'
REQUEST_SCHEMA = 'particle-slider-train-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + v)))


def validate_profile(profile):
    profile = deepcopy(profile)
    if not isinstance(profile, dict) or profile.get('schema') != PROFILE_SCHEMA:
        raise ValueError('expected ' + PROFILE_SCHEMA)
    for key in ('name', 'product'):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ValueError('profile requires ' + key)
    if profile.get('config_kind') not in ('session', 'native'):
        raise ValueError('profile config_kind must be session or native')
    if profile.get('status') not in ('native', 'fixture', 'source_only', 'blocked'):
        raise ValueError('profile must declare native, fixture, source_only or blocked status')
    if profile['status'] == 'blocked' and not profile.get('reason'):
        raise ValueError('blocked profile requires concrete reason')
    runtime = profile.setdefault('runtime', {})
    if not isinstance(runtime, dict):
        raise ValueError('runtime must be an object')
    for key in ('python', 'cwd', 'device'):
        if key in runtime and (not isinstance(runtime[key], str) or not runtime[key]):
            raise ValueError('runtime ' + key + ' must be a nonempty string')
    env = runtime.get('env', {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError('runtime env must contain string names and values')
    if profile['config_kind'] == 'native' and profile['status'] != 'blocked':
        command = profile.get('command')
        if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
            raise ValueError('native profile requires explicit argv command')
    if profile['config_kind'] == 'session':
        if profile.get('interface', 'image') not in ('image', 'response'):
            raise ValueError('session interface must be image or response')
        if not isinstance(profile.get('backend'), str) or not profile['backend']:
            raise ValueError('session profile requires backend')
    capabilities = profile.setdefault('capabilities', {})
    if not isinstance(capabilities, dict) or any(type(v) is not bool for v in capabilities.values()):
        raise ValueError('profile capabilities must contain boolean values')
    interval = profile.setdefault('default_checkpoint_every', 32)
    if type(interval) is not int or interval < 1:
        raise ValueError('default_checkpoint_every must be positive')
    digest(profile)
    return profile


def load_profile(reference, *, base=None):
    if isinstance(reference, dict):
        return validate_profile(reference), str(Path(base or '.').resolve())
    if not isinstance(reference, str) or not reference:
        raise ValueError('training profile is required')
    path = Path(reference).expanduser()
    if not path.is_absolute() and not path.is_file():
        path = Path(base or '.') / path
    if path.is_file():
        return validate_profile(read_json(path)), str(path.resolve().parent)
    matches = list(entry_points(group='particle_sliders.training_profiles', name=reference))
    if len(matches) != 1:
        raise ValueError(f'expected one training profile {reference!r}; found {len(matches)}; pass a training_profile.json path')
    value = matches[0].load()
    if callable(value):
        value = value()
    return validate_profile(value), str(Path(base or '.').resolve())


def _target(training, profile):
    path = profile.get('target_steps_path', 'options.steps' if profile['config_kind'] == 'session' else 'steps')
    value = training
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise ValueError('training requires explicit target budget at ' + path)
        value = value[key]
    if type(value) is not int or value < 1:
        raise ValueError('target update budget must be a positive integer')
    return value


def _profile_identity(profile):
    profile = deepcopy(profile)
    profile.pop('reporting', None)
    runtime = profile.get('runtime', {})
    for key in ('python', 'cwd', 'device'):
        runtime.pop(key, None)
    return profile


def semantic_identity(request):
    return dict(schema=REQUEST_SCHEMA, profile=_profile_identity(request['resolved']['profile']),
                training=request['training'], inputs=[{k: row[k] for k in ('name', 'sha256', 'size')}
                                                    for row in request.get('input_files', [])])


def validate_resolved(request):
    if request.get('schema') != REQUEST_SCHEMA or not isinstance(request.get('resolved'), dict):
        raise ValueError('expected resolved training request')
    profile = validate_profile(request['resolved']['profile'])
    if profile['name'] != request.get('profile'):
        raise ValueError('profile identity differs from resolved request')
    if digest(semantic_identity(request)) != request['resolved'].get('semantic_sha256'):
        raise ValueError('resolved training semantics changed')
    runtime_hash = digest(_profile_identity(profile).get('runtime', {}))
    if runtime_hash != request['resolved'].get('runtime_sha256'):
        raise ValueError('resolved compute identity changed')
    if _target(request['training'], profile) != request['resolved'].get('target_steps'):
        raise ValueError('resolved target budget changed')
    if not isinstance(request.get('execution'), dict) or not isinstance(request.get('output'), dict):
        raise ValueError('execution and output must be objects')
    execution, output = request['execution'], request['output']
    if not isinstance(execution.get('runner'), str) or not execution['runner']:
        raise ValueError('execution runner must be named')
    if not isinstance(output.get('root'), str) or not output['root']:
        raise ValueError('output root must be named')
    if type(output.get('checkpoint_every')) is not int or output['checkpoint_every'] < 1:
        raise ValueError('checkpoint interval must be positive')
    for key, choices in (('selection', ('raw', 'ema')), ('export_format', ('v1', 'v2', 'both'))):
        if profile['config_kind'] == 'session' or key in output:
            if output.get(key) not in choices:
                raise ValueError('unsupported artifact selection or export format')
            if profile['config_kind'] == 'native' and not profile['capabilities'].get(key, False):
                raise ValueError('native profile owns ' + key + '; override is unsupported')
    device = execution.get('device')
    fixed = profile.get('runtime', {}).get('device') if execution['runner'] == 'local' else None
    if device is not None and (not isinstance(device, str) or not device):
        raise ValueError('execution device must be a nonempty string')
    if fixed and device != fixed:
        raise ValueError('requested device differs from selected fixed runtime profile')
    if profile['config_kind'] == 'native' and execution['runner'] == 'local' and device and not fixed and not profile['capabilities'].get('device', False):
        raise ValueError('native profile does not support a device override')
    if profile['config_kind'] == 'session' and device and request['training'].get('options', {}).get('device') != device:
        raise ValueError('execution device differs from effective session options.device')
    if 'stop_after' in execution and (type(execution['stop_after']) is not int or
            not 0 < execution['stop_after'] <= request['resolved']['target_steps']):
        raise ValueError('invalid stop-after budget')
    reporting = request.get('reporting')
    if reporting is not None:
        validate_reporting(reporting)
        if digest(reporting) != request['resolved'].get('reporting_sha256'):
            raise ValueError('resolved reporting configuration changed')
    elif request['resolved'].get('reporting_sha256') is not None:
        raise ValueError('reporting configuration is missing')
    return profile


def validate_reporting(value):
    if not isinstance(value, dict) or value.get('kind') != 'hypergan-develop':
        raise ValueError('reporting kind must be hypergan-develop')
    required = {'kind', 'runtime_python', 'runtime_path', 'runtime_commit', 'runtime_sha256'}
    if set(value) != required or any(not isinstance(value.get(k), str) or not value[k] for k in required):
        raise ValueError('reporting requires exact runtime_python, runtime_path, runtime_commit and runtime_sha256')
    if re.fullmatch('[0-9a-f]{40}', value['runtime_commit']) is None or re.fullmatch('[0-9a-f]{64}', value['runtime_sha256']) is None:
        raise ValueError('reporting requires full source commit and SHA256')
    return value


def resolve_request(config, *, base=None, profile=None, runner=None, output=None, resume=None,
                    checkpoint_every=None, device=None, selection=None, export_format=None, stop_after=None, runtime_python=None, reporting=None, no_reporting=False, default_reporting=None):
    base = Path(base or '.').resolve()
    value = deepcopy(config)
    if not isinstance(value, dict):
        raise ValueError('training configuration must be an object')
    if value.get('schema') == REQUEST_SCHEMA:
        if set(value) - {'schema', 'profile', 'training', 'execution', 'output', 'input_files', 'reporting'}:
            raise ValueError('unknown training envelope fields')
        request = value
        ref = profile or request.get('profile')
    else:
        request = dict(schema=REQUEST_SCHEMA, training=value, execution={}, output={})
        ref = profile
        if ref is None and set(value) == {'backend', 'cohort', 'options'}:
            ref = dict(schema=PROFILE_SCHEMA, name=value['backend'], product=value['backend'],
                       config_kind='session', backend=value['backend'], interface='image', status='source_only',
                       runtime={}, capabilities={'resume': True, 'periodic_checkpoint': True, 'check': True})
        if ref is None:
            ref = str(base / 'training_profile.json')
    spec, directory = load_profile(ref, base=base)
    if runtime_python is not None:
        if not isinstance(runtime_python, str) or not runtime_python:
            raise ValueError('runtime-python must be a nonempty executable path')
        spec['runtime']['python'] = runtime_python
    request['profile'] = spec['name']
    if not isinstance(request.get('training'), dict):
        raise ValueError('training must contain the product-owned configuration')
    request.setdefault('execution', {})
    request.setdefault('output', {})
    if not isinstance(request['execution'], dict) or not isinstance(request['output'], dict):
        raise ValueError('execution and output must be objects')
    execution, out = request['execution'], request['output']
    execution['runner'] = runner or execution.get('runner', 'local')
    if not isinstance(execution['runner'], str) or not execution['runner']:
        raise ValueError('runner must be a nonempty name')
    if device is not None:
        execution['device'] = device
    fixed = spec.get('runtime', {}).get('device') if execution['runner'] == 'local' else None
    if fixed and execution.get('device', fixed) != fixed:
        raise ValueError('requested device differs from selected fixed runtime profile')
    if fixed:
        execution.setdefault('device', fixed)
    requested_device = execution.get('device')
    if spec['config_kind'] == 'session':
        if not isinstance(request['training'].get('options'), dict):
            raise ValueError('session training requires an options object')
        options = request['training']['options']
        if requested_device:
            if fixed and device is None and options.get('device', fixed) != fixed:
                raise ValueError('session options.device differs from selected fixed runtime profile')
            options['device'] = requested_device
        elif options.get('device'):
            execution['device'] = options['device']
    elif execution['runner'] == 'local' and requested_device and not fixed and not spec['capabilities'].get('device', False):
        raise ValueError('native profile does not support a device override')
    if output is None and 'root' not in out:
        import datetime
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        out['root'] = str(base / 'runs' / (re.sub(r'[^A-Za-z0-9._-]', '-', spec['name']) + '-' + stamp))
    else:
        out['root'] = str((base / str(output or out['root'])).resolve())
    out['checkpoint_every'] = checkpoint_every if checkpoint_every is not None else out.get('checkpoint_every', spec['default_checkpoint_every'])
    if type(out['checkpoint_every']) is not int or out['checkpoint_every'] < 1:
        raise ValueError('checkpoint interval must be a positive integer')
    for key, override, default in [('selection', selection, 'raw'), ('export_format', export_format, 'v1')]:
        explicit = override is not None or key in out
        supported = spec['config_kind'] == 'session' or spec['capabilities'].get(key, False)
        if explicit and not supported:
            raise ValueError('native profile owns ' + key + '; override is unsupported')
        if supported:
            out[key] = override if override is not None else out.get(key, default)

    if resume is not None:
        execution['resume'] = str(resume) if '://' in str(resume) else str((base / str(resume)).resolve())
    if execution.get('resume') and execution['runner'] == 'local' and '://' in execution['resume']:
        raise ValueError('local resume requires a local checkpoint path')
    if execution.get('resume') and not spec['capabilities'].get('resume', False):
        raise ValueError('selected profile does not support resume')
    if stop_after is not None:
        if type(stop_after) is not int or stop_after < 1:
            raise ValueError('stop-after must be positive')
        execution['stop_after'] = stop_after
    target = _target(request['training'], spec)
    if execution.get('stop_after', target) > target:
        raise ValueError('stop-after exceeds total update budget')
    inputs, names = [], set()
    for row in request.get('input_files', []):
        if not isinstance(row, dict) or not isinstance(row.get('name'), str):
            raise ValueError('input_files requires name and local_path')
        name = row['name']
        if name in names or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*', name) or '..' in Path(name).parts:
            raise ValueError('input name must be unique safe relative path')
        names.add(name)
        path = (base / row['local_path']).resolve()
        if not path.is_file():
            raise ValueError('missing input file: ' + name)
        size, checksum = path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get('sha256', checksum) != checksum or row.get('size', size) != size:
            raise ValueError('input checksum or size differs: ' + name)
        inputs.append(dict(name=name, local_path=str(path), size=size, sha256=checksum))
    request['input_files'] = inputs
    reporting_base = base
    selected_reporting = reporting if reporting is not None else request.get('reporting')
    if selected_reporting is None and 'reporting' not in request:
        if 'reporting' in spec:
            selected_reporting = spec['reporting']
            reporting_base = Path(directory)
        else:
            selected_reporting = default_reporting() if callable(default_reporting) else default_reporting
    if no_reporting:
        selected_reporting = None
    if selected_reporting is not None:
        selected_reporting = deepcopy(validate_reporting(selected_reporting))
        selected_reporting['runtime_path'] = str((reporting_base / selected_reporting['runtime_path']).resolve())
        if '/' in selected_reporting['runtime_python']:
            selected_reporting['runtime_python'] = os.path.abspath(reporting_base / os.path.expanduser(selected_reporting['runtime_python']))
        request['reporting'] = selected_reporting
    elif no_reporting or 'reporting' in request:
        request['reporting'] = None
    request['resolved'] = dict(profile=spec, profile_dir=directory, config_dir=str(base), target_steps=target)
    request['resolved']['semantic_sha256'] = digest(semantic_identity(request))
    request['resolved']['runtime_sha256'] = digest(_profile_identity(spec).get('runtime', {}))
    if selected_reporting is not None:
        request['resolved']['reporting_sha256'] = digest(selected_reporting)
    validate_resolved(request)
    return request


def materialize_training(request):
    files = {}
    for row in request.get('input_files', []):
        path = Path(row['local_path'])
        if path.stat().st_size != row['size'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('staged input changed: ' + row['name'])
        files[row['name']] = str(path.resolve())
    def visit(value):
        if isinstance(value, str) and value.startswith('input://'):
            name = value[8:]
            if name not in files:
                raise ValueError('undeclared input reference: ' + name)
            return files[name]
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value
    return visit(request['training'])
