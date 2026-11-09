"""Prospective slider weights are raw; optimizer scalar statistics stay upstream.

Historical averaged artifacts retain their provenance and require their original
archived implementation. They are never migrated or relabelled as raw here.
"""

NO_WEIGHT_EMA = 'Weight EMA is disabled; use ema_decay=None and raw current weights'


def require_no_weight_ema(decay):
    if decay is not None:
        raise ValueError(NO_WEIGHT_EMA)


def require_raw_selection(selection):
    if selection != 'raw':
        raise ValueError('Only raw current weights are supported; averaged artifacts require their archived loader')


def _value(value):
    """Decode safetensors str->str metadata; unrecognized text stays enabled."""
    if not isinstance(value, str):
        return value
    text = value.strip().lower()
    if text in ('', 'none', 'null', 'false'):
        return None
    try:
        return float(text)
    except ValueError:
        return value


def require_raw_provenance(metadata):
    """Refuse known averaged adapters without rewriting historical metadata.

    Historical ``live`` exports explicitly identify raw current weights even
    when the same training run also tracked EMA. Unmarked old EMA exports do
    not acquire raw provenance through a filename or the new default.
    """
    kind = metadata.get('weights_kind')
    if isinstance(kind, str) and kind.strip().lower() in ('ema', 'averaged') or _value(metadata.get('effective_ema')):
        raise ValueError('Averaged adapter is incompatible with raw-only inference; preserve its archived loader')
    enabled = any(_value(metadata.get(key)) not in (None, 0.0) for key in ('ema', 'ema_decay'))
    if kind not in ('live', 'raw') and enabled:
        raise ValueError('Adapter has EMA provenance without an explicit raw/live weight record')


def require_raw_checkpoint(path):
    """Read only adapter provenance; never decode weights or modify old artifacts."""
    import json
    from pathlib import Path
    path = Path(path)
    for sidecar in (path.with_suffix('.json'), Path(str(path) + '.json')):
        if sidecar.is_file():
            require_raw_provenance(json.loads(sidecar.read_text()))
    if path.suffix == '.safetensors' and path.is_file():
        from safetensors import safe_open
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            metadata = handle.metadata() or {}
        require_raw_provenance(metadata)
        if metadata.get('conceptmod'):
            require_raw_provenance(json.loads(metadata['conceptmod']))


def raw_checkpoint_admission(loader):
    """Validate provenance before an inner wrapper can inspect a native host.

    Place this outside attachment transaction decorators and inside classmethod.
    Preserve the loader signature for wrappers that bind host/path arguments.
    """
    from functools import wraps
    from inspect import signature
    parameters = signature(loader)

    @wraps(loader)
    def admitted(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        require_raw_checkpoint(bound.arguments['path'])
        return loader(*args, **kwargs)

    return admitted
