"""Optional HNDL architectures for conditional, bounded-response GAN critics.

Construction uses the released HNDL resolver and PyTorch builder. Importing this
module does not import HNDL. Native responses are flattened only inside the
critic; the game and its public penalty keep their original response geometry.
"""
import ast
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from textwrap import dedent

from torch import nn


HNDL_VERSION = '0.8.2'
# This helper authors small readout networks, rather than loading native models.
_READOUT_OPS = frozenset(('concat', 'linear', 'silu', 'gelu', 'relu',
                          'leaky_relu', 'tanh', 'layer_norm', 'add', 'mul'))


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _readout_source(source):
    if not isinstance(source, str) or not source.strip() or len(source.encode('utf-8')) > 16384:
        raise ValueError('HNDL readout source must be nonempty UTF-8 text of at most 16384 bytes')
    # Check before resolving: pretrained providers may resolve checkpoint metadata.
    # HNDL still owns parsing, graph validation, shape resolution and execution.
    # Match HNDL's public resolver: indented triple-quoted configs are valid.
    # Keep the caller's original text for resolution and checkpoint identity.
    tree = ast.parse(dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                or node.func.id not in _READOUT_OPS | {'range'}):
            raise ValueError('HNDL readout source permits only built-in readout operations')
    return source


class HNDLConditionalReadoutCritic(nn.Module):
    """A source-defined raw-logit critic accepted by ``ConditionalPG09``.

    ``response_width`` counts all non-batch response coordinates. The default
    matches the existing Linear/SiLU/Linear readout. Custom HNDL source receives
    named ``response`` and ``condition`` matrices and must end at a scalar linear
    readout. No pretrained providers, custom operators or model downloads occur.

    ``architecture()`` returns the source and resolved plan for experiment
    records. Their digests, HNDL's version and all its Python source digests are
    included in ``extra_repr()``, which the shared game signs for strict resume.
    A changed critic requires a fresh game/cohort, including legacy MLP critics.
    """
    score_bound = 0

    def __init__(self, response_width, condition_width, hidden=16, *, source=None):
        super().__init__()
        if any(type(value) is not int or value < 1
               for value in (response_width, condition_width, hidden)):
            raise ValueError('response_width, condition_width and hidden must be positive integers')
        if not (3, 11) <= sys.version_info[:2] < (3, 15):
            raise RuntimeError('the optional HNDL critic requires Python >=3.11,<3.15')
        try:
            import hndl
            from hndl.torch import build
        except ModuleNotFoundError as exc:
            if exc.name != 'hndl':
                raise
            raise ImportError('install particle-sliders-core[hndl] to use the optional HNDL critic') from exc
        hndl_version = version('hndl')
        if hndl_version != HNDL_VERSION:
            raise RuntimeError('the qualified HNDL critic requires hndl==' + HNDL_VERSION)
        source = _readout_source(source if source is not None else (
            'joined = concat(response, condition, name="joined")\n'
            f'features = linear(joined, {hidden}, name="hidden")\n'
            'features = silu(features, name="activation")\n'
            'linear(features, 1, name="logits")\n'))
        plan = hndl.resolve(source,
            input_shape={'response': ('B', response_width), 'condition': ('B', condition_width)},
            output_shape=('B', 1))
        plan_document = json.loads(plan.to_json())
        output_node = next((node for node in plan.nodes
                            if plan_document['output_ref'] == 'node:' + node.id + '/out'), None)
        if output_node is None or output_node.op != 'linear@1':
            raise ValueError('HNDL critic output must be a scalar linear raw-logit readout')
        self.network = build(plan, device='cpu')
        self.response_width, self.condition_width = response_width, condition_width
        root = Path(hndl.__file__).resolve().parent
        sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(root.rglob('*.py'))}
        if not sources:
            raise RuntimeError('HNDL Python sources are required for strict checkpoint identity')
        self._architecture = dict(protocol='hndl-conditional-readout-v1', source=source,
            plan=plan_document, response_width=response_width, condition_width=condition_width,
            hndl_version=hndl_version, hndl_sources=sources)
        self._identity = dict(source_sha256=_digest(source),
            plan_sha256=_digest(plan.to_json()), hndl_version=hndl_version,
            hndl_sources_sha256=_digest(json.dumps(sources, sort_keys=True)))

    def architecture(self):
        """JSON-compatible construction record, independent of mutable weights."""
        return deepcopy(self._architecture)

    def extra_repr(self):
        return json.dumps(dict(response_width=self.response_width,
                               condition_width=self.condition_width, **self._identity), sort_keys=True)

    def forward(self, response, condition):
        if (response.ndim < 2 or condition.ndim != 2 or
                response.shape[0] != condition.shape[0] or
                response.flatten(1).shape[1] != self.response_width or
                condition.shape[1] != self.condition_width):
            raise ValueError('response and condition differ from the HNDL readout shape contract')
        # The public low-level builder returns a port mapping. The single
        # unnamed output constraint has HNDL's default port name ``output``.
        return self.network(response=response.flatten(1), condition=condition)['output'].squeeze(-1)
