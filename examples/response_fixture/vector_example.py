"""Installed synthetic continuous-vector backend; no native/pretrained quality claim."""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch
from torch import nn

from particle_sliders import pg09
from particle_sliders.lifecycle import Binding
from particle_sliders.response import ResponseBackend, ResponseCapabilities
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg09_backends import AdapterParameters
from particle_sliders.pg09_audio import AudioRow, PreparedAudioResponse, AudioResponseCritic

PROFILE = getattr(pg09, 'PG_SOURCE_PROFILE', 'release')
CAPS = ResponseCapabilities('adapter_gan', 'synthetic_vector', 'native_final_hidden_fixture',
    'continuous_vector', (1, 32), pg09.PG_VERSION, PROFILE)
PROTOCOL = 'synthetic-vector-response-v1'
CLOSED = []


class Native:
    def __init__(self, model_file):
        value = Path(model_file).read_bytes()
        self.base = float(value)
        self.identity = dict(fixture='local scalar base; no pretrained model', sha256=hashlib.sha256(value).hexdigest())
        self.network = AdapterParameters([('edit', nn.Parameter(torch.zeros(32)))], self.scaled)

    @contextmanager
    def scaled(self, scale):
        yield

    def generated(self, prepared):
        with self.network.scaled(prepared.row.scale):
            return self.response(strength=prepared.row.scale, value=float(prepared.neutral[0, 0]))

    def response(self, *, strength=1., seed=0, value=None):
        neutral = torch.full((1, 32), self.base if value is None else value)
        delta = nn.functional.dropout(self.network.weights[0], .1, training=self.network.training)
        return neutral + delta.reshape(1, 32) * strength

    def metadata(self):
        return dict(model_identity=self.identity, protocol=PROTOCOL, response_domain='continuous_vector')

    def close(self):
        CLOSED.append(True)


def validate_inference(options, spec):
    if set(options) != {'model_file'} or spec.configuration != {'width': 32}:
        raise ValueError('native response needs only its local base and width')
    float(Path(options['model_file']).read_bytes())


def prepare_inference(options, spec):
    native = Native(options['model_file'])
    return InferenceBinding(native.network,
        native_identity(native, model_identity=native.identity, protocol=PROTOCOL),
        native.close, response=native.response)


INFERENCE = InferenceBackend('vector-fixture', CAPS, prepare_inference, validate_inference)


def validate_training(options):
    if not Path(options['training_file']).is_file():
        raise ValueError('missing local fixture training data')


def prepare_training(options):
    native = Native(options['model_file'])
    # These paired IDs label synthetic vector rows, not model tokens or waveforms.
    count = int(Path(options['training_file']).read_text())
    rows = [PreparedAudioResponse(AudioRow((1, i + 10), (2, i + 10)),
        torch.full((1, 32), native.base + .2 + .01 * i),
        torch.full((1, 32), native.base + .01 * i)) for i in range(count)]
    critic = AudioResponseCritic(32, 3)
    game = pg09.ConditionalPG09(native, rows, critic, conditions=torch.ones(count, 3),
        particles=None, response_domain='continuous_vector', total_steps=4, batch_size=2,
        pg_source_profile=PROFILE)
    spec = INFERENCE.specification(native, model_identity=native.identity,
        protocol=PROTOCOL, configuration={'width': 32})
    return Binding(game, native.close, inference=spec)


TRAINING = ResponseBackend('vector-fixture', CAPS, prepare_training, validate_training)
