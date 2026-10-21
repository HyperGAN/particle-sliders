"""Installed test plugin: native-only inference fixture, not a model qualification."""
from contextlib import contextmanager
from pathlib import Path
import hashlib

import torch
from torch import nn

from particle_sliders.lifecycle import Backend, Binding, Capabilities
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg09_backends import AdapterParameters, FrozenResponseAdapter

CAPS = Capabilities('adapter_gan', render=True, strength_min=-1., strength_max=1.,
                    base_precisions=('float32',), render_outputs=('latent',))
PROTOCOL = 'bounded-native-fixture-v1'
CLOSED = []


class Native(FrozenResponseAdapter):
    family = 'compact-fixture'
    time_coordinates = 'native fixture time'

    def __init__(self, model_file):
        self.model = nn.Conv2d(2, 2, 1, bias=False).eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
        self.identity = dict(model='CPU native fixture only',
            sha256=hashlib.sha256(Path(model_file).read_bytes()).hexdigest())
        network = AdapterParameters([('edit', nn.Parameter(torch.zeros(2, 2)))], self.scaled)
        super().__init__(network, model_identity=self.identity)

    @contextmanager
    def scaled(self, scale):
        yield

    def _velocity(self, condition, latent, timestep):
        return self.model(latent) + condition + self.network.scale * nn.functional.conv2d(latent, self.network.weights[0].reshape(2, 2, 1, 1))

    def response(self, *, value=.3, strength=1., seed=0):
        with self.network.scaled(strength):
            return self._velocity(torch.zeros(1, 2, 10, 10), torch.full((1, 2, 10, 10), value), torch.tensor(500.)).clone()

    def render(self, *, strength, seed, images, output_type='latent', value=.3):
        assert images is None and output_type == 'latent'
        return self.response(value=value, strength=strength, seed=seed)

    def close(self):
        CLOSED.append(True)


def validate_inference(options, spec):
    if set(options) != {'model_file'} or spec.configuration != {'channels': 2}:
        raise ValueError('only a local model file and native adapter geometry are needed')
    if not Path(options['model_file']).is_file():
        raise ValueError('missing staged native base')


def prepare_inference(options, spec):
    native = Native(options['model_file'])
    return InferenceBinding(native.network,
        native_identity(native, model_identity=native.identity, protocol=PROTOCOL),
        native.close, native.render, native.response)


INFERENCE = InferenceBackend('compact-fixture', CAPS, prepare_inference, validate_inference)


def validate_training(options):
    if not Path(options['training_file']).is_file():
        raise ValueError('missing training data')


def prepare_training(options):
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.field_critic import FieldCritic
    from particle_sliders.pg09_backends import ResponseRow
    native = Native(options['model_file'])
    data = torch.load(options['training_file'], weights_only=True)
    rows = [native.prepare(ResponseRow(torch.zeros_like(value), torch.full_like(value, .2),
            torch.zeros_like(value), value, timestep=torch.tensor(500.))) for value in data]
    critic = FieldCritic(2, 3, width=4)
    game = ConditionalPG09(native, rows, critic, conditions=torch.ones(len(rows), 3),
        particles=None, total_steps=4, batch_size=2, seed=7)
    spec = INFERENCE.specification(native, model_identity=native.identity,
                                  protocol=PROTOCOL, configuration={'channels': 2})
    return Binding(game, native.close, native.render, inference=spec)


TRAINING = Backend('compact-fixture', CAPS, prepare_training, validate_training)
