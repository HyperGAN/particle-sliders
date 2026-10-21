"""Metadata-only discovery; author now, run in the parent CPU slot."""
from particle_sliders.lifecycle import _rng_devices


def test_text_encoder_device_is_discovered_at_every_native_option_depth():
    assert _rng_devices({'device': 'cpu', 'text_device': 'cuda:4',
                         'native_args': {'text_device': 'cuda:2'}}) == [2, 4]
    assert _rng_devices({'text_device': 'cpu:0', 'device': 'cpu'}) == []
    assert _rng_devices({'model_path': 'cuda:7', 'text_device': 'cuda:3'}) == [3]
