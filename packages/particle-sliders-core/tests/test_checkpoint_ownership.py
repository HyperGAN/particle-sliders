"""Tiny CPU ownership contracts; no model loaders or training jobs required."""
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter
from particle_sliders.pg09_backends import checkpoint_with_adapter_scale


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7))
        self.scale = 1.

    @contextmanager
    def scaled(self, scale):
        before = self.scale
        self.scale = scale
        try:
            yield
        finally:
            self.scale = before


class Block(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale_owner = scale

    def forward(self, value):
        return (value * self.scale_owner.weight * self.scale_owner.scale).sin().square()


class InterruptMarker(Block):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == '_pg09_scale_checkpoint' and value is True:
            raise KeyboardInterrupt('partial marker installation')


def ownership(module):
    return {key: vars(module)[key] for key in ('forward', '_pg09_scale_checkpoint')
            if key in vars(module)}


@pytest.mark.parametrize('instance_forward', [False, True])
@pytest.mark.parametrize('existing_marker', [False, True])
def test_exact_descriptor_or_override_restore_and_idempotence(instance_forward, existing_marker):
    block = Block(Scale())
    if instance_forward:
        block.forward = block.forward
    if existing_marker:
        block._pg09_scale_checkpoint = False
    before = ownership(block)
    handle = checkpoint_with_adapter_scale(block, block.scale_owner)
    assert block._pg09_scale_checkpoint is True
    handle.close()
    handle.close()
    assert ownership(block) == before
    with pytest.raises(RuntimeError, match='already closed'):
        with handle:
            pass


def test_nested_distinct_blocks_unwind_on_interrupt():
    outer, inner = Block(Scale()), Block(Scale())
    with pytest.raises(KeyboardInterrupt, match='body'):
        with checkpoint_with_adapter_scale(outer, outer.scale_owner):
            with checkpoint_with_adapter_scale(inner, inner.scale_owner):
                raise KeyboardInterrupt('body')
    assert ownership(outer) == ownership(inner) == {}


def test_same_block_nested_refusal_keeps_outer_owner():
    block = Block(Scale())
    with checkpoint_with_adapter_scale(block, block.scale_owner):
        wrapper = block.forward
        with pytest.raises(ValueError, match='already checkpointed'):
            checkpoint_with_adapter_scale(block, block.scale_owner)
        assert block.forward is wrapper
        assert block._pg09_scale_checkpoint is True
    assert ownership(block) == {}


def test_nested_scopes_of_same_handle_restore_only_at_outer_exit():
    block = Block(Scale())
    handle = checkpoint_with_adapter_scale(block, block.scale_owner)
    with handle:
        wrapper = block.forward
        with pytest.raises(KeyboardInterrupt, match='inner'):
            with handle:
                raise KeyboardInterrupt('inner')
        assert block.forward is wrapper
        assert block._pg09_scale_checkpoint is True
    assert ownership(block) == {}
    handle.close()


def test_exit_stack_unwinds_completed_and_partially_installed_blocks():
    first, failing = Block(Scale()), InterruptMarker(Scale())
    with pytest.raises(KeyboardInterrupt, match='partial marker'):
        with ExitStack() as owned:
            owned.enter_context(checkpoint_with_adapter_scale(first, first.scale_owner))
            owned.enter_context(checkpoint_with_adapter_scale(failing, failing.scale_owner))
    assert ownership(first) == ownership(failing) == {}


@pytest.mark.parametrize('changed', ['forward', '_pg09_scale_checkpoint'])
def test_close_refuses_to_overwrite_another_owner(changed):
    block = Block(Scale())
    handle = checkpoint_with_adapter_scale(block, block.scale_owner)
    wrapped = ownership(block)
    setattr(block, changed, (lambda value: value) if changed == 'forward' else False)
    other = ownership(block)
    with pytest.raises(RuntimeError, match='ownership changed'):
        handle.close()
    assert ownership(block) == other
    vars(block).update(wrapped)
    handle.close()
    assert ownership(block) == {}


@pytest.mark.parametrize('strength', [0., -.75, .5, 1.])
def test_deferred_backward_after_handle_close_preserves_captured_scale(strength):
    expected, actual = Scale(), Scale()
    plain, wrapped = Block(expected), Block(actual)
    x = torch.tensor([.2, -.3], requires_grad=True)
    with expected.scaled(strength):
        reference = plain(x).sum()
    reference.backward()
    with checkpoint_with_adapter_scale(wrapped, actual):
        with actual.scaled(strength):
            result = wrapped(x.detach().clone().requires_grad_()).sum()
    # The wrapper is gone, but an already built autograd graph must still replay
    # with the forward's strength, independent of the caller's later strength.
    with actual.scaled(.125):
        result.backward()
        assert actual.scale == .125
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual.weight.grad, expected.weight.grad, rtol=0, atol=0)
    assert ownership(wrapped) == {}


def test_ignored_handle_retains_legacy_eager_installation():
    block = Block(Scale())
    checkpoint_with_adapter_scale(block, block.scale_owner)
    assert block._pg09_scale_checkpoint is True
    block(torch.tensor(.2)).backward()
    assert block.scale_owner.weight.grad is not None


def audio_fixture(failing=False):
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=32)
    # Blocks belong to the native model, but its frozen base is not an optimizer
    # owner. A separate tiny slider supplies the actual adapter leaves.
    model.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    if failing:
        model.blocks[1] = InterruptMarker(Scale())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, Scale()


def test_ar_adapter_owns_and_releases_its_checkpoint_handles():
    model, slider = audio_fixture()
    adapter = ARLastHiddenResponseAdapter(model, slider,
        model_identity={'fixture': 'checkpoint-owner'}, checkpoint_modules=model.blocks)
    assert all(block._pg09_scale_checkpoint is True for block in model.blocks)
    adapter.close_checkpointing()
    adapter.close_checkpointing()
    assert all(ownership(block) == {} for block in model.blocks)


def test_ar_adapter_partial_constructor_unwinds_wrappers():
    model, slider = audio_fixture(failing=True)
    with pytest.raises(KeyboardInterrupt, match='partial marker'):
        ARLastHiddenResponseAdapter(model, slider,
            model_identity={'fixture': 'checkpoint-owner'}, checkpoint_modules=model.blocks)
    assert all(ownership(block) == {} for block in model.blocks)
