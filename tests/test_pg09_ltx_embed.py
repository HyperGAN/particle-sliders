from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from particle_sliders.pg09 import ConditionalPG09
from conceptmod.textsliders import ltx25_backend as native_mod
from conceptmod.textsliders import ltx25_uni
from conceptmod.textsliders.pg09_ltx_embed import LTX25ConnectorResponseAdapter, LTXEmbedRow
from particle_sliders.pg09_text_video import ResponseCompatibilityError
from test_pg09_text_video import Critic, equal


def make_connector(preset='bispf_optimized', backend=None, *, shape=(1, 1026, 8)):
    torch.set_num_threads(1)
    if backend is None:
        torch.manual_seed(21)
        backend = native_mod.LTX25Backend(dummy=True, lora_rank=2, lora_alpha=2, te_last_n=1)
    adapter = LTX25ConnectorResponseAdapter(backend, native_shape=shape,
                    model_identity={'qualification': 'CPU random native text components'},
                    pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    rows = [adapter.prepare(LTXEmbedRow('portrait person neutral', 'portrait person smile', 'smile')),
            adapter.prepare(LTXEmbedRow('portrait child neutral', 'portrait child happy', 'happy'))]
    conditions = torch.cat([torch.cat((r.neutral.flatten(), r.row.common_mask.flatten().float(),
                                      torch.tensor([1., 0.])))[None] for r in rows])
    game = ConditionalPG09(adapter, rows, Critic(rows[0].target.numel(), conditions.shape[1]),
                           conditions=conditions, particles=None, **adapter.domain_options,
                           preset=preset, total_steps=4, batch_size=2, seed=15)
    return adapter, rows, game


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_default_connector_gan_gradients_and_no_mse_or_hold_objective(preset, monkeypatch):
    adapter, rows, game = make_connector(preset)
    modules = [adapter.backend.encoder, adapter.backend.connectors, adapter.backend.transformer]
    frozen = [(p, p.detach().clone()) for m in modules for p in m.parameters() if not p.requires_grad]
    before = deepcopy(adapter.network.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError('MSE/hold objectives forbidden')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    monkeypatch.setattr(ltx25_uni, 'ltx25_embed_match_loss', forbidden)
    monkeypatch.setattr(ltx25_uni, 'ltx25_unused_hold_loss', forbidden)
    for _ in range(3):
        metrics = game.step()
        assert metrics['g_loss'] == metrics['g_adv']
        assert metrics['prior_regularization'] == 0
        assert metrics['d_pen'] > 0
        assert game.penalty.last_stats['resolved_domain'] == 'continuous'
    assert all(p.grad is not None for p in adapter.network.parameters())
    assert any(not torch.equal(before[k], v) for k, v in adapter.network.state_dict().items())
    assert all(torch.equal(p, value) for p, value in frozen)
    assert game.prior is None
    assert game.state_dict()['config']['generator_arm'] == 'adapter_gan'
    assert torch.equal(adapter.render(rows[0], scale=0), rows[0].neutral)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_connector_exact_resume_weights_only_and_export(preset, tmp_path):
    adapter, rows, game = make_connector(preset)
    game.step()
    before = game.state_dict()
    torch.save(before, tmp_path / 'checkpoint.pt')
    checkpoint = torch.load(tmp_path / 'checkpoint.pt', weights_only=True)
    game.evaluate(0, seed=23)
    equal(before, game.state_dict())
    metrics = game.step()
    expected = game.state_dict()
    other, _, resumed = make_connector(preset)
    resumed.load_state_dict(checkpoint)
    assert resumed.step() == metrics
    equal(expected, resumed.state_dict())
    resumed.load_export_state_dict(game.export_state_dict())
    for scale in (0, .5, 1):
        assert torch.equal(adapter.render(rows[0], scale=scale), other.render(rows[0], scale=scale))
    assert game.state_dict()['config']['backend']['native_shape'] == [1, 1026, 8]


def test_lossless_vectorization_fixed_mask_and_native_preconnector_hold(monkeypatch):
    torch.manual_seed(21)
    backend = native_mod.LTX25Backend(dummy=True, lora_rank=2, lora_alpha=2, te_last_n=1)
    seen = []
    original = backend._run_connectors
    def connector(hidden, mask):
        seen.append((hidden.detach().clone(), mask.detach().clone()))
        return original(hidden, mask)
    monkeypatch.setattr(backend, '_run_connectors', connector)
    adapter, rows, _ = make_connector(backend=backend)
    prepared = rows[0]
    row = prepared.row
    with torch.no_grad(), adapter.network.scaled(0):
        positive = backend.encode_text(row.caption.positive, frozen=True)
        neutral = backend.encode_text(row.caption.neutral, frozen=True)
        held = ltx25_uni.apply_unused_hold(positive.embeds, neutral.embeds,
                        positive.token_ids, neutral.token_ids, row.hold_mask)
        padded, mask = native_mod.pad_pre_connector_sequence(held, positive.attention_mask, padding_side='left')
    assert torch.equal(seen[0][0], padded)
    assert torch.equal(seen[0][1], mask)
    assert torch.equal(row.common_mask, row.student_mask & row.target_mask)
    raw, _ = adapter.conditioning(prepared, scale=1)
    vector = adapter.generated(prepared)
    assert vector.shape == (1, 1026 * 8)
    assert torch.equal(vector.reshape(1, 1026, 8), raw * row.common_mask[..., None])
    assert torch.count_nonzero(vector.reshape(1, 1026, 8)[~row.common_mask]) == 0
    assert torch.equal(vector.reshape(1, 1026, 8)[row.common_mask], raw[row.common_mask])
    assert row.hold_mask.any() and (~row.hold_mask).any()
    assert adapter.metadata()['hold_objective'].startswith('none')


def test_changed_mask_shape_and_empty_intersection_fail_closed(monkeypatch):
    adapter, rows, _ = make_connector()
    original = adapter.backend.encode_post_connector_video
    def changed(*a, **kw):
        value, mask = original(*a, **kw)
        mask = mask.clone()
        mask[:, 0] = 1 - mask[:, 0]
        return value, mask
    monkeypatch.setattr(adapter.backend, 'encode_post_connector_video', changed)
    with pytest.raises(ResponseCompatibilityError, match='mask changed'):
        adapter.generated(rows[0])
    def empty(*a, **kw):
        value, mask = original(*a, **kw)
        return value, torch.zeros_like(mask)
    monkeypatch.setattr(adapter.backend, 'encode_post_connector_video', empty)
    with pytest.raises(ResponseCompatibilityError, match='no common valid'):
        adapter.prepare(rows[0].row.caption)
    monkeypatch.setattr(adapter.backend, 'encode_post_connector_video', original)
    wrong = replace(rows[0], row=replace(rows[0].row, native_shape=(1, 1, 1)))
    with pytest.raises(ResponseCompatibilityError, match='geometry differs'):
        adapter.generated(wrong)


def test_release_default_stops_before_native_backend_access():
    with pytest.raises(ResponseCompatibilityError, match='width 8208'):
        LTX25ConnectorResponseAdapter(object(), native_shape=(1, 1026, 8))


def native_components_backend(monkeypatch):
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    torch.set_num_threads(1)
    torch.manual_seed(29)
    encoder = Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
                         head_dim=8, global_head_dim=16, layer_types=['sliding_attention', 'full_attention'],
                         max_position_embeddings=128, sliding_window=32, attention_dropout=0.,
                         attention_k_eq_v=True, enable_moe_block=False, hidden_size_per_layer_input=0),
        vision_config=None, audio_config=None))
    connectors = LTX2TextConnectors(caption_channels=32, text_proj_in_factor=3,
        video_connector_num_attention_heads=4, video_connector_attention_head_dim=8,
        video_connector_num_layers=1, video_connector_num_learnable_registers=128,
        video_hidden_dim=32, video_gated_attn=True,
        audio_connector_num_attention_heads=2, audio_connector_attention_head_dim=8,
        audio_connector_num_layers=1, audio_connector_num_learnable_registers=128,
        audio_hidden_dim=16, audio_gated_attn=True, per_modality_projections=True, proj_bias=True, rope_type='split')
    transformer = LTX2VideoTransformer3DModel(num_layers=1, num_attention_heads=2, attention_head_dim=8,
        cross_attention_dim=32, audio_num_attention_heads=2, audio_attention_head_dim=8,
        audio_cross_attention_dim=16, caption_channels=32, rope_type='split', use_prompt_embeddings=False)
    pipe = SimpleNamespace(text_encoder=encoder, connectors=connectors, transformer=transformer,
                           tokenizer=native_mod.DummyTokenizer(), vae=None, audio_vae=None, vocoder=None)
    monkeypatch.setattr(native_mod, '_load_ltx25_pipeline', lambda *a, **kw: pipe)
    return native_mod.LTX25Backend(device='cpu', encoder_device='cpu', dummy=False, components=pipe,
                                   lora_rank=2, lora_alpha=2, te_last_n=1)


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_real_native_gemma_and_connector_classes_random_weights_gan_step(preset, monkeypatch):
    backend = native_components_backend(monkeypatch)
    adapter, rows, game = make_connector(preset, backend=backend, shape=(1, 1024, 32))
    base = [(p, p.detach().clone()) for m in (backend.encoder, backend.connectors, backend.transformer)
            for p in m.parameters() if not p.requires_grad]
    def forbidden(*a, **kw):
        raise AssertionError('MSE forbidden on actual native text classes')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    result = game.step()
    assert result['g_loss'] == result['g_adv']
    assert result['d_pen'] > 0
    assert all(p.grad is not None for p in adapter.network.parameters())
    assert all(torch.equal(p, value) for p, value in base)
    assert rows[0].target.shape == (1, 32768)
    sources = game.state_dict()['config']['implementation_sources']
    assert any('gemma4' in name for name in sources)
    assert any('connectors' in name for name in sources)
    assert not torch.cuda.is_initialized()
