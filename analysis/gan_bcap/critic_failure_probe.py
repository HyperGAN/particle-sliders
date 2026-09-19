#!/usr/bin/env python3
"""Cross saved critics and adapter spans on fixed training prompts, without updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lm_evaluate import cache_teachers, checkpoint_metadata, topology
from conceptmod.textsliders.train_lm_slider_music3 import (
    DEFAULT_MODEL, _load_rows, _encode_full, _set_scale, _span_delta_batch,
)
from conceptmod.textsliders.lm_adv import SpanTransformerD, rp_g_loss


def rms(x):
    return float(x.detach().square().mean().sqrt())


def probe(critic, fake, real, mask):
    x = fake.detach().clone().requires_grad_(True)
    with torch.no_grad():
        real_features = critic.features(real, mask)
        real_logits = critic(real, mask)
    features = critic.features(x, mask)
    logits = critic.head(critic.out_norm(features)).squeeze(-1)
    fm = F.mse_loss(features.mean(0), real_features.mean(0))
    adv = rp_g_loss(logits, real_logits)
    fm_grad = torch.autograd.grad(fm, x, retain_graph=True)[0]
    adv_grad = torch.autograd.grad(adv, x, retain_graph=True)[0]
    score_grad = torch.autograd.grad(logits.sum(), x)[0]
    norms = score_grad.flatten(1).norm(dim=-1) * critic.input_scale
    with torch.no_grad():
        normalized_fm = F.mse_loss(critic.out_norm(features).mean(0),
                                   critic.out_norm(real_features).mean(0))
        doubled_scores = critic.head(critic.out_norm(features * 2)).squeeze(-1)
        doubled_fm = F.mse_loss((2 * features).mean(0), (2 * real_features).mean(0))
    return dict(fake_feature_rms=rms(features), real_feature_rms=rms(real_features),
                raw_feature_matching=float(fm.detach()), normalized_feature_matching=float(normalized_fm),
                adversarial_loss=float(adv.detach()), score_real=float(real_logits.mean()),
                score_fake=float(logits.detach().mean()), fm_input_gradient_norm=float(fm_grad.norm()),
                adversarial_input_gradient_norm=float(adv_grad.norm()),
                calibrated_score_gradient_mean=float(norms.mean()), calibrated_score_gradient_max=float(norms.max()),
                fake_cap_penalty=float(F.relu(norms - 1).square().mean()),
                doubled_pooled_features_score_max_abs_change=float((doubled_scores - logits.detach()).abs().max()),
                doubled_both_feature_sets_fm_ratio=float(doubled_fm / fm.detach()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, nargs='+', required=True)
    parser.add_argument('--weights', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors.torch import load_file
    from conceptmod.textsliders.lora import LoRANetwork
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    metadata = [checkpoint_metadata(p) for p in args.weights]
    shape = topology(metadata[0][0])
    assert all(topology(m[0]) == shape for m in metadata)
    prompts = ROOT / metadata[0][0]['prompts_file']
    rows, _ = _load_rows(prompts)
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_MODEL / 'tokenizer'), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(str(DEFAULT_MODEL / 'language_model'),
                                            torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    lm.requires_grad_(False); lm.config.use_cache = False
    cached = cache_teachers(lm, tokenizer, [dict(id=f'train_{i}', group='train', row=r) for i,r in enumerate(rows)], device)
    network = LoRANetwork(lm, multiplier=0., **shape).to(device).eval().requires_grad_(False)
    span_sets = []
    for path, (_, _, step) in zip(args.weights, metadata):
        network.load_state_dict(load_file(str(path)), strict=True)
        with torch.no_grad():
            _set_scale(network, 0.)
            assert all(torch.equal(_encode_full(lm, r['input_ids'], r['attention_mask']).cpu(), r['neutral_hidden']) for r in cached)
            _set_scale(network, 1.)
            predicted = [_encode_full(lm, r['input_ids'], r['attention_mask']).cpu() for r in cached]
        fake, real, mask = _span_delta_batch(predicted, [r['positive_hidden'] for r in cached],
                                           [r['neutral_hidden'] for r in cached], [r['neutral_span'] for r in cached],
                                           [r['positive_span'] for r in cached], 'cpu')
        span_sets.append(dict(step=step, weights=str(path.resolve()), fake=fake, real=real, mask=mask))
        print(f'Captured adapter {step} lyric-span plus audio-start deltas', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spans_path = args.output.with_suffix('.pt')
    torch.save(span_sets, spans_path)
    del lm, network
    torch.cuda.empty_cache()
    results=[]; states=[]
    for path in args.states:
        state=torch.load(path, map_location='cpu', weights_only=True)
        settings=state['signature']['settings']; sd=state['modules']['critic']
        assert settings['adv_arch']=='tx' and settings['adv_condition']=='none'
        critic=SpanTransformerD(sd['proj.weight'].shape[1], width=settings['adv_width'],
                               n_layers=settings['adv_layers'], n_heads=settings['adv_heads'],
                               in_mode=settings['adv_in'], readout=settings['adv_readout']).to(device)
        critic.load_state_dict(sd,strict=True);critic.eval().requires_grad_(False)
        states.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           steps=state['completed_updates'], input_scale=float(critic.input_scale),
                           optimizer_settings={k:[{key:g.get(key) for key in ['lr','betas','eps','weight_decay']} for g in v['param_groups']]
                                               for k,v in state['optimizers'].items()}))
        for spans in span_sets:
            values=probe(critic,spans['fake'].to(device),spans['real'].to(device),spans['mask'].to(device))
            result=dict(critic_steps=state['completed_updates'],adapter_steps=spans['step'],**values)
            results.append(result);print(json.dumps(result),flush=True)
        del critic
    report=dict(states=states, results=results, zero_scale_exact_checks=len(cached)*len(span_sets),
                span_cache=str(spans_path.resolve()),span_cache_sha256=hashlib.sha256(spans_path.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                prompts_sha256=hashlib.sha256(prompts.read_bytes()).hexdigest(),
                scope='Read-only crossed critic/adapter probe on all four training prompt spans; no optimizer updates.',
                limitations=['Prefix-only bf16 forwards can differ numerically from training teacher-forced geometry.',
                             'Input gradient norms stop at critic inputs; they do not attribute full LoRA parameter gradients.',
                             'Post-state critic/adapter combinations are not a replay of the original pre-update losses.',
                             'Normalized features and doubled features are diagnostic counterfactuals, not trained fixes.',
                             'These snapshots cannot prove the initiating cause at updates 830–840.'])
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Saved {args.output}',flush=True)


if __name__=='__main__':
    main()
