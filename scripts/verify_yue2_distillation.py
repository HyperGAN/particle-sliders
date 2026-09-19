#!/usr/bin/env python3
"""Validate every final LoRA and audio comparison, then create download bundles."""
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/'models/yue2-lora-distill-20260917'
PAGE = ROOT/'eval/listen/yue2-lora-distill-20260917'
ANALYSIS = ROOT/'analysis/yue2_lora_distill_20260917'
COMFY = ROOT.parent/'.cache/comfyui-yue2-particle'
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import comfy.lora
from comfy.text_encoders.yue2 import YuE2TEModel


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    catalog = json.loads((ROOT.parent/'releases/yue2-concept-sliders/model/catalog.json').read_text())
    te = YuE2TEModel(device='meta', dtype=torch.bfloat16)
    mapping = comfy.lora.model_lora_keys_clip(te, {})
    expected = {f'model.layers.{i}.self_attn.{p}.weight'
                for i in range(28) for p in ('qkv_proj','o_proj')}
    reports = []
    native_files, comfy_files = [], []
    for spec in catalog['sliders']:
        concept = spec['id']
        directory = EXPERIMENT/concept
        metrics = json.loads((directory/'refined-evaluation.json').read_text())
        audio = json.loads((directory/'audio/summary.json').read_text())
        assert metrics['zero_exact'] and metrics['ordinary_lora_only']
        assert len(audio['recordings']) == 8
        checkpoint = directory/metrics['checkpoint']
        assert digest(checkpoint) == metrics['checkpoint_sha256'] == audio['checkpoint_sha256']
        converted = checkpoint.with_name(checkpoint.stem+'_comfyui.safetensors')
        native = load_file(checkpoint)
        with safe_open(checkpoint, framework='pt') as f:
            record = json.loads(f.metadata()['conceptmod'])
        assert record['format'] == 'conceptmod-yue2-ar-v1'
        assert record['teacher_sha256'] == spec['sha256']
        assert len(native) == 336
        assert all(k.endswith(('.lora_down.weight','.lora_up.weight','.alpha')) for k in native)
        assert all(torch.isfinite(t).all() for t in native.values())
        sd = load_file(converted)
        assert len(sd) == 168 and all(torch.isfinite(t).all() for t in sd.values())
        patches = comfy.lora.load_lora(sd, mapping)
        assert set(patches) == expected
        worst = 0.
        generator = torch.Generator().manual_seed(183)
        for layer in range(28):
            probe = torch.randn(8, 2048, generator=generator)
            for group in ('qkv_proj','o_proj'):
                members = ('q_proj','k_proj','v_proj') if group=='qkv_proj' else ('o_proj',)
                wanted = []
                for member in members:
                    name=f'adapters.model-layers-{layer}-self_attn-{member}'
                    wanted.append((probe @ native[name+'.lora_down.weight'].T) @ native[name+'.lora_up.weight'].T)
                wanted = torch.cat(wanted, dim=-1)
                stem=f'text_encoders.model.layers.{layer}.self_attn.{group}'
                a,b=sd[stem+'.lora_A.weight'].float(),sd[stem+'.lora_B.weight'].float()
                assert a.shape[0] == (24 if group=='qkv_proj' else 8)
                got=(probe @ a.T) @ b.T * sd[stem+'.alpha']/a.shape[0]
                relative=float((got-wanted).norm()/wanted.norm().clamp_min(1e-12))
                worst=max(worst,relative)
                assert relative < .02, (concept,layer,group,relative)
        scope=[]
        for entry in audio['recordings']:
            signal,rate=sf.read(directory/'audio'/entry['file'])
            assert rate == 48000 and signal.ndim==2 and signal.shape[1]==2
            assert np.isfinite(signal).all() and np.sqrt(np.mean(signal**2)) > 1e-6
            assert abs(len(signal)/rate-entry['seconds']) < 1e-6
        for row in range(2):
            left,_=sf.read(directory/'audio'/f'row{row}-plain-ar.flac')
            right,_=sf.read(directory/'audio'/f'row{row}-plain-all.flac')
            assert left.shape == right.shape
            assert np.array_equal(np.load(directory/'audio'/f'row{row}-plain-ar-semantic.npy'),
                                  np.load(directory/'audio'/f'row{row}-plain-all-semantic.npy'))
            scope.append(float(np.sum((left-right)**2)/np.sum(left**2)))
        reports.append(dict(concept=concept, native_sha256=digest(checkpoint),
            comfy_sha256=digest(converted), loaded_patches=len(patches),
            maximum_export_projection_relative_rmse=worst, audio_clips=8,
            acoustic_scope_relative_audio_mse=scope))
        native_files.extend([checkpoint, checkpoint.with_suffix('.json')])
        comfy_files.extend([converted, converted.with_name(converted.name+'.json')])
        old_error=directory/'error.json'
        if old_error.exists():
            old_error.rename(directory/'preflight-error-resolved.json')
        print(concept, '56 ComfyUI patches; 8 valid clips; export error',round(worst,6),flush=True)
    result=dict(complete=True,sliders=16,audio_clips=128,comfy_patches=896,
        converter_commit='a5c3dd8a9cdc34a86d633b9e1aed5e7efe4bc489',
        comfy_commit='387f98aa2822f684b8597959a52a467d88cc4806',
        torch_version=torch.__version__,reports=reports)
    for destination in (ANALYSIS/'verification.json',PAGE/'verification.json'):
        destination.write_text(json.dumps(result,indent=2)+'\n')
    license_text='''YuE2 distilled concept-slider weights\n\nDerived from ntc-ai/yue2-concept-sliders and m-a-p/YuE2-3B.\nWeights retain Creative Commons Attribution-NonCommercial 4.0 International terms:\nhttps://creativecommons.org/licenses/by-nc/4.0/\n\nExperimental approximations of the original particle sliders.\nSee README.md and verification.json for method, attribution and limits.\n'''
    (PAGE/'WEIGHTS_LICENSE.txt').write_text(license_text)
    for name,files in [('yue2-ordinary-loras-comfyui.zip',comfy_files),('yue2-ordinary-loras-native.zip',native_files)]:
        with zipfile.ZipFile(PAGE/name,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for file in files:
                z.write(file,file.name)
            for common in ['README.md','workflow.json','WORKFLOW_LICENSE','WEIGHTS_LICENSE.txt','verification.json','summary.json']:
                z.write(PAGE/common,common)
            if 'native' in name:
                for common in ['slider_runtime.py','USAGE.md','LICENSE']:
                    z.write(PAGE/common,common)
        print(name,digest(PAGE/name),flush=True)


if __name__=='__main__':
    main()
