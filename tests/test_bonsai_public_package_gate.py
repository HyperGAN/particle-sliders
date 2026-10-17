"""Public archived Bonsai reporting cannot restart its local training game."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from conceptmod.textsliders import train_lora_bonsai_gguf as cli


def forbidden(*args, **kwargs):
    raise AssertionError('inference path reached training/model preparation')


def test_positive_legacy_steps_reject_before_data_model_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'build_backend', forbidden)
    monkeypatch.setattr(cli, 'load_slider_rows', forbidden)
    monkeypatch.setattr(cli, 'download_weights', forbidden)
    args=cli.parse_args(['--steps','1','--allow_download','--prompts_file',str(tmp_path/'missing'),
                         '--save_dir',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='legacy Bonsai.*retired.*ConditionalPG09'):
        cli.train(args)
    assert not (tmp_path/'out').exists()


def test_zero_steps_requires_explicit_loaded_head_before_model_access(monkeypatch):
    monkeypatch.setattr(cli, 'build_backend', forbidden)
    with pytest.raises(ValueError,match='requires --load_head'):
        cli.train(cli.parse_args(['--steps','0']))


def test_report_only_preserves_failed_head_provenance_and_active_weights(tmp_path, monkeypatch):
    from particle_sliders.bonsai_native import BonsaiParticleHead
    from particle_sliders.reference import RoutedMLP
    head=BonsaiParticleHead()
    with torch.no_grad():
        head.up.weight.fill_(.01)
    source=tmp_path/'saved.safetensors'
    head.save(source,dict(weights_kind='ema',step=1200,quality_qualified=False,
                         heldout_transfer_passed=False,recipe='archival-failed-fixture'))
    contents=source.read_bytes(),source.with_suffix('.json').read_bytes()
    before=deepcopy(head.state_dict())
    monkeypatch.setattr(torch.optim.Adam,'__init__',forbidden)
    loaded=[]
    original=BonsaiParticleHead.load
    def load(path):
        value,metadata=original(path)
        loaded.append(value)
        return value,metadata
    monkeypatch.setattr(BonsaiParticleHead,'load',load)
    out=tmp_path/'out'
    result=cli.train(cli.parse_args(['--dummy','--steps','0','--load_head',str(source),
        '--name','review','--save_dir',str(out),'--report_scales=-1,0,0.5,1']))
    assert result['training_performed'] is False and result['optimizer_constructed'] is False
    assert result['head_class']=='particle_sliders.bonsai_native.BonsaiParticleHead'
    assert result['source_weights_kind']=='ema' and result['source_completed_steps']==1200
    assert result['source_metadata']['heldout_transfer_passed'] is False
    assert result['source_metadata']['quality_qualified'] is False
    assert result['quality_qualified'] is False
    assert type(loaded[0].bridge) is RoutedMLP
    assert all(torch.equal(value,loaded[0].state_dict()[key]) for key,value in before.items())
    assert contents==(source.read_bytes(),source.with_suffix('.json').read_bytes())
    assert not list(out.glob('*.safetensors'))
    receipt=json.loads((out/'review_inference.json').read_text())
    assert receipt==result
    records=json.loads((out/'report/readout_delta.json').read_text())
    assert set(r['scale'] for r in records)=={-1.,0.,.5,1.}
    assert all(r['delta_l2']<1e-5 for r in records if r['scale']==0)
    assert not torch.cuda.is_initialized()


def test_no_report_loads_shared_head_without_backend_or_prompts(tmp_path,monkeypatch):
    from particle_sliders.bonsai_native import BonsaiParticleHead
    source=tmp_path/'saved.safetensors'
    BonsaiParticleHead().save(source,dict(weights_kind='live',completed_steps=4))
    monkeypatch.setattr(cli,'build_backend',forbidden)
    monkeypatch.setattr(cli,'load_slider_rows',forbidden)
    result=cli.train(cli.parse_args(['--steps','0','--no_report','--load_head',str(source),
                                    '--save_dir',str(tmp_path/'out')]))
    assert result['source_weights_kind']=='live'
    assert result['source_completed_steps']==4
    assert result['report'] is None


def test_ambiguous_directory_requires_active_weight_file(tmp_path,monkeypatch):
    source=tmp_path/'source'
    source.mkdir()
    (source/'alpha_live_last.safetensors').write_bytes(b'live')
    (source/'zeta_last.safetensors').write_bytes(b'ema')
    monkeypatch.setattr(cli,'build_backend',forbidden)
    with pytest.raises(ValueError,match='unambiguous export'):
        cli.train(cli.parse_args(['--steps','0','--load_head',str(source),
                                 '--save_dir',str(tmp_path/'out')]))
    assert not (tmp_path/'out').exists()


def test_inference_does_not_replace_existing_receipt_or_report(tmp_path,monkeypatch):
    from particle_sliders.bonsai_native import BonsaiParticleHead
    source=tmp_path/'saved.safetensors'
    BonsaiParticleHead().save(source,{})
    args=cli.parse_args(['--steps','0','--no_report','--load_head',str(source),
                         '--save_dir',str(tmp_path),'--name','saved'])
    cli.train(args)
    prior=(tmp_path/'saved_inference.json').read_bytes()
    with pytest.raises(FileExistsError,match='destination already exists'):
        cli.train(args)
    assert prior==(tmp_path/'saved_inference.json').read_bytes()


def test_finite_head_overflow_fails_without_publishing_nonfinite_report(tmp_path):
    from particle_sliders.bonsai_native import BonsaiParticleHead
    source=tmp_path/'overflow.safetensors'
    head=BonsaiParticleHead()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.fill_(torch.finfo(torch.float32).max)
    head.save(source,{})
    out=tmp_path/'out'
    with pytest.raises((FloatingPointError,ValueError),match='nonfinite|JSON'):
        cli.train(cli.parse_args(['--dummy','--steps','0','--load_head',str(source),
            '--name','overflow','--save_dir',str(out)]))
    assert not (out/'report/readout_delta.json').exists()
    receipt=json.loads((out/'overflow_inference.json').read_text())
    assert receipt['status']=='failed'
    assert receipt['quality_qualified'] is False


def test_receipt_reservation_prevents_checked_then_concurrent_report_race(tmp_path):
    from conceptmod.textsliders.legacy_inference import report_destination, reporting_operation, source_digest
    source=tmp_path/'saved.safetensors'
    source.write_bytes(b'fixture')
    directory,first=report_destination(tmp_path/'out','same',source)
    _,second=report_destination(tmp_path/'out','same',source)
    receipt=dict(source_export=str(source),source_sha256=source_digest(source))
    with reporting_operation(first,dict(receipt)):
        with pytest.raises(FileExistsError):
            with reporting_operation(second,dict(receipt)):
                (directory/'report').mkdir()
    assert not (directory/'report').exists()
    assert json.loads(first.read_text())['status']=='completed'


def test_source_replacement_is_detected_and_never_stamped_completed(tmp_path):
    from conceptmod.textsliders.legacy_inference import reporting_operation, source_digest
    source=tmp_path/'saved.safetensors'
    source.write_bytes(b'before')
    path=tmp_path/'receipt.json'
    receipt=dict(source_export=str(source),source_sha256=source_digest(source))
    with pytest.raises(ValueError,match='source export changed'):
        with reporting_operation(path,receipt):
            source.write_bytes(b'after')
    observed=json.loads(path.read_text())
    assert observed['status']=='failed'
    assert observed['source_sha256']!=source_digest(source)


def test_invalid_receipt_is_rejected_before_output_creation(tmp_path):
    from conceptmod.textsliders.legacy_inference import save_receipt
    out=tmp_path/'new/receipt.json'
    with pytest.raises(ValueError):
        save_receipt(out,{'first':'valid','bad':float('nan')})
    assert not out.parent.exists()


@pytest.mark.parametrize('script', ['qualify_captions.py', 'queue_caption_qualification.py'])
def test_retired_campaign_cli_blocks_before_old_pinned_run_access(tmp_path,script):
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,str(root/'scripts'/script),'--model','bonsai']
    if script=='qualify_captions.py':
        command+=['--until','100']
    result=subprocess.run(command,cwd=tmp_path,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==2
    assert 'legacy native caption campaigns are retired' in result.stderr
    assert not list(tmp_path.iterdir())
