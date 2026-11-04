"""Exercise actual existing lifecycle, including exact split-resume state."""
import json
from pathlib import Path

import torch

from particle_sliders import lifecycle
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.execution.session_worker import main
from test_compact_inference import plugin


def test_periodic_session_and_exact_resume(tmp_path, plugin, monkeypatch):
    monkeypatch.setattr(lifecycle,'backend',lambda name:plugin.TRAINING)
    config=SessionConfig('compact-fixture','periodic-new-source-cohort',
        model_file=str(tmp_path/'model.pt'), training_file=str(tmp_path/'training.pt'),steps=4,ema_decay=.9)
    configfile=tmp_path/'config.json';configfile.write_text(json.dumps(config.as_dict()))
    first=tmp_path/'first'
    main(['--config',str(configfile),'--output',str(first),'--checkpoint-every','1','--stop-after','2'])
    assert len(list((first/'checkpoints').iterdir()))==2
    resumed=tmp_path/'resumed'
    main(['--config',str(configfile),'--output',str(resumed),'--checkpoint-every','1',
          '--resume',str(first/'checkpoints/step-000002')])
    uninterrupted=tmp_path/'full'
    main(['--config',str(configfile),'--output',str(uninterrupted),'--checkpoint-every','1'])
    a=torch.load(resumed/'checkpoints/step-000004/training.pt',weights_only=True)
    b=torch.load(uninterrupted/'checkpoints/step-000004/training.pt',weights_only=True)
    assert a['sha256']==b['sha256']
    result=json.loads((resumed/'result.json').read_text())
    assert result['completed_steps']==4 and result['rendered_images']==0 and result['lr_search'] is False
