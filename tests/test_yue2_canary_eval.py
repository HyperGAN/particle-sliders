"""The optional canary is unscored; hidden diagnostics do not claim toy gates."""
from contextlib import contextmanager
from types import SimpleNamespace
import torch
import pytest
from scripts.evaluate_yue2_arm_b import hidden_diagnostics,takes,page
from conceptmod.textsliders.yue2_backend import YuE2Backend


def test_heldout_geometry_keeps_zero_exact_and_minus_unscored(monkeypatch,tmp_path):
    class Slider:
        scale=0.
        @contextmanager
        def scaled(self,s):
            before=self.scale;self.scale=s
            try:yield
            finally:self.scale=before
    network=Slider()
    monkeypatch.setattr(YuE2Backend,'prefix',lambda self,style,lyrics: [style])
    def hidden(self,ids):
        delta=torch.tensor([2.,0.]) if ids==['positive'] else network.scale*torch.tensor([1.,1.])
        return (torch.tensor([3.,4.])+delta)[None,None,:]
    monkeypatch.setattr(YuE2Backend,'hidden',hidden)
    rows=[dict(neutral='neutral',positive='positive',lyrics='[verse]\nWe carry the canvas')]
    result=hidden_diagnostics(SimpleNamespace(),None,network,rows,True)
    assert result['kind']=='unscored_hidden_geometry'
    zero,half,full,minus=result['records']
    assert zero['zero_exact'] and zero['neutral_delta_norm']==0.
    assert half['caption_projection']==.25
    assert full['caption_projection']==full['orthogonal_ratio']==.5
    assert full['target_relative_error']==pytest.approx(2**-.5)
    assert minus['canary'] and minus['caption_projection']==-.5
    assert not any('hit' in row for row in result['records'])
    assert len(takes())==4 and len(takes(True))==5
    page(tmp_path,rows,[1709],'unipolar_gan',True)
    assert 'untrained, unscored canary' in (tmp_path/'index.html').read_text()
