"""CPU audit of the source recipe before any YuE2 Arm B training."""
from pathlib import Path
import ast
import hashlib
import inspect
import json
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.slider2d.adv import rp_d_loss, rp_g_loss, make_grad_regularizer
from conceptmod.textsliders import train_lm_slider_music3 as music
from conceptmod.textsliders.lm_adv import LMDiscriminator, rp_g_loss as legacy_g_loss
from conceptmod.textsliders.slider_targets import lm_slider_loss, lm_faithful_guard_e
from conceptmod.textsliders.yue2_uni import RECIPE as current_yue2

def main():
    torch.set_num_threads(1)
    torch.manual_seed(91)
    checks = {}
    real = torch.tensor([1.2, .4], dtype=torch.float64)
    fake = torch.tensor([-.3, .1], dtype=torch.float64, requires_grad=True)
    assert torch.equal(rp_d_loss(real, fake), F.softplus(fake-real).mean())
    assert torch.equal(rp_g_loss(real, fake), F.softplus(real-fake).mean())
    assert torch.equal(rp_g_loss(real, fake), legacy_g_loss(fake, real))
    assert (torch.autograd.grad(rp_g_loss(real, fake), fake)[0] < 0).all()
    checks['rpgan_loss_and_generator_gradient_sign'] = True
    checks['generator_helper_argument_order_differs'] = True

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([3.,4.],dtype=torch.float64))
        def forward(self, x):
            return x @ self.w
    d = Linear()
    xr, xf = torch.randn(4,2,dtype=torch.float64), torch.zeros(4,2,dtype=torch.float64)
    reg = make_grad_regularizer(coeff=1.,kappa=1.)
    penalty = reg(d,xr,xf)
    analytic = ((d.w.square().sum()+1e-12).sqrt()-1).relu().square()
    assert torch.allclose(penalty,analytic,rtol=1e-12,atol=1e-12)
    assert torch.allclose(torch.autograd.grad(penalty,d.w)[0],torch.autograd.grad(analytic,d.w)[0],rtol=1e-12,atol=1e-12)
    checks['vendored_penalty_value_and_parameter_gradient_including_zero_fakes'] = True
    assert reg.method == 'autograd' and reg.lazy_k == 1 and reg.target_anneal == 'none'
    checks['exact_autograd_every_update_no_anneal'] = True

    # Applying the same penalty to raw vs calibrated inputs changes units.
    # A shared code path is necessary; merely swapping the helper is not parity.
    scaled = LMDiscriminator(2,hidden_dim=8,in_mode='scaled',input_scale=3.).double()
    with torch.no_grad():
        for p in scaled.parameters(): p.fill_(.5)
    raw = torch.ones(4,2,dtype=torch.float64)
    expected = reg(scaled.net, raw/scaled.input_scale, raw/scaled.input_scale)
    naive = reg(scaled, raw, raw)
    assert not torch.allclose(expected,naive)
    checks['naive_regularizer_swap_changes_calibrated_cap_units'] = True

    p = torch.randn(2,5,requires_grad=True); n = torch.randn(2,5,requires_grad=True)
    tp = torch.randn(2,5); tn = torch.randn(2,5)
    pole = lm_slider_loss(p,n,tp,tn)
    assert torch.equal(pole,F.mse_loss(p,tp)+F.mse_loss(n,tn))
    assert torch.allclose(pole,2*F.mse_loss(torch.cat([p,n]),torch.cat([tp,tn])),rtol=1e-6,atol=1e-7)
    checks['pole_loss_is_sum_of_polarities_not_mean_or_two_cover_terms'] = True

    src=inspect.getsource(music.train)
    checks['music_train_calls_new_factory'] = 'make_music_grad_regularizer(' in src
    checks['music_train_calls_legacy_penalty'] = '_lm_adv.cap_penalty(' in src
    assert not checks['music_train_calls_new_factory']
    assert checks['music_train_calls_legacy_penalty']
    args=music.parse_args(['--prompts_file','x.yaml','--adv_b_cap','.5'])
    assert args.adv_b_cap == .5 and args.adv_reg_coeff == 1.
    checks['upstream_and_local_coefficient_flags_are_not_linked'] = True

    # The guard must preserve the caption midpoint and decline a subtraction
    # that would turn the pole into a blend; use the shared target function.
    neu=torch.zeros(1,3); pos=torch.tensor([[2.,.2,1.]]); neg=torch.tensor([[-2.,-.2,1.]])
    leak=torch.tensor([[0.,1.,0.]]); axis=torch.tensor([[1.,0.,0.]])
    plus,minus=lm_faithful_guard_e(pos,neg,neu,leak,slider_dir=axis)
    assert torch.allclose((plus+minus)/2,(pos+neg)/2)
    assert torch.allclose(plus,torch.tensor([[2.,0.,1.]]))
    bad_pos=torch.tensor([[.1,2.,1.]]); bad_neg=torch.tensor([[-.1,-2.,1.]])
    a,b=lm_faithful_guard_e(bad_pos,bad_neg,neu,leak,slider_dir=axis)
    assert torch.equal(a,bad_pos) and torch.equal(b,bad_neg)
    checks['guard_preserves_midpoint_and_refuses_destructive_subtraction'] = True

    paths=['analysis/slider2d/adv.py','analysis/slider2d/gan.py',
        'analysis/slider2d/grad_regularizers.py','conceptmod/textsliders/slider_targets.py',
        'conceptmod/textsliders/lm_adv.py','conceptmod/textsliders/train_lm_slider_music3.py',
        'conceptmod/textsliders/yue2_uni.py','conceptmod/textsliders/train_lora_yue2_fresh.py',
        'analysis/slider2d/notes/MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md',
        'analysis/slider2d/notes/music_arm_b_locked_smoke_20260909.sh','docs/music-arm-b-gates.md']
    report={'status':'source_recipe_audited; YuE2 Arm B not implemented or certified',
        'gpu_training_started':False,'checks':checks,'arm_b_shape':music.ARM_B,
        'existing_yue2_recipe':current_yue2,
        'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    destination=Path(__file__).with_name('verification.json')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':checks,'report':str(destination)},indent=2))

if __name__ == '__main__': main()
