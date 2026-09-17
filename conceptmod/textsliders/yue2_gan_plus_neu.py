"""Opt-in YuE2 adapter for the CPU-verified +/0 conditional RpGAN game."""
import torch

from conceptmod.textsliders import unipolar_gan as shared
from conceptmod.textsliders.yue2_arm_b import load_prompts, prepare, RECIPE as OLD_RECIPE

# LoRA parameters and the toy's direct residual have different units. Keep
# the native LR explicit; the shared equations/update order are identical.
RECIPE = dict(OLD_RECIPE, name='rpgan-bcap-plus-neu-yue2-v1',
    generator_objective='positive_and_neutral_rpgan_only', trained_scales=[0., 1.],
    critic_condition='slider_scale', g_lr=.0005, d_lr=.0005,
    g_optimizer='Adam', betas=[0., .99], g_weight_decay=0.,
    grad_clip_value=None, schedule='delayed_cosine', schedule_delay=80,
    min_lr_ratio=.05, toy_parameter_lr=.005,
    zero_training='exact_constant_rpgan_term; cached_base_delta',
    cap_condition_gradient=False)


def build_game(backend, network, fixed):
    device = next(backend.model.parameters()).device
    real = torch.cat([r['targets'] - r['neutral'] for r in fixed]).to(device)
    return shared.build_game(network, real, lr=RECIPE['g_lr'])


def update(backend, network, critic, g, d, rows, *, step, total_steps, checkpointing=True, grad_arm='b_cap'):
    device = next(critic.parameters()).device
    real = torch.cat([r['targets'] - r['neutral'] for r in rows]).to(device)
    def predict(i, scale, checkpointing):
        if scale == 0.:
            # Adapter multiplier zero is the frozen base exactly. Its GAN
            # term is log(2), with zero parameter gradient; no unused forward.
            return torch.zeros_like(real[i:i+1])
        row = rows[i]
        pred = backend.hidden(row['ids'], checkpointing=checkpointing)[:, row['prefix_len']-1].float()
        return pred - row['neutral'].to(device)
    return shared.update(network, critic, g, d, real, predict,
        step=step, total_steps=total_steps, checkpointing=checkpointing, grad_arm=grad_arm)
