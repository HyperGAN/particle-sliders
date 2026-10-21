"""Shared spatial field critic and scoped initialization; no campaign orchestration."""
import torch
from torch import nn


def seed_initialization_rng(seed, cuda_devices):
    """Seed only generators covered by the caller's active ``fork_rng`` scope."""
    torch.default_generator.manual_seed(seed)
    for index in cuda_devices:
        device = torch.device('cuda', index)
        state = torch.Generator(device=device).manual_seed(seed).get_state()
        torch.cuda.set_rng_state(state, device)


class FieldCritic(nn.Module):
    """Small native-field critic with context/time/sign FiLM and raw scalar logits."""
    def __init__(self, channels, condition_width, width=32):
        super().__init__()
        self.field = nn.Sequential(nn.Conv2d(channels, width, 1), nn.SiLU(),
                                   nn.Conv2d(width, width, 3, padding=1))
        self.condition = nn.Linear(condition_width, 2 * width)
        self.score = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, response, condition):
        if response.ndim == 5 and response.shape[2] == 1:
            response = response.squeeze(2)
        elif response.ndim != 4:
            raise ValueError('field critic requires 4D images or 5D fields with singleton temporal axis')
        field = self.field(response)
        gain, bias = self.condition(condition).chunk(2, dim=1)
        field = torch.nn.functional.silu(field * (1 + gain[:, :, None, None]) + bias[:, :, None, None])
        return self.score(field.mean((2, 3))).squeeze(-1)

