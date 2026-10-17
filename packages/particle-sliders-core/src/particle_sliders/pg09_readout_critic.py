"""Small raw-logit critic for bounded readout qualification."""
import torch
from torch import nn


class ConditionalReadoutCritic(nn.Module):
    def __init__(self, response_width, condition_width, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(response_width + condition_width, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, response, condition):
        return self.net(torch.cat((response.flatten(1), condition), dim=1)).squeeze(-1)
