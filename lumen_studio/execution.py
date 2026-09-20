"""Reproducible GPU execution; the ParticleGAN formulation is unchanged."""
import os

DETERMINISM = dict(algorithms=True, cublas_workspace=":4096:8",
                   cudnn_benchmark=False, cudnn_deterministic=True)

# This module is imported by the trainer before its runtime is allocated.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISM["cublas_workspace"]


def configure():
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


configure()
