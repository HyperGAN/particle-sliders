"""Separate Sana GAN cohort. Importing never loads a model or attaches hooks."""
from .backend import configuration, staged_model

__all__ = ['configuration', 'staged_model']
