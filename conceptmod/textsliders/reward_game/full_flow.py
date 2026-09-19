"""Unselected complete within-chunk flow derivative, with fixed chunk overlap.

This records all thirty flow steps. It still excludes autoregressive code and
conditioning changes and detaches reference overlap between audio chunks.
"""
from .acoustic_tail import TailCapture,replay_latents
from .core import IntegrityError


SCOPE='All thirty ordinary flow steps within each chunk; conditioning and reference overlap between chunks remain fixed'


class FullFlowCapture(TailCapture):
    def __init__(self,pipe):super().__init__(pipe,steps=30,retained=30)

    def result(self):
        result=super().result()
        result.update(version=2,gradient_scope=SCOPE,full_generation_gradient=False)
        validate(result)
        return result


def validate(capture):
    if capture.get('version')!=2 or capture.get('steps')!=30 or capture.get('retained_steps')!=30:
        raise IntegrityError('Complete within-chunk flow requires an explicitly recorded thirty-step capture')
    if capture.get('gradient_scope')!=SCOPE or capture.get('full_generation_gradient') is not False:
        raise IntegrityError('Complete within-chunk capture must declare fixed conditioning and overlap')
    if not capture.get('chunks'):raise IntegrityError('Missing captured chunks')
    for chunk in capture['chunks']:
        if chunk['calls']!=60 or set(chunk['steps'])!=set(range(30)):
            raise IntegrityError('Incomplete within-chunk flow capture')
    return capture


def replay(transformer,capture,**kwargs):
    validate(capture)
    return replay_latents(transformer,capture,**kwargs)
