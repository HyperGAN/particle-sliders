"""Per-model residual hooks; no pipeline classes or process globals are patched."""
from __future__ import annotations

from contextlib import AbstractContextManager
import math
import torch


class ResidualCapture(AbstractContextManager):
    """Use absolute cache positions, never sequence length alone, for the mask.

    Generation discovers the prompt boundary on the first uncached forward.
    Teacher forcing supplies it explicitly. All feedback inputs for both CFG
    rows are saved: a pooled activation alone is deliberately insufficient.
    Zero steering installs no intervention and does no vector arithmetic/RNG.
    """
    def __init__(self, lm, *, layers=(11, 23), direction=None, layer=None,
                 coefficient=0., residual_norm=1., prompt_length=None, capture=True):
        self.lm, self.layers = lm, tuple(layers)
        self.direction, self.layer = direction, layer
        self.coefficient, self.residual_norm = float(coefficient), float(residual_norm)
        self.prompt_length, self.capture = prompt_length, capture
        if not math.isfinite(self.coefficient) or not math.isfinite(self.residual_norm) or self.residual_norm <= 0:
            raise ValueError('Invalid residual intervention')
        if self.coefficient and (direction is None or layer is None or layer >= len(lm.model.layers)-1):
            raise ValueError('Nonzero steering needs a direction at an intermediate layer')
        if direction is not None and (direction.ndim != 1 or not torch.isfinite(direction).all()
                                      or abs(float(direction.float().norm())-1.) > 1e-4):
            raise ValueError('Direction must be a finite unit vector')
        self.handles, self.frames, self.prompt = [], [], None
        self.activations = {str(l): [] for l in self.layers}
        self.positions = []
        self.start, self.stop = 0, 0
        self.shift = None

    def _before(self, module, args, kwargs):
        embeds = kwargs.get('inputs_embeds')
        if embeds is None:
            ids = kwargs.get('input_ids', args[0] if args else None)
            embeds = module.embed_tokens(ids)
        cache = kwargs.get('past_key_values')
        offset = int(cache.get_seq_length()) if cache is not None else 0
        positions = kwargs.get('cache_position')
        if positions is not None:
            offset = int(positions[0])
        if self.prompt_length is None:
            if offset != 0:
                raise ValueError('Capture must start at an uncached prompt')
            self.prompt_length = embeds.shape[1]
        self.start = max(0, self.prompt_length-offset)
        self.stop = embeds.shape[1]
        if self.capture:
            if offset == 0:
                self.prompt = embeds[:, :self.prompt_length].detach().cpu().clone()
            if self.start < self.stop:
                self.frames.append(embeds[:, self.start:].detach().cpu().clone())
                self.positions.extend(range(offset+self.start, offset+self.stop))

    def _after_layer(self, layer):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.start >= self.stop:
                return output
            if self.coefficient != 0. and layer == self.layer:
                if self.shift is None:
                    self.shift = (self.direction.to(device=hidden.device, dtype=torch.float32)
                                  * (self.coefficient*self.residual_norm)).to(hidden.dtype)
                hidden = hidden.clone()
                hidden[:, self.start:self.stop] += self.shift
            if self.capture and str(layer) in self.activations:
                self.activations[str(layer)].append(hidden[:, self.start:self.stop].detach().cpu().clone())
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook

    def __enter__(self):
        if not self.capture and self.coefficient == 0.:
            return self
        if getattr(self.lm, '_reward_capture_owner', None) is not None:
            raise RuntimeError('A reward job already owns this model')
        self.lm._reward_capture_owner = self
        try:
            self.handles.append(self.lm.model.register_forward_pre_hook(self._before, with_kwargs=True))
            wanted = set(self.layers) if self.capture else set()
            if self.coefficient:
                wanted.add(self.layer)
            for layer in sorted(wanted):
                self.handles.append(self.lm.model.layers[layer].register_forward_hook(self._after_layer(layer)))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if getattr(self.lm, '_reward_capture_owner', None) is self:
            del self.lm._reward_capture_owner
        self.shift = None
        return False

    def trajectory(self):
        if self.prompt is None or not self.frames:
            raise ValueError('No generated feedback trajectory')
        return dict(prompt_embeds=self.prompt, frame_embeds=torch.cat(self.frames, 1),
                    positions=torch.tensor(self.positions), prompt_length=self.prompt_length,
                    activations={key: torch.cat(parts, 1) for key, parts in self.activations.items() if parts})


def pooled(trajectory, frames=500):
    """Equal weighting of two nominal audio windows and both recorded branches."""
    result = {}
    for layer, hidden in trajectory['activations'].items():
        h = hidden[:, :frames].float()
        if h.shape[1] < frames:
            raise ValueError('Incomplete trajectory cannot fit a short-screen direction')
        result[layer] = dict(windows=torch.stack([h[:, :250].mean(1), h[:, 250:500].mean(1)], 1),
                             branch_mean=h.mean(1), mean=h.mean((0, 1)),
                             residual_norms=h.norm(dim=-1))
    return result
