"""Bound the full-flow graph by recomputing each complete transformer call.

The ordinary projection merger remains unchanged. Reentrant checkpointing runs
the first call without a graph, then rebuilds one call during backward. The
transformer's existing block checkpointing remains enabled during that rebuild.
An unused scalar input enables first-step factor gradients with frozen inputs.
Only ordinary backward(), as used by the declared CE trainer, is supported.
"""
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


class RecomputedFlowForward:
    def __init__(self, transformer, *, max_rss_gib=28, max_swap_gib=2):
        self.transformer = transformer
        self.original = transformer.forward
        self.attached = False
        self.max_rss_kib = max_rss_gib * 1024**2
        self.max_swap_kib = max_swap_gib * 1024**2
        self.metrics = dict(no_grad_calls=0, recomputed_calls=0,
                            peak_observed_rss_kib=0, peak_observed_swap_kib=0)

    def memory_check(self):
        fields = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmSwap:')):
                key, value, _ = line.split()
                fields[key[:-1]] = int(value)
        rss, swap = fields.get('VmRSS', 0), fields.get('VmSwap', 0)
        self.metrics['peak_observed_rss_kib'] = max(self.metrics['peak_observed_rss_kib'], rss)
        self.metrics['peak_observed_swap_kib'] = max(self.metrics['peak_observed_swap_kib'], swap)
        if rss > self.max_rss_kib or swap > self.max_swap_kib:
            raise MemoryError(f'Full-flow process memory budget: RSS={rss} KiB, swap={swap} KiB')

    def attach(self):
        if self.attached:
            raise RuntimeError('Flow recomputation already attached')

        def forward(hidden_states, timestep, encoder_hidden_states, return_dict=True):
            def call(latent, time, condition, anchor):
                self.memory_check()
                key = 'recomputed_calls' if torch.is_grad_enabled() else 'no_grad_calls'
                self.metrics[key] += 1
                return self.original(hidden_states=latent, timestep=time,
                                     encoder_hidden_states=condition, return_dict=False)[0]

            # Frozen first-step inputs still require a checkpoint autograd node.
            # This scalar does not alter the function or enter an optimizer.
            anchor = hidden_states.new_zeros((), requires_grad=True)
            if torch.is_grad_enabled():
                output = checkpoint(call, hidden_states, timestep, encoder_hidden_states,
                                    anchor, use_reentrant=True, preserve_rng_state=True)
            else:
                output = call(hidden_states, timestep, encoder_hidden_states, anchor)
            if not return_dict:
                return (output,)
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output)

        self.transformer.forward = forward
        self.attached = True
        return self

    def detach(self):
        if self.attached:
            self.transformer.forward = self.original
            self.attached = False

