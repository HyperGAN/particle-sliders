"""One frozen Turbo runtime for target preparation, probes and Studio rendering."""
from __future__ import annotations

from collections import OrderedDict
import gc
import json
from pathlib import Path

import numpy as np
import torch

from ..contracts import digest, file_hash
from ..particles import Mixer
from ..provenance import model_identity, verify_environment


from ..runtime import Cancelled


class TurboRuntime:
    def __init__(self, model_dir, device="cuda:1", *, verify=True, checkpointing=False):
        from diffusers import ModularPipeline
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.lock = json.loads((self.model_dir / "anima-lock.json").read_text())
        if self.lock["model"] != "circlestone-labs/Anima:turbo-v1.1":
            raise ValueError("Expected the pinned official Turbo v1.1 conversion")
        if verify:
            verify_environment()
            for name, expected in self.lock["files"].items():
                if file_hash(self.model_dir / name) != expected:
                    raise ValueError(f"Model file hash mismatch: {name}")
        self.identity = model_identity(self.model_dir)
        self.pipe = ModularPipeline.from_pretrained(str(self.model_dir), local_files_only=True)
        self.pipe.load_components(dtype=self.dtype)
        for name in ("transformer", "text_encoder", "text_conditioner", "vae"):
            module = getattr(self.pipe, name)
            module.requires_grad_(False).eval().to(device=self.device)
        self.transformer = self.pipe.transformer
        if checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.mixer = Mixer(self.transformer)
        self.embeddings = OrderedDict()
        self.embedding_limit = 256
        self.scale = self.pipe.vae_scale_factor
        self.channels = self.transformer.config.in_channels

    @torch.no_grad()
    def encode(self, prompt):
        if prompt in self.embeddings:
            self.embeddings.move_to_end(prompt)
            return self.embeddings[prompt].to(self.device)
        tok = self.pipe.tokenizer([prompt], padding="longest", max_length=512,
                                   truncation=True, return_tensors="pt")
        ids, mask = tok.input_ids.to(self.device), tok.attention_mask.to(self.device)
        if ids.shape[-1] == 0:
            ids, mask = ids.new_zeros((1, 1)), mask.new_zeros((1, 1))
        qwen = self.pipe.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        qwen = qwen * mask.to(qwen.dtype).unsqueeze(-1)
        t5 = self.pipe.t5_tokenizer([prompt], padding="longest", max_length=512,
                                    truncation=True, return_tensors="pt")
        embed = self.pipe.text_conditioner(source_hidden_states=qwen,
            target_input_ids=t5.input_ids.to(self.device), target_attention_mask=t5.attention_mask.to(self.device),
            source_attention_mask=mask)
        self.embeddings[prompt] = embed.detach().cpu()
        if len(self.embeddings) > self.embedding_limit:
            self.embeddings.popitem(last=False)
        return embed

    def scheduler(self, steps):
        if steps not in (8, 10, 12):
            raise ValueError("Turbo uses 8, 10, or 12 steps")
        sched = self.pipe.scheduler.from_config(self.pipe.scheduler.config)
        sched.set_timesteps(sigmas=np.linspace(1., 1 / steps, steps), device=self.device)
        sched.set_begin_index(0)
        return sched

    def noise(self, seed, width, height):
        if width % 16 or height % 16 or min(width, height) < 512 or max(width, height) > 1536:
            raise ValueError("Image dimensions must be 512–1536 and divisible by 16")
        # CPU noise makes seeds stable across GPU handoffs and worker restarts.
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(1, self.channels, 1, height // self.scale, width // self.scale,
                           generator=g, dtype=torch.float32).to(self.device)

    def predict(self, latent, timestep, embedding):
        if latent.ndim != 5 or latent.shape[2] != 1:
            raise ValueError("Anima image latents must retain the singleton time axis")
        t = torch.as_tensor(timestep, device=self.device).reshape(-1).expand(latent.shape[0])
        t = t.to(self.dtype) / self.pipe.scheduler.config.num_train_timesteps
        return self.transformer(hidden_states=latent.to(self.dtype), timestep=t,
            encoder_hidden_states=embedding.to(device=self.device, dtype=self.dtype),
            padding_mask=torch.zeros(1, 1, latent.shape[-2] * self.scale,
                latent.shape[-1] * self.scale, device=self.device, dtype=self.dtype),
            return_dict=False)[0].float()

    @torch.no_grad()
    def decode(self, latent):
        vae = self.pipe.vae
        mean = torch.tensor(vae.config.latents_mean, device=self.device, dtype=self.dtype).view(1, -1, 1, 1, 1)
        inv_std = 1 / torch.tensor(vae.config.latents_std, device=self.device, dtype=self.dtype).view(1, -1, 1, 1, 1)
        decoded = vae.decode(latent.to(self.dtype) / inv_std + mean, return_dict=False)[0][:, :, 0]
        return self.pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    def step(self, scheduler, velocity, timestep, latent):
        # FlowMatchEuler casts its result to model_output.dtype. The official
        # Anima denoiser emits bf16; preserve that rounding in every trajectory.
        return scheduler.step(velocity.to(self.dtype), timestep, latent, return_dict=False)[0]

    @torch.no_grad()
    def render(self, prompt, seed, width=768, height=768, steps=10, progress=None, cancelled=None):
        embedding = self.encode(prompt)
        latent = self.noise(seed, width, height)
        scheduler = self.scheduler(steps)
        for index, timestep in enumerate(scheduler.timesteps):
            if cancelled and cancelled():
                raise Cancelled("Generation cancelled")
            velocity = self.predict(latent, timestep, embedding)
            latent = self.step(scheduler, velocity, timestep, latent)
            if progress:
                progress((index + 1) / steps)
        return self.decode(latent)

    def close(self):
        self.mixer.close()
        self.embeddings.clear()
        del self.transformer
        del self.pipe
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
