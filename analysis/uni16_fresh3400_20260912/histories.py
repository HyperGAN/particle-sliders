"""Exact history builder copied from the audited fresh-seed experiment."""
from hashlib import sha256
from pathlib import Path
import time
import torch
from conceptmod.textsliders import train_lm_slider_music3 as legacy
from conceptmod.textsliders.gan_v2.data import gather, digest, allowed_logits, sha

class HistoryFactory:
    def __init__(self, lm, tokenizer, network, row, initial, cache, model_dir, device):
        self.lm, self.network, self.row = lm, network, row
        self.initial, self.cache, self.model_dir, self.device = initial, cache, model_dir, device
        self.cache.mkdir(exist_ok=True)
        self.neutral = legacy._assemble(row.get("neutral") or row["target"], row["lyrics"])
        positive = legacy._assemble(row["positive"], row["lyrics"])
        self.nt, nm = legacy._tokenize(tokenizer, self.neutral, device)
        pt, pm = legacy._tokenize(tokenizer, positive, device)
        self.ns, ps = legacy._assert_lyric_span(self.nt, nm, pt, pm, tokenizer, row["lyrics"], where="fresh-history")
        self.ps = ps
        with torch.no_grad():
            self.ne = lm.model.embed_tokens(self.nt)
            self.pe = lm.model.embed_tokens(pt)
        self.depth = None

    @torch.no_grad()
    def get(self, seed):
        legacy._set_scale(self.network, 0.)
        path = legacy._endreg_cache_path(self.cache, str(self.model_dir), self.neutral, 250, seed)
        started = time.monotonic()
        if path.exists():
            blob = torch.load(path, map_location="cpu", weights_only=True)
        else:
            if self.depth is None:
                from diffusers import MiniMaxMusic3RVQDepthDecoder
                self.depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
                    str(self.model_dir / "rvq_depth_decoder"), torch_dtype=torch.bfloat16,
                    local_files_only=True,
                ).to(self.device).eval().requires_grad_(False)
            frames, ended = legacy._preroll_frames(self.lm, self.depth, self.nt, 250, seed, self.device)
            if frames is None:
                raise RuntimeError("Fresh history ended before its first frame; retain seed and investigate")
            blob = dict(frame_embeds=frames.cpu(), ended=bool(ended))
            temporary = path.with_suffix(".tmp")
            torch.save(blob, temporary)
            temporary.replace(path)
        frames = blob["frame_embeds"].to(self.device)
        _, end_teacher, nh = legacy._forward_teacher_forced(self.lm, self.ne, frames)
        _, _, ph = legacy._forward_teacher_forced(self.lm, self.pe, frames)
        neutral_span = gather(nh[:, :self.ne.shape[1]], self.ns)
        positive_span = gather(ph[:, :self.pe.shape[1]], self.ps)
        real = positive_span - neutral_span
        item = dict(
            prompt_index=0, history_seed=seed, prompt_hash=digest(self.row),
            lyric_hash=digest(self.row["lyrics"]), fixture=digest([self.row, seed, 250, 4]),
            prompt_embeds=self.ne.cpu(), frame_embeds=frames.cpu(),
            neutral_span=neutral_span.cpu(), condition=neutral_span.cpu(), real=real.cpu(),
            span_mask=self.ns.cpu(), end_teacher=end_teacher.cpu(),
            policy_teacher=allowed_logits(self.lm, ph[:, self.pe.shape[1]-1:])[0][::4].cpu(),
            continuation_teacher=ph[:, self.pe.shape[1]-1::4].float().cpu(), policy_stride=4,
            ended=bool(blob["ended"]), history_cache=str(path), history_cache_sha256=sha(path),
        )
        frame_hash = sha256(frames.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        audit = dict(seed=seed, frames=frames.shape[1], frame_sha256=frame_hash,
            cache_sha256=item["history_cache_sha256"], sampling_and_teacher_seconds=time.monotonic()-started,
            real_target_max_abs_change=float((item["real"]-self.initial["real"]).abs().max()),
            neutral_span_max_abs_change=float((item["neutral_span"]-self.initial["neutral_span"]).abs().max()))
        if item["end_teacher"].shape == self.initial["end_teacher"].shape:
            audit["ending_target_rms_change"] = float((item["end_teacher"]-self.initial["end_teacher"]).square().mean().sqrt())
        legacy._set_scale(self.network, 1.)
        return item, audit
