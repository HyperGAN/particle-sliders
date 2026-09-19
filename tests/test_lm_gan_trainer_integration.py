"""Exercise the real GAN trainer loop with a tiny CPU LM and cached frames."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from conceptmod.textsliders import train_lm_slider_music3 as trainer


class Qwen3Attention(nn.Module):
    """The production LoRA wrapper discovers this same attention class name."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden):
        return self.proj(hidden)


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.attn = Qwen3Attention()

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        positions = torch.arange(1, hidden.shape[1] + 1, device=hidden.device)[None, :, None]
        hidden = self.attn(hidden.cumsum(dim=1) / positions)
        # Represent numerical dependence on GEMM/attention sequence shape.
        # A prompt-only neutral cache is measurably wrong for this model,
        # while the matching teacher-forced neutral is exactly stationary.
        hidden = hidden + hidden.shape[1] * 0.01
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyBackbone()


@pytest.mark.parametrize("frames_present", [True, False])
@pytest.mark.parametrize("arch,adv,fm,txfm,parts,condition,bipolar,fm_mode", [
    ("none", 0, 0, 0, 0, "none", False, "batch"),
    ("mlp", 0, 0, 0, 0, "none", False, "batch"),
    ("mlp", 1, 1, 0, 0, "none", False, "batch"),
    ("tx", 1, 1, 0, 0, "none", False, "batch"),
    ("mlp", 0, 1, 0, 0, "none", False, "batch"),
    ("mlp", 0, 0, 1, 0, "none", False, "batch"),
    ("tx", 1, 1, 0, 1, "none", False, "batch"),
    ("tx", 1, 1, 0, 0, "row", False, "batch"),
    ("tx", 1, 1, 0, 1, "row", False, "batch"),
    ("mlp", 1, 1, 0, 0, "row", True, "batch"),
    ("mlp", 0, 0, 1, 0, "row", False, "batch"),
    ("tx", 0.1, 1, 0, 0, "row", False, "paired"),
    ("tx", 0.1, 1, 0, 1, "row", False, "paired"),
    ("mlp", 0.1, 1, 0, 0, "row", True, "paired"),
    ("mlp", 0, 0, 1, 0, "row", False, "paired"),
])
def test_cached_endreg_zero_lora_and_legacy_path(tmp_path, monkeypatch, arch, frames_present, adv, fm, txfm, parts, condition, bipolar, fm_mode):
    import transformers
    from safetensors.torch import load_file

    gan_on = bool(adv or fm or txfm)
    num_rows = 2 if parts or condition == "row" else 1
    original_conditional = trainer._lm_adv.RowConditionalD
    conditional_heads = []

    class RecordingConditionalD(original_conditional):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen_ids = []
            conditional_heads.append(self)

        def forward(self, delta, mask=None, row_ids=None):
            self.seen_ids.append(tuple(row_ids.detach().cpu().tolist()))
            return super().forward(delta, mask, row_ids)

    monkeypatch.setattr(trainer._lm_adv, "RowConditionalD", RecordingConditionalD)

    class CPUDeviceTorch:
        def device(self, spec):
            return torch.device("cpu")

        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(trainer, "torch", CPUDeviceTorch())
    original_adamw = torch.optim.AdamW
    generator_optimizers = []

    def capture_adamw(*args, **kwargs):
        opt = original_adamw(*args, **kwargs)
        generator_optimizers.append(opt)
        return opt

    monkeypatch.setattr(torch.optim, "AdamW", capture_adamw)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: _TinyLM())
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(trainer, "_load_rows", lambda path: ([{
        "neutral": "neutral", "target": "neutral", "positive": "positive",
        "negative": "negative", "lyrics": "a short lyric",
    }] * num_rows, {}))
    monkeypatch.setattr(trainer, "_assemble", lambda caption, lyrics: caption)

    def tokenize(tokenizer, text, device):
        caption_token = {"neutral": 1, "positive": 2, "negative": 3}[text]
        caption = [caption_token, caption_token] if text == "positive" else [caption_token]
        tokens = torch.tensor([caption + [10, 11, 12, 13, 14]])
        return tokens, torch.ones_like(tokens)

    monkeypatch.setattr(trainer, "_tokenize", tokenize)
    monkeypatch.setattr(trainer, "_assert_last_token_is_audio_start", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "_assert_lyric_span", lambda *a, **k: (
        torch.tensor([[0, 0, 1, 1, 1, 0]]), torch.tensor([[0, 0, 0, 1, 1, 1, 0]])
    ))
    monkeypatch.setattr(trainer, "_frame_margins", lambda lm, hidden: hidden.float().mean(dim=-1)[0])
    cached = tmp_path / "cached-preroll.pt"
    frames = torch.full((1, 3, 8), 0.25) if frames_present else None
    torch.save({"frame_embeds": frames, "ended": not frames_present}, cached)
    monkeypatch.setattr(trainer, "_endreg_cache_path", lambda *a, **k: cached)

    def forbidden_preroll(*args, **kwargs):
        raise AssertionError("existing frame cache must be reused")

    monkeypatch.setattr(trainer, "_preroll_frames", forbidden_preroll)
    args = trainer.parse_args([
        "--prompts_file", "unused.yaml", "--name", "gan-cpu", "--save_dir", str(tmp_path),
        "--lm_target", "faithful" if bipolar else "faithful_plus_neu_lyric", "--steps", "2", "--rank", "2", "--alpha", "2",
        "--adv_arch", arch, "--adv_in", "scaled", "--adv_weight", str(adv), "--fm_weight", str(fm),
        "--txfm_weight", str(txfm), "--parts", str(parts), "--adv_condition", condition, "--fm_mode", fm_mode,
        "--adv_hidden", "16", "--adv_width", "16", "--adv_layers", "1", "--adv_heads", "2",
        "--pole_weight", str(int(not gan_on)), "--endreg_weight", "1", "--endreg_frames", "3",
        "--endreg_cache", str(tmp_path), "--no-early_stop", "--save_every", "0", "--grad_account",
    ])
    check_resume = arch == "tx" and adv == 1 and fm == 1 and not parts and condition == "none"
    args.save_training_state = check_resume
    checkpoint = trainer.train(args)
    assert checkpoint.exists()
    report = json.loads(checkpoint.with_suffix(".json").read_text())
    first = report["first"]
    if gan_on:
        if not bipolar:
            assert first["mag_ratio"] == 0.0
        assert bool(first["gadv_norm"] > 0.0) == bool(adv)
        assert first["grad_norm"] > 0.0
        assert report["adv"]["input_scale"] > 0
        assert report["adv"]["cap_space"] == "teacher_rms"
        assert generator_optimizers[0].defaults["betas"][0] == 0
    else:
        # The repaired teacher coordinates and GAN optimizer are opt-in.
        # Preserve legacy supervised initial deltas and momentum.
        assert bool(first["mag_ratio"] > 0) == frames_present
        assert first["gadv_norm"] == 0
        assert report["adv"]["enabled"] is False
        assert report["adv"]["input_scale"] is None
        assert generator_optimizers[0].defaults["betas"][0] == 0.9
    assert first["edrift_p"] == 0.0
    assert report["endreg"]["rows_ended_naturally"] == num_rows * int(not frames_present)
    assert report["parts"]["enabled"] == bool(parts)
    assert report["txfm"]["enabled"] == bool(txfm)
    assert report["adv"]["condition"] == condition
    assert bool(conditional_heads) == (condition == "row")
    for head in conditional_heads:
        stride = 2 if bipolar else 1
        assert head.embedding.num_embeddings == num_rows * stride
        seen = {row_id for ids in head.seen_ids for row_id in ids}
        assert seen == set(range(num_rows * stride))
        assert torch.isfinite(head.embedding.weight).all()
        if bipolar:
            # A distinct condition for each polarity is essential: one row
            # ID for both signs still admits swapping the two teacher poles.
            assert (0, 1) in head.seen_ids and (2, 3) in head.seen_ids
            assert all((idx,) in head.seen_ids for idx in range(4))
    if parts:
        assert 0 < first["w_max"] < 1
    for record in (first, report["last"]):
        assert all(torch.isfinite(torch.tensor(value)) for value in record.values() if isinstance(value, float))
    weights = load_file(str(checkpoint))
    assert all(torch.isfinite(value).all() for value in weights.values())
    # lora_up starts at zero; this pins an actual generator update even on
    # FM-only and secondary-TX-only cards, where gadv_norm is correctly zero.
    assert any(torch.count_nonzero(value) for name, value in weights.items() if "lora_up" in name)
    if check_resume:
        from copy import deepcopy
        source_state = tmp_path / "gan-cpu_state.pt"
        assert source_state.exists()
        full = deepcopy(args)
        full.name = "gan-full"
        full.steps = 4
        full_weights = load_file(str(trainer.train(full)))
        resumed = deepcopy(full)
        resumed.name = "gan-resumed"
        resumed.resume_state = str(source_state)
        resumed_weights = load_file(str(trainer.train(resumed)))
        assert full_weights.keys() == resumed_weights.keys()
        assert all(torch.equal(full_weights[key], resumed_weights[key]) for key in full_weights)
        assert (tmp_path / "gan-full_train.jsonl").read_text() == (tmp_path / "gan-resumed_train.jsonl").read_text()
        changed = deepcopy(resumed)
        changed.name = "gan-changed"
        changed.lr *= 2
        with pytest.raises(ValueError, match="different training settings"):
            trainer.train(changed)

        from analysis.gan_bcap.stability_experiment import StabilityHooks
        baseline = torch.load(source_state, map_location="cpu", weights_only=True)["signature"]
        for mode, lr_scale, cap in [("noop", 1., None), ("lower_lr", .25, None), ("fm_cap", 1., 1e-6)]:
            branch = deepcopy(resumed)
            branch.name = f"gan-branch-{mode}"
            branch.lr = baseline["settings"]["lr"] * lr_scale
            if lr_scale != 1.:
                branch.adv_lr = baseline["settings"]["lr"] * trainer._lm_adv.D_LR_MULT
            telemetry = tmp_path / f"{mode}-telemetry.jsonl"
            with StabilityHooks(baseline_signature=baseline, g_lr_scale=lr_scale,
                                fm_input_grad_cap=cap, telemetry_path=telemetry,
                                diagnostics_every=1).installed():
                branch_weights = load_file(str(trainer.train(branch)))
            records = [json.loads(line) for line in telemetry.read_text().splitlines()]
            assert [row["step"] for row in records] == [3, 4]
            assert all(row["parameter_update_norm"] > 0 for row in records)
            saved = torch.load(tmp_path / f"{branch.name}_state.pt", map_location="cpu", weights_only=True)
            assert saved["optimizers"]["lora"]["param_groups"][0]["lr"] == branch.lr
            assert saved["optimizers"]["critic"]["param_groups"][0]["lr"] == baseline["settings"]["lr"] * trainer._lm_adv.D_LR_MULT
            if mode == "noop":
                assert all(torch.equal(full_weights[key], branch_weights[key]) for key in full_weights)
                assert (tmp_path / "gan-full_train.jsonl").read_text() == (tmp_path / f"{branch.name}_train.jsonl").read_text()
            elif mode == "fm_cap":
                assert any(row["fm_cap_active_rows"] > 0 for row in records)
                assert all(r["limited_input_gradient_norm"] <= cap * (1 + 1e-6) for row in records for r in row["fm_rows"])
            else:
                assert any(not torch.equal(full_weights[key], branch_weights[key]) for key in full_weights)
            next_run = deepcopy(branch)
            next_run.name += "-plain-resume"
            next_run.steps = 5
            next_run.resume_state = str(tmp_path / f"{branch.name}_state.pt")
            with pytest.raises(ValueError, match="different training settings"):
                trainer.train(next_run)
            if mode == "fm_cap":
                next_run.name = "gan-branch-fm-cap-resumed"
                with StabilityHooks(baseline_signature=baseline, g_lr_scale=lr_scale,
                                    fm_input_grad_cap=cap, telemetry_path=tmp_path / "cap-resumed.jsonl",
                                    diagnostics_every=1).installed():
                    continued = trainer.train(next_run)
                assert continued.exists()
                uninterrupted = deepcopy(branch)
                uninterrupted.name = "gan-branch-fm-cap-uninterrupted"
                uninterrupted.steps = 5
                with StabilityHooks(baseline_signature=baseline, g_lr_scale=lr_scale,
                                    fm_input_grad_cap=cap, telemetry_path=tmp_path / "cap-uninterrupted.jsonl",
                                    diagnostics_every=1).installed():
                    uninterrupted_weights = load_file(str(trainer.train(uninterrupted)))
                continued_weights = load_file(str(continued))
                assert all(torch.equal(continued_weights[k], v) for k,v in uninterrupted_weights.items())
                assert (tmp_path / f"{next_run.name}_train.jsonl").read_text() == (tmp_path / f"{uninterrupted.name}_train.jsonl").read_text()

        from analysis.gan_bcap.lyric_preservation_experiment import lyric_hooks
        for label, weight in [('unchanged', baseline['settings']['lyrichold_weight']), ('light', .1)]:
            branch = deepcopy(resumed)
            branch.name = f'gan-lyric-{label}'
            branch.lyrichold_weight = weight
            with lyric_hooks(baseline_signature=baseline, weight=weight,
                             telemetry_path=tmp_path / f'lyric-{label}.jsonl').installed():
                result = trainer.train(branch)
            lyric_weights = load_file(str(result))
            saved = torch.load(tmp_path / f'{branch.name}_state.pt', map_location='cpu', weights_only=True)
            assert saved['signature']['settings']['lyrichold_weight'] == weight
            assert saved['signature']['stability_experiment']['lyric_preservation']['weight'] == weight
            if label == 'unchanged':
                assert all(torch.equal(v, lyric_weights[k]) for k,v in full_weights.items())
                assert (tmp_path / 'gan-full_train.jsonl').read_text() == (tmp_path / f'{branch.name}_train.jsonl').read_text()
            else:
                assert any(not torch.equal(v, lyric_weights[k]) for k,v in full_weights.items())
                continued_args = deepcopy(branch)
                continued_args.name += '-continued'
                continued_args.steps = 5
                continued_args.resume_state = str(tmp_path / f'{branch.name}_state.pt')
                with lyric_hooks(baseline_signature=baseline, weight=weight,
                                 telemetry_path=tmp_path / 'lyric-continued.jsonl').installed():
                    continued_weights = load_file(str(trainer.train(continued_args)))
                full_args = deepcopy(branch)
                full_args.name += '-full'
                full_args.steps = 5
                with lyric_hooks(baseline_signature=baseline, weight=weight,
                                 telemetry_path=tmp_path / 'lyric-full.jsonl').installed():
                    all_weights = load_file(str(trainer.train(full_args)))
                assert all(torch.equal(v, continued_weights[k]) for k,v in all_weights.items())
                assert (tmp_path / f'{continued_args.name}_train.jsonl').read_text() == (tmp_path / f'{full_args.name}_train.jsonl').read_text()
                changed_args = deepcopy(continued_args)
                changed_args.name += '-changed'
                changed_args.lyrichold_weight = .2
                with lyric_hooks(baseline_signature=baseline, weight=.2,
                                 telemetry_path=tmp_path / 'lyric-invalid.jsonl').installed():
                    with pytest.raises(ValueError, match='different training settings'):
                        trainer.train(changed_args)

        from analysis.gan_bcap.parameter_step_limit import step_limited_hooks
        for label, maximum in [('inactive', 100.), ('active', 1e-5)]:
            branch = deepcopy(resumed)
            branch.name = f'gan-step-limit-{label}'
            telemetry = tmp_path / f'step-limit-{label}.jsonl'
            with step_limited_hooks(baseline_signature=baseline, weight=branch.lyrichold_weight,
                                    maximum=maximum, telemetry_path=telemetry).installed():
                step_weights = load_file(str(trainer.train(branch)))
            records = [json.loads(l) for l in telemetry.read_text().splitlines()]
            # Tiny bounds approach float32 weight resolution; allow one small
            # absolute rounding margin, while the float64 unit check is exact.
            assert all(r['parameter_update_norm'] <= maximum + 3e-8 for r in records)
            if label == 'inactive':
                assert all(torch.equal(v,step_weights[k]) for k,v in full_weights.items())
                assert (tmp_path/'gan-full_train.jsonl').read_text() == (tmp_path/f'{branch.name}_train.jsonl').read_text()
            else:
                assert any(not torch.equal(v,step_weights[k]) for k,v in full_weights.items())
                next_args = deepcopy(branch)
                next_args.name += '-continued'
                next_args.steps = 5
                next_args.resume_state = str(tmp_path/f'{branch.name}_state.pt')
                with step_limited_hooks(baseline_signature=baseline, weight=branch.lyrichold_weight,
                                        maximum=maximum, telemetry_path=tmp_path/'step-cap-continued.jsonl').installed():
                    next_weights = load_file(str(trainer.train(next_args)))
                full_args = deepcopy(branch)
                full_args.name += '-full'
                full_args.steps = 5
                with step_limited_hooks(baseline_signature=baseline, weight=branch.lyrichold_weight,
                                        maximum=maximum, telemetry_path=tmp_path/'step-cap-full.jsonl').installed():
                    all_weights = load_file(str(trainer.train(full_args)))
                assert all(torch.equal(v,next_weights[k]) for k,v in all_weights.items())
                assert (tmp_path/f'{next_args.name}_train.jsonl').read_text() == (tmp_path/f'{full_args.name}_train.jsonl').read_text()
                wrong = deepcopy(next_args)
                wrong.name += '-wrong'
                with step_limited_hooks(baseline_signature=baseline, weight=branch.lyrichold_weight,
                                        maximum=maximum*2, telemetry_path=tmp_path/'step-cap-wrong.jsonl').installed():
                    with pytest.raises(ValueError, match='different training settings'):
                        trainer.train(wrong)
