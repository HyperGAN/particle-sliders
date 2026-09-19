#!/usr/bin/env python3
"""Development-only ASR confidence/crop probe, independent of frozen pilot v1.

No lyric prompt, no decoding retries, no human labels used by measurement.
Scores are raw teacher-forced log probabilities of the decoder's own text,
not an assertion that confident transcription is correct. Whole-clip and
contiguous ten-second decoding expose missed phrases and unreliable tails.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slider_selection.features import digest, file_hash, write_json

MODEL = "openai/whisper-large-v3-turbo"
REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


def chunk_bounds(length, rate):
    ends = list(range(10 * rate, length, 10 * rate))
    if ends and length - ends[-1] < rate // 2:
        ends.pop()
    points = [0] + ends + [length]
    return list(zip(points[:-1], points[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "eval/slider-quality/pilot/manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/slider-quality/pilot/acoustic-probe.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "eval/slider-quality/asr-confidence-cache")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    os.environ.setdefault("HF_HOME", "/ml2/music/.cache/huggingface")
    import numpy as np
    import soundfile as sf
    import torch
    from torchaudio.functional import resample
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(4)
    torch.manual_seed(0)
    from slider_selection.dataset import load_manifest
    manifest = load_manifest(args.manifest)
    protocol = {"model": MODEL, "revision": REVISION, "dtype": "float16", "device": "cuda",
                "language": "en", "task": "transcribe", "beams": 1, "max_tokens": 128,
                "prompt": None, "script_sha256": file_hash(Path(__file__)),
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers")}}
    paths = sorted({path for r in manifest["records"] for path in [r["baseline"]] + [c["path"] for c in r["clips"]]})
    proc = model = None
    results = {}
    for index, path in enumerate(paths, 1):
        sha = file_hash(Path(path))
        cached = args.cache / f"{sha}-{digest(protocol)[:16]}.json"
        if cached.exists():
            results[path] = json.loads(cached.read_text())
            continue
        if model is None:
            proc = WhisperProcessor.from_pretrained(MODEL, revision=REVISION)
            model = WhisperForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16).to("cuda:0").eval()
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        audio = resample(torch.from_numpy(data.mean(axis=1)), sr, 16000).numpy()
        parts = [("whole", 0, len(audio))] + [("chunk", start, end) for start, end in chunk_bounds(len(audio), 16000)]
        decoded = []
        for kind, start, end in parts:
            if end - start > 30 * 16000:
                raise ValueError("Whole-clip probe is defined only for clips up to 30 seconds")
            inputs = proc(audio[start:end], sampling_rate=16000, return_tensors="pt").input_features.to("cuda:0", torch.float16)
            with torch.inference_mode():
                ids = model.generate(inputs, language="en", task="transcribe", do_sample=False,
                                     num_beams=1, max_new_tokens=128, return_timestamps=False)
                text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
                # Whisper versions differ in whether returned sequences retain
                # the decoder prefix. Rebuild it explicitly from lexical tokens.
                lexical_ids = ids[ids < model.config.eos_token_id].tolist()
                prefix = [model.config.decoder_start_token_id] + [token for _, token in
                          proc.get_decoder_prompt_ids(language="en", task="transcribe", no_timestamps=True)]
                if lexical_ids:
                    sequence = torch.tensor([prefix + lexical_ids], device="cuda:0")
                    logits = model(input_features=inputs, decoder_input_ids=sequence[:, :-1]).logits.float()
                    targets = sequence[:, 1:]
                    logprobs = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    values = logprobs[targets < model.config.eos_token_id].cpu().numpy()
                else:
                    values = np.array([], dtype=float)
            decoded.append({"kind": kind, "start_s": start/16000, "end_s": end/16000,
                            "text": text, "lexical_tokens": int(len(values)),
                            "mean_logprob": float(values.mean()) if len(values) else None,
                            "p10_logprob": float(np.quantile(values, .1)) if len(values) else None,
                            "token_logprobs": values.tolist()})
        result = {"sha256": sha, "protocol": protocol, "segments": decoded}
        write_json(cached, result)
        results[path] = result
        print(f"Measured {index}/{len(paths)}: {Path(path).parent.name}/{Path(path).name}", flush=True)
    write_json(args.out, {"scope": "Development probe, not a selection score", "manifest_id": manifest["id"],
                          "measurement_id": digest(protocol), "audio": results})


if __name__ == "__main__":
    main()
