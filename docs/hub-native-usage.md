# Load the native adapters

Use your working MiniMax Music 3 pipeline and its compatible PyTorch/diffusers environment. Keep the adapter and its JSON sidecar together. The release includes the existing [LoRANetwork implementation](source/lora.py); it has no package-relative imports.

Download the release with `huggingface_hub.snapshot_download`, including `source/lora.py`, the desired weight/sidecar and its prompts. This example assumes `pipe` is your already loaded Music 3 pipeline and `folder` is the snapshot directory:

```python
import importlib.util
import json
from pathlib import Path
import torch
from safetensors.torch import load_file

folder = Path(folder)
spec = importlib.util.spec_from_file_location("music3_release_lora", folder / "source/lora.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

catalog = json.loads((folder / "catalog.json").read_text())
entry = next(s for s in catalog["sliders"] if s["id"] == "female")
weights = folder / entry["weights"]
meta = json.loads(weights.with_suffix(".json").read_text())
host = pipe.language_model if meta["kind"] == "language_model" else pipe.transformer
device = next(host.parameters()).device
network = module.LoRANetwork(
    host,
    rank=meta["rank"],
    alpha=meta["alpha"],
    multiplier=1.0,
    target_replace=meta["target_replace"],
    train_method=meta["train_method"],
    delimiter=meta["delimiter"],
    prefix=meta["prefix"],
).to(device)
network.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
network.eval()

# Keep prompt, lyrics and seed fixed for an Off/On comparison.
# Use the neutral caption; do not replace it with the positive teacher caption.
for strength in (0.0, 1.0):
    network.set_lora_slider(strength)
    with torch.inference_mode(), network:
        audio = pipe(
            prompt=prompt,
            lyrics=lyrics,
            audio_duration=20,
            generator=torch.Generator(device).manual_seed(1709),
            output="audios",
        )[0]
    # Save each result with the pipeline's sampling_rate.
```

For the separate reward slider, select
`reward/refined-block-step2/reward-ce-robust-block_step2.safetensors`
instead. Its sidecar selects the acoustic host and full transformer-block topology. Load a fresh pipeline before attaching a different adapter through this example; stacking newly constructed wrappers on the same projections is not a merge strategy.

The native wrapper method is the concept comparison path. Reward research audio used ordinary full-delta merging in CPU FP32 followed by the model's BF16 cast; low-precision evaluation order can make wrapper output differ from those recordings. Use the recorded WAVs for exact published comparisons. The research package's native merge audits verified the selected reward export.

The adapters contain no base-model weights. There is no trained negative pole. The studio normalizes its fader mixture by host energy; to reproduce a direct multiplier of `1`, inspect the resolved multiplier instead of assuming the fader position equals the effective strength.

For the selected ComfyUI exports, use the [ComfyUI guide](comfyui/README.md). The native and ComfyUI catalog entries identify the same selected checkpoints in their respective formats.
