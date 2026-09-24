# Legacy: upstream Concept Sliders files

These files come from the upstream
[Concept Sliders](https://github.com/rohitgandikota/sliders) repository and
are kept for reference. They are **not maintained** in this fork. For working
copies, use upstream.

| Path | What it is |
|---|---|
| `notebooks/` | Upstream SD 1.x / SDXL / SDXL-Turbo inference and image-editing demos |
| `eval-scripts/` | Upstream image generation and CLIP/LPIPS scoring scripts |
| `prompts/` | Upstream image-evaluation prompt CSVs read by `eval-scripts/` |
| `images/` | Upstream paper figure |
| `trainscripts/imagesliders/` | Upstream paired-image slider trainer (`data/config*.yaml` paths assume the repository root as working directory) |
| `requirements.txt` | Upstream diffusion-era pins (torch 2.0, diffusers 0.20) |

The notebooks and `eval-scripts/generate_images_{sd1,xl}.py` import
`trainscripts.textsliders`, an upstream path that no longer exists;
`generate_images-uce.py` expects a `lora.py` in the working directory. This fork's text-slider code lives in
[`conceptmod/textsliders/`](../conceptmod/textsliders/), which also holds
this fork's SD 1.x/2.x, SDXL, SD3, Flux and Stable Cascade trainers.

Do **not** install `requirements.txt` into a Music 3, YuE2 or modern
image/video environment. Use [`requirements-dev.txt`](../requirements-dev.txt)
for the CPU test suite and each backend guide for its runtime.
