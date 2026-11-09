# Historical Concept Sliders source

These files come from [Concept Sliders](https://github.com/rohitgandikota/sliders)
and are retained for reference and attribution. They are outside the current
core distribution and are not maintained as working model integrations. Use
the upstream project for its original diffusion workflows.

| Path | Historical content |
|---|---|
| `notebooks/` | SD 1.x, SDXL and SDXL-Turbo demos |
| `eval-scripts/` and `prompts/` | Image generation, CLIP/LPIPS scoring and input CSVs |
| `images/` | Original paper figure |
| `trainscripts/imagesliders/` | Paired-image slider trainer |
| `requirements.txt` | Original diffusion-era dependency pins |

Some scripts assume old working directories or namespaces that are absent from
the current distribution. Reproduce them with their original complete checkout,
not by mixing the old requirements into a current product environment.

For current model usage, see the [product repositories](../docs/products.md).
For the shared core, see [development](../docs/development.md).
[History](../docs/history.md) explains immutable source cohorts.
