All 16 particle sliders have been distilled into experimental ordinary rank-8 LoRAs using both GPUs. The students contain only low-rank matrices and alpha values; they need no particle cloud, router, or custom ComfyUI node.

Use standard **Load LoRA**, connect MODEL and CLIP, and set **MODEL strength 0 / CLIP strength 1**. The workflow is included. Native Q/K/V rank-8 branches become a standard rank-24 fused QKV branch in the ComfyUI export; O remains rank 8.

- **ComfyUI ZIP:** all 16 converted LoRAs, workflow, method and validation results.
- **Native ZIP:** all 16 native LoRAs, standalone loader, usage guide and validation results.
- **Comparisons ZIP:** open `index.html` to hear 128 matched recordings with large audio controls.

The median held-out relative steering MSE across the 14 genre sliders is **0.0443**. Female and male are weaker approximations at **0.3832** and **0.3998**. These measure internal teacher fidelity, not perceptual quality. The original particle release and Space remain available.

Every export bound all 56 attention patches through the real ComfyUI LoRA loader, and all 128 diagnostic clips passed finite stereo 48 kHz signal checks. Audio comparisons use the native runtime, two held-out prompts, seed 1709, and a 20-second diagnostic guard. The fourth condition also adapts acoustic-prefix conditioning to mirror standard ComfyUI scope; it is not a full ComfyUI GPU audio render.

Read [DISTILLATION.md](https://github.com/mikkel/yue2-concept-sliders/blob/main/DISTILLATION.md) for the math, validation selection and limits. Weights retain **CC BY-NC 4.0** terms. `artifact-manifest.json` records download checksums.
