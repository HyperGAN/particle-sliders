# September 16 selected-checkpoint Hub release

Target: https://huggingface.co/ntc-ai/minimax-music3-concept-sliders

This release uses the final sixteen recommendations from the captured
`quality-later-v2` audit (80 candidates, all sixteen training budgets complete).
The user explicitly authorized publication, including ComfyUI conversion,
and clarified that the audit selections should be used instead of latest steps.

- `build_assets.py`: verifies selected IDs/hashes against the decision layer,
  checks native tensors against their pinned full states, copies all four
  matched three-arm cases per slider, and invokes the upstream converter CLI.
- `validate_comfyui.py`: checks the actual sixteen converted files using
  ComfyUI's CPU loader and unmerged Music 3 text-encoder topology. All 144
  projections per file must bind with no unused keys; q/k/v/o deltas are checked.
- `build_docs.py`: creates the model card, native/ComfyUI use guides, sample
  gallery and selection/method documentation. The prior reward showcase is retained.
- `release.py`: inventories and validates the package, publishes in one commit
  using the captured remote parent, then verifies remote file hashes and sizes.

The upstream converter is a shallow checkout under
`/tmp/music3-hf-release-20260916/conceptmod` at
`a8a9e898ea618d83f05505c5ece7c8e4ffa9c3df`. ComfyUI was checked at
`7a0b5eede3f9721c8faab290689893f36edc6d66`. Test-only dependencies were
installed under `/tmp/music3-hf-release-20260916/comfy-deps`; the training
Python environment and requirements were not modified.

Use `/home/mikkel/anaconda3/envs/minimax-music3/bin/python` with
`CUDA_VISIBLE_DEVICES=''`. ComfyUI validation also needs the converter and
isolated dependency directories on `PYTHONPATH`. No GPU generation or training
is part of this release build. Full ComfyUI audio generation is not claimed.

`published.json` records the committed revision; `remote-validation.json`
records post-upload checks. The staging package and original remote documents
are retained locally. Publication only adds new versioned weight/sample paths
and updates shared documentation/catalog/source files; historical weight and
sample paths are retained.
