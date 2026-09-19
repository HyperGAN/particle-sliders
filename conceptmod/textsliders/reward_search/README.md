# Reward CE search

Follow-up to `reward_sliders`, using its frozen scorer, capture hooks and ordinary
Music 3 LoRA loader. The user authorized experiments on both GPUs, prioritizing
fixed-length CE improvement over natural completion.

The [run record](../../../analysis/reward_search_20260908/README.md) includes the
frozen design, provenance, live stage, engineering amendments and scored decisions.

The first search refines the existing step-600 adapter's strength. Sixteen new
training families expand the original eight to 24. Their saved histories support
a refitted activation teacher and two new standard rank-8/alpha-8 students: the
bounded baseline recipe and feature-matching gradient limiting. Student rows
sample four windows throughout 20 seconds, always retaining the full preceding
prompt and feedback. Checkpoints and strengths are chosen by development CE.

Eight separate final families compare the frozen selection with Off and the
original v1 adapter. All treatment arms within a matched seed bundle run on one
physical GPU. No test-family results are used to select the candidate.

`setup.py` freezes and queues the initial collection. `renderer.py` atomically
claims bundles through `queue.py`. `resources.py` waits for the studio queue to
drain before borrowing GPU 0, keeps playback available, and restores the original
GPU-0 studio assignment afterward. `campaign.py` advances scored stages and
training automatically. `report.py` writes the readable progress record.

Use the existing `minimax-music3` environment. Do not install the repository's old
requirements. Source and protocol changes require an explicit amendment; original
audio, failures, source snapshots and optimizer states remain available.
