# History and retained evidence

Current usage lives in the [product repositories](products.md). Current shared
contracts are listed in the [documentation index](README.md).

The repository grew from Concept Sliders through embedded image, audio and text
experiments into a shared core. Older model instructions, release cards,
scoreboards and proposals described different code and objectives. They are
available at immutable revisions rather than presented as current guidance:

| Historical source | Use |
|---|---|
| [Pre-cleanup tree, `ba51374`](https://github.com/HyperGAN/particle-sliders/tree/ba513741b7d4be93d049c4e1d88168f86c41e783) | Full old docs, `MUSIC3.md`, `SCORING.md`, `LM-SCORING.md`, experiment notes and integration proposals |
| [Core 0.13.0, `285180e`](https://github.com/HyperGAN/particle-sliders/tree/285180ed9813f12e912bb6bce6bada00496fd19a) | Embedded model modules, old installed namespaces and their original loaders |
| [Archived model bindings](../legacy/model-bindings/core-0.13.0) | Replaced binding source retained for provenance |

Old checkpoints keep their original source, environment, model, data, optimizer,
step count and metric identity. A current API name or matching tensor shape does
not make an old training checkpoint portable. Do not edit source hashes to force
a restore. Render or score saved raw checkpoints when adding a missing
measurement; completed training does not need to be repeated.

## What remains in the tree

- JSON receipts, figures, audio and other captured evidence are retained as
  recorded. Their presence is not a current default or a new quality claim.
- [The November 6 architecture audit](audits/implicit-state-machines-20261106/README.md)
  records findings at that date. Subsequent fixes make some findings historical;
  use current contracts and code for present behavior.
- [Legacy sweep ownership](execution/sweep-owner-recovery.md) documents a retained
  script's process and recovery rules.
- `hub-formulation-uni16-gan.md` and `hub-native-usage.md` remain unchanged because
  the historical `scripts/publish_music3_hub_20260909.py` reads them as publishing
  inputs. They describe their historical cohort, not current usage.
- Excluded native source and research README files remain with their original
  code. Reproduce a historical workflow from its full pinned checkout, including
  its matching documentation and test suite.

The documentation cleanup removes superseded prose, not captured training
sources, full/raw checkpoints, experiment measurements or license attribution.
Historical references to EMA or older selection metrics describe those runs;
current training uses raw weights with EMA disabled. Old results are not
relabelled as the current search objective.
