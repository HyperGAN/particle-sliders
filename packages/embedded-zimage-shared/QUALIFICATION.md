# Qualification: embedded Z-Image shared-game preview

2026-10-31, Pop, Python3.12.15 / Torch2.6.0+cpu. All model tests use a small
random **actual Diffusers ZImageTransformer2DModel**, never downloaded model
weights. Configuration: dim32, one main layer, one refiner per stream, 4 query
heads / 2 KV heads, caption width16, channels4, native4×4 image lattice,
rotary axes4/2/2 with length128. All12 native attention projections are covered.
This is native API and CPU software conformance, not full-model/pretrained,
production sampling, GPU precision, resource or image-quality qualification.

## Results

- Built wheel and sdist using `python -m build --no-isolation`.
- Installed wheel: **29 passed**, no skips.
- Fresh public dependency environment plus installed sdist: **29 passed**, no skips.
- Historical `tests/test_zimage_slider.py`: **16 passed** (`0.95s`).
- Both installed environments: `pip check` → `No broken requirements found.`
- Clean public CPU resolution with `--ignore-installed --dry-run --report` succeeded.
- Public core VCS installation resolves exactly `7a8a321d04323d49319d0b08e6093a0ddce36b27`;
  public HyperGAN VCS installation resolves exactly `5f368a3770cf5bdae0448995c9f76e29af8049f7`.
  Their `direct_url.json` records were inspected, not inferred from version strings.
- All49 installed core source/graph/manifest files match the recursively hashed
  relative-path closure. A nested unexpected source rejects before attachment.
  Installed product Python/profile bytes match source, with imports resolving to
  site-packages. The builder verifies HNDL0.10 source closure, HyperGAN's six
  builder files and clean build provenance. Product graph source and plan digests
  are separately bound; the Fibo graph receipt does not stand in for this graph.
- The historical trainer, `slider_targets.py`, native `lora.py`, Z-Image YAML,
  historical test, root requirements-dev/pytest configuration and shared-core package
  have no source delta from base `3c25f785d14073e6f627c0fa9d2297a041c79913`.

The29 cases cover matched native outputs and all adapter/input/encoding VJPs
against the **existing** LoRANetwork at nonzero weights; critic output/input VJP
and all-parameter gradient-norm second derivatives against shared FieldCritic;
both public BISPF preset updates and actual optimizer ownership; exact4-step
versus2+2-step resume including EMA/RNG/history; native strengths0/0.5/1;
raw/EMA exports; native-only inference after deleting training rows; executable
CLI training/response; actual historical UNI safetensors refusal before factory;
model/shape/profile/source/rows/foreign-hook/attention-processor rejection before
native attachment; builder tamper and width2048 guard refusal; borrowed model
modes/requires-grad/existing gradients/checkpoint function restoration; nested
strength scopes; teacher/update/partial-attachment/cleanup failure; and late
checkpoint-load rollback with the original exception preserved. Review regressions
also bind complete rows bytes (including ignored trailers and changes between
preflight/preparation), prove identity does not deserialize weights and bounds
reads to1MiB, compare a multi-chunk safetensors digest with the native state,
reject nested extra core source, reject changed training paths before factory,
reject equal-weight reserialized files and assert exactly one native base load,
and relocate raw/EMA native-only exports after deleting original model/rows.

## Reproduce

Run modern installed tests outside the repository, so its existing pytest
`pythonpath=packages/particle-sliders-core/src` cannot substitute the old source
core for the installed modern package:

```sh
python -m build --no-isolation packages/embedded-zimage-shared
python -m pip install --no-deps packages/embedded-zimage-shared/dist/embedded_zimage_shared_preview-0.1.1-py3-none-any.whl
cd /tmp
/path/to/venv/bin/python -m pytest -c /dev/null --import-mode=importlib -q \
  /path/to/repository/packages/embedded-zimage-shared/tests
```

For a separate clean public environment (Python3.12 Linux x86_64), the actual
installation/resolution inputs were:

```sh
python -m pip install --index-url https://pypi.org/simple \
  'torch @ https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl' \
  packages/embedded-zimage-shared/dist/embedded_zimage_shared_preview-0.1.1.tar.gz pytest
```

No sibling previews, local source overlays, edited upstream PG files or full
models are required. Default historical installation remains separate.

## Remaining boundaries and review

This package is a new adapter GAN cohort, not an objective-preserving repin or a
claim that ordinary attention LoRA has a routed particle bank. No GAN loss,
penalty, optimizer, resume/EMA implementation or training loop was copied.

Shared release0.9.4/auto remains unchanged. Native continuous vectors≥170 and
rank3 packed responses remain unsupported. Real native fields stay4D through
the game and penalty; no flattening/geometry bypass is used.

Requested source/model/profile/shape and foreign cohort configuration mismatches
reject before attachment. Deep **same-config** training-packet signatures are
currently checked by the shared lifecycle after reversible preparation, before
parameter restoration. A packet-aware pre-prepare admission seam is a real shared
API gap, reported to the central core owner. Cleanup/rollback tests do not close
that admission gap. Full pre-hook rejection of every possible saved signature is
not claimed. Native-only inference source/model gates precede attachment.

Core7a8 remains held pending further lifecycle corrections/review, despite its
public availability in draft PR148. Root owns Claude
actual-diff review, fixes, return review, Ada routing and merge decisions. No
merge, model/package release, deployment or GPU run occurred. No CI workflow or
release configuration is changed by this package.

## Claude-requested bounded corrections

This update addresses complete rows-file identity, bounded streaming model-file
identity, recursive core closure and accurate relocation claims. Version0.1.1
uses cohort/protocolv2; prior0.1.0/v1 artifacts require their original source pin.
Training resume binds absolute paths; only native-only exports support relocated
byte-identical base files in the same pinned source/runtime environment.

The source delta from bca841ebb58d357f0ba93d76d53abce8d0de2c5e requires Claude
return review before landing. It does not close shared packet-aware pre-prepare
admission or admit core7a8/native training. No GPU/full-model work was added.
