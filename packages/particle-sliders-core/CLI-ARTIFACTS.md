# Explicit CLI artifacts

`particle-slider train --config local.json --out new-run` retains the existing
v1 `adapter.pt` default. Use `--export-format v2` for native-only `adapter-v2.pt`,
or `--export-format both` for both. `--selection raw|ema` applies to each export.
The product factory must supply its native-only `InferenceSpec` before training
starts when v2 is requested. Missing support fails before an update/output write.
The receipt hashes the actual selected files. `training.pt` remains the full,
strict source/cohort-bound resume checkpoint; v2 never replaces it.

Load v2 without training configuration or data:

```sh
particle-slider infer --artifact new-run/adapter-v2.pt --runtime local-native.json \
  --mode response --strength .5 --out response.pt
```

The product documents `local-native.json` and its registered native factory.
Rendering additionally requires that factory's qualified render capability.
V1 continues through its original session loader with the matching configuration.
New source pins require new cohorts; old receipts remain owned by old pins.

This additive source change and its regressions await a parent-admitted CPU slot.
