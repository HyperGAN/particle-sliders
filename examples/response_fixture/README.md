# Installed numerical response example (offline fixture)

Synthetic adapter-only vector example. No native model, token or audio renderer.
The fixture has installed CPU qualification; it makes no native media claim.
Use a separate environment with this branch's exact core/ParticleGAN dependency.
Build this tiny plugin with `python -m pip wheel --no-deps --no-build-isolation .`,
then install its wheel alongside the reviewed core wheel. Outside the source tree:

```sh
mkdir fresh-vector-run
cd fresh-vector-run
printf '.3' > base.txt
printf '4' > rows.txt
cat > config.json <<'JSON'
{"backend":"vector-fixture","cohort":"new-vector-example-source-cohort","options":{"model_file":"base.txt","training_file":"rows.txt","ema_decay":0.9}}
JSON
particle-slider capabilities --interface response --config config.json
particle-slider train --interface response --config config.json --out run \
  --stop-after 2 --export-format both --selection ema
particle-slider evaluate --interface response --config config.json \
  --resume run/training.pt --strength .5 --out response.pt
printf '{"model_file":"base.txt"}' > native.json
rm rows.txt
particle-slider infer --artifact run/adapter-v2.pt --runtime native.json \
  --mode response --strength .5 --out data-free-response.pt
```

To resume training, restore the same local rows file and use the same source pin,
configuration and original total budget. Inference needs only `base.txt` and the
v2 artifact; it constructs no game, teacher, critic or optimizer. Results remain
synthetic vectors; rendering fails closed. The core owns all session orchestration.
