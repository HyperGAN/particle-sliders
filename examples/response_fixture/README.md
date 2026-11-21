# Installed numerical response example (offline fixture)

Synthetic adapter-only vector example for core 0.15.2 / ParticleGAN 0.11.1. No native model, token or audio renderer.
The fixture has installed CPU qualification; it makes no native media claim.
Use a separate environment with this branch's exact core/ParticleGAN dependency.
From this example directory, run `python -m pip install .` to install the plugin
and its pinned shared runtime. To test the installed entry points, run the
following from a separate directory:

```sh
mkdir fresh-vector-run
cd fresh-vector-run
printf '.3' > base.txt
printf '4' > rows.txt
cat > config.json <<'JSON'
{"backend":"vector-fixture","cohort":"new-vector-example-source-cohort","options":{"steps":4,"model_file":"base.txt","training_file":"rows.txt","ema_decay":null}}
JSON
particle-slider capabilities --interface response --config config.json
particle-slider train --interface response --config config.json --out run \
  --stop-after 2 --export-format both --selection raw
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
