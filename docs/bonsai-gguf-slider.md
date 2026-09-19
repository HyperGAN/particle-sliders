# Bonsai-GGUF particle slider (opt-in)

Runs the YuE2 working game `anneal-routed-particle-error` (`--recipe
particle_bridge`) through the **shared**
`conceptmod/textsliders/particle_bridge_gan` module on frozen
**Ternary-Bonsai-2-27B** readouts. Opt-in only. Not a default, not a
production recipe. Propose-only: does not flip Music ARM_B, live
`--lm_target`, locked AdvConfig, or YuE2 defaults.

## Which model

Pinned: **`prism-ml/Ternary-Bonsai-2-27B-gguf` /
`Ternary-Bonsai-2-27B-PTQ1_0.gguf`** (5,946,648,928 bytes, Apache-2.0,
ungated), derived from `Qwen/Qwen3.8-27B`.

- GGUF arch `qwen35`: 64 blocks, hidden **5120**, 24 query / 4 kv heads,
  context **262144**, vocab 248320. Verified 2026-09-17 against the Hub
  card **and** a range-fetched GGUF header: magic `GGUF`, 851 tensors
  (402x type **143** ternary, 353x F32 norms/state, 96x type 30),
  `prism.hadamard.*` keys present. Discard these numbers if the card
  disagrees.
- PTQ1_0 is the smallest pack (dense trits, 1.75 bits/weight). PQ2_0
  (7.21 GB) is legitimate on the card but **not accepted here** — one
  pinned file so every smoke means the same bytes.
- Text-only: neither mmproj vision pack is loaded, ever. The F16 pack
  (53.8 GB) has **no code path** and is never downloaded.

## Stock llama.cpp is rejected

The ternary weights live in a rotated basis and need the fork's Hadamard
activation runtime plus its custom ternary kernels:

- **PTQ1_0 (type 143) / PQ2_0 (type 142)** sit past upstream's
  `GGML_TYPE_COUNT`: stock llama.cpp refuses them outright (safe
  failure). `transformers` cannot load them either.
- **The dev-repo Q2_0 band is dangerous**: stock loads it without
  warning and outputs garbage (known type + supported `qwen35` arch).
  It lives outside the model repo and is never touched here.
- Required runtime: the **[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)
  fork**. Run source of truth:
  **[PrismML-Eng/Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo)**
  (tested setup, pinned binaries, serving flags). Where anything here
  disagrees with it, it is right.
- The backend enforces this fail-closed: `resolve_weights` accepts only
  the pinned filename (F16 / mmproj / PQ2_0 / Q2_0-dev each get their
  own refusal message), `probe_gguf` verdicts the file header (arch,
  shape, `prism.hadamard.*` markers, ternary tensor types), and
  `check_fork_log` rejects stock load-log signatures — ambiguous logs
  never pass silently.

## Working game: YuE2 routed particle bridge (transferred, not reinvented)

Same game as the Qwen3-0.6B path: Rp paired-error GAN on
`e = T(student) - T(positive)` (same noise on real/fake, sigma anneals
0.03 over 8000), lazy b_cap every 4th update x4, particle VIC, EMA
0.995. No output MSE, FM, ending, hold, or anchor anywhere. Unipolar (+
vs raw positive); scale 0 bypasses exactly; `unconditional` rows are
unscored canaries.

### MATCH (read from the YuE2 source, never copied)

- Game math, LRs (G/D/particle 0.0006/0.0009/0.006), Adam betas (0,
  0.999), constant schedule, 128x4 standard-normal cloud, VIC coeff 1
  (std hinge + off-diag cov, 64-particle sample), b_cap L2 k=1 coeff 1
  lazy-4 x4, EMA 0.995, noise horizon 8000 — all imported from
  `particle_bridge_gan.REFERENCE` / `shared.update` / `shared.build_game`
  (tests assert module identity plus every number).
- Adapter math: router + `softmax(q P^T / sqrt(4))` route over one shared
  cloud, `up` zero-initialized, rank/alpha pinned 8/8.
- Save convention: EMA `_last` + live `_live_last` safetensors plus a
  `_last.json` sidecar, same as the YuE2 run and the tiny path. Format
  is its own `conceptmod-bonsai-gguf-particle-v1`.

### DRIFT vs the Qwen3-0.6B path (deliberate, honest)

- **Readout**: fork-server last-token embedding (`--embedding --pooling
  last`, dim 5120) instead of live-transformers last-hidden (dim 1024).
  The server euclidean-normalizes embeddings (`--embd-normalize 2`
  default), so the game sees normalized last-token states — the critic
  re-normalizes per-coordinate anyway.
- **Adapter placement**: a GGUF is inference-only frozen ternary — the
  fork has no gradient path, so full LoRA-in-GGUF training is
  impossible. The trainable slider is a **torch-side residual head** on
  the frozen readout (`readout_residual_zero_init`), not in-attention
  q/k/v/o LoRA like the 0.6B path. Same game, explicitly weaker (and
  explicitly labeled) placement.
- **Frozen-readout cache**: D and G phases share one fetch per row per
  update (the GGUF never changes) to keep CPU smokes tractable.
- Prompts: 3 plain caption rows, same shape as the tiny card.
- Scope: no `state.pt` resume, no campaign/queue machinery — a game
  smoke target, not a release pipeline.

## How to run

CPU / CI (no weights, no fork, no server):

```bash
python3 conceptmod/textsliders/train_lora_bonsai_gguf.py --dummy \
  --steps 20 --name bonsai-gguf-particle --save_dir models/bonsai-gguf-particle
```

Live (needs ~7 GB disk for weights, ~8 GB RAM to serve CPU-only):

```bash
# 1. Fork binary (Linux CPU example; Bonsai-demo pins the known-good tag)
curl -sL -o fork.tar.gz https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10685-7dffb15/llama-prism-b10685-7dffb15-bin-ubuntu-x64.tar.gz
tar -xzf fork.tar.gz
# 2. Pinned weights ONLY (never F16, never mmproj for this path)
python3 -c "from conceptmod.textsliders.bonsai_gguf_backend import download_weights; print(download_weights('models/bonsai-gguf', allow_download=True))"
# 3. Serve the frozen readout (text-only: no --mmproj, ever)
export LD_LIBRARY_PATH=$PWD/llama-prism-b10685-7dffb15${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
./llama-prism-b10685-7dffb15/llama-server -m models/bonsai-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --host 127.0.0.1 --port 8080 --embedding --pooling last -c 512 -ngl 0
# 4. Train the head on the live readout
python3 conceptmod/textsliders/train_lora_bonsai_gguf.py \
  --weights models/bonsai-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --server_url http://127.0.0.1:8080 \
  --steps 20 --name bonsai-gguf-live --save_dir models/bonsai-gguf-live
```

Reload + report only:

```bash
python3 conceptmod/textsliders/train_lora_bonsai_gguf.py \
  --weights models/bonsai-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --server_url http://127.0.0.1:8080 \
  --steps 0 --load_head models/bonsai-gguf-live \
  --name bonsai-gguf-reload --save_dir models/bonsai-gguf-reload
```

Omit `--allow_download` and pass a local `--weights` snapshot to avoid
re-downloading. The trainer writes `{name}_last.json` (sidecar,
REFERENCE numbers echoed) plus EMA `{name}_last.safetensors`, live
`{name}_live_last.safetensors`, and a `report/readout_delta.json`
diagnostic grid at scales 0 / 0.5 / 1 (report-only; the game has no MSE).

## Verified live smoke (2026-09-18, this VM)

- Fork `prism-b10685-7dffb15` ubuntu-x64 CPU binary; PTQ1_0 verified at
  exactly 5,946,648,928 bytes, magic `GGUF`.
- Server: `llama-server -m ...PTQ1_0.gguf --embedding --pooling last
  -c 512 -ngl 0` → log shows `model loaded`, listening on 127.0.0.1.
  `POST /embedding {"content": ...}` returns `[{"index":0,
  "embedding":[[5120 floats]]}]` — dim 5120, finite, distinct prompts
  give distinct vectors. `/tokenize` returns real token ids.
- 2-step live train on the real readout (seed 7, `--no_report`):

```text
bonsai-gguf particle step 1: g_adv=0.6931 vic=0.0823 d_loss=0.6931 cos_pos=1.0000 sigma=0.9996
bonsai-gguf particle step 2: g_adv=0.6774 vic=0.1330 d_loss=0.7524 cos_pos=0.9976 sigma=0.9991
```

  (g_adv 0.6931 = ln 2 at init, as in the dummy path.) EMA + live
  exports written (360 KB each); `--steps 0 --load_head` reload plus a
  6-row readout-delta report at scales 0/1 confirmed scale-0 exactness
  (l2 = 0, cos = 1) against the live server.
- Cost note: ~22 s per embedding on 4 CPU threads — a 2-step smoke is
  minutes, a 20-step run is roughly an hour CPU-only. GPU (`-ngl 99`
  on CUDA/Metal) follows Bonsai-demo flags.

## What it is NOT

- Not Music 3, not YuE2, not H3, not Music Arm B, not the tiny 0.6B path
  (that path still works; nothing in it was touched).
- Not the locked recipe: it does not read `locked_shared` and does not
  change any live `--lm_target` default (Music 3 stays `v9` / `hidden`).
- Not a default: nothing imports this backend unless the Bonsai trainer
  (or its test) asks for it.
- Not in-attention LoRA and not a quality slider: the head steers a
  frozen readout; 2-step smokes prove the game runs, not that a concept
  transferred.
