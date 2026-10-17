# Private substantive caption-backend qualification

The existing Qwen3-0.6B-Base and Ternary-Bonsai-2-27B PTQ1_0 backends are preserved. The original three garden, sea and rainy-street caption pairs stay unchanged. Native shared `particle_bridge_gan` is reused: absolute-positive-target whitening, MLP critic, G/D/particle rates .0006/.0009/.006, Adam (0,.999), batch64, 128x4 cloud, VIC, lazy b_cap every4, EMA.995, and the existing noise horizon8000. These campaigns do not switch to BISPF, paired-edit whitening, a new model, or a different student input.

## Budget and scheduling

The repository's default20 updates are smoke only. This isolated qualification defines **1200 native optimizer updates** as a substantive experiment: 300 lazy regularizer applications and six EMA time constants. This is a new explicit run budget, not a previously established production recipe. The noise schedule remains8000, including its existing native hold calculation. It is never compressed to the run length.

The closed sequence is100 updates, a measured projection gate, then exact continuation to1200. Checkpoints persist every20 updates. Heldout controls run at0,20,100,400,800,1200. The Tiny CPU100-update stage has a15-minute cap; its full continuation has a90-minute cap. The Bonsai frozen-cache preparation has a30-minute cap, its100-update stage10 minutes and its1200 continuation30 minutes. No automatic retry occurs. A failed projection, finite-state, recovery, cache, resource or replay gate blocks continuation.

All workers use CPU only, one Torch thread, nice15 and idle I/O priority. The own Bonsai CPU server alone uses four threads and zero GPU layers. Aggregate worker/server RSS stays below8GiB, host available memory above8GiB, and NVMe free space above100GiB. Existing GPU services and all training reservations are untouched.

The earlier480-second Bonsai20-update projection cap and600-second process timeout came from an assistant-authored provisional smoke runner. They were scheduling choices, not externally imposed safety limits. The parent explicitly authorized a bounded30-minute smoke allowance. `run_bonsai_20_authorized.py` records this provenance and preserves the original memory guards. The uncached native20-update smoke must pass before Bonsai substantive preparation begins.

## Recovery and fixed features

The optional observer preserves network/particles, critic parameters and normalization/attributes, both Adam optimizers (including the separate particle group), EMA, independent sampler RNG streams, Torch CPU/CUDA, Python and NumPy RNG. Target-bank fingerprints, recipe identity and checkpoint SHA256 must match. Files commit immutably at optimizer+EMA boundaries. Native exports record actual completed steps separately from planned steps. Split8 vs4+4 tests cover both lazy regularizer steps4 and8, ordinary unobserved parity, real tiny Qwen3 and the Bonsai head.

Bonsai's GGUF is frozen and has no attention-gradient path. An opt-in immutable cache avoids repeated identical27B forwards. Every prompt is fetched twice from the exact official CPU fork; both vectors must match bit for bit before caching. Keys include verified model SHA, runtime archive/commit, last-token pooling, normalization2, context512 and four-thread serving identity. Tensor SHA, shape, dtype and finite values are checked on every replay. Cached and uncached native dummy games produce identical full state; live preparation separately requires exact official vectors. Any nonrepeatability blocks the cache; it never silently enables approximate reuse. The cache covers only fixed training/evaluation prompts. No altered model, quantization band, embedding normalization or training goal is substituted.

## Quality gates and limitations

The native implementations feed **positive-caption inputs** to the student and compare with the same frozen-positive teacher. Thus zero adapter output already matches the noiseless training target. We preserve this behavior and explicitly test whether useful neutral-caption transfer emerges. A nonzero adapter or completed optimizer budget is insufficient evidence.

Three predeclared heldout caption pairs test garden/sea/street generalization. Scale0 must exactly match the base; native EMA exports must reload with exact tensors and readouts. Scale+1 must reduce relative target-edit MSE to at most.9 with cosine at least.25 on at least two of three pairs, with mean MSE at most.9. Unrelated-canary edit RMS must be at most.25 of the desired teacher-edit RMS; training-positive identity relative MSE must stay at most.01. Controls0/.5/1 and all metrics are retained. These thresholds are fixed before substantive training and are not tuned after observing results.

Tiny also retains deterministic32-token completion controls at each main milestone, for human review of relevance, leakage and repetition. Passing numeric readout gates does not establish generation quality. Bonsai trains a residual **readout head** only; it cannot steer the GGUF's token generation. Its quality claim is restricted to measured readout transfer. Neither backend is marked quality-qualified automatically. Full1200 runs with failed heldout gates retain their artifacts and are reported as failed qualification; no objective replacement or automatic retraining follows.
