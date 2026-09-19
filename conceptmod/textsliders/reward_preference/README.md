# Reward preference continuation

The user asked to stop spending renders on a weak 9/16 strength calibration and
continue training for more consistent wins. This isolated experiment starts with
the actual step-600 rank-8/alpha-8 LoRA, scaled to the calibrated half strength.
It changes the optimizer and objective explicitly; the unsuccessful V2 activation
teacher is not used and the frozen GAN experiments remain untouched.

`setup.py` selects the highest and lowest scored Off take within each of the 24
training families already captured. All 48 histories, scores, source files and
weights are bound by hashes. `replay.py` uses their saved sampling state and exact
feedback to recover semantic tokens. Every reconstructed feedback embedding must
match bit for bit. It neither inverts WAVs nor regenerates training audio. A
failure stops the run; approximate token guesses are never accepted.

`train.py` applies a preference loss to mean semantic log probabilities relative
to the initial adapter, with both CFG branches, reference KL and prompt anchors.
The first 500 emitted frames retain their full causal prefix. The discarded warmup
token is excluded. This is a music-specific surrogate inspired by
[Direct Preference Optimization](https://arxiv.org/abs/2305.18290); it is not the
full audio likelihood. Sampling top-k is omitted during training for continuous
support, and residual-code/acoustic likelihoods are not optimized. CE provides
pair labels, not a differentiable audio reward or token-aligned local rewards.

Two learning rates, 1e-5 and 3e-5, train for 30 and 60 updates with fresh AdamW
optimizers. Complete adapter/optimizer/sampler/RNG states allow bounded
continuation. Step 120 is allowed only for an arm whose step-60 four-comparison
screen wins at least three, improves mean CE by .02, and loses no more than .30
CE on any comparison. Select by win count, worst delta, then mean. The incumbent
remains eligible. A better promising candidate gets four fresh paired comparisons
(eight clips); no broad automatic evaluation follows a weak screen. At most 36
new clips are authorized by this fixed local protocol, including four incumbent
references. Original Off control audio is reused on the same physical GPU.

`campaign.py` owns child processes on both GPUs and pauses studio generation only
after its queue drains. Playback stays available. It restores GPU 0 to one studio
worker on completion/error. It exports normal 432-tensor LoRA files and records
results; it does not change the production registry or publish weights externally.

Run in the existing minimax-music3 environment:

```bash
python -m conceptmod.textsliders.reward_preference.campaign
```

Unit checks live in `tests/test_reward_preference.py`. Live progress is in
`analysis/reward_preference_20260908/README.md` and the corresponding user service.
