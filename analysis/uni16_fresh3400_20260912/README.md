# All 16 sliders, four prompt pairs, with remaining budgets capped at 2000

Live progress and milestone audio:
<http://100.90.104.57:8888/uni16-fresh3400-20260912/>

## September 15 GPU queue correction

The fixed GPU 1 queue finished while Acoustic Folk, Disco Funk and Reggaeton
remained in GPU 0's queue. GPU 0 was waiting for a game process to release
memory. The user reaffirmed using both GPUs and stopped that process.
`gpu-routing.json` now places Disco Funk on GPU 1; Acoustic Folk resumes from
1000 on GPU 0, followed by Reggaeton. `gpu_routing.py` applies this placement
in the budget wrapper without modifying the frozen catalog, training manifest
or checkpoint signatures. GPU locks and the existing memory check remain in
effect. Queue and placement tests passed before the service restarted.

## September 14 budget decision

The user approved stopping the eight unfinished sliders at **2000 updates**
to compare quality before spending more compute. Metal and Country resume
their current states; Acoustic Folk, House, Disco Funk, K-pop, Reggaeton and
Afrobeats follow in the two queues. Female, Lo-fi, Male, Pop, Hip-Hop, R&B,
Indie Rock and Pop Punk already completed 3400 updates and retain every sample.
The current combined budget is **43,200 updates**.

`budget.json` records the explicit per-slider targets and the user's approval.
The service now runs `budget_campaign.py`, which reuses the original queue,
GPU locks, warmup, trainer and renderer with a shorter list of milestones.
At 2000 it verifies the export and pinned full state, completes the sample
health check, and marks the current budget complete. It never automatically
extends those eight jobs. The full optimizer, critic, sampler and RNG state
supports a later explicitly chosen extension.

The original training manifest and signed source files remain immutable so
existing checkpoints resume with their original signatures. Their internal
3400 horizon and directory names describe that original recipe. The active
stopping budget is separate and is shown per slider on the progress page.
Saved evidence includes the jobs before the budget change, the graceful-stop
states and the new runner hashes. The service must use `budget_campaign.py`;
directly launching the historical `campaign.py` would use the old 3400 plan.

## Original training protocol

The user chose all four existing prompt pairs for every current studio slider,
authorized both graphics cards, and requested samples and health checks every
1000 updates. The intermediate idea of a one-row/four-row pilot was superseded
before any training launched. The full campaign runs 16 fresh retrains, split
into eight jobs on each physical A6000. GPU 0 begins with Female and GPU 1 with
Lo-fi. The existing studio and Hub exports are not replaced by this runner.

Each job starts with a fresh rank-8, alpha-8 LM attention LoRA and its own
critic. The first 600 updates reproduce the original four-row warm-up recipe.
Its full state then migrates into the baseline fresh-continuation trainer for
updates 601 through 3400. Generator/critic rates stay 0.0005/0.00075;
adversarial, feature-matching and ending losses retain weight 1. The critic,
both optimizers and RNG are carried over. EMA, LR decay, normalized features,
extra policy/hidden guards and effective-step limiting remain off. The
continuation's parameter-step bound remains 2.

During continuation one row is sampled per update in shuffled, balanced
passes over all four rows: 700 fresh updates per row, 2800 per slider. This
keeps the one-row minibatch of the successful fresh-history experiment while
broadening the contexts. Each update samples a fresh frozen-base continuation
with the adapter disabled. Seeds are unique across all 44,800 continuation
updates. Duplicate continuation tensors within a trajectory are rejected
without substituting a seed. Prompt-state style teachers remain fixed per row;
fresh histories change ending supervision. This does not claim fresh sampled
style-distribution training.

The 3400 endpoint follows the user's Lo-fi listening preference, which gave
that checkpoint 5/5 for sound, enjoyment and words on all four reused cases.
This is a fixed budget for the new campaign, not proof of convergence or an
optimal checkpoint for all genres. The four-row continuation is an explicit
data change relative to that one-row Lo-fi trajectory.

## Samples and health checks

At **1000, 2000, 3000 and 3400**, up to each slider's approved budget,
each GPU releases the trainer, renders the
checkpoint, checks audio health, and resumes the exact saved training state.
Every checkpoint uses the same two held-out arrangements and seeds 1709/2903,
20 seconds at strength 1. Adapter-off, positive-caption, published-660 and new
warm-up-600 controls are rendered once and reused with identity and hash checks.
No seed is bumped; short and silent results remain in the evidence.

Non-finite training state stops the affected run before exporting. Any new
candidate with RMS below 0.0001, at least 1% near-clipped samples, or a complete
four-case set with every clip under 4 seconds stops that slider for inspection.
All previous checkpoints and samples remain. Other sliders continue. A short
natural ending alone is a diagnostic flag, not an automatic stop. Preference
scores do not stop training. Musical diversity, repetition, pitch and lyric
quality still require listening; the page does not call these “no collapse.”

## Persistence

Continuation recovery state saves every 20 updates. Full milestone states,
generator exports, hashes, finiteness checks, row counts, and complete sampling
history are retained. Warm-up saves every 50 updates and resumes into a new
attempt directory to preserve earlier logs. Both services are enabled user
services; user lingering is enabled on this machine. Unexpected service failure
restarts the runner. Intentional shutdown lets continuation finish its current
update and save; warm-up resumes its latest 50-update recovery point.

```bash
systemctl --user status music-uni16-fresh3400.service
systemctl --user status music-uni16-fresh3400-page.service
journalctl --user -u music-uni16-fresh3400.service -n 30 --no-pager
```

`jobs/<slider>.json` records each queue job and its current log. Fresh training
state and exports live under `models/uni16-fresh3400-v1/<slider>/`. Samples and
health reports live on the listening page and under `samples/` here. Resuming a
health-stopped or numerically failed slider requires reviewing its retained
evidence first; restarting the campaign deliberately retries incomplete jobs.

The campaign holds `.music-gpu-0.lock` or `.music-gpu-1.lock` per model process
and waits if another workload occupies the card. The Lo-fi listening-test
renderer shares the GPU 1 lock, so newly requested comparison renders wait for
a free interval. The progress publisher is independent and cannot interrupt
training. Training source and prompt hashes are frozen; page presentation is
outside the training signature.

The original all-3400 plan was estimated at four to five days on both GPUs.
At the September 14 budget change, the remaining capped jobs were estimated
at roughly 23–24 hours, saving approximately one day of both GPUs' time.
The page reports observed updates rather than presenting estimates as deadlines.
