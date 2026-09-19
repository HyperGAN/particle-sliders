# Choose a training duration before adding another loss

Update: the user explicitly chose to test the step limit alone next, and
clarified that the current lyric-hold plus step-limit 900 sounds cool and is
not broken. The order proposed below is superseded by that instruction. The
[active experiment](stepcap_only_plan_20260905.json) resumes the original 600
state with hold 0 and a step bound of 2, saving 750/900 for matched listening.
The subjective 600-versus-900 preference remains open.

Working recommendation: the original repaired b_cap smoke recipe at 600
updates. This follows the user's current listening preference; it is not a
claim that 600 is universally optimal. Keep the original complete 600 state
and its exported adapter as the reference. The experimental lyric hold plus
step limit has not demonstrated a consistent improvement over it.

The present metrics disagree: later bounded checkpoints improve training
hidden-target alignment and generator adversarial loss, while feature-matching
loss and cached-frame hidden drift rise. Ending-margin drift stays similar.
These diagnostics do not identify a preferred training duration. Exact values
and definitions are in [the metric audit](other_metrics_20260905.json).

## First decision: duration for the original recipe

Compare the existing original-recipe 450, 600 and 750 adapters. These use the
same training recipe; 450/600 are from the same run, and 750 continues its full
600 state. Exclude the known broken original 900 from preference trials while
retaining it as a failure control in numerical reports.

Use a fixed listening set: three arrangements/lyric sheets and generation
seeds 7, 23 and 101, with matching duration and slider +1. This is nine matched
examples per checkpoint, not nine independent training runs. Include the main
training prompt and two existing development prompts. Keep every first render;
reuse audio only where all fixture and reference hashes match.

Present blinded 450-versus-600 and 750-versus-600 pairs. Record intended voice
character and lyric intelligibility separately, plus ties or unwanted musical
changes. Report the individual pairs and outcomes by prompt. ASR can help
locate questionable words; it cannot decide the listener's preference.

Keep 600 unless another checkpoint has a consistent audible advantage without
more lyric problems. A tie keeps the current reference. A mixed result
means a tradeoff or unresolved sample dependence, not an excuse to choose the
lowest training loss. Do not set a universal stopping count from one seed.

## Second decision: stabilize the chosen recipe

After selecting a provisional duration, test only the actual parameter-step
bound, keeping lyric hold disabled and all original recipe settings fixed.
The existing bounded run changes both the hold and the update bound, so it
cannot establish the bound's independent effect. Use the same listening set
for development; reserve fresh prompts for confirming any selected recipe.

Use the large recorded training/KL excursions to identify recurrence of known
failure. A numerically stable candidate still needs to beat the listening
reference. Retain complete states and intermediate exports every 50 updates
when testing another continuation; save frequency is not a quality threshold.

This document is the next experiment design, not a completed benchmark. No
new training or rendering was launched during this metrics and strategy review.
Run any subsequent GPU work only on physical GPU 1 after checking occupancy.
