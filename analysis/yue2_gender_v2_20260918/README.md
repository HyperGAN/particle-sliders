# Gender v2 prompts (2026-09-18)

Male/female one-word positives produced last-token edits only 5% of the neutral seed spread
(pop 14%, metal 29%); the paired gmix game could not be balanced at any sigma. These positives
add gender cues in Global Metadata (mix sentence) and throughout Vocal Details. Same neutrals,
lyrics, seeds and recipe as yue2_gmix_catalog_20260918. Launched via transient unit
`yue2-female-gmix-noise1600` so the catalog controller keeps off GPU 1. Driver: run_gender_v2.sh.

## Hold ratio follow-up
v2 prompts at the fixed hold of 1.0 (sigma/residual ~1.9) eroded: male d_loss 0.45→0.18, G grad 5→15 by step 700.
Added REFERENCE.noise_hold_ratio=1.3 (hold = edit_rms*1.3, per run: male 1.97, female 1.84, pop would be 1.56).
male_h13/female_h13 runs use it (unit yue2-gender-h13, run_gender_h13.sh). female v2 at hold 1.0 kept running as the comparison; male v2 stopped at ~step 720.

## Outcome (16:40)
| run | hold | steps | final residual | gain | cos_pos | last-50 d_loss | G grad |
|---|---|---|---|---|---|---|---|
| male v1 (one-word) | 1.0 | collapsed @575 | 1.31 | 0.68 | 0.01 | 0.01 | 15 |
| male v2 | 1.0 | stopped @719 | 0.51 | 0.90 | 0.91 | 0.17 | 14.5 |
| female v2 | 1.0 | 1600 | 0.47 | 0.91 | 0.96 | 0.22 | 12 |
| male_h13 | 1.97 | 1600 | 0.47 | 0.88 | 0.92 | 0.37 | 10 |
| female_h13 | 1.84 | 1600 | 0.48 | 0.91 | 0.95 | 0.45 | 9 |
Prompt rewrite is what made gender trainable (edit/neutral-spread 0.05 -> 0.10). The hold ratio keeps the critic
balanced (d_loss 0.37-0.45 vs 0.17-0.22) at equal residual. Probe best steps: male_h13 1600, female_h13 1500,
female v2 1200. All probes flag template t2 (keyboard-led, BPM 104): a few seeds (350851312, 1505790826,
2031050608) sit at residual 1-3 in every run, so t2 is the prompt to look at next, not the schedule.

Kept: male h13 and female h13. Male v2 is not kept. Female v2 was a tie on the full traces; h13 is the one we are keeping. Copies in `models/gender-yue2-gmix-h13-20260918/`.
