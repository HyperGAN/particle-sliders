# Improving reward-ce-v1

The user prioritizes more consistent Content Enjoyment gains and has deprioritized
the full-song completion issue. The next quality experiment should use fixed
20-second comparisons. Preserve the original full-song result as a separate record.

## Evidence

- The original pilot student won 12/16 seed pairs, averaging +0.3998 CE across
  four families. Three losses came from one instrumental house family. Three of
  the four losses were smaller than 0.11 CE; the largest was -1.0710 CE.
- The [post hoc excerpt diagnostic](results.json) covers all 24 paired takes
  from twelve additional families. It wins 16/24, averaging +0.0145 CE, with a
  whole-family bootstrap interval of [-0.1957, +0.1864]. Nine family means are
  positive. The largest seed regression is -1.4882 CE on a female guitar-pop
  arrangement. This does not establish a positive average transfer gain.
- The training set contained eight independent families and four takes per family.
  Both CFG branches became separate rows, giving 64 rows, but not 64 independent
  musical examples. The selected activation direction's development rank agreement
  was 0.50, indicating limited evidence of predictive transfer.
- Training used the first 128 feedback positions (approximately 5.12 seconds at
  25 frames per second), while direction fitting and audio selection covered
  20 seconds. Longer saved histories exist in the original capture artifacts.
- Development CE at multiplier 1 was 7.8901 for step 600 and 7.6465 for step 660.
  Further constant-rate updates alone have no demonstrated advantage.
- The small composition study lost CE when the reward adapter was combined with
  the existing controls, including against the matched reduced-style control.

## Recommended sequence

1. Refine strength calibration on development examples. The original grid was
   0.1, 0.3 and 1.0; compare 0.5, 0.75, 1.0 and 1.25 for the frozen step-600
   adapter. Include no-style, solo-style and mixed-style cells. Hold existing
   effective style multipliers fixed for the isolated effect, and also assess
   the studio's fixed-energy contract with matched reduced-style controls.
   Respect the existing total-energy limit and record exact multipliers.

2. Train a v2 candidate with broader musical support. Expand to at least 24
   independent training families with balanced voices, instrumental arrangements,
   and multiple style combinations. Use new examples resembling the observed
   weaknesses, rather than memorizing their exact sheets. Refit and causally
   check the reward teacher before training the new rank-8/alpha-8 student.
   Sample training targets throughout the available 20-second histories while
   retaining every target's exact preceding feedback and prompt. Compare this
   change under a frozen recipe; it is a hypothesis, not a proven improvement.

3. Declare development selection around consistency: use paired win rate,
   positive mean CE gain and the magnitude of regressions, then freeze the
   selected checkpoint and effective strength. Compare Off, v1 and the candidate
   on new held-out families and seeds. Both existing pilot and transfer examples
   are now diagnosis material. Improvement on those same examples is not fresh
   confirmation. Continue to report voice, lyric, style and diversity diagnostics
   separately from CE; natural completion is not the primary gate for this
   fixed-duration experiment.

This document recommends the next experiment. No new training, checkpoint change,
strength sweep or production deployment has been started by this diagnostic.
