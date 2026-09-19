# Decision after listening

The current evidence supports ordinary energy 2.8 as the **best tested equal-pair
strength on the first song**. It does not establish the optimal strength, a solo
default, or a universal merge recipe. The completed merger comparison tests
whether changing update directions helps at that same per-projection weight norm.

The [decoded user rankings](listening-feedback.json) show:

- Female + pop: KnOTS-TIES beats ordinary on both songs. TIES takes first on
  confirmation, but the user found that ranking hard to distinguish.
- Country + indie rock: ordinary wins both songs. Its confirmation lead over
  TIES is close; both are well ahead of KnOTS-TIES there.
- House + acoustic folk: KnOTS-TIES wins the familiar song and ordinary wins
  confirmation. TIES is last on both.

Keep ordinary addition as the production default. KnOTS-TIES remains a serious
candidate for an optional mode, especially for female + pop; it has demonstrated
a listening benefit over ordinary on both examples of that pair. Two fixtures
do not establish a reliable rule for choosing methods from pair names. The
current TIES setting is lower priority: it finishes last four times and its
only first place is uncertain. This does not reject every possible TIES variant.

The next focused merger experiment should compare KnOTS-TIES against ordinary
on female + pop with additional seeds and unequal fader ratios. A nearby pruning
density can follow if that advantage persists. This is a proposed next test,
not an already launched run or a production option. Deployment should compute
or cache the selected update directly, rather than depend on this study's
large disk artifacts.

Separately, use the already-rendered energy-2 references to check whether the
earlier energy-2.8 preference survives the second song. The new user rankings
cover A/B/C merger conditions only; no reference-energy preference was supplied.
Strength calibration over unequal ratios and three-control blends remains a
separate question, as does any change to solo energy.

The [retrospective metric check](metric-agreement.md) does not support replacing
listening with a score. Content Enjoyment's 13/18 pairwise agreement is only one
comparison ahead of the fixed ordinary > KnOTS-TIES > TIES order, and it gets
4/9 energy-study comparisons correct. Complexity does better on the older
energy choices (8/9), with 12/18 merger agreement. Evaluate these frozen metrics
on new preferences before fitting a compound objective or choosing a merger
automatically. Static weight geometry cannot explain the house preference
reversal across the two song/seed conditions.

## A possible strength rule to investigate

This is a hypothesis derived from the geometry and first-song preferences,
not a production change. Let faders have nonnegative shares `p_i` summing to
one, with studio energy `E`. The current multipliers are `E * p_i`.
For equal-norm, orthogonal updates, the mixture norm then scales with
`sqrt(sum(p_i^2))`. Dividing multipliers by that quantity preserves the solo
norm: an equal pair receives a factor `sqrt(2)`, turning an energy-2 pair into
the equivalent of approximately energy 2.83 under today's convention.

This is close to the preferred 2.8 examples. It is not proof that such a rule
is right: these adapters have unequal norms, and weight overlap does not
measure semantic conflict. The first checks should include unequal ratios,
three controls, and conflicting voice controls. A smooth, bounded blend
compensation may be useful if those listening tests support it; a blanket
increase in solo strength does not follow from these pair results.

## Why this is a bounded screen

TIES and KnOTS-TIES directly test pruning/sign conflict handling and shared
subspace alignment with the existing weights. A random pruning family would
also require pruning seeds, and learned gates require a separate fitting
objective and validation set. Those become worthwhile if this screen reveals
a specific failure they could address. This study does not claim to have
tested those additional methods or exhausted the merge literature.
