# Paired abstention on retained Tiny Aya outputs

The Tiny Aya grounding benchmark includes missing-evidence, answerable, and
altered-evidence prompts. The paired audit checks the frozen dataset and public
prediction hashes, then compares each strength to OFF on identical prompts.
The two ON arms use the same RAW native adapter at different inference
strengths. The diagnostic reports both correct abstention and abstention on
answerable prompts. The release's intended-effect selector is untouched.

| Matched subgroup | OFF | Strength 1 | Strength 2 |
| --- | ---: | ---: | ---: |
| Strict UNKNOWN on missing evidence | 118 / 192 | 191 / 192 | 192 / 192 |
| Abstention on answerable or altered evidence | 48 / 384 | 302 / 384 | 384 / 384 |

The OFF versus OFF control gave zero change for both measures. The contrast
shows how one aggregate abstention rate could hide the increased failure to
answer prompts with evidence. It concerns a saved slider strength comparison.
A domain fine-tuning study would need independent, matched pre- and
post-training outputs. Conflict cases are outside the two rows above.

The [Tiny Aya audit implementation](https://github.com/HyperGAN/tiny-aya-particle-sliders/pull/15)
runs without a model call. Its tests verify strict scoring, matched prompts,
and partial coverage. Use its CLI on the retained
[release manifest](https://huggingface.co/datasets/ntc-ai/evidence-dial-benchmark/resolve/bbebf36c42294ea0814a397147b5809f1e846b04/curated-retry-v1/results.json),
[prediction files](https://huggingface.co/datasets/ntc-ai/evidence-dial-benchmark/tree/bbebf36c42294ea0814a397147b5809f1e846b04/curated-retry-v1/predictions),
and [frozen holdout](https://huggingface.co/datasets/ntc-ai/evidence-dial-grounding-tests/resolve/7e98c915ca74489ae0991fa96d25a5cf52b9cae4/balanced-grounding-v2/holdout.jsonl).
