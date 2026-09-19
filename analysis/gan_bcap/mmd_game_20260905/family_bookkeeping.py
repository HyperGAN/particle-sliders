"""Bookkeeping helpers for the user's explicitly authorized non-MMD experiments.

These labels and round outcomes never alter the locked audio judge or its scores.
"""
RESEARCH_FAMILIES = ('mmd', 'gan', 'distillation', 'calibration', 'hybrid')
ALL_FAMILIES = (*RESEARCH_FAMILIES, 'reference')


def validate_budget(family, budget):
    if family not in RESEARCH_FAMILIES:
        raise ValueError('Unknown research method family')
    if budget < 0 or (budget == 0 and family not in ('calibration', 'hybrid')):
        raise ValueError('A training family needs a positive update budget; calibration or hybrid construction may use zero')


def round_result(entries, round_name, initial_best_scores, current_champions):
    candidates = [entry for entry in entries.values()
                  if entry['round'] == round_name and entry['family'] in RESEARCH_FAMILIES]
    eligible = [entry for entry in candidates if entry['eligible']]
    best = max(eligible, key=lambda entry: entry['score']) if eligible else None
    mmd = [entry for entry in eligible if entry['family'] == 'mmd']
    best_mmd = max(mmd, key=lambda entry: entry['score']) if mmd else None

    def gain(entry, name):
        previous = initial_best_scores[name]
        return entry['score']-previous if entry is not None and previous is not None else None

    overall_gain = gain(best, 'overall')
    mmd_gain = gain(best_mmd, 'mmd')
    if not candidates:
        outcome = 'failed experiment'
    elif best is None:
        outcome = 'ineligible candidate'
    elif current_champions['overall'] == best['id'] and (overall_gain is None or overall_gain > 1e-9):
        outcome = 'new overall lead'
    elif best_mmd and (mmd_gain is None or mmd_gain > 1e-9):
        outcome = 'new MMD personal best'
    else:
        outcome = 'no new record'
    return dict(candidates=[entry['id'] for entry in candidates], best_candidate=best['id'] if best else None,
        outcome=outcome, gain_vs_previous_mmd=mmd_gain, gain_vs_previous_overall=overall_gain)
