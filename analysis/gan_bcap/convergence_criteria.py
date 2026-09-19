"""Explicit, conservative stopping decisions for checkpoint comparisons.

Statistical intervals describe a fixed metric on representative independent
prompt groups. They do not validate that metric as human musical judgment.
The incumbent and comparison budget must be locked before confirmation data.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import statistics
import sys

# Isolated analysis dependency; the training environment is unchanged.
sys.path.append(str(Path(__file__).resolve().parents[3]/'.cache/gan-convergence/python'))
from scipy.stats import t


def paired_gain_bound(challenger, incumbent, groups, *, minimum_groups=8,
                      minimum_seeds=4, worthwhile_gain=.05, family_alpha=.05,
                      planned_comparisons=12):
    """One-sided t interval over prompt means, with Bonferroni allocation.

    Seeds within a prompt are averaged first, not counted as independent
    prompts. This is an approximate interval under prompt-level sampling
    assumptions, not a distribution-free or indefinitely sequential bound.
    """
    if (not challenger or challenger.keys() != incumbent.keys()
            or challenger.keys() != groups.keys()):
        raise ValueError('Need the same unique matched fixtures and group labels')
    if minimum_groups < 2 or minimum_seeds < 1 or planned_comparisons < 1:
        raise ValueError('Invalid declared evaluation budget')
    if not 0 < family_alpha < 1 or worthwhile_gain < 0:
        raise ValueError('Invalid confidence or worthwhile-gain setting')
    grouped = defaultdict(list)
    for key in challenger:
        a, b = challenger[key], incumbent[key]
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Nonfinite score')
        grouped[groups[key]].append(a-b)
    values = [statistics.mean(v) for v in grouped.values()]
    result = dict(prompt_groups=len(values), samples=len(challenger),
        minimum_seeds_observed=min(map(len, grouped.values())),
        mean_gain=statistics.mean(values), worthwhile_gain=worthwhile_gain,
        alpha_per_comparison=family_alpha/planned_comparisons,
        method='Prompt-mean paired t upper bound with predeclared Bonferroni allocation',
        upper_gain_bound=None, lower_gain_bound=None, decision='evaluate_more')
    if len(values) < minimum_groups or result['minimum_seeds_observed'] < minimum_seeds:
        result['reason']='Too few prompt groups or matched seeds'
        return result
    standard_error = statistics.stdev(values)/math.sqrt(len(values))
    critical = float(t.ppf(1-family_alpha/planned_comparisons, len(values)-1))
    result.update(upper_gain_bound=result['mean_gain']+critical*standard_error,
                  lower_gain_bound=result['mean_gain']-critical*standard_error,
                  standard_error=standard_error)
    if result['upper_gain_bound'] <= worthwhile_gain:
        result.update(decision='no_worthwhile_gain', reason='Upper gain bound is within tolerance')
    elif result['lower_gain_bound'] > worthwhile_gain:
        result.update(decision='worthwhile_gain', reason='Lower gain bound exceeds tolerance')
    else:
        result['reason']='Interval still includes both a worthwhile gain and no worthwhile gain'
    return result


def stopping_decision(comparisons, *, quality, unstable=False, budget_exhausted=False,
                      patience=3, independent_confirmation=False):
    """Keep scientific and operational outcomes distinct; never infer quality."""
    if patience < 1:
        raise ValueError('Patience must be positive')
    if quality not in ['pass', 'fail', 'unvalidated']:
        raise ValueError('Quality must be explicitly supplied by calibrated checks')
    if unstable:
        return 'stop_unstable_keep_previous'
    plateau = (len(comparisons) >= patience and
               all(r['decision']=='no_worthwhile_gain' for r in comparisons[-patience:]))
    if plateau and quality == 'fail':
        return 'stalled_change_recipe'
    if plateau and quality == 'pass' and independent_confirmation:
        return 'stop_keep_best'
    if budget_exhausted:
        return 'budget_stopped_not_converged'
    if quality == 'unvalidated':
        return 'validate_judge_and_quality'
    if plateau:
        return 'run_reserved_confirmation'
    if comparisons and comparisons[-1]['decision']=='worthwhile_gain':
        return 'continue_training'
    if not comparisons or comparisons[-1]['decision']=='evaluate_more':
        return 'evaluate_more'
    return 'continue_until_patience'


def game_evidence(generator_probes, discriminator_probes, *, g_tolerance=.05,
                  d_tolerance=.02):
    """Large observed unilateral gain disproves the working stationarity tolerance.

    Failed/weak optimization cannot certify stationarity, even when all gains
    are small. This is deliberately asymmetric.
    """
    if not generator_probes or not discriminator_probes:
        return 'insufficient_probes'
    if (max(p['fractional_improvement'] for p in generator_probes) > g_tolerance or
            max(p['available_improvement'] for p in discriminator_probes) > d_tolerance):
        return 'not_stationary_under_tested_tolerances'
    return 'small_observed_gain_requires_stronger_probes'
