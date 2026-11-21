"""Audit retained Vibe scores and best-of-N receipts without a judge or model."""

import argparse
import json
import math
from pathlib import Path

from . import SCHEMA, digest, file_digest, prompt_pair, read_json, score_value


def _summary(rows):
    if not rows:
        return {'pairs': 0, 'mean_gain': None}
    return {'pairs': len(rows), 'mean_gain': sum(row['gain'] for row in rows) / len(rows)}


def audit_collection(state, *, verify_media=False):
    """Reconcile saved scores with ranking and curation, with no reranking."""
    identity = state['identity']
    if (identity.get('schema') != SCHEMA or identity.get('selection') != 'raw'
            or identity.get('metric') != 'equal-case-mean-positive-minus-neutral'):
        raise ValueError('expected a Vibe RAW matched-effect collection')
    best_of, pool_size = identity['best_of'], identity['pool_size']
    if (type(best_of) is not int or best_of < 1 or type(pool_size) is not int or pool_size < 1
            or type(identity['seed']) is not int or identity['seed'] < 0):
        raise ValueError('invalid Vibe frozen cohort sizes or seed')
    all_cases, selected, rounds = [], [], []
    checked_media = {}

    for index, round_ in enumerate(state['rounds']):
        ranking = round_['ranking']
        if round_['index'] != index or len(ranking) != pool_size:
            raise ValueError('round does not match the frozen Vibe cohort')
        prompt_keys = set()
        fitnesses = []
        expected_seeds = list(range(identity['seed'] + index * best_of,
                                    identity['seed'] + (index + 1) * best_of))
        for row in ranking:
            prompt = prompt_pair(row['prompt'])
            key = prompt['neutral'], prompt['positive']
            if key in prompt_keys:
                raise ValueError('duplicate candidate prompt pair in round')
            prompt_keys.add(key)
            cases = row['cases']
            if [case['seed'] for case in cases] != expected_seeds:
                raise ValueError('candidate does not cover exactly the frozen matched seeds')

            gains = []
            for case in cases:
                scores = []
                for side in ('neutral', 'positive'):
                    receipt = case[side]
                    output = receipt['output']
                    if (('seed' in output and output['seed'] != case['seed'])
                            or ('prompt' in output and output['prompt'] != prompt[side])):
                        raise ValueError('output does not bind the case seed and prompt')
                    if verify_media:
                        path = output['path']
                        if path not in checked_media:
                            checked_media[path] = file_digest(path)
                        if checked_media[path] != output.get('sha256'):
                            raise ValueError('output media differs from its saved hash')
                    scores.append(score_value(receipt['judgment'])['score'])
                gain = scores[1] - scores[0]
                if gain != case['gain']:
                    raise ValueError('saved case gain differs from the matched effect scores')
                gains.append(gain)
            fitness = row['fitness']
            winner = max(range(best_of), key=gains.__getitem__)
            # Python versions differ in their float summation. Verify the
            # arithmetic within rounding error, and keep the saved tie/order.
            if (type(fitness) not in (int, float)
                    or not math.isclose(fitness, math.fsum(gains) / best_of, rel_tol=0, abs_tol=1e-15)
                    or row['best'] != winner):
                raise ValueError('saved fitness or best-of-N index differs from measured effect')
            fitnesses.append(fitness)
            all_cases.extend(cases)
            case = cases[winner]
            selected.append(dict(id=digest(dict(round=index, prompt=prompt)), round=index,
                                 prompt=prompt, seed=case['seed'], fitness=fitness, gain=case['gain'],
                                 neutral=case['neutral'], positive=case['positive']))

        if fitnesses != sorted(fitnesses, reverse=True):
            raise ValueError('saved ranking is out of descending effect order')
        rounds.append({'round': index, 'candidates': len(fitnesses),
                       'top_effect_gain': fitnesses[0], 'tied_top_candidates': fitnesses.count(fitnesses[0])})

    if state['selected'] != selected:
        raise ValueError('saved selected contexts differ from the round evidence')
    return {'schema': 'particle-slider-vibe-receipt-audit-v1', 'collection_sha256': digest(state),
            'rounds': rounds, 'all_matched_cases': _summary(all_cases),
            'best_of_n_contexts': _summary(selected), 'media_files_checked': len(checked_media),
            'scope': 'descriptive curation audit of saved intended-effect scores'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path, help='retained collection.json')
    parser.add_argument('--verify-media', action='store_true', help='also verify the original output files')
    args = parser.parse_args(argv)
    try:
        report = audit_collection(read_json(args.collection), verify_media=args.verify_media)
    except (OSError, TypeError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
