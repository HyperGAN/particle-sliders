from copy import deepcopy
import json
import math

import pytest

from particle_sliders.parameter_search.metric import (
    algebraic_controls, canonical, decoded_chunks, digest, score, score_chunks, validate_grid, write_raw,
)


def data(gain=0.):
    grid = [dict(example=dict(scene=scene, seed=seed, path='neutral'), positions=[
        dict(position=dict(surface='image', t=0), shape=[1, 2]),
        dict(position=dict(surface='coupled_audio', t=1), shape=[1, 2]),
    ]) for scene, seed in [('room', 11), ('outdoors', 12)]]
    chunks = []
    for example in grid:
        for i, position in enumerate(example['positions']):
            target = [2., -1.] if i == 0 else [.001, 0.]
            chunks.append(dict(example=example['example'], **position, offset=0,
                neutral=[0., 0.], target=target, student=[gain * t for t in target]))
    return grid, chunks


@pytest.mark.parametrize('gain,expected', [(0., 1.), (1., 0.), (.5, .25), (-1., 4.), (2., 1.)])
def test_full_coordinate_controls_both_surfaces_and_chunking(tmp_path, gain, expected):
    grid, chunks = data(gain)
    path = tmp_path / 'responses.jsonl'
    write_raw(path, {'selection': 'raw'}, chunks)
    result = score(path, grid, {'selection': 'raw'})
    assert result['error'] == pytest.approx(expected, abs=1e-14)
    assert result['signed_edit_gain'] == pytest.approx(gain)
    assert set(result['per_scene']) == {'room', 'outdoors'}
    assert all(row['elements'] == 4 for row in result['examples'])
    split = []
    for row in chunks:
        for i in range(2):
            split.append(dict(row, offset=i, **{key: [row[key][i]] for key in ('student', 'target', 'neutral')}))
    result2 = score_chunks(split, grid, reference_sha256=result['reference_sha256'])
    assert result2 == result
    assert list(decoded_chunks(path)) == chunks


def test_pool_before_dividing_and_equal_examples():
    grid, chunks = data(1.)
    chunks[1]['student'] = [1., 0.]  # weak late edit, a large late student error
    chunks[3]['student'] = [0., 0.]
    result = score_chunks(chunks, grid)
    expected = (((1 - .001) ** 2 / (5 + .001 ** 2)) + .001 ** 2 / (5 + .001 ** 2)) / 2
    assert result['error'] == pytest.approx(expected)
    assert result['error'] < .1  # averaging per-position ratios would be enormous
    chunks[1]['student'][1] = 3.  # orthogonal coupled-audio corruption MUST count
    assert score_chunks(chunks, grid)['error'] > .9


def test_zero_position_valid_whole_example_degenerate_rejected():
    grid, chunks = data()
    chunks[1]['target'] = [0., 0.]
    assert score_chunks(chunks, grid)['error'] == 1
    chunks[0]['target'] = [0., 0.]
    with pytest.raises(ValueError, match='degenerate'):
        score_chunks(chunks, grid)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'duplicate', 'wrong_label', 'wrong_shape', 'wrong_offset', 'wrong_count', 'nonfinite', 'nonnumeric'])
def test_rejects_wrong_or_incomplete_native_grid(mutation):
    grid, chunks = data()
    if mutation == 'missing': chunks.pop()
    if mutation == 'extra': chunks.append(deepcopy(chunks[-1]))
    if mutation == 'duplicate': chunks[1] = deepcopy(chunks[0])
    if mutation == 'wrong_label': chunks[0]['example'] = dict(scene='other', seed=11, path='neutral')
    if mutation == 'wrong_shape': chunks[0]['shape'] = [2, 1]
    if mutation == 'wrong_offset': chunks[0]['offset'] = 1
    if mutation == 'wrong_count': chunks[0]['student'].append(2.)
    if mutation == 'nonfinite': chunks[0]['student'][0] = math.nan
    if mutation == 'nonnumeric': chunks[0]['student'][0] = True
    with pytest.raises(ValueError):
        score_chunks(chunks, grid)


def test_header_frozen_reference_binary_and_algebraic_controls(tmp_path):
    grid, chunks = data(.7)
    path = tmp_path / 'responses.jsonl'
    write_raw(path, {'selection': 'raw', 'step': 12}, chunks)
    result = score(path, grid, {'selection': 'raw', 'step': 12})
    for binding in ({'selection': 'ema', 'step': 12}, {'selection': 'raw', 'step': 11}):
        with pytest.raises(ValueError, match='binding'):
            score(path, grid, binding)
    with pytest.raises(ValueError, match='reference'):
        score(path, grid, {'selection': 'raw', 'step': 12}, reference_sha256='wrong')
    control = algebraic_controls(path, grid, reference_sha256=result['reference_sha256'])
    assert control['observed'] == pytest.approx(control['expected'])
    assert 'not trained/native OFF' in control['evidence']
    binary = path.with_suffix('.bin')
    binary.write_bytes(binary.read_bytes()[:-8])
    with pytest.raises(ValueError, match='truncated'):
        score(path, grid, {'selection': 'raw', 'step': 12})


def test_duplicate_grid_and_unit_contract():
    grid, chunks = data(.5)
    duplicate = deepcopy(grid)
    duplicate.append(deepcopy(duplicate[0]))
    with pytest.raises(ValueError, match='duplicate'):
        validate_grid(duplicate)
    expected = score_chunks(chunks, grid)['error']
    for row in chunks:
        for key in ('student', 'target', 'neutral'):
            row[key] = [10 * v for v in row[key]]
    assert score_chunks(chunks, grid)['error'] == pytest.approx(expected)


@pytest.mark.parametrize('mutation', ['seed_float_alias', 'shape_float_alias', 'bool_offset', 'float_offset'])
def test_exact_labels_keep_numeric_types(mutation):
    grid, chunks = data()
    row = chunks[0]
    if mutation == 'seed_float_alias': row['example'] = dict(row['example'], seed=11.)
    if mutation == 'shape_float_alias': row['shape'] = [1., 2.]
    if mutation == 'bool_offset': row['offset'] = False
    if mutation == 'float_offset': row['offset'] = 0.
    with pytest.raises(ValueError, match='labels'):
        score_chunks(chunks, grid)


@pytest.mark.parametrize('mutation', ['tiny_denominator', 'pooled_overflow', 'huge_integer'])
def test_nonfinite_statistics_are_explicit_unscoreable_errors(mutation):
    grid, chunks = data()
    if mutation == 'tiny_denominator':
        for row in chunks:
            row['target'] = [1e-160, 0.]
            row['student'] = [1e20, 0.]
    elif mutation == 'pooled_overflow':
        chunks[0]['student'] = [1e154, 1e154]  # individual squares finite; pooled SSE overflows
    else:
        chunks[0]['student'][0] = 10 ** 1000
    with pytest.raises(ValueError, match='nonfinite'):
        score_chunks(chunks, grid)


def test_examples_have_equal_weight_even_with_different_teacher_energies():
    grid, chunks = data()
    for row in chunks[2:]:
        row['target'] = [t * 1000 for t in row['target']]
        row['student'] = list(row['target'])
    assert score_chunks(chunks, grid)['error'] == pytest.approx(.5)


def test_transport_chunks_do_not_round_pooled_statistics_twice():
    # fsum([1e16, 1, 1]) differs from fsum([fsum([1e16, 1]), 1]).
    grid = [dict(example={'scene': 'rounding', 'seed': 1}, positions=[
        dict(position={'surface': 'vector'}, shape=[3])])]
    row = dict(example=grid[0]['example'], **grid[0]['positions'][0], offset=0,
               student=[1e8, 1., 1.], target=[0., 0., 0.], neutral=[-1., -1., -1.])
    one = score_chunks([row], grid)
    for sizes in ([1, 1, 1], [2, 1], [1, 2]):
        chunks, offset = [], 0
        for size in sizes:
            chunks.append(dict(row, offset=offset, **{
                key: row[key][offset:offset + size] for key in ('student', 'target', 'neutral')}))
            offset += size
        assert score_chunks(chunks, grid) == one
    assert one['examples'][0]['student_teacher_sse'] == math.fsum([1e16, 1., 1.])


def test_overflowing_preservation_diagnostics_do_not_veto_finite_primary_metric():
    grid = [dict(example={'scene': 'overflow', 'seed': 1}, positions=[
        dict(position={'surface': 'vector'}, shape=[1])])]
    row = dict(example=grid[0]['example'], **grid[0]['positions'][0], offset=0,
               student=[1e154], target=[0.], neutral=[-1e154])
    result = score_chunks([row], grid)
    assert result['error'] == 1.
    assert result['examples'][0]['student_edit_energy'] is None
    assert result['examples'][0]['student_teacher_edit_dot'] is None
    assert result['signed_edit_gain'] is None


def test_fp64_operands_precede_subtraction_and_nonzero_threshold_is_refused():
    import torch
    grid = [dict(example={'scene': 'precision', 'seed': 1}, positions=[
        dict(position={'surface': 'vector'}, shape=[1])])]
    # These operands are exact FP32 values, but their subtraction in FP32 loses
    # the low part. Lists convert operands first, as products must do on CPU.
    student = torch.tensor([1e8], dtype=torch.float32)
    target = torch.tensor([3.], dtype=torch.float32)
    row = dict(example=grid[0]['example'], **grid[0]['positions'][0], offset=0,
        student=student.tolist(), target=target.tolist(), neutral=[0.])
    result = score_chunks([row], grid)
    difference = 1e8 - 3.
    assert result['examples'][0]['student_teacher_sse'] == difference * difference
    assert result['examples'][0]['student_teacher_sse'] != float((student - target).double().square()[0])
    with pytest.raises(ValueError, match='zero'):
        score_chunks([row], grid, min_edit_energy=1e-100)
