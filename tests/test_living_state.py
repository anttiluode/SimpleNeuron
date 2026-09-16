import json
from pathlib import Path

import pytest

from simple_neuron.experiment import run_gate4, run_v1


def _assert_close(actual, expected, *, rel=1e-12, abs_tol=1e-12):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_close(actual[key], expected[key], rel=rel, abs_tol=abs_tol)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for a_item, e_item in zip(actual, expected):
            _assert_close(a_item, e_item, rel=rel, abs_tol=abs_tol)
        return
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=rel, abs=abs_tol)
        return
    assert actual == expected


def test_gate4_reports_living_state_against_stateless_attacker():
    gate = run_gate4(seeds=8)
    assert gate['classification'] in {'PASS_LIVING_STATE', 'RESIDENT_STATE_NOT_NEEDED'}
    assert gate['living_context_accuracy_mean'] >= 0.0
    assert gate['stateless_context_accuracy_mean'] >= 0.0
    assert gate['interpolation_error_max'] >= 0.0
    assert gate['same_ping_state_separation_mean'] >= 0.0


def test_v1_extends_v0_with_gate4_without_renaming_old_gates():
    receipt = run_v1(seeds=4)
    assert receipt['version'] == 'v1'
    assert list(receipt['gates']) == ['gate0', 'gate1', 'gate2', 'gate3', 'gate4']


def test_frozen_v1_receipt_matches_canonical_run():
    frozen = json.loads(Path('results/v1.json').read_text())
    _assert_close(run_v1(seeds=64), frozen)
