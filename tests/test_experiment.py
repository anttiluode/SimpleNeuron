import json
from pathlib import Path

import pytest

from simple_neuron.experiment import run_v0


def _assert_receipts_close(actual, expected, *, rel=1e-12, abs_tol=1e-12):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_receipts_close(actual[key], expected[key], rel=rel, abs_tol=abs_tol)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for a_item, e_item in zip(actual, expected):
            _assert_receipts_close(a_item, e_item, rel=rel, abs_tol=abs_tol)
        return
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=rel, abs=abs_tol)
        return
    assert actual == expected


def test_v0_receipt_is_deterministic():
    a = run_v0(seeds=8)
    b = run_v0(seeds=8)
    assert a == b


def test_v0_receipt_contains_falsifiable_knee_classification():
    receipt = run_v0(seeds=8)
    assert receipt['gates']['gate3']['classification'] in {
        'KNEE_EARNS_ROLE', 'KNEE_NOT_NEEDED'
    }


def test_gate1_contains_paired_and_shuffled_alignment():
    gate1 = run_v0(seeds=8)['gates']['gate1']
    assert 'paired_alignment_mean' in gate1
    assert 'shuffled_alignment_mean' in gate1
    assert 'alignment_delta' in gate1


def test_receipt_comparison_allows_machine_precision_float_drift():
    _assert_receipts_close({'metric': 1.0 + 5e-15}, {'metric': 1.0})


def test_receipt_comparison_rejects_material_float_drift():
    with pytest.raises(AssertionError):
        _assert_receipts_close({'metric': 1.0 + 1e-6}, {'metric': 1.0})


def test_frozen_v0_receipt_matches_canonical_run():
    frozen = json.loads(Path('results/v0.json').read_text())
    _assert_receipts_close(run_v0(seeds=64), frozen)
