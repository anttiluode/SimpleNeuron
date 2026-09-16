import json
from pathlib import Path

from simple_neuron.experiment import run_v0


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


def test_frozen_v0_receipt_matches_canonical_run():
    frozen = json.loads(Path('results/v0.json').read_text())
    assert run_v0(seeds=64) == frozen
