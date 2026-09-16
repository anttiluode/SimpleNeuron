#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple_neuron.experiment import run_v1


def main() -> None:
    parser = argparse.ArgumentParser(description='Run SimpleNeuron v1 gates')
    parser.add_argument('--seeds', type=int, default=64)
    parser.add_argument('--out', type=Path, default=Path('results/v1.json'))
    args = parser.parse_args()

    receipt = run_v1(seeds=args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')

    print('SimpleNeuron v1')
    for gate, result in receipt['gates'].items():
        print(f"{gate}: {result['classification']}")
    print(f'written: {args.out}')


if __name__ == '__main__':
    main()
