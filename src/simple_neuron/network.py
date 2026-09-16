from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable
import numpy as np

from .core import ResidentNeuron
from .transducer import DirectTransducer


@dataclass(frozen=True)
class Route:
    source: int
    target: int
    port: int
    delay: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.delay < 1:
            raise ValueError('delay must be >= 1')


class SparseNetwork:
    def __init__(
        self,
        neurons: Iterable[ResidentNeuron],
        routes: Iterable[Route],
        transducers: Iterable[object] | None = None,
    ) -> None:
        self.neurons = list(neurons)
        self.routes = list(routes)
        if not self.neurons:
            raise ValueError('network needs at least one neuron')
        n = len(self.neurons)
        for route in self.routes:
            if not (0 <= route.source < n and 0 <= route.target < n):
                raise ValueError('route source/target out of range')
            if not (0 <= route.port < self.neurons[route.target].n_ports):
                raise ValueError('route port out of range')
        if transducers is None:
            self.transducers = [DirectTransducer(neuron.n_ports) for neuron in self.neurons]
        else:
            self.transducers = list(transducers)
            if len(self.transducers) != n:
                raise ValueError('one transducer per neuron required')
        self.current_step = 0
        self._queue: dict[int, list[tuple[int, int, float]]] = defaultdict(list)

    def step(
        self,
        external_port_drive: dict[int, np.ndarray] | None = None,
        publish: bool = True,
    ) -> dict:
        external_port_drive = external_port_drive or {}
        incoming = [np.zeros(neuron.n_ports, dtype=float) for neuron in self.neurons]

        for target, port, weight in self._queue.pop(self.current_step, []):
            incoming[target][port] += weight

        for idx, drive in external_port_drive.items():
            if not (0 <= idx < len(self.neurons)):
                raise ValueError('external drive neuron index out of range')
            arr = np.asarray(drive, dtype=float)
            if arr.shape != incoming[idx].shape:
                raise ValueError('external port drive shape mismatch')
            incoming[idx] += arr

        states = []
        soma_values = []
        spikes = []
        effective_drives = []
        for neuron, transducer, events in zip(self.neurons, self.transducers, incoming):
            drive = np.asarray(transducer.step(events), dtype=float)
            state, soma_value, spike = neuron.step(port_drive=drive, publish=publish)
            states.append(state)
            soma_values.append(soma_value)
            spikes.append(spike)
            effective_drives.append(drive.copy())

        for route in self.routes:
            if spikes[route.source]:
                due = self.current_step + route.delay
                self._queue[due].append((route.target, route.port, float(route.weight)))

        out = {
            'step': self.current_step,
            'states': states,
            'soma_values': soma_values,
            'spikes': spikes,
            'port_events': [x.copy() for x in incoming],
            'effective_drives': effective_drives,
        }
        self.current_step += 1
        return out
