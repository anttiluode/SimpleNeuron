from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _as_events(events: np.ndarray, n_ports: int) -> np.ndarray:
    arr = np.asarray(events, dtype=float)
    if arr.shape != (n_ports,):
        raise ValueError('events must have shape (n_ports,)')
    return arr


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class DirectTransducer:
    n_ports: int

    def __post_init__(self) -> None:
        self.state = np.zeros(self.n_ports, dtype=float)

    def reset(self) -> None:
        self.state.fill(0.0)

    def step(self, events: np.ndarray) -> np.ndarray:
        events = _as_events(events, self.n_ports)
        self.state = events.copy()
        return events.copy()


@dataclass
class LinearTraceTransducer:
    n_ports: int
    decay: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.decay < 1.0):
            raise ValueError('decay must be in [0, 1)')
        self.state = np.zeros(self.n_ports, dtype=float)

    def reset(self) -> None:
        self.state.fill(0.0)

    def step(self, events: np.ndarray) -> np.ndarray:
        events = _as_events(events, self.n_ports)
        self.state = self.decay * self.state + events
        return self.state.copy()


@dataclass
class SoftKneeTransducer:
    n_ports: int
    decay: float
    threshold: float
    softness: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.decay < 1.0):
            raise ValueError('decay must be in [0, 1)')
        if self.softness <= 0.0:
            raise ValueError('softness must be positive')
        self.state = np.zeros(self.n_ports, dtype=float)
        one_gate = float(_sigmoid((1.0 - self.threshold) / self.softness))
        self._single_pulse_norm = one_gate if one_gate > 1e-12 else 1e-12

    def reset(self) -> None:
        self.state.fill(0.0)

    def step(self, events: np.ndarray) -> np.ndarray:
        events = _as_events(events, self.n_ports)
        self.state = self.decay * self.state + events
        gate = _sigmoid((self.state - self.threshold) / self.softness)
        return (self.state * gate / self._single_pulse_norm).copy()
