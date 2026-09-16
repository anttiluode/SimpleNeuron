from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def stable_operator(dim: int, radius: float = 0.95, seed: int = 0) -> np.ndarray:
    if dim <= 0:
        raise ValueError('dim must be positive')
    if not (0.0 <= radius < 1.0):
        raise ValueError('radius must be in [0, 1)')
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim))
    eig_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if eig_radius == 0.0:
        return np.zeros((dim, dim), dtype=float)
    return A * (radius / eig_radius)


@dataclass
class ResidentNeuron:
    A: np.ndarray
    B: np.ndarray
    soma: np.ndarray
    threshold: float
    state: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.soma = np.asarray(self.soma, dtype=float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError('A must be square')
        dim = self.A.shape[0]
        if self.B.ndim != 2 or self.B.shape[0] != dim:
            raise ValueError('B must have shape (state_dim, n_ports)')
        if self.soma.shape != (dim,):
            raise ValueError('soma must have shape (state_dim,)')
        if self.state is None:
            self.state = np.zeros(dim, dtype=float)
        else:
            self.state = np.asarray(self.state, dtype=float).copy()
            if self.state.shape != (dim,):
                raise ValueError('state must have shape (state_dim,)')

    @property
    def n_ports(self) -> int:
        return self.B.shape[1]

    def step(
        self,
        port_drive: np.ndarray | None = None,
        sensory_drive: np.ndarray | None = None,
        publish: bool = True,
    ) -> tuple[np.ndarray, float, int]:
        if port_drive is None:
            port_drive = np.zeros(self.n_ports, dtype=float)
        port_drive = np.asarray(port_drive, dtype=float)
        if port_drive.shape != (self.n_ports,):
            raise ValueError('port_drive must have shape (n_ports,)')
        if sensory_drive is None:
            sensory_drive = np.zeros_like(self.state)
        sensory_drive = np.asarray(sensory_drive, dtype=float)
        if sensory_drive.shape != self.state.shape:
            raise ValueError('sensory_drive must have shape (state_dim,)')

        self.state = self.A @ self.state + sensory_drive + self.B @ port_drive
        soma_value = float(self.soma @ self.state)
        spike = int(bool(publish) and soma_value > self.threshold)
        return self.state.copy(), soma_value, spike
