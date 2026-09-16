from __future__ import annotations

import numpy as np


def oja_port_update(
    B: np.ndarray,
    local_state: np.ndarray,
    port_events: np.ndarray,
    eta: float,
) -> np.ndarray:
    B = np.asarray(B, dtype=float)
    x = np.asarray(local_state, dtype=float)
    events = np.asarray(port_events, dtype=float)
    if B.ndim != 2:
        raise ValueError('B must be 2-D')
    if x.shape != (B.shape[0],):
        raise ValueError('local_state must match B state dimension')
    if events.shape != (B.shape[1],):
        raise ValueError('port_events must match B port dimension')
    if eta < 0:
        raise ValueError('eta must be non-negative')

    updated = B.copy()
    for k, event in enumerate(events):
        if event == 0.0:
            continue
        b = updated[:, k]
        y = float(b @ x)
        b = b + eta * float(event) * y * (x - y * b)
        norm = float(np.linalg.norm(b))
        if norm > 1e-12:
            b = b / norm
        updated[:, k] = b
    return updated
