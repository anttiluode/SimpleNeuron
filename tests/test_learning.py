import numpy as np
from simple_neuron.learning import oja_port_update


def test_inactive_port_does_not_change():
    B = np.eye(2)
    updated = oja_port_update(B, np.array([0.5, 0.5]), np.array([1.0, 0.0]), eta=0.1)
    assert np.allclose(updated[:, 1], B[:, 1])


def test_active_port_moves_toward_local_state():
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = np.array([1.0, 1.0]) / np.sqrt(2.0)
    before = float(B[:, 0] @ x)
    updated = oja_port_update(B, x, np.array([1.0, 0.0]), eta=0.2)
    after = float(updated[:, 0] @ x)
    assert after > before
    assert np.isclose(np.linalg.norm(updated[:, 0]), 1.0)
