import numpy as np
from simple_neuron.core import ResidentNeuron, stable_operator


def test_stable_operator_respects_requested_radius():
    A = stable_operator(dim=4, radius=0.92, seed=3)
    spectral_radius = max(abs(np.linalg.eigvals(A)))
    assert spectral_radius <= 0.9200001


def test_silence_decays_displaced_state():
    A = np.eye(2) * 0.8
    B = np.eye(2)
    neuron = ResidentNeuron(A=A, B=B, soma=np.array([1.0, 0.0]), threshold=10.0,
                            state=np.array([1.0, 0.0]))
    before = np.linalg.norm(neuron.state)
    neuron.step(port_drive=np.zeros(2))
    assert np.linalg.norm(neuron.state) < before


def test_simultaneous_ports_add_before_publication():
    A = np.zeros((2, 2))
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    neuron = ResidentNeuron(A=A, B=B, soma=np.array([1.0, 1.0]), threshold=99.0)
    state, soma_value, spike = neuron.step(port_drive=np.array([0.25, 0.75]))
    assert np.allclose(state, [0.25, 0.75])
    assert soma_value == 1.0
    assert spike == 0


def test_publication_suppression_does_not_erase_state():
    A = np.eye(2)
    B = np.eye(2)
    neuron = ResidentNeuron(A=A, B=B, soma=np.array([1.0, 0.0]), threshold=0.5)
    state, _, spike = neuron.step(port_drive=np.array([1.0, 0.0]), publish=False)
    assert spike == 0
    assert np.allclose(state, [1.0, 0.0])
