import numpy as np
from simple_neuron.core import ResidentNeuron
from simple_neuron.network import Route, SparseNetwork


def make_neuron(threshold):
    return ResidentNeuron(A=np.zeros((2, 2)), B=np.eye(2),
                          soma=np.array([1.0, 0.0]), threshold=threshold)


def test_route_delivers_same_bit_to_named_target_port_after_delay():
    source = make_neuron(threshold=0.5)
    target = make_neuron(threshold=99.0)
    net = SparseNetwork([source, target], [Route(0, 1, port=1, delay=1, weight=1.0)])
    net.step(external_port_drive={0: np.array([1.0, 0.0])})
    out = net.step()
    assert np.allclose(out['states'][1], [0.0, 1.0])


def test_changing_only_port_changes_target_state_direction():
    source_a = make_neuron(0.5); target_a = make_neuron(99.0)
    source_b = make_neuron(0.5); target_b = make_neuron(99.0)
    net_a = SparseNetwork([source_a, target_a], [Route(0, 1, 0, 1, 1.0)])
    net_b = SparseNetwork([source_b, target_b], [Route(0, 1, 1, 1, 1.0)])
    net_a.step(external_port_drive={0: np.array([1.0, 0.0])}); a = net_a.step()['states'][1]
    net_b.step(external_port_drive={0: np.array([1.0, 0.0])}); b = net_b.step()['states'][1]
    assert not np.allclose(a, b)
