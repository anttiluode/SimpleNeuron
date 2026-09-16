import numpy as np
from simple_neuron.transducer import DirectTransducer, LinearTraceTransducer, SoftKneeTransducer


def test_direct_is_identity():
    t = DirectTransducer(2)
    assert np.allclose(t.step(np.array([1.0, 0.25])), [1.0, 0.25])


def test_linear_trace_accumulates_and_decays():
    t = LinearTraceTransducer(1, decay=0.5)
    assert np.allclose(t.step(np.array([1.0])), [1.0])
    assert np.allclose(t.step(np.array([0.0])), [0.5])
    assert np.allclose(t.step(np.array([1.0])), [1.25])


def test_soft_knee_matches_single_pulse_gain():
    t = SoftKneeTransducer(1, decay=0.5, threshold=1.25, softness=0.15)
    out = t.step(np.array([1.0]))
    assert np.allclose(out, [1.0], atol=1e-12)


def test_soft_knee_is_more_burst_selective_than_linear_trace():
    linear = LinearTraceTransducer(1, decay=0.7)
    knee = SoftKneeTransducer(1, decay=0.7, threshold=1.25, softness=0.15)
    linear.step(np.array([1.0])); knee.step(np.array([1.0]))
    l2 = linear.step(np.array([1.0]))[0]
    k2 = knee.step(np.array([1.0]))[0]
    assert k2 > l2
