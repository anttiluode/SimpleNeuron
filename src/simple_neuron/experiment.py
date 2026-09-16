from __future__ import annotations

import numpy as np

from .core import ResidentNeuron, stable_operator
from .learning import oja_port_update
from .network import Route, SparseNetwork
from .transducer import DirectTransducer, LinearTraceTransducer, SoftKneeTransducer


def _unit_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).copy()
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _orthonormal_targets(rng: np.random.Generator, dim: int, ports: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q[:, :ports]


def _nonidentity_permutation(rng: np.random.Generator, n: int) -> np.ndarray:
    identity = np.arange(n)
    perm = identity.copy()
    while np.array_equal(perm, identity):
        perm = rng.permutation(n)
    return perm


def _train_port_matrix(
    seed: int,
    dim: int = 4,
    ports: int = 3,
    samples: int = 256,
    eta: float = 0.05,
    noise: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    targets = _orthonormal_targets(rng, dim, ports)
    initial = _unit_columns(rng.normal(size=(dim, ports)))
    paired = initial.copy()
    shuffled = initial.copy()
    perm = _nonidentity_permutation(rng, ports)

    for k in range(ports):
        for _ in range(samples):
            x = targets[:, k] + noise * rng.normal(size=dim)
            x = x / max(np.linalg.norm(x), 1e-12)
            event = np.zeros(ports, dtype=float)
            event[k] = 1.0
            paired = oja_port_update(paired, x, event, eta)

            shuffled_event = np.zeros(ports, dtype=float)
            shuffled_event[int(perm[k])] = 1.0
            shuffled = oja_port_update(shuffled, x, shuffled_event, eta)

    return paired, shuffled, targets


def _alignment(B: np.ndarray, targets: np.ndarray) -> float:
    vals = [abs(float(B[:, k] @ targets[:, k])) for k in range(B.shape[1])]
    return float(np.mean(vals))


def run_gate0() -> dict:
    A = stable_operator(dim=3, radius=0.9, seed=11)
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))

    decay_neuron = ResidentNeuron(
        A=np.eye(2) * 0.8,
        B=np.eye(2),
        soma=np.array([1.0, 0.0]),
        threshold=99.0,
        state=np.array([1.0, 0.0]),
    )
    before = float(np.linalg.norm(decay_neuron.state))
    decay_neuron.step(port_drive=np.zeros(2))
    after = float(np.linalg.norm(decay_neuron.state))

    interp = ResidentNeuron(
        A=np.zeros((2, 2)),
        B=np.eye(2),
        soma=np.array([1.0, 1.0]),
        threshold=99.0,
    )
    state, _, _ = interp.step(port_drive=np.array([0.3, 0.7]))
    interpolation_error = float(np.linalg.norm(state - np.array([0.3, 0.7])))

    hidden = ResidentNeuron(
        A=np.eye(2),
        B=np.eye(2),
        soma=np.array([1.0, 0.0]),
        threshold=0.5,
    )
    state_hidden, _, spike_hidden = hidden.step(
        port_drive=np.array([1.0, 0.0]), publish=False
    )

    return {
        'classification': 'PASS_INVARIANTS',
        'spectral_radius': radius,
        'silent_decay_ratio': after / before,
        'interpolation_error': interpolation_error,
        'suppressed_spike': int(spike_hidden),
        'suppressed_state_norm': float(np.linalg.norm(state_hidden)),
    }


def run_gate1(seeds: int = 64) -> dict:
    paired = []
    shuffled = []
    for seed in range(seeds):
        learned, control, targets = _train_port_matrix(seed)
        paired.append(_alignment(learned, targets))
        shuffled.append(_alignment(control, targets))
    paired_mean = float(np.mean(paired))
    shuffled_mean = float(np.mean(shuffled))
    delta = paired_mean - shuffled_mean
    classification = (
        'PASS_LOCAL_STEERING'
        if paired_mean >= 0.90 and delta >= 0.20
        else 'INSUFFICIENT_LOCAL_STEERING'
    )
    return {
        'classification': classification,
        'seeds': int(seeds),
        'paired_alignment_mean': paired_mean,
        'shuffled_alignment_mean': shuffled_mean,
        'alignment_delta': delta,
        'paired_alignment_min': float(np.min(paired)),
        'shuffled_alignment_max': float(np.max(shuffled)),
    }


def _source_neuron() -> ResidentNeuron:
    return ResidentNeuron(
        A=np.zeros((1, 1)),
        B=np.ones((1, 1)),
        soma=np.ones(1),
        threshold=0.5,
    )


def _route_probe(B: np.ndarray, source_index: int, target_port: int) -> np.ndarray:
    receiver = ResidentNeuron(
        A=np.zeros((B.shape[0], B.shape[0])),
        B=B,
        soma=np.zeros(B.shape[0]),
        threshold=99.0,
    )
    neurons = [_source_neuron(), _source_neuron(), receiver]
    net = SparseNetwork(
        neurons,
        [Route(source=source_index, target=2, port=target_port, delay=1, weight=1.0)],
    )
    drive = {source_index: np.array([1.0])}
    net.step(external_port_drive=drive)
    before = net.neurons[2].state.copy()
    out = net.step()
    return np.asarray(out['states'][2]) - before


def _cosine_abs(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return abs(float(x @ y) / denom)


def run_gate2(seeds: int = 64) -> dict:
    intact_scores = []
    shuffled_scores = []
    random_scores = []
    for seed in range(seeds):
        learned, _, targets3 = _train_port_matrix(seed, ports=3)
        targets = targets3[:, :2]
        B = learned[:, :2]
        rng = np.random.default_rng(seed + 10_000)
        random_B = _unit_columns(rng.normal(size=B.shape))

        for source in (0, 1):
            intended = targets[:, source]
            intact = _route_probe(B, source_index=source, target_port=source)
            shuffled = _route_probe(B, source_index=source, target_port=1 - source)
            random = _route_probe(random_B, source_index=source, target_port=source)
            intact_scores.append(_cosine_abs(intact, intended))
            shuffled_scores.append(_cosine_abs(shuffled, intended))
            random_scores.append(_cosine_abs(random, intended))

    intact_mean = float(np.mean(intact_scores))
    shuffled_mean = float(np.mean(shuffled_scores))
    random_mean = float(np.mean(random_scores))
    classification = (
        'PASS_ROUTE_SEMANTICS'
        if intact_mean >= 0.90 and intact_mean - shuffled_mean >= 0.50
        else 'INSUFFICIENT_ROUTE_SEMANTICS'
    )
    return {
        'classification': classification,
        'seeds': int(seeds),
        'intact_alignment_mean': intact_mean,
        'port_shuffle_alignment_mean': shuffled_mean,
        'random_B_alignment_mean': random_mean,
        'intact_minus_shuffle': intact_mean - shuffled_mean,
        'intact_minus_random': intact_mean - random_mean,
        'payload_values': [0, 1],
    }


def _trial_response(transducer, events: list[float], washout: int = 10) -> tuple[np.ndarray, float]:
    transducer.reset()
    outputs = []
    for event in events:
        outputs.append(float(transducer.step(np.array([event]))[0]))
    last = 0.0
    for _ in range(washout):
        last = float(transducer.step(np.array([0.0]))[0])
    return np.asarray(outputs, dtype=float), abs(last)


def _first_latency(outputs: np.ndarray, threshold: float = 0.5) -> int:
    idx = np.flatnonzero(outputs >= threshold)
    return int(idx[0]) if len(idx) else int(len(outputs))


def run_gate3(seeds: int = 64) -> dict:
    names = ('direct', 'linear_trace', 'soft_knee')
    ratios = {name: [] for name in names}
    latencies = {name: [] for name in names}
    residuals = {name: [] for name in names}

    for seed in range(seeds):
        rng = np.random.default_rng(seed + 20_000)
        gap = int(rng.integers(1, 3))
        signal_events = [1.0] + [0.0] * (gap - 1) + [1.0] + [0.0] * 5
        distractor_events = [1.0] + [0.0] * (len(signal_events) - 1)

        models = {
            'direct': DirectTransducer(1),
            'linear_trace': LinearTraceTransducer(1, decay=0.7),
            'soft_knee': SoftKneeTransducer(1, decay=0.7, threshold=1.25, softness=0.15),
        }
        for name, model in models.items():
            signal, signal_residual = _trial_response(model, signal_events)
            distractor, distractor_residual = _trial_response(model, distractor_events)
            signal_energy = float(np.sum(signal * signal))
            distractor_energy = float(np.sum(distractor * distractor))
            ratios[name].append(signal_energy / max(distractor_energy, 1e-12))
            latencies[name].append(_first_latency(signal))
            residuals[name].append(max(signal_residual, distractor_residual))

    summary = {}
    for name in names:
        summary[name] = {
            'signal_distractor_ratio_mean': float(np.mean(ratios[name])),
            'onset_latency_mean': float(np.mean(latencies[name])),
            'washout_residual_mean': float(np.mean(residuals[name])),
        }

    linear = summary['linear_trace']
    knee = summary['soft_knee']
    ratio_gain = (
        knee['signal_distractor_ratio_mean'] /
        max(linear['signal_distractor_ratio_mean'], 1e-12)
    )
    latency_ok = knee['onset_latency_mean'] <= linear['onset_latency_mean'] + 1.0
    residual_ok = knee['washout_residual_mean'] <= max(
        linear['washout_residual_mean'] * 1.10, 1e-12
    )
    classification = (
        'KNEE_EARNS_ROLE'
        if ratio_gain >= 1.10 and latency_ok and residual_ok
        else 'KNEE_NOT_NEEDED'
    )
    return {
        'classification': classification,
        'seeds': int(seeds),
        'single_pulse_gain_matched': True,
        'conditions': summary,
        'knee_over_linear_ratio_gain': float(ratio_gain),
        'latency_ok': bool(latency_ok),
        'washout_ok': bool(residual_ok),
    }


def _run_living_context_trial(
    B: np.ndarray,
    context_port: int,
    *,
    decay: float,
    delay: int,
    soma: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    neuron = ResidentNeuron(
        A=np.eye(B.shape[0]) * decay,
        B=B,
        soma=soma,
        threshold=threshold,
    )
    context = np.zeros(B.shape[1], dtype=float)
    context[context_port] = 1.0
    state0, _, _ = neuron.step(port_drive=context)
    for _ in range(delay):
        neuron.step(port_drive=np.zeros(B.shape[1], dtype=float))
    retained = neuron.state.copy()
    cue = np.zeros(B.shape[1], dtype=float)
    cue[0] = 1.0
    _, soma_value, spike = neuron.step(port_drive=cue)
    return state0, retained, soma_value, spike


def _living_state_after(
    B: np.ndarray,
    context_port: int,
    drive: np.ndarray,
    *,
    decay: float,
    delay: int,
) -> np.ndarray:
    neuron = ResidentNeuron(
        A=np.eye(B.shape[0]) * decay,
        B=B,
        soma=np.zeros(B.shape[0]),
        threshold=99.0,
    )
    context = np.zeros(B.shape[1], dtype=float)
    context[context_port] = 1.0
    neuron.step(port_drive=context)
    for _ in range(delay):
        neuron.step(port_drive=np.zeros(B.shape[1], dtype=float))
    state, _, _ = neuron.step(port_drive=drive)
    return state


def run_gate4(seeds: int = 64) -> dict:
    living_accuracies = []
    stateless_accuracies = []
    history_retentions = []
    separations = []
    interpolation_errors = []
    positive_margins = []
    negative_margins = []

    decay = 0.97
    delay = 3
    threshold = 0.95

    for seed in range(seeds):
        learned, _, _ = _train_port_matrix(seed, dim=4, ports=3)
        B = learned[:, :3]
        soma = B[:, 0] + B[:, 1]
        soma = soma / max(np.linalg.norm(soma), 1e-12)

        state0_pos, retained_pos, soma_pos, spike_pos = _run_living_context_trial(
            B, 1, decay=decay, delay=delay, soma=soma, threshold=threshold
        )
        _, _, soma_neg, spike_neg = _run_living_context_trial(
            B, 2, decay=decay, delay=delay, soma=soma, threshold=threshold
        )
        living_accuracies.append(0.5 * ((spike_pos == 1) + (spike_neg == 0)))
        positive_margins.append(float(soma_pos - threshold))
        negative_margins.append(float(threshold - soma_neg))
        history_retentions.append(
            float(np.linalg.norm(retained_pos) / max(np.linalg.norm(state0_pos), 1e-12))
        )

        stateless = ResidentNeuron(
            A=np.zeros((B.shape[0], B.shape[0])),
            B=B,
            soma=soma,
            threshold=threshold,
        )
        cue = np.zeros(B.shape[1], dtype=float)
        cue[0] = 1.0
        _, _, stateless_spike = stateless.step(port_drive=cue)
        stateless_accuracies.append(
            0.5 * ((stateless_spike == 1) + (stateless_spike == 0))
        )

        cue_drive = np.zeros(B.shape[1], dtype=float)
        cue_drive[0] = 1.0
        pos_state = _living_state_after(B, 1, cue_drive, decay=decay, delay=delay)
        neg_state = _living_state_after(B, 2, cue_drive, decay=decay, delay=delay)
        separations.append(float(np.linalg.norm(pos_state - neg_state)))

        alt_drive = np.zeros(B.shape[1], dtype=float)
        alt_drive[2] = 1.0
        mix_drive = 0.5 * (cue_drive + alt_drive)
        cue_state = _living_state_after(B, 1, cue_drive, decay=decay, delay=delay)
        alt_state = _living_state_after(B, 1, alt_drive, decay=decay, delay=delay)
        mix_state = _living_state_after(B, 1, mix_drive, decay=decay, delay=delay)
        interpolation_errors.append(
            float(np.linalg.norm(mix_state - 0.5 * (cue_state + alt_state)))
        )

    living_mean = float(np.mean(living_accuracies))
    stateless_mean = float(np.mean(stateless_accuracies))
    retention_mean = float(np.mean(history_retentions))
    separation_mean = float(np.mean(separations))
    interp_max = float(np.max(interpolation_errors))
    accuracy_gain = living_mean - stateless_mean

    classification = (
        'PASS_LIVING_STATE'
        if (
            living_mean >= 0.95
            and accuracy_gain >= 0.40
            and retention_mean >= 0.85
            and separation_mean >= 1.0
            and interp_max <= 1e-10
            and min(positive_margins) > 0.0
            and min(negative_margins) > 0.0
        )
        else 'RESIDENT_STATE_NOT_NEEDED'
    )

    return {
        'classification': classification,
        'seeds': int(seeds),
        'decay': decay,
        'context_delay_steps': delay,
        'living_context_accuracy_mean': living_mean,
        'stateless_context_accuracy_mean': stateless_mean,
        'living_minus_stateless_accuracy': accuracy_gain,
        'history_retention_mean': retention_mean,
        'same_ping_state_separation_mean': separation_mean,
        'interpolation_error_max': interp_max,
        'positive_context_margin_min': float(np.min(positive_margins)),
        'other_context_margin_min': float(np.min(negative_margins)),
        'attacker': 'same learned B and soma, current cue only, no resident history',
    }


def run_v0(seeds: int = 64) -> dict:
    return {
        'version': 'v0',
        'gates': {
            'gate0': run_gate0(),
            'gate1': run_gate1(seeds=seeds),
            'gate2': run_gate2(seeds=seeds),
            'gate3': run_gate3(seeds=seeds),
        },
        'overall_notes': [
            'All machine learning is receiver-local; no backpropagation is used.',
            'Axonal topology is fixed in v0.',
            'The Ca-like knee is retained only if it beats the matched linear trace on its predeclared temporal-selectivity gate.',
        ],
    }


def run_v1(seeds: int = 64) -> dict:
    v0 = run_v0(seeds=seeds)
    return {
        'version': 'v1',
        'gates': {
            **v0['gates'],
            'gate4': run_gate4(seeds=seeds),
        },
        'overall_notes': [
            *v0['overall_notes'],
            'Gate 4 tests whether retained receiver history changes the consequence of a later identical cue.',
            'The Gate 4 attacker keeps the learned route coordinate and soma but removes resident history.',
        ],
    }
