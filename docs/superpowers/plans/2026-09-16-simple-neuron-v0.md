# SimpleNeuron v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and falsify a minimal resident-state neuron-like machine in which sparse routed events locally learn how to steer receiver state, with a Ca-like soft knee retained only if it beats simpler temporal controls.

**Architecture:** Each unit owns a stable linear resident state `x`, fixed soma/AIS readout, and receiving-port steering matrix `B`. `B[:, k]` is learned only from the event on port `k` and the receiver's local state using route-conditioned Oja learning. Sparse binary spikes are routed through an explicit delayed graph. The optional receiver-side temporal stage is compared as direct, linear leaky trace, and leaky trace plus soft knee.

**Tech Stack:** Python 3.11+, NumPy, pytest, JSON receipts, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-16-simple-neuron-v0-design.md`

## Global Constraints

- Pure Python + NumPy + pytest.
- No PyTorch, autograd, task-loss optimizer, or backpropagation in the neuron-like machine.
- All machine learning is local to receiving state/port variables.
- Axonal topology is fixed in v0.
- The Ca-like knee is optional and CI must accept either `KNEE_EARNS_ROLE` or `KNEE_NOT_NEEDED`.
- Scientific failures remain recorded; engineering tests enforce invariants, determinism, and receipt schema rather than preferred empirical signs.

---

## File structure

```text
pyproject.toml
README.md
.github/workflows/ci.yml
src/simple_neuron/__init__.py
src/simple_neuron/core.py
src/simple_neuron/learning.py
src/simple_neuron/transducer.py
src/simple_neuron/network.py
src/simple_neuron/experiment.py
experiments/run_v0.py
results/v0.json
tests/test_core.py
tests/test_learning.py
tests/test_transducer.py
tests/test_network.py
tests/test_experiment.py
```

### Task 1: Resident-state core and soma/AIS boundary

**Files:**
- Create: `pyproject.toml`
- Create: `src/simple_neuron/__init__.py`
- Create: `src/simple_neuron/core.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces `stable_operator(dim: int, radius: float, seed: int) -> np.ndarray`.
- Produces `ResidentNeuron(A, B, soma, threshold, state=None)`.
- Produces `ResidentNeuron.step(port_drive=None, sensory_drive=None, publish=True) -> tuple[np.ndarray, float, int]`.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_core.py -q`

Expected: collection/import failure because `simple_neuron.core` does not exist.

- [ ] **Step 3: Implement minimal core**

Implement `ResidentNeuron` as a linear state-space unit:

```python
x_next = A @ x + sensory_drive + B @ port_drive
soma_value = float(soma @ x_next)
spike = int(publish and soma_value > threshold)
```

Validate dimensions in `__post_init__`. `stable_operator` draws a deterministic random matrix and rescales it to the requested spectral radius.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_core.py -q`

Expected: 4 passed.

- [ ] **Step 5: Commit**

Commit message: `feat: add resident-state neuron core`

---

### Task 2: Route-conditioned local Oja steering

**Files:**
- Create: `src/simple_neuron/learning.py`
- Test: `tests/test_learning.py`

**Interfaces:**
- Produces `oja_port_update(B, local_state, port_events, eta) -> np.ndarray`.
- Consumes the receiver-local state, current port-event vector, and current steering matrix only.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_learning.py -q`

Expected: import failure for missing `simple_neuron.learning`.

- [ ] **Step 3: Implement minimal local learner**

For each active port `k`:

```python
b = B[:, k]
y = float(b @ local_state)
b = b + eta * event * y * (local_state - y * b)
b = b / (norm(b) + 1e-12)
```

Return a copy; never mutate the caller's matrix in place.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_learning.py -q`

Expected: 2 passed.

- [ ] **Step 5: Commit**

Commit message: `feat: add local Oja port learning`

---

### Task 3: Direct, linear-trace, and soft-knee receiver stages

**Files:**
- Create: `src/simple_neuron/transducer.py`
- Test: `tests/test_transducer.py`

**Interfaces:**
- Produces `DirectTransducer(n_ports)` with `step(events) -> np.ndarray`.
- Produces `LinearTraceTransducer(n_ports, decay)` with `step(events) -> np.ndarray`.
- Produces `SoftKneeTransducer(n_ports, decay, threshold, softness)` with `step(events) -> np.ndarray`.
- All transducers expose `.state` and `.reset()`.

- [ ] **Step 1: Write failing tests**

```python
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
    assert k2 / 1.0 > l2 / 1.0
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_transducer.py -q`

Expected: import failure for missing `simple_neuron.transducer`.

- [ ] **Step 3: Implement minimal transducers**

Use `trace <- decay * trace + events` for both temporal models. For the knee use:

```python
gate = sigmoid((trace - threshold) / softness)
raw = trace * gate
```

and divide by the analytically computed `raw` value at `trace=1` so a fresh isolated unit pulse produces exactly `1.0`.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_transducer.py -q`

Expected: 4 passed.

- [ ] **Step 5: Commit**

Commit message: `feat: add receiver temporal transducers`

---

### Task 4: Sparse delayed axonal routing

**Files:**
- Create: `src/simple_neuron/network.py`
- Test: `tests/test_network.py`

**Interfaces:**
- Produces immutable `Route(source, target, port, delay, weight)`.
- Produces `SparseNetwork(neurons, routes, transducers=None)`.
- Produces `SparseNetwork.step(external_port_drive=None, publish=True) -> dict`.
- Route queue delivers source spikes after exactly `delay` integer steps to the named target port.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_network.py -q`

Expected: import failure for missing `simple_neuron.network`.

- [ ] **Step 3: Implement minimal sparse network**

Maintain an integer-step delivery queue. On each step:

1. collect due route deliveries per target port;
2. add any external port drive;
3. pass port events through the target's transducer if present;
4. step all neurons once;
5. schedule each emitted source spike onto every matching route at `current_step + delay`.

No route learning in v0.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_network.py -q`

Expected: 2 passed.

- [ ] **Step 5: Commit**

Commit message: `feat: add sparse delayed axonal routing`

---

### Task 5: Deterministic scientific Gates 0-3

**Files:**
- Create: `src/simple_neuron/experiment.py`
- Create: `experiments/run_v0.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Produces `run_gate0() -> dict`.
- Produces `run_gate1(seeds=64) -> dict`.
- Produces `run_gate2(seeds=64) -> dict`.
- Produces `run_gate3(seeds=64) -> dict`.
- Produces `run_v0(seeds=64) -> dict` with top-level schema `{version, gates, overall_notes}`.

- [ ] **Step 1: Write failing receipt tests**

```python
from simple_neuron.experiment import run_v0


def test_v0_receipt_is_deterministic():
    a = run_v0(seeds=8)
    b = run_v0(seeds=8)
    assert a == b


def test_v0_receipt_contains_falsifiable_knee_classification():
    receipt = run_v0(seeds=8)
    assert receipt['gates']['gate3']['classification'] in {
        'KNEE_EARNS_ROLE', 'KNEE_NOT_NEEDED'
    }


def test_gate1_contains_paired_and_shuffled_alignment():
    gate1 = run_v0(seeds=8)['gates']['gate1']
    assert 'paired_alignment_mean' in gate1
    assert 'shuffled_alignment_mean' in gate1
    assert 'alignment_delta' in gate1
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experiment.py -q`

Expected: import failure for missing `simple_neuron.experiment`.

- [ ] **Step 3: Implement Gate 0**

Return invariant measurements for stability, silent decay, additive interpolation, and publication suppression.

- [ ] **Step 4: Implement Gate 1**

For each seed:

1. create `d=4`, `ports=3` target directions from a seeded orthonormal basis;
2. initialize random unit `B` columns;
3. for each port, present 256 noisy local-state samples centered on its assigned target direction and set only that port event to 1;
4. apply `oja_port_update` after each sample;
5. freeze `B` and measure absolute cosine alignment against the assigned target direction;
6. run a matched shuffled control using the same target samples but a seeded permutation of port identities.

Record means and paired delta. Classification is descriptive: `PASS_LOCAL_STEERING` only when paired mean >= 0.90 and paired-minus-shuffled mean >= 0.20; otherwise `INSUFFICIENT_LOCAL_STEERING`.

- [ ] **Step 5: Implement Gate 2**

Build a two-source/one-receiver routed system. The two source neurons emit the same binary spike payload, but their routes land on two receiver ports whose `B` columns were locally trained on different directions in Gate-1 style tapes. Measure the cosine of the receiver's post-ping displacement with the intended direction under:

- intact ports;
- target-port shuffle;
- unlearned random `B`.

Use identical initial receiver states and source spike tapes across controls.

- [ ] **Step 6: Implement Gate 3**

Create a one-port tape with legitimate two-pulse bursts separated by quiet periods and isolated distractor pulses. For each transducer, feed the same tape into a fixed steering vector and measure:

- total steering energy in a short window after legitimate bursts;
- total steering energy after isolated distractors;
- signal/distractor ratio;
- first threshold-crossing latency of useful steering;
- residual state norm after a fixed quiet washout.

The soft knee earns its role only if its mean signal/distractor ratio exceeds the linear trace by at least 10%, its mean onset latency is no more than one step worse, and its mean washout residual is no more than 10% worse. Otherwise classify `KNEE_NOT_NEEDED`.

- [ ] **Step 7: Implement CLI**

`experiments/run_v0.py` accepts `--seeds` and `--out`, calls `run_v0`, prints a compact summary, and writes sorted/indented JSON.

- [ ] **Step 8: Run GREEN**

Run: `pytest tests/test_experiment.py -q`

Expected: 3 passed.

Then run: `python experiments/run_v0.py --seeds 64 --out /tmp/simple-neuron-v0.json`

Expected: valid JSON receipt with all four gates and a knee classification.

- [ ] **Step 9: Commit**

Commit message: `feat: add deterministic SimpleNeuron v0 gates`

---

### Task 6: Freeze receipt, documentation, and CI

**Files:**
- Create: `results/v0.json`
- Create: `README.md`
- Create: `.github/workflows/ci.yml`
- Modify: `tests/test_experiment.py`

**Interfaces:**
- Frozen receipt is generated by `python experiments/run_v0.py --seeds 64 --out results/v0.json`.
- CI runs unit tests and regenerates a temporary v0 receipt on Python 3.11 and 3.12.

- [ ] **Step 1: Add failing frozen-receipt test**

Add to `tests/test_experiment.py`:

```python
import json
from pathlib import Path
from simple_neuron.experiment import run_v0


def test_frozen_v0_receipt_matches_canonical_run():
    frozen = json.loads(Path('results/v0.json').read_text())
    assert run_v0(seeds=64) == frozen
```

Run: `pytest tests/test_experiment.py::test_frozen_v0_receipt_matches_canonical_run -q`

Expected: FAIL because `results/v0.json` does not exist.

- [ ] **Step 2: Generate canonical receipt**

Run: `python experiments/run_v0.py --seeds 64 --out results/v0.json`

- [ ] **Step 3: Run frozen test GREEN**

Run: `pytest tests/test_experiment.py::test_frozen_v0_receipt_matches_canonical_run -q`

Expected: PASS.

- [ ] **Step 4: Write README from the frozen results**

README must contain:

- the one-sentence architecture: **keep rich state resident; sparse routed events steer it through locally learned receiver coordinates**;
- exact equations for `A`, `B`, soma, and AIS;
- Gate 0-3 table populated only from `results/v0.json`;
- explicit statement that axonal topology is fixed in v0;
- explicit Ca-knee result, including `KNEE_NOT_NEEDED` if that is what the receipt says;
- biology fence and links to `AnttisNeuron`, `GrowingAnttisNeuron`, `NewMachine`, `FusionMachine`, and `AnotherOddThing` as motivation rather than validation.

- [ ] **Step 5: Add CI**

Matrix Python 3.11/3.12. Steps:

```yaml
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
  with:
    python-version: ${{ matrix.python-version }}
- run: python -m pip install -e '.[test]'
- run: pytest -q
- run: python experiments/run_v0.py --seeds 8 --out /tmp/v0-smoke.json
```

- [ ] **Step 6: Full verification**

Run locally:

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/run_v0.py --seeds 64 --out /tmp/v0-verify.json
python - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('/tmp/v0-verify.json').read_text()) == json.loads(Path('results/v0.json').read_text())
print('receipt match')
PY
```

Expected: all tests pass and `receipt match` prints.

- [ ] **Step 7: Commit**

Commit message: `docs: freeze SimpleNeuron v0 result`

---

## Plan self-review

- Spec coverage: resident state, local Oja learning, soma/AIS separation, sparse delayed routing, direct/linear/knee ablation, deterministic scientific receipts, and biology fence are each assigned to tasks.
- No route-learning task is included because the spec explicitly defers axonal topology learning.
- No production implementation precedes its failing test.
- The same `B` convention is used throughout: shape `(state_dim, n_ports)`, columns are receiver steering directions.
- The knee's scientific classification is allowed to fail positively; CI tests determinism/schema rather than forcing `KNEE_EARNS_ROLE`.
