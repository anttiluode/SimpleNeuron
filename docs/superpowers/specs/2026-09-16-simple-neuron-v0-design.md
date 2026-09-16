# SimpleNeuron v0 Design

## Purpose

Build the smallest falsifiable version of the current neuron-inspired architecture:

- rich state remains resident in the receiver;
- incoming axonal events are sparse and mostly steer that resident state rather than carry a full representation;
- the meaning of an incoming route lives partly in the receiving port's learned steering vector;
- dendritic state evolves continuously through a stable local operator;
- soma/AIS converts resident state into sparse publish events;
- axonal structure is an explicit sparse routing graph;
- all learning in the neuron-like machine is local. No backpropagation is used to train the machine.

The design is computational first. Biological vocabulary is motivational and does not imply that the equations are a literal neuron model.

## Core state-space primitive

Each receiver owns a resident dendritic state

```text
x in R^d
```

with a stable recurrent operator `A` and a matrix `B` whose columns are receiving-port steering vectors:

```text
x[t+1] = A @ x[t] + E @ sensory[t] + B @ q[t]
```

where:

- `A` is fixed in v0 and normalized to spectral radius < 1;
- `E @ sensory[t]` is an optional external/world drive used to establish a local state during learning and evaluation;
- `q[t]` is the vector of effective incoming routed events after the optional receiver-side temporal transducer;
- `B[:, k]` says what direction an event arriving at port `k` pushes the local resident state.

The linear dendritic core is deliberate. It keeps modal/state-space interpretation exact and makes simultaneous inputs interpolate naturally. Nonlinearity is applied later at the soma/AIS boundary rather than hidden inside the first gate.

## Local port learning

Each receiving port learns the state direction historically associated with that route using route-conditioned Oja learning. For active port `k`:

```text
y_k = dot(B[:, k], x)
B[:, k] += eta * event_k * y_k * (x - y_k * B[:, k])
normalize(B[:, k])
```

The update uses only:

- the presynaptic event on that local port;
- the receiver's local resident state;
- the port's own steering vector.

No global target, task loss, or backpropagated gradient is available to the rule.

The first scientific claim is therefore precise:

> A receiving port can locally learn a steering direction from the local states that historically co-occur with events on that route, and a later event on that route can steer the resident state toward that learned direction.

A shuffled event/state-pairing control tests whether apparent alignment comes from the state distribution alone.

## Soma and AIS

The soma is a readout/mixer of resident state:

```text
s[t] = m dot x[t]
```

The AIS publishes a sparse binary event:

```text
spike[t] = 1[s[t] > theta]
```

`m` and `theta` are fixed in v0. The point of this gate is architectural separation: resident state can continue evolving even when nothing is published.

## Axonal routing

A network contains an explicit sparse route tensor mapping source spikes to receiver ports with integer delays:

```text
(source neuron, target neuron, target port, delay, weight)
```

A source spike does not carry a rich vector. The route identity, arrival time, target port, target steering matrix `B`, and target current state determine its effect.

Axonal topology is fixed in v0. Learning the routing graph is intentionally deferred so that v0 can isolate whether the receiving-port primitive earns its place.

## Receiver-side temporal transducer and the Ca-like knee

The Ca-like stage is optional and must earn itself. It sits between arriving axonal events and the dendritic steering matrix.

Three matched conditions are required:

1. `direct`: `q[t] = event[t]`;
2. `linear_trace`: `c[t+1] = lambda*c[t] + event[t]`, `q[t] = c[t]`;
3. `soft_knee`: the same leaky trace followed by a smooth local knee.

The soft-knee output is normalized so an isolated unit pulse has the same immediate gain as the direct/linear conditions. Its only intended capability is temporal selectivity: clustered pings may cross into a higher-gain regime while isolated distractors decay below it.

The knee is not retained merely because it is biologically suggestive. It earns a role only if it improves burst-vs-isolated-event discrimination beyond the matched linear trace without unacceptable latency or instability. Otherwise v0 classifies it as `KNEE_NOT_NEEDED` and the simpler direct/linear mechanism remains the architecture.

## Frozen gates

### Gate 0 — resident-state and interpolation invariants

Verify deterministically that:

- `A` is stable;
- silence decays a displaced state;
- two simultaneous routed events combine as the sum of their steering vectors before the soma threshold;
- suppressing publication does not erase resident state.

This is an engineering gate, not a scientific win.

### Gate 1 — route-conditioned local steering learning

Construct multiple recurring local state directions. Each input port is active only while one direction is externally established. Train `B` using only the local Oja update, freeze it, remove the external drive, and probe each port.

Measure:

- cosine alignment between learned `B[:, k]` and the state direction paired with port `k`;
- steering accuracy after learning;
- the same metrics under an event/state-pairing shuffle.

Predeclared success rule: the paired condition must show both high absolute alignment and a clear positive margin over the shuffled control across deterministic seeds. CI records the result and does not tune the threshold after seeing it.

### Gate 2 — sparse routed network

Connect several resident units with a fixed sparse delayed route graph. Train receiver port semantics locally as in Gate 1. Then remove some external drive and test whether identical binary source events cause different downstream state movements solely because they travel through different routes into different learned ports.

Controls:

- route identity intact;
- target-port shuffle;
- unlearned random `B`;
- dense vector-message oracle as a ceiling, not as a matched architecture.

The claim is only that route identity plus resident receiver structure can carry useful information beyond a one-bit payload.

### Gate 3 — does the Ca-like knee earn a role?

Use a fixed event tape containing legitimate clustered pings and isolated distractors. Compare `direct`, `linear_trace`, and `soft_knee` with identical route weights and single-pulse gain.

Measure:

- legitimate/distractor steering-energy ratio;
- correct downstream state-direction margin;
- onset latency;
- stability after the event train ends.

Classification:

- `KNEE_EARNS_ROLE` only if the soft knee improves useful discrimination beyond the linear trace without a material stability or latency cost;
- otherwise `KNEE_NOT_NEEDED`.

CI must accept either scientific outcome.

## Implementation boundaries

The first version is pure Python + NumPy + pytest. No PyTorch, no autograd, no optimizer package, and no hidden learned global loss.

Proposed modules:

```text
src/simple_neuron/core.py       resident state, soma/AIS, stepping
src/simple_neuron/learning.py   local Oja port update
src/simple_neuron/transducer.py direct/linear/soft-knee receiver stages
src/simple_neuron/network.py    sparse delayed axonal routing
experiments/run_v0.py           deterministic Gates 0-3 receipt
results/v0.json                 frozen canonical receipt
tests/                          mechanism and invariant tests
```

## Out of scope for v0

- learning axonal topology;
- branch-local active-channel nonlinearities;
- spike waveform models;
- structural growth;
- reward-modulated global learning;
- backpropagation through the neuron-like machine;
- claims that the system is a literal biological neuron model;
- claims that continuous cognitive manifolds are explained by this toy.

## Interpretation rule

The architectural hypothesis is:

> Keep rich state resident. Let sparse routed events steer it through locally learned receiving coordinates. Publish sparsely. Treat route identity and receiver state as part of the message semantics.

Every extra mechanism, especially the Ca-like knee, must beat a simpler destructive control before it becomes part of the claimed machine.
