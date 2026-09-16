# SimpleNeuron

> **Keep rich state resident. Let sparse routed events steer it through locally learned receiver coordinates.**

SimpleNeuron is a deliberately small research machine extracted from the recent `AnttisNeuron` / `GrowingAnttisNeuron` / `NewMachine` / `FusionMachine` / `AnotherOddThing` thread.

It is **not** a claim that biological neurons literally run these equations. The biological picture motivated a computational question that can fail cleanly:

> If a receiver already contains a rich continuously evolving state, can a tiny routed event learn to mean “push that state this way” without backpropagating a global loss?

v0 said **yes in a controlled synthetic setting** for local route meaning, and gave a Ca-like soft knee one narrowly defined job: detect temporally clustered pings better than a matched linear trace. v1 adds the missing state test: does earlier routed history remain resident long enough to change what a later identical cue does?

## The machine

Each unit owns resident dendritic state

```text
x[t+1] = A @ x[t] + sensory[t] + B @ q[t]
```

where:

- `A` is a stable local state-space operator;
- `B[:, k]` is the steering direction associated with receiving port `k`;
- `q[t]` is the effective routed input after an optional temporal transducer.

The dendritic core is linear on purpose. Simultaneous inputs therefore interpolate naturally in state space rather than being forced into a winner-take-all code.

The soma reads the current resident state:

```text
s[t] = m @ x[t]
```

and the AIS-like boundary publishes only when that local mixture crosses a threshold:

```text
spike[t] = 1[s[t] > theta]
```

Suppressing publication does **not** erase the resident state.

A sparse axonal route is just

```text
(source neuron, target neuron, target port, integer delay, weight)
```

so the event payload can remain one bit. Meaning comes from the combination of **source identity + route + landing port + learned receiver steering matrix + receiver's current state**.

## Local learning: a route learns what direction it means

Each receiving port updates only from its own event and the receiver's own local state:

```text
y_k = B[:, k] dot x
B[:, k] += eta * event_k * y_k * (x - y_k * B[:, k])
normalize(B[:, k])
```

This is route-conditioned Oja learning. There is no task-loss optimizer and no backpropagation through the machine.

The interpretation is simple:

```text
this route keeps arriving while my local state looks like X
                    ↓
this port gradually becomes a steering direction toward X
                    ↓
later, the same tiny routed event can push the resident state toward X
```

## Does the Ca-like knee belong here?

v0 does **not** assume it does.

The receiving side compares three transducers with the same immediate gain for one isolated unit pulse:

```text
direct
    event -> dendrite

linear trace
    event -> leaky local trace -> dendrite

soft knee
    event -> same leaky trace -> smooth threshold/gain -> dendrite
```

The soft knee is allowed to stay only if it adds useful temporal selectivity beyond the linear trace. Its job is therefore not “calcium magic”; it is specifically **clustered-event sensitivity**.

## Frozen results

Frozen receipts: [`results/v0.json`](results/v0.json) and [`results/v1.json`](results/v1.json), each with 64 deterministic seeds. v0 remains unchanged; v1 adds Gate 4.

| Gate | Result |
|---|---:|
| 0 — invariants | `PASS_INVARIANTS` |
| 1 — local port steering | `PASS_LOCAL_STEERING` |
| 2 — one-bit route semantics | `PASS_ROUTE_SEMANTICS` |
| 3 — Ca-like temporal knee | `KNEE_EARNS_ROLE` |
| 4 — living resident state | `PASS_LIVING_STATE` |

### Gate 0 — resident state really is resident

- operator spectral radius: **0.900000**
- silent one-step decay ratio: **0.800000**
- simultaneous-port interpolation error: **0.000000**
- suppressed publication leaves state norm: **1.000000** while spike = **0**

This is an engineering invariant gate, not a scientific result.

### Gate 1 — local route-conditioned learning

Three receiving ports were each paired with a different local state direction. Only local Oja updates were allowed.

Across 64 seeds:

- paired learned alignment: **0.999187**
- event/state-shuffled alignment: **0.223379**
- paired minus shuffled: **+0.775808**
- worst paired seed: **0.998569**

So the receiver can locally learn what state direction historically accompanies a route.

### Gate 2 — the route carries semantics beyond the bit

Two source units emit the **same binary payload**. Only their target port differs.

Across 64 seeds:

- intact learned-port alignment: **0.999188**
- target-port shuffle: **0.018331**
- random unlearned `B`: **0.443779**
- intact minus port shuffle: **+0.980857**

The event itself does not describe the downstream state change. The route selects a locally learned coordinate in an already resident receiver.

### Gate 3 — the soft knee earns one narrow job

The event tape contains legitimate clustered pings and isolated distractor pings. A single isolated unit pulse has matched immediate gain in all three conditions.

Mean legitimate/distractor steering-energy ratio:

- direct: **2.0000**
- linear trace: **3.1975**
- soft knee: **90.8641**

Soft-knee / linear ratio gain: **28.4171×**.

Mean onset latency is **0 steps** for all three conditions. Mean washout residual is **0.007635** for the linear trace and **0.000012** for the soft knee.

This is a **constructed mechanism witness**. The knee is explicitly designed to convert temporal clustering into nonlinear gain, so this does *not* establish that calcium-like nonlinearities are generally better, biologically required, or useful on arbitrary tasks. It establishes only that such a local knee has a concrete computational role that a matched linear trace does not reproduce on this gate.

### Gate 4 — the receiver's past changes a later identical cue

Gate 4 keeps the learned receiving matrix `B` from v0 but finally gives the receiver a nonzero life before the cue. A context event lands first, the state evolves silently for three steps under `A = 0.97 I`, and only then does the same cue arrive. One context should make the soma/AIS publish; an alternative context should not.

Across 64 seeds:

- living receiver context accuracy: **1.0000**
- stateless attacker accuracy: **0.5000**
- living minus stateless: **+0.5000**
- retained context norm after the silent delay: **0.912673** of the original
- final-state separation under the **same later cue**: **1.256281**
- simultaneous-drive interpolation error: **2.29e-16** maximum

The attacker is deliberately given the **same learned `B` and the same soma**. It loses only the earlier resident state and sees the current cue. Because its current input is identical in both histories, it cannot distinguish the two contexts.

This establishes a narrow but important point: **route semantics alone are not the whole computation. Resident history can change the consequence of the same later event.** In the current linear dendrite, the instantaneous steering vector `B[:, k]` itself is still additive and state-independent; the state dependence appears in the resulting full state and at the soma/AIS publication boundary. So Gate 4 does not yet establish a nonlinear attractor or state-dependent dendritic gain. It establishes that keeping state alive adds a computational degree of freedom that a reset/stateless lookup lacks.

## What v1 is — and is not

The current object is roughly:

```text
resident dynamical state
        ↑
locally learned receiving coordinates
        ↑
sparse delayed routed pings
        ↑
AIS-like publish boundary
        ↑
other resident dynamical states
```

The rich thing mostly stays resident. The small thing travels.

v1 still does **not** learn axonal topology. The route graph is fixed so that we can isolate the receiver-side claim first. It also has no structural growth, active dendritic channels, reward, global optimizer, spike waveform model, or biological calibration.

## Lineage

The computational ingredients came from several earlier projects:

- [`AnttisNeuron`](https://github.com/anttiluode/AnttisNeuron) — dendritic modes, local Oja specialization, soma/AIS separation.
- [`GrowingAnttisNeuron`](https://github.com/anttiluode/GrowingAnttisNeuron) — input address versus physical receiver operator.
- [`NewMachine`](https://github.com/anttiluode/NewMachine) — keep resident state distinct from publication.
- [`FusionMachine`](https://github.com/anttiluode/FusionMachine) — tiny event payload plus route identity, with resident computation at the destination.
- [`AnotherOddThing`](https://github.com/anttiluode/AnotherOddThing) — local traces, active perturbation, and the idea that a message can poke computation already living at the receiver.

Those repositories motivate this architecture. They do not validate a biological interpretation of SimpleNeuron.

## Run it

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/run_v0.py --seeds 64 --out results/v0.json
python experiments/run_v1.py --seeds 64 --out results/v1.json
```

The frozen receipts are regression-tested structurally, with tight numerical tolerance for cross-platform floating-point drift. CI runs the suite on Python 3.11 and 3.12 and executes a smaller deterministic v1 scientific smoke run.

## Next honest gate

With resident history now earning a role in Gate 4, the next clean question is:

> Can strictly local pre/post timing learn **where** a sparse event should be routed, while the receiver independently learns **what that route means** in its own state space?

That would join the two slow variables we deliberately separated here:

```text
learn the receiving coordinate
            +
learn which axon reaches it
```
