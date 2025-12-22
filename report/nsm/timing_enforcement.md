[← Return to Main Index](../index.md)

# 8.4 Timing Enforcement: The Causality Constraint

## The Temporal Security Requirement

### Physical Foundation

The Noisy Storage Model's security rests on a **temporal assumption**: the adversary must store qubits for a minimum duration $\Delta t$ before learning the measurement bases. During this storage period, decoherence accumulates, degrading the adversary's quantum information.

Unlike channel noise (enforced automatically by transmission imperfections), **storage noise requires active timing enforcement**. Without it, a dishonest Bob can measure immediately upon reception, bypassing storage entirely.

### Protocol Timing (Wehner et al. [1])

The original NSM-OT protocol specifies:

> **Protocol Step 3**: "At time $t = T$, Alice sends $I_{\bar{C}}, I_C, F_{\bar{C}}, F_C$ to Bob."

where $T = \Delta t$ is measured from when Bob receives the last qubit ($t = 0$).

**Security assumption**: During $[0, \Delta t]$, any qubits stored by a dishonest Bob undergo depolarization $\mathcal{N}_r$. The protocol **must not proceed** until $\Delta t$ has elapsed.

---

## Markovian Noise and the Wait Time

### The Semigroup Property

König et al. [2] proved security assuming **Markovian** storage noise:

$$
\mathcal{F}_{t_1 + t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}
$$

**Physical interpretation**: Decoherence is **memoryless**—noise accumulated from $t$ to $t + dt$ depends only on the current state, not history.

For the depolarizing channel:

$$
r(t_1 + t_2) = r(t_1) \cdot r(t_2) = e^{-\Gamma(t_1 + t_2)}
$$

### Why Timing Matters

Consider two scenarios:

**Scenario A** (Honest wait): Bob stores qubits, waits $\Delta t$, then measures after receiving bases.
- Storage fidelity: $r = e^{-\Gamma \Delta t}$
- Adversary's advantage: Bounded by NSM security proof

**Scenario B** (Immediate measurement): Bob measures immediately upon reception, before $\Delta t$.
- Storage fidelity: $r = 1$ (no storage)
- Adversary's advantage: Unbounded (defeats NSM security)

**The timing barrier** prevents Scenario B by withholding classical information until $\Delta t$ elapses.

---

## Discrete-Event Simulation Challenges

### NetSquid Event Timeline

SquidASM (built on NetSquid) uses **discrete-event simulation** (DES): time advances only when events are processed.

```
Event Timeline:
  t = 0 ns:      Alice generates EPR pair
  t = 50 ns:     Photon arrives at Bob (fiber delay)
  t = 50 ns:     Bob measures in random basis
  t = 100 ns:    Bob stores result in classical memory
  t = ??? ns:    Alice reveals bases (MUST BE ≥ Δt!)
```

**Problem**: Without explicit enforcement, Alice's basis revelation occurs at the next available event (e.g., $t = 101$ ns), **violating** the $\Delta t = 10^6$ ns requirement.

### Why Naive Approaches Fail

**Python `time.sleep()`**: Blocks the Python interpreter, not the NetSquid simulation clock. The simulation timeline advances only when events are processed—real-world delays are irrelevant.

**Busy-wait loop**: NetSquid's event engine is cooperative; it progresses only when `sim_run()` is called or events are yielded. A busy-wait prevents event processing, deadlocking the simulation.

### Correct Approach: Event-Based Barrier

Insert a **virtual event** at $t = t_{\text{measure}} + \Delta t$ that must be processed before proceeding:

```
yield from wait_for_event(duration=delta_t_ns)
```

**Effect**: The simulation timeline blocks until $\Delta t$ has elapsed in simulation time, then resumes. This is a **causal barrier**—no events can fire before the barrier is lifted.

---

## Formal Specification

### State Machine

The timing barrier implements a three-state machine:

$$
\text{IDLE} \xrightarrow{\text{mark\_quantum\_complete()}} \text{WAITING} \xrightarrow{\text{wait\_delta\_t()}} \text{READY} \xrightarrow{\text{reset()}} \text{IDLE}
$$

**State transitions**:
1. **IDLE → WAITING**: Called after Bob completes quantum measurements; records $t_{\text{start}}$
2. **WAITING → READY**: Simulation time advances by $\Delta t$
3. **READY → IDLE**: Reset for next protocol iteration

### Invariants

At all times:

$$
\text{(state = READY)} \implies (\text{sim\_time()} \geq t_{\text{start}} + \Delta t)
$$

$$
\text{(basis\_revelation)} \implies (\text{state = READY})
$$

### Generator Protocol

The barrier is implemented as a Python generator yielding NetSquid events:

```python
def wait_delta_t(self) -> Generator:
    """Block until Δt has elapsed in simulation time."""
    assert self.state == TimingBarrierState.WAITING
    
    remaining = (self.start_time + self.delta_t_ns) - sim_time()
    if remaining > 0:
        yield from wait(duration=remaining)
    
    self.state = TimingBarrierState.READY
```

---

## Physical Justification

### Experimental Wait Times

Erven et al. [3] used $\Delta t = 1$ ms in their experimental demonstration.

**Justification**: Room-temperature quantum memories exhibit decoherence times $T_2 \sim 100$ μs to 1 ms. A 1 ms wait ensures:

$$
r(1 \text{ ms}) = e^{-\Delta t / T_2} \approx e^{-10} \approx 4.5 \times 10^{-5} \text{ (for } T_2 = 100 \text{ μs)}
$$

or

$$
r(1 \text{ ms}) \approx 0.75 \text{ (for } T_2 \approx 3.5 \text{ ms)}
$$

depending on the storage medium.

### Practical Enforcement

In a real-world implementation, Alice enforces $\Delta t$ by:

1. Recording $t_{\text{quantum\_end}}$ when all qubits are transmitted
2. Computing $t_{\text{reveal}} = t_{\text{quantum\_end}} + \Delta t$
3. Withholding basis information until $t_{\text{reveal}}$
4. Using authenticated channels to prevent early revelation

Bob cannot force Alice to reveal bases early (cryptographic authentication ensures message integrity and timing).

---

## Critique: Assumption Validity

### Non-Markovian Effects

The Markovian assumption may fail for:

1. **Structured environments**: Coupling to discrete bath modes
2. **Low-temperature systems**: Long bath correlation times
3. **Strong coupling**: Non-perturbative system-bath interaction

**Consequence**: If $\mathcal{F}_{t_1 + t_2} \neq \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}$, the security proof may not apply.

### Finite Wait Time Effects

For finite $\Delta t$, the adversary might:

1. **Partial measurement**: Measure some qubits immediately, store others
2. **Adaptive storage**: Store only qubits where early measurement suggests value
3. **Collective strategies**: Entangle stored qubits with ancilla systems

König et al. [2] addressed (1) and (2) in their security proof; (3) remains an active research area.

### Simulation Limitations

The SquidASM simulation enforces $\Delta t$ as an **idealized causal barrier**. Real implementations face:

- Clock synchronization errors
- Network latency variability
- Side-channel timing leakage

These are not modeled in the current simulation.

---

## Timing Parameter Selection

### Tradeoffs

| Larger $\Delta t$ | Smaller $\Delta t$ |
|-------------------|--------------------|
| More storage noise (better security) | Less storage noise (weaker security) |
| Lower protocol throughput | Higher protocol throughput |
| More susceptible to clock drift | Less susceptible to clock drift |

### Recommended Values

| Application | $\Delta t$ | Rationale |
|-------------|------------|-----------|
| High security | 10 ms | Ensures $r < 0.1$ for most storage |
| Balanced | 1 ms | Erven et al. experimental regime |
| High throughput | 100 μs | Marginal security, high rate |

---

## References

[1] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.*, vol. 100, 220502, 2008.

[2] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory*, vol. 58, no. 3, pp. 1962–1984, 2012.

[3] C. Erven et al., "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model," *Nat. Commun.*, vol. 5, 3418, 2014.

---

[← Return to Main Index](../index.md) | [← Previous: Noise Models](./noise_models.md)
