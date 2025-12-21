[← Return to Main Index](../index.md)

# 1.2 Problem Scope & Objectives

## 1.2.1 Problem Statement

Implementing cryptographic protocols in quantum network simulators presents unique challenges that differ fundamentally from both abstract protocol design and physical hardware deployment. This project addresses the following core problem:

> **How can we implement a $\binom{2}{1}$-Oblivious Transfer protocol secured by the Noisy Storage Model within a discrete-event quantum network simulator, ensuring faithful representation of physical constraints, rigorous security parameter enforcement, and modular, maintainable architecture?**

This problem decomposes into three interconnected challenges:

### Challenge 1: Bridging Theory and Simulation

**Theoretical Framework**: The NSM security proof [1] operates on abstract parameters—storage rate $\nu$, depolarizing parameter $r$, and waiting time $\Delta t$. Security follows from the inequality $C_{\mathcal{N}} \cdot \nu < 1/2$.

**Simulation Reality**: SquidASM/NetSquid [14,15] models quantum networks using:
- NetSquid's discrete-event kernel with explicit timing
- Quantum gate noise models (depolarizing, dephasing, amplitude damping)
- Network topology with link attenuation and detector inefficiency
- Qubit measurement with finite detector efficiency and dark counts

**Challenge**: Establish a rigorous mapping from NSM parameters $(r, \nu, \Delta t)$ to SquidASM configuration parameters (gate fidelities, link losses, detector efficiency) such that security conditions are verifiable within simulation.

### Challenge 2: Rate-Compatible Reconciliation Under NSM Constraints

**Standard QKD Reconciliation**: In Quantum Key Distribution, syndrome information leaks to a passive eavesdropper (Eve). Reconciliation efficiency $f = \text{leak}_{\text{EC}} / (n \cdot h(\text{QBER}))$ determines secure key rate, with typical targets $f \in [1.05, 1.2]$ [2,12].

**NSM-OT Reconciliation**: In oblivious transfer under NSM, syndrome information leaks **directly to Bob**, who is a potential adversary. The extractable secure key length is:

$$
\ell \leq n \cdot \left[ H_{\min}^{\epsilon}(X|E) - \text{leak}_{\text{EC}} - \log_2\left(\frac{2}{\epsilon^2}\right) \right]
$$

where $\text{leak}_{\text{EC}} = |\Sigma| + |\text{Hash}| + |\text{Revealed}|$ must be **strictly bounded**.

**Challenge**: Implement rate-compatible LDPC reconciliation [2,3] that:
1. Adapts to varying QBER without pre-estimation (blind mode)
2. Minimizes syndrome leakage via efficient puncturing strategies
3. Operates at finite block length ($n = 4096$) with verifiable error correction success
4. Supports both baseline (QBER-aware) and blind (QBER-adaptive) strategies

### Challenge 3: Architectural Discipline in Complex Simulation Environments

**Challenge**: Design a **domain-driven architecture** that:
1. Aligns package structure with protocol phases (not software patterns)
2. Enforces Single Responsibility Principle
3. Uses explicit phase contracts (dataclasses) for inter-phase communication
4. Separates simulation concerns (timing, noise) from protocol logic
5. Enables independent testing of protocol components

## 1.2.2 Project Objectives

Given these challenges, the Caligo project defines the following objectives:

### Objective 1: Simulation-Native Protocol Implementation

**Goal**: Implement the four-phase protocol with native SquidASM integration.

**Success Criteria**:
- EPR pair generation using SquidASM's `create_epr` primitives
- Explicit timing barriers enforcing NSM waiting time $\Delta t$
- Configurable noise models (depolarizing, link loss, detector inefficiency)
- Discrete-event execution with reproducible simulation logs
- Support for both sequential and parallel EPR generation strategies

**Deliverables**:
- `caligo.quantum`: EPR generation with batching strategies
- `caligo.sifting`: Basis sifting and QBER estimation
- `caligo.reconciliation`: Rate-compatible LDPC reconciliation
- `caligo.amplification`: Toeplitz privacy amplification

### Objective 2: Hybrid Rate-Compatible Reconciliation

**Goal**: Implement both baseline and blind reconciliation strategies using a unified rate-compatible LDPC framework.

**Success Criteria**:
- Mother code $\mathcal{C}_{R_0=0.5}$ constructed via ACE-PEG algorithm [16]
- Hybrid puncturing: untainted regime ($R \leq 0.625$) + ACE-guided regime ($R > 0.625$)
- Fine-grained rate adaptation ($\Delta R = 0.01$) spanning $[0.5, 0.9]$
- Belief Propagation decoder with configurable iteration limits
- Rigorous leakage accounting: $\text{leak}_{\text{EC}} = (1-R_{\text{eff}}) \cdot n + |\text{hash}| + |\text{revealed}|$
- Circuit-breaker pattern: reconciliation aborts if $\text{leak}_{\text{EC}} > \text{leak}_{\text{budget}}$

**Deliverables**:
- `caligo.reconciliation.baseline`: QBER-aware strategy
- `caligo.reconciliation.blind`: QBER-adaptive strategy with bit revelation
- `caligo.reconciliation.ldpc_encoder`: Syndrome computation with puncturing
- `caligo.reconciliation.ldpc_decoder`: JIT-compiled BP decoder (Numba)
- `caligo.reconciliation.leakage_tracker`: NSM-aware leakage accounting

### Objective 3: NSM Parameter Enforcement

**Goal**: Establish verifiable mappings from NSM security parameters to SquidASM physical configurations.

**Success Criteria**:
- `NSMParameters` dataclass with validation: $Q_{\text{channel}} < Q_{\text{storage}}$
- Depolarizing noise calculation: $p_{\text{depolar}} = \frac{1-r}{4}$
- QBER decomposition: $Q_{\text{total}} = Q_{\text{channel}} + (1 - Q_{\text{channel}}) \cdot Q_{\text{storage}}$
- SquidASM injection: direct mapping to `network_config.yaml`
- Runtime timing barrier: `TimingBarrier.wait(delta_t_ns)`

**Deliverables**:
- `caligo.nsm_config`: NSM parameter validation and translation
- SquidASM configuration generator with noise model injection
- Timing enforcement primitives

### Objective 4: Modular, Testable Architecture

**Goal**: Design a codebase that adheres to software engineering best practices while remaining accessible to quantum information scientists.

**Success Criteria**:
- Single Responsibility Principle
- Explicit phase contracts: `dataclasses` for inter-phase data transfer
- Type hints throughout: `mypy --strict` compliance
- Numpydoc-formatted docstrings for all public APIs
- Test coverage ≥ 90% on protocol logic (excluding SquidASM integration)
- Logging via `LogManager` (no `print()` statements)

**Deliverables**:
- `caligo.types`: Domain primitives (`ObliviousKey`, `MeasurementRecord`, phase contracts)
- `caligo.utils`: Cross-cutting utilities (logging, entropy calculations, bitarray helpers)
- Comprehensive test suite with phase-contract validation

## 1.2.3 Non-Objectives (Scope Boundaries)

To maintain focus, the following are **explicitly out of scope**:

### Hardware Implementation
- Physical qubit control (ion traps, superconducting circuits, photonics)
- Real-time measurement feedback and adaptive basis selection
- Physical noise characterization and calibration

**Rationale**: Caligo is a simulation platform for protocol validation, not a hardware control stack.

### Multi-Party Extensions
- $\binom{n}{1}$-OT (1-out-of-n oblivious transfer)
- Multi-party computation protocols beyond two-party OT
- Network routing and multi-hop quantum communication

**Rationale**: The NSM security proof extends to multi-party settings [17], but implementation complexity exceeds project scope.

### Alternative Error Correction Codes
- Turbo codes, Polar codes, or Fountain codes for reconciliation
- Algebraic codes (BCH, Reed-Solomon) for structured puncturing

**Rationale**: LDPC codes with rate-compatible puncturing are well-established in QKD literature [2,12]. Alternative codes offer marginal gains at significant implementation cost.

### Statistical Analysis Automation
- Parameter sweep frameworks for $(r, \nu, \Delta t)$ exploration
- Automated plotting and visualization dashboards
- Monte Carlo simulation orchestration

**Rationale**: These are valuable tools but peripheral to core protocol implementation. They can be developed as external scripts leveraging Caligo's API.

---

## References

[1] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.* 100, 220502 (2008).

[2] D. Elkouss, J. Martinez-Mateo, D. Lancho, and V. Martin, "Rate Compatible Protocol for Information Reconciliation: An Application to QKD" (2010).

[3] D. Elkouss, J. Martinez-Mateo, and V. Martin, "Untainted Puncturing for Irregular Low-Density Parity-Check Codes," *IEEE Wireless Commun. Lett.* 1(6), 585-588 (2012).

[12] J. Martinez-Mateo, D. Elkouss, and V. Martin, "Blind Reconciliation," *Quantum Inf. Comput.* 12(9&10), 791-812 (2012).

[14] SquidASM Documentation, QuTech Delft. https://github.com/QuTech-Delft/squidasm

[15] T. Coopmans et al., "NetSquid, a NETwork Simulator for QUantum Information using Discrete events," *Commun. Phys.* 4, 164 (2021).

[16] T. Tian, C. Jones, J. D. Villasenor, and R. D. Wesel, "Construction of Irregular LDPC Codes with Low Error Floors," *IEEE ICC* (2003).

[17] R. König and B. M. Terhal, "The Bounded-Storage Model in the Presence of a Quantum Adversary," *IEEE Trans. Inf. Theory* 54(2), 749-762 (2008).

[18] E. Kiktenko et al., "Post-processing procedure for industrial quantum key distribution systems," *J. Phys.: Conf. Ser.* 741, 012081 (2016).

[19] M. Pompili et al., "Realization of a multinode quantum network of remote solid-state qubits," *Science* 372(6539), 259-264 (2021).

---

[← Return to Main Index](../index.md) | [← Previous: Introduction](./introduction.md) | [Next: Document Structure →](./structure.md)
