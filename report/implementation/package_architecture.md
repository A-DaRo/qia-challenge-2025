[← Return to Main Index](../index.md)

# Appendix A: Simulation Framework Architecture

## Scope

This appendix documents the software architecture of the Caligo simulation environment. The emphasis is on how the computational structure maps onto the physical and information-theoretic layers of the NSM Oblivious Transfer protocol—not on programming conventions or API specifications.

---

## Architectural Philosophy

### Simulation as Physical Apparatus

The simulation framework is conceptualized as a *computational laboratory apparatus* that:

1. **Prepares** quantum states according to specified density matrices
2. **Evolves** states through CPTP channels with parametrized noise
3. **Measures** observables in specified bases
4. **Processes** classical post-measurement data through information-theoretic primitives

Each software component corresponds to a well-defined physical or mathematical operation.

### Layered Abstraction

The architecture employs three abstraction layers corresponding to the physical hierarchy:

| Layer | Physical Correspondence | Responsibility |
|-------|------------------------|----------------|
| **Quantum** | EPR generation, storage, measurement | State preparation, CPTP evolution, POVM realization |
| **Information** | Sifting, reconciliation, amplification | Syndrome extraction, error correction, entropy extraction |
| **Protocol** | NSM OT execution, security parameter enforcement | Timing control, abort conditions, key output |

---

## Component Taxonomy

### Quantum Simulation Layer

The quantum layer interfaces with the NetSquid discrete-event simulator:

**EPR Source Model**:
- Parameterized by fidelity $F$ and generation rate $\lambda$
- Outputs Werner states $\rho_F = F|\Phi^+\rangle\langle\Phi^+| + (1-F)\mathbb{I}_4/4$
- Sequential or statistical generation modes

**Memory Model**:
- T1-T2 decoherence: Lindblad evolution with $(\gamma_1, \gamma_\phi)$
- Depolarizing approximation: Effective parameter $r(t) = e^{-\Gamma t}$
- Configurable time-dependent vs. time-independent noise

**Measurement Model**:
- BB84 basis selection: Random choice from $\{Z, X\}$
- POVM realization: Projective measurements $\{|0\rangle\langle 0|, |1\rangle\langle 1|\}$ or $\{|+\rangle\langle +|, |-\rangle\langle -|\}$

### Information Processing Layer

**Sifting Module**:
- Input: Raw measurement records $(b_A^i, x_A^i, b_B^i, x_B^i)$
- Operation: Basis comparison, matching-basis retention
- Output: Sifted key strings $(X_A, X_B)$ with length $\approx n/2$

**Reconciliation Module**:
- Input: Sifted keys, QBER estimate $\hat{Q}$
- Operation: LDPC syndrome computation, belief propagation decoding
- Output: Reconciled key $X_R$ with residual error $\epsilon_{\text{rec}} < 10^{-6}$

**Amplification Module**:
- Input: Reconciled key $X_R$, min-entropy estimate $k$
- Operation: Toeplitz matrix multiplication
- Output: Final key $K$ of length $\ell = k - 2\log_2(1/\varepsilon)$

### Protocol Coordination Layer

**State Machine**:
- Tracks protocol phase: INIT → QUANTUM → SIFT → RECONCILE → AMPLIFY → OUTPUT
- Enforces abort conditions: QBER threshold, minimum entropy, timing bounds
- Manages classical communication rounds

**Security Parameter Enforcement**:
- $\varepsilon_{\text{sec}}$: Security parameter for LHL
- $Q_{\text{max}}$: QBER abort threshold (default 11%)
- $\Delta t_{\text{min}}$: Minimum storage delay for NSM guarantee

---

## Data Flow Model

The simulation follows a pipeline architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Quantum Layer                                │
│  EPR Source ─→ Channel ─→ Memory ─→ Measurement                     │
│     (ρ_F)       (𝒩_loss)    (𝒩_r)     (POVM)                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼ (b_i, x_i) measurement records
┌─────────────────────────────────────────────────────────────────────┐
│                    Information Layer                                │
│  Sifting ─→ QBER Est. ─→ Reconciliation ─→ Privacy Amp.             │
│   (n/2)      (Q̂)           (syndrome)        (Toeplitz)             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼ K: final key
┌─────────────────────────────────────────────────────────────────────┐
│                     Protocol Layer                                  │
│  Security Check ─→ Key Output ─→ Verification                       │
│   (abort?)          (ℓ bits)       (optional)                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Semantics

### Parameter Categories

**Physical Parameters** (mapped to experimental apparatus):
- `source_fidelity`: $F$ in Werner state model
- `t1_time`, `t2_time`: Memory coherence times (ns)
- `storage_delay`: $\Delta t$ for NSM security

**Information Parameters** (tunable protocol choices):
- `code_rate`: $R$ for LDPC codes
- `num_raw_qubits`: $n$ initial EPR pairs
- `reconciliation_efficiency`: $f$ in $R = 1 - f \cdot h(Q)$

**Security Parameters** (non-negotiable bounds):
- `qber_threshold`: Maximum tolerated $Q$
- `security_epsilon`: $\varepsilon$ in LHL
- `abort_on_failure`: Protocol behavior on threshold violation

### Configuration Files

YAML-formatted configuration files specify complete experimental setups:

```yaml
# Conceptual structure (not actual syntax)
physical:
  source_fidelity: 0.97
  storage_noise_model: "depolarizing"
  storage_parameter_r: 0.85

information:
  reconciliation_method: "ldpc_bp"
  code_rate: 0.45
  num_raw_qubits: 4096

security:
  qber_threshold: 0.11
  security_epsilon: 1e-10
```

---

## Validation and Verification

### Unit Invariants

Each module enforces mathematical invariants:

- **Sifting**: $|X_{\text{sift}}| = \sum_i \mathbb{1}[b_A^i = b_B^i]$
- **Reconciliation**: $H(X_R) \cdot s = 0$ (syndrome satisfaction)
- **Amplification**: $\ell \leq \lfloor k - 2\log_2(1/\varepsilon) \rfloor$

### Integration Tests

End-to-end protocol executions verify:
- QBER consistency: $|\hat{Q}_{\text{measured}} - Q_{\text{theory}}| < \delta_Q$
- Key rate bounds: $R_{\text{key}} \leq R_{\text{Lupo}}$
- Abort correctness: Protocol terminates when $Q > Q_{\text{threshold}}$

---

## References

[1] T. Coopmans et al., "NetSquid, a NETwork Simulator for QUantum Information using Discrete events," *Commun. Phys.*, vol. 4, 164, 2021.

---

[← Return to Main Index](../index.md) | [Next: Module Specifications](./module_specs.md)
