[← Return to Main Index](../index.md)

# Abstract

Quantum cryptography offers security guarantees based on physical laws rather than computational assumptions. The Noisy Storage Model (NSM) provides a framework for implementing cryptographic primitives under realistic physical constraints—specifically, that quantum storage undergoes decoherence. This report presents **Caligo**, a simulation-native implementation of the $\binom{2}{1}$-Oblivious Transfer (OT) protocol secured by the NSM.

Oblivious Transfer is a fundamental cryptographic primitive where a sender (Alice) holds two secret strings $(S_0, S_1)$, and a receiver (Bob) with choice bit $C \in \{0,1\}$ obtains $S_C$ without Alice learning $C$, and without Bob learning $S_{1-C}$. Traditional implementations rely on computational hardness assumptions. The NSM-based approach, first proposed by Wehner et al. [1], exploits quantum storage noise to achieve information-theoretic security.

**Caligo** (Latin: "fog/mist"—evoking the obscured nature of oblivious transfer) is a ground-up implementation designed with native integration to the **SquidASM/NetSquid** discrete-event quantum network simulation framework. The protocol operates in four phases:

1. **Phase I (Quantum Layer)**: EPR pair generation and BB84-style measurements
2. **Phase II (Sifting)**: Basis sifting and Quantum Bit Error Rate (QBER) estimation
3. **Phase III (Reconciliation)**: Rate-compatible LDPC error correction
4. **Phase IV (Amplification)**: Toeplitz hashing for secure key extraction

Key innovations include:

- **Hybrid Puncturing Strategy**: A two-regime approach combining untainted puncturing (stopping-set protection) with ACE/EMD-guided intentional puncturing for finite-length rate-compatible LDPC codes [2,3]
- **Simulation-Native Architecture**: Domain-driven design aligned with NetSquid's discrete-event model, explicit timing enforcement, and configurable noise models
- **Rigorous Leakage Accounting**: Circuit-breaker patterns enforcing NSM constraints at architectural boundaries
- **Parallel EPR Generation**: Batching strategies optimized for quantum network simulators

This report provides a comprehensive technical account of Caligo's theoretical foundations, architectural design, implementation details, and validation results, demonstrating a complete path from NSM security parameters to operational quantum network simulation.

---

**Keywords**: Oblivious Transfer, Noisy Storage Model, Quantum Cryptography, SquidASM, LDPC Codes, Privacy Amplification

---

[← Return to Main Index](../index.md) | [Next: Introduction →](./introduction.md)
