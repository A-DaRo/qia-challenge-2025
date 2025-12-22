# Caligo: Finite-Size Security in the Noisy Storage Model

**A Simulation Study of $\binom{2}{1}$-Oblivious Transfer**

**Authors:** Alessandro Da Ros, GitHub Copilot (AI Agent)  
**Date:** 2025  
**Framework:** SquidASM Quantum Network Simulator

---

## Abstract

This report presents a rigorous analysis of finite-size security bounds for 1-out-of-2 oblivious transfer in the Noisy Storage Model (NSM). Using the SquidASM discrete-event quantum network simulator, we verify theoretical predictions for extractable key length as a function of block length, quantum bit error rate, and storage noise parameters. The central finding is the characterization of **Death Valley**—the finite-size regime where protocol parameters that would yield positive key rates asymptotically fail due to statistical penalties from parameter estimation and privacy amplification.

→ [Read Full Abstract](./introduction/abstract.md)

---

## Table of Contents

### Part I: Foundations

**1. Introduction**
- 1.1 [Introduction: Secure Two-Party Computation](./introduction/introduction.md)
- 1.2 [Research Questions and Scope](./introduction/scope.md)

**2. Theoretical Foundations**
- 2.1 [The Noisy Storage Model](./foundations/nsm_model.md) — CPTP maps, depolarizing channel, security condition
- 2.2 [Cryptographic Primitives](./foundations/primitives.md) — ROT definition, WSE, min-entropy bounds
- 2.3 [Simulation Framework](./foundations/squidasm_framework.md) — SquidASM as numerical experiment
- 2.4 [Key Literature](./foundations/protocol_literature.md) — Wehner 2008, König 2012, Lupo 2023

### Part II: Protocol Specification

**3. Protocol Architecture**
- 3.1 [Protocol Specification](./architecture/protocol_overview.md) — Four-phase pipeline, timing diagram
- 3.2 [Security Model](./architecture/security_model.md) — Adversary capabilities, QBER thresholds
- 3.3 [Implementation Architecture](./architecture/domain_design.md) — Module structure, data flow

**4. Quantum Distribution**
- 4.1 [EPR Pair Generation](./quantum/epr_generation.md) — Bell states, Werner states, QBER-fidelity relation
- 4.2 [Generation Modes](./quantum/generation_modes.md) — Sequential vs. parallel
- 4.3 [Batching Strategies](./quantum/batching.md)
- 4.4 [Measurement Protocol](./quantum/measurement.md) — BB84 encoding, basis selection

**5. Sifting and Parameter Estimation**
- 5.1 [Basis Sifting](./sifting/basis_sifting.md) — Index partitioning, commitment scheme
- 5.2 [QBER Estimation](./sifting/qber_estimation.md) — Hoeffding bounds, confidence intervals
- 5.3 [Security Verification](./sifting/security_checks.md) — Threshold enforcement

### Part III: Post-Processing

**6. Information Reconciliation**
- 6.1 [LDPC Framework](./reconciliation/ldpc_framework.md) — Slepian-Wolf, syndrome coding, belief propagation
- 6.2 [Baseline Strategy](./reconciliation/baseline_strategy.md) — Fixed-rate reconciliation
- 6.3 [Blind Reconciliation](./reconciliation/blind_strategy.md) — Rate-adaptive protocol
- 6.4 [Hybrid Puncturing](./reconciliation/hybrid_puncturing.md) — Rate-compatible coding
- 6.5 [Leakage Accounting](./reconciliation/leakage_accounting.md) — Security impact

**7. Privacy Amplification**
- 7.1 [Toeplitz Hashing](./amplification/toeplitz_hashing.md) — Two-universal hash functions
- 7.2 [Extractable Length](./amplification/extractable_length.md) — Leftover hash lemma, key length formula
- 7.3 [Key Derivation](./amplification/key_derivation.md) — OT output construction

### Part IV: Physical Model Analysis

**8. NSM Parameter Analysis**
- 8.1 [Parameter Space](./nsm/parameter_space.md) — $(\Delta t, r, \nu)$ configuration
- 8.2 [Physical Mapping](./nsm/physical_mapping.md) — Decoherence times, storage platforms
- 8.3 [Noise Models](./nsm/noise_models.md) — Depolarizing, dephasing, amplitude damping
- 8.4 [Timing Enforcement](./nsm/timing_enforcement.md) — Markovian assumption validity

### Part V: Results

**9. Implementation**
- 9.1 [Package Architecture](./implementation/package_architecture.md)
- 9.2 [Numerical Methods](./implementation/numerical_optimization.md)
- 9.3 [Module Specifications](./implementation/module_specs.md)

**10. Exploration and Validation**
- 10.1 [Exploration Overview](./exploration/overview.md)
- 10.2 [Active Learning](./exploration/active_learning.md)
- 10.3 [Reproducibility](./exploration/harness_and_usage.md)

**11. Results and Analysis**
- 11.1 [Validation Methodology](./results/test_strategy.md)

**12. Conclusions**
- [Summary and Future Directions](./conclusions/summary.md)

---

## Key Results

| Finding | Section | Significance |
|---------|---------|--------------|
| Death Valley characterization | §3.2.5 | Identifies finite-size regime boundaries |
| QBER threshold validation | §3.2.4 | Confirms 11% conservative, 22% hard limit |
| Min-entropy bound comparison | §2.1.7 | Dupuis-König vs. Lupo bounds |
| Reconciliation efficiency | §6.1.6 | Measured $f \approx 1.08$ for $Q = 0.05$ |

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\mathcal{F}$ | Storage channel (CPTP map) |
| $\mathcal{N}_r$ | Depolarizing channel with parameter $r$ |
| $C_\mathcal{N}$ | Classical capacity |
| $H_{\min}^\varepsilon$ | $\varepsilon$-smooth min-entropy |
| $Q$ | Quantum bit error rate |
| $\ell$ | Extractable key length |
| $\varepsilon_{\text{sec}}$ | Security parameter |

---

*This report constitutes the technical documentation for the Caligo project, developed for the QIA Challenge 2025.*
