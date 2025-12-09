# QIA Challenge 2025: Technical Report

**Authors**: Alessandro Da Ros, GitHub Copilot (AI Agent)
**Date**: December 2025

---

## Table of Contents

### 1. [Introduction](./introduction/overview.md)
*   1.1 Context and Motivation
*   1.2 Literature Background
*   1.3 Research Insights and Problem Statement
*   1.4 Main Research Question

### 2. Foundations
*   2.1 [The SquidASM Framework](./foundations/squidasm_framework.md)
    *   *Three-tier architecture: SquidASM → NetQASM → NetSquid*
    *   *Quantum state representation and noise models*
    *   *EPRSocket API and Future abstraction*
    *   *E-HOK integration patterns*
*   2.2 [Theoretical Underpinnings](./foundations/theory.md) *(Planned)*
    *   *Deep dive into EPR correlations and Oblivious Transfer security definitions.*

### 3. Baseline Protocol Implementation
*   3.1 [Architecture & Design Patterns](./baseline_protocol/architecture.md)
    *   *Layered architecture, Strategy Pattern, Template Method Pattern*
    *   *Component interactions and state machine formalization*
    *   *Quantum batching and classical communication protocols*
*   3.2 [Phase I: Quantum Generation](./baseline_protocol/quantum_generation.md) *(Planned)*
    *   *Implementation of batching and EPR measurement.*
*   3.3 [Phase II: Commitment & Verification](./baseline_protocol/commitment.md) *(Planned)*
    *   *Hash-based commitment logic and security checks.*
*   3.4 [Phase III-V: Post-Processing](./baseline_protocol/post_processing.md) *(Planned)*
    *   *Sifting, LDPC Reconciliation, and Privacy Amplification.*

### 4. Results & Validation
*   4.1 [Simulation Setup](./results/simulation_setup.md) *(Planned)*
    *   *Network topology and noise models.*
*   4.2 [Baseline Performance Metrics](./results/baseline_metrics.md) *(Planned)*
    *   *QBER analysis, key generation rates, and execution time.*

### 5. Future Extensions
*   5.1 [Advanced Protocols](./extensions/advanced_protocols.md) *(Planned)*
    *   *Roadmap for MET-LDPC and Noisy Storage Model integration.*

---
*This report documents the technical implementation and validation of the E-HOK protocol.*
