# QIA Challenge 2025: Technical Report

**Authors**: Alessandro Da Ros, GitHub Copilot (AI Agent)  
**Date**: December 2025  
**Project**: Caligo - $\binom{2}{1}$-Oblivious Transfer via the Noisy Storage Model

---

## Table of Contents

### 0. Front Matter
*   [Abstract](./introduction/abstract.md)

### 1. Introduction
*   1.1 [Introduction to Caligo](./introduction/introduction.md)
*   1.2 [Problem Scope & Objectives](./introduction/scope.md)
*   1.3 [Document Structure](./introduction/structure.md)

### 2. Theoretical Foundations
*   2.1 [The Noisy Storage Model (NSM)](./foundations/nsm_model.md)
*   2.2 [Cryptographic Primitives](./foundations/primitives.md)
*   2.3 [The SquidASM Simulation Framework](./foundations/squidasm_framework.md)
*   2.4 [Protocol Literature Review](./foundations/protocol_literature.md)

### 3. Protocol Architecture
*   3.1 [Protocol Overview: Four-Phase Pipeline](./architecture/protocol_overview.md)
*   3.2 [Security Model & NSM Parameters](./architecture/security_model.md)
*   3.3 [Domain-Driven Design](./architecture/domain_design.md)

### 4. Phase I: Quantum Layer
*   4.1 [EPR Pair Generation](./quantum/epr_generation.md)
*   4.2 [Sequential vs. Parallel Generation](./quantum/generation_modes.md)
*   4.3 [Batching Strategies](./quantum/batching.md)
*   4.4 [Measurement & Basis Selection](./quantum/measurement.md)

### 5. Phase II: Sifting & QBER Estimation
*   5.1 [Basis Sifting Protocol](./sifting/basis_sifting.md)
*   5.2 [QBER Estimation](./sifting/qber_estimation.md)
*   5.3 [Security Verification](./sifting/security_checks.md)

### 6. Phase III: Information Reconciliation
*   6.1 [Rate-Compatible LDPC Framework](./reconciliation/ldpc_framework.md)
*   6.2 [Baseline Reconciliation Strategy](./reconciliation/baseline_strategy.md)
*   6.3 [Blind Reconciliation Strategy](./reconciliation/blind_strategy.md)
*   6.4 [Hybrid Puncturing Architecture](./reconciliation/hybrid_puncturing.md)
*   6.5 [Leakage Accounting](./reconciliation/leakage_accounting.md)

### 7. Phase IV: Privacy Amplification
*   7.1 [Toeplitz Hashing](./amplification/toeplitz_hashing.md)
*   7.2 [Extractable Length Calculation](./amplification/extractable_length.md)
*   7.3 [Key Derivation](./amplification/key_derivation.md)

### 8. NSM Parameters & Physical Models
*   8.1 [Parameter Space](./nsm/parameter_space.md)
*   8.2 [NSM-to-Physical Mapping](./nsm/physical_mapping.md)
*   8.3 [Noise Model Configuration](./nsm/noise_models.md)
*   8.4 [Timing Enforcement](./nsm/timing_enforcement.md)

### 9. Implementation Details
*   9.1 [Package Architecture](./implementation/package_architecture.md)
*   9.2 [Numerical Optimization](./implementation/numerical_optimization.md)
*   9.3 [Module Specifications](./implementation/module_specs.md)

### 10. Validation & Results
*   10.1 [Test Strategy](./results/test_strategy.md)
*   10.2 [Performance Metrics](./results/performance_metrics.md)
*   10.3 [QBER Analysis](./results/qber_analysis.md)
*   10.4 [Security Parameter Validation](./results/security_validation.md)

### 11. Discussion
*   11.1 [Key Achievements](./discussion/achievements.md)
*   11.2 [Technical Challenges](./discussion/challenges.md)
*   11.3 [Lessons Learned](./discussion/lessons_learned.md)
*   11.4 [Future Work](./discussion/future_work.md)

### 12. Conclusions
*   [Summary & Final Remarks](./conclusions/summary.md)

### Appendices
*   A. [Mathematical Proofs](./appendices/proofs.md)
*   B. [Code Listings](./appendices/code_listings.md)
*   C. [Simulation Configuration](./appendices/simulation_config.md)
*   D. [Glossary of Terms](./appendices/glossary.md)

### References
*   [Bibliography](./references/bibliography.md)

---

*This report documents the design, implementation, and validation of Caligo, a simulation-native $\binom{2}{1}$-Oblivious Transfer protocol built on the Noisy Storage Model.*
