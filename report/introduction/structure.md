[← Return to Main Index](../index.md)

# 1.3 Document Structure

This report is organized to serve multiple reading strategies—sequential for comprehensive understanding, or targeted for specific technical aspects. Each chapter is self-contained with cross-references to related sections.

## Reading Paths

### Path 1: Theory-First (Cryptographers & Quantum Information Scientists)

For readers interested in security foundations and mathematical rigor:

1. [Abstract](./abstract.md) — High-level overview
2. **Chapter 2: Theoretical Foundations** — NSM model, cryptographic primitives, literature review
3. **Chapter 3: Protocol Architecture** — Security model and parameter mappings
4. **Chapter 6: Information Reconciliation** — LDPC framework, hybrid puncturing, leakage accounting
5. **Chapter 10: Validation & Results** — Security parameter verification, QBER analysis

### Path 2: Implementation-First (Software Engineers & Implementers)

For readers focused on architecture and code organization:

1. [Abstract](./abstract.md) — High-level overview
2. [Problem Scope](./scope.md) — Objectives and success criteria
3. **Chapter 3: Protocol Architecture** — Domain-driven design principles
4. **Chapter 9: Implementation Details** — Package architecture, SquidASM integration, module specifications
5. **Chapter 10: Validation & Results** — Test strategy and performance metrics

### Path 3: Protocol-First (Experimentalists & Protocol Designers)

For readers interested in end-to-end protocol flow:

1. [Abstract](./abstract.md) — High-level overview
2. **Chapter 3.1: Protocol Overview** — Four-phase pipeline
3. **Chapter 4: Phase I (Quantum Layer)** — EPR generation and measurement
4. **Chapter 5: Phase II (Sifting)** — Basis sifting and QBER estimation
5. **Chapter 6: Phase III (Reconciliation)** — Error correction strategies
6. **Chapter 7: Phase IV (Amplification)** — Privacy amplification and key extraction

---

## Chapter Summaries

### Chapter 0: Front Matter
- **Abstract**: Executive summary of objectives, methods, and key results

### Chapter 1: Introduction
- **1.1 Introduction to Caligo**: Motivation, NSM foundations, protocol overview
- **1.2 Problem Scope**: Challenges, objectives, success metrics, research questions
- **1.3 Document Structure**: (This section)

### Chapter 2: Theoretical Foundations
- **2.1 The Noisy Storage Model**: Security inequality, physical justification, attack models
- **2.2 Cryptographic Primitives**: OT definitions, weak string erasure, composability
- **2.3 SquidASM Framework**: NetSquid discrete-event model, quantum operations, timing semantics
- **2.4 Protocol Literature**: Wehner et al. (2008), Elkouss et al. (2010/2012), Lemus et al. (2020)

### Chapter 3: Protocol Architecture
- **3.1 Protocol Overview**: Four-phase pipeline, data flow, phase contracts
- **3.2 Security Model**: NSM parameters, finite-size key rate, leakage budget enforcement
- **3.3 Domain-Driven Design**: Package structure rationale, modularity principles

### Chapter 4: Phase I (Quantum Layer)
- **4.1 EPR Pair Generation**: Bell state preparation, fidelity considerations
- **4.2 Sequential vs. Parallel**: Throughput vs. memory tradeoffs, batching strategies
- **4.3 Batching Strategies**: Optimal batch size, memory consumption analysis
- **4.4 Measurement**: BB84 basis selection, outcome recording, timing synchronization

### Chapter 5: Phase II (Sifting & QBER Estimation)
- **5.1 Basis Sifting**: Index set partitioning ($I_0, I_1$ for Alice; $I_+, I_\times$ for Bob)
- **5.2 QBER Estimation**: Statistical sampling, confidence intervals, finite-size corrections
- **5.3 Security Verification**: QBER threshold checks, abort conditions

### Chapter 6: Phase III (Information Reconciliation)
- **6.1 Rate-Compatible LDPC Framework**: Mother code, effective rate calculation, puncturing theory
- **6.2 Baseline Strategy**: QBER-aware rate selection, single-pass reconciliation
- **6.3 Blind Strategy**: Iterative bit revelation, adaptive rate escalation, stopping criteria
- **6.4 Hybrid Puncturing**: Untainted regime (stopping-set protection) + ACE-guided regime (topology management)
- **6.5 Leakage Accounting**: Syndrome length, hash verification, revealed bits, circuit-breaker enforcement

### Chapter 7: Phase IV (Privacy Amplification)
- **7.1 Toeplitz Hashing**: Two-universal hash functions, construction via LFSR
- **7.2 Extractable Length**: Min-entropy estimation, leftover hash lemma, finite-size penalties
- **7.3 Key Derivation**: Final key extraction for Alice $(S_0, S_1)$ and Bob $(S_C)$

### Chapter 8: NSM Parameters & Physical Models
- **8.1 Parameter Space**: Valid ranges for $(r, \nu, \Delta t, F, \eta, e_{\text{det}}, P_{\text{dark}})$
- **8.2 NSM-to-Physical Mapping**: Depolarizing probability, QBER decomposition, timing translation
- **8.3 Noise Models**: SquidASM gate noise, link attenuation, detector inefficiency
- **8.4 Timing Enforcement**: Discrete-event barriers, synchronization primitives

### Chapter 9: Implementation Details
- **9.1 Package Architecture**: Directory structure, module responsibilities, import graph
- **9.2 SquidASM Integration**: `NetQASMConnection`, program compilation, qubit lifecycle
- **9.3 Numerical Optimization**: Numba JIT compilation, vectorized operations, memory management
- **9.4 Module Specifications**: API contracts, input validation, error handling

### Chapter 10: Validation & Results
- **10.1 Test Strategy**: Unit tests, integration tests, phase-contract validation
- **10.2 Performance Metrics**: EPR success rate, reconciliation efficiency, key rate, simulation time
- **10.3 QBER Analysis**: Channel vs. storage contributions, parameter sweep results
- **10.4 Security Validation**: Min-entropy bounds, leakage verification, finite-size analysis

### Chapter 11: Discussion
- **11.1 Key Achievements**: Hybrid puncturing, simulation-native design, modular architecture
- **11.2 Technical Challenges**: Finite-length LDPC optimization, SquidASM timing subtleties
- **11.3 Lessons Learned**: Architecture anti-patterns, simulation vs. hardware tradeoffs
- **11.4 Future Work**: Multi-party extensions, alternative codes, hardware deployment path

### Chapter 12: Conclusions
- **Summary**: Project contributions, research question answers, impact statement

### Appendices
- **A. Mathematical Proofs**: Hybrid puncturing theorem, leakage bounds
- **B. Code Listings**: Key algorithms (BP decoder, Toeplitz hash, timing barrier)
- **C. Simulation Configuration**: Example `network_config.yaml`, parameter sweep scripts
- **D. Glossary**: Definitions of technical terms and acronyms

---

## Notation and Conventions

### Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $n$ | Block length (post-sifting raw key size) |
| $k$ | LDPC dimension (mother code) |
| $R_0$ | Mother code rate ($R_0 = k/n$) |
| $R_{\text{eff}}$ | Effective rate after puncturing/shortening |
| $p$ | Number of punctured bits |
| $s$ | Number of shortened bits |
| $Q$ | Quantum Bit Error Rate (QBER) |
| $h(x)$ | Binary entropy: $-x \log_2 x - (1-x) \log_2(1-x)$ |
| $f$ | Reconciliation efficiency: $\text{leak}_{\text{EC}} / (n \cdot h(Q))$ |
| $\ell$ | Extractable secure key length |
| $\epsilon$ | Security parameter (failure probability) |
| $H_{\min}^{\epsilon}(X\|E)$ | Smooth min-entropy |
| $\nu$ | Storage rate (fraction of qubits stored) |
| $r$ | Depolarizing parameter ($0 \leq r \leq 1$) |
| $\Delta t$ | NSM waiting time |
| $C_{\mathcal{N}}$ | Classical capacity of depolarizing channel |

### Code Conventions

- **Module names**: `snake_case` (e.g., `ldpc_encoder.py`)
- **Class names**: `PascalCase` (e.g., `ReconciliationOrchestrator`)
- **Function names**: `snake_case` (e.g., `compute_syndrome()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `QBER_HARD_LIMIT`)
- **Type hints**: Enforced throughout (Python 3.10+ syntax)

### File Paths

All file references use workspace-relative paths formatted as clickable Markdown links:
- Example: [caligo/reconciliation/ldpc_encoder.py](../../caligo/caligo/reconciliation/ldpc_encoder.py)
- Line references: [ldpc_encoder.py:66-108](../../caligo/caligo/reconciliation/ldpc_encoder.py#L66-L108)

### Acronyms

| Acronym | Expansion |
|---------|-----------|
| **ACE** | Approximate Cycle Extrinsic Message Degree |
| **BB84** | Bennett-Brassard 1984 (QKD protocol) |
| **BP** | Belief Propagation |
| **EMD** | Extrinsic Message Degree |
| **EPR** | Einstein-Podolsky-Rosen (entangled pair) |
| **FER** | Frame Error Rate |
| **JIT** | Just-In-Time (compilation) |
| **LDPC** | Low-Density Parity-Check |
| **LFSR** | Linear Feedback Shift Register |
| **LOC** | Lines of Code |
| **NSM** | Noisy Storage Model |
| **OT** | Oblivious Transfer |
| **PEG** | Progressive Edge-Growth |
| **QBER** | Quantum Bit Error Rate |
| **QKD** | Quantum Key Distribution |
| **ROT** | Randomized Oblivious Transfer |
| **SRP** | Single Responsibility Principle |
| **WSE** | Weak String Erasure |

---

## Document Metadata

- **Version**: 1.0  
- **Date**: December 2025  
- **Format**: Markdown with LaTeX math (KaTeX)  
- **Citation Style**: IEEE (numbered references)  
- **Word Count**: ~25,000 (estimated, excluding code listings)  
- **Target Audience**: Graduate-level quantum information science, cryptography, or software engineering background

---

## Accessibility Notes

- **Math Rendering**: Inline equations use `$...$`; display equations use `$$...$$` (KaTeX-compatible)
- **Cross-References**: All chapter/section links use relative paths for portability
- **Code Blocks**: Syntax-highlighted Python/YAML with line numbers where relevant
- **Figures**: Alt text provided for accessibility (where applicable)
- **Tables**: Markdown format with alignment for readability

---

[← Return to Main Index](../index.md) | [← Previous: Problem Scope](./scope.md) | [Next Chapter: Theoretical Foundations →](../foundations/nsm_model.md)
