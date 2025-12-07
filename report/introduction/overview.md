[← Return to Main Index](../index.md)

# 1. Introduction

## 1.1 Context and Motivation

The rapid advancement of quantum information technologies has necessitated the development of cryptographic primitives that leverage quantum mechanical properties for enhanced security. While Quantum Key Distribution (QKD) has successfully addressed the problem of generating identical shared secrets between two parties, the domain of Secure Multiparty Computation (SMC) requires a different primitive: **Oblivious Transfer (OT)**.

In an OT scenario, a sender (Alice) transfers information to a receiver (Bob) such that Bob receives only a subset of the information, and Alice remains ignorant of which subset Bob received. This primitive is foundational for privacy-preserving computations. The **Entanglement-based Hybrid Oblivious Key (E-HOK)** protocol represents a convergence of these fields, utilizing quantum entanglement to generate "Oblivious Keys"—correlated bitstrings where the receiver has partial, verifiable ignorance.

This project focuses on the implementation and validation of a baseline E-HOK protocol within the **SquidASM** simulation framework. By establishing a robust, modular baseline, we aim to create a platform for future industrial research into advanced quantum network applications.

## 1.2 Literature Background

The theoretical foundation of this project rests on the intersection of quantum key distribution and oblivious transfer protocols.

### 1.2.1 Entanglement-Based Distribution
The core quantum generation mechanism derives from the work of **Ekert (1991)** [1], who proposed using Einstein-Podolsky-Rosen (EPR) pairs for key distribution. In this scheme, Alice and Bob measure entangled qubits (e.g., the Bell state $|\Phi^+\rangle$) in randomly chosen bases.
$$ |\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) $$
When bases match, outcomes are correlated; when they differ, outcomes are uncorrelated. This property, fundamental to the BB84 and E91 protocols, is repurposed in E-HOK not just to discard mismatched bases, but to define the "oblivious" regions of the key.

### 1.2.2 Quantum Oblivious Keys
**Lemus et al. (2020)** [2] formalized the concept of generating Oblivious Keys for SMC. Their work demonstrates that by combining quantum distribution with classical commitment schemes, one can achieve a hybrid protocol where:
1.  **Alice** holds a full key $K$.
2.  **Bob** holds a key $\bar{K}$ and a "knowledge mask" indicating which bits of $K$ he knows (where bases matched) and which he does not.
3.  **Security** is guaranteed by the monogamy of entanglement and classical commitments.

### 1.2.3 Classical Post-Processing
The baseline implementation draws upon established classical cryptographic techniques to secure the quantum phase:
*   **Commitment Schemes**: To prevent "look-ahead" cheating, the receiver must commit to their measurement outcomes before basis announcement. **Halevi and Micali** [3] provide the theoretical basis for computationally binding hash-based commitments, which we implement using SHA-256 in the baseline.
*   **Information Reconciliation**: To correct quantum channel errors (QBER), we utilize **Low-Density Parity-Check (LDPC)** codes, which offer efficient syndrome decoding close to the Shannon limit [4].
*   **Privacy Amplification**: **Toeplitz matrices** are employed as universal hash functions to compress the key and eliminate information leaked during reconciliation or to an eavesdropper [2].

## 1.3 Research Insights and Problem Statement

A review of the literature reveals that while theoretical models for Quantum OT exist, practical, verifiable implementations in discrete-event simulators are scarce. Most existing simulations focus on pure QKD (maximizing shared key) rather than Oblivious Keys (managing partial knowledge).

**Key Insight**: The transition from QKD to E-HOK requires a fundamental architectural shift. The system must not only track the *value* of the bits but also the *provenance* of the knowledge (i.e., the "Oblivious" state). Furthermore, the integration of strict commitment phases is critical to prevent active adversaries in the semi-honest model.

### 1.4 Main Research Question

Based on the theoretical requirements and the capabilities of the SquidASM framework, we formulate the following primary research question for the baseline implementation:

> **RQ1**: How can an entanglement-based oblivious key distribution protocol be architected within the SquidASM framework to ensure verifiable partial knowledge (obliviousness) while maintaining modularity for future cryptographic extensions?

This question drives the design of the modular "Manager/Worker" architecture, the definition of the `ObliviousKey` data structure, and the strict separation of quantum generation, commitment, and reconciliation phases.

## References

[1] A. K. Ekert, "Quantum cryptography based on Bell’s theorem," *Physical Review Letters*, vol. 67, no. 6, pp. 661–663, 1991.
[2] P. Lemus, et al., "Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation," *arXiv preprint arXiv:1909.11701*, 2020.
[3] S. Halevi and S. Micali, "Practical and Provably-Secure Commitment Schemes from Collision-Free Hashing," *CRYPTO*, 1996.
[4] D. J. C. MacKay, "Good error-correcting codes based on very sparse matrices," *IEEE Transactions on Information Theory*, vol. 45, no. 2, pp. 399–431, 1999.

---
[← Return to Main Index](../index.md)
