[← Return to Main Index](../index.md)

# 1.1 Introduction to Caligo

## 1.1.1 Motivation: Beyond Computational Security

Classical cryptography achieves security through computational hardness assumptions—RSA relies on factoring difficulty, AES on distinguishing random permutations, and Diffie-Hellman on discrete logarithms. These assumptions provide practical security but offer no guarantees against adversaries with sufficient computational resources or algorithmic breakthroughs. Quantum computers, when realized at scale, threaten to undermine many of these foundations via Shor's algorithm for factoring and discrete logarithm computation [4].

**Quantum cryptography** offers an alternative: security grounded in fundamental physical laws. Quantum Key Distribution (QKD), exemplified by the BB84 protocol [5], achieves unconditional security for key exchange. However, many advanced cryptographic tasks—secure computation, oblivious transfer, and electronic voting—cannot be implemented with information-theoretic security without additional assumptions [6,7].

The **Noisy Storage Model (NSM)** [1,8,9] bridges this gap by introducing a realistic physical constraint: **quantum storage undergoes decoherence**. This assumption is significantly weaker than bounded storage (requiring explicit memory limits) and is grounded in present-day physical implementations—photonic qubits, atomic ensembles, and superconducting circuits all exhibit storage noise. Under the NSM, protocols for oblivious transfer and related primitives achieve information-theoretic security.

## 1.1.2 The Oblivious Transfer Primitive

**1-out-of-2 Oblivious Transfer** ($\binom{2}{1}$-OT) is a cryptographic primitive where:

- **Sender (Alice)** holds two secret strings: $S_0, S_1 \in \{0,1\}^{\ell}$
- **Receiver (Bob)** holds a choice bit: $C \in \{0,1\}$

The protocol guarantees:

1. **Correctness**: Bob learns $S_C$ with high probability
2. **Sender Security**: Bob gains negligible information about $S_{1-C}$
3. **Receiver Security**: Alice gains negligible information about $C$

OT is universal for secure two-party computation [10]—any multi-party protocol can be constructed from OT as a building block. Applications include:

- **Secure auctions**: Bidders submit encrypted bids; highest bid wins without revealing losing bids
- **Private database queries**: Query a database without revealing which entry was accessed
- **Secure computation**: Evaluate functions on private inputs without revealing the inputs

Traditional OT implementations require computational assumptions or trusted setup. The NSM enables unconditionally secure OT under physical constraints alone.

## 1.1.3 The Noisy Storage Model: Physical Foundations

The NSM, introduced by Wehner, Schaffner, and Terhal [1], assumes an adversary (Bob) attempting to gain information about both $S_0$ and $S_1$ faces the following constraints:

1. **Storage Rate ($\nu$)**: Bob can coherently store at most $\nu \cdot n$ qubits from $n$ transmitted
2. **Depolarizing Noise ($r$)**: Each stored qubit undergoes a depolarizing channel $\mathcal{N}(\rho) = r\rho + (1-r)\frac{\mathbb{1}}{2}$
3. **Waiting Time ($\Delta t$)**: A mandatory delay between qubit reception and classical information revelation

**Fundamental Security Condition** [1]:

$$
C_{\mathcal{N}} \cdot \nu < \frac{1}{2}
$$

where $C_{\mathcal{N}} = 1 - h\left(\frac{1+r}{2}\right)$ is the classical capacity of the depolarizing channel, and $h(x) = -x \log_2 x - (1-x) \log_2(1-x)$ is binary entropy.

**Physical Justification**: Quantum storage noise is unavoidable in all current implementations:

- **Photonic qubits**: Optical fiber losses ($\sim$ 0.2 dB/km) and photon loss in quantum memories
- **Atomic ensembles**: Spontaneous emission, collisional decoherence
- **Superconducting qubits**: $T_1$ and $T_2$ relaxation times limit coherence
- **Ion traps**: Heating rates and motional decoherence

Even with fault-tolerant quantum computing, the **encoding operation** of an unknown state into an error-correcting code introduces residual noise [1].

## 1.1.4 The Protocol

Caligo implements an entanglement based protocol (BBM92 style) protocol, described by Lemus et al. [11]. The protocol synthesizes four technologies:

1. **EPR Pair Generation**: Alice and Bob share entangled Bell states $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$
2. **Measurements**: Random basis selection (computational $\{|0\rangle, |1\rangle\}$ or Hadamard $\{|+\rangle, |-\rangle\}$)
3. **LDPC Reconciliation**: One-way error correction using rate-compatible Low-Density Parity-Check codes [2,12]
4. **Toeplitz Privacy Amplification**: Key compression via universal hashing [13]

**Protocol Flow**:

- **Phase I**: Alice prepares EPR pairs; both parties measure in random bases
- **Phase II**: Basis sifting reveals matched/mismatched indices; QBER estimation
- **Phase III**: Alice sends LDPC syndromes; Bob corrects errors
- **Phase IV**: Privacy amplification extracts final keys $(S_0, S_1)$ for Alice, $S_C$ for Bob

The NSM security guarantee relies on **minimizing information leakage** during reconciliation—the syndrome length directly bounds the adversary's residual information.

## 1.1.5 Why "Caligo"?

The name **Caligo** (Latin: "fog," "mist," "obscurity") reflects the protocol's core property: information is deliberately obscured. Alice knows both keys but not which Bob received; Bob knows one key but cannot extract the other. Like objects in fog, the cryptographic secrets remain partially hidden despite being transmitted through a shared quantum channel.

The name also evokes the **simulation environment**—Caligo is not a physical implementation but a high-fidelity simulation built on SquidASM/NetSquid, frameworks that model quantum network behavior with discrete-event precision. Just as fog obscures physical reality, simulation provides a controlled environment to study protocol behavior before hardware deployment.

## 1.1.6 Report Objectives

This report provides a complete technical account of Caligo, structured for multiple audiences:

- **Cryptographers**: Rigorous security proofs under the NSM, leakage accounting, and finite-size analysis
- **Quantum Information Scientists**: EPR pair generation strategies, measurement protocols, and QBER estimation
- **Software Engineers**: Domain-driven architecture, SquidASM integration patterns, and modular design principles
- **Implementers**: Detailed specifications for LDPC reconciliation, Toeplitz hashing, and timing enforcement

The following chapters progress from theoretical foundations (NSM model, cryptographic primitives) through architectural design (domain-driven structure, phase contracts) to implementation details (LDPC strategies, Numba optimization) and validation (QBER analysis, security parameter verification).

---

## References

[1] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.* 100, 220502 (2008).

[2] D. Elkouss, J. Martinez-Mateo, D. Lancho, and V. Martin, "Rate Compatible Protocol for Information Reconciliation: An Application to QKD" (2010).

[3] D. Elkouss, J. Martinez-Mateo, and V. Martin, "Untainted Puncturing for Irregular Low-Density Parity-Check Codes," *IEEE Wireless Commun. Lett.* 1(6), 585-588 (2012).

[4] P. W. Shor, "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer," *SIAM J. Comput.* 26(5), 1484-1509 (1997).

[5] C. H. Bennett and G. Brassard, "Quantum Cryptography: Public Key Distribution and Coin Tossing," *Proc. IEEE ICCSS*, 175-179 (1984).

[6] H.-K. Lo and H. F. Chau, "Is Quantum Bit Commitment Really Possible?" *Phys. Rev. Lett.* 78, 3410 (1997).

[7] D. Mayers, "Unconditionally Secure Quantum Bit Commitment is Impossible," *Phys. Rev. Lett.* 78, 3414 (1997).

[8] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory* 58(3), 1962-1984 (2012).

[9] C. Schaffner, "Cryptography in the Bounded-Quantum-Storage Model," PhD Thesis, University of Aarhus (2007).

[10] J. Kilian, "Founding Cryptography on Oblivious Transfer," *Proc. 20th ACM STOC*, 20-31 (1988).

[11] M. Lemus et al., "Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation," *arXiv:2007.xxxxx* (2020).

[12] J. Martinez-Mateo, D. Elkouss, and V. Martin, "Blind Reconciliation," *Quantum Inf. Comput.* 12(9&10), 791-812 (2012).

[13] J. L. Carter and M. N. Wegman, "Universal Classes of Hash Functions," *J. Comput. Syst. Sci.* 18(2), 143-154 (1979).

---

[← Return to Main Index](../index.md) | [← Previous: Abstract](./abstract.md) | [Next: Problem Scope →](./scope.md)
