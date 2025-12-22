[← Return to Main Index](../index.md)

# 4.4 Measurement Protocol and Basis Selection

## Introduction

The measurement phase of an NSM protocol implements the BB84 encoding scheme [1], where each party independently selects one of two mutually unbiased bases. The security of the protocol rests on the information-theoretic properties of these measurements: when Alice and Bob select matching bases, their outcomes are perfectly correlated (up to channel noise); when bases mismatch, outcomes are completely uncorrelated.

## BB84 Measurement Bases

### Mathematical Definition

The two measurement bases correspond to Pauli eigenstates:

**Computational basis (Z):**
$$
|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}
$$

**Hadamard basis (X):**
$$
|+\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad |-\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}
$$

The bases are related by the Hadamard transformation:
$$
H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad H|0\rangle = |+\rangle, \quad H|1\rangle = |-\rangle
$$

### Mutual Unbiasedness

The Z and X bases are **mutually unbiased**: measuring a Z-eigenstate in the X basis (or vice versa) yields uniformly random outcomes:

$$
|\langle 0|+\rangle|^2 = |\langle 0|-\rangle|^2 = |\langle 1|+\rangle|^2 = |\langle 1|-\rangle|^2 = \frac{1}{2}
$$

This property is essential for NSM security: a dishonest party who measures in the wrong basis gains no information about the bit value.

## POVM Formulation

### Projective Measurements

For basis choice $\theta \in \{0, 1\}$, the measurement is described by the POVM:

$$
\{M_0^\theta, M_1^\theta\} = \begin{cases}
\{|0\rangle\langle 0|, |1\rangle\langle 1|\} & \text{if } \theta = 0 \text{ (Z-basis)} \\
\{|+\rangle\langle +|, |-\rangle\langle -|\} & \text{if } \theta = 1 \text{ (X-basis)}
\end{cases}
$$

The outcome probability for state $\rho$ is:
$$
P(a = k | \theta) = \text{Tr}[M_k^\theta \rho]
$$

### Imperfect Measurements

Real detectors introduce errors. The effective POVM becomes:

$$
\tilde{M}_k^\theta = (1 - e_{\text{det}}) M_k^\theta + e_{\text{det}} M_{1-k}^\theta
$$

where $e_{\text{det}}$ is the detector error rate (probability of misidentifying the outcome).

## Basis Selection Requirements

### Uniformity Constraint

For security, basis choices must be **uniformly random and independent**:

$$
P(\theta_i = 0) = P(\theta_i = 1) = \frac{1}{2} \quad \forall i
$$

This ensures that a dishonest Alice gains no information about Bob's measurement strategy before basis revelation.

**Critical security requirement** [2]: If Bob's basis choices are non-uniform, Alice can exploit this to gain partial information about his choice bit $C$ in the OT protocol. The protocol requires:

> *"An honest Bob can perform symmetrization over all his detectors to make them all equally efficient, so that Alice cannot make a better guess of his choice bit than a random guess."*

### Expected Basis Match Rate

For uniform random basis selection:

$$
P(\theta_A = \theta_B) = \frac{1}{2}
$$

The expected number of matching-basis pairs is $n/2$, with standard deviation $\sqrt{n}/2$.

## Correlation Statistics

### Bell State Correlations

For the maximally entangled state $|\Phi^+\rangle$ with perfect detectors:

| Alice basis | Bob basis | $P(a = b)$ |
|-------------|-----------|------------|
| Z | Z | 1 |
| X | X | 1 |
| Z | X | 1/2 |
| X | Z | 1/2 |

### QBER Decomposition

The total QBER decomposes into independent contributions:

$$
Q = \underbrace{\frac{1-F}{2}}_{Q_{\text{source}}} + \underbrace{e_{\text{det}}}_{Q_{\text{detector}}} + \underbrace{(1-\eta) \cdot \frac{P_{\text{dark}}}{2}}_{Q_{\text{dark}}}
$$

where:
- $F$ is the source fidelity to $|\Phi^+\rangle$
- $e_{\text{det}}$ is the intrinsic detector error
- $\eta$ is the detection efficiency
- $P_{\text{dark}}$ is the dark count probability

For the Erven et al. experiment [2]: $Q \approx 0.029$ (predominantly source-limited).

## Sifted Key Statistics

### Expected Sifted Length

After basis sifting, the expected key length is:

$$
\mathbb{E}[n_{\text{sifted}}] = \frac{n_{\text{raw}}}{2} \cdot (1 - P_{\text{loss}})
$$

where $P_{\text{loss}}$ accounts for detection inefficiency and dark counts.

### Variance and Confidence Intervals

The sifted length follows approximately:

$$
n_{\text{sifted}} \sim \mathcal{N}\left(\frac{n}{2}(1-P_{\text{loss}}), \frac{n}{4}(1-P_{\text{loss}})\right)
$$

For $n = 10^6$ and $P_{\text{loss}} = 0.1$:
$$
n_{\text{sifted}} \approx 450{,}000 \pm 474 \quad (1\sigma)
$$

---

## References

[1] C. H. Bennett and G. Brassard, "Quantum cryptography: Public key distribution and coin tossing," *Proc. IEEE Int. Conf. Comput. Syst. Signal Process.*, pp. 175–179, 1984.

[2] C. Erven et al., "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model," *Nat. Commun.*, vol. 5, 3418, 2014.

---

[← Return to Main Index](../index.md) | [Next: Basis Sifting](../sifting/basis_sifting.md)
