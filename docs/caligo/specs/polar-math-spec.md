# Polar Codes: Mathematical Specification

<metadata>
version: 1.0.0
status: draft
created: 2026-02-03
purpose: Algebraic ground truth for Polar codec implementation
dependencies:
  - adr/0001-polar-codec-adoption.md
  - specs/siso-codec-protocol.md
</metadata>

---

## Executive Summary

<overview>
This document defines the mathematical foundations of Polar Codes as required for the HPC-Ready Polar Codec implementation. It serves as the **algebraic ground truth**, abstracting away code to focus purely on the mathematical structure. All implementation agents MUST conform to the definitions herein.

**Scope:** Channel polarization transform, generator matrix construction, decoding graph topology, and permutation contracts.

**Non-Scope:** Algorithm pseudocode, software interfaces, performance optimizations.
</overview>

---

## 1. Notation Conventions

<notation>

| Symbol | Meaning | Domain |
|--------|---------|--------|
| $N$ | Block length | $N = 2^m$, $m \in \mathbb{Z}^+$ |
| $K$ | Information bits (dimension) | $0 < K \leq N$ |
| $m$ | Polarization depth | $m = \log_2 N$ |
| $\mathbf{u}$ | Information + frozen bits vector | $\mathbf{u} \in \{0,1\}^N$ |
| $\mathbf{x}$ | Codeword (transmitted) | $\mathbf{x} \in \{0,1\}^N$ |
| $\mathbf{y}$ | Received vector (channel output) | $\mathbf{y} \in \mathcal{Y}^N$ |
| $\mathcal{A}$ | Information bit indices | $\mathcal{A} \subset [N]$, $|\mathcal{A}| = K$ |
| $\mathcal{F}$ | Frozen bit indices | $\mathcal{F} = [N] \setminus \mathcal{A}$ |
| $u_i^j$ | Sub-vector $(u_i, u_{i+1}, \ldots, u_j)^T$ | $j \geq i$ |
| $[N]$ | Index set $\{0, 1, \ldots, N-1\}$ | — |

**Arithmetic:** All binary operations are in $\mathbb{F}_2$ (XOR for addition, AND for multiplication).

</notation>

---

## 2. Algebraic Definition: The Generator Matrix

### 2.1 The Polarizing Kernel

<definition id="kernel">

The fundamental building block is the $2 \times 2$ Arıkan kernel:

$$
F_2 = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}
$$

This kernel implements channel combining: given two independent uses of channel $W$, it synthesizes two new channels with different reliabilities.

**Reference:** [List_Decoding_of_Polar_Codes.md] §II, [Fast_Polar_Decoders_Algorithm_and_Implementation.md] §II.A Fig. 1.

</definition>

### 2.2 Recursive Construction via Kronecker Product

<definition id="generator-matrix">

For block length $N = 2^m$, the **generator matrix** is constructed recursively:

$$
G_N = F_2^{\otimes m} = \underbrace{F_2 \otimes F_2 \otimes \cdots \otimes F_2}_{m \text{ times}}
$$

where $\otimes$ denotes the Kronecker product. Explicitly:

$$
F_2^{\otimes m} = F_2 \otimes F_2^{\otimes (m-1)}
$$

with base case $F_2^{\otimes 1} = F_2$.

**Examples:**

For $N = 2$ ($m = 1$):
$$
G_2 = F_2 = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}
$$

For $N = 4$ ($m = 2$):
$$
G_4 = F_2 \otimes F_2 = \begin{bmatrix} 
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 \\
1 & 1 & 1 & 1 
\end{bmatrix}
$$

For $N = 8$ ($m = 3$):
$$
G_8 = F_2 \otimes G_4 = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1
\end{bmatrix}
$$

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II Eq. (1).

</definition>

### 2.3 The Bit-Reversal Permutation Matrix

<definition id="bit-reversal">

The **bit-reversal permutation matrix** $B_N$ reorders indices according to bit-reversed binary representation:

$$
B_N[i, j] = \begin{cases}
1 & \text{if } j = \text{bitrev}_m(i) \\
0 & \text{otherwise}
\end{cases}
$$

where $\text{bitrev}_m(i)$ reverses the $m$-bit binary representation of $i$.

**Definition of $\text{bitrev}_m$:** For $i \in [N]$ with binary representation $(b_{m-1}, b_{m-2}, \ldots, b_1, b_0)$:
$$
\text{bitrev}_m(i) = \sum_{k=0}^{m-1} b_k \cdot 2^{m-1-k}
$$

**Example for $N = 8$ ($m = 3$):**

| $i$ | Binary | Reversed | $\text{bitrev}_3(i)$ |
|-----|--------|----------|----------------------|
| 0 | 000 | 000 | 0 |
| 1 | 001 | 100 | 4 |
| 2 | 010 | 010 | 2 |
| 3 | 011 | 110 | 6 |
| 4 | 100 | 001 | 1 |
| 5 | 101 | 101 | 5 |
| 6 | 110 | 011 | 3 |
| 7 | 111 | 111 | 7 |

**Properties:**
- $B_N$ is a permutation matrix: $B_N^{-1} = B_N^T = B_N$
- $B_N^2 = I_N$
- $B_N$ is symmetric

**Reference:** [Fast_Polar_Decoders_Algorithm_and_Implementation.md] §II.A (bit-reversed indexing).

</definition>

### 2.4 Complete Generator Matrix with Bit-Reversal

<definition id="full-generator">

The complete polar code generator matrix incorporating bit-reversal is:

$$
\mathbf{G}_m = F_2^{\otimes m} \cdot B_N
$$

or equivalently, using the convention in [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md]:

$$
\mathbf{G}_m \triangleq \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}^{\otimes m} B_m
$$

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II Eq. (1).

</definition>

---

## 3. Encoding Transformation

<definition id="encoding">

**Encoding:** Given information vector $\mathbf{u} \in \{0,1\}^N$ (with frozen positions set to known values, typically 0):

$$
\mathbf{x} = \mathbf{G}_m \mathbf{u}
$$

Equivalently, using the factored form:
$$
\mathbf{x} = (F_2^{\otimes m} B_N) \mathbf{u} = F_2^{\otimes m} (B_N \mathbf{u})
$$

**Interpretation:**
1. Apply bit-reversal permutation to $\mathbf{u}$: $\mathbf{u}' = B_N \mathbf{u}$
2. Apply polarizing transform: $\mathbf{x} = F_2^{\otimes m} \mathbf{u}'$

**Complexity:** $O(N \log N)$ binary operations via recursive butterfly structure.

</definition>

---

## 4. The Decoding Graph: Layer-Phase Indexing

### 4.1 Bit-Channel Definition

<definition id="bit-channels">

The polarizing transform synthesizes $N$ **bit-channels** $W_\lambda^{(\phi)}$, indexed by:
- **Layer** $\lambda$: $0 \leq \lambda \leq m$
- **Phase** $\phi$: $0 \leq \phi < 2^\lambda$

For notational convenience, define $\Lambda = 2^\lambda$.

The bit-channel $W_\lambda^{(\phi)}$ has:
- **Input:** single bit $u_\phi$
- **Output:** $(y_0^{\Lambda-1}, u_0^{\phi-1})$ — the received symbols plus previously decoded bits

**Transition probability:**
$$
W_\lambda^{(\phi)}(y_0^{\Lambda-1}, u_0^{\phi-1} | u_\phi)
$$

**Reference:** [List_Decoding_of_Polar_Codes.md] §II.A Eq. (1)-(3).

</definition>

### 4.2 Recursive Bit-Channel Equations

<definition id="recursive-channels">

The bit-channels satisfy the following recursions. Let $0 \leq \psi < \Lambda/2$:

**Even phase** ($\phi = 2\psi$):
$$
W_\lambda^{(2\psi)}(y_0^{\Lambda-1}, u_0^{2\psi-1} | u_{2\psi}) = \sum_{u_{2\psi+1} \in \{0,1\}} \frac{1}{2} W_{\lambda-1}^{(\psi)}(y_0^{\Lambda/2-1}, u_{0,\text{even}}^{2\psi-1} \oplus u_{0,\text{odd}}^{2\psi-1} | u_{2\psi} \oplus u_{2\psi+1}) \cdot W_{\lambda-1}^{(\psi)}(y_{\Lambda/2}^{\Lambda-1}, u_{0,\text{odd}}^{2\psi-1} | u_{2\psi+1})
$$

**Odd phase** ($\phi = 2\psi + 1$):
$$
W_\lambda^{(2\psi+1)}(y_0^{\Lambda-1}, u_0^{2\psi} | u_{2\psi+1}) = \frac{1}{2} W_{\lambda-1}^{(\psi)}(y_0^{\Lambda/2-1}, u_{0,\text{even}}^{2\psi-1} \oplus u_{0,\text{odd}}^{2\psi-1} | u_{2\psi} \oplus u_{2\psi+1}) \cdot W_{\lambda-1}^{(\psi)}(y_{\Lambda/2}^{\Lambda-1}, u_{0,\text{odd}}^{2\psi-1} | u_{2\psi+1})
$$

**Base case** ($\lambda = 0$):
$$
W_0^{(0)}(y | u) = W(y | u)
$$

where $W$ is the underlying physical channel.

**Reference:** [List_Decoding_of_Polar_Codes.md] §II.A Eq. (4)-(5).

</definition>

### 4.3 Branch Numbering

<definition id="branch-number">

During decoding, computations are tracked by a **branch number** $\beta$:

$$
0 \leq \beta < 2^{m - \lambda}
$$

The combined index within layer $\lambda$ uses quotient-remainder representation:

$$
i = (\phi, \beta)_\lambda = \phi + 2^\lambda \cdot \beta
$$

This establishes an array addressing scheme where $P_\lambda[\phi, \beta]$ stores the probability pair for phase $\phi$, branch $\beta$ at layer $\lambda$.

**Reference:** [List_Decoding_of_Polar_Codes.md] §II.B Eq. (6)-(8).

</definition>

---

## 5. LLR-Domain Computation

### 5.1 Decision Log-Likelihood Ratios

<definition id="llr-definition">

For numerically stable decoding, computations are performed in the **log-likelihood ratio (LLR)** domain:

$$
L_m^{(i)} \triangleq \ln\left( \frac{W_m^{(i)}(\mathbf{y}, \hat{\mathbf{u}}_0^{i-1} | 0)}{W_m^{(i)}(\mathbf{y}, \hat{\mathbf{u}}_0^{i-1} | 1)} \right), \quad i \in [N]
$$

**Channel LLRs** (base case):
$$
L_0^{(i)} \triangleq \ln\left( \frac{W(y_i | 0)}{W(y_i | 1)} \right), \quad \forall i \in [N]
$$

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II Eq. (7).

</definition>

### 5.2 LLR Update Rules

<definition id="llr-updates">

The LLR values propagate through the decoding tree via two update functions:

**$f$ function (check-node, "min-sum"):**
$$
f(\alpha, \beta) \triangleq \ln\left( \frac{e^{\alpha+\beta} + 1}{e^{\alpha} + e^{\beta}} \right)
$$

**$g$ function (variable-node, "repetition"):**
$$
g(\alpha, \beta, u) \triangleq (-1)^u \alpha + \beta
$$

**Min-Sum Approximation** (hardware-friendly):
$$
\tilde{f}(\alpha, \beta) = \text{sign}(\alpha) \cdot \text{sign}(\beta) \cdot \min(|\alpha|, |\beta|)
$$

**Recursion for even/odd phases:**
$$
L_s^{(2i)} = f(L_{s-1}^{(i \bmod 2^{s-1})}, L_{s-1}^{(2^{s-1} + i \bmod 2^{s-1})})
$$
$$
L_s^{(2i+1)} = g(L_{s-1}^{(i \bmod 2^{s-1})}, L_{s-1}^{(2^{s-1} + i \bmod 2^{s-1})}, u_s^{(2i)})
$$

for $s = m, m-1, \ldots, 1$.

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II Eq. (8a)-(8b), Eq. (9).

</definition>

### 5.3 Partial Sum Updates

<definition id="partial-sums">

The **partial sums** $u_s^{(i)}$ track decoded bit estimates:

**Initialization:** $u_m^{(i)} \triangleq \hat{u}_i$ for all $i \in [N]$

**Recursion:**
$$
u_{s-1}^{(i \bmod 2^{s-1})} = u_s^{(2i)} \oplus u_s^{(2i+1)}
$$
$$
u_{s-1}^{(2^{s-1} + i \bmod 2^{s-1})} = u_s^{(2i+1)}
$$

for $s = m, m-1, \ldots, 1$.

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II.

</definition>

---

## 6. Permutation Contract

<permutation_contract>

### 6.1 The Ambiguity

Different sources apply $B_N$ at different points. This section resolves the ambiguity by establishing the **canonical convention** for this implementation.

### 6.2 Canonical Convention (Arıkan Standard)

We adopt the convention from [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II Eq. (1):

$$
\mathbf{x} = \mathbf{G}_m \mathbf{u} = (F_2^{\otimes m} B_m) \mathbf{u}
$$

**Interpretation:**
| Vector | Indexing | Description |
|--------|----------|-------------|
| $\mathbf{u}$ | **Natural order** | Information + frozen bits, indexed $u_0, u_1, \ldots, u_{N-1}$ in decoding order |
| $B_m \mathbf{u}$ | Bit-reversed | Intermediate: $\mathbf{u}$ after bit-reversal permutation |
| $\mathbf{x}$ | **Natural order** | Codeword, indexed $x_0, x_1, \ldots, x_{N-1}$ in transmission order |

### 6.3 Equivalent Formulation

By commutativity properties, an equivalent formulation applies $B_N$ to the output:

$$
\mathbf{x}' = B_N (F_2^{\otimes m} \mathbf{u})
$$

This produces codeword $\mathbf{x}'$ in bit-reversed order relative to $\mathbf{x}$:
$$
x'_i = x_{\text{bitrev}_m(i)}
$$

### 6.4 Implementation Contract

<invariants>

**INV-PERM-1:** The encoder accepts $\mathbf{u}$ in natural index order $(u_0, u_1, \ldots, u_{N-1})$.

**INV-PERM-2:** The encoder outputs $\mathbf{x}$ in natural index order $(x_0, x_1, \ldots, x_{N-1})$.

**INV-PERM-3:** The decoder accepts channel LLRs $L_0^{(i)}$ in natural index order.

**INV-PERM-4:** The decoder outputs $\hat{\mathbf{u}}$ in natural index order.

**INV-PERM-5:** Bit-reversal permutation is applied **internally** during encoding and decoding, transparent to the API.
</invariants>

### 6.5 Hardware Optimization Note

For hardware implementations, **bit-reversed indexing** may be used internally for reduced routing complexity, as noted in [Fast_Polar_Decoders_Algorithm_and_Implementation.md] §II.A:

> "In [1], bit-reversed indexing is used, which changes the generator matrix by multiplying it with a bit-reversal operator $B$, so that $G = BF$. [...] it was shown in [3] that bit-reversed indexing significantly reduced data-routing complexity in a hardware implementation."

When bit-reversed indexing is used **internally**, the external API must still conform to **INV-PERM-1** through **INV-PERM-5**.

</permutation_contract>

---

## 7. Frozen Bit Selection

<definition id="frozen-bits">

The $(N, K)$ polar code is defined by the **frozen set** $\mathcal{F} \subset [N]$ with $|\mathcal{F}| = N - K$.

**Selection criterion:** $\mathcal{F}$ contains the indices of the $N - K$ least reliable synthetic channels $W_m^{(i)}$.

**Reliability metrics:**
1. **Bhattacharyya parameter:** $Z(W_m^{(i)})$ — smaller is more reliable
2. **Mutual information:** $I(W_m^{(i)})$ — larger is more reliable
3. **Density evolution** (Gaussian approximation) — channel-specific

**Frozen bit values:** Conventionally $u_i = 0$ for all $i \in \mathcal{F}$. The frozen pattern is known to both encoder and decoder.

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II.A.1.

</definition>

---

## 8. Summary of Key Equations

<summary>

| Concept | Equation | Reference |
|---------|----------|-----------|
| Kernel | $F_2 = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}$ | [List] §II |
| Generator | $\mathbf{G}_m = F_2^{\otimes m} B_m$ | [LLR] §II Eq. (1) |
| Encoding | $\mathbf{x} = \mathbf{G}_m \mathbf{u}$ | [LLR] §II Eq. (1) |
| Bit-reversal | $\text{bitrev}_m(i) = \sum_{k=0}^{m-1} b_k \cdot 2^{m-1-k}$ | [Fast] §II.A |
| Channel LLR | $L_0^{(i)} = \ln\frac{W(y_i|0)}{W(y_i|1)}$ | [LLR] §II Eq. (7) |
| $f$ update | $f(\alpha, \beta) = \ln\frac{e^{\alpha+\beta}+1}{e^\alpha + e^\beta}$ | [LLR] §II Eq. (8a) |
| $g$ update | $g(\alpha, \beta, u) = (-1)^u \alpha + \beta$ | [LLR] §II Eq. (8b) |
| Min-sum | $\tilde{f}(\alpha,\beta) = \text{sign}(\alpha)\text{sign}(\beta)\min(|\alpha|,|\beta|)$ | [LLR] §II Eq. (9) |
| Layer/Phase | $\lambda \in [0, m]$, $\phi \in [0, 2^\lambda)$ | [List] §II.A Eq. (1)-(2) |

**Citation Key:**
- [LLR] = LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md
- [List] = List_Decoding_of_Polar_Codes.md
- [Fast] = Fast_Polar_Decoders_Algorithm_and_Implementation.md

</summary>

---

## References

<references>

1. **[LLR-Based SCL Decoding]** Balatsoukas-Stimming, A., Parizi, M. B., & Burg, A. (2015). *LLR-Based Successive Cancellation List Decoding of Polar Codes.* IEEE Transactions on Signal Processing, 63(19). — Primary source for LLR-domain formulation, Eq. (1), (7)-(9).

2. **[List Decoding of Polar Codes]** Tal, I., & Vardy, A. (2015). *List Decoding of Polar Codes.* IEEE Transactions on Information Theory, 61(5), 2213-2226. — Primary source for layer-phase indexing, bit-channel recursions Eq. (4)-(5).

3. **[Fast Polar Decoders]** Sarkis, G., Giard, P., Vardy, A., Thibeault, C., & Gross, W. J. (2014). *Fast Polar Decoders: Algorithm and Implementation.* IEEE Journal on Selected Areas in Communications, 32(5), 946-957. — Primary source for bit-reversal discussion, hardware indexing.

4. **[Original Polar Codes]** Arıkan, E. (2009). *Channel Polarization: A Method for Constructing Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels.* IEEE Transactions on Information Theory, 55(7), 3051-3073. — Foundational paper (not included in literature directory).

</references>

---

## Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-03 | Context Engineer | Initial mathematical specification |
