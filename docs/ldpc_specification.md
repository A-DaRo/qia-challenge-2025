# LDPC Error Correction and Parameter Estimation Specification

**Version:** 1.0
**Based on:** "Post-processing procedure for industrial quantum key distribution systems" (Kiktenko et al., 2016)
**Context:** QIA Challenge 2025 - E-HOK Protocol Implementation

## 1. Introduction

This document specifies the technical requirements for implementing the **LDPC-based Information Reconciliation** and **Integrated Parameter Estimation** phases of the E-HOK protocol. This approach replaces the Cascade protocol used in earlier iterations, offering higher efficiency close to the Shannon limit and better suitability for high-speed hardware implementations.

The implementation follows a **Reverse Reconciliation** scheme where Bob's key is the reference, and Alice corrects her key to match Bob's.

---

## 2. LDPC Error Correction (Reconciliation)

### 2.1. System Parameters & Matrix Pool

The system utilizes a fixed frame size for LDPC codes but achieves rate adaptability through a pool of matrices and a shortening technique.

*   **Frame Size ($n$):** 4096 bits.
*   **Code Rates ($R$):** A discrete set of rates from 0.5 to 0.9 with a step of 0.05.
    *   $R \in \{0.50, 0.55, 0.60, \dots, 0.90\}$.
*   **Matrix Construction:**
    *   Matrices must be constructed using the **Progressive Edge-Growth (PEG)** algorithm.
    *   Degree distributions should follow the optimized polynomials from Richardson et al. [17].
    *   **Storage:** Matrices are pre-generated and stored. Both Alice and Bob must have access to the identical pool of matrices indexed by rate $R$.

### 2.2. Rate Selection Strategy

For each block of the sifted key, an appropriate code rate is selected dynamically based on the current estimated QBER ($\text{QBER}_{\text{est}}$).

**Selection Criterion:**
Choose the minimal rate $R$ from the pool that satisfies:
$$ \frac{1-R}{h_b(\text{QBER}_{\text{est}})} < f_{\text{crit}} $$

*   $h_b(p)$: Binary entropy function, $h_b(p) = -p \log_2 p - (1-p) \log_2 (1-p)$.
*   $f_{\text{crit}}$: Critical efficiency parameter. **Value:** $1.22$.
    *   *Note:* This represents the tolerable ratio between disclosed information and the Shannon limit.

### 2.3. Shortening Technique

To fine-tune the code rate and decrease the frame error rate (FER) without changing the matrix, a shortening technique is applied. This allows processing a variable number of actual key bits ($m$) within the fixed frame size ($n$).

**1. Calculate Shortened Bits ($n_s$):**
$$ n_s = \left\lfloor n - \frac{m_{\text{target}}}{f_{\text{crit}} \cdot h_b(\text{QBER}_{\text{est}})} \right\rfloor $$
*   *Correction:* In the reference paper, $m$ is the number of information bits. Here, we solve for the number of *shortened* bits to achieve the target efficiency.
*   Effectively, we process $k_{\text{eff}} = n - n_s$ bits of the sifted key per LDPC block.

**2. Block Construction:**
Alice and Bob construct a block of length $n$ composed of:
*   **Payload:** $n - n_s$ bits from their respective sifted keys ($K_{\text{sift}}^A$ and $K_{\text{sift}}^B$).
*   **Padding:** $n_s$ pseudo-random bits.
    *   **Source:** A synchronized Pseudo-Random Number Generator (PRNG) seeded with a shared value (e.g., derived from previous successful blocks or an initial seed).
    *   **Values:** Must be identical for Alice and Bob.
    *   **Position:** Appended to the end or interleaved (fixed convention required).

### 2.4. Syndrome Exchange (Reverse Reconciliation)

1.  **Bob (Reference):**
    *   Selects the matrix $H_R$ based on the rate selection.
    *   Constructs his block $X_B$ (sifted bits + padding).
    *   Computes syndrome $S_B = H_R \cdot X_B \pmod 2$.
    *   Sends $S_B$ to Alice over the classical channel.

2.  **Alice (Correction):**
    *   Receives $S_B$.
    *   Constructs her block $X_A$ (sifted bits + padding).
    *   Computes her syndrome $S_A = H_R \cdot X_A \pmod 2$.
    *   Computes the difference syndrome: $\Delta S = S_A \oplus S_B$.
    *   *Note:* Since the padding bits are identical, they cancel out in the difference, but they stabilize the decoding graph.

### 2.5. Decoding (Belief Propagation)

Alice employs the **Iterative Sum-Product Algorithm** (Belief Propagation) to find the error vector $e$ such that $H_R \cdot e = \Delta S$.

*   **Graph:** Tanner graph corresponding to $H_R$.
*   **Input LLRs (Log-Likelihood Ratios):**
    *   For payload bits: Derived from $\text{QBER}_{\text{est}}$.
        $$ LLR_i = \ln \left( \frac{1 - \text{QBER}_{\text{est}}}{\text{QBER}_{\text{est}}} \right) $$
    *   For padding bits: $LLR_i = \infty$ (since Alice knows these bits match Bob's perfectly, error probability is 0).
*   **Iterations:** Maximum 60 iterations.
*   **Output:**
    *   If converged: Error vector $e$. Alice corrects her key: $X'_A = X_A \oplus e$.
    *   If not converged: Mark block as **Unverified**.

### 2.6. Verification (Polynomial Hashing)

To ensure the corrected key $X'_A$ is identical to $X_B$, a verification step is mandatory.

*   **Algorithm:** $\epsilon$-universal polynomial hashing (PolyR).
*   **Hash Size:** 50 bits.
*   **Collision Probability:** $\epsilon_{\text{ver}} < 2 \times 10^{-12}$.
*   **Procedure:**
    1.  Bob computes hash $T_B = \text{PolyR}(X_B[\text{payload}])$.
    2.  Bob sends $T_B$ to Alice.
    3.  Alice computes $T_A = \text{PolyR}(X'_A[\text{payload}])$.
    4.  **Check:**
        *   If $T_A = T_B$: Block is **Verified**. Added to $K_{\text{ver}}$.
        *   If $T_A \neq T_B$: Block is **Unverified**. Discarded.

---

## 3. Parameter Estimation (Improved Mechanism)

Instead of sacrificing a subset of bits for sampling, this specification uses **Integrated Parameter Estimation**. This method utilizes the results of the error correction process to estimate the QBER, maximizing the key yield.

### 3.1. Calculation Logic

The QBER is estimated over a window of $N$ LDPC blocks (e.g., $N=256$).

$$ \text{QBER}_{\text{est}} = \frac{1}{N} \left( \sum_{i \in V} \text{QBER}_i + \frac{|\bar{V}|}{2} \right) $$

Where:
*   $V$: Set of indices of **Verified** blocks (successful decoding + matching hash).
*   $\bar{V}$: Set of indices of **Unverified** blocks (decoding failed or hash mismatch).
*   $\text{QBER}_i$: The actual error rate observed in block $i$.
    *   Calculated as: $\frac{\text{HammingWeight}(e_i)}{\text{Length}(X_A[\text{payload}])}$.
    *   *Note:* Only count errors in the payload section, not the padding.
*   $|\bar{V}| / 2$: Unverified blocks are conservatively assumed to have a QBER of 0.5 (uncorrelated/fully eavesdropped).

### 3.2. Proposed Class/Function Signature:

```python
@dataclass
class LDPCBlockResult:
    verified: bool
    error_count: int
    block_length: int

def estimate_qber_from_ldpc(
    block_results: List[LDPCBlockResult]
) -> float:
    """
    Estimate QBER from LDPC reconciliation results.
    
    Implements Eq (3) from Kiktenko et al. (2016).
    
    Parameters
    ----------
    block_results : List[LDPCBlockResult]
        List of results from N LDPC blocks.
        
    Returns
    -------
    float
        Estimated QBER.
    """
    total_qber_sum = 0.0
    N = len(block_results)
    
    if N == 0:
        return 0.5 # Conservative default
        
    for res in block_results:
        if res.verified:
            # QBER_i = errors / length
            total_qber_sum += res.error_count / res.block_length
        else:
            # Unverified blocks contribute 0.5
            total_qber_sum += 0.5
            
    return total_qber_sum / N
```

### 3.3. Advantages over Sampling

1.  **Efficiency:** No bits are discarded solely for estimation.
2.  **Accuracy:** Uses the entire key for estimation rather than a small sample.
3.  **Security:** The conservative assumption (0.5 for failures) prevents attacks that selectively jam blocks to hide errors.

---

## 4. Summary of Workflow

1.  **Initialization:** Load LDPC matrices ($R=0.5 \dots 0.9$). Sync PRNG.
2.  **Sifting:** Produce $K_{\text{sift}}^A, K_{\text{sift}}^B$.
3.  **Block Loop:**
    *   Estimate current QBER (from previous rounds or initial sample).
    *   Select Rate $R$ and Shortening $n_s$.
    *   Construct Blocks (Payload + Padding).
    *   **Bob:** Send Syndrome $S_B$ and Hash $T_B$.
    *   **Alice:** Decode $\Delta S$. Verify Hash.
    *   **Result:** Mark as Verified/Unverified.
4.  **Estimation:** Update $\text{QBER}_{\text{est}}$ using the Integrated method.
5.  **Privacy Amplification:** Use the calculated $\text{QBER}_{\text{est}}$ and leakage information (syndrome lengths + hash lengths) to distill the final key.
