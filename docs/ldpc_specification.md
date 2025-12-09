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

This section details the mathematical foundation (Degree Distributions), the construction engine (PEG Algorithm), and the implementation specifications (Storage).

#### 2.1.1. Optimized Degree Distributions (The Polynomials)

The performance of an LDPC code in the "waterfall region" (the signal-to-noise ratio where the error rate drops precipitously) is dictated by its degree distribution. For the Binary Symmetric Channel (BSC) characteristic of QKD, we utilize **irregular** LDPC codes where the degrees of variable nodes (columns) and check nodes (rows) vary according to specific probability distributions.

The degree distributions are defined by two polynomials, $\lambda(x)$ and $\rho(x)$, representing the edge-perspective degree distribution of Variable Nodes (VN) and Check Nodes (CN) respectively.

**1. Mathematical Definitions**
*   **$\lambda(x) = \sum_{i=2}^{d_{v_{max}}} \lambda_i x^{i-1}$**: $\lambda_i$ is the fraction of edges in the graph that connect to a Variable Node of degree $i$.
*   **$\rho(x) = \sum_{j=2}^{d_{c_{max}}} \rho_j x^{j-1}$**: $\rho_j$ is the fraction of edges in the graph that connect to a Check Node of degree $j$.

The code rate $R$ is determined by the distributions as:
$$ R = 1 - \frac{\int_0^1 \rho(x) dx}{\int_0^1 \lambda(x) dx} = 1 - \frac{\sum (\rho_j / j)}{\sum (\lambda_i / i)} $$

**2. Optimized Coefficients (Reference Example: Rate 0.5)**
The specific polynomials used in the system are derived via **Density Evolution** (as described by Richardson et al.) to maximize the threshold on the BSC.

For a code rate $R=0.5$ (the base rate of the pool), the system utilizes the standard high-performance distribution optimized for the BSC (sourced from Richardson/Urbanke [17] & Elkouss et al.):

**Variable Node Distribution $\lambda(x)$:**
The distribution relies heavily on degree-2 and degree-3 nodes to ensure easy decoding startup, with high-degree nodes to ensure connectivity and low error floors.

| Degree ($i$) | Coefficient ($\lambda_i$) |
| :--- | :--- |
| 2 | 0.234029 |
| 3 | 0.212425 |
| 6 | 0.146898 |
| 7 | 0.102840 |
| 13 | 0.000780 |
| 14 | 0.000320 |
| 18 | 0.302708 |

$$ \lambda(x) = 0.2340x + 0.2124x^2 + 0.1469x^5 + 0.1028x^6 + 0.0008x^{12} + 0.0003x^{13} + 0.3027x^{17} $$

**Check Node Distribution $\rho(x)$:**
Check nodes are typically "concentrated" (nearly regular) to minimize processing complexity. For Rate 0.5:
$$ \rho(x) = 0.7187x^7 + 0.2813x^8 $$
*(Approximately 71.9% of edges connect to degree-8 check nodes, and 28.1% to degree-9 check nodes).*

**Note on Higher Rates ($R > 0.5$):**
For rates 0.55 through 0.90, the system uses "puncturing" optimizations or distinct polynomials derived via the same Density Evolution method, typically maintaining a high concentration of degree-2 variable nodes ($\lambda_2 \approx 0.2-0.3$) to support the belief propagation flow.

#### 2.1.2. The Progressive Edge-Growth (PEG) Algorithm

While the polynomials define *how many* nodes of each degree exist, they do not define *where* the edges connect. Random connection often creates short cycles (girth 4), which devastate decoding performance.

The **PEG Algorithm** (Hu et al. [16]) constructs the graph edge-by-edge to maximize the **local girth** (length of the shortest cycle) for every variable node.

**1. Algorithm Logic**
We denote the graph as $(V, C, E)$, where $V$ are variable nodes and $C$ are check nodes.
For a variable node $v_j$, we expand a tree to depth $l$ to find the set of check nodes $N^l_{v_j}$ that are reachable from $v_j$ within $l$ hops. We place the new edge $(v_j, c)$ connecting to a check node $c$ that is **not** in $N^l_{v_j}$ (maximizing girth) or is in the furthest possible layer.

**2. Full Pseudo-Code Implementation**

```python
def generate_peg_matrix(n, m, lambda_coeffs, rho_coeffs):
    """
    Constructs LDPC Matrix H using PEG algorithm.
    n: Block length (4096)
    m: Number of check nodes (n * (1-R))
    lambda_coeffs: List of tuples (degree, probability_lambda)
    rho_coeffs: List of tuples (degree, probability_rho) (Used to cap CN degrees)
    """
    
    # 1. Convert edge-perspective probabilities to integer node degrees
    # Calculate total edges E based on distributions
    # Assign target degrees to Variable Nodes (V) and Check Nodes (C)
    # Sort V in non-decreasing order of degree (critical for PEG performance)
    V_nodes = assign_degrees(n, lambda_coeffs, sort=True) 
    C_nodes = assign_degrees(m, rho_coeffs, sort=False)
    
    # Current degrees of check nodes (initially 0)
    current_cn_degrees = [0] * m
    
    # Adjacency list for the graph (Check Nodes to Variable Nodes)
    H_graph = [set() for _ in range(m)] 
    
    # 2. Iterate through every Variable Node v_j from 0 to n-1
    for j in range(n):
        degree_v = V_nodes[j].target_degree
        
        # Add k edges for this variable node
        for k in range(degree_v):
            
            if k == 0:
                # First edge: Select CN with minimum current degree to balance load
                c_selected = get_min_degree_check_node(current_cn_degrees)
            else:
                # Subsequent edges: Use Tree Expansion to maximize girth
                c_selected = peg_tree_expand(j, H_graph, m, current_cn_degrees)
            
            # 3. Update Graph State
            H_graph[c_selected].add(j)
            current_cn_degrees[c_selected] += 1
            
    return convert_to_csr_matrix(H_graph)

def peg_tree_expand(v_idx, H_graph, m, current_cn_degrees):
    """
    Performs BFS from variable node v_idx to find best check node candidate.
    """
    # Initialize layers for BFS
    # N_l contains check nodes reachable at depth l
    N_l = set()
    
    # We also track Variable nodes visited to prevent cycles in BFS
    visited_V = {v_idx}
    visited_C = set()
    
    # Root of tree is v_idx. 
    # Level 0: Check nodes currently connected to v_idx
    # Since H_graph stores C->V, we must look at C nodes that contain v_idx
    # Optimization: In real impl, maintain V->C list for fast expansion
    current_level_C = [c for c, v_list in enumerate(H_graph) if v_idx in v_list]
    
    for c in current_level_C:
        visited_C.add(c)
        N_l.add(c)
        
    l = 0
    max_depth = 10 # Heuristic limit to prevent infinite loops in dense graphs
    
    while l < max_depth:
        # Check if the complement of N_l (unreachable nodes) is non-empty
        if len(N_l) == m:
            break # All check nodes are reachable, stop expansion
            
        # Expand one level deeper:
        # 1. From current Check Nodes, find all connected Variable Nodes
        next_level_V = set()
        for c in current_level_C:
             # Get all V neighbors of check node c, excluding visited
            neighbors = H_graph[c]
            for v in neighbors:
                if v not in visited_V:
                    next_level_V.add(v)
                    visited_V.add(v)
        
        if not next_level_V:
            break # Graph connectivity stops
            
        # 2. From these Variable Nodes, find all connected Check Nodes
        next_level_C = []
        cardinality_increase = False
        
        for v in next_level_V:
            # (In full implementation, need V->C adjacency list here)
            # Find neighbors of v
            c_neighbors = get_check_neighbors(v) 
            for c in c_neighbors:
                if c not in visited_C:
                    visited_C.add(c)
                    N_l.add(c)
                    next_level_C.append(c)
                    cardinality_increase = True
        
        if not cardinality_increase:
            break
            
        current_level_C = next_level_C
        l += 1

    # SELECTION STRATEGY:
    # Candidate set Omega = All Check Nodes NOT in N_l (Reachable set)
    Omega = [c for c in range(m) if c not in N_l]
    
    if Omega:
        # If candidates exist that create NO cycles of length 2*(l+1), pick one.
        # Tie-breaker: Pick the one with minimum current degree.
        return select_min_degree(Omega, current_cn_degrees)
    else:
        # If all nodes are reachable, we must create a cycle.
        # Select a node from the furthest level reached (maximize cycle length).
        return select_min_degree(current_level_C, current_cn_degrees)
```

#### 2.1.3. Matrix Storage and Access

Since the PEG algorithm is computationally intensive, matrices are not generated on-the-fly. They are generated once ("offline") and stored permanently in the system.

**1. The Matrix Pool**
*   **Format:** Compressed Sparse Row (CSR). This is critical for the efficiency of the sum-product decoding algorithm.
    *   `ptr`: Array of indices pointing to the start of each row (check node).
    *   `indices`: Array containing the column indices (variable nodes) for the non-zero elements.
    *   `data`: Array of 1s (implicit in binary LDPC, often omitted).
*   **File Structure:** A binary file containing the header (N, M, Rate) followed by the CSR arrays.
    *   `ldpc_4096_rate0.50.npz`
    *   `ldpc_4096_rate0.55.npz`
    *   ...
    *   `ldpc_4096_rate0.90.npz`

**2. Synchronization**
*   Alice and Bob must possess identical binary files.
*   **Checksumming:** Upon system initialization, the SHA-256 hash of the matrix pool is compared between Alice and Bob via the public channel (authenticated) to guarantee the `indices` arrays are perfectly aligned. Any discrepancy results in a `System Halt`.

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

---

## 5. Implementation Plan

### 5.1. Architecture Overview

The LDPC implementation replaces the existing Cascade-based reconciliation with a modular, rate-adaptive LDPC system. This implementation follows **Reverse Reconciliation** (Bob = Reference, Alice = Corrects).

**Design Principles:**
*   **No Backward Compatibility:** This is the new baseline.
*   **Modular Design:** Separate concerns (matrix management, syndrome computation, decoding, verification).
*   **Testability:** All components must be unit-testable in isolation.
*   **Type Safety:** Strict type hints throughout.

### 5.2. Core Data Structures

#### 5.2.1. `LDPCBlockResult` (ehok/core/data_structures.py)

```python
@dataclass
class LDPCBlockResult:
    """
    Result of processing a single LDPC block during reconciliation.
    
    Used for integrated QBER estimation and tracking reconciliation progress.
    
    Attributes
    ----------
    verified : bool
        True if block passed hash verification, False otherwise.
    error_count : int
        Number of bit errors detected in this block (Hamming weight of error vector).
    block_length : int
        Length of the payload (excludes padding bits).
    syndrome_length : int
        Number of syndrome bits transmitted for this block.
    hash_bits : int
        Number of hash verification bits transmitted (default 50).
    
    Notes
    -----
    Unverified blocks contribute 0.5 to QBER estimation (conservative assumption).
    """
    verified: bool
    error_count: int
    block_length: int
    syndrome_length: int
    hash_bits: int = 50
```

#### 5.2.2. `LDPCMatrixPool` (ehok/core/data_structures.py)

```python
@dataclass
class LDPCMatrixPool:
    """
    Pool of pre-generated LDPC matrices at different code rates.
    
    Attributes
    ----------
    frame_size : int
        Fixed frame size n = 4096 bits.
    matrices : Dict[float, sp.spmatrix]
        Dictionary mapping code rate R to parity check matrix H.
        Keys: {0.50, 0.55, 0.60, ..., 0.90}.
    rates : np.ndarray
        Sorted array of available code rates for binary search.
    checksum : str
        SHA-256 checksum of the matrix pool for synchronization verification.
    
    Notes
    -----
    Matrices must be constructed using PEG algorithm with optimized degree distributions.
    Both Alice and Bob must load identical matrices (deterministic construction or shared files).
    The checksum ensures synchronization and detects file corruption or tampering.
    """
    frame_size: int
    matrices: Dict[float, sp.spmatrix]
    rates: np.ndarray
    checksum: str
```

#### 5.2.3. `LDPCReconciliationResult` (ehok/core/data_structures.py)

```python
@dataclass
class LDPCReconciliationResult:
    """
    Complete result of LDPC reconciliation phase.
    
    Attributes
    ----------
    corrected_key : np.ndarray
        Final reconciled key (verified blocks only).
    qber_estimate : float
        Estimated QBER from integrated estimation.
    total_leakage : int
        Total information leakage (syndrome + hash bits).
    blocks_processed : int
        Total number of LDPC blocks processed.
    blocks_verified : int
        Number of blocks that passed verification.
    blocks_discarded : int
        Number of blocks discarded due to decoding failure or hash mismatch.
    """
    corrected_key: np.ndarray
    qber_estimate: float
    total_leakage: int
    blocks_processed: int
    blocks_verified: int
    blocks_discarded: int
```

### 5.3. Interface Refactoring

#### 5.3.1. Updated `IReconciliator` Interface (ehok/interfaces/reconciliation.py)

**BREAKING CHANGE:** The interface is redesigned for block-based, iterative reconciliation.

```python
class IReconciliator(ABC):
    """
    Abstract interface for LDPC-based information reconciliation.
    
    This interface supports block-based reconciliation with integrated QBER estimation
    following the Kiktenko et al. (2016) industrial QKD post-processing procedure.
    """
    
    @abstractmethod
    def select_rate(self, qber_est: float) -> float:
        """
        Select optimal LDPC code rate for current estimated QBER.
        
        Parameters
        ----------
        qber_est : float
            Current estimated QBER.
        
        Returns
        -------
        rate : float
            Selected code rate R from the pool.
        
        Notes
        -----
        Selection criterion: minimal R satisfying (1-R) / h_b(QBER_est) < f_crit.
        """
        pass
    
    @abstractmethod
    def compute_shortening(self, rate: float, qber_est: float, target_payload: int) -> int:
        """
        Calculate number of shortened bits for a given rate and target payload.
        
        Parameters
        ----------
        rate : float
            Selected LDPC code rate.
        qber_est : float
            Current estimated QBER.
        target_payload : int
            Desired number of sifted key bits to process.
        
        Returns
        -------
        n_s : int
            Number of shortened (padding) bits to add.
        """
        pass
    
    @abstractmethod
    def reconcile_block(
        self,
        key_block: np.ndarray,
        syndrome: np.ndarray,
        rate: float,
        n_shortened: int,
        prng_seed: int
    ) -> Tuple[np.ndarray, bool, int]:
        """
        Reconcile a single LDPC block (Alice's side: decode).
        
        Parameters
        ----------
        key_block : np.ndarray
            Alice's noisy key block (payload only, length < frame_size).
        syndrome : np.ndarray
            Difference syndrome received from Bob.
        rate : float
            LDPC code rate used for this block.
        n_shortened : int
            Number of shortened bits to add as padding.
        prng_seed : int
            Synchronized PRNG seed for padding generation.
        
        Returns
        -------
        corrected_block : np.ndarray
            Error-corrected key block (payload only).
        converged : bool
            True if BP decoder converged, False otherwise.
        error_count : int
            Number of errors corrected (Hamming weight of error vector).
        
        Notes
        -----
        - Constructs full frame (payload + padding).
        - Runs BP decoding on difference syndrome.
        - Returns only the payload portion.
        """
        pass
    
    @abstractmethod
    def compute_syndrome_block(
        self,
        key_block: np.ndarray,
        rate: float,
        n_shortened: int,
        prng_seed: int
    ) -> np.ndarray:
        """
        Compute syndrome for a single block (Bob's side: reference).
        
        Parameters
        ----------
        key_block : np.ndarray
            Bob's key block (payload only).
        rate : float
            LDPC code rate.
        n_shortened : int
            Number of shortened bits.
        prng_seed : int
            PRNG seed for padding.
        
        Returns
        -------
        syndrome : np.ndarray
            Syndrome S = H @ (payload || padding) mod 2.
        """
        pass
    
    @abstractmethod
    def verify_block(self, block_alice: np.ndarray, block_bob: np.ndarray) -> Tuple[bool, bytes]:
        """
        Verify that corrected block matches reference using polynomial hashing.
        
        Parameters
        ----------
        block_alice : np.ndarray
            Alice's corrected block.
        block_bob : np.ndarray
            Bob's reference block.
        
        Returns
        -------
        verified : bool
            True if hashes match.
        hash_value : bytes
            50-bit hash for transmission/comparison.
        """
        pass
    
    @abstractmethod
    def estimate_leakage_block(self, syndrome_length: int, hash_bits: int = 50) -> int:
        """
        Estimate information leakage for a single block.
        
        Parameters
        ----------
        syndrome_length : int
            Length of transmitted syndrome.
        hash_bits : int
            Number of verification hash bits (default 50).
        
        Returns
        -------
        leakage : int
            Total leakage in bits.
        """
        pass
```

### 5.4. Implementation Classes

#### 5.4.1. `PEGMatrixGenerator` (ehok/implementations/reconciliation/peg_generator.py)

**Responsibility:** Generate LDPC matrices using the PEG algorithm with optimized degree distributions.

```python
@dataclass
class DegreeDistribution:
    """
    Degree distribution specification for LDPC code.
    
    Attributes
    ----------
    degrees : List[int]
        List of node degrees.
    probabilities : List[float]
        Edge-perspective probability for each degree (lambda_i or rho_j).
    
    Notes
    -----
    Sum of probabilities must equal 1.0.
    """
    degrees: List[int]
    probabilities: List[float]
    
    def __post_init__(self):
        assert abs(sum(self.probabilities) - 1.0) < 1e-6, "Probabilities must sum to 1"

class PEGMatrixGenerator:
    """
    Progressive Edge-Growth algorithm for LDPC matrix construction.
    
    Constructs parity check matrices that maximize local girth to improve
    belief propagation performance.
    
    Parameters
    ----------
    n : int
        Block length (number of variable nodes).
    rate : float
        Target code rate.
    lambda_dist : DegreeDistribution
        Variable node degree distribution.
    rho_dist : DegreeDistribution
        Check node degree distribution.
    max_tree_depth : int
        Maximum BFS depth for girth maximization (default 10).
    seed : Optional[int]
        Random seed for tie-breaking (default None).
    
    Notes
    -----
    Implementation follows Hu et al. [16] with optimizations for large graphs.
    """
    
    def __init__(
        self,
        n: int,
        rate: float,
        lambda_dist: DegreeDistribution,
        rho_dist: DegreeDistribution,
        max_tree_depth: int = 10,
        seed: Optional[int] = None
    ) -> None:
        ...
    
    def generate(self) -> sp.csr_matrix:
        """
        Generate LDPC parity check matrix using PEG algorithm.
        
        Returns
        -------
        H : sp.csr_matrix
            Parity check matrix in CSR format, shape (m, n).
        
        Notes
        -----
        Algorithm steps:
        1. Assign target degrees to all variable and check nodes.
        2. Sort variable nodes by degree (non-decreasing).
        3. For each variable node, add edges using tree expansion to maximize girth.
        4. Convert adjacency structure to CSR matrix.
        """
        ...
    
    def _assign_node_degrees(self) -> Tuple[List[int], List[int]]:
        """
        Convert edge-perspective distributions to node degree assignments.
        
        Returns
        -------
        vn_degrees : List[int]
            Target degree for each variable node (length n).
        cn_degrees : List[int]
            Target degree for each check node (length m).
        """
        ...
    
    def _peg_tree_expand(
        self,
        vn_idx: int,
        adjacency: List[Set[int]],
        current_cn_degrees: List[int]
    ) -> int:
        """
        Perform BFS tree expansion to find optimal check node connection.
        
        Parameters
        ----------
        vn_idx : int
            Variable node index to connect.
        adjacency : List[Set[int]]
            Current graph adjacency (CN -> VN).
        current_cn_degrees : List[int]
            Current degree of each check node.
        
        Returns
        -------
        cn_idx : int
            Selected check node index that maximizes girth.
        
        Notes
        -----
        Implements Algorithm 1 from Hu et al. [16]:
        1. Expand BFS tree from vn_idx to depth l.
        2. Find check nodes NOT reachable (candidates).
        3. If candidates exist, select minimum-degree node.
        4. Otherwise, select from furthest reachable layer.
        """
        ...
    
    def _build_vn_to_cn_index(self, adjacency: List[Set[int]]) -> List[List[int]]:
        """
        Build reverse index (VN -> CN) from CN -> VN adjacency.
        
        Optimization for fast tree expansion.
        """
        ...
    
    def verify_matrix_properties(self, H: sp.csr_matrix) -> Dict[str, Any]:
        """
        Verify generated matrix satisfies design specifications.
        
        Returns
        -------
        properties : Dict[str, Any]
            - 'rate': Actual code rate
            - 'avg_vn_degree': Average variable node degree
            - 'avg_cn_degree': Average check node degree
            - 'min_girth': Minimum girth detected (approximate)
            - 'cycle_4_count': Number of 4-cycles (should be 0)
        
        Raises
        ------
        ValueError
            If matrix violates critical properties.
        """
        ...

# Optimized degree distributions (Section 2.1.1 reference)
DEGREE_DISTRIBUTIONS = {
    0.50: {
        'lambda': DegreeDistribution(
            degrees=[2, 3, 6, 7, 13, 14, 18],
            probabilities=[0.234029, 0.212425, 0.146898, 0.102840, 0.000780, 0.000320, 0.302708]
        ),
        'rho': DegreeDistribution(
            degrees=[8, 9],
            probabilities=[0.7187, 0.2813]
        )
    },
    # Additional rates 0.55-0.90 would be defined here with their optimized distributions
}
```

#### 5.4.2. `LDPCMatrixManager` (ehok/implementations/reconciliation/ldpc_matrix_manager.py)

**Responsibility:** Load, store, provide access to LDPC matrices, and verify synchronization.

```python
class LDPCMatrixManager:
    """
    Manager for LDPC matrix pool with rate selection and synchronization.
    
    Parameters
    ----------
    matrix_directory : Path
        Directory containing pre-generated LDPC matrix files.
    frame_size : int
        Fixed frame size (default 4096).
    f_crit : float
        Critical efficiency parameter (default 1.22).
    verify_checksums : bool
        If True, compute and verify SHA-256 checksums of loaded matrices (default True).
    """
    
    def __init__(
        self,
        matrix_directory: Path,
        frame_size: int = 4096,
        f_crit: float = 1.22,
        verify_checksums: bool = True
    ) -> None:
        ...
    
    def load_matrices(self) -> LDPCMatrixPool:
        """
        Load all LDPC matrices from files.
        
        Returns
        -------
        pool : LDPCMatrixPool
            Complete matrix pool.
        
        Notes
        -----
        Expected file naming: ldpc_4096_rate{R:.2f}.npz
        Example: ldpc_4096_rate0.50.npz, ldpc_4096_rate0.55.npz, ...
        
        File format (scipy.sparse.save_npz):
        - 'data': CSR data array (implicit 1s for binary)
        - 'indices': CSR column indices
        - 'indptr': CSR row pointers
        - 'shape': Matrix dimensions (m, n)
        """
        ...
    
    def compute_pool_checksum(self) -> str:
        """
        Compute SHA-256 checksum of entire matrix pool.
        
        Returns
        -------
        checksum : str
            Hexadecimal SHA-256 hash of all matrix indices concatenated.
        
        Notes
        -----
        Checksum is computed over the CSR 'indices' arrays for all rates.
        This ensures Alice and Bob have bit-identical matrix structures.
        Order: R=0.50, 0.55, ..., 0.90.
        """
        ...
    
    def verify_synchronization(self, peer_checksum: str) -> bool:
        """
        Verify that local matrix pool matches peer's pool.
        
        Parameters
        ----------
        peer_checksum : str
            Checksum received from peer via authenticated classical channel.
        
        Returns
        -------
        synchronized : bool
            True if checksums match.
        
        Raises
        ------
        MatrixSynchronizationError
            If checksums do not match (critical security failure).
        
        Notes
        -----
        This method MUST be called during protocol initialization before
        any reconciliation blocks are processed. Mismatch indicates either:
        1. Different matrix files (configuration error)
        2. File corruption
        3. Potential tampering (security breach)
        """
        ...
    
    def select_rate(self, qber_est: float) -> float:
        """
        Select minimal rate satisfying efficiency criterion.
        
        Parameters
        ----------
        qber_est : float
            Current estimated QBER.
        
        Returns
        -------
        rate : float
            Selected code rate from pool.
        
        Notes
        -----
        Implements: (1-R) / h_b(qber_est) < f_crit
        Uses binary search on sorted rate array for O(log k) complexity.
        """
        ...
    
    def get_matrix(self, rate: float) -> sp.spmatrix:
        """
        Retrieve parity check matrix for given rate.
        
        Parameters
        ----------
        rate : float
            Code rate (must exist in pool).
        
        Returns
        -------
        H : sp.csr_matrix
            Parity check matrix.
        
        Raises
        ------
        ValueError
            If rate not in pool.
        """
        ...
    
    @staticmethod
    def binary_entropy(p: float) -> float:
        """
        Compute binary entropy h_b(p).
        
        Parameters
        ----------
        p : float
            Probability in [0, 1].
        
        Returns
        -------
        entropy : float
            h_b(p) = -p*log2(p) - (1-p)*log2(1-p)
        
        Notes
        -----
        Handles edge cases: h_b(0) = h_b(1) = 0.
        """
        ...
```

#### 5.4.3. `LDPCBeliefPropagation` (ehok/implementations/reconciliation/ldpc_bp_decoder.py)

**Responsibility:** Implement Sum-Product (BP) decoder with LLR initialization.

```python
class LDPCBeliefPropagation:
    """
    Iterative Sum-Product Algorithm for LDPC decoding.
    
    Parameters
    ----------
    H : sp.spmatrix
        Parity check matrix.
    max_iterations : int
        Maximum BP iterations (default 60).
    llr_scale : float
        Scaling factor for LLR initialization.
    """
    
    def __init__(
        self,
        H: sp.spmatrix,
        max_iterations: int = 60,
        llr_scale: float = 1.0
    ) -> None:
        ...
    
    def decode(
        self,
        received_block: np.ndarray,
        target_syndrome: np.ndarray,
        qber_est: float,
        padding_mask: np.ndarray
    ) -> Tuple[np.ndarray, bool, int]:
        """
        Decode a block to match target syndrome.
        
        Parameters
        ----------
        received_block : np.ndarray
            Alice's noisy block (payload + padding).
        target_syndrome : np.ndarray
            Difference syndrome ΔS = S_A ⊕ S_B.
        qber_est : float
            Estimated QBER for LLR initialization.
        padding_mask : np.ndarray
            Boolean mask: True for padding bits (LLR=∞), False for payload.
        
        Returns
        -------
        corrected_block : np.ndarray
            Corrected block.
        converged : bool
            True if converged to target syndrome.
        iterations_used : int
            Number of iterations before convergence or timeout.
        
        Notes
        -----
        - LLR initialization: ln((1-QBER)/QBER) for payload, ∞ for padding.
        - Convergence check: H @ corrected_block = target_syndrome (mod 2).
        """
        ...
    
    def _initialize_llrs(
        self,
        block: np.ndarray,
        qber_est: float,
        padding_mask: np.ndarray
    ) -> np.ndarray:
        """Initialize log-likelihood ratios."""
        ...
    
    def _sum_product_iteration(self, llrs: np.ndarray) -> np.ndarray:
        """Perform one iteration of sum-product message passing."""
        ...
    
    def _compute_aposteriori_llrs(self, llrs: np.ndarray) -> np.ndarray:
        """Compute a posteriori LLRs for hard decision."""
        ...
```

#### 5.4.3. `PolynomialHashVerifier` (ehok/implementations/reconciliation/polynomial_hash.py)

**Responsibility:** ε-universal polynomial hashing for block verification.

```python
class PolynomialHashVerifier:
    """
    ε-universal polynomial hash for LDPC block verification.
    
    Parameters
    ----------
    hash_bits : int
        Hash size in bits (default 50).
    seed : Optional[int]
        Seed for hash polynomial generation (must match between Alice/Bob).
    
    Notes
    -----
    Collision probability: ε_ver < 2 × 10^-12 for 50-bit hash.
    """
    
    def __init__(self, hash_bits: int = 50, seed: Optional[int] = None) -> None:
        ...
    
    def compute_hash(self, data: np.ndarray) -> bytes:
        """
        Compute polynomial hash of data.
        
        Parameters
        ----------
        data : np.ndarray
            Input bit array.
        
        Returns
        -------
        hash_value : bytes
            Hash value (50 bits packed into bytes).
        """
        ...
    
    def verify(self, data1: np.ndarray, data2: np.ndarray) -> bool:
        """
        Verify that two blocks have matching hashes.
        
        Returns
        -------
        match : bool
            True if hashes match.
        """
        ...
```

#### 5.4.4. `LDPCReconciliator` (ehok/implementations/reconciliation/ldpc_reconciliator.py)

**Responsibility:** Main reconciliation orchestrator implementing `IReconciliator`.

```python
class LDPCReconciliator(IReconciliator):
    """
    LDPC-based reconciliation with integrated QBER estimation.
    
    Implements reverse reconciliation: Bob (reference) sends syndrome,
    Alice (corrector) decodes and corrects her key.
    
    Parameters
    ----------
    matrix_manager : LDPCMatrixManager
        Manager for LDPC matrix pool.
    hash_verifier : PolynomialHashVerifier
        Block verification hasher.
    frame_size : int
        LDPC frame size (default 4096).
    max_iterations : int
        Maximum BP iterations (default 60).
    f_crit : float
        Critical efficiency parameter (default 1.22).
    """
    
    def __init__(
        self,
        matrix_manager: LDPCMatrixManager,
        hash_verifier: PolynomialHashVerifier,
        frame_size: int = 4096,
        max_iterations: int = 60,
        f_crit: float = 1.22
    ) -> None:
        ...
    
    def select_rate(self, qber_est: float) -> float:
        """Delegate to matrix manager."""
        ...
    
    def compute_shortening(
        self, rate: float, qber_est: float, target_payload: int
    ) -> int:
        """
        Calculate shortening: n_s = floor(n - m_target / (f_crit * h_b(qber_est))).
        """
        ...
    
    def reconcile_block(
        self,
        key_block: np.ndarray,
        syndrome: np.ndarray,
        rate: float,
        n_shortened: int,
        prng_seed: int
    ) -> Tuple[np.ndarray, bool, int]:
        """
        Reconcile single block (Alice's side).
        
        Steps:
        1. Generate padding using synchronized PRNG.
        2. Construct full block (payload || padding).
        3. Compute Alice's syndrome S_A.
        4. Compute difference syndrome ΔS = S_A ⊕ syndrome.
        5. Run BP decoder to find error vector e.
        6. Correct: key' = key ⊕ e.
        7. Return corrected payload (strip padding).
        """
        ...
    
    def compute_syndrome_block(
        self,
        key_block: np.ndarray,
        rate: float,
        n_shortened: int,
        prng_seed: int
    ) -> np.ndarray:
        """
        Compute syndrome (Bob's side).
        
        Steps:
        1. Generate padding using PRNG.
        2. Construct block (payload || padding).
        3. Compute S_B = H @ block (mod 2).
        4. Return syndrome.
        """
        ...
    
    def verify_block(
        self, block_alice: np.ndarray, block_bob: np.ndarray
    ) -> Tuple[bool, bytes]:
        """Delegate to hash verifier."""
        ...
    
    def estimate_leakage_block(self, syndrome_length: int, hash_bits: int = 50) -> int:
        """Leakage = syndrome_length + hash_bits."""
        ...
    
    def _generate_padding(self, n_shortened: int, prng_seed: int) -> np.ndarray:
        """Generate synchronized padding bits."""
        ...
```

#### 5.4.5. `IntegratedQBEREstimator` (ehok/implementations/reconciliation/qber_estimator.py)

**Responsibility:** Implement integrated QBER estimation from LDPC results.

```python
class IntegratedQBEREstimator:
    """
    Integrated QBER estimation from LDPC block results.
    
    Implements Eq (3) from Kiktenko et al. (2016).
    
    Parameters
    ----------
    window_size : int
        Number of blocks to use for estimation (default 256).
    """
    
    def __init__(self, window_size: int = 256) -> None:
        ...
    
    def estimate_qber(self, block_results: List[LDPCBlockResult]) -> float:
        """
        Estimate QBER from block results.
        
        QBER_est = (1/N) * (Σ_{i∈V} QBER_i + |V̄|/2)
        
        Parameters
        ----------
        block_results : List[LDPCBlockResult]
            Results from N LDPC blocks.
        
        Returns
        -------
        qber_est : float
            Estimated QBER.
        
        Notes
        -----
        - Verified blocks contribute their measured QBER.
        - Unverified blocks conservatively contribute 0.5.
        """
        ...
    
    def update_rolling_estimate(self, new_result: LDPCBlockResult) -> float:
        """Update QBER estimate with new block result (rolling window)."""
        ...
```

### 5.5. Protocol Integration

**CRITICAL:** Matrix pool synchronization MUST occur during initialization before any reconciliation.

#### 5.5.0. Initialization Phase: Matrix Pool Synchronization

Both Alice and Bob must verify matrix pool synchronization before protocol execution.

```python
def _initialize_ldpc_reconciliation(self) -> Generator[EventExpression, None, None]:
    """
    Initialize LDPC reconciliation with matrix pool synchronization.
    
    This method MUST be called before any reconciliation blocks are processed.
    Ensures Alice and Bob have bit-identical LDPC matrix pools.
    
    Yields
    ------
    EventExpression
        Classical communication events for checksum exchange.
    
    Raises
    ------
    MatrixSynchronizationError
        If matrix pool checksums do not match.
    """
    logger.info("=== LDPC INITIALIZATION: Matrix Pool Synchronization ===")
    
    # Compute local checksum
    local_checksum = self.matrix_manager.compute_pool_checksum()
    logger.info(f"Local matrix pool checksum: {local_checksum[:16]}...")
    
    # Exchange checksums (authenticated classical channel)
    if self.is_alice:
        # Alice sends first
        self.context.csockets[self.PEER_NAME].send(local_checksum)
        peer_checksum_msg = yield from self.context.csockets[self.PEER_NAME].recv()
        peer_checksum = peer_checksum_msg
    else:
        # Bob receives first
        peer_checksum_msg = yield from self.context.csockets[self.PEER_NAME].recv()
        peer_checksum = peer_checksum_msg
        self.context.csockets[self.PEER_NAME].send(local_checksum)
    
    # Verify synchronization
    if not self.matrix_manager.verify_synchronization(peer_checksum):
        error_msg = (
            f"Matrix pool synchronization FAILED. "
            f"Local: {local_checksum[:16]}..., "
            f"Peer: {peer_checksum[:16]}..."
        )
        logger.critical(error_msg)
        raise MatrixSynchronizationError(error_msg)
    
    logger.info("Matrix pool synchronization VERIFIED. Protocol may proceed.")
```

#### 5.5.1. Alice's Reconciliation Phase (ehok/protocols/alice.py)

```python
def _phase4_reconciliation(
    self, alice_key: np.ndarray
) -> Generator[EventExpression, None, LDPCReconciliationResult]:
    """
    Phase 4: LDPC Reconciliation (Alice = Corrector).
    
    Yields
    ------
    EventExpression
        Classical communication events.
    
    Returns
    -------
    result : LDPCReconciliationResult
        Complete reconciliation result with corrected key and QBER estimate.
    """
    logger.info("=== PHASE 4: LDPC Information Reconciliation (Alice) ===")
    
    # Initialize
    qber_est = 0.05  # Initial conservative estimate
    block_results: List[LDPCBlockResult] = []
    corrected_blocks: List[np.ndarray] = []
    total_leakage = 0
    
    key_length = len(alice_key)
    target_payload = self.reconciliator.frame_size - 200  # Reserve for shortening
    blocks_to_process = (key_length + target_payload - 1) // target_payload
    
    for block_idx in range(blocks_to_process):
        # Extract payload
        start = block_idx * target_payload
        end = min(start + target_payload, key_length)
        key_block = alice_key[start:end]
        
        # Select rate and shortening
        rate = self.reconciliator.select_rate(qber_est)
        n_shortened = self.reconciliator.compute_shortening(rate, qber_est, len(key_block))
        
        # Receive syndrome and hash from Bob
        syndrome_msg = yield from self.context.csockets[self.PEER_NAME].recv()
        syndrome = np.frombuffer(bytes.fromhex(syndrome_msg), dtype=np.uint8)
        
        hash_msg = yield from self.context.csockets[self.PEER_NAME].recv()
        bob_hash = bytes.fromhex(hash_msg)
        
        # Reconcile block
        prng_seed = self._derive_prng_seed(block_idx)
        corrected_block, converged, error_count = self.reconciliator.reconcile_block(
            key_block, syndrome, rate, n_shortened, prng_seed
        )
        
        # Verify
        alice_hash = self.reconciliator.hash_verifier.compute_hash(corrected_block)
        verified = (alice_hash == bob_hash)
        
        if verified:
            corrected_blocks.append(corrected_block)
            block_results.append(LDPCBlockResult(
                verified=True,
                error_count=error_count,
                block_length=len(key_block),
                syndrome_length=len(syndrome),
                hash_bits=50
            ))
        else:
            # Discard unverified block
            block_results.append(LDPCBlockResult(
                verified=False,
                error_count=0,
                block_length=len(key_block),
                syndrome_length=len(syndrome),
                hash_bits=50
            ))
        
        # Update QBER estimate
        qber_est = self.qber_estimator.estimate_qber(block_results[-256:])
        
        # Track leakage
        total_leakage += self.reconciliator.estimate_leakage_block(len(syndrome), 50)
    
    # Concatenate verified blocks
    final_key = np.concatenate(corrected_blocks)
    
    return LDPCReconciliationResult(
        corrected_key=final_key,
        qber_estimate=qber_est,
        total_leakage=total_leakage,
        blocks_processed=blocks_to_process,
        blocks_verified=len(corrected_blocks),
        blocks_discarded=blocks_to_process - len(corrected_blocks)
    )
```

#### 5.5.2. Bob's Reconciliation Phase (ehok/protocols/bob.py)

```python
def _phase4_reconciliation(
    self, bob_key: np.ndarray
) -> Generator[EventExpression, None, LDPCReconciliationResult]:
    """
    Phase 4: LDPC Reconciliation (Bob = Reference).
    
    Yields
    ------
    EventExpression
        Classical communication events.
    
    Returns
    -------
    result : LDPCReconciliationResult
        Reconciliation metadata (Bob keeps original key).
    """
    logger.info("=== PHASE 4: LDPC Information Reconciliation (Bob) ===")
    
    qber_est = 0.05
    total_leakage = 0
    key_length = len(bob_key)
    target_payload = self.reconciliator.frame_size - 200
    blocks_to_process = (key_length + target_payload - 1) // target_payload
    
    for block_idx in range(blocks_to_process):
        start = block_idx * target_payload
        end = min(start + target_payload, key_length)
        key_block = bob_key[start:end]
        
        rate = self.reconciliator.select_rate(qber_est)
        n_shortened = self.reconciliator.compute_shortening(rate, qber_est, len(key_block))
        
        prng_seed = self._derive_prng_seed(block_idx)
        syndrome = self.reconciliator.compute_syndrome_block(
            key_block, rate, n_shortened, prng_seed
        )
        
        bob_hash = self.reconciliator.hash_verifier.compute_hash(key_block)
        
        # Send syndrome and hash to Alice
        self.context.csockets[self.PEER_NAME].send(syndrome.tobytes().hex())
        yield from self.context.csockets[self.PEER_NAME].recv()  # Sync
        
        self.context.csockets[self.PEER_NAME].send(bob_hash.hex())
        yield from self.context.csockets[self.PEER_NAME].recv()  # Sync
        
        total_leakage += self.reconciliator.estimate_leakage_block(len(syndrome), 50)
    
    return LDPCReconciliationResult(
        corrected_key=bob_key,  # Bob keeps reference key
        qber_estimate=qber_est,
        total_leakage=total_leakage,
        blocks_processed=blocks_to_process,
        blocks_verified=blocks_to_process,  # Bob assumes all verified
        blocks_discarded=0
    )
```

### 5.6. Testing Strategy

#### 5.6.1. Unit Tests

**File:** `ehok/tests/test_ldpc_reconciliation.py`

```python
class TestPEGMatrixGenerator:
    """Test PEG matrix generation."""
    
    def test_generate_matrix_rate_050(self):
        """Generate matrix with R=0.5 and verify dimensions."""
        generator = PEGMatrixGenerator(
            n=4096,
            rate=0.50,
            lambda_dist=DEGREE_DISTRIBUTIONS[0.50]['lambda'],
            rho_dist=DEGREE_DISTRIBUTIONS[0.50]['rho']
        )
        H = generator.generate()
        
        assert H.shape[1] == 4096
        expected_m = int(4096 * (1 - 0.50))
        assert abs(H.shape[0] - expected_m) < 10  # Allow small deviation
        ...
    
    def test_peg_girth_maximization(self):
        """Verify PEG maximizes local girth (no 4-cycles)."""
        ...
    
    def test_degree_distribution_realization(self):
        """Verify generated matrix realizes target degree distribution."""
        ...
    
    def test_matrix_verification(self):
        """Test matrix property verification."""
        ...

class TestLDPCMatrixManager:
    """Test matrix pool management and rate selection."""
    
    def test_load_matrices(self):
        """Verify all matrices loaded with correct dimensions."""
        manager = LDPCMatrixManager(matrix_directory=Path("configs/ldpc_matrices"))
        pool = manager.load_matrices()
        
        assert pool.frame_size == 4096
        assert len(pool.matrices) == 9  # 0.50 to 0.90
        assert len(pool.checksum) == 64  # SHA-256 hex
        ...
    
    def test_compute_pool_checksum(self):
        """Verify checksum computation is deterministic."""
        ...
    
    def test_verify_synchronization_success(self):
        """Verify synchronization succeeds with matching checksums."""
        ...
    
    def test_verify_synchronization_failure(self):
        """Verify synchronization fails with mismatched checksums."""
        manager = LDPCMatrixManager(matrix_directory=Path("configs/ldpc_matrices"))
        manager.load_matrices()
        
        fake_checksum = "0" * 64
        
        with pytest.raises(MatrixSynchronizationError):
            manager.verify_synchronization(fake_checksum)
        ...
    
    def test_select_rate_low_qber(self):
        """For QBER=0.01, should select high rate (e.g., R=0.90)."""
        ...
    
    def test_select_rate_high_qber(self):
        """For QBER=0.10, should select low rate (e.g., R=0.50)."""
        ...
    
    def test_binary_entropy(self):
        """Verify binary entropy calculation."""
        assert abs(LDPCMatrixManager.binary_entropy(0.5) - 1.0) < 1e-6
        assert abs(LDPCMatrixManager.binary_entropy(0.0) - 0.0) < 1e-6
        assert abs(LDPCMatrixManager.binary_entropy(1.0) - 0.0) < 1e-6
        ...

class TestLDPCBeliefPropagation:
    """Test BP decoder."""
    
    def test_decode_no_errors(self):
        """Decoder should converge in 1 iteration with no errors."""
        ...
    
    def test_decode_correctable_errors(self):
        """Decoder should correct errors below code capacity."""
        ...
    
    def test_decode_failure_high_errors(self):
        """Decoder should fail to converge with excessive errors."""
        ...
    
    def test_llr_initialization(self):
        """Verify LLR initialization for payload and padding."""
        ...

class TestPolynomialHashVerifier:
    """Test block verification."""
    
    def test_hash_identical_blocks(self):
        """Identical blocks should produce identical hashes."""
        ...
    
    def test_hash_different_blocks(self):
        """Different blocks should produce different hashes (with high probability)."""
        ...
    
    def test_collision_probability(self):
        """Empirically verify collision rate < 2×10^-12."""
        ...

class TestIntegratedQBEREstimator:
    """Test QBER estimation logic."""
    
    def test_estimate_all_verified(self):
        """With all blocks verified, QBER should equal measured average."""
        ...
    
    def test_estimate_all_unverified(self):
        """With all blocks unverified, QBER should equal 0.5."""
        ...
    
    def test_estimate_mixed(self):
        """Mixed verified/unverified blocks should produce weighted estimate."""
        ...

class TestLDPCReconciliator:
    """Integration tests for full reconciliator."""
    
    def test_reconcile_block_success(self):
        """Test successful block reconciliation with low QBER."""
        ...
    
    def test_reconcile_block_failure(self):
        """Test block reconciliation failure with high QBER."""
        ...
    
    def test_compute_syndrome_deterministic(self):
        """Syndrome computation should be deterministic."""
        ...
    
    def test_shortening_calculation(self):
        """Verify shortening formula correctness."""
        ...
```

#### 5.6.2. Integration Tests

**File:** `ehok/tests/test_ldpc_integration.py`

```python
class TestLDPCProtocolIntegration:
    """End-to-end protocol tests."""
    
    def test_matrix_synchronization_success(self):
        """Test successful matrix pool synchronization between Alice and Bob."""
        # Simulate both parties loading identical matrix files
        # Verify checksum exchange and validation succeeds
        ...
    
    def test_matrix_synchronization_failure(self):
        """Test protocol abort on matrix synchronization failure."""
        # Simulate Bob loading corrupted matrix file
        # Verify MatrixSynchronizationError is raised
        # Verify protocol does not proceed to reconciliation
        ...
    
    def test_alice_bob_reconciliation_low_qber(self):
        """Test full reconciliation with QBER=0.05."""
        # Generate correlated keys with 5% errors
        # Run full block-based reconciliation
        # Verify final keys match
        # Verify QBER estimate converges to ~0.05
        ...
    
    def test_alice_bob_reconciliation_high_qber(self):
        """Test reconciliation with QBER=0.10 (expect some block failures)."""
        # Generate correlated keys with 10% errors
        # Verify some blocks are discarded (unverified)
        # Verify conservative QBER estimation (includes 0.5 for unverified)
        ...
    
    def test_leakage_accounting(self):
        """Verify leakage calculation matches transmitted bits."""
        # Track all syndrome and hash transmissions
        # Verify total_leakage equals actual bits disclosed
        # Verify leakage is compatible with privacy amplification bounds
        ...
    
    def test_qber_estimation_convergence(self):
        """Verify QBER estimate converges to true value over multiple blocks."""
        # Process N=256 blocks
        # Verify rolling QBER estimate converges within tolerance
        ...
    
    def test_rate_adaptation(self):
        """Test dynamic rate selection adapts to changing QBER."""
        # Start with low QBER (should select high rate)
        # Simulate QBER increase mid-protocol
        # Verify rate selection adapts downward
        ...
    
    def test_peg_matrix_determinism(self):
        """Verify PEG matrix generation is deterministic with fixed seed."""
        # Generate matrix with seed=42
        # Generate matrix again with seed=42
        # Verify bit-identical CSR structures
        ...
```

### 5.7. Exception Handling

#### New Exception Class (ehok/core/exceptions.py)

```python
class MatrixSynchronizationError(SecurityException):
    """
    Raised when LDPC matrix pool synchronization fails between Alice and Bob.
    
    This exception indicates that the matrix pools are not bit-identical,
    which would cause catastrophic reconciliation failure. Possible causes:
    1. Different matrix generation seeds
    2. File corruption
    3. Configuration mismatch
    4. Potential tampering (security breach)
    
    Protocol MUST abort immediately when this exception is raised.
    """
    pass
```

### 5.8. Code Removal and Deprecation

#### Files to Remove:
*   All Old-LDPC-related implementations in `ehok/implementations/reconciliation/`
*   Any tests specifically for LDPC in `ehok/tests/`

#### Interface Changes:
*   **BREAKING:** `IReconciliator` signature changed from `compute_syndrome()` / `reconcile()` to block-based methods
*   Old method calls in `alice.py` and `bob.py` must be replaced

### 5.9. Configuration and Constants

#### Update `ehok/core/constants.py`:

```python
# LDPC Reconciliation Parameters
LDPC_FRAME_SIZE = 4096
LDPC_MAX_ITERATIONS = 60
LDPC_F_CRIT = 1.22
LDPC_HASH_BITS = 50
LDPC_QBER_WINDOW_SIZE = 256

# LDPC Code Rates
LDPC_AVAILABLE_RATES = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])

# Matrix file naming convention
LDPC_MATRIX_FILE_PATTERN = "ldpc_{frame_size}_rate{rate:.2f}.npz"

# PEG Algorithm Parameters
PEG_MAX_TREE_DEPTH = 10  # Maximum BFS depth for girth maximization
PEG_DEFAULT_SEED = 42    # Default seed for deterministic matrix generation

# Degree Distributions (from Section 2.1.1)
# Rate 0.5 optimized for BSC (Richardson/Urbanke)
LDPC_DEGREE_DIST_R050_LAMBDA = {
    'degrees': [2, 3, 6, 7, 13, 14, 18],
    'probabilities': [0.234029, 0.212425, 0.146898, 0.102840, 0.000780, 0.000320, 0.302708]
}
LDPC_DEGREE_DIST_R050_RHO = {
    'degrees': [8, 9],
    'probabilities': [0.7187, 0.2813]
}

# Additional degree distributions for rates 0.55-0.90 would be defined here
```

### 5.10. Documentation Updates

#### Files to Update:
*   `docs/e-hok-baseline.md`: Replace Cascade references with LDPC
*   `docs/e-hok-baseline-tests.md`: Update reconciliation test specifications
*   `README.md`: Update feature list

### 5.11. Migration Checklist

**Phase 1: Core Components**
- [ ] Implement `DegreeDistribution` dataclass in `ehok/core/data_structures.py`
- [ ] Implement `PEGMatrixGenerator` in `ehok/implementations/reconciliation/peg_generator.py`
- [ ] Define all degree distributions for rates 0.50-0.90 in constants
- [ ] Generate and validate LDPC matrix files (rates 0.50-0.90) using PEG algorithm
- [ ] Verify matrix properties (girth, degree distribution, no 4-cycles)

**Phase 2: Infrastructure**
- [ ] Add `MatrixSynchronizationError` exception to `ehok/core/exceptions.py`
- [ ] Update `LDPCMatrixPool` dataclass to include `checksum` field
- [ ] Implement `LDPCMatrixManager` with matrix loading and checksum computation
- [ ] Implement matrix pool synchronization verification logic
- [ ] Update `ehok/core/constants.py` with PEG parameters and degree distributions

**Phase 3: Reconciliation**
- [ ] Implement `LDPCBeliefPropagation` decoder with LLR initialization
- [ ] Implement `PolynomialHashVerifier` for block verification
- [ ] Implement `IntegratedQBEREstimator` with rolling window support
- [ ] Refactor `IReconciliator` interface for block-based reconciliation
- [ ] Implement new `LDPCReconciliator` class integrating all components
- [ ] Add PRNG padding generation for shortening

**Phase 4: Protocol Integration**
- [ ] Add `_initialize_ldpc_reconciliation()` method to base protocol class
- [ ] Refactor `alice.py` Phase 4 with matrix synchronization and block loop
- [ ] Refactor `bob.py` Phase 4 with matrix synchronization and syndrome transmission
- [ ] Update protocol initialization to call matrix synchronization before reconciliation
- [ ] Implement PRNG seed derivation for block padding synchronization

**Phase 5: Testing**
- [ ] Write unit tests for `PEGMatrixGenerator` (girth, degree distribution)
- [ ] Write unit tests for `LDPCMatrixManager` (loading, checksum, synchronization)
- [ ] Write unit tests for `LDPCBeliefPropagation` (decoding, LLR initialization)
- [ ] Write unit tests for `PolynomialHashVerifier` (hashing, collision probability)
- [ ] Write unit tests for `IntegratedQBEREstimator` (estimation logic)
- [ ] Write unit tests for `LDPCReconciliator` (block reconciliation, shortening)
- [ ] Write integration tests for matrix synchronization (success and failure)
- [ ] Write integration tests for full Alice-Bob reconciliation (various QBER)
- [ ] Write integration tests for rate adaptation and QBER estimation convergence
- [ ] Write integration tests for leakage accounting

**Phase 6: Cleanup and Documentation**
- [ ] Remove all old Cascade-related implementations
- [ ] Remove old simple LDPC reconciliator (non-block-based)
- [ ] Update `docs/e-hok-baseline.md` with LDPC details (replace Cascade)
- [ ] Update `docs/e-hok-baseline-tests.md` with new test specifications
- [ ] Update `README.md` with PEG algorithm and block-based reconciliation features
- [ ] Document matrix file generation procedure and tooling
- [ ] Add troubleshooting guide for matrix synchronization failures

**Phase 7: Validation**
- [ ] Run full test suite and verify all tests pass
- [ ] Perform end-to-end protocol validation with various QBER scenarios
- [ ] Verify matrix synchronization catches file corruption
- [ ] Benchmark reconciliation performance (throughput, convergence rate)
- [ ] Validate degree distribution realization in generated matrices
- [ ] Verify information leakage calculations match theoretical bounds
