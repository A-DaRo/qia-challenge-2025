# Successive Cancellation List (SCL) Decoder: Algorithmic Reference

<metadata>
version: 1.0.0
status: draft
created: 2026-02-03
purpose: Rigorous pseudocode specification for SCL decoder implementation
dependencies:
  - specs/polar-math-spec.md
  - adr/0001-polar-codec-adoption.md
</metadata>

---

## Executive Summary

<overview>

This document provides the **algorithmic blueprint** for implementing the Successive Cancellation List (SCL) decoder in Rust. It transcribes the core algorithms from Tal & Vardy's seminal work, **adapted to the LLR domain** for numerical stability per Balatsoukas-Stimming et al.

**Critical Adaptation Note:** The original Tal & Vardy algorithms operate on **probabilities** ($W$). This specification replaces all probability arithmetic with **Log-Likelihood Ratio (LLR)** arithmetic, following the LLR-based formulation which provides:
- Numerical stability (avoids underflow)
- Simplified path metric updates
- Hardware-friendly min-sum approximation

**Complexity:** $O(L \cdot N \log N)$ time, $O(L \cdot N)$ space via "lazy-copy" memory management.
</overview>

---

## 1. Data Structures

<data_structures>

### 1.1 Global Parameters

| Symbol | Type | Description |
|--------|------|-------------|
| $N$ | `usize` | Block length ($N = 2^m$) |
| $m$ | `usize` | Polarization depth ($m = \log_2 N$) |
| $L$ | `usize` | List size (max concurrent paths) |
| $\mathcal{A}$ | `BitSet` | Information bit indices |
| $\mathcal{F}$ | `BitSet` | Frozen bit indices |

### 1.2 Core Arrays

**Reference:** [List_Decoding_of_Polar_Codes.md] Algorithm 5 (`initializeDataStructures`).

| Array | Shape | Description |
|-------|-------|-------------|
| `inactivePathIndices` | Stack[$L$] | Stack of unused path indices |
| `activePath` | `bool[$L$]` | `activePath[ℓ]` = true iff path $\ell$ is active |
| `arrayPointer_L` | `ptr[$m+1$][$L$]` | **LLR arrays** (replaces `arrayPointer_P`) |
| `arrayPointer_C` | `ptr[$m+1$][$L$]` | Bit-pair arrays for partial sums |
| `pathIndexToArrayIndex` | `usize[$m+1$][$L$]` | Maps path → physical array index |
| `inactiveArrayIndices` | Stack[$m+1$][$L$] | Stack of free array indices per layer |
| `arrayReferenceCount` | `usize[$m+1$][$L$]` | Reference count per array |

### 1.3 LLR Arrays (Adaptation from Probability Arrays)

<llr_adaptation>

**CRITICAL ADAPTATION:** Tal & Vardy's `arrayPointer_P[λ][s]` stores **probability pairs** $(W(\cdot|0), W(\cdot|1))$.

We replace this with `arrayPointer_L[λ][s]` storing **LLR values**:

$$
L_\lambda[\beta] = \ln\left(\frac{W_\lambda(y|\beta, 0)}{W_\lambda(y|\beta, 1)}\right)
$$

| Layer | Array Size | Content |
|-------|-----------|---------|
| $\lambda = 0$ | $N$ | Channel LLRs: $L_0^{(i)} = \ln\frac{W(y_i|0)}{W(y_i|1)}$ |
| $\lambda > 0$ | $2^{m-\lambda}$ | Computed LLRs from recursive updates |

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II Eq. (7).
</llr_adaptation>

### 1.4 Path Metric Array

<path_metric>

**Additional structure** for LLR-based decoding:

| Array | Shape | Description |
|-------|-------|-------------|
| `pathMetric` | `f32[$L$]` | $PM_\ell^{(i)}$ — cumulative path penalty |

Initialized to $PM_\ell^{(0)} = 0$ for all paths.

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §III Eq. (10).
</path_metric>

</data_structures>

---

## 2. LLR Update Rules

<llr_update_rules>

### 2.1 The $f$-Function (Check-Node Update)

**Source:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II Eq. (8a).

**Exact form:**
$$
f(\alpha, \beta) = \ln\left(\frac{e^{\alpha + \beta} + 1}{e^\alpha + e^\beta}\right) = 2 \tanh^{-1}\left(\tanh\frac{\alpha}{2} \cdot \tanh\frac{\beta}{2}\right)
$$

**Min-Sum Approximation** (hardware-friendly):

$$
\tilde{f}(\alpha, \beta) = \text{sign}(\alpha) \cdot \text{sign}(\beta) \cdot \min(|\alpha|, |\beta|)
$$

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II Eq. (9).

```
FUNCTION f_minsum(α: f32, β: f32) -> f32:
    RETURN sign(α) * sign(β) * min(|α|, |β|)
```

### 2.2 The $g$-Function (Variable-Node Update)

**Source:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II Eq. (8b).

$$
g(\alpha, \beta, u) = (-1)^u \cdot \alpha + \beta = \begin{cases}
\alpha + \beta & \text{if } u = 0 \\
-\alpha + \beta & \text{if } u = 1
\end{cases}
$$

```
FUNCTION g(α: f32, β: f32, u: bit) -> f32:
    IF u = 0 THEN
        RETURN α + β
    ELSE
        RETURN -α + β
```

### 2.3 Path Metric Update Function $\phi$

**Source:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §III Eq. (11a)-(11b).

**Exact form:**
$$
\phi(\mu, \lambda, u) = \mu + \ln(1 + e^{-(1-2u)\lambda})
$$

**Approximation** (for efficient implementation):
$$
\tilde{\phi}(\mu, \lambda, u) = \begin{cases}
\mu & \text{if } u = \frac{1}{2}[1 - \text{sign}(\lambda)] \\
\mu + |\lambda| & \text{otherwise}
\end{cases}
$$

**Interpretation:** If the chosen bit $u$ agrees with the LLR sign, no penalty. Otherwise, penalty = $|\lambda|$.

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §III Eq. (12).

```
FUNCTION φ_approx(μ: f32, λ: f32, u: bit) -> f32:
    // u agrees with LLR direction if u = (1 - sign(λ)) / 2
    LET llr_suggests = IF λ >= 0 THEN 0 ELSE 1
    IF u = llr_suggests THEN
        RETURN μ                    // No penalty
    ELSE
        RETURN μ + |λ|              // Penalty = |LLR|
```

</llr_update_rules>

---

## 3. Low-Level Memory Management Algorithms

<low_level_algorithms>

### 3.1 Initialize Data Structures

**Source:** [List_Decoding_of_Polar_Codes.md] Algorithm 5.

```
ALGORITHM initializeDataStructures():
    // Allocate path tracking
    inactivePathIndices ← new Stack(capacity=L)
    activePath ← new bool[L], initialized to false
    
    // Allocate array pointers (LLR adaptation)
    arrayPointer_L ← new ptr[m+1][L]      // ⚠️ ADAPTED: was arrayPointer_P
    arrayPointer_C ← new ptr[m+1][L]
    pathIndexToArrayIndex ← new usize[m+1][L]
    inactiveArrayIndices ← new Stack[m+1](capacity=L each)
    arrayReferenceCount ← new usize[m+1][L], initialized to 0
    
    // ⚠️ ADAPTED: Path metrics for LLR-based decoding
    pathMetric ← new f32[L], initialized to 0.0
    
    // Allocate physical arrays
    FOR λ = 0 TO m DO
        FOR s = 0 TO L-1 DO
            // ⚠️ ADAPTED: LLR arrays instead of probability pairs
            arrayPointer_L[λ][s] ← new f32[2^(m-λ)]
            arrayPointer_C[λ][s] ← new BitPair[2^(m-λ)]
            arrayReferenceCount[λ][s] ← 0
            push(inactiveArrayIndices[λ], s)
    
    FOR ℓ = 0 TO L-1 DO
        activePath[ℓ] ← false
        push(inactivePathIndices, ℓ)
```

### 3.2 Assign Initial Path

**Source:** [List_Decoding_of_Polar_Codes.md] Algorithm 6.

```
ALGORITHM assignInitialPath() -> pathIndex:
    ℓ ← pop(inactivePathIndices)
    activePath[ℓ] ← true
    
    // Associate arrays with path index
    FOR λ = 0 TO m DO
        s ← pop(inactiveArrayIndices[λ])
        pathIndexToArrayIndex[λ][ℓ] ← s
        arrayReferenceCount[λ][s] ← 1
    
    // ⚠️ ADAPTED: Initialize path metric
    pathMetric[ℓ] ← 0.0
    
    RETURN ℓ
```

### 3.3 Clone Path (Lazy Copy)

**Source:** [List_Decoding_of_Polar_Codes.md] Algorithm 7.

```
ALGORITHM clonePath(ℓ: pathIndex) -> pathIndex:
    ℓ' ← pop(inactivePathIndices)
    activePath[ℓ'] ← true
    
    // Make ℓ' reference same arrays as ℓ (LAZY COPY)
    FOR λ = 0 TO m DO
        s ← pathIndexToArrayIndex[λ][ℓ]
        pathIndexToArrayIndex[λ][ℓ'] ← s
        arrayReferenceCount[λ][s] ← arrayReferenceCount[λ][s] + 1
    
    // ⚠️ ADAPTED: Clone path metric
    pathMetric[ℓ'] ← pathMetric[ℓ]
    
    RETURN ℓ'
```

### 3.4 Kill Path

**Source:** [List_Decoding_of_Polar_Codes.md] Algorithm 8.

```
ALGORITHM killPath(ℓ: pathIndex):
    // Mark path as inactive
    activePath[ℓ] ← false
    push(inactivePathIndices, ℓ)
    
    // Disassociate arrays (decrement reference counts)
    FOR λ = 0 TO m DO
        s ← pathIndexToArrayIndex[λ][ℓ]
        arrayReferenceCount[λ][s] ← arrayReferenceCount[λ][s] - 1
        IF arrayReferenceCount[λ][s] = 0 THEN
            push(inactiveArrayIndices[λ], s)
```

### 3.5 Get Array Pointer (Copy-on-Write)

**Source:** [List_Decoding_of_Polar_Codes.md] Algorithm 9.

```
ALGORITHM getArrayPointer_L(λ: layer, ℓ: pathIndex) -> ptr:
    // ⚠️ ADAPTED: Returns LLR array pointer (was getArrayPointer_P)
    s ← pathIndexToArrayIndex[λ][ℓ]
    
    IF arrayReferenceCount[λ][s] = 1 THEN
        // Array is private to this path
        s' ← s
    ELSE
        // Array is shared — make a private copy
        s' ← pop(inactiveArrayIndices[λ])
        COPY arrayPointer_L[λ][s] INTO arrayPointer_L[λ][s']
        arrayReferenceCount[λ][s] ← arrayReferenceCount[λ][s] - 1
        arrayReferenceCount[λ][s'] ← 1
        pathIndexToArrayIndex[λ][ℓ] ← s'
    
    RETURN arrayPointer_L[λ][s']

ALGORITHM getArrayPointer_C(λ: layer, ℓ: pathIndex) -> ptr:
    // Identical logic for bit-pair arrays
    s ← pathIndexToArrayIndex[λ][ℓ]
    
    IF arrayReferenceCount[λ][s] = 1 THEN
        s' ← s
    ELSE
        s' ← pop(inactiveArrayIndices[λ])
        COPY arrayPointer_C[λ][s] INTO arrayPointer_C[λ][s']
        arrayReferenceCount[λ][s] ← arrayReferenceCount[λ][s] - 1
        arrayReferenceCount[λ][s'] ← 1
        pathIndexToArrayIndex[λ][ℓ] ← s'
    
    RETURN arrayPointer_C[λ][s']
```

</low_level_algorithms>

---

## 4. Mid-Level Recursive Algorithms

<mid_level_algorithms>

### 4.1 Recursively Calculate LLRs

**Source:** [List_Decoding_of_Polar_Codes.md] Algorithm 10, **ADAPTED to LLR domain**.

<llr_adaptation_block>

**⚠️ CRITICAL ADAPTATION:**

Original Tal & Vardy (Algorithm 10) computes probability updates:
```
P_λ[β][u'] ← Σ_{u''} (1/2) · P_{λ-1}[2β][u' ⊕ u''] · P_{λ-1}[2β+1][u'']
```

We replace with LLR updates using $f$ and $g$ functions:
- **Even phase:** $L_\lambda[\beta] = f(L_{\lambda-1}[2\beta], L_{\lambda-1}[2\beta+1])$
- **Odd phase:** $L_\lambda[\beta] = g(L_{\lambda-1}[2\beta], L_{\lambda-1}[2\beta+1], C_\lambda[\beta][0])$

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] §II Eq. (8a)-(8b).
</llr_adaptation_block>

```
ALGORITHM recursivelyCalcL(λ: layer, φ: phase):
    // ⚠️ ADAPTED: Was recursivelyCalcP (probabilities)
    IF λ = 0 THEN RETURN      // Base case: channel LLRs
    
    ψ ← ⌊φ/2⌋
    
    // Recurse first, if needed (even phase only)
    IF φ mod 2 = 0 THEN
        recursivelyCalcL(λ-1, ψ)
    
    // Perform LLR calculation for all active paths
    FOR ℓ = 0 TO L-1 DO
        IF activePath[ℓ] = false THEN CONTINUE
        
        L_λ ← getArrayPointer_L(λ, ℓ)
        L_{λ-1} ← getArrayPointer_L(λ-1, ℓ)
        C_λ ← getArrayPointer_C(λ, ℓ)
        
        FOR β = 0 TO 2^{m-λ} - 1 DO
            IF φ mod 2 = 0 THEN
                // ⚠️ ADAPTED: f-function instead of probability sum
                // Original: P[β][u'] ← Σ (1/2) P[2β][u'⊕u''] · P[2β+1][u'']
                // LLR form: L[β] ← f(L[2β], L[2β+1])
                L_λ[β] ← f_minsum(L_{λ-1}[2β], L_{λ-1}[2β+1])
            ELSE
                // ⚠️ ADAPTED: g-function instead of probability product
                // Original: P[β][u''] ← (1/2) P[2β][u'⊕u''] · P[2β+1][u'']
                // LLR form: L[β] ← g(L[2β], L[2β+1], u')
                u' ← C_λ[β][0]
                L_λ[β] ← g(L_{λ-1}[2β], L_{λ-1}[2β+1], u')
    
    // Note: No normalization needed in LLR domain!
    // (Tal & Vardy lines 20-25 for probability normalization are removed)
```

### 4.2 Recursively Update Partial Sums

**Source:** [List_Decoding_of_Polar_Codes.md] Algorithm 11.

*No adaptation needed — operates on bits, not probabilities.*

```
ALGORITHM recursivelyUpdateC(λ: layer, φ: phase):
    REQUIRE φ is odd
    
    ψ ← ⌊φ/2⌋
    
    FOR ℓ = 0 TO L-1 DO
        IF activePath[ℓ] = false THEN CONTINUE
        
        C_λ ← getArrayPointer_C(λ, ℓ)
        C_{λ-1} ← getArrayPointer_C(λ-1, ℓ)
        
        FOR β = 0 TO 2^{m-λ} - 1 DO
            // XOR for left child, copy for right child
            C_{λ-1}[2β][ψ mod 2] ← C_λ[β][0] ⊕ C_λ[β][1]
            C_{λ-1}[2β+1][ψ mod 2] ← C_λ[β][1]
    
    // Recurse if ψ is odd
    IF ψ mod 2 = 1 THEN
        recursivelyUpdateC(λ-1, ψ)
```

</mid_level_algorithms>

---

## 5. High-Level Decoding Algorithms

<high_level_algorithms>

### 5.1 Continue Paths at Unfrozen Bit (Path Fork/Prune)

**Source:** [List_Decoding_of_Polar_Codes.md] Algorithm 13, **ADAPTED to LLR-based path metrics**.

<llr_adaptation_block>

**⚠️ CRITICAL ADAPTATION:**

Original Tal & Vardy ranks paths by **probability** $P_m[0][u]$.

We rank by **path metric** $PM_\ell^{(i)}$ (lower is better):
- Compute candidate metrics: $PM_{\ell,u} = \phi(PM_\ell^{(i-1)}, L_m^{(i)}[\ell], u)$
- Keep $L$ paths with **smallest** metrics

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md] Algorithm 3, lines 8-20.
</llr_adaptation_block>

```
ALGORITHM continuePaths_UnfrozenBit(φ: phase):
    // ⚠️ ADAPTED: Use path metrics instead of probabilities
    metricForks ← new f32[L][2]
    i ← 0
    
    // Populate candidate metrics for each active path
    FOR ℓ = 0 TO L-1 DO
        IF activePath[ℓ] = true THEN
            L_m ← getArrayPointer_L(m, ℓ)
            λ ← L_m[0]                           // Decision LLR
            
            // ⚠️ ADAPTED: Compute path metrics for both forks
            // Original: probForks[ℓ][u] ← P_m[0][u]
            // LLR form: metricForks[ℓ][u] ← φ(pathMetric[ℓ], λ, u)
            metricForks[ℓ][0] ← φ_approx(pathMetric[ℓ], λ, 0)
            metricForks[ℓ][1] ← φ_approx(pathMetric[ℓ], λ, 1)
            i ← i + 1
        ELSE
            metricForks[ℓ][0] ← +∞              // Invalid path
            metricForks[ℓ][1] ← +∞
    
    ρ ← min(2·i, L)
    contForks ← new bool[L][2], initialized to false
    
    // ⚠️ ADAPTED: Select ρ SMALLEST metrics (not largest probabilities)
    // Original: "contForks[ℓ][b] is true iff probForks[ℓ][b] is one of ρ largest"
    // LLR form: "contForks[ℓ][b] is true iff metricForks[ℓ][b] is one of ρ smallest"
    // This is achievable in O(L) time via selection algorithm
    SELECT ρ smallest entries from metricForks, mark in contForks
    
    // First: kill paths where both forks are discontinued
    FOR ℓ = 0 TO L-1 DO
        IF activePath[ℓ] = false THEN CONTINUE
        IF contForks[ℓ][0] = false AND contForks[ℓ][1] = false THEN
            killPath(ℓ)
    
    // Then: continue surviving paths
    FOR ℓ = 0 TO L-1 DO
        IF contForks[ℓ][0] = false AND contForks[ℓ][1] = false THEN
            CONTINUE
        
        C_m ← getArrayPointer_C(m, ℓ)
        
        IF contForks[ℓ][0] = true AND contForks[ℓ][1] = true THEN
            // Both forks survive — clone path
            C_m[0][φ mod 2] ← 0
            pathMetric[ℓ] ← metricForks[ℓ][0]     // ⚠️ ADAPTED
            
            ℓ' ← clonePath(ℓ)
            C_m ← getArrayPointer_C(m, ℓ')
            C_m[0][φ mod 2] ← 1
            pathMetric[ℓ'] ← metricForks[ℓ][1]    // ⚠️ ADAPTED
        ELSE
            // Exactly one fork survives
            IF contForks[ℓ][0] = true THEN
                C_m[0][φ mod 2] ← 0
                pathMetric[ℓ] ← metricForks[ℓ][0]
            ELSE
                C_m[0][φ mod 2] ← 1
                pathMetric[ℓ] ← metricForks[ℓ][1]
```

### 5.2 SCL Decoder Main Loop

**Source:** [List_Decoding_of_Polar_Codes.md] Algorithm 12, **ADAPTED**.

```
ALGORITHM SCL_Decode(y: f32[N], L: listSize) -> bits[N]:
    // === Initialization ===
    initializeDataStructures()
    ℓ ← assignInitialPath()
    
    // Load channel LLRs
    L_0 ← getArrayPointer_L(0, ℓ)
    FOR β = 0 TO N-1 DO
        // ⚠️ ADAPTED: Store LLRs, not probabilities
        // Original: P_0[β][0] ← W(y_β|0), P_0[β][1] ← W(y_β|1)
        // LLR form: L_0[β] ← ln(W(y_β|0) / W(y_β|1))
        L_0[β] ← channelLLR(y[β])
    
    // === Main Loop ===
    FOR φ = 0 TO N-1 DO
        recursivelyCalcL(m, φ)
        
        IF φ ∈ 𝓕 THEN                          // Frozen bit
            FOR ℓ = 0 TO L-1 DO
                IF activePath[ℓ] = false THEN CONTINUE
                
                C_m ← getArrayPointer_C(m, ℓ)
                L_m ← getArrayPointer_L(m, ℓ)
                
                // Set frozen bit value (typically 0)
                C_m[0][φ mod 2] ← frozenValue[φ]
                
                // ⚠️ ADAPTED: Update path metric at frozen bit
                // Reference: [LLR-Based] Algorithm 3, line 6
                pathMetric[ℓ] ← φ_approx(pathMetric[ℓ], L_m[0], frozenValue[φ])
        ELSE                                    // Information bit
            continuePaths_UnfrozenBit(φ)
        
        IF φ mod 2 = 1 THEN
            recursivelyUpdateC(m, φ)
    
    // === Select Best Path ===
    // ⚠️ ADAPTED: Select minimum path metric (not maximum probability)
    ℓ_best ← 0
    pm_best ← +∞
    FOR ℓ = 0 TO L-1 DO
        IF activePath[ℓ] = false THEN CONTINUE
        IF pathMetric[ℓ] < pm_best THEN
            ℓ_best ← ℓ
            pm_best ← pathMetric[ℓ]
    
    // Extract codeword
    C_0 ← getArrayPointer_C(0, ℓ_best)
    RETURN (C_0[β][0])_{β=0}^{N-1}
```

</high_level_algorithms>

---

## 6. CRC-Aided Path Selection

<crc_aided_selection>

### 6.1 CA-SCL Final Path Selection

**Source:** [CRC-Aided_Decoding_of_Polar_Codes.md] §III.A Step (A.4).

When CRC is enabled, the final path selection replaces the simple "minimum metric" selection:

```
ALGORITHM selectPathWithCRC(crcPoly: polynomial, crcLen: int) -> (bits[K], success: bool):
    // Collect all active paths sorted by path metric (ascending)
    candidates ← []
    FOR ℓ = 0 TO L-1 DO
        IF activePath[ℓ] = true THEN
            candidates.append((ℓ, pathMetric[ℓ]))
    
    SORT candidates BY metric ASCENDING  // Lower metric = more likely
    
    // Examine paths in order of likelihood
    FOR (ℓ, _) IN candidates DO
        C_0 ← getArrayPointer_C(0, ℓ)
        decoded ← extractInfoBits(C_0, 𝓐)
        
        // Check CRC
        IF verifyCRC(decoded, crcPoly, crcLen) = true THEN
            RETURN (decoded, true)          // First passing path wins
    
    // No path passed CRC — decoding failure
    // Option 1: Return path with lowest metric anyway
    // Option 2: Return failure flag
    RETURN (extractInfoBits(C_0[candidates[0].ℓ], 𝓐), false)
```

**Reference:** [CRC-Aided_Decoding_of_Polar_Codes.md] §III.A:
> "The paths in the list are examined one-by-one with decreasing metrics [increasing likelihood]. The decoder outputs the first path passing the CRC detection as the estimation sequence."

### 6.2 Incremental CRC Computation

For efficiency, CRC can be updated incrementally as information bits are decoded:

```
ALGORITHM updateCRC(crcState: bits[r], newBit: bit, crcPoly: bits[r+1]) -> bits[r]:
    // LFSR-style update: shift in new bit, XOR with polynomial if MSB = 1
    msb ← crcState[r-1]
    crcState ← (crcState << 1) | newBit
    IF msb = 1 THEN
        crcState ← crcState ⊕ crcPoly[0:r]
    RETURN crcState
```

This is performed for each information bit, allowing early termination if a path's CRC becomes inconsistent.

</crc_aided_selection>

---

## 7. Complexity Analysis

<complexity>

### 7.1 Time Complexity

**Reference:** [List_Decoding_of_Polar_Codes.md] Theorem 8.

| Function | Complexity |
|----------|------------|
| `initializeDataStructures` | $O(L \cdot m)$ |
| `assignInitialPath` | $O(m)$ |
| `clonePath` | $O(m)$ |
| `killPath` | $O(m)$ |
| `getArrayPointer_L/C(λ, ℓ)` | $O(2^{m-\lambda})$ |
| `recursivelyCalcL(m, ·)` total | $O(L \cdot m \cdot N)$ |
| `recursivelyUpdateC(m, ·)` total | $O(L \cdot m \cdot N)$ |
| `continuePaths_UnfrozenBit` | $O(L \cdot m)$ |
| **SCL Decoder total** | $O(L \cdot N \log N)$ |

### 7.2 Space Complexity

**Reference:** [List_Decoding_of_Polar_Codes.md] Theorem 7.

| Component | Space |
|-----------|-------|
| LLR arrays ($L$ banks × $(m+1)$ layers) | $O(L \cdot N)$ |
| Bit-pair arrays | $O(L \cdot N)$ |
| Path metrics | $O(L)$ |
| Bookkeeping arrays | $O(L \cdot m)$ |
| **Total** | $O(L \cdot N)$ |

</complexity>

---

## 8. Summary: Adaptation from Probability to LLR Domain

<adaptation_summary>

| Tal & Vardy (Probability) | This Spec (LLR) | Reference |
|---------------------------|-----------------|-----------|
| `arrayPointer_P[λ][s]` — probability pairs | `arrayPointer_L[λ][s]` — single LLR | [LLR-Based] §II |
| $P_\lambda[\beta][u] = W_\lambda(\cdot|u)$ | $L_\lambda[\beta] = \ln\frac{W(\cdot|0)}{W(\cdot|1)}$ | [LLR-Based] Eq. (7) |
| Even: $\sum_{u''} \frac{1}{2} P[u' \oplus u''] \cdot P[u'']$ | $f(L[2\beta], L[2\beta+1])$ | [LLR-Based] Eq. (8a) |
| Odd: $\frac{1}{2} P[u' \oplus u''] \cdot P[u'']$ | $g(L[2\beta], L[2\beta+1], u')$ | [LLR-Based] Eq. (8b) |
| Normalization (lines 20-25) | **Removed** — LLRs don't underflow | [LLR-Based] §III |
| Path ranking: max probability | Path ranking: min metric $PM_\ell$ | [LLR-Based] Thm. 1 |
| Metric: $\prod_j W_n^{(j)}(\cdot|\hat{u}_j)$ | Metric: $\sum_j \ln(1 + e^{-(1-2\hat{u}_j)L_n^{(j)}})$ | [LLR-Based] Eq. (10) |

</adaptation_summary>

---

## References

<references>

1. **[List_Decoding_of_Polar_Codes.md]** Tal, I. & Vardy, A. (2015). *List Decoding of Polar Codes.* IEEE Trans. Information Theory, 61(5). — Primary source: Algorithms 5-13, lazy-copy memory management, complexity proofs.

2. **[LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md]** Balatsoukas-Stimming, A., Parizi, M. B., & Burg, A. (2015). *LLR-Based Successive Cancellation List Decoding of Polar Codes.* IEEE Trans. Signal Processing, 63(19). — LLR adaptation: Theorem 1, Eq. (8)-(12), Algorithm 3.

3. **[CRC-Aided_Decoding_of_Polar_Codes.md]** Niu, K. & Chen, K. (2012). *CRC-Aided Decoding of Polar Codes.* IEEE Communications Letters, 16(10). — CRC-aided selection: §III.A Step (A.4).

</references>

---

## Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-03 | Context Engineer | Initial algorithmic specification with LLR adaptation |
