# Caligo Phase C: Security Layer Specification

**Document Type:** Formal Specification  
**Version:** 1.0  
**Date:** December 16, 2025  
**Status:** Draft  
**Parent Document:** [caligo_architecture.md](caligo_architecture.md)  
**Prerequisites:** [phase_a_spec.md](phase_a_spec.md), [phase_b_spec.md](phase_b_spec.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Scope & Deliverables](#2-scope--deliverables)
3. [Theoretical Foundations](#3-theoretical-foundations)
4. [Package: `security/`](#4-package-security)
5. [Mathematical Formulas & Bounds](#5-mathematical-formulas--bounds)
6. [Integration with Simulation Layer](#6-integration-with-simulation-layer)
7. [Testing Strategy](#7-testing-strategy)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [References](#9-references)

---

## 1. Executive Summary

**Phase C** implements the security analysis layer that validates protocol feasibility and computes entropy bounds for key extraction. This phase is **essential** because it determines whether a secure oblivious key can be generated given the physical parameters.

| Component | Purpose | Est. LOC |
|-----------|---------|----------|
| `security/bounds.py` | NSM entropy calculations (Max Bound, etc.) | ~200 |
| `security/feasibility.py` | Pre-flight security checks | ~180 |
| `security/finite_key.py` | Finite-size statistical corrections | ~150 |
| `security/__init__.py` | Package exports | ~20 |

**Critical Insight from Literature:**

> "Our result [...] shows that even for a very high storage rate, there is a positive rate of λ for many reasonable settings."
> — Schaffner et al. (2009)

> "In order to achieve secure OT, the overall trusted noise should be below ≈22%, for a noisy but unbounded quantum memory."
> — Lupo et al. (2023), Section VI

Phase C encodes these theoretical bounds as executable validation logic.

---

## 2. Scope & Deliverables

### 2.1 In Scope

| Deliverable | Description |
|-------------|-------------|
| `caligo/security/bounds.py` | NSM entropy bounds calculation |
| `caligo/security/feasibility.py` | Pre-flight feasibility checker |
| `caligo/security/finite_key.py` | Finite-size corrections |
| `caligo/security/__init__.py` | Package exports |
| Unit tests | Full coverage of all bounds |

### 2.2 Out of Scope

- Protocol phase implementations (Phase D)
- Privacy amplification hashing (Phase D)
- SquidASM program definitions (Phase E)
- Simulation infrastructure (Phase B)

### 2.3 Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Phase A types | — | `SecurityError`, `NSMViolationError`, `QBERThresholdExceeded` |
| Phase B simulation | — | `NSMParameters` dataclass |
| NumPy | ≥1.21 | Numerical computation |
| SciPy | ≥1.7 | Special functions (Lambert W, etc.) |

---

## 3. Theoretical Foundations

### 3.1 The Noisy Storage Model Security Paradigm

ROT security fundamentally differs from QKD. In QKD, Alice and Bob collaborate against an external eavesdropper Eve. In E-HOK, **Alice and Bob distrust each other**, and security derives from physical limitations on the adversary's quantum memory.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QKD vs. ROT Security Models                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────┐    ┌──────────────────────────────────┐   │
│  │         QKD Model            │    │       ROT (NSM) Model            │   │
│  ├──────────────────────────────┤    ├──────────────────────────────────┤   │
│  │                              │    │                                  │   │
│  │  Alice ←──trusted──→ Bob     │    │  Alice ←──distrustful──→ Bob     │   │
│  │          │                   │    │          │                       │   │
│  │          │                   │    │          │                       │   │
│  │          ▼                   │    │          ▼                       │   │
│  │        Eve                   │    │    (Each may cheat)              │   │
│  │    (adversary)               │    │                                  │   │
│  │                              │    │  Security assumption:            │   │
│  │  Security assumption:        │    │  Adversary's quantum memory      │   │
│  │  Channel estimation gives    │    │  is NOISY (storage decoherence)  │   │
│  │  bound on Eve's information  │    │                                  │   │
│  │                              │    │  Key formula:                    │   │
│  │  Key formula:                │    │  ℓ = n·h_min(r) - leak - Δ       │   │
│  │  ℓ = n·(1-h(Q)) - leak       │    │                                  │   │
│  └──────────────────────────────┘    └──────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 The Central Security Condition

**Source:** König et al. (2012), Theorem I.1; Lupo et al. (2023), Section VI

For security to hold in the NSM, the classical capacity of the adversary's storage channel, multiplied by the storage rate, must be bounded:

$$
C_{\mathcal{N}} \cdot \nu < \frac{1}{2}
$$

Where:
- $C_{\mathcal{N}}$ — Classical capacity of the noisy storage channel
- $\nu$ — Storage rate (fraction of qubits adversary can store)

**Implementation:** This constraint is checked in `feasibility.py` before protocol execution.

### 3.3 The "Strictly Less" Condition

**Source:** Schaffner et al. (2009), Corollary 7; Lupo et al. (2023), Section VI

The trusted noise (channel + honest device errors) must be **strictly less** than the untrusted noise (adversary's storage decoherence):

$$
h\left(\frac{1+r_{trusted}}{2}\right) < h\left(\frac{1+r_{untrusted}}{2}\right)
$$

For the depolarizing channel with storage parameter $r$:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   "Strictly Less" Condition Visualization                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Entropy leaked                                                             │
│  via syndrome        h(P_error) ◄── Error correction information            │
│        │                                                                    │
│        │          ┌─────────────────────────────────────────────────┐       │
│        │          │   If h(P_error) ≥ h_min(r)                      │       │
│        │          │   → Security IMPOSSIBLE                         │       │
│        │          │   → Adversary gains more than loses to storage  │       │
│        │          └─────────────────────────────────────────────────┘       │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   SECURE REGION: h(P_error) < h_min(r)                               │  │
│   │                                                                      │  │
│   │   For r ≥ r̃ ≈ 0.78:   h((1+r)/2) > h(P_error)                        │  │
│   │   For r < r̃:           1/2 > h(P_error) → P_error < 11%              │  │
│   │                                                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Critical Thresholds:                                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • QBER < 11%  (Conservative, Schaffner Corollary 7)                        │
│  • QBER < 22%  (Absolute limit, Lupo Eq. (43))                              │
│  • r̃ ≈ 0.7798 (Storage noise threshold, 2h⁻¹(1/2) - 1)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 The "Rational Adversary" Insight

**Source:** Lupo et al. (2023), Section V.A

A key insight is that a **rational adversary** will not store qubits if storage noise exceeds a certain threshold, because immediate measurement yields more information:

$$
h_{min}^{rational} = \min\left\{\frac{1}{2}, \max\left\{\Gamma[1-\log_2(1+3r^2)], 1-r\right\}\right\}
$$

**Interpretation:**
- If storage noise is high ($r$ small), measuring immediately is optimal
- Immediate measurement in random BB84 basis yields $h_{min} = 1/2$
- Protocol only needs to consider rational adversarial strategies

---

## 4. Package: `security/`

### 4.1 Module: `bounds.py` (~200 LOC)

**Purpose:** Implement the NSM entropy bounds for privacy amplification key length calculation.

#### 4.1.1 The Γ (Gamma) Function

**Source:** Lupo et al. (2023), Eq. (24)-(25)

```python
def gamma_function(x: float) -> float:
    """
    Compute the Γ function for collision entropy regularization.
    
    The Γ function maps collision entropy rate to min-entropy rate,
    accounting for the relationship between Rényi entropies of
    different orders.
    
    Parameters
    ----------
    x : float
        Collision entropy rate h₂ ∈ [0, 1].
    
    Returns
    -------
    float
        Regularized min-entropy rate Γ(x).
    
    Mathematical Definition
    -----------------------
    Γ(x) = x                    if x ≥ 1/2
    Γ(x) = g⁻¹(x)              if x < 1/2
    
    where g(y) = -y·log₂(y) - (1-y)·log₂(1-y) + y - 1
    
    References
    ----------
    - Lupo et al. (2023), Eq. (24)-(25)
    - Dupuis et al. (2014), Theorem 1
    
    Examples
    --------
    >>> gamma_function(0.7)
    0.7
    >>> gamma_function(0.3)  # Returns g⁻¹(0.3) via numerical inversion
    0.198...
    """
```

**Implementation Notes:**
- For $x \geq 1/2$: Return $x$ directly
- For $x < 1/2$: Numerically invert $g(y)$ using `scipy.optimize.brentq`
- The function $g(y) = h(y) + y - 1$ where $h$ is binary entropy

#### 4.1.2 Collision Entropy Rate

**Source:** Lupo et al. (2023), Eq. (27)

```python
def collision_entropy_rate(r: float) -> float:
    """
    Compute collision entropy rate for depolarizing storage channel.
    
    For a depolarizing channel that preserves state with probability r
    and completely depolarizes with probability (1-r), the collision
    entropy rate is:
    
        h₂ = 1 - log₂(1 + 3r²)
    
    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
        r = 0: Complete depolarization (best for security)
        r = 1: Perfect storage (worst for security)
    
    Returns
    -------
    float
        Collision entropy rate h₂ ∈ [0, 1].
    
    References
    ----------
    - Lupo et al. (2023), Eq. (27)
    
    Examples
    --------
    >>> collision_entropy_rate(0.0)
    1.0
    >>> collision_entropy_rate(0.5)
    0.678...
    >>> collision_entropy_rate(1.0)
    -1.0  # Log₂(4) = 2, so 1 - 2 = -1 (edge case)
    """
```

#### 4.1.3 The Dupuis-König Bound (Collision Entropy Bound)

**Source:** Dupuis et al. (2014); Lupo et al. (2023), Eq. (29)

```python
def dupuis_konig_bound(r: float) -> float:
    """
    Compute min-entropy bound from collision entropy (Dupuis-König).
    
    This bound is derived from the collision entropy of the stored
    quantum state after depolarizing noise:
    
        h_A = Γ[1 - log₂(1 + 3r²)]
    
    This bound is tighter for HIGH noise (small r) storage.
    
    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
    
    Returns
    -------
    float
        Min-entropy rate h_A ∈ [0, 1].
    
    References
    ----------
    - Dupuis et al. (2014), Theorem 1
    - Lupo et al. (2023), Eq. (28)-(29)
    
    Notes
    -----
    For r approaching 1 (perfect storage), this bound approaches 0,
    but remains non-trivial even for r arbitrarily close to 1.
    """
```

#### 4.1.4 The Lupo Bound (Virtual Erasure Bound)

**Source:** Lupo et al. (2023), Eq. (35)

```python
def lupo_virtual_erasure_bound(r: float) -> float:
    """
    Compute min-entropy bound from virtual erasure argument (Lupo).
    
    This bound treats depolarized qubits as virtually erased, giving
    an adversary flag information about which qubits were corrupted:
    
        h_B = 1 - r
    
    This bound is tighter for LOW noise (large r) storage.
    
    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
    
    Returns
    -------
    float
        Min-entropy rate h_B ∈ [0, 1].
    
    References
    ----------
    - Lupo et al. (2023), Eq. (34)-(35)
    
    Notes
    -----
    The physical intuition: if about (1-r) fraction of qubits are
    depolarized, an adversary knows nothing about those bits,
    contributing (1-r) bits of entropy per qubit on average.
    """
    return 1.0 - r
```

#### 4.1.5 The Max Bound (Combined Optimal Bound)

**Source:** Lupo et al. (2023), Eq. (36)

```python
def max_bound_entropy(r: float) -> float:
    """
    Compute the optimal min-entropy bound by selecting the maximum.
    
    The "Max Bound" extracts strictly more key than either individual
    bound by selecting the optimal one for each noise regime:
    
        h_min = max{ Γ[1 - log₂(1 + 3r²)], 1 - r }
    
    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
    
    Returns
    -------
    float
        Optimal min-entropy rate h_min ∈ [0, 1].
    
    References
    ----------
    - Lupo et al. (2023), Eq. (36)
    
    Notes
    -----
    Crossover point: The two bounds are equal at r ≈ 0.82.
    - For r < 0.82: Dupuis-König (collision) bound is better
    - For r > 0.82: Lupo (virtual erasure) bound is better
    
    Examples
    --------
    >>> max_bound_entropy(0.1)  # High noise → collision bound
    0.957...
    >>> max_bound_entropy(0.9)  # Low noise → virtual erasure bound
    0.100
    """
```

#### 4.1.6 Rational Adversary Bound

**Source:** Lupo et al. (2023), Eq. (37)-(38)

```python
def rational_adversary_bound(r: float) -> float:
    """
    Compute min-entropy assuming a rational adversary.
    
    A rational adversary will not store qubits if immediate measurement
    (without waiting for basis information) yields more information.
    Measuring in random BB84 basis yields 1/2 bit of min-entropy.
    
        h_min^rational = min{ 1/2, max{ Γ[1-log(1+3r²)], 1-r } }
    
    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
    where:
        $r = 1$: Perfect storage (no noise)
        $r = 0$: Complete depolarization (maximum noise)
    
    Returns
    -------
    float
        Min-entropy rate for rational adversary ∈ [0, 0.5].
    
    References
    ----------
    - Lupo et al. (2023), Eq. (37)-(38)
    
    Notes
    -----
    For r < 0.5: Storing is irrational; honest behavior (measure
    immediately) is the best strategy.
    """
    return min(0.5, 1.0 - r)
```

#### 4.1.7 Bounded Storage Extension

**Source:** Lupo et al. (2023), Eq. (49)-(51)

```python
def bounded_storage_entropy(r: float, nu: float) -> float:
    """
    Compute min-entropy for noisy AND bounded quantum storage.
    
    When the adversary can store at most a fraction ν of received
    qubits, the entropy bound is improved:
    
        h_min = (1-ν)/2 + ν·max{ Γ[1-log(1+3r²)], 1-r }
    
    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
    nu : float
        Storage rate ∈ [0, 1]. Fraction of qubits adversary can store.
        ν = 0: No storage capability
        ν = 1: Can store all qubits (pure noisy storage model)
    
    Returns
    -------
    float
        Min-entropy rate for bounded noisy storage.
    
    References
    ----------
    - Lupo et al. (2023), Eq. (49)
    - König et al. (2012), Corollary I.2
    
    Examples
    --------
    >>> bounded_storage_entropy(0.75, 1.0)  # Full storage
    0.25
    >>> bounded_storage_entropy(0.75, 0.5)  # Half storage
    0.375
    """
```

#### 4.1.8 Strong Converse Error Exponent

**Source:** Lupo et al. (2023), Eq. (16); König et al. (2012)

```python
def strong_converse_exponent(r: float, rate: float) -> float:
    """
    Compute strong converse error exponent for depolarizing channel.
    
    The error exponent γ_r(R) quantifies how fast the success
    probability of decoding decays above channel capacity:
    
        P_succ^{N^⊗n}(nR) ≲ 2^{-n·γ_r(R)}
    
    For the depolarizing channel:
    
        γ_r(R) = 1 + max_{α>1} [(α-1)(R-1) - log[(1+r)^α + (1-r)^α]] / α
    
    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
    rate : float
        Attempted transmission rate R (bits per qubit).
    
    Returns
    -------
    float
        Error exponent γ_r(R) ≥ 0.
        Returns 0 if R ≤ C_N (capacity).
    
    References
    ----------
    - Lupo et al. (2023), Eq. (16)
    - König et al. (2012), strong converse theorem
    """
```

---

### 4.2 Module: `feasibility.py` (~180 LOC)

**Purpose:** Pre-flight validation that protocol can achieve positive key rate.

#### 4.2.1 QBER Threshold Check

**Source:** Schaffner et al. (2009), Corollary 7; Lupo et al. (2023), Section VI

```python
# Security threshold constants (to be stored in top level yaml config)
QBER_CONSERVATIVE_THRESHOLD = 0.11  # Schaffner Corollary 7
QBER_ABSOLUTE_THRESHOLD = 0.22      # Lupo Eq. (43)
R_TILDE = 0.7798                     # 2·h⁻¹(1/2) - 1

class FeasibilityChecker:
    """
    Pre-flight security feasibility validation.
    
    This class validates whether the E-HOK protocol can produce a
    secure oblivious key given the physical parameters. All checks
    MUST pass before protocol execution.
    
    Attributes
    ----------
    storage_noise_r : float
        Adversary's storage noise parameter.
    storage_rate_nu : float
        Adversary's storage rate (fraction of qubits storable).
    expected_qber : float
        Expected quantum bit error rate from honest devices/channel.
    security_parameter : float
        Target security parameter ε_sec.
    
    References
    ----------
    - Schaffner et al. (2009), Corollary 7
    - Lupo et al. (2023), Section VI
    - König et al. (2012), Theorem I.1
    """
```

#### 4.2.2 The "Strictly Less" Check

```python
def check_strictly_less_condition(
    self,
    trusted_qber: float,
    storage_noise_r: float
) -> FeasibilityResult:
    """
    Verify the fundamental NSM security condition.
    
    Security requires that information leaked via error correction
    is strictly less than the min-entropy from storage decoherence:
    
        h(P_error) < h_min(r)
    
    For depolarizing storage (Corollary 7 of Schaffner):
        - If r ≥ r̃ ≈ 0.78: h((1+r)/2) > h(P_error)
        - If r < r̃:         1/2 > h(P_error) → P_error < 11%
    
    Parameters
    ----------
    trusted_qber : float
        Total trusted noise (channel + honest device errors).
    storage_noise_r : float
        Adversary's storage noise parameter.
    
    Returns
    -------
    FeasibilityResult
        Contains is_feasible, margin, and diagnostic message.
    
    Raises
    ------
    QBERThresholdExceeded
        If QBER exceeds absolute threshold (22%).
    NSMViolationError
        If "strictly less" condition cannot be satisfied.
    
    References
    ----------
    - Schaffner et al. (2009), Corollary 7
    - Lupo et al. (2023), Section VI
    """
```

#### 4.2.3 Storage Capacity Constraint

**Source:** König et al. (2012), Corollary I.2

```python
def check_storage_capacity_constraint(
    self,
    storage_noise_r: float,
    storage_rate_nu: float
) -> FeasibilityResult:
    """
    Verify the noisy storage capacity constraint.
    
    For security to hold, the classical capacity of the adversary's
    storage channel multiplied by storage rate must satisfy:
    
        C_N · ν < 1/2
    
    For depolarizing channel with parameter r:
        C_N = 1 - h((1+r)/2)
    
    Parameters
    ----------
    storage_noise_r : float
        Storage channel noise parameter.
    storage_rate_nu : float
        Fraction of qubits adversary can store.
    
    Returns
    -------
    FeasibilityResult
        Contains is_feasible, capacity_product, and margin.
    
    Raises
    ------
    NSMViolationError
        If C_N · ν ≥ 1/2.
    
    References
    ----------
    - König et al. (2012), Corollary I.2
    """
```

#### 4.2.4 Batch Size Feasibility

**Source:** Erven et al. (2014), Eq. (8); Tomamichel et al. (2012)

```python
def check_batch_size_feasibility(
    self,
    n_raw_bits: int,
    expected_qber: float,
    storage_noise_r: float,
    syndrome_rate: float,
    epsilon_sec: float = 1e-10
) -> FeasibilityResult:
    """
    Check if batch size yields positive extractable key length.
    
    Computes the expected final key length using the E-HOK formula:
    
        ℓ = n · h_min(r) - |Σ| - 2·log₂(1/ε_sec) - Δ_finite
    
    Where:
        - h_min(r) = max{Γ[1-log(1+3r²)], 1-r}  [Max Bound]
        - |Σ| = n · h(Q) · f                     [Syndrome leakage]
        - Δ_finite = finite-key correction       [Statistical penalty]
    
    Parameters
    ----------
    n_raw_bits : int
        Number of raw bits after sifting.
    expected_qber : float
        Expected QBER for error correction rate.
    storage_noise_r : float
        Adversary storage noise parameter.
    syndrome_rate : float
        Error correction efficiency factor f (typically 1.1-1.5).
    epsilon_sec : float, optional
        Security parameter. Default: 1e-10.
    
    Returns
    -------
    FeasibilityResult
        Contains is_feasible, expected_key_length, and min_n recommendation.
    
    Raises
    ------
    SecurityError
        If expected key length ≤ 0 ("Death Valley" scenario).
    
    References
    ----------
    - Erven et al. (2014), Eq. (8)
    - Lupo et al. (2023), Eq. (43)
    """
```

#### 4.2.5 Full Pre-Flight Check

```python
def run_preflight_checks(
    self,
    config: "ProtocolConfiguration"
) -> PreflightReport:
    """
    Execute all feasibility checks before protocol execution.
    
    This is the main entry point that validates whether the protocol
    can succeed given the provided configuration. All checks must pass
    for the protocol to proceed.
    
    Checks Performed (in order):
    1. QBER threshold (conservative 11%, hard limit 22%)
    2. Storage capacity constraint (C_N · ν < 1/2)
    3. "Strictly less" condition (h(P_error) < h_min(r))
    4. Batch size feasibility (expected ℓ > 0)
    
    Parameters
    ----------
    config : ProtocolConfiguration
        Complete protocol configuration including physical parameters.
    
    Returns
    -------
    PreflightReport
        Comprehensive report with all check results and recommendations.
    
    Raises
    ------
    SecurityError
        If any check fails with unrecoverable parameters.
    
    Notes
    -----
    The protocol MUST abort if this method raises SecurityError.
    The PreflightReport contains detailed diagnostics for debugging.
    """
```

---

### 4.3 Module: `finite_key.py` (~150 LOC)

**Purpose:** Finite-size statistical corrections for real-world key extraction.

#### 4.3.1 Statistical Fluctuation μ

**Source:** Erven et al. (2014), Eq. discussion; Tomamichel et al. (2012)

```python
def compute_statistical_fluctuation(
    n: int,
    k: int,
    epsilon_pe: float
) -> float:
    """
    Compute the statistical fluctuation penalty μ.
    
    For finite key lengths, the QBER estimate from k test bits has
    statistical uncertainty that must be accounted for:
    
        μ = √[(n+k)/(n·k) · (k+1)/k] · √[ln(4/ε_PE)]
    
    This penalty is ADDED to the measured QBER for security calculations.
    
    Parameters
    ----------
    n : int
        Number of bits used for key extraction (after test sampling).
    k : int
        Number of bits used for QBER estimation (test sample).
    epsilon_pe : float
        Parameter estimation failure probability.
    
    Returns
    -------
    float
        Statistical fluctuation μ to add to measured QBER.
    
    References
    ----------
    - Tomamichel et al. (2012), Theorem 1
    - Erven et al. (2014), Security section
    
    Examples
    --------
    >>> compute_statistical_fluctuation(100000, 5000, 1e-10)
    0.015...  # About 1.5% penalty
    >>> compute_statistical_fluctuation(1000, 100, 1e-10)
    0.12...   # Much larger for small samples
    """
```

#### 4.3.2 Hoeffding Bound for Detection Statistics

**Source:** Erven et al. (2014), Eq. (1)

```python
def hoeffding_detection_interval(
    n: int,
    p_expected: float,
    epsilon: float
) -> tuple[float, float]:
    """
    Compute secure interval for detection statistics using Hoeffding.
    
    For n independent Bernoulli trials with probability p, the number
    of successes S satisfies (via Hoeffding's inequality):
    
        Pr[|S - p·n| > ζ·n] ≤ 2·exp(-2·ζ²·n)
    
    Setting failure probability to ε gives:
        ζ = √[ln(1/ε) / (2n)]
    
    Parameters
    ----------
    n : int
        Total number of trials (e.g., photon emission events).
    p_expected : float
        Expected probability (e.g., detection efficiency).
    epsilon : float
        Allowed failure probability for the bound.
    
    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) for expected counts.
    
    References
    ----------
    - Hoeffding (1963)
    - Erven et al. (2014), Correctness section
    
    Notes
    -----
    Alice uses this to verify Bob's reported detection counts.
    If counts fall outside interval → protocol aborts.
    """
```

#### 4.3.3 Finite-Key Length Calculation

**Source:** Erven et al. (2014), Eq. (8); Lupo et al. (2023), Section VI

```python
def compute_finite_key_length(
    n: int,
    qber_measured: float,
    storage_noise_r: float,
    storage_rate_nu: float,
    ec_efficiency: float,
    epsilon_sec: float,
    epsilon_cor: float
) -> int:
    """
    Compute the extractable secure key length for finite resources.
    
    Full formula incorporating all finite-size effects:
    
        ℓ = ⌊n/2 · ν · γ^{N_r}(R) - n · f · h(p_err) - log₂(1/(2ε))⌋
    
    Simplified asymptotic form (n large):
    
        ℓ ≈ n · h_min(r) - n · f · h(Q) - 2·log₂(1/ε_sec)
    
    Parameters
    ----------
    n : int
        Number of raw bits after sifting.
    qber_measured : float
        Measured QBER from test sample (NOT including μ penalty yet).
    storage_noise_r : float
        Adversary storage noise parameter.
    storage_rate_nu : float
        Adversary storage rate.
    ec_efficiency : float
        Error correction efficiency f (typically 1.1-1.5).
    epsilon_sec : float
        Security parameter.
    epsilon_cor : float
        Correctness parameter.
    
    Returns
    -------
    int
        Maximum extractable key length ℓ (non-negative).
        Returns 0 if key is not extractable.
    
    References
    ----------
    - Erven et al. (2014), Eq. (8)
    - Lupo et al. (2023), Eq. (43)
    
    Notes
    -----
    The μ penalty is computed internally and applied to QBER.
    """
```

---

## 5. Mathematical Formulas & Bounds

### 5.1 Summary Table of All Bounds

| Bound Name | Formula | Best For | Reference |
|------------|---------|----------|-----------|
| **Dupuis-König** | $h_A = \Gamma[1 - \log_2(1 + 3r^2)]$ | High noise (small r) | Lupo Eq. (29) |
| **Lupo Virtual Erasure** | $h_B = 1 - r$ | Low noise (large r) | Lupo Eq. (35) |
| **Max Bound** | $h_{min} = \max\{h_A, h_B\}$ | All regimes | Lupo Eq. (36) |
| **Rational Adversary** | $h_{min}^{rat} = \min\{1/2, 1-r\}$ | All regimes | Lupo Eq. (38) |
| **Bounded Storage** | $h = \frac{1-\nu}{2} + \nu \cdot h_{min}$ | ν < 1 | Lupo Eq. (49) |

### 5.2 Numeric Reference Values

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               Min-Entropy Rates for Common Storage Noise Values             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    r     │ Dupuis-König │   Lupo    │ Max Bound │ Rational │                │
│  (noise) │   (h_A)      │   (h_B)   │   (h)     │  (h_rat) │   Notes        │
│ ─────────┼──────────────┼───────────┼───────────┼──────────┼─────────────   │
│   0.0    │    1.000     │   1.000   │   1.000   │   0.500  │ Perfect        │
│   0.1    │    0.957     │   0.900   │   0.957   │   0.500  │ ← DK better    │
│   0.2    │    0.895     │   0.800   │   0.895   │   0.500  │                │
│   0.3    │    0.805     │   0.700   │   0.805   │   0.500  │                │
│   0.4    │    0.688     │   0.600   │   0.688   │   0.500  │                │
│   0.5    │    0.542     │   0.500   │   0.542   │   0.500  │ ← Equal        │
│   0.6    │    0.374     │   0.400   │   0.400   │   0.400  │                │
│   0.7    │    0.204     │   0.300   │   0.300   │   0.300  │ ← Lupo better  │
│   0.75   │    0.133     │   0.250   │   0.250   │   0.250  │ (Erven exp.)   │
│   0.8    │    0.076     │   0.200   │   0.200   │   0.200  │                │
│   0.82   │    0.057     │   0.180   │   0.180   │   0.180  │ ← Crossover    │
│   0.9    │    0.016     │   0.100   │   0.100   │   0.100  │                │
│   1.0    │    -1.00     │   0.000   │   0.000   │   0.000  │ No security    │
│                                                                             │
│ Key: DK = Dupuis-König bound; Lupo = Virtual erasure bound                  │
│ Crossover at r ≈ 0.82 where both bounds equal                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Critical Threshold Constants

```python
# security/bounds.py (to be stored in top level yaml configuration)

# QBER Thresholds (Source: Schaffner Corollary 7, Lupo Section VI)
QBER_CONSERVATIVE = 0.11      # 11% - safe for most storage parameters
QBER_ABSOLUTE_MAX = 0.22      # 22% - theoretical maximum, h((1+0.78)/2) = 0.5

# Storage Noise Thresholds (Source: Schaffner Corollary 7)
R_TILDE = 0.7798              # 2·h⁻¹(1/2) - 1, crossover for optimal attack
R_CROSSOVER = 0.82            # Where Dupuis-König and Lupo bounds equal

# Security Parameters (Source: Erven et al. Table I)
DEFAULT_EPSILON_SEC = 1e-10   # Standard security parameter
DEFAULT_EPSILON_COR = 1e-6    # Correctness parameter

# Erven Experimental Values (for validation)
ERVEN_STORAGE_NOISE_R = 0.75  # Assumed storage parameter
ERVEN_STORAGE_RATE_NU = 0.002 # Assumed fraction storable
ERVEN_QBER_MAX = 0.0238       # Maximum tolerable at η=0.25
```

---

## 6. Integration with Simulation Layer

### 6.1 Data Flow from Phase B

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Phase B → Phase C Integration                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FROM Phase B (simulation/physical_model.py):                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ NSMParameters                                                          │ │
│  │   storage_noise_r : float      # Maps to depolarizing r               │ │
│  │   storage_rate_nu : float      # Fraction of qubits storable          │ │
│  │   delta_t_ns : float           # Wait time in nanoseconds             │ │
│  │   channel_fidelity : float     # Fidelity of EPR distribution         │ │
│  │   detection_eff_eta : float    # Total detection efficiency           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                       │                                                     │
│                       ▼                                                     │
│  TO Phase C (security/feasibility.py):                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ FeasibilityChecker.run_preflight_checks()                              │ │
│  │   ├── compute_expected_qber(channel_fidelity, detection_eff_eta)      │ │
│  │   ├── check_qber_threshold(expected_qber)                              │ │
│  │   ├── check_storage_capacity(storage_noise_r, storage_rate_nu)         │ │
│  │   ├── check_strictly_less(expected_qber, storage_noise_r)              │ │
│  │   └── check_batch_size(n, expected_qber, storage_noise_r, ...)        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                       │                                                     │
│                       ▼                                                     │
│  OUTPUT:                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ PreflightReport                                                        │ │
│  │   is_feasible : bool           # Can protocol proceed?                │ │
│  │   expected_key_rate : float    # Bits per raw qubit                   │ │
│  │   min_batch_size : int         # Minimum n for positive ℓ             │ │
│  │   security_margin : float      # How far above thresholds             │ │
│  │   warnings : list[str]         # Near-threshold conditions            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Expected QBER from Physical Parameters

**Source:** Erven et al. (2014), Table I and discussion

```python
def compute_expected_qber(
    channel_fidelity: float,
    detection_efficiency: float,
    dark_count_prob: float,
    intrinsic_error: float
) -> float:
    """
    Compute expected QBER from physical device parameters.
    
    The QBER arises from multiple sources:
    1. EPR pair infidelity: (1 - F)/2 contribution
    2. Detector dark counts: P_dark / P_click contribution
    3. Intrinsic detection error: e_det direct contribution
    
    Parameters
    ----------
    channel_fidelity : float
        EPR pair fidelity F ∈ [0.5, 1].
    detection_efficiency : float
        Total detection efficiency η ∈ (0, 1].
    dark_count_prob : float
        Dark count probability per detection window.
    intrinsic_error : float
        Intrinsic detector error rate e_det.
    
    Returns
    -------
    float
        Expected QBER ∈ [0, 0.5].
    
    References
    ----------
    - Erven et al. (2014), Table I
    """
```

---

## 7. Testing Strategy

### 7.1 Unit Tests for Bounds

```python
# tests/security/test_bounds.py

class TestEntropyBounds:
    """Unit tests for NSM entropy bound calculations."""
    
    def test_gamma_function_above_half(self):
        """Γ(x) = x for x ≥ 0.5."""
        assert gamma_function(0.7) == pytest.approx(0.7)
        assert gamma_function(0.5) == pytest.approx(0.5)
    
    def test_gamma_function_below_half(self):
        """Γ(x) = g⁻¹(x) for x < 0.5, verified numerically."""
        # g(0.2) ≈ 0.278, so Γ(0.278) ≈ 0.2
        x = 0.3
        result = gamma_function(x)
        assert 0 < result < 0.5
    
    def test_collision_entropy_perfect_storage(self):
        """h₂ = 1 - log₂(4) = -1 for perfect storage (edge case)."""
        assert collision_entropy_rate(1.0) == pytest.approx(-1.0)
    
    def test_collision_entropy_complete_depolarization(self):
        """h₂ = 1 for complete depolarization."""
        assert collision_entropy_rate(0.0) == pytest.approx(1.0)
    
    def test_max_bound_crossover(self):
        """At r ≈ 0.82, both bounds should be approximately equal."""
        r = 0.82
        dk = dupuis_konig_bound(r)
        lupo = lupo_virtual_erasure_bound(r)
        assert dk == pytest.approx(lupo, abs=0.01)
    
    def test_max_bound_selects_optimal(self):
        """Max bound should select the larger of the two bounds."""
        for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
            dk = dupuis_konig_bound(r)
            lupo = lupo_virtual_erasure_bound(r)
            max_b = max_bound_entropy(r)
            assert max_b == pytest.approx(max(dk, lupo))
    
    def test_rational_adversary_capped_at_half(self):
        """Rational adversary bound never exceeds 1/2."""
        for r in [0.0, 0.3, 0.5, 0.7, 1.0]:
            assert rational_adversary_bound(r) <= 0.5


class TestLiteratureValues:
    """Verify numerical values against published literature."""
    
    @pytest.mark.parametrize("r,expected", [
        (0.1, 0.957),   # Lupo Fig. 1
        (0.5, 0.542),   # Approx from Figure 1
        (0.75, 0.25),   # Erven assumption: h_min = 1-r = 0.25
    ])
    def test_entropy_values_match_literature(self, r, expected):
        """Entropy values should match published figures."""
        result = max_bound_entropy(r)
        assert result == pytest.approx(expected, abs=0.02)
```

### 7.2 Property-Based Tests

```python
# tests/security/test_bounds_properties.py

from hypothesis import given, strategies as st

class TestBoundProperties:
    """Property-based tests for entropy bounds."""
    
    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_max_bound_in_valid_range(self, r):
        """Max bound should always be in [0, 1]."""
        result = max_bound_entropy(r)
        assert 0.0 <= result <= 1.0
    
    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_rational_bound_in_valid_range(self, r):
        """Rational bound should always be in [0, 0.5]."""
        result = rational_adversary_bound(r)
        assert 0.0 <= result <= 0.5
    
    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0)
    )
    def test_bounded_storage_monotonic_in_nu(self, r, nu):
        """More storage (larger ν) should decrease entropy."""
        h_full = bounded_storage_entropy(r, 1.0)
        h_bounded = bounded_storage_entropy(r, nu)
        assert h_bounded >= h_full
```

### 7.3 Integration Tests

```python
# tests/security/test_feasibility_integration.py

class TestFeasibilityIntegration:
    """Integration tests with Phase B configuration."""
    
    def test_erven_experimental_params_feasible(self):
        """Erven et al. experimental parameters should be feasible."""
        params = NSMParameters(
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            delta_t_ns=1e9,  # 1 second
            channel_fidelity=0.99,
            detection_eff_eta=0.015
        )
        checker = FeasibilityChecker(params)
        report = checker.run_preflight_checks()
        assert report.is_feasible
    
    def test_qber_above_22_percent_fails(self):
        """QBER > 22% should always fail."""
        params = NSMParameters(
            storage_noise_r=0.75,
            storage_rate_nu=1.0,
            channel_fidelity=0.5,  # 25% QBER from infidelity alone
        )
        checker = FeasibilityChecker(params)
        with pytest.raises(QBERThresholdExceeded):
            checker.run_preflight_checks()
    
    def test_storage_capacity_violation(self):
        """C_N · ν ≥ 0.5 should fail."""
        params = NSMParameters(
            storage_noise_r=0.99,  # Near-perfect storage
            storage_rate_nu=1.0,   # Can store everything
        )
        checker = FeasibilityChecker(params)
        with pytest.raises(NSMViolationError):
            checker.run_preflight_checks()
```

---

## 8. Acceptance Criteria

### 8.1 Functional Requirements

| ID | Requirement | Verification Method |
|----|-------------|---------------------|
| **SEC-F01** | `max_bound_entropy(r)` returns correct value for all r ∈ [0,1] | Unit tests against Table 5.2 |
| **SEC-F02** | `gamma_function(x)` correctly inverts g(y) for x < 0.5 | Numerical verification |
| **SEC-F03** | `check_qber_threshold()` raises at 22% hard limit | Exception test |
| **SEC-F04** | `check_strictly_less()` validates h(Q) < h_min(r) | Property test |
| **SEC-F05** | `check_storage_capacity()` validates C_N·ν < 0.5 | Integration test |
| **SEC-F06** | `compute_finite_key_length()` matches Erven Eq. (8) | Reference value test |
| **SEC-F07** | All checks compose in `run_preflight_checks()` | End-to-end test |

### 8.2 Non-Functional Requirements

| ID | Requirement | Threshold |
|----|-------------|-----------|
| **SEC-N01** | Entropy bound computation time | < 1ms per call |
| **SEC-N02** | Full preflight check time | < 10ms |
| **SEC-N03** | Memory usage | < 10 MB |
| **SEC-N04** | Test coverage | ≥ 95% line coverage |

### 8.3 Documentation Requirements

| ID | Requirement |
|----|-------------|
| **SEC-D01** | All public functions have Numpydoc docstrings |
| **SEC-D02** | Each formula cites source paper and equation number |
| **SEC-D03** | Module-level docstrings explain theoretical context |

---

## 9. References

### 9.1 Primary Sources

| Citation | Full Reference | Key Contribution |
|----------|----------------|------------------|
| **König et al. (2012)** | R. König, S. Wehner, J. Wullschleger. "Unconditional Security From Noisy Quantum Storage." *IEEE Trans. Inf. Theory* 58(3), 2012. | NSM definition, storage capacity constraint |
| **Schaffner et al. (2009)** | C. Schaffner, B. Terhal, S. Wehner. "Robust Cryptography in the Noisy-Quantum-Storage Model." *QIC* 9(11&12), 2009. | 11% QBER threshold, Corollary 7, robust protocol |
| **Lupo et al. (2023)** | C. Lupo et al. "Error-tolerant oblivious transfer in the noisy-storage model." arXiv:2308.05098, 2023. | Max Bound, virtual erasure, 22% limit |
| **Erven et al. (2014)** | C. Erven et al. "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model." *Phys. Rev. Lett.* 2014. | Experimental parameters, finite-key formula |
| **Dupuis et al. (2014)** | F. Dupuis, O. Fawzi, S. Wehner. "Entanglement Sampling and Applications." 2014. | Collision entropy bound |

### 9.2 Secondary Sources

| Citation | Full Reference | Relevance |
|----------|----------------|-----------|
| **Tomamichel et al. (2012)** | M. Tomamichel et al. "Tight Finite-Key Analysis for Quantum Cryptography." *Nat. Commun.* 2012. | Statistical fluctuation μ formula |
| **Hoeffding (1963)** | W. Hoeffding. "Probability Inequalities for Sums of Bounded Random Variables." *JASA* 1963. | Detection interval bounds |

### 9.3 Internal Documents

| Document | Path | Relevance |
|----------|------|-----------|
| Phase I Spec | `docs/implementation plan/phase_I.md` | Feasibility requirements |
| Phase IV Analysis | `docs/implementation plan/phase_IV_analysis.md` | Max Bound derivation |
| Phase B Spec | `docs/caligo/phase_b_spec.md` | NSMParameters interface |
| Architecture | `docs/caligo/caligo_architecture.md` | Package structure |

---

## Appendix A: Mathematical Derivations

### A.1 Binary Entropy Function

$$h(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$

Properties:
- $h(0) = h(1) = 0$
- $h(1/2) = 1$
- Symmetric: $h(p) = h(1-p)$

### A.2 Depolarizing Channel Capacity

For depolarizing channel with parameter $r$:

$$C_{\mathcal{N}} = 1 - h\left(\frac{1+r}{2}\right)$$

At $r = 0$ (complete depolarization): $C_{\mathcal{N}} = 0$
At $r = 1$ (perfect storage): $C_{\mathcal{N}} = 1$

### A.3 The g Function Inversion

The function $g(y)$ for $\Gamma$ calculation:

$$g(y) = -y \log_2(y) - (1-y) \log_2(1-y) + y - 1 = h(y) + y - 1$$

Properties:
- $g(0) = -1$
- $g(1/2) = 1/2$
- $g(1) = 0$
- Monotonically increasing on $[0, 1]$

Numerical inversion uses Brent's method on the interval $[0, 1]$.

---

## Appendix B: Code Template

```python
# caligo/security/bounds.py
"""
NSM entropy bounds for E-HOK protocol security analysis.

This module implements the entropy calculations required for
privacy amplification key length determination in the Noisy
Storage Model.

References
----------
- König et al. (2012), IEEE Trans. Inf. Theory
- Schaffner et al. (2009), QIC
- Lupo et al. (2023), arXiv:2308.05098
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import NamedTuple

from caligo.utils.math import binary_entropy


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

QBER_CONSERVATIVE_THRESHOLD: float = 0.11
"""Conservative QBER limit from Schaffner Corollary 7."""

QBER_ABSOLUTE_THRESHOLD: float = 0.22
"""Absolute QBER limit from Lupo Section VI."""

R_TILDE: float = 0.7798
"""Storage noise threshold: 2·h⁻¹(1/2) - 1."""

R_CROSSOVER: float = 0.82
"""Where Dupuis-König and Lupo bounds are equal."""


# ═══════════════════════════════════════════════════════════════════════════
# Core Bound Functions
# ═══════════════════════════════════════════════════════════════════════════

def gamma_function(x: float) -> float:
    """
    Compute the Γ function for collision entropy regularization.
    
    [Full docstring as specified in Section 4.1.1]
    """
    if x >= 0.5:
        return x
    
    # Numerically invert g(y) = h(y) + y - 1
    def g(y: float) -> float:
        if y <= 0 or y >= 1:
            return float('inf')
        return binary_entropy(y) + y - 1.0
    
    # Find y such that g(y) = x
    return brentq(lambda y: g(y) - x, 1e-10, 0.5)


def max_bound_entropy(r: float) -> float:
    """
    Compute the optimal min-entropy bound.
    
    [Full docstring as specified in Section 4.1.5]
    """
    h_dk = dupuis_konig_bound(r)
    h_lupo = lupo_virtual_erasure_bound(r)
    return max(h_dk, h_lupo)


# [Additional implementations as specified...]
```

---

*End of Phase C Specification*
