# E-HOK Baseline: Formal Testing & Validation Specification

**Document Version:** 1.0  
**Purpose:** Formal specification of test requirements, validation criteria, and verification procedures for the E-HOK baseline protocol.  
**Framework:** SquidASM (NetQASM + NetSquid)  
**Language:** Python 3.10+

---

## Executive Summary

This document provides a mathematically rigorous specification of testing requirements for the E-HOK baseline protocol. Tests are structured hierarchically: **unit tests** validate individual components, **integration tests** verify subsystem interactions, and **system tests** validate end-to-end protocol properties including security, correctness, and performance.

**Design Principles:**
1. **Minimality:** No redundant tests; each test validates a unique property
2. **Formality:** All acceptance criteria mathematically defined
3. **Verifiability:** All requirements are deterministic and falsifiable
4. **Traceability:** Each test maps to specific requirements in `e-hok-baseline.md`

---

## Table of Contents

1. [Testing Hierarchy & Scope](#1-testing-hierarchy--scope)
2. [Phase 0: Foundation Tests](#2-phase-0-foundation-tests)
3. [Phase 1: Quantum Generation Tests](#3-phase-1-quantum-generation-tests)
4. [Phase 2: Commitment Tests](#4-phase-2-commitment-tests)
5. [Phase 3: Sifting & Sampling Tests](#5-phase-3-sifting--sampling-tests)
6. [Phase 4: Reconciliation Tests](#6-phase-4-reconciliation-tests)
7. [Phase 5: Privacy Amplification Tests](#7-phase-5-privacy-amplification-tests)
8. [Phase 6: Integration Tests](#8-phase-6-integration-tests)
9. [Phase 7: System Verification Tests](#9-phase-7-system-verification-tests)
10. [Statistical & Performance Tests](#10-statistical--performance-tests)

---

## 1. Testing Hierarchy & Scope

### 1.1 Test Classification

| Level | Scope | Dependencies | Execution Time | Coverage Goal |
|-------|-------|--------------|----------------|---------------|
| **Unit** | Single function/class | Mocked external dependencies | < 1s per test | ≥90% line coverage |
| **Integration** | Component interactions | Real dependencies, minimal network | < 10s per test | All interfaces verified |
| **System** | End-to-end protocol | Full SquidASM network simulation | < 60s per test | Security properties proven |

### 1.2 Test Environment Requirements

**Hardware:**
- CPU: 4 cores minimum (for NetSquid parallelization)
- RAM: 8 GB minimum
- Storage: 100 MB (for LDPC matrices)

**Software:**
- Python 3.10+
- SquidASM v0.10+
- NetQASM v0.12+
- NetSquid v1.1+ (licensed)
- pytest v7.0+
- numpy v1.24+
- scipy v1.10+

**Network Configuration:** All tests use isolated YAML configurations with explicit noise parameters.

---

## 2. Phase 0: Foundation Tests

### 2.1 Unit Test: Data Structure Validation

**Test ID:** `test_foundation::test_data_structures`  
**Requirement:** Data structures must enforce type constraints and value ranges.

#### Test Case 2.1.1: ObliviousKey Construction

**Preconditions:**
```python
key_value = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
knowledge_mask = np.array([0, 0, 1, 1, 0], dtype=np.uint8)
security_param = 1e-9
qber = 0.03
final_length = 5
```

**Operation:**
```python
ok = ObliviousKey(
    key_value=key_value,
    knowledge_mask=knowledge_mask,
    security_param=security_param,
    qber=qber,
    final_length=final_length
)
```

**Postconditions (∀ must hold):**
1. `ok.key_value.dtype == np.uint8`
2. `ok.knowledge_mask.dtype == np.uint8`
3. `len(ok.key_value) == len(ok.knowledge_mask) == final_length`
4. `∀i: ok.key_value[i] ∈ {0, 1}`
5. `∀i: ok.knowledge_mask[i] ∈ {0, 1}`
6. `0 ≤ ok.qber ≤ 1`
7. `ok.security_param > 0`

**Failure Injection Tests:**
| Invalid Input | Expected Behavior |
|---------------|-------------------|
| `key_value` contains 2 | `ValueError` or assertion failure |
| `key_value.dtype == float64` | `ValueError` or type error |
| `len(key_value) ≠ len(knowledge_mask)` | `ValueError` |
| `security_param < 0` | `ValueError` |

**Acceptance Criterion:**
$$\text{PASS} \iff \text{all postconditions hold} \land \text{all failure injections raise exceptions}$$

---

### 2.2 Unit Test: Abstract Interface Enforcement

**Test ID:** `test_foundation::test_abstract_interfaces`  
**Requirement:** Abstract base classes cannot be instantiated; concrete classes must implement all methods.

#### Test Case 2.2.1: ICommitmentScheme

**Preconditions:** None

**Operation:**
```python
try:
    scheme = ICommitmentScheme()
    instantiation_succeeded = True
except TypeError:
    instantiation_succeeded = False
```

**Postcondition:**
$$\text{instantiation\_succeeded} = \text{False}$$

#### Test Case 2.2.2: Concrete Implementation Verification

**Preconditions:**
```python
from ehok.implementations.commitment.sha256_commitment import SHA256Commitment
```

**Operation:**
```python
scheme = SHA256Commitment()
has_commit = callable(getattr(scheme, 'commit', None))
has_verify = callable(getattr(scheme, 'verify', None))
has_open = callable(getattr(scheme, 'open_subset', None))
```

**Postconditions:**
1. `has_commit = True`
2. `has_verify = True`
3. `has_open = True`
4. `isinstance(scheme, ICommitmentScheme) = True`

**Acceptance Criterion:**
$$\text{PASS} \iff \text{ABC non-instantiable} \land \text{concrete class implements all abstract methods}$$

---

### 2.3 Unit Test: Exception Hierarchy

**Test ID:** `test_foundation::test_exception_hierarchy`  
**Requirement:** Custom exceptions must form correct inheritance chain.

#### Test Case 2.3.1: Inheritance Verification

**Preconditions:** Exception classes defined in `ehok/core/exceptions.py`

**Operation & Postconditions:**

| Exception Class | Parent Class | Verification |
|----------------|--------------|--------------|
| `EHOKException` | `Exception` | `issubclass(EHOKException, Exception)` |
| `SecurityException` | `EHOKException` | `issubclass(SecurityException, EHOKException)` |
| `QBERTooHighError` | `SecurityException` | `issubclass(QBERTooHighError, SecurityException)` |
| `ProtocolError` | `EHOKException` | `issubclass(ProtocolError, EHOKException)` |
| `ReconciliationFailedError` | `ProtocolError` | `issubclass(ReconciliationFailedError, ProtocolError)` |
| `CommitmentVerificationError` | `SecurityException` | `issubclass(CommitmentVerificationError, SecurityException)` |

**Acceptance Criterion:**
$$\text{PASS} \iff \forall \text{ (class, parent) pairs: verification holds}$$

---

### 2.4 Unit Test: Logging Infrastructure

**Test ID:** `test_foundation::test_logging`  
**Requirement:** Logging must use SquidASM LogManager, never print().

#### Test Case 2.4.1: Logger Creation

**Preconditions:**
```python
from ehok.utils.logging import get_logger
```

**Operation:**
```python
logger = get_logger("test_module")
```

**Postconditions:**
1. `logger is not None`
2. `hasattr(logger, 'info')`
3. `hasattr(logger, 'debug')`
4. `hasattr(logger, 'error')`
5. Logger name contains "ehok.test_module"

#### Test Case 2.4.2: No Print Statements in Codebase

**Preconditions:** All Python files in `ehok/` directory

**Operation:**
```bash
grep -r "print(" ehok/ --include="*.py" | grep -v "test_" | grep -v "#"
```

**Postcondition:**
$$\text{matches} = \emptyset \quad \text{(no print statements found)}$$

**Acceptance Criterion:**
$$\text{PASS} \iff \text{logger created} \land \text{no print() in production code}$$

---

## 3. Phase 1: Quantum Generation Tests

### 3.1 Unit Test: Basis Selection Randomness

**Test ID:** `test_quantum::test_basis_randomness`  
**Requirement:** Basis choices must be uniformly random and independent.

#### Test Case 3.1.1: Uniform Distribution

**Preconditions:**
```python
from ehok.quantum.basis_selection import BasisSelector
selector = BasisSelector()
N = 10000
```

**Operation:**
```python
bases = selector.generate_bases(N)
```

**Postconditions:**
Let $n_Z = \sum_{i=1}^{N} \mathbb{1}[bases[i] = 0]$ (count of Z-basis)  
Let $n_X = N - n_Z$ (count of X-basis)

**Statistical Hypothesis Test:**
$$H_0: P(basis = Z) = 0.5 \quad \text{vs} \quad H_1: P(basis = Z) \neq 0.5$$

**Test Statistic:**
$$z = \frac{n_Z - N/2}{\sqrt{N/4}}$$

Under $H_0$, $z \sim \mathcal{N}(0, 1)$ approximately for large $N$.

**Acceptance Criterion:**
$$|z| < 3 \quad \text{(3-sigma confidence interval)}$$

**Numerical Example:**
For $N = 10000$:
- Expected: $n_Z = 5000$
- Acceptance range: $[4850, 5150]$ (3σ ≈ 150)

#### Test Case 3.1.2: Independence Test

**Preconditions:** Same as 3.1.1

**Operation:**
```python
bases_alice = selector.generate_bases(N)
bases_bob = selector.generate_bases(N)
```

**Postcondition:**
Compute empirical joint distribution:
$$p_{ij} = \frac{\#\{k : bases\_alice[k] = i \land bases\_bob[k] = j\}}{N}$$

**Expected (under independence):**
$$p_{00} \approx p_{01} \approx p_{10} \approx p_{11} \approx 0.25$$

**Chi-Square Test:**
$$\chi^2 = N \sum_{i,j \in \{0,1\}} \frac{(p_{ij} - 0.25)^2}{0.25}$$

Under independence, $\chi^2 \sim \chi^2(3)$ (3 degrees of freedom).

**Acceptance Criterion:**
$$\chi^2 < 11.34 \quad \text{(95% confidence, df=3)}$$

---

### 3.2 Integration Test: EPR Generation & Measurement

**Test ID:** `test_quantum::test_epr_generation_perfect`  
**Requirement:** EPR generation with perfect link must produce perfectly correlated outcomes in matching bases.

#### Test Case 3.2.1: Perfect Link Correlation

**Network Configuration:**
```yaml
links:
  - stack1: alice
    stack2: bob
    typ: perfect
    cfg: {}
```

**Preconditions:**
- Alice and Bob both use Z-basis (basis = 0) for all measurements
- Number of EPR pairs: $N = 100$

**Operation:**
1. Alice: `results_alice = epr_socket.create_measure(number=N)`
2. Bob: `results_bob = epr_socket.recv_measure(number=N)`
3. Extract outcomes: `outcomes_alice = [r.measurement_outcome for r in results_alice]`

**Postcondition (Perfect Correlation):**
$$\forall i \in [0, N): outcomes\_alice[i] = outcomes\_bob[i]$$

**Quantitative Criterion:**
$$\text{agreement\_rate} = \frac{\sum_{i=0}^{N-1} \mathbb{1}[outcomes\_alice[i] = outcomes\_bob[i]]}{N} = 1.0$$

**Acceptance Criterion:**
$$\text{agreement\_rate} = 1.0 \quad \text{(100% correlation)}$$

---

### 3.3 Integration Test: Noisy Link QBER

**Test ID:** `test_quantum::test_epr_generation_noisy`  
**Requirement:** Depolarizing noise with fidelity $F$ must produce QBER consistent with theory.

#### Test Case 3.3.1: QBER Estimation

**Network Configuration:**
```yaml
links:
  - stack1: alice
    stack2: bob
    typ: depolarise
    cfg:
      fidelity: 0.96  # F = 0.96
      prob_success: 1.0
```

**Theoretical QBER:**
For depolarizing channel, QBER in matching bases:
$$\text{QBER}_{\text{theory}} = \frac{3}{4}(1 - F) = 0.75 \times 0.04 = 0.03 = 3\%$$

**Preconditions:**
- Both nodes use Z-basis for all measurements
- Number of EPR pairs: $N = 5000$ (large for statistical significance)

**Operation:**
1. Generate $N$ EPR pairs
2. Measure all in Z-basis
3. Compute empirical QBER:
$$\text{QBER}_{\text{emp}} = \frac{\sum_{i=0}^{N-1} \mathbb{1}[outcomes\_alice[i] \neq outcomes\_bob[i]]}{N}$$

**Confidence Interval:**
For binomial distribution with $p = 0.03$, $n = 5000$:
$$\sigma = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.03 \times 0.97}{5000}} \approx 0.0024$$

**Acceptance Criterion (3σ):**
$$|\text{QBER}_{\text{emp}} - 0.03| < 3 \times 0.0024 = 0.0072$$

Equivalently:
$$\text{QBER}_{\text{emp}} \in [0.0228, 0.0372] \quad \text{or roughly } [2.3\%, 3.7\%]$$

---

### 3.4 Unit Test: Batching Manager

**Test ID:** `test_quantum::test_batching_manager`  
**Requirement:** Batching manager must correctly partition total pairs into memory-constrained batches.

#### Test Case 3.4.1: Batch Size Computation

**Preconditions:**
```python
from ehok.quantum.batching_manager import BatchingManager
manager = BatchingManager(batch_size=5, total_pairs=10003)
```

**Operation:**
```python
batch_sizes = manager.compute_batch_sizes()
```

**Postconditions:**
1. `len(batch_sizes) == 2001` (ceiling of 10003/5)
2. `sum(batch_sizes) == 10003`
3. `all(b == 5 for b in batch_sizes[:-1])` (all but last batch have size 5)
4. `batch_sizes[-1] == 3` (last batch has remainder)

**Mathematical Invariant:**
$$\sum_{i=1}^{M} b_i = N \quad \text{where } M = \lceil N / B \rceil$$

**Acceptance Criterion:**
$$\text{PASS} \iff \text{all postconditions hold}$$

---

## 4. Phase 2: Commitment Tests

### 4.1 Unit Test: SHA-256 Commitment

**Test ID:** `test_commitment::test_sha256_commit_verify`  
**Requirement:** SHA-256 commitment must be computationally binding and verifiable.

#### Test Case 4.1.1: Commit-Verify Correctness

**Preconditions:**
```python
from ehok.implementations.commitment.sha256_commitment import SHA256Commitment
scheme = SHA256Commitment()
data = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.uint8)
```

**Operation:**
```python
commitment, decom_info = scheme.commit(data)
is_valid = scheme.verify(commitment, data, decom_info)
```

**Postconditions:**
1. `len(commitment) == 32` (SHA-256 outputs 256 bits = 32 bytes)
2. `is_valid == True`
3. `isinstance(commitment, bytes)`

#### Test Case 4.1.2: Binding Property (Negative Test)

**Preconditions:** Same commitment and decom_info as above

**Operation:**
```python
fake_data = np.array([1, 0, 0, 1, 0, 1, 0, 0], dtype=np.uint8)  # Different from original
is_valid_fake = scheme.verify(commitment, fake_data, decom_info)
```

**Postcondition:**
$$is\_valid\_fake = \text{False}$$

**Security Property (Informal):**
Finding $data' \neq data$ such that $\text{SHA256}(data') = \text{SHA256}(data)$ requires $\approx 2^{128}$ operations (collision resistance).

**Acceptance Criterion:**
$$\text{PASS} \iff \text{correct data verifies} \land \text{incorrect data rejected}$$

---

### 4.2 Unit Test: Subset Opening

**Test ID:** `test_commitment::test_subset_opening`  
**Requirement:** Commitment scheme must support selective opening of subset indices.

#### Test Case 4.2.1: Correct Subset Opening

**Preconditions:**
```python
scheme = SHA256Commitment()
full_data = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
commitment, decom_info = scheme.commit(full_data)
test_indices = np.array([1, 3, 7, 9], dtype=np.int64)
```

**Operation:**
```python
subset_data, proof = scheme.open_subset(test_indices, full_data, decom_info)
is_valid = scheme.verify_subset_opening(commitment, test_indices, subset_data, proof)
```

**Postconditions:**
1. `len(subset_data) == len(test_indices)`
2. `∀i ∈ [0, len(test_indices)): subset_data[i] == full_data[test_indices[i]]`
3. `is_valid == True`

**Mathematical Property:**
$$\text{subset\_data} = \text{full\_data}|_{T} \quad \text{where } T = \text{test\_indices}$$

#### Test Case 4.2.2: Tampered Subset Rejected

**Preconditions:** Same as 4.2.1

**Operation:**
```python
tampered_subset = subset_data.copy()
tampered_subset[0] = 1 - tampered_subset[0]  # Flip first bit
is_valid_tampered = scheme.verify_subset_opening(
    commitment, test_indices, tampered_subset, proof
)
```

**Postcondition:**
$$is\_valid\_tampered = \text{False} \quad \text{or exception raised}$$

**Acceptance Criterion:**
$$\text{PASS} \iff \text{correct subset accepted} \land \text{tampered subset rejected}$$

---

## 5. Phase 3: Sifting & Sampling Tests

### 5.1 Unit Test: Basis Matching

**Test ID:** `test_sifting::test_basis_matching`  
**Requirement:** Sifting must correctly identify matching and non-matching basis indices.

#### Test Case 5.1.1: Matching Indices Identification

**Preconditions:**
```python
from ehok.core.sifting import SiftingManager
sifter = SiftingManager()
bases_alice = np.array([0, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
bases_bob =   np.array([0, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
```

**Expected Sets:**
- Matching: $I_0 = \{0, 2, 3, 5, 6\}$ (indices where bases match)
- Non-matching: $I_1 = \{1, 4, 7\}$ (indices where bases differ)

**Operation:**
```python
I_0, I_1 = sifter.identify_matching_bases(bases_alice, bases_bob)
```

**Postconditions:**
1. `set(I_0) == {0, 2, 3, 5, 6}`
2. `set(I_1) == {1, 4, 7}`
3. `len(I_0) + len(I_1) == len(bases_alice)`
4. `set(I_0).intersection(set(I_1)) == ∅`

**Mathematical Property:**
$$I_0 = \{i : bases\_alice[i] = bases\_bob[i]\}$$
$$I_1 = \{i : bases\_alice[i] \neq bases\_bob[i]\}$$
$$I_0 \cap I_1 = \emptyset \land I_0 \cup I_1 = [0, N)$$

**Acceptance Criterion:**
$$\text{PASS} \iff \text{all postconditions hold}$$

---

### 5.2 Unit Test: QBER Estimation

**Test ID:** `test_sifting::test_qber_estimation`  
**Requirement:** QBER estimation on test set must be statistically accurate.

#### Test Case 5.2.1: Known Error Rate

**Preconditions:**
```python
sifter = SiftingManager()
N = 1000
outcomes_alice = np.random.randint(0, 2, size=N, dtype=np.uint8)
# Introduce 5% errors in Bob's outcomes
outcomes_bob = outcomes_alice.copy()
error_indices = np.random.choice(N, size=50, replace=False)  # 50 errors
outcomes_bob[error_indices] = 1 - outcomes_bob[error_indices]
test_set = np.arange(N)  # Use all indices for test
```

**Operation:**
```python
qber = sifter.estimate_qber(outcomes_alice, outcomes_bob, test_set)
```

**Postcondition:**
$$qber = 50/1000 = 0.05 = 5\%$$

**Acceptance Criterion:**
$$|qber - 0.05| < 10^{-10} \quad \text{(exact computation)}$$

#### Test Case 5.2.2: QBER Abort Threshold

**Preconditions:**
```python
from ehok.core.constants import QBER_THRESHOLD
# QBER_THRESHOLD = 0.11
```

**Test Cases:**

| QBER Value | Expected Behavior |
|------------|-------------------|
| 0.10 | No exception (below threshold) |
| 0.11 | No exception (at threshold) |
| 0.12 | `QBERTooHighError` raised |
| 0.15 | `QBERTooHighError` raised |

**Acceptance Criterion:**
$$\text{PASS} \iff \forall \text{QBER} > \tau: \text{abort} \land \forall \text{QBER} \leq \tau: \text{continue}$$

Where $\tau = 0.11$ is the QBER threshold.

---

### 5.3 Unit Test: Test Set Selection

**Test ID:** `test_sifting::test_test_set_selection`  
**Requirement:** Test set must be deterministic from seed and match expected fraction.

#### Test Case 5.3.1: Fraction Verification

**Preconditions:**
```python
from ehok.core.constants import TEST_SET_FRACTION
# TEST_SET_FRACTION = 0.1
sifter = SiftingManager()
I_0 = np.arange(5000)  # Assume 5000 matching bases
seed = 42
```

**Operation:**
```python
test_set, key_set = sifter.select_test_set(I_0, seed=seed)
```

**Postconditions:**
1. `len(test_set) == int(5000 * 0.1) == 500`
2. `len(key_set) == 5000 - 500 == 4500`
3. `set(test_set).union(set(key_set)) == set(I_0)`
4. `set(test_set).intersection(set(key_set)) == ∅`

**Determinism Test:**
```python
test_set_2, key_set_2 = sifter.select_test_set(I_0, seed=seed)
assert np.array_equal(test_set, test_set_2)  # Same seed → same selection
```

**Acceptance Criterion:**
$$|test\_set| = \lfloor |I_0| \times f \rfloor \land test\_set \cap key\_set = \emptyset$$

Where $f = 0.1$ is TEST_SET_FRACTION.

---

## 6. Phase 4: Reconciliation Tests

### 6.1 Unit Test: Syndrome Computation

**Test ID:** `test_reconciliation::test_syndrome_computation`  
**Requirement:** Syndrome must satisfy $S = H \cdot k \mod 2$.

#### Test Case 6.1.1: Known Key Syndrome

**Preconditions:**
```python
from ehok.implementations.reconciliation.ldpc_reconciliator import LDPCReconciliator
import scipy.sparse as sp

# Simple parity-check matrix: 3x6
H = sp.csr_matrix([
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1]
])
reconciliator = LDPCReconciliator(H)
key = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
```

**Expected Syndrome (manual calculation):**
```
S[0] = (1⊕0⊕1) mod 2 = 0
S[1] = (0⊕1⊕0) mod 2 = 1
S[2] = (1⊕1⊕1) mod 2 = 1
```

**Operation:**
```python
syndrome = reconciliator.compute_syndrome(key)
```

**Postcondition:**
$$syndrome = [0, 1, 1]$$

**Mathematical Verification:**
$$S = H \cdot k \mod 2 \quad \text{(matrix-vector product in GF(2))}$$

**Acceptance Criterion:**
$$\text{computed syndrome} = \text{expected syndrome}$$

---

### 6.2 Integration Test: Error Correction

**Test ID:** `test_reconciliation::test_error_correction`  
**Requirement:** LDPC BP decoder must correct errors below code capacity.

#### Test Case 6.2.1: Known Error Pattern

**Preconditions:**
```python
# Load regular LDPC matrix with rate 0.5, design QBER ≈ 10%
H = load_ldpc_matrix("ldpc_1000_rate05.npz")
reconciliator = LDPCReconciliator(H)

# Generate random key
np.random.seed(42)
n = H.shape[1]
alice_key = np.random.randint(0, 2, size=n, dtype=np.uint8)

# Introduce 5% errors
num_errors = int(0.05 * n)
error_positions = np.random.choice(n, size=num_errors, replace=False)
bob_key = alice_key.copy()
bob_key[error_positions] = 1 - bob_key[error_positions]

# Alice computes syndrome
syndrome = reconciliator.compute_syndrome(alice_key)
```

**Operation:**
```python
bob_corrected_key, converged = reconciliator.decode(bob_key, syndrome, max_iter=50)
```

**Postconditions:**
1. `converged == True` (BP converges)
2. `np.array_equal(bob_corrected_key, alice_key)` (perfect correction)
3. `error_rate = np.sum(bob_corrected_key != alice_key) / n == 0`

**Performance Requirement:**
Convergence in $< 50$ iterations for QBER $\leq 10\%$.

**Acceptance Criterion:**
$$\text{PASS} \iff \text{converged} \land \text{(corrected key = Alice's key)}$$

#### Test Case 6.2.2: Failure at High QBER

**Preconditions:** Same LDPC matrix, but introduce 15% errors (above capacity)

**Operation:**
```python
num_errors = int(0.15 * n)
error_positions = np.random.choice(n, size=num_errors, replace=False)
bob_key_bad = alice_key.copy()
bob_key_bad[error_positions] = 1 - bob_key_bad[error_positions]
bob_corrected, converged = reconciliator.decode(bob_key_bad, syndrome, max_iter=50)
```

**Expected Behavior:**
$$\text{converged} = \text{False} \quad \text{(decoder fails to converge)}$$

OR

$$\text{error\_rate} = \frac{\sum (bob\_corrected \neq alice\_key)}{n} > 0.01 \quad \text{(residual errors)}$$

**Acceptance Criterion:**
$$\text{PASS} \iff \text{decoder recognizes failure (either non-convergence or residual errors)}$$

---

### 6.3 Unit Test: Leakage Estimation

**Test ID:** `test_reconciliation::test_leakage_estimation`  
**Requirement:** Information leakage must be bounded by syndrome length.

#### Test Case 6.3.1: Leakage Formula

**Preconditions:**
```python
m = 500  # Number of parity checks (syndrome length)
qber = 0.05
```

**Theoretical Upper Bound:**
$$\text{leakage} \leq m = 500 \text{ bits}$$

**Operation:**
```python
reconciliator = LDPCReconciliator(H)  # Assume H.shape[0] = m
leakage = reconciliator.estimate_leakage(m, qber)
```

**Postcondition:**
$$\text{leakage} \leq m$$

**Tighter Bound (Shannon limit):**
For rate $R = k/n = 0.5$ code:
$$\text{leakage} \approx m + \log_2(\text{# iterations}) \times \text{message\_overhead}$$

**Acceptance Criterion:**
$$m \leq \text{leakage} \leq m \times 1.2 \quad \text{(20% overhead allowed)}$$

---

## 7. Phase 5: Privacy Amplification Tests

### 7.1 Unit Test: Toeplitz Matrix Construction

**Test ID:** `test_privacy_amplification::test_toeplitz_construction`  
**Requirement:** Toeplitz matrix must be correctly generated from seed.

#### Test Case 7.1.1: Matrix Structure

**Preconditions:**
```python
from ehok.implementations.privacy_amplification.toeplitz_amplifier import ToeplitzAmplifier
amplifier = ToeplitzAmplifier()
input_length = 6
output_length = 4
```

**Operation:**
```python
seed = amplifier.generate_hash_seed(input_length, output_length)
```

**Postconditions:**
1. `len(seed) == output_length + input_length - 1 == 9`
2. `seed.dtype == np.uint8`
3. `∀i: seed[i] ∈ {0, 1}`

**Toeplitz Structure Verification:**

For seed `s = [s0, s1, ..., s8]`, the matrix should be:
```
T = [
    [s5, s6, s7, s8, s9, s10],  # (actually s[5], s[6], ..., s[10] doesn't exist)
    [s4, s5, s6, s7, s8, s9],   # Error in indexing - need clarification
    ...
]
```

**Corrected Property:**
Row $i$ of Toeplitz matrix $T$ is constructed from seed as:
$$T_{i,j} = seed[i - j + (n-1)] \quad \text{where } n = \text{input\_length}$$

**Acceptance Criterion:**
$$\text{PASS} \iff \text{seed length correct} \land \text{seed binary}$$

---

### 7.2 Integration Test: Compression & Leftover Hash

**Test ID:** `test_privacy_amplification::test_compression`  
**Requirement:** Compression must satisfy leftover hash lemma bounds.

#### Test Case 7.2.1: Length Reduction

**Preconditions:**
```python
amplifier = ToeplitzAmplifier()
sifted_length = 1000
qber = 0.05
leakage = 500  # From reconciliation
epsilon = 1e-9
```

**Operation:**
```python
final_length = amplifier.compute_final_length(sifted_length, qber, leakage, epsilon)
```

**Expected Calculation:**
```python
h_qber = -0.05*log2(0.05) - 0.95*log2(0.95) ≈ 0.286
min_entropy = 1000 * (1 - 0.286) ≈ 714
epsilon_cost = 2 * log2(1/1e-9) ≈ 59.8
final_length ≈ 714 - 500 - 60 - 100 ≈ 54
```

**Postcondition:**
$$final\_length > 0 \land final\_length < sifted\_length$$

**Security Constraint:**
$$final\_length \leq sifted\_length \times [1 - h(qber)] - leakage - 2\log_2(1/\epsilon) - \text{margin}$$

**Acceptance Criterion:**
$$\text{PASS} \iff \text{computed length satisfies security bound}$$

#### Test Case 7.2.2: Actual Compression

**Preconditions:**
```python
key = np.random.randint(0, 2, size=1000, dtype=np.uint8)
seed = amplifier.generate_hash_seed(1000, 54)
```

**Operation:**
```python
compressed = amplifier.compress(key, seed)
```

**Postconditions:**
1. `len(compressed) == 54`
2. `compressed.dtype == np.uint8`
3. `∀i: compressed[i] ∈ {0, 1}`

**Determinism:**
```python
compressed2 = amplifier.compress(key, seed)
assert np.array_equal(compressed, compressed2)  # Same seed → same output
```

**Acceptance Criterion:**
$$\text{PASS} \iff \text{output length correct} \land \text{deterministic}$$

---

### 7.3 Statistical Test: Output Uniformity

**Test ID:** `test_privacy_amplification::test_output_uniformity`  
**Requirement:** Compressed keys must be statistically indistinguishable from uniform random.

#### Test Case 7.3.1: Chi-Square Uniformity Test

**Preconditions:**
```python
num_trials = 10000
output_length = 10  # Short for statistical testing
amplifier = ToeplitzAmplifier()
```

**Operation:**
```python
outputs = []
for _ in range(num_trials):
    key = np.random.randint(0, 2, size=100, dtype=np.uint8)
    seed = amplifier.generate_hash_seed(100, output_length)
    compressed = amplifier.compress(key, seed)
    outputs.append(tuple(compressed))

# Count frequency of each 10-bit string
from collections import Counter
counts = Counter(outputs)
```

**Expected Distribution:**
Under uniform randomness, each of $2^{10} = 1024$ strings should appear $\approx 10000/1024 \approx 9.77$ times.

**Chi-Square Test:**
$$\chi^2 = \sum_{i=0}^{1023} \frac{(O_i - E)^2}{E}$$

Where $O_i$ is observed count, $E = 10000/1024 \approx 9.77$.

Under uniformity, $\chi^2 \sim \chi^2(1023)$.

**Acceptance Criterion:**
$$\chi^2 < 1101 \quad \text{(95% confidence for df=1023)}$$

**Note:** This is a weak test. For production, use NIST Statistical Test Suite.

---

## 8. Phase 6: Integration Tests

### 8.1 Integration Test: Phase Sequencing

**Test ID:** `test_integration::test_phase_sequencing`  
**Requirement:** Protocol phases must execute in strict order with proper synchronization.

#### Test Case 8.1.1: Commitment Before Basis Reveal

**Objective:** Verify Alice waits for Bob's commitment before sending bases.

**Network Configuration:** 2 nodes, perfect link

**Alice's Program Logic:**
```python
# Phase 1: Quantum generation
generate_epr_pairs()

# Phase 2: MUST wait for commitment
commitment = yield from csocket.recv()  # Blocking receive

# Phase 3: Only now send bases
csocket.send(bases)
```

**Bob's Program Logic:**
```python
# Phase 1: Quantum generation
generate_epr_pairs()

# Phase 2: Send commitment FIRST
csocket.send(compute_commitment())

# Phase 3: Wait for bases
bases = yield from csocket.recv()
```

**Verification:**
1. Monitor message timestamps in logs
2. Verify: $t_{commit} < t_{bases}$ (commitment sent before bases)
3. Verify: Alice's receive blocks until Bob sends

**Acceptance Criterion:**
$$\text{PASS} \iff \text{commitment transmitted before basis reveal} \land \text{no timeout errors}$$

---

### 8.2 Integration Test: Synchronization Points

**Test ID:** `test_integration::test_synchronization`  
**Requirement:** Classical communication must be properly synchronized with connection.flush().

#### Test Case 8.2.1: Flush After EPR Generation

**Preconditions:** 2 nodes, generate 100 EPR pairs

**Alice's Code:**
```python
results = epr_socket.create_measure(number=100)
yield from context.connection.flush()  # REQUIRED before accessing results
outcomes = [r.measurement_outcome for r in results]
```

**Bob's Code:**
```python
results = epr_socket.recv_measure(number=100)
yield from context.connection.flush()  # REQUIRED
outcomes = [r.measurement_outcome for r in results]
```

**Failure Injection:**
Remove `flush()` and verify that accessing `measurement_outcome` raises error or returns undefined values.

**Acceptance Criterion:**
$$\text{PASS} \iff \text{flush() required before result access} \land \text{omitting flush() causes failure}$$

---

## 9. Phase 7: System Verification Tests

### 9.1 System Test: Honest Execution (No Noise)

**Test ID:** `test_system::test_honest_execution_perfect`  
**Requirement:** End-to-end protocol must succeed with perfect link, producing secure oblivious keys.

**Requirement Traceability:** Maps to Test 1 in `e-hok-baseline.md`

#### Test Configuration

**Network:**
```yaml
links:
  - stack1: alice
    stack2: bob
    typ: perfect
    cfg: {}
```

**Parameters:**
- EPR pairs: $N = 1000$ (reduced for faster testing)
- Expected matching bases: $|I_0| \approx 500$ (50% probability)
- Expected QBER: $0\%$ (perfect link)

#### Execution

**Operation:**
```python
results = run(
    network_config,
    programs={"alice": AliceEHOKProgram(), "bob": BobEHOKProgram()},
    num_times=1
)
alice_result = results[0]["alice"]
bob_result = results[0]["bob"]
```

#### Postconditions

##### P1: Protocol Success
$$alice\_result["success"] = \text{True} \land bob\_result["success"] = \text{True}$$

##### P2: QBER Verification
$$alice\_result["qber"] = 0.0 \quad \text{(perfect link)}$$

##### P3: Key Agreement
```python
alice_key = alice_result["oblivious_key"].key_value
bob_key = bob_result["oblivious_key"].key_value
assert np.array_equal(alice_key, bob_key)
```

##### P4: Oblivious Property (Bob's Knowledge Mask)

Bob's knowledge mask must indicate which bits are "unknown":
```python
bob_mask = bob_result["oblivious_key"].knowledge_mask
# mask[i] = 0 if Bob measured in matching basis (I_0)
# mask[i] = 1 if Bob measured in non-matching basis (I_1)

fraction_unknown = np.mean(bob_mask)
# Expected: ~0% unknown after privacy amplification (all matching bases used)
assert fraction_unknown < 0.1  # Less than 10% unknown
```

**Note:** After privacy amplification, knowledge mask structure may differ from raw measurements. This requires clarification in implementation.

##### P5: Final Key Length

Expected maximum length (no errors, no leakage):
$$m \approx |I_0| \times [1 - h(0)] - 0 - 2\log_2(1/10^{-9}) - 100$$
$$m \approx 500 \times 1 - 60 - 100 = 340 \text{ bits}$$

**Acceptance Range:**
$$300 \leq alice\_result["final\_count"] \leq 400$$

#### Acceptance Criterion

$$\text{PASS} \iff \bigwedge_{i=1}^{5} P_i \text{ (all postconditions hold)}$$

---

### 9.2 System Test: Noise Tolerance (5% QBER)

**Test ID:** `test_system::test_noise_tolerance`  
**Requirement:** Protocol must succeed under realistic noise conditions.

**Requirement Traceability:** Maps to Test 2 in `e-hok-baseline.md`

#### Test Configuration

**Network:**
```yaml
links:
  - stack1: alice
    stack2: bob
    typ: depolarise
    cfg:
      fidelity: 0.95
      prob_success: 1.0
```

**Expected QBER:** $0.75 \times 0.05 = 0.0375 = 3.75\%$

**Parameters:**
- EPR pairs: $N = 5000$
- Expected sifted: $|I_0| \approx 2500$

#### Postconditions

##### P1: QBER Within Expected Range
$$0.03 \leq alice\_result["qber"] \leq 0.045 \quad \text{(3% to 4.5%)}$$

##### P2: Reconciliation Success
$$alice\_result["reconciliation\_success"] = \text{True}$$

##### P3: Key Agreement After Reconciliation
```python
# Keys must match after reconciliation but before PA
assert np.array_equal(alice_reconciled_key, bob_reconciled_key)
```

##### P4: Final Key Length Reduction

Due to noise and leakage:
$$m < |I_0| \times 0.8 \quad \text{(at least 20% reduction)}$$

##### P5: Security Parameter
$$alice\_result["oblivious\_key"].security\_param \leq 10^{-9}$$

#### Statistical Requirement

Run test $n = 10$ times with different random seeds:
$$\text{success\_rate} = \frac{\text{\# successful runs}}{n} \geq 0.9 \quad \text{(90% success rate)}$$

#### Acceptance Criterion

$$\text{PASS} \iff \text{all postconditions hold in } \geq 90\% \text{ of runs}$$

---

### 9.3 System Test: QBER Abort Threshold

**Test ID:** `test_system::test_qber_abort`  
**Requirement:** Protocol must abort when QBER exceeds threshold.

#### Test Configuration

**Network:**
```yaml
links:
  - stack1: alice
    stack2: bob
    typ: depolarise
    cfg:
      fidelity: 0.80  # Very noisy: QBER ≈ 15%
      prob_success: 1.0
```

**Expected QBER:** $0.75 \times 0.20 = 0.15 = 15\%$

#### Postconditions

##### P1: Exception Raised
```python
from ehok.core.exceptions import QBERTooHighError
with pytest.raises(QBERTooHighError):
    run(network_config, programs)
```

OR

##### P1': Graceful Abort
```python
alice_result = results[0]["alice"]
assert alice_result["success"] == False
assert "QBER_EXCEEDED" in alice_result["abort_reason"]
```

##### P2: Log Messages
Verify log contains:
- "QBER = 0.15 exceeds threshold 0.11"
- "Protocol ABORT"

#### Acceptance Criterion

$$\text{PASS} \iff \text{protocol aborts when QBER} > 0.11$$

---

### 9.4 System Test: Commitment Ordering Security

**Test ID:** `test_system::test_commitment_ordering_security`  
**Requirement:** Protocol must enforce commitment-before-reveal ordering.

**Requirement Traceability:** Maps to Test 3 in `e-hok-baseline.md`

#### Test Configuration

Implement malicious Alice who attempts to violate protocol ordering:

**Malicious Alice Program:**
```python
class MaliciousAliceProgram(Program):
    def run(self, context):
        # Phase 1: Quantum generation
        outcomes, bases = generate_epr_pairs()
        
        # ATTACK: Send bases BEFORE receiving commitment
        context.csockets["bob"].send(bases.tobytes().hex())
        
        # Now try to receive commitment (should fail)
        try:
            commitment = yield from context.csockets["bob"].recv()
            # If we reach here, security property violated
            return {"status": "SECURITY_VIOLATED"}
        except TimeoutError:
            return {"status": "TIMEOUT"}
        except Exception as e:
            return {"status": "PROTOCOL_ERROR", "error": str(e)}
```

**Honest Bob Program:** Uses standard protocol

#### Expected Behaviors (any is acceptable)

##### Option 1: Timeout
Bob never sends commitment because he detects ordering violation.
$$alice\_result["status"] = \text{"TIMEOUT"}$$

##### Option 2: Protocol Error
Bob raises `SecurityException` when detecting early basis reveal.
$$bob\_result["status"] = \text{"PROTOCOL\_ERROR"}$$

##### Option 3: Commitment Invalidation
Bob proceeds but commitment verification fails.
$$\text{"COMMITMENT\_FAILED"} \in logs$$

#### Unacceptable Behavior

$$alice\_result["status"] \neq \text{"SECURITY\_VIOLATED"}$$

(Protocol must NOT allow completion if ordering violated)

#### Acceptance Criterion

$$\text{PASS} \iff \text{protocol prevents or detects ordering violation}$$

---

## 10. Statistical & Performance Tests

### 10.1 Statistical Test: Basis Independence

**Test ID:** `test_statistical::test_basis_independence`  
**Requirement:** Alice and Bob's basis choices must be independent.

#### Test Configuration

- Iterations: $n = 100$
- EPR pairs per iteration: $N = 1000$
- Total samples: $100 \times 1000 = 100,000$

#### Operation

```python
joint_counts = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}

for iteration in range(100):
    alice_bases, bob_bases = run_quantum_generation(N=1000)
    for i in range(1000):
        pair = (alice_bases[i], bob_bases[i])
        joint_counts[pair] += 1

# Normalize
total = sum(joint_counts.values())
joint_probs = {k: v/total for k, v in joint_counts.items()}
```

#### Expected Distribution (under independence)

$$P(A=i, B=j) = P(A=i) \times P(B=j) = 0.5 \times 0.5 = 0.25$$

For all $(i, j) \in \{0,1\}^2$

#### Chi-Square Test

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E)^2}{E}$$

Where:
- $O_{ij}$ = observed count for pair $(i,j)$
- $E = 100000 / 4 = 25000$ (expected count)

**Degrees of Freedom:** $df = 3$ (4 cells - 1)

**Critical Value:** $\chi^2_{0.95, 3} = 7.81$

#### Acceptance Criterion

$$\chi^2 < 7.81 \quad \text{(95% confidence)}$$

---

### 10.2 Performance Test: Throughput

**Test ID:** `test_performance::test_throughput`  
**Requirement:** Protocol must process target number of EPR pairs efficiently.

#### Test Configuration

- EPR pairs: $N = 10,000$
- Network: 2 nodes, depolarizing link (fidelity = 0.97)
- Measurement: Wall-clock time and simulation time

#### Operation

```python
import time
start_wall = time.time()
start_sim = ns.sim_time()

results = run(network_config, programs)

end_wall = time.time()
end_sim = ns.sim_time()

wall_time = end_wall - start_wall
sim_time = (end_sim - start_sim) / 1e9  # Convert ns to seconds
```

#### Performance Criteria

##### P1: Simulation Time
$$sim\_time < 30 \text{ seconds (simulated)}$$

##### P2: Wall-Clock Time
$$wall\_time < 120 \text{ seconds (real)}$$

##### P3: Throughput
$$\text{throughput} = \frac{N}{sim\_time} > 333 \text{ EPR pairs/sec (simulated)}$$

##### P4: Final Key Rate
$$\text{key\_rate} = \frac{final\_length}{sim\_time} > 10 \text{ bits/sec (simulated)}$$

#### Acceptance Criterion

$$\text{PASS} \iff \bigwedge_{i=1}^{4} P_i$$

---

### 10.3 Performance Test: Scalability

**Test ID:** `test_performance::test_scalability`  
**Requirement:** Runtime must scale linearly with number of EPR pairs.

#### Test Configuration

Run protocol with varying $N$:
- $N_1 = 1,000$
- $N_2 = 5,000$
- $N_3 = 10,000$
- $N_4 = 20,000$ (if memory permits)

#### Operation

```python
results = {}
for N in [1000, 5000, 10000, 20000]:
    t_start = ns.sim_time()
    run_protocol(N=N)
    t_end = ns.sim_time()
    results[N] = (t_end - t_start) / 1e9
```

#### Linear Scaling Test

Fit linear model: $t(N) = a + b \cdot N$

**Expected:** $R^2 > 0.95$ (strong linear correlation)

**Slope Interpretation:**
$$b \approx \text{time per EPR pair} < 10 \text{ ms (simulated)}$$

#### Acceptance Criterion

$$R^2 > 0.95 \land b < 0.01 \text{ (seconds per pair)}$$

---

### 10.4 Memory Test: Batching Effectiveness

**Test ID:** `test_performance::test_memory_batching`  
**Requirement:** Batching must enable processing beyond quantum memory limits.

#### Test Configuration

- Quantum memory: 5 qubits
- EPR pairs: $N = 10,000$ (far exceeds memory)
- Batch size: 5

#### Operation

```python
# Monitor memory allocation during execution
memory_tracker = MemoryTracker()
results = run(network_config, programs, memory_tracker=memory_tracker)

max_qubits_allocated = memory_tracker.get_max_allocation()
```

#### Postconditions

##### P1: Memory Constraint Respected
$$max\_qubits\_allocated \leq 5$$

##### P2: All Pairs Processed
$$results["alice"]["raw\_count"] = 10000$$

##### P3: Batches Executed
$$\text{\# batches} = \lceil 10000 / 5 \rceil = 2000$$

#### Acceptance Criterion

$$\text{PASS} \iff P_1 \land P_2 \land P_3$$

---

## Appendix A: Test Execution Matrix

| Phase | Test ID | Type | Duration | Dependencies | Priority |
|-------|---------|------|----------|--------------|----------|
| 0 | test_data_structures | Unit | < 1s | None | P0 |
| 0 | test_abstract_interfaces | Unit | < 1s | None | P0 |
| 0 | test_exception_hierarchy | Unit | < 1s | None | P0 |
| 0 | test_logging | Unit | < 1s | None | P0 |
| 1 | test_basis_randomness | Unit | 2s | Phase 0 | P1 |
| 1 | test_epr_generation_perfect | Integration | 5s | SquidASM | P1 |
| 1 | test_epr_generation_noisy | Integration | 10s | SquidASM | P1 |
| 1 | test_batching_manager | Unit | < 1s | Phase 0 | P1 |
| 2 | test_sha256_commit_verify | Unit | < 1s | Phase 0 | P1 |
| 2 | test_subset_opening | Unit | < 1s | Phase 0 | P1 |
| 3 | test_basis_matching | Unit | < 1s | Phase 0 | P1 |
| 3 | test_qber_estimation | Unit | < 1s | Phase 0 | P1 |
| 3 | test_test_set_selection | Unit | < 1s | Phase 0 | P1 |
| 4 | test_syndrome_computation | Unit | < 1s | Phase 0 | P1 |
| 4 | test_error_correction | Integration | 5s | scipy | P1 |
| 4 | test_leakage_estimation | Unit | < 1s | Phase 0 | P1 |
| 5 | test_toeplitz_construction | Unit | < 1s | Phase 0 | P1 |
| 5 | test_compression | Integration | 2s | Phase 0 | P1 |
| 5 | test_output_uniformity | Statistical | 30s | Phase 5 | P2 |
| 6 | test_phase_sequencing | Integration | 10s | Phases 1-5 | P1 |
| 6 | test_synchronization | Integration | 5s | Phase 1 | P1 |
| 7 | test_honest_execution_perfect | System | 30s | All phases | P0 |
| 7 | test_noise_tolerance | System | 60s | All phases | P0 |
| 7 | test_qber_abort | System | 20s | All phases | P1 |
| 7 | test_commitment_ordering_security | System | 15s | All phases | P0 |
| Stat | test_basis_independence | Statistical | 120s | Phase 1 | P2 |
| Perf | test_throughput | Performance | 120s | All phases | P2 |
| Perf | test_scalability | Performance | 300s | All phases | P2 |
| Perf | test_memory_batching | Performance | 60s | Phase 1 | P1 |

**Priority Levels:**
- **P0:** Critical (blocking release)
- **P1:** High (required for validation)
- **P2:** Medium (desired for quality assurance)

---

## Appendix B: Continuous Integration Pipeline

### B.1 Test Stages

```yaml
stages:
  - stage: unit_tests
    command: pytest tests/test_*.py -k "unit" -v
    timeout: 60s
    requirements: Phase 0 complete
    
  - stage: integration_tests
    command: pytest tests/test_*.py -k "integration" -v
    timeout: 300s
    requirements: Phase 0-5 complete
    
  - stage: system_tests
    command: pytest tests/test_integration.py -v
    timeout: 600s
    requirements: All phases complete
    
  - stage: performance_tests
    command: pytest tests/test_performance.py -v
    timeout: 900s
    requirements: System tests pass
```

### B.2 Coverage Requirements

```yaml
coverage_targets:
  overall: 85%
  critical_paths:
    ehok/core/: 95%
    ehok/implementations/: 90%
    ehok/protocols/: 95%
    ehok/quantum/: 90%
```

---

## Appendix C: Test Data & Fixtures

### C.1 Network Configurations

**File:** `tests/fixtures/network_perfect.yaml`
```yaml
stacks:
  - name: alice
    qdevice_typ: generic
    qdevice_cfg: {num_qubits: 5, T1: 1e9, T2: 1e9}
  - name: bob
    qdevice_typ: generic
    qdevice_cfg: {num_qubits: 5, T1: 1e9, T2: 1e9}
links:
  - stack1: alice
    stack2: bob
    typ: perfect
```

**File:** `tests/fixtures/network_noisy_5pct.yaml`
```yaml
links:
  - stack1: alice
    stack2: bob
    typ: depolarise
    cfg: {fidelity: 0.95, prob_success: 1.0}
```

### C.2 LDPC Matrices

Pre-generated matrices stored in `tests/fixtures/ldpc_matrices/`:
- `ldpc_100_rate05.npz` (for fast unit tests)
- `ldpc_1000_rate05.npz` (for integration tests)
- `ldpc_5000_rate05.npz` (for system tests)

### C.3 Pytest Fixtures

```python
# tests/conftest.py
import pytest
import numpy as np

@pytest.fixture
def random_seed():
    """Fixed seed for reproducible tests."""
    np.random.seed(42)

@pytest.fixture
def network_perfect():
    """Perfect link configuration."""
    from squidasm.run.stack.config import StackNetworkConfig
    return StackNetworkConfig.from_file("tests/fixtures/network_perfect.yaml")

@pytest.fixture
def network_noisy():
    """Noisy link configuration."""
    from squidasm.run.stack.config import StackNetworkConfig
    return StackNetworkConfig.from_file("tests/fixtures/network_noisy_5pct.yaml")
```

---

## Appendix D: Formal Verification Checklist

For each system test, verify:

- [ ] **Correctness:** Output matches mathematical specification
- [ ] **Completeness:** All phases executed without skip
- [ ] **Security:** Commitment-before-reveal enforced
- [ ] **Abort Safety:** Protocol aborts on QBER > threshold
- [ ] **Key Agreement:** Alice and Bob produce identical keys
- [ ] **Oblivious Property:** Bob's mask correctly identifies unknown bits
- [ ] **Performance:** Meets throughput and latency requirements
- [ ] **Determinism:** Same seed produces same results (where applicable)
- [ ] **Error Handling:** Exceptions propagate correctly
- [ ] **Logging:** All critical events logged without print()

---

**End of Formal Testing Specification**
