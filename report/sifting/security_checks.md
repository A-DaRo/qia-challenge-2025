# 5.3 Security Verification

## Overview

Phase II concludes with security verification to ensure NSM constraints are satisfied before proceeding to reconciliation.

## Verification Checklist

### 1. QBER Threshold Check

**Invariant**: $Q \leq Q_{\text{threshold}}$ (default 11%)

```python
def verify_qber_threshold(qber: float, threshold: float = 0.11) -> None:
    if qber > threshold:
        raise QBERThresholdExceeded(
            f"QBER {qber:.4f} exceeds conservative threshold {threshold}"
        )
```

### 2. NSM Security Condition

**Invariant**: $Q_{\text{channel}} < Q_{\text{storage}} = \frac{1-r}{2}$

```python
def verify_nsm_security(qber: float, nsm_params: NSMParameters) -> None:
    q_storage = (1 - nsm_params.storage_noise_r) / 2
    if qber >= q_storage:
        raise SecurityError(
            f"NSM security violated: QBER {qber:.4f} >= "
            f"storage threshold {q_storage:.4f}"
        )
```

### 3. Sufficient Sifted Bits

**Invariant**: $n_{\text{sifted}} \geq n_{\text{min}}$ (minimum for reconciliation)

```python
def verify_sufficient_bits(n_sifted: int, n_min: int = 10_000) -> None:
    if n_sifted < n_min:
        raise InsufficientDataError(
            f"Sifted key length {n_sifted} < minimum {n_min}"
        )
```

### 4. Detection Efficiency Validation

**Invariant**: Observed detection rate consistent with expected $\eta$

```python
class DetectionValidator:
    def validate(
        self,
        num_detected: int,
        num_attempts: int,
        expected_eta: float,
    ) -> ValidationResult:
        observed_eta = num_detected / num_attempts
        
        # Hoeffding bound on detection efficiency
        epsilon = self._compute_bound(num_attempts)
        
        is_valid = abs(observed_eta - expected_eta) <= epsilon
        
        return ValidationResult(
            is_valid=is_valid,
            observed_eta=observed_eta,
            expected_eta=expected_eta,
            bound_epsilon=epsilon,
        )
```

## Abort Conditions

If any verification fails, protocol **aborts** with diagnostic information:

```python
class AbortReason(str, Enum):
    SIFTING_QBER_EXCEEDED = "PHASE_II_QBER_THRESHOLD_EXCEEDED"
    SIFTING_NSM_VIOLATED = "PHASE_II_NSM_SECURITY_VIOLATED"
    SIFTING_INSUFFICIENT_SIFTED = "PHASE_II_INSUFFICIENT_SIFTED_BITS"
    SIFTING_DETECTION_ANOMALY = "PHASE_II_DETECTION_EFFICIENCY_ANOMALY"
```

## Pre-Flight Feasibility Check

**Recommendation**: Run feasibility check *before* protocol execution to avoid wasted EPR generation.

```python
from caligo.security import FeasibilityChecker

checker = FeasibilityChecker()
result = checker.check(
    nsm_params=NSMParameters(...),
    expected_qber=compute_expected_qber(channel_params),
)

if not result.is_feasible:
    print(f"Infeasible: {result.failure_reason}")
    print(f"Recommendations: {result.recommendations}")
    sys.exit(1)
```

## References

- König, R., et al. (2012). "The operational meaning of min- and max-entropy." *IEEE Trans. Inf. Theory*, 58(3), 1962-1984.
- Caligo implementation: [`caligo/sifting/detection_validator.py`](../../caligo/caligo/sifting/detection_validator.py)
