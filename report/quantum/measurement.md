# 4.4 Measurement & Basis Selection

## Basis Selection

### Random Basis Choice

```python
class BasisSelector:
    def choose(self) -> int:
        return np.random.randint(0, 2)  # 0=Z, 1=X
```

**Bases**:
- $Z = \{|0\rangle, |1\rangle\}$ (computational basis)
- $X = \{|+\rangle, |-\rangle\}$ where $|\pm\rangle = \frac{1}{\sqrt{2}}(|0\rangle \pm |1\rangle)$

### Security Requirement

Bases must be **uniformly random** to prevent basis-dependent attacks:
$$
P(b_i = 0) = P(b_i = 1) = 0.5
$$

## Measurement Execution

### NetQASM Measurement

```python
class MeasurementExecutor:
    def measure(self, qubit, basis: int) -> Generator:
        if basis == 1:  # X-basis
            yield from qubit.H()  # Hadamard gate
        
        outcome = yield from qubit.measure()
        return int(outcome)
```

### Measurement Buffer

```python
@dataclass
class MeasurementRecord:
    outcome: int
    basis: int
    round_id: int
    timestamp: float

class MeasurementBuffer:
    def record(self, outcome: int, basis: int, round_id: int) -> None:
        self._records.append(MeasurementRecord(outcome, basis, round_id, timestamp))
    
    def finalize(self) -> QuantumPhaseResult:
        return QuantumPhaseResult(
            measurement_outcomes=np.array([r.outcome for r in self._records]),
            basis_choices=np.array([r.basis for r in self._records]),
            # ...
        )
```

## Statistical Properties

**Expected Basis Match Rate**:
$$
P(b_A = b_B) = 0.5 \quad \text{(uniform random bases)}
$$

**Expected Sifted Key Length**:
$$
n_{\text{sifted}} = 0.5 \times n_{\text{raw}} \times (1 - P_{\text{loss}})
$$

## References

- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
- Caligo implementation: [`caligo/quantum/measurement.py`](../../caligo/caligo/quantum/measurement.py)
