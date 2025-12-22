[← Return to Main Index](../index.md)

# 10.3 Experimental Reproducibility and Statistical Methodology

## Introduction

Scientific validity of simulation results requires rigorous reproducibility guarantees. This section documents the statistical methodology for generating reproducible results and the validation protocols ensuring that reported performance metrics accurately reflect the underlying physical model.

---

## Reproducibility Framework

### Determinism Requirements

A simulation run is *reproducible* if identical inputs yield identical outputs. For quantum network simulation, this requires controlling:

1. **Quantum state preparation**: Seeded random number generators for Werner state mixing
2. **Measurement outcomes**: Seeded PRNG for simulated projective measurements
3. **Classical processing**: Deterministic LDPC construction and BP decoding
4. **Parameter sampling**: Seeded LHS and active learning acquisition

### Seed Hierarchy

The simulation employs a hierarchical seed structure:

$$
\text{seed}_{\text{trial}} = H(\text{seed}_{\text{master}}, \text{trial\_index})
$$

where $H$ is a cryptographic hash function (SHA-256 truncated to 64 bits).

**Benefits**:
- Master seed uniquely identifies experimental campaign
- Trial seeds are independent but reproducible
- Parallel execution yields identical results to sequential

### Random Number Generation

**Algorithm**: Mersenne Twister (MT19937) with 64-bit state.

**Independence verification**: For $n$ trials with seeds $s_1, \ldots, s_n$, verify:
$$
\text{Corr}(X_i, X_j) < \epsilon_{\text{indep}} \quad \forall i \neq j
$$

where $X_i$ is the PRNG output sequence from seed $s_i$.

---

## Statistical Analysis Protocol

### Point Estimates

For each protocol metric (key rate, QBER, FER), report the sample mean:

$$
\bar{X} = \frac{1}{N}\sum_{i=1}^N X_i
$$

**Unbiasedness**: $\mathbb{E}[\bar{X}] = \mu$ for i.i.d. samples.

### Confidence Intervals

**Normal approximation** (CLT-based, $N \geq 30$):
$$
\bar{X} \pm z_{\alpha/2} \frac{s}{\sqrt{N}}
$$

where $s$ is the sample standard deviation and $z_{\alpha/2} = 1.96$ for 95% confidence.

**Bootstrap intervals** (for non-normal distributions):
1. Resample $N$ observations with replacement, $B$ times
2. Compute statistic $\hat{\theta}_b$ for each bootstrap sample
3. Report $[\hat{\theta}_{(\alpha/2)}, \hat{\theta}_{(1-\alpha/2)}]$ percentiles

**Exact binomial intervals** (for proportions like FER):
$$
P_{\text{lo}} = \text{Beta}^{-1}\left(\frac{\alpha}{2}; k, n-k+1\right), \quad P_{\text{hi}} = \text{Beta}^{-1}\left(1-\frac{\alpha}{2}; k+1, n-k\right)
$$

where $k$ is the number of failures in $n$ trials.

### Sample Size Determination

For target precision $\delta$ with confidence $1-\alpha$:

$$
N = \left(\frac{z_{\alpha/2} \sigma}{\delta}\right)^2
$$

**Example**: To estimate QBER within $\pm 0.5\%$ with 95% confidence, assuming $\sigma \approx 0.03$:
$$
N = \left(\frac{1.96 \times 0.03}{0.005}\right)^2 \approx 139 \text{ trials}
$$

---

## Experimental Design

### Full Factorial Design

For $k$ parameters each at $L$ levels:
- **Design points**: $L^k$
- **Advantage**: Captures all interaction effects
- **Limitation**: Exponential growth

**Applicability**: Only for $k \leq 3$ parameters.

### Fractional Factorial Design

For larger $k$, use $2^{k-p}$ fractional designs:
- Alias structure determines confounded effects
- Resolution III: Main effects aliased with 2-factor interactions
- Resolution V: Main effects and 2-factor interactions estimable

**Caligo application**: For 5 parameters, use $2^{5-1}$ design (16 points) with generator $E = ABCD$.

### Response Surface Methodology

**Central Composite Design (CCD)**: Augments factorial design with:
- Center points (for curvature detection)
- Axial points (star design at distance $\alpha$ from center)

**Fitted model**:
$$
\hat{y} = \beta_0 + \sum_i \beta_i x_i + \sum_{i < j} \beta_{ij} x_i x_j + \sum_i \beta_{ii} x_i^2
$$

---

## Validation Protocols

### Ground Truth Comparison

Where analytical results exist, validate simulation against theory:

| Metric | Theory | Validation |
|--------|--------|------------|
| QBER | $(1-F)/2$ | $\|\hat{Q} - Q_{\text{theory}}\| < 3\sigma/\sqrt{n}$ |
| Sifting rate | $0.5$ | $\|\hat{R}_{\text{sift}} - 0.5\| < 0.01$ |
| Key rate bound | Lupo formula | $\hat{\ell} \leq \ell_{\text{Lupo}} + \epsilon$ |

### Sanity Checks

**Monotonicity**: Key rate should decrease with increasing QBER:
$$
Q_1 < Q_2 \implies \mathcal{R}(Q_1) \geq \mathcal{R}(Q_2)
$$

**Boundary conditions**:
- $\mathcal{R}(Q = 0, r = 1) > 0$ (perfect conditions yield positive rate)
- $\mathcal{R}(Q > 0.22) = 0$ (protocol aborts above threshold)

### Regression Testing

Maintain a suite of canonical test cases with known outputs:

1. **Deterministic LDPC**: Fixed $H$ matrix, verify syndrome computation
2. **Known QBER**: Inject controlled errors, verify estimation
3. **Key length**: For fixed $(n, Q, r)$, verify $\ell$ matches formula

**Failure criterion**: Any change in output beyond $\epsilon_{\text{reg}} = 10^{-10}$ triggers investigation.

---

## Data Management

### Output Schema

Each experimental run produces structured data:

```
{
  "metadata": {
    "seed": <master_seed>,
    "timestamp": <ISO8601>,
    "version": <code_version>
  },
  "parameters": {
    "n": <int>,
    "Q": <float>,
    "r": <float>,
    "R": <float>
  },
  "results": {
    "key_length": <int>,
    "qber_estimated": <float>,
    "fer": <float>,
    "trials": <int>
  }
}
```

### Persistence Strategy

**Atomic writes**: Each result is written atomically to prevent corruption.

**Append-only logs**: Historical data is never overwritten, enabling audit trails.

**Checkpointing**: Long-running sweeps checkpoint every $N$ evaluations.

---

## Computational Reproducibility

### Environment Specification

Reproducibility requires exact specification of:

1. **Software versions**: Python, NumPy, SciPy, NetSquid
2. **Hardware**: CPU model (for floating-point determinism)
3. **Configuration**: All non-default parameters

### Containerization

Docker images freeze the complete environment:

```dockerfile
FROM python:3.10-slim
RUN pip install squidasm==X.Y.Z netqasm==A.B.C
COPY caligo/ /app/caligo/
```

**Verification**: Identical image SHA256 across machines.

---

## Reporting Standards

### Minimum Viable Report

Every published result must include:

1. **Parameter values**: All $(n, Q, r, R, \varepsilon)$ settings
2. **Sample size**: Number of trials $N$
3. **Confidence level**: $1 - \alpha$ (typically 95%)
4. **Seed**: Master seed for reproducibility
5. **Code version**: Git commit hash or release tag

### Figure Generation

All figures are generated from raw data via deterministic scripts:

$$
\text{Raw Data} \xrightarrow{\text{script}} \text{Processed Data} \xrightarrow{\text{plot}} \text{Figure}
$$

**No manual intervention** between data and publication figure.

---

## References

[1] NIST/SEMATECH, "e-Handbook of Statistical Methods," 2012. [Online]. Available: https://www.itl.nist.gov/div898/handbook/

[2] D. C. Montgomery, *Design and Analysis of Experiments*, 8th ed., Wiley, 2012.

---

[← Return to Main Index](../index.md) | [Next: Simulation Results](../results/test_strategy.md)
