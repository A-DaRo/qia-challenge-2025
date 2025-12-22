[← Return to Main Index](../index.md)

# 10. Parameter Space Exploration: Methodology

## Introduction

The Caligo protocol operates in a high-dimensional parameter space $(n, Q, r, R, \varepsilon)$ where protocol viability—defined as successful key extraction with positive rate—exhibits complex, non-convex structure. This chapter presents the mathematical framework for systematic exploration of this space, with particular focus on identifying the *Death Valley* boundary that separates viable from non-viable regimes.

---

## The Exploration Problem

### Parameter Space Definition

The protocol parameter space is:

$$
\Theta = \{(n, Q, r, R, \varepsilon) : n \in \mathbb{Z}^+, Q \in [0, 0.5], r \in [0, 1], R \in (0, 1), \varepsilon > 0\}
$$

**Physical constraints** reduce this to:
$$
\Theta_{\text{phys}} = \{(n, Q, r, R, \varepsilon) \in \Theta : Q \leq 0.22, \, C_{\mathcal{N}_r} \cdot \nu < 1/2\}
$$

### Objective Function

The primary objective is the *key rate*:

$$
\mathcal{R}(n, Q, r, R) = \frac{\ell}{n} = \frac{1}{n}\left\lfloor n \cdot h_{\min}(r) - \text{leak}_{EC} - 2\log_2(1/\varepsilon)\right\rfloor
$$

This is a *black-box* function: no closed-form expression exists when accounting for finite-size effects, reconciliation efficiency, and protocol overhead.

### Death Valley Definition

**Definition**. The *Death Valley* region $\mathcal{D} \subset \Theta$ is:
$$
\mathcal{D} = \{(n, Q, r, R) : \mathcal{R}(n, Q, r, R) = 0 \text{ or protocol aborts}\}
$$

The *Death Valley boundary* $\partial\mathcal{D}$ is the critical surface where key extraction transitions from viable to non-viable.

---

## Exploration Strategy

### Multi-Phase Approach

1. **Coverage phase**: Space-filling sampling to understand global structure
2. **Exploitation phase**: Targeted sampling near boundary regions
3. **Refinement phase**: High-resolution probing of critical transitions

### Computational Constraints

Each evaluation of $\mathcal{R}(\theta)$ requires:
- Full protocol execution ($\sim 100$ ms)
- Multiple trials for statistical confidence ($\sim 10-100$ trials)
- Total cost: $\sim 1-10$ seconds per parameter point

With $\sim 10^5$ parameter combinations of interest, exhaustive search is infeasible. Efficient sampling is essential.

---

## Latin Hypercube Sampling

### Theoretical Foundation

For a $d$-dimensional parameter space, Latin Hypercube Sampling (LHS) [1] partitions each dimension into $N$ equal intervals and ensures exactly one sample in each interval along each axis.

**Space-filling property**: LHS achieves $O(1/N)$ covering radius compared to $O(1/N^{1/d})$ for simple random sampling—exponentially better in high dimensions.

### Mathematical Formulation

Let $\pi_j : \{1, \ldots, N\} \to \{1, \ldots, N\}$ be random permutations for each dimension $j = 1, \ldots, d$.

The $i$-th sample is:
$$
x_i^{(j)} = \frac{\pi_j(i) - U_{ij}}{N}
$$

where $U_{ij} \sim \text{Uniform}(0, 1)$ are independent.

**Variance reduction**: For functions with moderate interaction effects, LHS reduces variance by factor $O(1/N)$ compared to simple random sampling [1].

### Implementation

For the Caligo parameter space:

| Dimension | Range | Scale |
|-----------|-------|-------|
| $n$ | $[512, 8192]$ | Log |
| $Q$ | $[0.01, 0.15]$ | Linear |
| $r$ | $[0.5, 0.99]$ | Linear |
| $R$ | $[0.3, 0.9]$ | Linear |

Log-scale for $n$ ensures adequate coverage across orders of magnitude.

---

## Gaussian Process Surrogates

### Motivation

After initial LHS exploration, a *surrogate model* approximates the objective function to guide subsequent sampling. Gaussian Processes (GPs) are ideal because they:

1. Provide uncertainty estimates (not just point predictions)
2. Naturally interpolate between observed points
3. Enable principled acquisition functions

### GP Regression Model

A GP defines a distribution over functions:
$$
f(\theta) \sim \mathcal{GP}(m(\theta), k(\theta, \theta'))
$$

where $m(\theta)$ is the mean function and $k(\theta, \theta')$ is the covariance kernel.

**Squared exponential kernel**:
$$
k(\theta, \theta') = \sigma_f^2 \exp\left(-\frac{\|\theta - \theta'\|^2}{2\ell^2}\right)
$$

**Posterior distribution**: Given observations $\mathcal{D} = \{(\theta_i, y_i)\}_{i=1}^n$:

$$
\mu(\theta^*) = k_*^T (K + \sigma_n^2 I)^{-1} y
$$
$$
\sigma^2(\theta^*) = k(\theta^*, \theta^*) - k_*^T (K + \sigma_n^2 I)^{-1} k_*
$$

where $k_* = [k(\theta^*, \theta_1), \ldots, k(\theta^*, \theta_n)]^T$ and $K_{ij} = k(\theta_i, \theta_j)$.

---

## Acquisition Functions

### Expected Improvement

The Expected Improvement (EI) acquisition function [2] balances exploration and exploitation:

$$
\text{EI}(\theta) = \mathbb{E}[\max(f(\theta) - f^*, 0)]
$$

where $f^* = \max_i f(\theta_i)$ is the current best observation.

**Closed form** (for GP posterior):
$$
\text{EI}(\theta) = (\mu(\theta) - f^* - \xi)\Phi(Z) + \sigma(\theta)\phi(Z)
$$

where $Z = \frac{\mu(\theta) - f^* - \xi}{\sigma(\theta)}$, $\Phi$ is the standard normal CDF, and $\phi$ is the PDF.

### Exploration Parameter

The parameter $\xi \geq 0$ controls exploration:
- $\xi = 0$: Pure exploitation (greedy)
- $\xi > 0$: Increased exploration of uncertain regions

For boundary detection, we use $\xi \approx 0.01$ to encourage probing near the transition.

### Probability of Improvement

Alternative acquisition for boundary detection:
$$
\text{PI}(\theta) = P(f(\theta) > f^* + \xi) = \Phi(Z)
$$

---

## Active Learning Loop

### Algorithm

```
Input: Initial samples D₀ from LHS, budget B
Initialize: GP model from D₀

for t = 1 to B:
    1. Fit GP to current data Dₜ₋₁
    2. θₜ = argmax_θ EI(θ; GP)
    3. yₜ = evaluate protocol at θₜ
    4. Dₜ = Dₜ₋₁ ∪ {(θₜ, yₜ)}
    
Output: DB, trained GP surrogate
```

### Boundary-Focused Variant

To specifically target the Death Valley boundary:

$$
\text{EI}_{\text{boundary}}(\theta) = \text{EI}(\theta) \cdot \mathbb{1}[|\mu(\theta)| < \delta]
$$

This focuses sampling on regions where the predicted key rate is near zero.

---

## Computational Considerations

### GP Scaling

Standard GP inference scales as $O(n^3)$ for $n$ observations. For large exploration campaigns:

- **Sparse GPs**: Inducing point methods reduce to $O(nm^2)$ for $m \ll n$ inducing points
- **Local GPs**: Fit separate models in parameter subregions
- **Neural network surrogates**: For very large datasets

### Parallelization

**Batch acquisition**: Select multiple points simultaneously using:
$$
\theta_{1:k} = \argmax_{\theta_1, \ldots, \theta_k} \sum_{i=1}^k \text{EI}(\theta_i | \theta_{1:i-1})
$$

with "fantasized" observations for pending evaluations.

---

## Validation of Surrogate Quality

### Cross-Validation Metrics

- **RMSE**: $\sqrt{\frac{1}{n}\sum_i (\mu(\theta_i) - y_i)^2}$ on held-out data
- **NLL**: $-\frac{1}{n}\sum_i \log p(y_i | \theta_i, \mathcal{D})$ tests calibration
- **Coverage**: Fraction of true values within predicted 95% interval

### Boundary Accuracy

For Death Valley boundary detection:
$$
\text{IoU} = \frac{|\hat{\mathcal{D}} \cap \mathcal{D}|}{|\hat{\mathcal{D}} \cup \mathcal{D}|}
$$

where $\hat{\mathcal{D}}$ is the predicted Death Valley region.

---

## References

[1] M. D. McKay, R. J. Beckman, and W. J. Conover, "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code," *Technometrics*, vol. 21, no. 2, pp. 239–245, 1979.

[2] D. R. Jones, M. Schonlau, and W. J. Welch, "Efficient Global Optimization of Expensive Black-Box Functions," *J. Glob. Optim.*, vol. 13, no. 4, pp. 455–492, 1998.

---

[← Return to Main Index](../index.md) | [Next: Active Learning Details](./active_learning.md)
