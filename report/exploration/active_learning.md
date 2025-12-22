[← Return to Main Index](../index.md)

# 10.2 Active Learning and Surrogate Models

## Introduction

This section provides detailed mathematical treatment of the surrogate modeling and active learning components used for parameter space exploration. The goal is to efficiently identify the viable operating region of the Caligo protocol with minimal simulation budget.

---

## Gaussian Process Details

### Kernel Selection

The choice of covariance kernel encodes prior beliefs about the objective function.

**Matérn 5/2 kernel** (preferred for physical systems):
$$
k_{\text{M52}}(r) = \sigma_f^2 \left(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2}\right)\exp\left(-\frac{\sqrt{5}r}{\ell}\right)
$$

where $r = \|\theta - \theta'\|$.

**Properties**:
- Twice differentiable (appropriate for smooth key rate function)
- Finite correlation length (unlike squared exponential)
- Better extrapolation near boundaries

**Automatic Relevance Determination (ARD)**:
$$
k_{\text{ARD}}(\theta, \theta') = \sigma_f^2 \exp\left(-\frac{1}{2}\sum_{d=1}^D \frac{(\theta_d - \theta'_d)^2}{\ell_d^2}\right)
$$

Separate length scales $\ell_d$ per dimension capture anisotropic behavior.

### Hyperparameter Optimization

Hyperparameters $\phi = (\sigma_f, \{\ell_d\}, \sigma_n)$ are optimized by maximizing log marginal likelihood:

$$
\log p(y | X, \phi) = -\frac{1}{2}y^T K_y^{-1} y - \frac{1}{2}\log|K_y| - \frac{n}{2}\log 2\pi
$$

where $K_y = K + \sigma_n^2 I$.

**Gradient-based optimization**: L-BFGS-B with multiple random restarts to avoid local optima.

---

## Acquisition Function Theory

### Expected Improvement Derivation

For a GP with posterior $f(\theta) \sim \mathcal{N}(\mu(\theta), \sigma^2(\theta))$, the expected improvement over current best $f^*$:

$$
\text{EI}(\theta) = \int_{f^*}^{\infty} (f - f^*) \cdot p(f | \theta, \mathcal{D}) \, df
$$

**Derivation**:
$$
\text{EI}(\theta) = \int_{f^*}^{\infty} (f - f^*) \cdot \frac{1}{\sigma(\theta)\sqrt{2\pi}} \exp\left(-\frac{(f - \mu(\theta))^2}{2\sigma^2(\theta)}\right) df
$$

Substituting $z = (f - \mu(\theta))/\sigma(\theta)$:
$$
\text{EI}(\theta) = \sigma(\theta) \int_{z^*}^{\infty} (\mu(\theta) + \sigma(\theta)z - f^*) \phi(z) \, dz
$$

where $z^* = (f^* - \mu(\theta))/\sigma(\theta)$.

**Closed form**:
$$
\text{EI}(\theta) = (\mu(\theta) - f^*)\Phi(-z^*) + \sigma(\theta)\phi(z^*)
$$

### Lower Confidence Bound

Alternative acquisition for exploration:
$$
\text{LCB}(\theta) = \mu(\theta) - \kappa \sigma(\theta)
$$

**Regret bound** (Srinivas et al., 2010): With $\kappa_t = \sqrt{2\log(t^{d/2+2}\pi^2/3\delta)}$, the cumulative regret is $O^*(\sqrt{T\gamma_T})$ where $\gamma_T$ is the information gain.

### Knowledge Gradient

For noisy observations, the Knowledge Gradient [3] accounts for expected improvement in the posterior:

$$
\text{KG}(\theta) = \mathbb{E}\left[\max_{\theta'} \mu^{(n+1)}(\theta') - \max_{\theta'} \mu^{(n)}(\theta') \,\bigg|\, \text{sample at } \theta\right]
$$

Computationally intensive but provides better performance with noisy evaluations.

---

## Multi-Objective Formulation

### Pareto Optimization

The Caligo exploration actually involves multiple objectives:

1. **Key rate** $\mathcal{R}$: Maximize
2. **Reconciliation efficiency** $f_{\text{eff}}$: Minimize (closer to Shannon limit)
3. **Protocol latency** $T_{\text{total}}$: Minimize

**Pareto dominance**: $\theta_1$ dominates $\theta_2$ ($\theta_1 \succ \theta_2$) if $\theta_1$ is at least as good in all objectives and strictly better in at least one.

**Pareto front**: The set of non-dominated solutions $\mathcal{P}^* = \{\theta : \nexists \theta' \text{ s.t. } \theta' \succ \theta\}$.

### Expected Hypervolume Improvement

For multi-objective optimization, use Expected Hypervolume Improvement (EHVI):

$$
\text{EHVI}(\theta) = \mathbb{E}[\text{HV}(\mathcal{P} \cup \{f(\theta)\}) - \text{HV}(\mathcal{P})]
$$

where $\text{HV}(\mathcal{P})$ is the hypervolume dominated by the Pareto front relative to a reference point.

---

## Boundary Detection Strategy

### Level Set Estimation

The Death Valley boundary corresponds to the level set $\{\theta : f(\theta) = 0\}$.

**Straddle heuristic**: Sample where uncertainty about threshold crossing is highest:
$$
a_{\text{straddle}}(\theta) = -|f^*|\cdot\Phi\left(\frac{\mu(\theta)}{\sigma(\theta)}\right) - |f^*|\cdot\Phi\left(\frac{-\mu(\theta)}{\sigma(\theta)}\right)
$$

### Entropy-Based Acquisition

Information-theoretic approach: maximize information about the level set boundary.

**Entropy Search** [4]:
$$
a_{\text{ES}}(\theta) = H[p(\theta^* | \mathcal{D})] - \mathbb{E}_{y}[H[p(\theta^* | \mathcal{D} \cup \{(\theta, y)\})]]
$$

where $\theta^*$ is the location of the boundary.

---

## Surrogate Training Protocol

### Data Normalization

Before GP fitting, normalize inputs and outputs:

**Input normalization**: Scale each dimension to $[0, 1]$:
$$
\tilde{\theta}_d = \frac{\theta_d - \theta_d^{\min}}{\theta_d^{\max} - \theta_d^{\min}}
$$

**Output normalization**: Standardize:
$$
\tilde{y} = \frac{y - \bar{y}}{s_y}
$$

where $\bar{y}$ and $s_y$ are sample mean and standard deviation.

### Training Algorithm

```
Input: Observations D = {(θᵢ, yᵢ)}ᵢ₌₁ⁿ
Output: Trained GP model

1. Normalize inputs θ̃ᵢ and outputs ỹᵢ
2. Initialize hyperparameters φ₀ (length scales, noise variance)
3. for restart = 1 to N_restarts:
   a. Perturb initial φ₀
   b. Optimize φ by maximizing log marginal likelihood
   c. Record solution if best so far
4. Compute posterior mean μ(·) and variance σ²(·)
5. Return GP model
```

### Model Selection

**Nested cross-validation** for kernel selection:
1. Outer loop: Hold out 20% data for testing
2. Inner loop: 5-fold CV for hyperparameter tuning
3. Select kernel family with best outer-loop NLL

---

## Convergence Criteria

### Stopping Conditions

The active learning loop terminates when:

1. **Budget exhausted**: $t = T_{\max}$
2. **Negligible improvement**: $\max_\theta \text{EI}(\theta) < \epsilon_{\text{EI}}$
3. **Boundary converged**: Surrogate boundary stable for $k$ iterations

### Convergence Diagnostics

**Boundary stability**: Track changes in estimated Death Valley boundary:
$$
\Delta_t = \int_\Theta |\mathbb{1}[\mu_t(\theta) > 0] - \mathbb{1}[\mu_{t-1}(\theta) > 0]| \, d\theta
$$

Convergence when $\Delta_t < \delta_{\text{boundary}}$ for consecutive iterations.

**Posterior contraction**: Monitor average posterior variance:
$$
\bar{\sigma}^2_t = \frac{1}{|\Theta_{\text{test}}|}\sum_{\theta \in \Theta_{\text{test}}} \sigma_t^2(\theta)
$$

Expect $\bar{\sigma}^2_t \to 0$ as $t \to \infty$ in explored regions.

---

## Theoretical Guarantees

### Regret Bounds

For GP-UCB with Matérn kernel (Srinivas et al., 2010):

**Cumulative regret**:
$$
R_T = \sum_{t=1}^T (f(\theta^*) - f(\theta_t)) = O^*\left(T^{\frac{d+1}{d+2}}\right)
$$

This is sublinear: the average regret per step vanishes.

### Sample Complexity for Level Set

For level set estimation with GP-based methods (Gotovos et al., 2013):

Under smoothness assumptions, $O((\log(1/\delta)/\epsilon^2)^d)$ samples suffice to identify the level set with error $\epsilon$ and confidence $1-\delta$.

---

## References

[1] C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006.

[2] N. Srinivas, A. Krause, S. Kakade, and M. Seeger, "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design," *Proc. ICML*, 2010.

[3] P. I. Frazier, W. B. Powell, and S. Dayanik, "A Knowledge-Gradient Policy for Sequential Information Collection," *SIAM J. Control Optim.*, vol. 47, no. 5, pp. 2410–2439, 2008.

[4] P. Hennig and C. J. Schuler, "Entropy Search for Information-Efficient Global Optimization," *J. Mach. Learn. Res.*, vol. 13, pp. 1809–1837, 2012.

---

[← Return to Main Index](../index.md) | [Next: Harness and Reproducibility](./harness_and_usage.md)
