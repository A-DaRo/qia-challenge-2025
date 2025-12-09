[← Return to Main Index](../index.md)

# 2.3 Quantum Advantage

This section summarizes the industrial motivation for E-HOK and evaluates its quantum advantage—highlighting both benefits and drawbacks with mathematically grounded criteria.

## Industrial Application: High-Rate Secure MPC

Secure multiparty computation (SMC) pipelines (e.g., finance/health analytics) require millions of oblivious transfers (OTs). E-HOK pre-distributes **oblivious keys** at quantum key distribution (QKD) rates, then converts them into fast OTs on demand, reducing online latency for SMC back-ends [1, 2].

- **Resource Precomputation:** The OKD phase (steps 1–7 of $\pi_{QOT}$) is input-independent; keys can be buffered during off-peak windows, amortizing quantum channel use [1].
- **Integration Point:** The resulting oblivious keys $(k, \bar{k}, x)$ directly instantiate 1-out-of-2 OT one-time pads for circuit-gate evaluation (Kilian reduction) without public-key cryptography at run time [1, 2].

## Quantum (Dis)Advantage

*Advantage and disadvantage bullets are evaluated against classical public-key OT baselines and physical-constraint alternatives (noisy/bounded storage).* 

### Advantages

- **Throughput & Complexity:** Classical OT based on RSA/ECC exponentiation scales superlinearly (typical software OT extension scales near $O(n^2 \log n)$ in message length). In $\pi_{QOT}$, state preparation and measurement are linear in signals, $O(k(n+m))$, and post-processing is linear-time LDPC + Toeplitz hashing [1, Sec. 3.2; 2, Sec. 4]. This supports MHz-scale OKD rates observed in QKD hardware [2].
- **Symmetric-Crypto Assumption Only:** Security reduces to collision resistance of a hash-based commitment plus quantum uncertainty from basis mismatch; no trapdoor OWF or public-key setup is required [1, Sec. 2.1].
- **Forward-Looking Post-Quantum Posture:** Even against quantum computers, symmetric hashes remain conjecturally hard to invert (Grover gives only quadratic speed-up). Thus, compared to classical OT relying on RSA/ECC, E-HOK avoids known quantum breaks while keeping efficiency [1].

### Disadvantages / Limits

- **Computational (Not Unconditional) Security:** Hash-based commitments are only computationally binding; unconditional bit-commitment is impossible without extra physical constraints (bounded/noisy storage) [4; 5]. Therefore E-HOK’s security ultimately inherits the hash collision bound $\approx 2^{-256}$, not information-theoretic guarantees.
- **Channel/Device Imperfections:** Practical rates hinge on QBER staying below the LDPC operating region (e.g., $e \lesssim 11\%$ for rate-0.5–0.9 pools); excess noise collapses the advantage by increasing reconciliation leakage and abort probability [2, Sec. 5].
- **Quantum Infrastructure Cost:** Requires entanglement distribution plus authenticated classical channels; in low-volume settings the capital/operational cost may outweigh the speedup over purely classical OT.
- **Composability Caveat:** Without adopting noisy-storage or MDI variants, side-channel or detector attacks are out of scope; the present baseline is computationally secure but not universally composable against all physical attacks [4, 5].

### Takeaway

E-HOK offers a **performance-driven quantum advantage**: linear-time OT resource generation at QKD speeds under only symmetric-crypto assumptions. The trade-off is that security is **computational** (hash-based commitment) and **operationally contingent** on maintaining low QBER and trusted-device assumptions; unconditional guarantees would require the noisy-storage or MDI extensions left for future work.

---

## References

[1] Lemus, M., Ramos, M.F., Yadav, P., Silva, N.A., Muga, N.J., Souto, A., Paunković, N., Mateus, P., and Pinto, A.N. (2020). [Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation](../literature/Generation%20and%20Distribution%20of%20Quantum%20Oblivious%20Keys%20for%20Secure%20Multiparty%20Computation.md). *arXiv:1909.11701v2*.

[2] Lemus, M., Schiansky, P., Goulão, M., Bozzio, M., Elkouss, D., Paunković, N., Mateus, P., and Walther, P. (2025). [Performance of Practical Quantum Oblivious Key Distribution](../literature/Performance%20of%20Practical%20Quantum%20Oblivious%20Key%20Distribution.md). *arXiv:2501.03973*.

[3] Kilian, J. (1988). Founding Cryptography on Oblivious Transfer. *STOC '88*, 20-31. *(For MPC reduction)*.

[4] Wehner, S., Schaffner, C., and Terhal, B.M. (2008). [Cryptography from Noisy Storage](../literature/Cryptography%20from%20Noisy%20Storage.md). *Phys. Rev. Lett.* 100, 220502.

[5] König, R., Wehner, S., and Wullschleger, J. (2012). [Unconditional Security from Noisy Quantum Storage](../literature/Unconditional-security-from-noisy-quantum-storage.md). *IEEE Trans. Inf. Theory* 58(3), 1962-1984.

---

[← Return to Main Index](../index.md)
