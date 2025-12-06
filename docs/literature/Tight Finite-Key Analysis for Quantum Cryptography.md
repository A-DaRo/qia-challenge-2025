**Tight Finite-Key Analysis for Quantum Cryptography**
Marco Tomamichel, 1,* Charles Ci Wen Lim, 2,† Nicolas Gisin, 2 and Renato Renner¹
¹Institute for Theoretical Physics, ETH Zurich, 8093 Zurich, Switzerland
² Group of Applied Physics, University of Geneva, 1211 Geneva, Switzerland
Despite enormous theoretical and experimental progress in quantum cryptography, the
security of most current implementations of quantum key distribution is still not rigorously
established. One significant problem is that the security of the final key strongly depends on
the number, $M$, of signals exchanged between the legitimate parties—yet existing security
proofs are often only valid asymptotically, for unrealistically large values of $M$. Another
challenge is that most security proofs are very sensitive to small differences between the
physical devices used by the protocol and the theoretical model used to describe them. Here,
we show that these gaps between theory and experiment can be simultaneously overcome
by using a recently developed proof technique based on the uncertainty relation for smooth
entropies.

Quantum Key Distribution (QKD), invented by Bennett and Brassard [1] and by Ekert [2], can
be considered the first application of quantum information science, and commercial products have
already become available. Accordingly, QKD has been an object of intensive study over the past
few years. On the theory side, the security of several variants of QKD protocols against general
attacks has been proved [3–8]. At the same time, experimental techniques have reached a state of
development that enables key distribution at MHz rates over distances of 100 km [9–11].

Despite these developments, there is still a large gap between theory and practice, in the sense
that the security claims are based on assumptions that are not (or cannot be) met by experimental
implementations. For example, the proofs often rely on theoretical models of the devices (such
as photon sources and detectors) that do not take into account experimentally unavoidable im-
perfections (see [12] for a discussion). In this work, we consider prepare-and-measure quantum
key distribution protocols, like the original Bennett-Brassard 1984 (BB84) protocol [1]. Here, one
party prepares quantum systems (e.g. the polarization degrees of freedom of photons) and sends
them through an insecure quantum channel to another party who then measures the systems. In
order to analyze the security of such protocols, the physical devices used by both parties to prepare
and measure quantum systems are replaced by theoretical device models. The goal, from a theory
perspective, is to make these theoretical models as general as possible so that they can accom-
modate imperfect physical devices independently of their actual implementation. (This approach,
in the context of entanglement-based protocols, also led to the development of device-independent
quantum cryptography—see [13, 14] for recent results.)

Another weakness of many security proofs is the **asymptotic resource assumption**, i.e., the as-
sumption that an arbitrarily large number $M$ of signals can be exchanged between the legitimate
parties and used for the computation of the final key. This assumption is quite common in the lit-
erature, and security proofs are usually only valid asymptotically as $M$ tends to infinity. However,
the asymptotic resource assumption cannot be met by practical realizations—in fact, the key is
often computed from a relatively small number of signals ($M \ll 10^6$). This problem has recently
received increased attention and explicit bounds on the number of signals required to guarantee
security have been derived [15–21].

In this work, we apply a novel proof technique [22] that allows us to overcome the above diffi-
culties. In particular, we derive almost tight bounds on the minimum value $M$ required to achieve
a given level of security. The technique is based on an entropic formulation of the uncertainty re-
lation [23] or, more precisely, its generalization to smooth entropies [22]. Compared to preexisting
methods, our technique is rather direct. It therefore avoids various estimates, including the de
Finetti Theorem [24] and the Post-Selection technique [25], that have previously led to too pes-
simistic bounds. Roughly speaking, our result is a lower bound on the achievable key rate which
deviates from the asymptotic result (where $M$ is infinitely large) only by terms that are caused
by (probably unavoidable) statistical fluctuations in the parameter estimation step. Moreover, we
believe that the theoretical device model used for our security analysis is as general as possible for
protocols of the prepare-and-measure type.

# RESULTS

## Security Definitions

We follow the discussion of composable security [26] and first take an abstract view on QKD
protocols. A QKD protocol describes the interaction between two players, Alice and Bob. Both
players can generate fresh randomness and have access to an insecure quantum channel as well
as an authenticated (but otherwise insecure) classical channel. (Note that, using an authentica-
tion protocol, any insecure channel can be turned into an authentic channel. The authentication
protocol will however use some key material, as discussed in [27].)

The QKD protocol outputs a key, $S$, on Alice’s side and an estimate of that key, $\hat{S}$, on Bob’s
side. This key is usually an $\ell$-bit string, where $\ell$, depends on the noise level of the channel, as
well as the security and correctness requirements on the protocol. The protocol may also abort, in
which case we set $S = \hat{S} = \bot$.

In the following, we define what it means for a QKD protocol to be **secure**. Roughly speaking,
the protocol has to (approximately) satisfy two criteria, called **correctness** and **secrecy**. These
criteria are conditions on the probability distribution of the protocol output, $S$ and $\hat{S}$, as well as
the information leaked to an adversary, $E$. These depend, in general, on the attack strategy of the
adversary, who is assumed to have full control over the quantum channel connecting Alice and
Bob, and has access to all messages sent over the authenticated classical channel.

A QKD protocol is called **correct** if, for any strategy of the adversary, $\hat{S} = S$. It is called $\varepsilon_{\text{cor}}$-
**correct** if it is $\varepsilon_{\text{cor}}$-indistinguishable from a correct protocol. In particular, a protocol is $\varepsilon_{\text{cor}}$-correct
if $\text{Pr}[S \neq \hat{S}] \leq \varepsilon_{\text{cor}}$.

In order to define the secrecy of a key, we consider the quantum state $\rho_{SE}$ that describes the
correlation between Alice’s classical key $S$ and the eavesdropper, $E$ (for any given attack strategy).
A key is called $\Delta$-**secret** from $E$ if it is $\Delta$-close to a uniformly distributed key that is uncorrelated
with the eavesdropper, i.e. if

$$
\frac{1}{2} \|\rho_{SE} - \omega_S \otimes \rho_E\|_1 \leq \Delta, \quad (1)
$$

where $\omega_S$ denotes the fully mixed state on $S$ and $\rho_E$ is the marginal state on the system $E$. For a
motivation and discussion of this particular secrecy criterion (in particular the choice of the norm)
we refer to [28].

A QKD protocol is called **secret** if, for any attack strategy, $\Delta = 0$ whenever the protocol outputs
a key. It is called $\varepsilon_{\text{sec}}$-**secret** if it is $\varepsilon_{\text{sec}}$-indistinguishable from a secret protocol. In particular, a
protocol is $\varepsilon_{\text{sec}}$-secret if it outputs $\Delta$-secure keys with $(1 - P_{\text{abort}}) \Delta < \varepsilon_{\text{sec}}$, where $P_{\text{abort}}$ is the
probability that the protocol aborts. (To see that this suffices to ensure $\varepsilon_{\text{sec}}$-indistinguishability,
note that the secrecy condition is trivially fulfilled if the protocol aborts.)

In some applications it is reasonable to consider correctness and secrecy of protocols separately,
since there may be different requirements on the correctness of the key (i.e., that Bob’s key agrees
with Alice’s, implying that messages encrypted by Alice are correctly decrypted by Bob) and
secrecy. In fact, in many realistic applications, an incorrect decoding of the transmitted data
would be detected so that the data can be resent. For such applications, $\varepsilon_{\text{cor}}$ may be chosen larger
than $\varepsilon_{\text{sec}}$.

However, secrecy of the protocol alone as defined above does not ensure that Bob’s key is secret
from the eavesdropper as well. One is thus often only interested in the **overall security** of the
protocol (which automatically implies secrecy of Bob’s key).

A QKD protocol is called **secure** if it is correct and secret. It is called $\varepsilon$-**secure** if it is $\varepsilon$-
indistinguishable from a secure protocol. In particular, a protocol is $\varepsilon$-secure if it is $\varepsilon_{\text{cor}}$-correct
and $\varepsilon_{\text{sec}}$-secret with $\varepsilon_{\text{cor}} + \varepsilon_{\text{sec}} \leq \varepsilon$.

Finally, the **robustness**, $\varepsilon_{\text{rob}}$, is the probability that the protocol aborts even though the eaves-
dropper is inactive. (More precisely, one assumes a certain channel model which corresponds to
the characteristics of the channel in the absence of an adversary. For protocols based on qubits,
the standard channel model used in the literature is the depolarizing channel. We also chose this
channel model for our analysis in the discussion section, thus enabling a comparison to the existing
results.) Note that a trivial protocol that always aborts is secure according to the above definitions,
and a robustness requirement is therefore necessary. In this work, we include the robustness $\varepsilon_{\text{rob}}$
in our estimate for the expected key rate (when the eavesdropper is inactive) and then optimize
over the protocol parameters to maximize this rate.

## Device Model

Recall that Alice and Bob are connected by an insecure quantum channel. On one side of
this channel, Alice controls a device allowing her to prepare quantum states in two bases, $X$ and
$Z$. In an optimal scenario, the prepared states are qubits and the two bases are diagonal, e.g.
$\mathcal{X} = \{|0\rangle, |1\rangle\}$ and $\mathcal{Z} = \{|+\rangle, |-\rangle\}$ with $|\pm\rangle := (|0\rangle \pm |1\rangle)/\sqrt{2}$. More generally, we characterize the
quality of a source by its **preparation quality**, $q$. The preparation quality—as we will see in the
following—is the only device parameter relevant for our security analysis. It achieves its maximum
of $q = 1$ if the prepared states are qubits and the bases are diagonal, as in the example above.
In the following, we discuss two possible deviations from a perfect source and how they can be
characterized in terms of $q$.

Firstly, if the prepared states are guaranteed to be qubits, we characterize the quality of Alice’s
device by the maximum fidelity it allows between states prepared in the $X$ basis and states prepared
in the $Z$ basis. Namely, we have $q = -\log \max |\langle \psi_X | \psi_Z \rangle|^2$, where the maximization is over all states
$|\psi_X\rangle$ and $|\psi_Z\rangle$ prepared in the $X$ and $Z$ basis, respectively. (In this work, $\log$ denotes the binary
logarithm.) The maximum $q = 1$ is achieved if the basis states are prepared in diagonal bases, as
is the case in the BB84 protocol.

In typical optical schemes, qubits are realized by polarization states of single photons. An ideal
implementation therefore requires a single-photon source in Alice’s laboratory. In order to take
into account sources that emit weak coherent light pulses instead, the analysis presented in this
paper can be extended using photon tagging [29] and decoy states [30]. This approach—although
beyond the scope of the present article—can be incorporated into our finite-key analysis. (See
also [31–33] for recent results on the finite-key analysis of such protocols.)

Secondly, consider a source that prepares states in the following way: The source produces two
entangled particles and then sends out one of them while the other is measured in one of two
bases. The choice of basis for the measurement decides whether the states are prepared in the $X$
or $Z$ basis. Together with the measurement outcome, which is required to be uniformly random
for use in our protocol, this determines which of the four states is prepared. For such a source,
the preparation quality is given by $q = -\log \max \|\sqrt{M_x} \sqrt{N_z}\|_\infty^2$, where $\{M_x\}_x$ and $\{N_z\}_z$ are the
elements of the positive operator valued measurements (POVMs) that are used to prepare the
state in the $X$ and the $Z$ basis, respectively. If the produced state is that of two fully entangled
qubits and the measurements are projective measurements in diagonal bases, we recover BB84 and
$q = 1$ [34]. Sources of this type have recently received increased attention since they can be used
as heralded single photon sources [35, 36] and have applications in (device independent) quantum
cryptography [37–39].

On the other side of the channel, Bob controls a device allowing him to measure quantum
systems in two bases corresponding to $X$ and $Z$. We will derive security bounds that are valid
independently of the actual implementation of this device as long as the following condition is
satisfied: we require that the probability that a signal is detected in Bob’s device is independent
of the basis choices ($X$ or $Z$) by Alice and Bob. Note that this assumption is necessary. In fact,
if it is not satisfied (which is the case for some implementations) a loophole arises that can be
used to eavesdrop on the key without being detected [40]. (Remarkably, this assumption can be
enforced device-independently: Bob simply substitutes a random bit whenever his device fails to
detect Alice’s signal. If this is done, however, the expected error rate may increase significantly.)

Finally, we assume that it is theoretically possible to devise an apparatus for Bob which delays
all the measurements in the $X$-basis until after parameter estimation, but produces the exact same
measurement statistics as the actual device he uses. This assumption is satisfied if Bob’s actual
measurement device is memoryless. (To see this, note that we could (in theorey) equip such a device
with perfect quantum memory that stores the received state until after the parameter estimation
has been done.) The assumption is already satisfied if the measurement statistics are unaffected
when the memory of the actual device is reset after each measurement. It is an open question
whether this assumption can be further relaxed.

## Protocol Definition

We now define a family of protocols, $\Phi[n, k, \ell, Q_{\text{tol}}, \varepsilon_{\text{cor}}, \text{leak}_{\text{EC}}]$, which is parametrized by the
**block size**, $n$, the number of bits used for **parameter estimation**, $k$, the **secret key length**, $\ell$, the
**channel error tolerance**, $Q_{\text{tol}}$, the required **correctness**, $\varepsilon_{\text{cor}}$, and the **error correction leakage**, $\text{leak}_{\text{EC}}$.
The protocol is asymmetric, so that the number of bits measured in the two bases ($n$ bits in the $X$
basis and $k$ bits in the $Z$ basis) are not necessarily equal [41].

These protocols are described in Table I.

## Security Analysis

The following two theorems constitute the main technical result of our paper, stating that the
protocols described above are both $\varepsilon_{\text{cor}}$-correct and $\varepsilon_{\text{sec}}$-secure if the secret key length is chosen
appropriately. Correctness is guaranteed by the error correction step of the protocol, where a hash
of Alice’s raw key is compared with the hash of its estimate on Bob’s side. The following holds:

**Theorem 1. The protocol $\Phi[n, k, \ell, Q_{\text{tol}}, \varepsilon_{\text{cor}}, \text{leak}_{\text{EC}}]$ is $\varepsilon_{\text{cor}}$-correct.**

The protocols are $\varepsilon_{\text{sec}}$-secure if the length of the extracted secret key does not exceed a certain
length. Asymptotically for large block sizes $n$, the reductions of the key length due to finite statistics
and security parameters can be neglected, and a secret key of length $\ell_{\text{max}} = n(q - h(Q_{\text{tol}})) - \text{leak}_{\text{EC}}$
can be extracted securely. Here, $h$ denotes the binary entropy function. Since our statistical sample
is finite, we have to add to the tolerated channel noise a term $\mu \approx \sqrt{1/k} \cdot \ln(1/\varepsilon_{\text{sec}})$ that accounts

**TABLE I. Protocol Definition.**

**State Preparation:** The first four steps of the protocol are repeated for $i = 1, 2, \ldots, M$ until the condition
in the Sifting step is met.
Alice chooses a basis $a_i \in \{X, Z\}$, where $X$ is chosen with probability $p_x = (1 + \sqrt{k/n})^{-1}$ and $Z$
with probability $p_z = 1 - p_x$. (These probabilities are chosen in order to minimize the number $M$
of exchanged particles before Alice and Bob agree on the basis $X$ for $n$ particles and on the basis $Z$
for $k$ particles.) Next, Alice chooses a uniformly random bit $y_i \in \{0, 1\}$ and prepares the qubit in a
state of basis $a_i$, given by $y_i$. Alternatively, if the source is entanglement-based, Alice will ask it to
prepare a state in the basis $a_i$ and record the output in $Y_i$.
**Distribution:** Alice sends the qubit over the quantum channel to Bob. (Recall that Eve is allowed to
arbitrarily interact with the system and we do not make any assumptions about what Bob receives.)
**Measurement:** Bob also chooses a basis, $b_i \in \{X, Z\}$, with probabilities $p_x$ and $p_z$, respectively. He
measures the system received from Alice in the chosen basis and stores the outcome in $y'_i \in \{0, 1, \emptyset\}$,
where ‘$\emptyset$’ is the symbol produced when no signal is detected.
**Sifting:** Alice and Bob broadcast their basis choices over the classical channel. We define the sets $\mathcal{X} :=$
$\{i : a_i = b_i = X \land y'_i \neq \emptyset\}$ and $\mathcal{Z} := \{i : a_i = b_i = Z \land y'_i \neq \emptyset\}$. The protocol repeats the first steps as
long as either $|\mathcal{X}| < n$ or $|\mathcal{Z}| < k$.
**Parameter Estimation:** Alice and Bob choose a random subset of size $n$ of $\mathcal{X}$ and store the respective
bits, $y_i$ and $y'_i$, into raw key strings $X$ and $X'$, respectively.
Next, they compute the average error $\lambda := \frac{1}{|\mathcal{Z}|} \sum_{i \in \mathcal{Z}} y_i \oplus y'_i$, where the sum is over all $i \in \mathcal{Z}$. The
protocol aborts if $\lambda > Q_{\text{tol}}$.
**Error Correction:** An information reconciliation scheme that broadcasts at most $\text{leak}_{\text{EC}}$ bits of classical
error correction data is applied. This allows Bob to compute an estimate, $\hat{X}$, of $X$.
Then, Alice computes a bit string (a hash) of length $\lceil \log(1/\varepsilon_{\text{cor}}) \rceil$ by applying a random universal₂
hash function [42] to $X$. She sends the choice of function and the hash to Bob. If the hash of $X$
disagrees with the hash of $\hat{X}$, the protocol aborts.
**Privacy Amplification:** Alice extracts $\ell$ bits of secret key $S$ from $X$ using a random universal₂ hash
function [43, 44]. (Instead of choosing a universal₂ hash function, which requires at least $n$ bits of
random seed, one could instead employ almost two-universal₂ hash functions [45] or constructions
based on Trevisan’s extractor [46]. These techniques allow for a reduction in the random seed length
while the security claims remain almost unchanged.) The choice of function is communicated to Bob,
who uses it to calculate $\hat{S}$.

for statistical fluctuations. Furthermore, the security parameters lead to a small reduction of the
key rate logarithmic in $\varepsilon_{\text{cor}}$ and $\varepsilon_{\text{sec}}$. The following theorem holds:

**Theorem 2. The protocol $\Phi[n, k, \ell, Q_{\text{tol}}, \varepsilon_{\text{cor}}, \text{leak}_{\text{EC}}]$ using a source with preparation quality $q$ is $\varepsilon_{\text{sec}}$-secret if the
secret key length $\ell$ satisfies**

$$
\ell \leq n(q - h(Q_{\text{tol}} + \mu)) - \text{leak}_{\text{EC}} - \log \frac{2}{\varepsilon_{\text{sec}} \varepsilon_{\text{cor}}}, \quad \text{where} \quad \mu := \sqrt{\frac{n + k}{n k} \frac{k + 1}{k}} \ln \frac{4}{\varepsilon_{\text{sec}}}. \quad (2)
$$

A sketch of the proof of these two statements follows in the methods section and a rigorous proof
of slightly more general versions of the theorems presented above can be found in Supplementary
Material 1.

# DISCUSSION

In this section, we discuss the asymptotic behavior of our security bounds and compare numer-
ical bounds on the key rate for a finite number of exchanged signals with previous results. For this
purpose, we assume that the quantum channel in the absence of an eavesdropper can be described
as a **depolarizing channel** with **quantum bit error rate** $Q$. (Note that this assumption is not needed
for the security analysis of the previous section.) The numerical results are computed for a perfect
single-photon source, i.e. $q = 1$. Furthermore, finite detection efficiencies and channel losses are
not factored into the key rates, i.e. the expected secret key rate calculated here can be understood
as the **expected key length per detected signal**.

The efficiency of a protocol $\Phi$ is characterized in terms of its **expected secret key rate**,

$$
r(\Phi, Q) := (1 - \varepsilon_{\text{rob}}) \frac{\ell}{M(n, k)}, \quad (3)
$$

where $M(n, k)$ is the expected number of qubits that need to be exchanged until $n$ raw key bits
and $k$ bits for parameter estimation are gathered (see protocol description).

[IMAGE: Fig. 1. Expected Key Rate as Function of the Block Size. Plot of expected key rate $r$ as a function of the block size $n$ for channel bit error rates $Q \in \{1\%, 2.5\%, 5\%\}$ (from left to right). The security rate is fixed to $\varepsilon/\ell = 10^{-14}$. The plot shows Expected Secret Key Rate, $r$, on the vertical axis, ranging from $10^{-3}$ to $10^0$, and Post-Processing Block Size, $n$, on the horizontal axis, ranging from $10^3$ to $10^7$. The curve for $1\%$ QBER is highest, followed by $2.5\%$ and $5\%$.]

Before presenting numerical results for the **optimal expected key rates for finite $n$**, let us quickly
discuss its asymptotic behavior for arbitrarily large $n$. It is easy to verify that the key rate
asymptotically reaches $r_{\text{max}}(Q) = 1 - 2h(Q)$ for arbitrary security bounds $\varepsilon > 0$. To see this, note
that error correction can be achieved with a leakage rate of $h(Q)$ (see, e.g. [47]). Furthermore, if
we choose, for instance, $k$ proportional to $\sqrt{n}$, the statistical deviation in (S3), $\mu$, vanishes and
the ratio between the raw key length, $n$, and the expected number of exchanged qubits, $M(n, k)$,
approaches one as $n$ tends to infinity, i.e., $n/M(n, k) \rightarrow 1$. This asymptotic rate is optimal [48].
Finally, the deviations of the key length in (S3) from its asymptotic limit can be explained as
fluctuations that are due to the finiteness of the statistical samples we consider and the error
bounds we chose. These terms are necessary for any finite-key analysis. In particular, one expects
a statistical deviation $\mu$ that scales with the inverse of the square root of the sample size $k$ as
in (S3) from any statistical estimation of the error rate. In this sense our result is tight.

To obtain our results for finite block sizes $n$, we fix a security bound $\varepsilon$ and define an **optimized
$\varepsilon$-secure protocol**, $\Phi^*[n, \varepsilon]$, that results from a maximization of the expected secret key rate over
all $\varepsilon$-secure protocols with block size $n$. For the purpose of this optimization, we assume an error
correction leakage of $\text{leak}_{\text{EC}} = \xi n h(Q_{\text{tol}})$ with $\xi = 1.1$. Moreover, we bound the robustness $\varepsilon_{\text{rob}}$
by the probability that the measured security parameter exceeds $Q_{\text{tol}}$, which (for depolarizing
channels) decays exponentially in $Q_{\text{tol}} - Q$. (Note that, for general quantum channels, the error
rate in the $X$ and $Z$ bases may be different. Hence, the error correction leakage is in general not
a function of $Q_{\text{tol}}$ but of the expected error rate in the $X$ basis. Similarly, $\varepsilon_{\text{rob}}$ generally is the
sum of the robustness of parameter estimation as above and the robustness of the error correction
scheme. In this discussion, the analysis is simplified since we consider a depolarizing channel, and,
thus, the expected error rate is the same in both bases.)

In Figure 1, we present the expected key rates $r = r(\Phi, Q)$ of the optimal protocols $\Phi^*[n, \varepsilon]$
as a function of the block size $n$. These rates are given for a fixed value of the security rate $\varepsilon/\ell$,
i.e., the amount by which the security bound $\varepsilon$ increases per generated key bit. (In other words,
$\varepsilon/\ell$ can be seen as the probability of key leakage per key bit.) The plot shows that significant key
rates can be obtained already for $n = 10^4$.

In Table II, we provide selected numerical results for the optimal protocol parameters that
correspond to block sizes $n = \{10^4, 10^5, 10^6\}$ and quantum bit error rates $Q \in \{1\%, 2.5\%\}$. These
block sizes exemplify current hardware limitations in practical QKD systems.

**TABLE II. Optimized parameters for a given security rate $\varepsilon/\ell = 10^{-14}$. The column labeled $r_{\text{rel}}$ shows the
deviation of the expected secret key rate from the corresponding asymptotic value, i.e., $r_{\text{rel}} := r/(1-2h(Q))$.**

| $n$ | $Q$ (%) | $r$ (%) | $r_{\text{rel}}$ (%) | $p_x$ (%) | $Q_{\text{tol}}$ (%) | $\varepsilon_{\text{rob}}$ (%) |
|:---:|:-------:|:-------:|:--------:|:-------:|:----------:|:----------:|
| $10^4$ | $1.0$ | $11.7$ | $14.0$ | $38.2$ | $2.48$ | $2.3$ |
| | $2.5$ | $6.8$ | $10.4$ | $43.0$ | $3.78$ | $3.0$ |
| $10^5$ | $1.0$ | $30.4$ | $36.4$ | $22.0$ | $2.14$ | $0.8$ |
| | $2.5$ | $21.5$ | $32.6$ | $23.3$ | $3.58$ | $1.0$ |
| $10^6$ | $1.0$ | $47.8$ | $57.1$ | $12.5$ | $1.73$ | $0.6$ |
| | $2.5$ | $35.7$ | $53.9$ | $13.7$ | $3.21$ | $0.7$ |

In Figure 2, we compare our optimal key rates with the maximal key rates that can be shown
secure using the finite key analysis of Scarani and Renner [18]. For comparison with previous work,
we plot the rate $\ell/N$, i.e. the ratio between key length and block size, instead of the expected secret
key rate as defined by Eq. (3). We show a major improvement in the minimum block size required
to produce a provably secret key. The improvements are mainly due to a more direct evaluation
of the smooth min-entropy via the entropic uncertainty relation and the use of statistics optimized
specifically to the problem at hand (cf. Supplementary Note 2).

In conclusion, this article gives tight finite-key bounds for secure quantum key distribution with
an asymmetric BB84 protocol. Our novel proof technique, based on the uncertainty principle,
offers a conceptual improvement over earlier proofs that relied on a tomography of the state shared
between Alice and Bob. Most previous security proofs against general adversaries, e.g. [7, 18, 20,
21], are arranged in two steps: An analysis of the security against adversaries restricted to collective
attacks and a lifting of this argument to general attacks. The lifting is often possible without a
significant loss in key rate using modern techniques [24, 25]; hence, the main difference lies in

[IMAGE: Fig. 2. Comparison of Key Rate with Earlier Results. The plots show the rate $\ell/N$ as a function of the sifted key size $N = n + k$ for various channel bit error rates $Q$ (as in Fig. 1) and a security bound of $\varepsilon = 10^{-10}$. The (curved) dashed lines show the rates that can be proven secure using [18]. The horizontal dashed lines indicate the asymptotic rates for $Q \in \{1\%, 2.5\%, 5\%\}$ (from top to bottom). The plot shows Secret Key Rate, $\ell/N$, on the vertical axis, ranging from $10^{-2}$ to $10^0$, and Sifted Key Length, $N$, on the horizontal axis, ranging from $10^3$ to $10^7$. The curved solid lines for the current result (green, blue, red) are above and reach the horizontal asymptotic limit faster than the dashed lines for the earlier results.]

the first part. In security proofs against collective attacks Alice and Bob usually do tomography
on their shared state, i.e., they characterize the density matrix of their shared state. Since the
eavesdropper can be assumed to hold a purification of this state, it is then possible to bound the
von Neumann entropy of the eavesdropper on Alice’s measurement result. The min-entropy of
the eavesdropper is in turn bounded using the quantum asymptotic equipartition property [7, 49],
introducing a penalty scaling with $1/\sqrt{n}$ on the key rate. (A notable exception is [20], where the
min-entropy is bounded directly from the results of tomography.)

In contrast, our approach bounds the min-entropy directly and does not require us to do to-
mography on the state shared between Alice and Bob. In fact, we are only interested in one
correlation (between $Z$ and $Z'$) and, thus, our statistics can be produced more efficiently. (Note,
however, that this is also the reason why our approach does not reach the asymptotic key rate for
the 6-state protocol [50]. There, full tomography puts limits on Eve’s information that go beyond
the uncertainty relation in [22].) Finally, since our considerations are rather general, we believe
that they can be extended to other QKD protocols.

# METHODS

## Correctness

The required correctness is ensured in the error correction step of the protocol, when Alice
and Bob compute and evaluate a random hash function of their keys. If these hash values dis-
agree, the protocol aborts and both players output empty keys. (These keys are trivially correct.)
Since arbitrary errors in the key will be detected with high probability when the hash values are
compared [42], we can guarantee that Alice’s and Bob’s secret keys are also the same with high
probability.

## Secrecy

In order to establish the secrecy of the protocols, we consider a **gedankenexperiment** in which
Alice and Bob, after choosing a basis according to probabilities $p_x$ and $p_z$ as usual, prepare and
measure everything in the $Z$ basis. We denote the bit strings of length $n$ that replace the raw keys
$X$ and $X'$ in this hypothetical protocol as $Z$ and $Z'$, respectively. The secrecy then follows from
the fact that, if Alice has a choice of encoding a string of $n$ uniform bits in either the $X$ or $Z$ basis,
the following holds: the better Bob is able to estimate Alice’s string if she prepared in the $Z$ basis,
the worse Eve is able to guess Alice’s string if she prepared in the $X$ basis. This can be formally
expressed in terms of an **uncertainty relation for smooth entropies** [22],

$$
H_{\min}^{\varepsilon'}(X|E) + H_{\max}^{\varepsilon'}(Z|Z') \geq nq, \quad (4)
$$

where $\varepsilon' \geq 0$ is called a **smoothing parameter** and $q$, as we will see below, is the **preparation qual-
ity** defined previously. The **smooth min-entropy**, $H_{\min}^{\varepsilon'}(X|E)$, introduced in [7], characterizes the
average probability that Eve guesses $X$ correctly using her optimal strategy with access to the cor-
relations stored in her quantum memory [51]. The **smooth max-entropy**, $H_{\max}^{\varepsilon'}(Z|Z')$, corresponds
to the number of additional bits that are needed in order to reconstruct the value of $Z$ using $Z'$
up to a failure probability $\varepsilon$ [52]. For precise mathematical definitions of the smooth min- and
max-entropy, we refer to [53].

The sources we consider in this article are either a) qubit sources or b) sources that create BB84-
states by measuring part of an entangled state. In case b), a comparison with [22] reveals that
the bound on the uncertainty is given by $-\log c$, where $c$ is the overlap of the two measurement
employed in the source. For general POVMs, $\{M_x\}_x$ for preparing in the $X$ basis and $\{N_z\}_z$ for
preparing in the $Z$ basis, this overlap is given by $c = \max \|\sqrt{M_x} \sqrt{N_z}\|_\infty^2$. This justifies the
definition of the preparation quality $q = -\log c$ (as defined in Section I.B of the main text) for
such sources.

Note that in the gedankenexperiment picture—the observed average error, $\lambda$, is calculated
from $k$ measurements sampled at random from $n + k$ measurements in the $Z$ basis. Hence, if $\lambda$ is
small, we deduce that, with high probability, $Z$ and $Z'$ are highly correlated and, thus, $H_{\max}^{\varepsilon'}(Z|Z')$
is small. In fact, since the protocol aborts if $\lambda$ exceeds $Q_{\text{tol}}$, the following bound on the smooth
max-entropy (conditioned on the correlation test passing) holds:

$$
H_{\max}^{\varepsilon'}(Z|Z') \leq n h(Q_{\text{tol}} + \mu), \quad (5)
$$

where $\mu$ takes into account statistical fluctuations and depends on the security parameter via $\varepsilon$.
Eq. (5) is shown in Supplementary Note 2 using an upper bound by Serfling [54] on the probability
that the average error on the sample, $\lambda$, deviates by more than $\mu$ from the average error on the
total string. (See also [55].)

In addition to the uncertainty relation, our analysis employs the **Quantum Leftover Hash
Lemma** [7, 45], which gives a direct operational meaning to the smooth min-entropy. It asserts
that, using a random universal₂ hash function, it is possible to extract a $\Delta$-secret key of length $\ell$
from $X$, where

$$
\Delta = 2\varepsilon + \frac{1}{2} \sqrt{2^{\ell - H_{\min}^{\varepsilon'}(X|E')}}. \quad (\text{S}2)
$$

Here, $E'$ summarizes all information Eve learned about $X$ during the protocol—including the
classical communication sent by Alice and Bob over the authenticated channel. Furthermore, the
extracted secret key is independent of the randomness that is used to choose the hash function.

The following theorem gives a sufficient condition for which a protocol $\Phi$ using a source with
preparation quality $q$ is $\varepsilon_{\text{sec}}$-secret. The minimum value $\varepsilon_{\text{sec}}$ for which it is $\varepsilon_{\text{sec}}$-secret is called the
**secrecy** of the protocol and is denoted by $\varepsilon_{\text{sec}}(\Phi, q)$.

**Theorem 2. The protocol $\Phi[n, k, \ell, Q_{\text{tol}}, \varepsilon_{\text{cor}}, \text{leak}_{\text{EC}}]$ using a source with preparation quality $q$ is**

$$
\varepsilon_{\text{sec}}\text{-secret for some } \varepsilon_{\text{sec}} > 0 \text{ if } \ell \text{ satisfies}^1 \\
\ell \leq \max_{\varepsilon, \varepsilon'} \left[ n(q - h(Q_{\text{tol}} + \mu(\varepsilon))) - 2 \log \frac{1}{2\varepsilon'} - \text{leak}_{\text{EC}} - \log \frac{2}{\varepsilon_{\text{cor}}} \right], \quad (\text{S}3)
$$

where we optimize over $\varepsilon > 0$ and $\tilde{\varepsilon} > 0$ s.t. $2\varepsilon + \tilde{\varepsilon} \leq \varepsilon_{\text{sec}}$ and

$$
\mu(\varepsilon) := \sqrt{\frac{n + k}{n k} \frac{k + 1}{k}} \ln \frac{4}{\varepsilon}, \quad (\text{S}4)
$$

*Proof.* In the *gedankenexperiment* picture described above, $A$ is a random variable calculated from
at least $k$ measurements sampled at random from $n + k$ measurements in the $Z$ basis. Hence, if $\lambda$
is small, we deduce that, with high probability, $Z$ and $Z'$ are highly correlated and $H_{\max}^{\varepsilon'}(Z|Z')$ is
small. This is elaborated in Lemma 3, where it is shown that, conditioned on the event that the
correlation test passed ($\lambda \leq Q_{\text{tol}}$), the following bound on the smooth max-entropy holds,

$$
H_{\max}^{\varepsilon'}(Z|Z')_{\rho} \leq n h(Q_{\text{tol}} + \mu(\varepsilon)), \quad (\text{S}5)
$$

where $\varepsilon' = \varepsilon / P_{\text{pass}}$ and $P_{\text{pass}} \geq 1 - P_{\text{abort}}$ is the probability that the correlation test passes.
Here, $\rho$ is the state of the system conditioned on the event that the correlation test passed. More
precisely, we consider the state $\rho_{ABE}$ of the $N$ systems shared between Alice and Bob as well
as Eve’s information. Moreover, the classical joint probability distributions $P_{XX'}$ and $P_{ZZ'}$ are
induced by the respective measurement on $A$ and $B$. (Note that these states are well-defined
since, by assumption, we know that the measurement of the $n$ bits used for key generation can be
postponed until after parameter estimation.)

We now apply the uncertainty relation, $H_{\min}^{\varepsilon'}(X|E)_{\rho} \geq nq - H_{\max}^{\varepsilon'}(Z|Z')_{\rho}$, on this state to find
a lower bound on the min-entropy that Eve has about Alice’s bits prepared in the $X$ basis. Since a
maximum of $\text{leak}_{\text{EC}} + \lceil \log(1/\varepsilon_{\text{cor}}) \rceil \leq \text{leak}_{\text{EC}} + \log(2/\varepsilon_{\text{cor}})$ bits of information about $X$ are revealed
during error correction, we find$^2$

$$
H_{\min}^{\varepsilon'}(X|E')_{\rho} \geq H_{\min}^{\varepsilon'}(X|E)_{\rho} - \text{leak}_{\text{EC}} - \log \frac{2}{\varepsilon_{\text{cor}}} \quad (\text{S}6) \\
\geq nq - H_{\max}^{\varepsilon'}(Z|Z')_{\rho} - \text{leak}_{\text{EC}} - \log \frac{2}{\varepsilon_{\text{cor}}} \quad (\text{S}7) \\
\geq n(q - h(Q_{\text{tol}} + \mu(\varepsilon))) - \text{leak}_{\text{EC}} - \log \frac{2}{\varepsilon_{\text{cor}}} \quad (\text{S}8)
$$

Thus, combining this with (S2) and using the proposed key length (S3), we find, for all $\varepsilon$ and $\tilde{\varepsilon}$,

$$
\Delta \leq 2\varepsilon' + \frac{1}{2} \sqrt{2^{\ell - H_{\min}^{\varepsilon'}(X|E')}} \leq 2\varepsilon' + \tilde{\varepsilon}. \quad (\text{S}9)
$$

The security of the protocol now follows since $(1 - P_{\text{abort}}) \Delta \leq 2\varepsilon + \tilde{\varepsilon} \leq \varepsilon_{\text{sec}}$.

# SUPPLEMENTARY NOTE 2: STATISTICS

This section covers the statistical analysis of the classical data collected during the run of the
BB84-type protocols described in this work. A more general framework for such an analysis can
be found in [19]

We use the notation of the previous sections and define $N := n + k$. The fraction of bits that
are used for parameter estimation is denoted as $\nu$, i.e. $k = \nu N$ and $n = (1 - \nu) N$.
The statistical analysis is based on a *gedankenexperiment*, where Alice and Bob measure all
$N$ states with $i \in \mathcal{X} \cup \mathcal{Z}$ in the control basis, $Z$, resulting in strings $Z_{\text{tot}}$ and $Z'_{\text{tot}}$ for Alice and
Bob, respectively. The following random variables are of interest to us. The **relative Hamming
distance** between Alice’s and Bob’s bit-string is defined as $\Lambda_{\text{tot}} = \frac{1}{N} |Z_{\text{tot}} \oplus Z'_{\text{tot}}|$, where $|\cdot|$ denotes
the Hamming weight. Similarly, $\Lambda = \Lambda_{\mathcal{P} \mathcal{Z}}$ denotes the **relative Hamming distances** between the
random subsets $Z_{\mathcal{P} \mathcal{Z}}$ of $Z_{\text{tot}}$ and $Z'_{\mathcal{P} \mathcal{Z}}$ of $Z'_{\text{tot}}$ used for **parameter estimation**. Finally, $\Lambda_{\text{key}}$ is the
**relative Hamming distance** between the remainders of the strings, denoted $Z = Z_{\text{key}}$ and $Z' = Z'_{\text{key}}$.
Clearly,

$$
\Lambda_{\text{tot}} = \nu \Lambda + (1 - \nu) \Lambda_{\text{key}}.
$$

The $k$ bits used for parameter estimation are chosen at random from $N$ bits. Hence, if we
fix $\Lambda_{\text{tot}} = \lambda_{\text{tot}}$ for the moment, the random variables $\Lambda$ and $\Lambda_{\text{key}}$ can be seen as emanating from
sampling without replacement. We apply the bound [54]

$$
\text{Pr} [\Lambda_{\text{key}} \geq \lambda_{\text{tot}} + \delta | \Lambda_{\text{tot}} = \lambda_{\text{tot}}] \leq e^{-2 \frac{n k}{n+k} \delta^2} \quad (\text{S}10)
$$

We now derive a bound on the probability that $\Lambda_{\text{key}}$ exceeds $\Lambda$ by more than a constant $\mu$ condi-
tioned on the event that we passed the correlation test. (Note that, while $\Lambda$ is accessible during
the protocol, $\Lambda_{\text{key}}$ is the quantity we are actually interested in.) We find, using Bayes’ theorem,

$$
\text{Pr} [\Lambda_{\text{key}} \geq \Lambda + \mu | “\text{pass}” ] \leq \frac{1}{P_{\text{pass}}} \text{Pr} [\Lambda_{\text{key}} \geq \Lambda + \mu], \quad (\text{S}11)
$$

where we keep $P_{\text{pass}} = \text{Pr}[“\text{pass}”] = \text{Pr}[\Lambda \leq Q_{\text{tol}}]$ as a parameter and further bound

$$
\text{Pr} [\Lambda_{\text{key}} \geq \Lambda + \mu] = \text{Pr} [\Lambda_{\text{key}} \geq \Lambda_{\text{tot}} + \nu \mu] \quad (\text{S}12) \\
= \sum_{\lambda_{\text{tot}}} \text{Pr} [\Lambda_{\text{tot}} = \lambda_{\text{tot}}] \text{Pr} [\Lambda_{\text{key}} \geq \lambda_{\text{tot}} + \nu \mu | \Lambda_{\text{tot}} = \lambda_{\text{tot}}] \leq e^{-2 \frac{n k}{N k} (\nu \mu)^2}. \quad (\text{S}13)
$$

We used (S10) to bound each summand individually. Finally, defining $\varepsilon := e^{-\frac{n k}{N} (\nu \mu)^2}$, we write

$$
\text{Pr} [\Lambda_{\text{key}} \geq \Lambda + \mu | “\text{pass}”] \leq \frac{\varepsilon^2}{P_{\text{pass}}} \quad (\text{S}14)
$$

The above result can be used to bound the uncertainty Bob has about Alice’s measurement
outcomes in the $Z$-basis, as expressed using the smooth max-entropy of $Z$ given $Z'$ and $\Lambda$. The
entropy is evaluated for the probability distribution conditioned on the event that the correlation
test passed, which we denote $P_{ZZ'\Lambda}(z, z', \lambda) = \text{Pr}[Z = z \land Z' = z' \land \Lambda = \lambda | “\text{pass}”]$.

**Lemma 3.** Let $\varepsilon > 0$. Then

$$
H_{\max}^{\varepsilon'}(Z|Z') \leq n h(Q_{\text{tol}} + \mu), \quad \text{where} \quad \varepsilon' := \frac{\varepsilon}{\sqrt{P_{\text{pass}}}} \quad \text{and} \quad \mu := \sqrt{\frac{n + k}{n k} \frac{k + 1}{k}} \ln \frac{1}{\varepsilon} \quad (\text{S}15)
$$

*Proof.* According to (S14), the probability that $\Lambda_{\text{key}}$ exceeds $\Lambda$ by more than $\mu$ is bounded. In
fact, we can find a probability distribution,

$$
Q_{ZZ'\Lambda}(z, z', \lambda) := \begin{cases} \frac{P_{ZZ'\Lambda}(z, z', \lambda)}{\text{Pr}[\Lambda_{\text{key}} < \Lambda + \mu | “\text{pass}”]} & \text{if } \Lambda_{\text{key}}(z, z') < \lambda + \mu \\ 0 & \text{else} \end{cases} \quad (\text{S}16)
$$

which is $\varepsilon'$-close to $P_{ZZ'\Lambda}$ in terms of the purified distance. To see this, note that the fidelity
between the two distributions satisfies

$$
F(P, Q) := \sum_{Z, Z', \Lambda} \sqrt{P_{ZZ'\Lambda}(z, z', \lambda) Q_{ZZ'\Lambda}(z, z', \lambda)} = \sqrt{\text{Pr}[\Lambda_{\text{key}} < \Lambda + \mu | “\text{pass}”]}, \quad (\text{S}17)
$$

which can be bounded using (S14). The purified distance between the distributions is then given by
$P(P, Q) := \sqrt{1 - F^2(P, Q)} = \varepsilon'$. Hence, under the distribution $Q$, we have $\Lambda_{\text{key}} < \Lambda + \mu \leq Q_{\text{tol}} + \mu$
with certainty. In particular, the total number of errors on $n$ bits, $W := n \Lambda_{\text{key}}$, satisfies

$$
W \leq \lceil n(Q_{\text{tol}} + \mu) \rceil. \quad (\text{S}18)
$$

The max-entropy, $H_{\max}(Z|Z')$, is upper bounded by the minimum number of bits of additional
information about $Z$ needed to perfectly reconstruct $Z$ from $Z'$ [52]. This value can in turn be
upper bounded by the logarithm of the maximum support of $Z$ conditioned on any value $Z' = z'$.
Since the total number of errors under $Q$ satisfies (S18), we may write

$$
H_{\max}(Z|Z')_{\rho} \leq H_{\max}(Z|Z')_Q \leq \log \sum_{w=0}^{\lceil n(Q_{\text{tol}}+\mu) \rceil} \binom{n}{w} \leq n h(Q_{\text{tol}} + \mu). \quad (\text{S}19)
$$

The last inequality is shown in [55], Section 1.4. This concludes the proof of Lemma 3.

**REFERENCES**

[1] C. H. Bennett and G. Brassard. Quantum cryptography: Public key distribution and coin tossing. In
*Proc. IEEE Int. Conf. on Comp., Sys. and Signal Process.*, pages 175–179, Bangalore, India (1984).
[2] A. K. Ekert. Quantum cryptography based on Bell’s theorem. *Phys. Rev. Lett.*, 67, 661–663 (1991).
[3] H.-K. Lo and H. F. Chau. Unconditional Security of Quantum Key Distribution over Arbitrarily Long
Distances. *Science*, 283, 2050–2056 (1999).
[4] P. Shor and J. Preskill. Simple Proof of Security of the BB84 Quantum Key Distribution Protocol.
*Phys. Rev. Lett.*, 85, 441–444 (2000).
[5] E. Biham, M. Boyer, P. O. Boykin, T. Mor, and V. Roychowdhury. A Proof of the Security of Quantum
Key Distribution. *J. Cryptology*, 19, 381–439 (2006).
[6] D. Mayers. Unconditional security in quantum cryptography. *J. ACM*, 48, 351–406 (2001).
[7] R. Renner. Security of Quantum Key Distribution. PhD thesis, ETH Zurich. Preprint arXiv:0512258
(2005).
[8] R. Renner, N. Gisin, and B. Kraus. Information-theoretic security proof for quantum-key-distribution
protocols. *Phys. Rev. A*, 72, 012332 (2005).
[9] H. Takesue, S. W. Nam, Q. Zhang, R. H. Hadfield, T. Honjo, K. Tamaki, and Y. Yamamoto. Quantum
key distribution over 40 dB channel loss using superconducting single photon detectors. *Nat. Photon.*,
1, 343–357 (2007).
[10] A. R. Dixon, Z. L. Yuan, J. F. Dynes, A. W. Sharpe, and A. J. Shields. Gigahertz decoy quantum key
distribution with 1 Mbit/s secure key rate. *Opt. Express*, 16, 18790–18979 (2008).
[11] D. Stucki, N. Walenta, F. Vannel, R. T. Thew, N. Gisin, H. Zbinden, S. Gray, C. R. Towery, and S. Ten.
High rate, long-distance quantum key distribution over 250 km of ultra low loss fibres. *New J. Phys.*,
11, 75003 (2009).
[12] V. Scarani and C. Kurtsiefer. The black paper of quantum cryptography: real implementation problems.
Preprint arXiv:0906.4547 (2009).
[13] E. Hänggi. Device-Independent Quantum Key Distribution. PhD thesis, ETH Zurich. Preprint
arXiv:1012.3878 (2010).
[14] L. Masanes, S. Pironio, and A. Acín. Secure device-independent quantum key distribution with causally
independent measurement devices. *Nat. Commun.*, 2, 238 (2011).
[15] M. Hayashi. Practical evaluation of security for quantum key distribution. *Phys. Rev. A*, 74, 022307
(2006).
[16] T. Meyer, H. Kampermann, M. Kleinmann, and D. Bruß. Finite key analysis for symmetric attacks in
quantum key distribution. *Phys. Rev. A*, 74, 042340 (2006).
[17] H. Inamori, N. Lütkenhaus, and D. Mayers. Unconditional security of practical quantum key distribu-
tion. *Eur. Phys. J. D*, 41, 599–627 (2007).
[18] V. Scarani and R. Renner. Quantum Cryptography with Finite Resources: Unconditional Security
Bound for Discrete-Variable Protocols with One-Way Postprocessing. *Phys. Rev. Lett.*, 100, 200501
(2008).
[19] N. Bouman and S. Fehr. Sampling in a Quantum Population, and Applications. Preprint
arXiv:0907.4246 (2009).
[20] S. Bratzik, M. Mertz, H. Kampermann, and D. Bruß. Min-entropy and quantum key distribution:
Nonzero key rates for small numbers of signals. *Phys. Rev. A*, 83, 022330 (2011).
[21] L. Sheridan, T. P. Le, and V. Scarani. Finite-key security against coherent attacks in quantum key
distribution. *New J. Phys.*, 12, 123019 (2010).
[22] M. Tomamichel and R. Renner. Uncertainty Relation for Smooth Entropies. *Phys. Rev. Lett.*, 106,
110506 (2011).
[23] M. Berta, M. Christandl, R. Colbeck, J. M. Renes, and R. Renner. The uncertainty principle in the
presence of quantum memory. *Nat. Phys.*, 6, 659–662 (2010).
[24] R. Renner. Symmetry of large physical systems implies independence of subsystems. *Nat. Phys.*, 3,
645–649 (2007).
[25] M. Christandl, R. König, and R. Renner. Postselection Technique for Quantum Channels with Appli-
cations to Quantum Cryptography. *Phys. Rev. Lett.*, 102, 020504 (2009).
[26] R. Canetti. Universally composable security: a new paradigm for cryptographic protocols. In *Proc.
IEEE Int. Conf. on Cluster Comput.*, pages 136–145. IEEE (2001).
[27] J. Müller-Quade and R. Renner. Composability in quantum cryptography. *New J. Phys.*, 11, 085006
(2009).
[28] R. König, R. Renner, A. Bariska, and U. Maurer. Small Accessible Quantum Information Does Not
Imply Security. *Phys. Rev. Lett.*, 98, 140502 (2007).
[29] N. Lütkenhaus. Security against individual attacks for realistic quantum key distribution. *Phys. Rev.
A*, 61, 052304 (2000).
[30] H.-K. Lo, X. Ma, and K. Chen. Decoy State Quantum Key Distribution. *Phys. Rev. Lett.*, 94, 230504
(2005).
[31] J. Hasegawa, M. Hayashi, T. Hiroshima, and A. Tomita. Security analysis of decoy state quantum key
distribution incorporating finite statistics. Preprint arXiv:0707.3541 (2007).
[32] R. Y. Q. Cai and V. Scarani. Finite-key analysis for practical implementations of quantum key distri-
bution. *New Journal of Physics*, 11, 045024 (2009).
[33] T.-T. Song, J. Zhang, S.-J. Qin, F. Gao, and Q.-Y. Wen. Finite-key analysis for quantum key distri-
bution with decoy states. *Quant. Inf. Comput.*, 11, 0374–0389 (2011).
[34] C. H. Bennett, G. Brassard, N. D. Mermin. Quantum cryptography without Bell’s theorem. *Phys.
Rev. Lett.*, 68, 557–559 (1992).
[35] T. Pittman, B. Jacobs, and J. Franson. Heralding single photons from pulsed parametric down-
conversion. *Optics Commun.*, 246, 545–550 (2005).
[36] G. Y. Xiang, T. C. Ralph, A. P. Lund, N. Walk, and G. J. Pryde. Heralded noiseless linear amplification
and distillation of entanglement. *Nat. Photon.*, 4, 316–319 (2010).
[37] N. Gisin, S. Pironio, and N. Sangouard. Proposal for Implementing Device-Independent Quantum Key
Distribution Based on a Heralded Qubit Amplifier. *Phys. Rev. Lett.*, 105, 070501 (2010).
[38] M. Curty and T. Moroder. Heralded-qubit amplifiers for practical device-independent quantum key
distribution. *Phys. Rev. A*, 84, 010304 (2011).
[39] D. Pitkänen, X.F. Ma, R. Wickert, P. van Loock, and N. Lütkenhaus. Efficient Heralding of Photonic
Qubits with Apllications to Device Independent Quantum Key Distribution. *Phys. Rev. A*, 84, 022325
(2011).
[40] L. Lydersen, C. Wiechers, C. Wittmann, D. Elser, J. Skaar, and V. Makarov. Hacking commercial
quantum cryptography systems by tailored bright illumination. *Nat. Photon.*, 4, 686–689 (2010).
[41] H.-K. Lo, H. Chau, and M. Ardehali. Efficient Quantum Key Distribution Scheme and a Proof of Its
Unconditional Security. *J. Cryptology*, 18, 2, 33–165 (2004).
[42] J. L. Carter and M. N. Wegman. Universal classes of hash functions. *J. Comp. Syst. Sci.*, 18, 2.
143–154 (1979).
[43] C. H. Bennett, G. Brassard, C. Crepeau, and U. M. Maurer. Generalized privacy amplification. *IEEE
Trans. on Inf. Theory*, 41, 6, 1915–1923 (1995).
[44] R. Renner and R. König. Universally Composable Privacy Amplification Against Quantum Adversaries.
In *Proc. TCC, 3378 of LNCS*, pages 407–425, Cambridge, USA (2005).
[45] M. Tomamichel, C. Schaffner, A. Smith, and R. Renner. Leftover Hashing Against Quantum Side
Information. *IEEE Trans. on Inf. Theory*, 57, 5524–5535 (2011).
[46] A. De, C. Portmann, T. Vidick, and R. Renner. Trevisan’s Extractor in the Presence of Quantum Side
Information. Preprint arXiv:0912.5514 (2009).
[47] T. M. Cover and J. A. Thomas. *Elements of Information Theory*. Wiley (1991).
[48] R. Renner, N. Gisin, and B. Kraus. Information-theoretic security proof for quantum-key-distribution
protocols. *Phys. Rev. A*, 72, 012332 (2005).
[49] M. Tomamichel, R. Colbeck, and R. Renner. A Fully Quantum Asymptotic Equipartition Property.
*IEEE Trans. on Inf. Theory*, 55, 5840–5847 (2009).
[50] D. Bruß. Optimal Eavesdropping in Quantum Cryptography with Six States. *Phys. Rev. Lett.*, 81,
3018–3021 (1998).
[51] R. König, R. Renner, and C. Schaffner. The Operational Meaning of Min- and Max-Entropy. *IEEE
Trans. on Inf. Theory*, 55, 4337–4347 (2009).
[52] J. M. Renes and R. Renner. One-Shot Classical Data Compression with Quantum Side Information
and the Distillation of Common Randomness or Secret Keys. Preprint arXiv:1008.0452 (2010).
[53] M. Tomamichel, R. Colbeck, and R. Renner. Duality Between Smooth Min- and Max-Entropies. *IEEE
Trans. on Inf. Theory*, 54, 4674–4681 (2010).
[54] R. J. Serfling. Probability Inequalities for the Sum in Sampling without Replacement. *Ann. Stat.*,
2(1):39–48 (1974).
[55] J. H. van Lint. *Introduction to Coding Theory*. Graduate Texts in Mathematics. Springer (1999).