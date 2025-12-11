Quantum Information and Computation, Vol. 9, No. 11\&12 (2009) 0963-0996
© Rinton Press

# ROBUST CRYPTOGRAPHY IN THE NOISY-QUANTUM-STORAGE MODEL

**CHRISTIAN SCHAFFNER**
Centrum Wiskunde & Informatica (CWI)
P.O. Box 94079, 1090 GB Amsterdam, The Netherlands

**BARBARA TERHAL**
IBM, Watson Research Center
P.O. Box 218, Yorktown Heights, NY, USA

**STEPHANIE WEHNER**
Caltech, Institute for Quantum Information
1200 E California Blvd, Pasadena CA 91125, USA

Received February 17, 2009
Revised July 29, 2009

It was shown in [42] that cryptographic primitives can be implemented based on the assumption that quantum storage of qubits is noisy. In this work we analyze a protocol for the universal task of oblivious transfer that can be implemented using quantum-key-distribution (QKD) hardware in the practical setting where honest participants are unable to perform noise-free operations. We derive trade-offs between the amount of storage noise, the amount of noise in the operations performed by the honest participants and the security of oblivious transfer which are greatly improved compared to the results in [42]. As an example, we show that for the case of depolarizing noise in storage we can obtain secure oblivious transfer as long as the quantum bit-error rate of the channel does not exceed $11\%$ and the noise on the channel is strictly less than the quantum storage noise. This is optimal for the protocol considered. Finally, we show that our analysis easily carries over to quantum protocols for secure identification.

**Keywords:** quantum cryptography, noisy-storage model
**Communicated by:** R Cleve & J Watrous

## 1 Introduction

The noisy-quantum-storage model [42] is based on the assumption that it is difficult to store quantum states. Based on current practical and near-future technical limitations, we assume that any state placed into quantum storage is affected by noise. At the same time the model assumes that preparation, transmission and measurement of simple unentangled quantum states can be performed with much lower levels of noise. The present-day technology of quantum key distribution with photonic qubits demonstrates this contrast between a relatively simple technology for preparation/transmission/measurement versus a limited capability for quantum storage.

$^{\text{a}}$c.schaffner@cwi.nl
$^{\text{b}}$bterhal@gmail.com
$^{\text{c}}$wehner@caltech.edu

963

964 Robust cryptography in the noisy-quantum-storage model

Table 1. Summary of previous results and the results in this paper. The allowed quantum bit-error rate (QBR) is the maximum effective error-rate on the actions of the honest parties below which we can prove the security of the cryptographic scheme.

| | Allowed QBR | Secure 1-2 OT | Secure Identification |
| :--- | :--- | :--- | :--- |
| PRL [42] | None | Yes | No |
| Unpublished [41] | 2.9% | Yes | No |
| This work | 11% (optimal) | Yes | Yes |

Almost all interesting cryptographic tasks are impossible to realize without any restrictions on the participating players, neither classically nor with the help of quantum information, see e.g. [27, 30, 29, 28, 31]. It is therefore an important task to come up with a cryptographic model which restricts the capabilities of adversarial players and in which these tasks become feasible. It turns out that all such two-party protocols can be based on a simple primitive called 1-2 Oblivious Transfer (1-2 OT) [24, 19], first introduced in [44, 35, 16]. In 1-2 OT, the sender Alice starts off with two bit strings $S_0$ and $S_1$, and the receiver Bob holds a choice bit $C$. The protocol allows Bob to retrieve $S_C$ in such a way that Alice does not learn any information about $C$ (thus, Bob cannot simply ask for $S_C$). At the same time, Alice must be ensured that Bob only learns $S_C$, and no information about the other string $S_{\bar{C}}$ (thus, Alice cannot simply send him both $S_0$ and $S_1$). A 1-2 OT protocol is called unconditionally secure when neither Alice nor Bob can break these conditions, even when given unlimited resources.

## 2 Results

In this work we focus on the setting where the honest parties are unable to perform perfect operations and experience errors themselves, where we analyze individual-storage attacks. These honest-party errors can be modeled as bit-errors on an effective channel connecting the honest parties. In unpublished work, we have shown that for the case of depolarizing noise in storage, security can be obtained if the actions of the honest parties are noisy but their error rate does not exceed $2.9\%$ [41]. This threshold is too low to be of any practical value. In particular, this result left open the question whether security can be obtained in a real-life scenario.

Using a very different analysis, we are now able to show that in the setting of individual-storage attacks 1-2 oblivious transfer and secure identification can be achieved in the noisy-storage model with depolarizing storage noise, as long as the quantum bit-error rate of the channel does not exceed $11\%$ and the noise on the channel is strictly less than the noise during quantum storage. This is optimal for the protocol considered.

Our result is of great practical significance, since it paves the way to achieve security in a real-life implementation. Our main new Theorems 4 and 6 relate the security of the 1-2 OT protocol to an uncertainty lower bound on the conditional Shannon entropy. In order to prove these theorems, we need to relate the Shannon entropy to the smooth min-entropy and establish several new properties of the smooth min-entropy, see Section 3.2.1.

We evaluate the uncertainty lower bounds on the conditional Shannon entropy in the practically-interesting case of depolarizing noise resulting in Theorems 5 and 7. From this analysis we obtain the clear-cut result that, depending on the amount of storage noise, the

965 C. Schaffner, B. Terhal, and S. Wehner

adversary's optimal storage attack is to either store the incoming state as is, or to measure it immediately in one of the two BB84 bases.

### 2.1 The Noisy-Quantum-Storage Model and Individual-Storage Attacks

The noisy-storage model assumes that any quantum state that is placed into quantum storage is affected by some noise described by a quantum operation $N$. Practically, noise can arise as a result of transferring the qubit onto a different physical carrier, for example the transfer of a photonic qubit onto an atomic ensemble or atomic state. In addition, a quantum state will undergo noise once it has been transferred into 'storage' if such quantum memory is not $100\%$ reliable.

In principle, one may like to prove security against an adversary that can perform any operation on the incoming quantum states. Here however we analyze the restricted case where the adversary Bob performs **individual-storage attacks**. More precisely, Bob's actions are of the following form as depicted in Figure 1.

1.  Bob may choose to (partially) measure (a subset of) his qubits immediately upon reception using an error-free product measurement, i.e., when he receives the $j$th qubit, he may apply any measurement $P_j$ of his choosing.
2.  In addition, he can store each incoming qubit, or post-measurement state from a prior partial measurement, *separately* and wait until he gets additional information from Alice (at Step 3 in Protocol 1). During storage, the $j$th qubit is thereby affected by some noise described by a quantum operation $N_j$ acting independently on each qubit. Note that such quantum operation $N_j$ could come about from encoding an incoming qubit into an error-correcting code and decoding it right before receiving Alice's additional information.
3.  Once Bob obtains the additional information he may perform an arbitrary coherent measurement $M$ on his stored qubits and stored classical data.

We would like to note that we can also derive security if we would allow Bob to initially perform any, non-product, destructive measurement on the incoming qubits. By destructive we mean that there is no post-measurement quantum data left. The reason is that we have previously shown in Lemma 2 in [42], that destructive product measurements are optimal for Bob if he is not allowed to keep any post-measurement information. Hence this optimality of product measurements reduces such more general destructive measurement model to our model of individual-storage attacks. Measurements in present-day technology with single photon qubits in which photons are detected, are in fact always destructive, hence our model includes many realistic attacks. Intuitively, using entangling operations between the incoming qubits should be of little help in either extracting more information from these independent, uncorrelated, BB84 qubits or in better preserving these qubits against noise when the noise is extremely low and more is lost than gained by measuring some qubits right away and using part of the newly freed space to encode the remaining qubits. Of course this remains to be proven (see also Conclusion). What can help is to entangle an incoming qubit individually with ancilla qubits in order to store the incoming qubit in an encoded or other more robust

966 Robust cryptography in the noisy-quantum-storage model

[IMAGE: Fig. 1. Individual-Storage Attacks. A diagram showing the process for multiple qubits. For the $j$-th qubit, an input `quantum` state goes through an operation $P_j$ and outputs a `classical outcome` and a `quantum` state. This quantum state then goes through a noise operation $N_j$. The classical outcome and the noisy quantum state are inputs, along with `additional information`, to a final `Measure` operation $M$, which outputs a `guess`.]

form. This attack is covered in our model as an effective noisy operation $N_j$ on incoming qubit $j$.

In the following, we use the quantum operation $\mathcal{S}_j$ to denote the combined quantum operations of Bob's initial (partial) measurement and the noise.

### 2.2 Related work

Our model is closely related to the bounded-quantum-storage model, which assumes that the adversary has a limited amount of quantum memory at his disposal [11, 12]. Within this 'bounded-quantum-storage model' OT can be implemented securely as long as a dishonest receiver Bob can store at most $n/4 - O(1)$ qubits coherently, where $n$ is the number of qubits transmitted from Alice to Bob. This approach assumes an explicit limit on the physical number of qubits (or more precisely, the rank of the adversary's quantum state). However, at present we do not know of any practical physical situation which enforces such a limit for quantum information. As was pointed out in [39, 14], the original bounded-quantum-storage analysis applies in the case of noise levels which are so large such that the dishonest player's quantum storage has an effective noise-free Hilbert space with dimension at most $2^{n/4}$. The advantage of our model is that we can evaluate the security parameters of a protocol explicitly in terms of the strength of the noise, even when the noise rate is very low.

Precursors of the idea of basing cryptographic security on storage-noise are already present in [3], but no rigorous analysis was carried through in that paper. We furthermore note that our security proof does not exploit the noise in the communication channel (which has been done in the classical setting to achieve cryptographic tasks, see e.g. [9, 10, 8]), but is solely based on the fact that the dishonest receiver's quantum storage is noisy. A model based on classical noisy storage is akin to the setting of a classical noisy channel, if the operations are

967 C. Schaffner, B. Terhal, and S. Wehner

noisy, or the classical bounded-storage model, both of which are difficult to enforce in practice. Another technical limitation has been considered in [38] where a bit-commitment scheme was shown secure under the assumption that the dishonest committer can only measure a limited amount of qubits coherently. Our analysis differs in that we can in fact allow any coherent destructive measurement at the end of the protocol.

### 2.3 Outline

In Section 3, we introduce some notation and the necessary technical tools. In Section 4, we define the security of 1-2 OT, present the protocol and prove its security in the case when honest players do not experience noise. In Section 5 we then consider the example of depolarizing storage noise explicitly. The lengthy proof of Theorem 5 can be found in Appendix B. In Section 6 we show how to obtain security if the honest players are unable to perform perfect quantum operations. Finally, we point out in Section 7 how our analysis carries over to other protocols.

## 3 Preliminaries

We start by introducing the necessary definitions, tools and technical lemmas that we need in the remainder of this text.

### 3.1 Basic Concepts

We use $x \in_{R} \mathcal{X}$ to denote the uniform random choice of an element from a set $\mathcal{X}$. We further use $x_{\mathcal{T}}$ to denote the string $x=x_1,...,x_n$ restricted to the bits indexed by the set $\mathcal{T} \subseteq \{1,...,n\}$. For a binary random variable $C$, we denote by $\bar{C}$ the bit different from $C$.

Let $\mathcal{B}(\mathcal{H})$ denote the set of all bounded operators on a finite-dimensional Hilbert space $\mathcal{H}$. Let $\mathcal{P}(\mathcal{H}) \subset \mathcal{B}(\mathcal{H})$ denote the subset of positive semi-definite Hermitian operators on $\mathcal{H}$, and let $\mathcal{S}(\mathcal{H}) \subset \mathcal{P}(\mathcal{H})$ denote the subset of all quantum states on $\mathcal{H}$, i.e. $\rho \in \mathcal{S}(\mathcal{H})$ iff $\rho \in \mathcal{B}(\mathcal{H})$ with $\rho \ge 0$ and $Tr(\rho)=1$. $Tr_A : \mathcal{B}(\mathcal{H}_{AB}) \to \mathcal{B}(\mathcal{H}_B)$ is the partial trace over system $A$. We denote by $id_A$ the identity operator on system $A$. Let $|0\rangle_+, |1\rangle_+, |0\rangle_{\times} := (|0\rangle_+ + |1\rangle_+)/\sqrt{2}, |1\rangle_{\times} := (|0\rangle_+ - |1\rangle_+)/\sqrt{2}$ denote the BB84-states corresponding to the encoding of a classical bit into the computational or Hadamard basis, respectively.

**Classical-Quantum States** A cq-state $\rho_{XE}$ is a state that is partly classical, partly quantum, and can be written as
$$
\rho_{XE} = \sum_{x \in \mathcal{X}} P_X(x) |x\rangle\langle x| \otimes \rho_E^x .
$$
Here, $X$ is a classical random variable distributed over the finite set $\mathcal{X}$ according to distribution $P_X$, $\{|x\rangle\}_{x \in \mathcal{X}}$ is a set of orthonormal states and the register $E$ is in state $\rho_E^x$ when $X$ takes on value $x$.

**Distance measures** The $L_1$-norm of an operator $A \in \mathcal{B}(\mathcal{H})$ is defined as $||A||_1 := Tr \sqrt{A^{\dagger}A}$. The fidelity between two quantum states $\rho, \sigma$ is defined as $F(\rho, \sigma) := ||\sqrt{\rho}\sqrt{\sigma}||_1$. For pure states it takes on the easy form $F(|\phi\rangle\langle\phi|, |\psi\rangle\langle\psi|) = |\langle\phi|\psi\rangle|$. The related quantity $C(\rho, \sigma) := \sqrt{1 - F^2(\rho, \sigma)}$ is a convenient distance measure on normalized states [1]. It is invariant

968 Robust cryptography in the noisy-quantum-storage model

under purifications and equals the trace distance for pure states, i.e. $C(|\psi\rangle\langle\psi|, |\phi\rangle\langle\phi|) = \sqrt{1 - |\langle\psi|\phi\rangle|^2} = \frac{1}{2} |||\psi\rangle\langle\psi| - |\phi\rangle\langle\phi|||_1$.

**Non-uniformity** We can say that a quantum adversary has little information about $X$ if the distribution $P_X$ given his quantum state is close to uniform. Formally, this distance is quantified by the **non-uniformity** of $X$ given $\rho_E = \sum_x P_X(x) \rho_E^x$ defined as
$$
d(X|E)_{\rho} := \frac{1}{2} \left\| id_{\mathcal{X}}/\lvert \mathcal{X} \rvert \otimes \rho_E - \sum_{x} P_X(x) |x\rangle\langle x| \otimes \rho_E^x \right\|_1 . \tag{1}
$$
Intuitively, $d(X|E) \le \epsilon$ means that the distribution of $X$ is $\epsilon$-close to uniform even given $\rho_E$, i.e., $\rho_E$ gives hardly any information about $X$. A simple property of the non-uniformity which follows from its definition is that it does not change given independent information. Formally,
$$
d(X|E, D) = d(X|E) \tag{2}
$$
for any cqq-state of the form $\rho_{XED} = \rho_{XE} \otimes \rho_D$.

### 3.2 Entropic Quantities

Throughout this paper we use a number of entropic quantities. The **binary-entropy function** is defined as $h(p) := -p\log p - (1-p)\log(1-p)$, where $\log$ denotes the logarithm base 2 throughout this paper. The **von Neumann entropy** of a quantum state $\rho$ is given by
$$
H(\rho) := -Tr(\rho \log \rho) .
$$
For a bipartite state $\rho_{AB} \in \mathcal{S}(\mathcal{H}_{AB})$, we use the shorthand
$$
H(A|B) := H(\rho_{AB}) - H(\rho_B)
$$
to denote the **conditional von Neumann entropy** of the state $\rho_{AB}$ given the quantum state $\rho_B = Tr_A(\rho_{AB}) \in \mathcal{S}(\mathcal{H}_B)$. Of particular importance to us are the following quantities introduced by Renner [36]. Let $\rho_{AB} \in \mathcal{S}(\mathcal{H}_{AB})$. Then the **conditional min-entropy** of $\rho_{AB}$ relative to $B$ is defined by the following semi-definite program
$$
H_{\infty}(A|B)_{\rho} := -\log \min_{\sigma_B \in \mathcal{P}(\mathcal{H}_B)} Tr(\sigma_B) .
$$
$$
\rho_{AB} \le id_A \otimes \sigma_B
$$
For a cq-state $\rho_{XE}$ one can show [26] that the conditional min-entropy is the (negative logarithm of the) **guessing probability** $^{\text{d}}$
$$
H_{\infty}(X|E)_{\rho} = - \log P_{\text{guess}}(X|E)_{\rho}, \tag{3}
$$
where $P_{\text{guess}}(X|E)_{\rho}$ is defined as the maximum success probability of guessing $X$ by measuring the $E$-register of $\rho_{XE}$. Formally, for any (not necessarily normalized) cq-state $\rho_{XE}$, we define
$$
P_{\text{guess}}(X|E)_{\rho} := \sup_{\{M_x\}_{x \in \mathcal{X}}} \sum_{x \in \mathcal{X}} P_X(x) Tr(M_x \rho_E^x),
$$
$^{\text{d}}$Such an "operational meaning" of conditional min-entropy can also be formulated for general qq-states [26].

969 C. Schaffner, B. Terhal, and S. Wehner

where the supremum ranges over all positive-operator valued measurements (POVMs) with measurement elements $\{M_x\}_{x \in \mathcal{X}}$, i.e. $M_x \ge 0$ and $\sum_x M_x = id_E$. If all information in $E$ is classical, we recover the fact that the classical min-entropy is the negative logarithm of the average maximum guessing probability.

In our proofs we also need smooth versions of these entropic quantities. The idea is to no longer consider the min-entropy of a fixed state $\rho_{AB}$, but take the supremum over the min-entropy of states $\tilde{\rho}_{AB}$ which are close to $\rho_{AB}$, and which may have considerably larger min-entropy. In a cryptographic setting, we are often not interested in the min-entropy of a concrete state $\rho_{AB}$, but in the maximal min-entropy we can get from states in the neighborhood of $\rho_{AB}$, i.e. deviating only slightly from the real situation $\rho_{AB}$. These smooth quantities have some nice properties which are needed in our security proof. For $\epsilon \ge 0$, the $\epsilon$-smooth min-entropy of $\rho_{AB}$ is given by
$$
H_{\infty}^{\epsilon}(A|B)_{\rho} := \sup_{\tilde{\rho}_{AB} \in \mathcal{K}^{\epsilon}(\rho_{AB})} H_{\infty}(A|B)_{\tilde{\rho}},
$$
where $\mathcal{K}^{\epsilon}(\rho_{AB}) := \{\tilde{\rho}_{AB} \in \mathcal{P}(\mathcal{H}_{AB}) \mid C(\tilde{\rho}_{AB}, \rho_{AB}) \le \epsilon \text{ and } Tr(\tilde{\rho}_{AB}) \le 1\}$. If the quantum states $\rho$ are clear from the context, we drop the subscript of the entropies.

#### 3.2.1 Properties of The Conditional Smooth Min-Entropy

In our security analysis we make use of the following properties of smooth min-entropy. First, we need the chain rule whose simple proof can be found in Appendix A:

**Lemma 1 (Chain Rule)** For any ccq-state $\rho_{XYE} \in \mathcal{S}(\mathcal{H}_{XYE})$ and for all $\epsilon \ge 0$, it holds that
$$
H_{\infty}^{\epsilon}(X|YE) \ge H_{\infty}^{\epsilon}(XY|E) - \log |\mathcal{Y}|,
$$
where $|\mathcal{Y}|$ is the alphabet size of the random variable $Y$.

Secondly, we prove the additivity of the smooth conditional min-entropy (see Appendix A):

**Lemma 2 (Additivity)** Let $\rho_{AB}$ and $\rho_{A'B'}$ be two independent qq-states. For $\epsilon \ge 0$, it holds that
$$
H_{\infty}^{\epsilon^2}(AA'|BB')_{\rho} \le H_{\infty}^{\epsilon}(A|B)_{\rho} + H_{\infty}^{\epsilon}(A'|B')_{\rho} .
$$
Thirdly, adding a classical register can only increase the smooth min-entropy (see Appendix A):

**Lemma 3 (Monotonicity)** For a ccq-state $\rho_{XYE}$ and for all $\epsilon \ge 0$, it holds that
$$
H_{\infty}^{\epsilon}(XY|E) \ge H_{\infty}^{\epsilon}(Y|E).
$$
At last, we deduce a lower bound on the conditional smooth min-entropy of product states. The following theorem is a straightforward generalization of Theorem 7 in [40] (see also [36, Theorem 3.3.6]) to the case where the states are independently, but not necessarily identically distributed. The theorem states that for a large number of independent states, the conditional

970 Robust cryptography in the noisy-quantum-storage model

smooth min-entropy can be lower-bounded by the conditional Shannon entropy. We note that it is a common feature of equipartition theorems for classical or quantum information that the assumption of i.i.d. sources can be replaced by the weaker assumption of non-i.i.d. but independent sources (see Appendix A for the proof).

**Theorem 1 (adapted from [40])** For $i=1,...,n$, let $\rho_i \in \mathcal{S}(\mathcal{H}_{AB})$ be density operators. Then, for any $\epsilon > 0$,
$$
H_{\infty}^{\epsilon}(A^n|B^n)_{\otimes_{i=1}^n \rho_i} \ge \sum_{i=1}^n [H(A_i|B_i)_{\rho_i}] - \delta(\epsilon, \gamma)\sqrt{n},
$$
where, for $n \ge \frac{5}{\epsilon^2} \log \frac{2}{\epsilon^2}$, the error is given by
$$
\delta(\epsilon, \gamma) := 4 \log \frac{\gamma}{\sqrt{n}} \sqrt{\log \frac{2}{\epsilon^2}}
$$
and the single-system entropy contribution by
$$
\gamma \le 2 \max_i \sqrt{rank(\rho_{A_i})} + 1 .
$$
For the case of independent cq-states in Hilbert spaces with the same dimensions, we obtain

**Corollary 2** For $i=1,...,n$, let $\rho_{X_i B_i}$ be cq-states over (copies of) the same space $\mathcal{H}_X \otimes \mathcal{H}_B$. Then for every $\epsilon > 0$ and $n \ge \frac{5}{\epsilon^2} \log \frac{2}{\epsilon^2}$,
$$
H_{\infty}^{\epsilon}(X^n|B^n)_{\otimes_{i=1}^n \rho_{X_i B_i}} \ge \sum_{i=1}^n H(X|B)_{\rho_{X_i B_i}} - \delta n , \tag{4}
$$
where $\delta := \frac{\sqrt{\log(2/\epsilon^2)}}{n} 4 \log(2\sqrt{\text{dim } \mathcal{H}_X} + 1)$.

We use the properties of the smooth min-entropy to prove the following two lemmas. These lemmas show that the (smooth) min-entropy of two independent strings can be split.

**Lemma 4** Let $\epsilon \ge 0$, and let $\rho_{X_0 E_0}, \rho_{X_1 E_1}$ be two independent cq-states with
$$
H_{\infty}^{\epsilon}(X_0 X_1 | E_0 E_1) \ge \alpha .
$$
Additionally, let $S_0, S_1$ be classical random variables distributed over $\{0,1\}^{\ell}$. Then, there exists a random variable $D \in \{0,1\}$ such that $H_{\infty}^{\epsilon}(X_{\bar{D}} D S_{\bar{D}} | E_0 E_1) \ge \alpha/2$.

**Proof.** From the additivity of smooth min-entropy (Lemma 2) it follows that we can split the min-entropy as
$$
H_{\infty}^{\epsilon^2}(X_0|E_0)_{\rho} + H_{\infty}^{\epsilon^2}(X_1|E_1)_{\rho} \ge H_{\infty}^{\epsilon}(X_0 X_1 | E_0 E_1) \ge \alpha,
$$
and therefore, there exists $D \in \{0,1\}$ such that
$$
H_{\infty}^{\epsilon^2}(X_{\bar{D}} D S_{\bar{D}} | E_0 E_1) \ge \alpha/2,
$$
where we used the monotonicity of smooth min-entropy (Lemma 3). $\Box$

971 C. Schaffner, B. Terhal, and S. Wehner

**Lemma 5** Let $\epsilon \ge 0$. Let $\rho_{XE} = \otimes_{i=0}^{m-1} \rho_{X_i E_i}$ be a cq-state consisting of $m$ independent cq-substates such that $H_{\infty}^{\epsilon}(X_i X_j | E) \ge \alpha$ for all $i \ne j$. Then there exists a random variable $V$ over $\{1,...,m\}$ such that for any $v \in \{1,...,m\}$ with $P[V \ne v] > 0$
$$
H_{\infty}^{\epsilon^2}(X_{\bar{V}} | E_V, V \ne v)_{\rho} \ge \alpha/2 - \log(m).
$$

**Proof.** Let $V \in \{0, ..., m-1\}$ be the index which achieves the minimum of $H_{\infty}^{\epsilon^2}(X_i|E_i)$, i.e. $H_{\infty}^{\epsilon^2}(X_{\bar{V}}|E_V) = \min_i H_{\infty}^{\epsilon^2}(X_{\bar{i}}|E_i)$. By the additivity of smooth min-entropy (Lemma 2), we have for all $v \ne V$,
$$
\alpha \le H_{\infty}^{\epsilon^2}(X_{\bar{V}} X_V | E)_{\rho} \le H_{\infty}^{\epsilon^2}(X_{\bar{V}}|E_{\bar{V}})_{\rho} + H_{\infty}^{\epsilon^2}(X_V|E_V)_{\rho} .
$$
It follows that $H_{\infty}^{\epsilon^2}(X_{\bar{V}}|E_V, V \ne v)_{\rho} \ge \alpha/2$. The chain rule (Lemma 1) then leads to the claim.
$\Box$

### 3.3 Tools

We also require the following technical results. This lemma is well-known, see [2] or [32] for a proof.

**Lemma 6 (Chernoff's inequality)** Let $X_1,..., X_n$ be identically and independently distributed random variables with Bernoulli distribution, i.e. $X_i=1$ with probability $p$ and $X_i=0$ with probability $1-p$. Then $S := \sum_{i=1}^n X_i$ has a binomial distribution with parameters $(n, p)$ and it holds that
$$
Pr \left[ |S - pn| > \epsilon n \right] \le 2e^{-2\epsilon^2 n} .
$$

**Privacy Amplification** The OT protocol makes use of two-universal hash functions. These hash functions are used for privacy amplification similar as in quantum key distribution. A class $\mathcal{F}$ of functions $f : \{0,1\}^n \to \{0,1\}^l$ is called **two-universal**, if for all $x \ne y \in \{0,1\}^n$ and $f \in \mathcal{F}$ chosen uniformly at random from $\mathcal{F}$, we have $Pr[f(x) = f(y)] \le 2^{-l}$ [6]. The following theorem expresses how the application of hash functions can increase the privacy of a random variable $X$ given a quantum adversary holding $\rho_E$, the function $F$ and a classical random variable $U$:

**Theorem 3 ([36, 12])** Let $\mathcal{F}$ be a class of two-universal hash functions from $\{0,1\}^n$ to $\{0,1\}^{\ell}$. Let $F$ be a random variable that is uniformly and independently distributed over $\mathcal{F}$, and let $\rho_{XUE}$ be a ccq-state. Then, for any $\epsilon \ge 0$,
$$
d(F(X)|F, U, E) \le 2^{-\frac{1}{2} (H_{\infty}^{\epsilon}(X|UE) - \ell)} + \epsilon .
$$

## 4 1-2 Oblivious Transfer

### 4.1 Security Definition and Protocol

In this section we prove the security of a randomized version of 1-2 OT (Theorem 4) from which we can easily obtain 1-2 OT. In such a randomized 1-2 OT protocol, Alice does not input two strings herself, but instead receives two strings $S_0, S_1 \in \{0,1\}^{\ell}$ chosen uniformly at

972 Robust cryptography in the noisy-quantum-storage model

random. Randomized OT (ROT) can easily be converted into OT. After the ROT protocol is completed, Alice uses her strings $S_0, S_1$ obtained from ROT as one-time pads to encrypt her original inputs $\hat{S}_0$ and $\hat{S}_1$, i.e. she sends an additional classical message consisting of $\hat{S}_0 \oplus S_0$ and $\hat{S}_1 \oplus S_1$ to Bob. Bob can retrieve the message of his choice by computing $S_C \oplus (\hat{S}_C \oplus S_C) = \hat{S}_C$. He stays completely ignorant about the other message $\hat{S}_{\bar{C}}$ since he is ignorant about $S_{\bar{C}}$. The security of a quantum protocol implementing ROT is formally defined in [12] and justified in [17] (see also [43]).

**Definition 1** An $\epsilon$-secure 1-2 ROT is a protocol between Alice and Bob, where Bob has input $C \in \{0,1\}$, and Alice has no input.

*   (**Correctness**) If both parties are honest, then for any distribution of Bob's input $C$, Alice gets outputs $S_0, S_1 \in \{0,1\}^{\ell}$ which are $\epsilon$-close to uniform and independent of $C$ and Bob learns $Y = S_C$ except with probability $\epsilon$.
*   (**Security against dishonest Alice**) If Bob is honest and obtains output $Y$, then for any cheating strategy of Alice resulting in her state $\rho_A$, there exist random variables $S'_0$ and $S'_1$ such that $Pr[Y = S'_C] \ge 1 - \epsilon$ and $C$ is independent of $S'_0, S'_1$ and $\rho_A^{\dag}$.
*   (**Security against dishonest Bob**) If Alice is honest, then for any cheating strategy of Bob resulting in his state $\rho_B$, there exists a random variable $D \in \{0,1\}$ such that $d(S_{\bar{D}} S_D | D \rho_B) \le \epsilon$.

For convenience, we choose $\{+, \times\}$ instead of $\{0, 1\}$ as domain of Bob's choice bit $C$. We consider the same protocol for ROT as in [12].

**Protocol 1 ([12]) 1-2 ROT**

1.  Alice picks $X \in_{R} \{0,1\}^n$ and $\Theta \in_{R} \{+, \times\}^n$. Let $I_b = \{i \mid \Theta_i = b\}$ for $b \in \{+, \times\}$. At time $t=0$, she sends $|X_i\rangle_{\Theta_i}, ..., |X_n\rangle_{\Theta_n}$ to Bob.
2.  Bob measures all qubits in the basis corresponding to his choice bit $C \in \{+, \times\}$. He obtains outcome $X' \in \{0,1\}^n$.
3.  Alice picks two hash functions $F_+, F_{\times} \in_{R} \mathcal{F}$, where $\mathcal{F}$ is a class of two-universal hash functions. At the reveal time $t=T_{\text{rev}}$, she sends $I_+, I_{\times}, F_+, F_{\times}$ to Bob. Alice outputs $S_+ = F_+(X|_{I_+})$ and $S_{\times} = F_{\times}(X|_{I_{\times}})$.$^{\text{f}}$
4.  Bob outputs $S_C = F_C(X|_{I_C}')$.

### 4.2 Security Analysis

We show in this section that Protocol 1 is secure according to Definition 1, in case the dishonest receiver is restricted to individual-storage attacks.

$^{\text{e}}$Existence of the random variables $S'_0, S'_1$ has to be understood as follows: given the cq-state $\rho_{YA}$ of honest Bob and dishonest Alice, there exists a cccq-state $\rho_{Y S'_0 S'_1 A}$ such that tracing out the registers of $S'_0, S'_1$ yields the original state $\rho_{YA}$ and the stated properties hold.
$^{\text{f}}$If $X|_{I_b}$ is less than $n$ bits long Alice pads the string $X|_{I_b}$ with $0$'s to get an $n$ bit-string in order to apply the hash function to $n$ bits.

973 C. Schaffner, B. Terhal, and S. Wehner

[IMAGE: Fig. 2. Bob performs a partial measurement $P_i$, followed by noise $N_i$, and outputs a guess bit $x_g$ depending on his classical measurement outcome, the remaining quantum state, and the additional basis information. The diagram shows an input state going through $P_i$ to output a `classical` outcome (basis information) and a `quantum` state. The quantum state goes through a noise operation $N_i$. Both outputs are inputs to a `Measure` operation which outputs $x_g$. The text "basis information" is below the $P_i$ operation.]

**Correctness** First of all, note that it is clear that the protocol fulfills its task correctly. Bob can determine the string $X'_{I_C}$ (except with negligible probability $2^{-n}$ the set $I_C$ is non-empty) and hence obtains $S_C$. Alice's outputs $S_+, S_{\times}$ are perfectly independent of each other and of $C$.

**Security against Dishonest Alice** Security holds in the same way as shown in [12]. As the protocol is non-interactive, Alice never receives any information from Bob at all, and Alice's input strings can be extracted by letting her interact with an unbounded receiver.

**Security against Dishonest Bob** Proving that the protocol is secure against Bob requires more work. Our goal is to show that there exists a $D \in \{+, \times\}$ such that Bob is completely ignorant about $S_{\bar{D}}$.

Recall that in round $i$, honest Alice picks $X_i \in_R \{0,1\}$ and $\Theta_i \in_R \{+, \times\}$ and sends $|X_i\rangle_{\Theta_i}$ to dishonest Bob. Bob can subsequently do a partial measurement to obtain the classical outcome $K_i$ and store the remaining quantum state in register $E_i$, which is then subject to noise (see Figure 2). It is important to note that Bob's initial partial measurement does not depend on the basis information $\Theta$. Since we are modeling individual-storage attacks, the overall state (as viewed by Bob) for Alice and Bob right before time $T_{\text{rev}}$ is of the form
$$
\rho_{X \Theta K E} = \bigotimes_{i=1}^n \rho_{X_i \Theta_i K_i E_i} ,
$$
with
$$
\rho_{X_i \Theta_i K_i E_i} = \frac{1}{4} \sum_{x_i, \theta_i, k_i} P_{k_i|x_i, \theta_i} |x_i\rangle\langle x_i| \otimes |\theta_i\rangle\langle \theta_i| \otimes |k_i\rangle\langle k_i| \otimes \mathcal{N}_i \left( \rho_{x_i \theta_i}^{k_i} \right) . \tag{5}
$$
where we use $X_i$ to denote Alice's system corresponding to her choice of bit $x_i$, $\Theta_i$ for the system corresponding to her choice of basis $\theta_i$, and $K_i$ and $E_i$ for Bob's systems corresponding to the classical outcome $k_i$ (with probability $P_{k_i|x_i, \theta_i}$) of his partial measurement and his remaining quantum system respectively.

It is clear that a dishonest receiver will have some uncertainty about the bit $X_i$, given that he either measured the register $E$ without the correct basis information and that storage noise occurred on the post-measurement quantum state. To formalize this uncertainty, let us call $t$ an **uncertainty lower bound** on the conditional Shannon entropy if, for all $i = 1, ..., n$, we have
$$
H(X|\Theta K E) = H(\rho_{X \Theta K E}) - H(\rho_{\Theta K E_i}) \ge t. \tag{6}
$$

974 Robust cryptography in the noisy-quantum-storage model

The parameter $t$ thereby depends on the specific kind of noise in the quantum storage. In Section 5, we evaluate the uncertainty lower-bound $t$ for the case of depolarizing noise.

The following theorem shows that as long as $\ell \le t n/4$, the protocol is secure except with probability $\epsilon$. Since we are performing 1-out-of-2 oblivious transfer of $\ell$-bit strings, $\ell$ corresponds to the "amount" of oblivious transfer we can perform for a given security parameter $\epsilon$ and number of qubits $n$. In QKD, $\ell$ corresponds to the length of the key generated.

**Theorem 4** Protocol 1 is $2\epsilon$-secure against a dishonest receiver Bob according to Definition 1, if $n \ge \frac{5}{\epsilon^2} \log \frac{2}{\epsilon^2}$ and
$$
\ell \le \frac{1}{4} \left( t - \delta \right) n + \frac{1}{2} - \log \left( \frac{1}{\epsilon} \right) ,
$$
where $\delta = 8\sqrt{\log(2/\epsilon^4)}/n$, and $t$ is the uncertainty lower bound on the conditional Shannon entropy fulfilling Eq. (6).

**Proof.** We need to show the existence of a binary random variable $D$ such that $S_{\bar{D}}$ is $\epsilon$-close to uniform from Bob's point of view. As noted above, the overall state of Alice and Bob before time $T_{\text{rev}}$ has a product form. After time $T_{\text{rev}}$, dishonest Bob holds the classical registers $\Theta, K$, the quantum register $E$ as well as classical information about the hash functions $F_+, F_{\times}$. To prove security, we first lower-bound Bob's uncertainty about $X$ in terms of min-entropy, use Lemma 4 to obtain $D$ and then apply the privacy amplification theorem.

First of all, we know from Corollary 2 that the smooth min-entropy of an $n$-fold tensor state is roughly equal to $n$ times the von Neumann entropy of its substates. Hence, applying Corollary 2 to our setting with $B_i := \Theta_i K_i E_i$ and $\log(2\sqrt{\text{dim } \mathcal{H}_{X_i}} + 1) = \log(2\sqrt{2} + 1) \le 2$ we obtain for $n \ge \frac{5}{\epsilon^2} \log \frac{2}{\epsilon^2}$ that
$$
H_{\infty}^{\epsilon}(X^n|\Theta K E) \ge \sum_{i=1}^n H(X_i|\Theta_i K_i E_i) - \delta n \ge (t - \delta)n,
$$
with $\delta = 8\sqrt{\log(2/\epsilon^4)}/n$. We used Equation (4) in the first inequality and the last follows by Definition (6) of the uncertainty bound $t$.

For ease of notation, we use $X_+$ and $X_{\times}$ to denote $X|_{I_+}$ and $X|_{I_{\times}}$, the sequences of bits $X_i$ where $\Theta_i = +$ and $\Theta_i = \times$, respectively. From $H_{\infty}^{\epsilon^2}(X_+ X_{\times} | \Theta K E)_{\rho} \ge (t - \delta)n$ and Lemma 4 it follows that $D \in \{+, \times\}$ exists such that
$$
H_{\infty}^{\epsilon^2}(X_{\bar{D}} S_D | \Theta K E)_{\rho} \ge \frac{(t - \delta)n}{2}.
$$
The rest of the security proof is analogous to the derivation in [12]: It follows from the chain rule (Lemma 1) and the monotonicity (Lemma 3) of the smooth min-entropy that
$$
H_{\infty}^{\epsilon^2}(X_{\bar{D}}|\Theta D S_{\bar{D}} K E) \ge H_{\infty}^{\epsilon^2}(X_{\bar{D}} D S_{\bar{D}}|\Theta K E) - (\ell+1)
$$
$$
\ge \frac{(t - \delta)n}{2} - 1 - \ell.
$$
The privacy amplification Theorem 3 yields
$$
d(F_{\bar{D}}(X_{\bar{D}}) | \mathcal{F}_{\bar{D}}, D S_{\bar{D}} K E) \le 2^{-\frac{1}{2}(\frac{(t - \delta)n}{2} - 1 - 2\ell)} + \epsilon \tag{7}
$$

975 C. Schaffner, B. Terhal, and S. Wehner

which is smaller than $2\epsilon$ as long as
$$
\frac{(t - \delta)n}{4} + \frac{1}{2} - \ell \ge \log \left( \frac{1}{\epsilon} \right),
$$
from which our claim follows.
$\Box$

We note that one can improve on the extractable length $\ell$ by using a quantum version of Wullschleger's distributed-privacy-amplification theorem [45]. Since this technique is specific to oblivious transfer and does not apply to our extension to the case of secure identification, we do not go into the details here.

## 5 Example: Depolarizing Noise

In this section, we consider the case when Bob's storage is affected by depolarizing noise as described by the quantum operation
$$
\mathcal{N}(\rho) = r\rho + (1 - r)\frac{id}{2} . \tag{8}
$$
Depolarization noise will leave the input state $\rho$ intact with probability $r$, but replace it with the completely mixed state with probability $1-r$. In order to give explicit security parameters for this setting, our goal is to prove an uncertainty bound $t$ for the conditional von Neumann entropy $H(X_i|\Theta_i K_i E_i)$ as in Eq. (6). Exploiting the symmetries in the setting, we derive in Appendix B the following result. We drop the index $i$ in this Theorem.

**Theorem 5** Let $\mathcal{N}$ be the depolarizing quantum operation given by Eq. (8) and let $H(X|\Theta K E)$ be the conditional von Neumann entropy of one qubit. Then
$$
H(X|\Theta K E) \ge
\begin{cases}
h \left( \frac{1+r}{2} \right) & \text{for } r \ge \tilde{r}, \\
1/2 & \text{for } r < \tilde{r},
\end{cases}
$$
where $\tilde{r} := 2h^{-1}(1/2) - 1 \approx 0.7798$.

Our result shows that when the probability of retaining the input state $\rho$ is small, $r < 0.7798$, the best attack for Bob is to measure everything right away in the computational basis. For this measurement, we have $H(X|\Theta K E) \ge 1/2$. If the depolarizing rate is low, i.e. $r > 0.7798$, our result says that the best strategy for Bob is to simply store the qubit as is.

Our result may seem contradictory to our previous error trade-off obtained in [41], where Bob's best strategy was to either store the qubit as is or measure it in the Breidbart basis depending on the amount of depolarizing noise. Note, however, that the quantity we optimize in this work is the **von Neumann entropy** and not the **guessing probability** considered in [41]. This phenomenon is similar to the setting of QKD, where Eve's strategy that optimizes her guessing probability is different from the one that optimizes the entropy [18]. In general, the von Neumann entropy is larger than the min-entropy (which corresponds to the guessing probability). Corollary 2 provides the explanation why the von Neumann entropy is the relevant quantity in the setting of individual-storage attacks.

976 Robust cryptography in the noisy-quantum-storage model

## 6 Robust Oblivious Transfer

In a practical setting, honest Alice and honest Bob are not able to perform perfect quantum operations or transmit qubits through a noiseless channel. We must therefore modify the ROT protocol to make it robust against noise for the honest parties. The protocol we consider is a small modification of the protocol considered in [39]. The idea is to let Alice send additional error-correcting information which can help honest Bob to retrieve $S_C$ as desired. The main difficulty in the analysis of the extended protocol is the fact that we have to assume a worst-case scenario: If Bob is dishonest, we give him access to a perfect noise-free quantum channel with Alice and he only experiences noise during storage.

We can divide the noise on the channel into two categories, which we consider separately: First, we consider **erasure noise** (in practice corresponding to photon loss) during preparation, transmission and measurement of the qubits by the honest parties. Let $1-P_{\text{erase}}$ be the total probability for an honest Bob to measure and detect a photon in the $\{+, \times\}$-basis given that an honest Alice prepares a weak pulse in her lab and sends it to him. The probability $P_{\text{erase}}$ is determined, among other things, by the mean photon number in the pulse, the loss on the channel and the quantum efficiency of the detector. In our protocol we assume that the erasure rate $P_{\text{erase}}$ is independent for every pulse and independent of whether qubits were encoded or measured in the $+$ or $\times$-basis whenever Bob is honest. This assumption is necessary to guarantee the correctness and the security against a cheating Alice only. Fortunately, this assumption is well matched with the possible physical implementations of the protocol.

Any other noise source during preparation, transmission and measurement can be characterized as an effective classical noisy channel resulting in the output bits $X'$ that Bob obtains at Step 3 of Protocol 2. For simplicity, we model this compound noise source as a classical **binary symmetric channel** acting independently on each bit of $X$. Typical noise sources for polarization-encoded qubits are depolarization during transmission, dark counts in Bob's detector and misaligned polarizing beam-splitters. Let the effective bit-error probability, called the **quantum bit-error rate** in quantum key distribution, of this binary symmetric channel be $P_{\text{error}} < 1/2$.

### 6.1 Protocol

In this section we present the modified version of the ROT protocol. Before engaging in the actual protocol, Alice and Bob agree on a small enough security-error probability $\epsilon > 0$ that they are willing to tolerate. In addition, they determine the system parameters $P_{\text{erase}}$ and $P_{\text{error}}$ similarly to Step 1 of the protocol in [3]. Furthermore, they agree on a family $\{\mathcal{C}_n\}$ of linear error-correcting codes of length $n$ capable of efficiently correcting $n P_{\text{error}}$ errors [8]. For any string $x \in \{0,1\}^n$, error-correction is done by sending the syndrome information $\text{syn}(x)$ to Bob from which he can correctly recover $x$ if he holds an output $x' \in \{0,1\}^n$ obtained by flipping each bit of $x$ independently with probability $P_{\text{error}}$. It is known that for large enough $n$, the code $\mathcal{C}_n$ can be chosen such that its rate is arbitrarily close to $1 - h(P_{\text{error}})$ and the syndrome length (the number of parity check bits) is asymptotically bounded by $|\text{syn}(x)| \approx h(P_{\text{error}})n$ [8]. We assume that the players have synchronized clocks. In each time slot, Alice sends one qubit to Bob.

**Protocol 2 Robust 1-2 ROT($\mathcal{C}, t, \epsilon$)**

1.  Alice picks $X \in_{R} \{0,1\}^n$ and $\Theta \in_{R} \{+, \times\}^n$.

977 C. Schaffner, B. Terhal, and S. Wehner

2.  For $i=1,...,n$: In time slot $t=i$, Alice sends $|X_i\rangle_{\Theta_i}$ as a phase- or polarization-encoded weak pulse of light to Bob.
3.  In each time slot, Bob measures the incoming qubit in the basis corresponding to his choice bit $C \in \{+, \times\}$ and records whether he detects a photon or not. He obtains some bit-string $X' \in \{0,1\}^m$ with $m \le n$.
4.  Bob reports back to Alice in which time slots he received a qubit. Alice restricts herself to the set of $m \le n$ bits that Bob did not report as missing. Let this set of qubits be $\mathcal{S}_{\text{remain}}$ with $|\mathcal{S}_{\text{remain}}| = m$.
5.  Let $I_b = \{i \in \mathcal{S}_{\text{remain}} \mid \Theta_i = b\}$ for $b \in \{+, \times\}$ and let $m_b = |I_b|$. Alice aborts the protocol if either $m_+$ or $m_{\times}$ are outside the interval $[(1-P_{\text{erase}}-\epsilon)n/2, (1-P_{\text{erase}}+\epsilon)n/2]$. If this is not the case, Alice picks two two-universal hash functions $F_+, F_{\times} \in_{R} \mathcal{F}$. At time $t = n + T_{\text{rev}}$, Alice sends $I_+, I_{\times}, F_+, F_{\times}$, and the syndromes $\text{syn}(X|_{I_+})$ and $\text{syn}(X|_{I_{\times}})$ according to codes of appropriate length $m_b$ to Bob. Alice outputs $S_+ = F_+(X|_{I_+})$ and $S_{\times} = F_{\times}(X|_{I_{\times}})$.
6.  Bob uses $\text{syn}(X|_{I_C})$ to correct the errors on his output $X'_{I_C}$. He obtains the corrected bit-string $X_{\text{cor}}$ and outputs $S'_C = F_C(X_{\text{cor}})$.

### 6.2 Security Analysis

**Correctness** By assumption, $P_{\text{erase}}$ is independent for every pulse and independent of the basis in which Alice sent the qubits. Thus, by Chernoff's Inequality (Lemma 6), $\mathcal{S}_{\text{remain}}$ is, except with negligible probability, a random subset of $m$ qubits independent of the value of $\Theta$ and such that $(1-P_{\text{erase}}-\epsilon)n \le m \le (1-P_{\text{erase}}+\epsilon)n$. This implies that in Step 5 the protocol is aborted with a probability only exponentially small in $n$. The codes are chosen such that Bob can decode except with negligible probability. These facts imply that if both parties are honest, the protocol is correct (i.e. $S_C = S'_C$) with exponentially small probability of error.

**Security against Dishonest Alice** Even though in this scenario Bob does communicate to Alice, the information about which qubits were erased is (by assumption) independent of the basis in which he measured and thus of his choice bit $C$. Hence Alice does not learn anything about his choice bit $C$. Her input strings can be extracted as in the analysis of Protocol 1.

**Security against Dishonest Bob** We prove the following:

**Theorem 6** Protocol 2 is secure against a dishonest receiver Bob with error of at most $2\epsilon$, if $n \ge \frac{5}{\epsilon^2} \log \frac{2}{\epsilon^2}$ and
$$
\ell \le \left( t - \delta - h(P_{\text{error}}) \right) \frac{(1 - P_{\text{erase}})n}{2} + \frac{1}{2} - \log \left( \frac{1}{\epsilon} \right), \tag{9}
$$
where $\delta = 8\sqrt{\log(2/\epsilon^4)}/((1-P_{\text{erase}}-\epsilon)n)$, and $t$ is the uncertainty bound on the conditional Shannon entropy fulfilling Eq. (6).

978 Robust cryptography in the noisy-quantum-storage model

**Proof (Sketch).** First of all, we note that Bob can always make Alice abort the protocol by reporting back an insufficient number of received qubits. If Alice does not abort the protocol in Step 5, we have that $(1-P_{\text{erase}}-\epsilon)n/2 \le m_+, m_{\times} \le (1-P_{\text{erase}}+\epsilon)n/2$. We define $D$ as in the security proof of Protocol 1. The security analysis is the same, but we need to subtract the amount of error correcting information $|\text{syn}(X|_{I_{\bar{D}}})|$ from the entropy of the dishonest receiver. If Alice does not abort the protocol in Step 5, we have that $|\text{syn}(X|_{I_{\bar{D}}})| \le h(P_{\text{error}}) (1-P_{\text{erase}}+\epsilon)n/2$. Hence,
$$
H_{\infty}^{\epsilon^2}(X_{\bar{D}}|\mathcal{F}_{\bar{D}} D S_{\bar{D}} \text{syn}(X|_{I_{\bar{D}}}) K E)
$$
$$
\ge H_{\infty}^{\epsilon^2}(X_{\bar{D}} D S_{\bar{D}} \text{syn}(X|_{I_{\bar{D}}})|\mathcal{F}_{\bar{D}} K E) - (\ell+1) - h(P_{\text{error}}) m/2
$$
$$
\ge (t - \delta)(1 - P_{\text{erase}} - \epsilon)n/2 - (\ell+1) - h(P_{\text{error}})(1 - P_{\text{erase}} + \epsilon)n/2 - 1 - \ell
$$
$$
\ge \left( t - \delta - h(P_{\text{error}}) \right) \frac{(1 - P_{\text{erase}})n}{2} - \frac{(t - \delta + h(P_{\text{error}}))\epsilon n}{2} - 1 - \ell ,
$$
$\approx_{\le 2\epsilon}$

where $(t - \delta + h(P_{\text{error}})) \le 2$ since $t \le 1$. Using this inequality to bound the security parameter via the privacy amplification Theorem 3 gives the claimed bound on $\ell$, Eq. (9). $\Box$

**Remarks** Note that it is only possible to choose a code $\mathcal{C}$ that satisfies the stated parameters **asymptotically**. For a real---finite block-length---code, deviations from this asymptotic behavior need to be taken into account. For the sake of clarity we have omitted these details in the analysis above. Secondly, the dishonest parties need to obtain an estimate for $P_{\text{error}}$ prior to the protocol. One approach would be to use a worst case estimate based what is possible with present-day technology. Alternatively, one could follow Step 1 of the protocol in [3] as suggested above. However, one needs to analyze this estimation procedure in a practical setting. Thirdly, when weak photon sources are used in this protocol, one needs to analyze the security threat due to the presence of multi-photon emissions which Bob can exploit in photon-number-splitting attacks as in QKD. See [41] for a first discussion of the effect of such attacks.

### 6.3 Depolarizing Noise

As an example, we again consider the security trade-off when Bob's storage is affected by depolarizing noise. It follows directly from Theorems 3, 6 and 5 that

**Corollary 7** Let $\mathcal{N}$ be the depolarizing quantum operation given by Eq. (8). Then the protocol can be made secure (by choosing a sufficiently large $n$) as long as
$$
h \left( \frac{1+r}{2} \right) > h(P_{\text{error}}) \quad \text{for } r \ge \tilde{r},
$$
$$
1/2 > h(P_{\text{error}}) \quad \text{for } r < \tilde{r},
$$
where $\tilde{r} := 2h^{-1}(1/2) - 1 \approx 0.7798$.

979 C. Schaffner, B. Terhal, and S. Wehner

Hence, our security parameters are greatly improved from our previous analysis [41]. For $r < \tilde{r}$ we can now obtain security as long as the quantum bit error rate $P_{\text{error}} \lesssim 0.11$, compared to $0.029$ before. For the case of $r \ge \tilde{r}$, we can essentially show security as long as the noise on the channel is strictly less than the noise in Bob's quantum storage. Note that we cannot hope to construct a protocol that is both correct and secure when the noise of the channel exceeds the noise in Bob's quantum storage. However, it remains an open question whether it is possible to construct a protocol or improve the analysis of the current protocol such that security can be achieved even for very small $n$.

Corollary 7 puts a restriction on the noise rate of the honest protocol. Yet, since our protocols are particularly interesting at short distances (e.g. in the case of secure identification we describe below), we can imagine free-space implementations over very short distances such that depolarization noise during transmission is negligible and the main noise source is due to Bob's honest measurements.

In the near-future, if good photonic memories become available (see e.g. [23, 4, 7, 15, 37, 34] for recent progress), we may anticipate that storing the qubit is a better attack than a direct measurement. Note, however, that we are free in our protocol to stretch the reveal time $T_{\text{rev}}$ between Bob's reception of the qubits and his reception of the classical basis information, say, to seconds, which means that one has to consider the overall noise rate on a qubit that is stored for seconds.

In terms of long-term security, fault-tolerant photonic computation (e.g., with the KLM scheme [25]) might allow a dishonest Bob to encode the incoming quantum information into a fault-tolerant quantum memory. Such an encoding would guarantee that the effective noise rate in storage can be made arbitrarily small. The encoding of a single unknown state is **not** a fault-tolerant quantum operation however. Hence, even in the presence of a quantum computer, there is a residual storage noise rate due to the unprotected encoding operation. The question of security then becomes a question of a trade-off between this residual noise rate versus the intrinsic noise rate for honest parties. Intuitively, it might be possible to arrange the setting such that tasks of honest players are always technically easier (and/or cheaper) to perform than the ones for dishonest players. Possibly, this intrinsic gap can be exploited for cryptographic purposes. The current paper can be appreciated as a first step in this direction.

## 7 Extension to Secure Identification

In this section, we like to point out how our model of noisy quantum storage with individual-storage attacks also applies to protocols that achieve more advanced tasks such as secure **identification**. The protocol from [13] allows a user $U$ to identify him/herself to a server $S$ by means of a personal identification number (PIN). This task can be achieved by securely evaluating the equality function on the player's inputs. In other words, both $U$ and $S$ input passwords $W_U$ and $W_S$ into the protocol and the server learns as output whether $W_U = W_S$ or not. The protocol proposed in [13] is secure against an unbounded user $U$ and a quantum-memory bounded server $S$ in the sense that it is guaranteed that if a dishonest player starts with quantum side information which is uncorrelated with the honest player's password $W$, the

980 Robust cryptography in the noisy-quantum-storage model

only thing the dishonest player can do is guess a possible $W'$ and learn whether $W = W'$ or not while not learning anything more than this mere bit of information about the honest user's password $W$. This protocol can also be (non-trivially) extended to additionally withstand man-in-the-middle attacks.

The security proof against a quantum-memory bounded dishonest server (and man-in-the-middle attacks) relies heavily on the uncertainty relation first derived in [12] and used for proving the security of 1-2 OT. This uncertainty relation guarantees a lower bound on the smooth min-entropy of the encoded string $X$ from the dishonest player's point of view. As we establish a similar type of lower bound (Cor. 2 and Eq. (6)) on the smooth min-entropy in the noisy-storage model, the security proof for the identification scheme (and its extension) translates to our model.

In terms of the proof of Proposition 3.1 of [13], the pair $X_i, X_j$ has essentially $t \cdot d$ bits of min-entropy given $\Theta, K$, and $E$, where $t$ is the uncertainty lower bound on the conditional Shannon entropy from Eq. (6) and $d$ is the minimal distance of the code used in the identification scheme. Lemma 5 implies that there exists $W'$ (called $V$ in Lemma 5) such that if $W \ne W'$ then $X_{\bar{W}}$ has essentially $td/2 - \log(m)$ bits of min-entropy given $W, W', \Theta, K, E$. Privacy amplification then guarantees that $F(X_{\bar{W}})$ is $\epsilon'$-close to uniform and independent of $F, W, W', \Theta, K, E$, conditioned on $W \ne W'$, where $\epsilon' \approx 2^{-\frac{1}{2}(\frac{td}{2} - \log(m) - \ell)}$. Security against a dishonest server with noisy quantum storage follows as in [13] for an error parameter $\epsilon$ which is exponentially small in $td - 2\log(m) - 2\ell$.

## 8 Conclusion

We have obtained improved security parameters for oblivious transfer in the noisy-quantum-storage model. Yet, it remains to prove security against general coherent noisy attacks. The problem with analyzing a coherent attack of Bob described by some quantum operation $\mathcal{S}$ affecting all his incoming qubits is not merely a technical one: one first needs to determine a realistic noise model in this setting. Symmetrizing the protocol as in the proof of QKD [36] and using de Finetti type arguments does not immediately work here. However, one can analyze a specific type of coherent noise, one that essentially corresponds to an eavesdropping attack in QKD. Note that the 1-2 OT protocol can be seen as two runs of QKD interleaved with each other. The strings $f(x|_{I_+})$ and $f(x|_{I_{\times}})$ are then the two keys generated. The noise must be such that it leaves Bob with exactly the same information as the eavesdropper Eve in QKD. In this case, it follows from the security of QKD that the dishonest Bob (learning exactly the same information as the eavesdropper Eve) does not learn anything about the two keys.

Clearly, there is a strong relation between QKD and the protocol for 1-2 OT, and one may wonder whether other QKD protocols can be used to perform oblivious transfer in our model. Intuitively, this is indeed the case, but it remains to evaluate explicit parameters for the security of the resulting protocols.

It will be interesting to extend our results to a security analysis of a noise-robust protocol in a realistic physical setting, where, for example, the use of weak laser pulses allows the possibility of photon-number-splitting attacks. Such a comprehensive security analysis has been carried out in [20] for quantum key distribution.

981 C. Schaffner, B. Terhal, and S. Wehner

## Acknowledgments

We thank Robert König and Renato Renner for useful discussions about the additivity of the smooth min-entropy and the permission to include Lemma 2. CS is supported by EU fifth framework project QAP IST 015848 and the NWO VICI project 2004-2009. SW is supported by NSF grant number PHY-04056720.

## References

1.  G. Alexei, N. Langford, and M. Nielsen. Distance measures to compare real and ideal quantum processes. Physical Review A, 71(6):062310, 2005.
2.  N. Alon and J. Spencer. The Probabilistic Method. Series in Discrete Mathematics and Optimization. Wiley-Interscience, 2nd edition, 2000.
3.  C. H. Bennett, G. Brassard, C. Crépeau, and M.-H. Skubiszewska. Practical quantum oblivious transfer. In CRYPTO '91: Proceedings of the 11th Annual International Cryptology Conference on Advances in Cryptology, pages 351–366. Springer-Verlag, 1992.
4.  A. D. Boozer, A. Boca, R. Miller, T. E. Northup, and H. J. Kimble. Reversible state transfer between light and a single trapped atom, 2007. quant-ph/0702248.
5.  S. Boyd and L. Vandenberghe. Convex Optimization. Cambridge University Press, 2004.
6.  J. L. Carter and M. N. Wegman. Universal classes of hash functions. Journal of Computer and System Sciences, 18:143–154, 1979.
7.  T. Chanelière, D.N. Matsukevich, S.D. Jenkins, S.-Y. Lan, T.A.B. Kennedy, and A. Kuzmich. Storage and retrieval of single photons between remote quantum memories. Nature, 438:pp. 833–836, 2005.
8.  C. Crépeau. Efficient cryptographic protocols based on noisy channels. In *Advances in Cryptology - Proceedings of EUROCRYPT '97*, 1997.
9.  C. Crépeau and J. Kilian. Achieving oblivious transfer using weakened security assumptions. In *Proceedings of 29th IEEE FOCS*, pages 42–52, 1988.
10. C. Crépeau, K. Morozov, and S. Wolf. Efficient unconditional oblivious transfer from almost any noisy channel. In *International Conference on Security in Communication Networks (SCN)*, volume 4 of Lecture Notes in Computer Science, pages 47–59, 2004.
11. I. B. Damgård, S. Fehr, L. Salvail, and C. Schaffner. Cryptography in the Bounded-Quantum-Storage Model. In *Proceedings of 46th IEEE FOCS*, pages 449–458, 2005.
12. I. B. Damgård, S. Fehr, R. Renner, L. Salvail, and C. Schaffner. A tight high-order entropic quantum uncertainty relation with applications. In *Advances in Cryptology - CRYPTO '07*, volume 4622 of Lecture Notes in Computer Science, pages 360–378. Springer-Verlag, 2007.
13. I. B. Damgård, S. Fehr, L. Salvail, and C. Schaffner. Secure identification and QKD in the bounded-quantum-storage model. In *Advances in Cryptology - CRYPTO '07*, volume 4622 of Lecture Notes in Computer Science, pages 342–359. Springer-Verlag, 2007.
14. I. B. Damgård, S. Fehr, L. Salvail, and C. Schaffner. Cryptography in the bounded-quantum-storage model. SIAM Journal on Computing, 37(6):1865–1890, 2008.
15. M.D. Eisaman, A. André, F. Massou, M. Fleischauer, A.S. Zibrov, and M. D. Lukin. Electromagnetically induced transparency with tunable single-photon pulses. Nature, 438:pp. 837–841, 2005.
16. S. Even, O. Goldreich, and A. Lempel. A randomized protocol for signing contracts. *Communications of the ACM*, 28(6):637–647, 1985.
17. S. Fehr and C. Schaffner. Composing quantum protocols in a classical environment. In *Theory of Cryptography - TCC '09*, volume 5444 of Lecture Notes in Computer Science, pages 350–367. Springer, 2009.
18. N. Gisin, G. Ribordy, W. Tittel, and H. Zbinden. Quantum cryptography. *Reviews of Modern Physics*, 74:145–195, 2002.
19. O. Goldreich and R. Vainish. How to solve any protocol problem — an efficiency improvement. In *Advances in Cryptology - CRYPTO '87*, volume 293 of Lecture Notes in Computer Science,

982 Robust cryptography in the noisy-quantum-storage model

pages 73–86. Springer, 1988.
20. D. Gottesman, H.-K. Lo, N. Lutkenhaus, and J. Preskill. Security of quantum key distribution with imperfect devices. Quant. Inf. Comp, 5:325–360, 2004. quant-ph/0212066.
21. M. Hayashi. Quantum Information - An introduction. Springer, 2006.
22. R. A. Horn and C. R. Johnson. Matrix Analysis. Cambridge University Press, 1985.
23. B. Julsgaard, J. Sherson, J. I. Cirac, J. Fiurasek, and E. S. Polzik. Experimental demonstration of quantum memory for light. Nature, 432:pp. 482–485, 2004.
24. J. Kilian. Founding cryptography on oblivious transfer. In *Proceedings of 20th ACM STOC*, pages 20–31, 1988.
25. E. Knill, R. Laflamme, and G. Milburn. A scheme for efficient quantum computation with linear optics. Nature, 409:46–52, 2001.
26. R. König, R. Renner, and C. Schaffner. The operational meaning of min- and max-entropy. IEEE Transactions on Information Theory, 2009. arxiv:0807.1338.
27. H-K. Lo. Insecurity of quantum secure computations. Physical Review A, 56:1154, 1997.
28. H-K. Lo and H. F. Chau. Is quantum bit commitment really possible? Physical Review Letters, 78:3410, 1997.
29. H-K. Lo and H.F. Chau. Why quantum bit commitment and ideal quantum coin tossing are impossible. In *Proceedings of PhysComp96*, 1996. quant-ph/9605026.
30. D. Mayers. The trouble with quantum bit commitment. quant-ph/9603015, 1996.
31. D. Mayers. Unconditionally secure quantum bit commitment is impossible. Physical Review Letters, 78:3414–3417, 1997.
32. R. Motwani and R. Prabhakar. Randomized Algorithms. Cambridge University Press, 1995.
33. M. A. Nielsen and I. L. Chuang. Quantum Computation and Quantum Information. Cambridge University Press, 2000.
34. T. B. Pittman and J. D. Franson. Cyclical quantum memory for photonic qubits. Phys. Rev. A, 66(6):062302, Dec 2002.
35. M. Rabin. How to exchange secrets by oblivious transfer. Technical report, Aiken Computer Laboratory, Harvard University, 1981. Technical Report TR-81.
36. R. Renner. Security of Quantum Key Distribution. PhD thesis, ETH Zurich, 2005. quant-ph/0512258.
37. W. Rosenfeld, S. Berner, J. Volz, M. Weber, and H. Weinfurter. Remote preparation of an atomic quantum memory. Physical Review Letters, 98:0505004, 2007.
38. L. Salvail. Quantum bit commitment from a physical assumption. In *Proceedings of CRYPTO'98*, volume 1462 of Lecture Notes in Computer Science, pages 338–353, 1998.
39. C. Schaffner. Cryptography in the Bounded-Quantum-Storage Model. PhD thesis, University of Aarhus, 2007. arxiv:0709.0289.
40. M. Tomamichel, R. Colbeck, and R. Renner. A fully quantum asymptotic equipartition property. arxiv:0811.1221, 2008.
41. S. Wehner, C. Schaffner, and B. Terhal. Cryptography from noisy photonic storage. arxiv:0711.2895, 2007.
42. S. Wehner, C. Schaffner, and B. M. Terhal. Cryptography from noisy storage. Physical Review Letters, 100(22):220502, 2008.
43. S. Wehner and J. Wullschleger. Security in the bounded-quantum-storage model. In ICALP 2008, 2008.
44. S. Wiesner. Conjugate coding. Sigact News, 15(1), 1983.
45. J. Wullschleger. Oblivious-transfer amplification. In *Advances in Cryptology - EUROCRYPT '07*, Lecture Notes in Computer Science. Springer-Verlag, 2007.

## Appendix A

In this Appendix we provide the technical proofs of the Lemmas and the Theorem in Section 3.2.1. We restate the claims for convenience.

983 C. Schaffner, B. Terhal, and S. Wehner

**Proof of Lemma 1 (Chain Rule)**

**Lemma A.1 (Chain Rule)** For any ccq-state $\rho_{XYE} \in \mathcal{S}(\mathcal{H}_{XYE})$ and for all $\epsilon \ge 0$, it holds that
$$
H_{\infty}^{\epsilon}(X|YE) \ge H_{\infty}^{\epsilon}(XY|E) - \log |\mathcal{Y}|,
$$
where $|\mathcal{Y}|$ is the alphabet size of the random variable $Y$.

**Proof.** For $\epsilon=0$, it follows from Eq. (3) that we need to show that
$$
P_{\text{guess}}(XY|E)_{\rho} \le P_{\text{guess}}(X|YE)_{\rho} \cdot \frac{1}{|\mathcal{Y}|} \tag{A.1}
$$
For a given value $y$, let $\{M_x^y\}_x$ be the POVM on register $E$ which optimally guesses $X$ given $Y$. A particular strategy of guessing $X$ and $Y$ from $E$ is to guess a value of $y$ uniformly at random from $Y$ and subsequently measure $E$ with the POVM $\{M_x^y\}_x$. The success probability of this strategy is exactly the r.h.s of (A.1). Clearly, the optimal guessing probability $P_{\text{guess}}(XY|E)$ can only be better than this particular strategy. For $\epsilon > 0$, let $\tilde{\rho}_{XYE} \in \mathcal{K}^{\epsilon}(\rho_{XYE})$ be the state in the $\epsilon$-ball around $\rho_{XYE}$ that maximizes the min-entropy $H_{\infty}(XY|E)$. The technique from Remark 3.2.4 in [36] can be used to show that $\tilde{\rho}_{XYE}$ is a ccq-state. By the derivation above for $\epsilon=0$, we obtain that
$$
P_{\text{guess}}(XY|E)_{\tilde{\rho}} \le P_{\text{guess}}(X|YE)_{\tilde{\rho}} \cdot \frac{1}{|\mathcal{Y}|}
$$
$$
\le \min_{\tilde{\rho}_{XYE} \in \mathcal{K}^{\epsilon}(\rho_{XYE})} P_{\text{guess}}(X|YE)_{\tilde{\rho}} \cdot \frac{1}{|\mathcal{Y}|}
$$
which proves the lemma by taking the negative logarithms and using Eq. (3). $\Box$

**Proof of Lemma 2 (Additivity)**

To show additivity of the smooth min-entropy we will employ semidefinite programming, where we refer to [5] for in-depth information. Here, we will use semidefinite programming in the language of [26] to express the primal and dual optimization problem given by parameters $c \in \mathcal{V}_1$ and $b \in \mathcal{V}_2$ in vector spaces $\mathcal{V}_1$ and $\mathcal{V}_2$ with inner products $(\cdot, \cdot)_1$ and $(\cdot, \cdot)_2$. We will optimize over variables $v_1 \in \mathcal{K}_1$ and $v_2 \in \mathcal{K}_2$, where $\mathcal{K}_1 \subset \mathcal{V}_1$ and $\mathcal{K}_2 \subset \mathcal{V}_2$ are convex cones in the respective vector spaces. In our application below, these will simply be the cones of positive-semidefinite matrices. We can then write
$$
\gamma^{\text{primal}} = \min_{v_1 \ge 0} (v_1, c)_1 \quad \text{and} \quad \gamma^{\text{dual}} = \max_{v_2 \ge 0} (b, v_2)_2, \tag{A.2}
$$
$$
A v_1 \ge b \quad A^* v_2 \le c
$$
where $A: \mathcal{V}_1 \to \mathcal{V}_2$ is a linear map defining the particular problem we wish to solve. We use $A^*: \mathcal{V}_2 \to \mathcal{V}_1$ to denote its dual map satisfying
$$
(A v_1, v_2)_2 = (v_1, A^* v_2)_1 \quad \text{for all } v_1 \in \mathcal{V}_1, v_2 \in \mathcal{V}_2 .
$$
Note that we have $\gamma^{\text{primal}} \ge \gamma^{\text{dual}}$ by weak duality. In this case our SDPs will be strongly feasible, giving us $\gamma^{\text{primal}} = \gamma^{\text{dual}}$ known as strong duality. Our proof is based on the same

984 Robust cryptography in the noisy-quantum-storage model

idea as [41, Lemma 2] applied to the smoothed setting. We thank Robert König for allowing us to include the following.

**Lemma A.2 (Additivity (König and Wehner))** Let $\rho_{AB}$ and $\rho_{A'B'}$ be two independent qq-states. For $\epsilon \ge 0$, it holds that
$$
H_{\infty}^{\epsilon^2}(AA'|BB')_{\rho} \le H_{\infty}^{\epsilon}(A|B)_{\rho} + H_{\infty}^{\epsilon}(A'|B')_{\rho} .
$$

**Proof.** In order to prove additivity, it is important to realize that the smooth conditional min-entropy can be written as semi-definite program:
$$
H_{\infty}(A|B)_{\rho} = \max_{\tilde{\rho}_{AB} \in \mathcal{K}^{\epsilon}(\rho_{AB})} H_{\infty}(A|B)_{\tilde{\rho}}
$$
$$
= \max_{\tilde{\rho}_{AB} \in \mathcal{K}^{\epsilon}(\rho_{AB})} - \log \min_{\substack{\sigma_B \ge 0 \\ \tilde{\rho}_{AB} \le id_A \otimes \sigma_B}} Tr(\sigma_B) \tag{A.3}
$$
$$
= - \log \min_{\substack{\sigma_B \ge 0 \\ \tilde{\rho}_{AB} \in \mathcal{K}^{\epsilon}(\rho_{AB}) \\ \tilde{\rho}_{AB} \le id_A \otimes \sigma_B}} Tr(\sigma_B). \tag{A.4}
$$
where $\sigma_B \in \mathcal{P}(\mathcal{H}_B)$ throughout. Let $|\Psi\rangle_{ABC}$ be a purification of $\rho_{AB}$. Then, all states $\tilde{\rho}_{AB} \in \mathcal{K}^{\epsilon}(\rho_{AB})$ can be obtained by an extension $\tilde{\rho}_{ABC} \ge 0$ such that $Tr(\tilde{\rho}_{ABC}) \le 1$, and $Tr(\tilde{\rho}_{ABC} |\Psi\rangle\langle \Psi|_{ABC}) \ge 1 - \delta$ with $\delta = \epsilon^2$. Therefore, we can write
$$
H_{\infty}^{\epsilon}(A|B)_{\rho} = - \log \min_{\substack{\sigma_B \ge 0 \\ \tilde{\rho}_{ABC} \ge 0 \\ Tr(\tilde{\rho}_{ABC} |\Psi\rangle\langle \Psi|_{ABC}) \ge 1 - \delta \\ Tr(\tilde{\rho}_{ABC}) \le 1 \\ \tilde{\rho}_{AB} \le id_A \otimes \sigma_B}} Tr(\sigma_B),
$$
where the minimum is taken over all $\sigma_B \in \mathcal{P}(\mathcal{H}_B)$ and $\tilde{\rho}_{ABC} \in \mathcal{P}(\mathcal{H}_{ABC})$, which is a semi-definite program (SDP). Our goal will be to determine the dual of this semidefinite program which will then allow us to put an upper bound on the smooth min-entropy as desired.

We now first show how to convert the primal of this semidefinite program into the form of Eq. (A.2). Let $\mathcal{V}_1 = Herm(\mathcal{H}_B) \oplus Herm(\mathcal{H}_{ABC})$ where $Herm(\mathcal{H})$ is the (real) vector space of Hermitian operators on $\mathcal{H}$. Let $\mathcal{K}_1 \subset \mathcal{V}_1$ be the cone of positive semi-definite operators. Let $c = id_B \oplus 0_{ABC}$ where $0_{ABC}$ is the zero-operator on $\mathcal{H}_{ABC}$. Let the inner product be defined as $(v_1, v'_1)_1 = Tr(v_1 v'_1)$. Note that this allows us to express our objective function as
$$
(\sigma_B \oplus \tilde{\rho}_{ABC}, c)_1 = Tr(\sigma_B).
$$
It remains to rewrite the constraints in the appropriate form. To this end, we need to define $\mathcal{V}_2 = \mathbb{R} \oplus \mathbb{R} \oplus Herm(\mathcal{H}_A) \oplus Herm(\mathcal{H}_B)$, $\mathcal{K}_2 \subset \mathcal{V}_2$ the cone of positive semi-definite operators and take the inner product to have the same form $(v_2, v'_2)_2 = Tr(v_2 v'_2)$. We then let $b \in \mathcal{V}_2$ be given as
$$
b = (1 - \delta) \oplus (-1) \oplus 0_{AB},
$$
and define the map
$$
A(\sigma_B \oplus \tilde{\rho}_{ABC}) = Tr(\tilde{\rho}_{ABC} |\Psi\rangle\langle \Psi|_{ABC}) \oplus (-Tr(\tilde{\rho}_{ABC})) \oplus (id_A \otimes \sigma_B - \tilde{\rho}_{AB}) .
$$

985 C. Schaffner, B. Terhal, and S. Wehner

Note that $v_1 = \sigma_B \oplus \tilde{\rho}_{ABC} \ge 0$ and $A(v_1) \ge b$ now exactly represent our constraints.

We now use this formalism to find the dual. Note that we may write any $v_2 \in \mathcal{V}_2$ with $v_2 \ge 0$ as $v_2 = r \oplus s \oplus Q_{AB}$ where $Q_{AB} \in \mathcal{P}(\mathcal{H}_A \otimes \mathcal{H}_B)$ and $r, s \in \mathbb{R}$. To find the dual map $A^*$ note that
$$
(A v_1, v_2)_2 = r Tr(\tilde{\rho}_{ABC} |\Psi\rangle\langle \Psi|_{ABC}) - s Tr(\tilde{\rho}_{ABC}) + Tr(Q_{AB}(id_A \otimes \sigma_B - \tilde{\rho}_{AB}))
$$
$$
= r Tr(\tilde{\rho}_{ABC} |\Psi\rangle\langle \Psi|_{ABC}) - s Tr(\tilde{\rho}_{ABC}) + Tr(Q_B \sigma_B) - Tr((Q_{AB} \otimes id_C) \tilde{\rho}_{ABC}),
$$
and we therefore have
$$
A^*(v_2) = (Q_B \oplus r |\Psi\rangle\langle \Psi|_{ABC} - s id_{ABC}) + (Q_B \oplus 0_{ABC}) - (0_B \oplus Q_{AB} \otimes id_C),
$$
which is all we require using Eq. (A.2). To find a more intuitive interpretation of the dual note that $A^*(v_2) \le c$ is equivalent to
$$
id_B \ge Q_B , \tag{A.5}
$$
$$
Q_{AB} \otimes id_C \ge r |\Psi\rangle\langle \Psi|_{ABC} - s id_{ABC}, \tag{A.6}
$$
and $(b, v_2)_2 = r(1 - \delta) - s$. The dual can thus be written as
$$
\gamma^{\text{dual}} = \max_{\substack{r \ge 0, s \ge 0 \\ id_B \ge Q_B \\ Q_{AB} \otimes id_C \ge r |\Psi\rangle\langle \Psi|_{ABC} - s id_{ABC}}} r(1 - \delta) - s .
$$

We now use the dual formulation to upper bound the smooth min-entropy of the combined state $\rho_{AB} \otimes \rho_{A'B'}$ and parameter $\delta$ by finding a lower bound to the dual semidefinite program. Let $\gamma(\delta)$ denote the optimal solution of the dual of the SDP for the combined state for error $\delta$. For each individual state, we may solve the above SDP, where we let $Q_{AB}, r$ and $s$ denote the optimal solution for state $\rho_{AB}$ with parameter $\delta$ and optimal value $\gamma(\delta)$, and let $Q_{A'B'}, r'$ and $s'$ denote the optimal solution for state $\rho_{A'B'}$ with parameter $\delta'$ and optimal value $\gamma(\delta')$. We now use these solutions to construct a solution (not necessarily the optimal one) for the combined state $\rho_{AB} \otimes \rho_{A'B'}$. Let $Q = Q_{AB} \otimes Q_{A'B'}$, $\tilde{r} = r r'$ and $\tilde{s} = r s'(1 - \delta) + s r'(1 - \delta') - s s'$. Note that $r s' \ge 0$ and $r'(1 - \delta') - s' \ge 0$ for the optimal $r', s'$ and hence
$$
\tilde{r} \ge 0, \tilde{s} \ge 0,
$$
$$
id_{BB'} \ge Q_{BB'},
$$
$$
Q_{AA'BB'} \otimes id_{CC'} \ge (r |\Psi\rangle\langle \Psi|_{ABC} - s id_{ABC}) \otimes (r' |\Psi\rangle\langle \Psi|_{A'B'C'} - s' id_{A'B'C'})
$$
$$
\ge \tilde{r} |\Psi\rangle\langle \Psi|_{ABC} \otimes |\Psi\rangle\langle \Psi|_{A'B'C'} - \tilde{s} id_{ABC} \otimes id_{A'B'C'},
$$
and thus $Q$ is indeed a feasible solution for the combined problem. Choosing $\tilde{\delta}$ as
$$
\tilde{\delta} = \delta + \delta' - \delta \delta'
$$
we have
$$
\gamma(\tilde{\delta}) \ge \tilde{r} (1 - \tilde{\delta}) - \tilde{s} = \gamma(\delta) \gamma'(\delta') .
$$

986 Robust cryptography in the noisy-quantum-storage model

We hence obtain
$$
H_{\infty}^{\sqrt{\delta}}(\tilde{A}| \tilde{B}) \le H_{\infty}^{\delta}(A|B) + H_{\infty}^{\delta}(A'|B').
$$
For $\delta = \delta'$, we have
$$
\tilde{\delta} = 2\delta - \delta^2 \ge \delta^2 .
$$
Putting everything together we thus have
$$
H_{\infty}^{\sqrt{\delta}}(A^n|B^n) \le H_{\infty}^{\delta}(A|B) + H_{\infty}^{\delta}(A'|B'),
$$
from which the result follows since $\delta = \epsilon^2$.
$\Box$

**Proof of Lemma 3 (Monotonicity)**

**Lemma A.3 (Monotonicity)** For a ccq-state $\rho_{XYE}$ and for all $\epsilon \ge 0$, it holds that
$$
H_{\infty}^{\epsilon}(XY|E) \ge H_{\infty}^{\epsilon}(Y|E) .
$$

**Proof.** For $\epsilon=0$, the lemma follows from Eq. (3), that is, guessing $XY$ from $E$ is harder than guessing only $Y$ from $E$ and therefore, $P_{\text{guess}}(XY|E) \le P_{\text{guess}}(Y|E)$.

For $\epsilon > 0$ the idea behind the argument is similar. Let the maximum in $H_{\infty}^{\epsilon}(Y|E)$ be achieved by a density matrix $\tilde{\rho}_{YE}$, i.e. $H_{\infty}^{\epsilon}(Y|E) = H_{\infty}(Y|E)_{\tilde{\rho}}$, such that $C(\rho_{YE}, \tilde{\rho}_{YE}) \le \epsilon$ and $Tr(\tilde{\rho}_{YE}) \le 1$. Remark 3.2.4 in [36] shows that $\tilde{\rho}_{YE}$ is a cq-state. We can express this min-entropy in terms of the guessing probability, Eq. (3), and thus
$$
H_{\infty}(Y|E)_{\tilde{\rho}} = - \log P_{\text{guess}}(Y|E)_{\tilde{\rho}_{YE}} \le - \log P_{\text{guess}}(XY|E)_{\tilde{\rho}_{XYE}}, \tag{A.7}
$$
where $\tilde{\rho}_{XYE}$ is any ccq-state which has $\tilde{\rho}_{YE}$ as its reduced state, i.e $Tr_X(\tilde{\rho}_{XYE}) = \tilde{\rho}_{YE}$. Now we would like to show that one can choose an extension $\tilde{\rho}_{XYE}$ such that $C(\rho_{XYE}, \tilde{\rho}_{XYE}) = \sqrt{1 - F(\rho_{XYE}, \tilde{\rho}_{XYE})^2} \le \epsilon$ and $Tr(\tilde{\rho}_{XYE}) \le 1$. If we can determine such an extension, we can upper-bound the r.h.s. in Eq. (A.7) by $H_{\infty}^{\epsilon}(XY|E)$ which is the supremum of $-\log P_{\text{guess}}(XY|E)$ over states in the $\epsilon$-neighborhood of $\rho_{XYE}$. This would prove the Lemma.

Let $|\Psi\rangle_{XYEC}$ be a purification of $\rho_{XYE}$ and hence also a purification of $\rho_{YE}$. By Uhlmann's theorem (see e.g. [33]), we have for the fidelity $F(\rho_{YE}, \tilde{\rho}_{YE})$ between $\rho_{YE}$ and $\tilde{\rho}_{YE}$ that
$$
F(\rho_{YE}, \tilde{\rho}_{YE}) = \max_{|\tilde{\Psi}\rangle_{XYEC}} |\langle \Psi | \tilde{\Psi} \rangle| := F(|\Psi\rangle\langle \Psi|, |\tilde{\Psi}\rangle\langle \tilde{\Psi}|),
$$
where $|\tilde{\Psi}\rangle_{XYEC}$ is the purification of $\tilde{\rho}_{YE}$ achieving the maximum. The monotonicity property of the fidelity under taking the partial trace gives
$$
F(|\Psi\rangle\langle \Psi|, |\tilde{\Psi}\rangle\langle \tilde{\Psi}|) \le F(Tr_C(|\Psi\rangle\langle \Psi|), Tr_C(|\tilde{\Psi}\rangle\langle \tilde{\Psi}|)) = F(\rho_{XYE}, \tilde{\rho}_{XYE}),
$$
where $\tilde{\rho}_{XYE} := Tr_C(|\tilde{\Psi}\rangle\langle \tilde{\Psi}|_{XYEC})$. Hence
$$
\sqrt{1 - \epsilon^2} \le F(\rho_{YE}, \tilde{\rho}_{YE}) \le F(\rho_{XYE}, \tilde{\rho}_{XYE}), \tag{A.8}
$$

987 C. Schaffner, B. Terhal, and S. Wehner

and therefore, $C(\rho_{XYE}, \tilde{\rho}_{XYE}) \le \epsilon$. If $Tr(\tilde{\rho}_{XYE}) > 1$, it follows that also $Tr(\tilde{\rho}_{YE}) > 1$ which contradicts the assumption. Therefore, it must be the case that $Tr(\tilde{\rho}_{XYE}) \le 1$.

It remains to show that $\tilde{\rho}_{XYE}$ is a ccq-state. Because of
$$
F(\rho_{YE}, \tilde{\rho}_{YE}) = F(|\Psi\rangle\langle \Psi|, |\tilde{\Psi}\rangle\langle \tilde{\Psi}|)
$$
$$
\le F(Tr_C(|\Psi\rangle\langle \Psi|), Tr_C(|\tilde{\Psi}\rangle\langle \tilde{\Psi}|)) = F(\rho_{XYE}, \tilde{\rho}_{XYE}) \le F(\rho_{YE}, \tilde{\rho}_{YE}),
$$
these quantities are all equal and in particular, we could do a measurement on the $X$-register of $\tilde{\rho}_{XYE}$ without increasing the fidelity. Hence, we can assume the optimal purification $|\tilde{\Psi}\rangle_{XYEC}\langle \tilde{\Psi}|_{XYEC}$ is such that $\tilde{\rho}_{XYE}$ is a ccq-state.
$\Box$

**Proof of Theorem 1**

**Theorem A.1** For $i=1,...,n$, let $\rho_i \in \mathcal{S}(\mathcal{H}_{AB})$ be density operators. Then, for any $\epsilon > 0$,
$$
H_{\infty}^{\epsilon}(A^n|B^n)_{\otimes_{i=1}^n \rho_i} \ge \sum_{i=1}^n [H(A_i|B_i)_{\rho_i}] - \delta(\epsilon, \gamma)\sqrt{n},
$$
where, for $n \ge \frac{5}{\epsilon^2} \log \frac{2}{\epsilon^2}$, the error is given by
$$
\delta(\epsilon, \gamma) := 4 \log \frac{\gamma}{\sqrt{n}} \sqrt{\log \frac{2}{\epsilon^2}}
$$
and the single-system entropy contribution by
$$
\gamma \le 2 \max_i \sqrt{\text{rank}(\rho_{A_i})} + 1 .
$$

**Proof.** The proof is analogous to the proof of Theorem 7 in [40]. For convenience, we point out where their proof needs to be adapted. We need the following definitions. Let $\mathcal{H}'_{AB}$ be a copy of $\mathcal{H}_{AB}$ and let $|\gamma\rangle := \sum_i |i\rangle|i\rangle$ be the unnormalized fully entangled state on $\mathcal{H}_{AB} \otimes \mathcal{H}'_{AB}$. Define the purification $|\phi\rangle := (\sqrt{\rho_{AB}} \otimes id_{A'B'})|\gamma\rangle$ of $\rho_{AB}$ and let $1 < \alpha < 2$, $\beta := \alpha - 1$, and $X := \rho_{AB} \otimes (id_{A'B'} \rho_{B'}^{-1})^T$. The conditional $\alpha$-entropy is defined as $H_{\alpha}(A|B)_{\rho|\rho} := \frac{1}{1-\alpha} \log Tr(\rho_{AB} (id_A \otimes \sigma_B)^{\beta/(1-\alpha)})$. The authors of [40] prove the following lower bound
$$
H_{\alpha}(A|B)_{\rho|\rho} \ge H(A|B)_{\rho} - \frac{1}{\beta \ln 2} \log (\langle \phi | r_{\beta}(X) |\phi\rangle) , \tag{A.9}
$$
where $r_{\beta}(t) := t^{\beta} - \beta \ln t - 1$.
Let $\rho = \rho_{AB}^1 \otimes ... \otimes \rho_{AB}^n$. Then, as in Equation (27) of [40], we have
$$
H_{\infty}^{\epsilon}(A^n|B^n)_{\rho} \ge H_{\infty}^{\epsilon}(A^n|B^n)_{\rho|\rho} \ge H_{\alpha}(A^n|B^n)_{\rho|\rho} - \frac{1}{\beta \ln 2} \log \frac{2}{\epsilon^2}
$$
$$
= \sum_{i=1}^n H_{\alpha}(A_i|B_i)_{\rho^i|\rho^i} - \frac{1}{\beta \ln 2} \log \frac{2}{\epsilon^2}
$$
$$
\ge \sum_{i=1}^n \left( H(A_i|B_i)_{\rho^i} - \frac{1}{\beta \ln 2} \log (\langle \phi | r_{\beta}(X^i) |\phi\rangle) \right) - \frac{1}{\beta \ln 2} \log \frac{2}{\epsilon^2} , \tag{A.10}
$$

988 Robust cryptography in the noisy-quantum-storage model

where we used (A.9) in the last step.

Let us define the single-system entropy contributions $\gamma_i^2 := (\langle \phi | \sqrt{X_i} + 1/\sqrt{X_i} + id | \phi \rangle)$ of which we know that they are all $\ge 3$ and let $\gamma_{\max}$ be the largest of them. By choosing an appropriate $\mu \ge 0$ such that
$$
\frac{1}{\beta} = \frac{1}{2\mu\sqrt{n}} \sqrt{\frac{5}{\log \frac{2}{\epsilon^2}}} \frac{1}{2\log \gamma_{\max}} \le \min \left\{ \frac{1}{4}, \frac{1}{2\log \gamma_{\max}} \right\} ,
$$
we can bound
$$
\frac{1}{\beta \ln 2} \langle \phi | r_{\beta}(X^i) |\phi\rangle \le \frac{2}{\mu \sqrt{n}} \log^2(\gamma^i) \le \frac{2}{\mu \sqrt{n}} \log^2(\gamma_{\max}) ,
$$
Therefore, we can further lower bound (A.10) as
$$
H_{\infty}^{\epsilon}(A^n|B^n)_{\rho} \ge \sum_{i=1}^n H(A|B)_{\rho^i} - \sum_{i=1}^n \frac{2}{\mu \sqrt{n}} \log^2(\gamma_{\max}) - 2\sqrt{n}\log \frac{2}{\epsilon^2}
$$
$$
\ge \sum_{i=1}^n H(A|B)_{\rho^i} - \frac{2\sqrt{n}}{\mu} \left( \frac{1}{n} \sum_{i=1}^n \log^2(\gamma_{\max}) + \mu \log \frac{2}{\epsilon^2} \right) ,
$$
and the rest of the derivation goes as after Equation (28) in [40].

In order to obtain the upper bound on $\gamma$, we notice that $H_{1/2}(A|B)_{\rho|\rho} \le H_{1/2}(A)_{\rho} \le H_0(A)_{\rho} = \log(\text{rank}(\rho_A))$. $\Box$

## Appendix B

In this appendix, we use the symmetries inherent in our problem to prove Theorem 5 in a series of steps.

**Theorem B.1** Let $\mathcal{N}$ be the depolarizing quantum operation given by Eq. (8) and let $H(X|\Theta K E)$ be the conditional von Neumann entropy of one qubit. Then
$$
H(X|\Theta K E) \ge
\begin{cases}
h \left( \frac{1+r}{2} \right) & \text{for } r \ge \tilde{r}, \\
1/2 & \text{for } r < \tilde{r},
\end{cases}
$$
where $\tilde{r} := 2h^{-1}(1/2) - 1 \approx 0.7798$.

### Setting the Stage

In order to prove the theorem, we find Bob's strategy which minimizes $H(X|\Theta K E)$ as a function of the depolarizing noise parameter $r$. As depicted in Figure 2, in each round the dishonest receiver Bob receives one of the four possible BB84 states $\rho_{x\theta}$ at random. On such state he may then perform any (partial) measurement $M$ given by measurement operators $M = \{F_k\}$ such that $\sum_k F_k^{\dagger} F_k = id$. For clarity of notation, we do not use a subscript to indicate the round $i$ as in the Figure. We denote by $E$ the register containing the renormalized post-measurement state
$$
\rho_{x\theta}^{k, M} = \frac{F_k \rho_{x\theta} F_k^{\dagger}}{P_{k|x\theta}^M} ,
$$

989 C. Schaffner, B. Terhal, and S. Wehner

to which the depolarizing quantum operation $\mathcal{N}$ is applied. Here
$$
P_{k|x\theta}^M = Tr(F_k \rho_{x\theta} F_k^{\dagger})
$$
is the probability to measure outcome $k$ when given state $\rho_{x\theta}$. We omit the superscript $M$ if it is clear which measurement is used. Note that we may write
$$
\rho_{x\theta k}^M = \frac{1}{4} P_{k|x\theta}^M \rho_{x\theta k},
$$
and
$$
\sum_{x} P_{k|x\theta}^M \rho_{x\theta k} = \frac{1}{4} Tr \left( F_k^{\dagger} ( \rho_{0\theta} + \rho_{1\theta} ) F_k \right) = \frac{1}{4} Tr(F_k^{\dagger} F_k),
$$
$$
\rho_{x\theta k}^M = \frac{P_{k|x\theta}^M}{P_{k|x\theta}} \rho_{x\theta k} .
$$
Here we have used the fact that Alice chooses the basis and bit in each round uniformly and independently at random.

First of all, note that for a cq-state $\rho_{YE} = \sum_y P_Y(y) |y\rangle\langle y| \otimes \rho_E^y$, the von Neumann entropy can be expanded as
$$
H(Y|E) = H(Y) + \sum_y P_Y(y) H(\rho_E^y) .
$$
Using this expansion, we can write
$$
H(X|\Theta K E)_M
$$
$$
= H(X\Theta K E)_M - H(\Theta K E)_M \tag{B.1}
$$
$$
= H(X\Theta K)_M + \sum_{x\theta k} P_{x\theta k}^M H \left( \mathcal{N} \left( \rho_{x\theta}^k \right) \right) - H(\Theta K)_M - \sum_{\theta k} P_{\theta k}^M H \left( \sum_x P_{x|\theta k}^M \mathcal{N} \left( \rho_{x\theta}^k \right) \right)
$$
$$
= H(X|\Theta K)_M + \sum_{x\theta k} P_{x\theta k}^M H \left( \mathcal{N} \left( \rho_{x\theta}^k \right) \right) - \sum_{\theta k} P_{\theta k}^M H \left( \sum_x P_{x|\theta k}^M \mathcal{N} \left( \rho_{x\theta}^k \right) \right) . \tag{B.2}
$$
We use the notation $H(X|\Theta K E)_M$ to emphasize that we consider the conditional von Neumann entropy when Bob performed a partial measurement $M$. In the following, we use the shorthand
$$
B(M) := H(X|\Theta K E)_M .
$$

### Using Symmetries to Reduce Degrees of Freedom

Our goal is to minimize $B(M)$ over all possible measurements $M = \{F_k\}$ as a function of $r$. We proceed in three steps. First, we simplify our problem considerably until we are left with a single Hermitian measurement operator over which we need to minimize the entropy. Second, we show that the optimal measurement operator is diagonal in the computational basis. And finally, we show that depending on the amount of noise, this measurement operator is either proportional to the identity, or proportional to a rank one projector.

First, we prove a property of the function $B(M)$ for a composition of two measurements. Intuitively, the following statement uses the fact that if we choose one measurement with probability $\alpha$ and another measurement with probability $\beta$ our average success probability is the average of the success probabilities obtained via the individual measurements:

990 Robust cryptography in the noisy-quantum-storage model

**Claim 1** Let $F = \{F_k\}_{k=1}^f$ and $G = \{G_k\}_{k=f+1}^{f+g}$ be two measurements. Then, for $0 \le \alpha \le 1$ and a combined measurement $M = \alpha F + (1 - \alpha) G := \{\sqrt{\alpha} F_k\}_{k=1}^f \cup \{\sqrt{1 - \alpha} G_k\}_{k=f+1}^{f+g}$, we have
$$
B(\alpha F + (1 - \alpha) G) = \alpha B(F) + (1 - \alpha) B(G) .
$$

**Proof.** Let $F = \{F_k\}_{k=1}^f$ and $G = \{G_k\}_{k=1}^g$ be measurements, $0 \le \alpha \le 1$ and let $M := \{\sqrt{\alpha} F_k\}_{k=1}^f \cup \{\sqrt{1 - \alpha} G_k\}_{k=1}^g$.

It is easy to verify that we have the following relations for $1 \le k \le f$: $P_{k|x\theta}^M = \alpha P_{k|x\theta}^F$, $\rho_{x\theta k}^M = \rho_{x\theta k}^F$, $P_{k|x\theta}^M \rho_{x\theta k}^M = \alpha P_{k|x\theta}^F \rho_{x\theta k}^F$ and analogously for $f+1 \le k \le f+g$.

We consider the three summands in Eq. (B.2) separately. For the first term we get
$$
H(X|\Theta K)_M = \sum_{\theta k} P_{\theta k}^M h(\rho_{0|\theta k}^M)
$$
$$
= \sum_{\theta} \sum_{k=1}^f \alpha P_{k|\theta}^F h(\rho_{0|\theta k}^F) + \sum_{\theta} \sum_{k=f+1}^{f+g} (1 - \alpha) P_{k|\theta}^G h(\rho_{0|\theta k}^G)
$$
$$
= \alpha H(X|\Theta K)_F + (1 - \alpha) H(X|\Theta K)_G .
$$
For the second term, we obtain
$$
\sum_{x\theta k} P_{x\theta k}^M H \left( \mathcal{N} \left( \rho_{x\theta}^k \right) \right)
$$
$$
= \alpha \sum_{x\theta} \sum_{k=1}^f P_{x\theta k}^F H \left( \mathcal{N} \left( \rho_{x\theta}^{k, F} \right) \right) + (1 - \alpha) \sum_{x\theta} \sum_{k=f+1}^{f+g} P_{x\theta k}^G H \left( \mathcal{N} \left( \rho_{x\theta}^{k, G} \right) \right) .
$$
The third term yields
$$
\sum_{\theta k} P_{\theta k}^M H \left( \sum_x P_{x|\theta k}^M \mathcal{N} \left( \rho_{x\theta}^k \right) \right)
$$
$$
= \alpha \sum_{\theta} \sum_{k=1}^f P_{\theta k}^F H \left( \sum_x P_{x|\theta k}^F \mathcal{N} \left( \rho_{x\theta}^{k, F} \right) \right) + (1 - \alpha) \sum_{\theta} \sum_{k=f+1}^{f+g} P_{\theta k}^G H \left( \sum_x P_{x|\theta k}^G \mathcal{N} \left( \rho_{x\theta}^{k, G} \right) \right) .
$$
$\Box$

We can now make a series of observations.

**Claim 2** Let $M = \{F_k\}$ and $G = \{id, X, Z, XZ\}$. Then for all $g \in G$ we have $B(M) = B(g M g^{\dagger})$.

**Proof.** First of all, note that for all $g \in G$, $g$ can at most exchange the roles of $0$ and $1$. That is, we can perform a bit flip before the measurement which we can correct for afterwards by applying classical post-processing. Furthermore, since $g \in G$ is Hermitian and unitary we have
$$
P_{k|x\theta}^{g M g^{\dagger}} = \frac{1}{4} Tr(g F_k^{\dagger} g^{\dagger} (\rho_{0\theta} + \rho_{1\theta}) g F_k g^{\dagger}) = \frac{1}{4} Tr(F_k^{\dagger} F_k) = P_{k|x\theta}^M,
$$

991 C. Schaffner, B. Terhal, and S. Wehner

and hence there exists a bijection $f : \{0, 1\} \to \{0, 1\}$ such that
$$
P_{k|x\theta}^{g M g^{\dagger}} = P_{k|f(x)\theta}^M .
$$
Again, we consider the three summands in Eq. (B.2) separately. For the first term, observe that $h(\rho_{0|\theta k}^{g M g^{\dagger}}) = h(\rho_{f(0)|\theta k}^M)$.
$$
H(X|\Theta K)_{g M g^{\dagger}} = \sum_{\theta k} P_{\theta k}^{g M g^{\dagger}} h(\rho_{0|\theta k}^{g M g^{\dagger}}) = \sum_{\theta k} P_{\theta k}^M h(\rho_{f(0)|\theta k}^M) = H(X|\Theta K)_M .
$$
To analyze the second term, note that we can write
$$
P_{x\theta k}^{g M g^{\dagger}} = P_{f(x)\theta k}^M ,
$$
and for depolarizing noise $\mathcal{N}(U \rho U^{\dagger}) = U \mathcal{N}(\rho) U^{\dagger}$, in addition the von Neumann entropy itself is invariant under unitary operations $H(g \mathcal{N}(\rho) g^{\dagger}) = H(\mathcal{N}(\rho))$. Putting everything together, we obtain
$$
\sum_{x\theta k} P_{x\theta k}^{g M g^{\dagger}} H \left( \mathcal{N} \left( \rho_{x\theta k}^{g M g^{\dagger}} \right) \right) = \sum_{x\theta k} P_{f(x)\theta k}^M H \left( \mathcal{N} \left( \rho_{f(x)\theta k}^M \right) \right) .
$$
By a similar argument, we derive the equality for the third term
$$
\sum_{\theta k} P_{\theta k}^{g M g^{\dagger}} H \left( \sum_x P_{x|\theta k}^{g M g^{\dagger}} \mathcal{N} \left( \rho_{x\theta k}^{g M g^{\dagger}} \right) \right) = \sum_{\theta k} P_{\theta k}^M H \left( \sum_x P_{x|\theta k}^M \mathcal{N} \left( \rho_{x\theta k}^M \right) \right) .
$$
$\Box$

**Claim 3** Let $G = \{id, X, Z, XZ\}$. There exists a measurement operator $F$ such that the minimum of $B(M)$ over all measurements $M$ is achieved by a measurement proportional to $\{g F g^{\dagger} \mid g \in G\}$.

**Proof.** Let $M = \{F_k\}$ be a measurement. Let $K = |M|$ be the number of measurement operators. Clearly, $M = \{F_{g, k}\}$ with
$$
\tilde{F}_{g, k} = \frac{1}{\sqrt{2}} g F_k g^{\dagger} ,
$$
is also a quantum measurement since $\sum_{g, k} \tilde{F}_{g, k}^{\dagger} \tilde{F}_{g, k} = id$. It follows from Claims 1 and 2 that $B(\tilde{M}) = B(M)$. Define operators
$$
N_{g, k} = \frac{1}{\sqrt{2Tr(F_k^{\dagger} F_k)}} g F_k g^{\dagger} .
$$
Note that
$$
\sum_{g \in G} N_{g, k} = \frac{1}{\sqrt{2Tr(F_k^{\dagger} F_k)}} \sum_{u, v \in \{0, 1\}} X^u Z^v F_k X^u Z^v \propto id .
$$

992 Robust cryptography in the noisy-quantum-storage model

(see for example Hayashi [21]). Hence $M_k = \{N_{g, k}\}$ is a valid quantum measurement. Now, note that $M$ can be obtained from $M_1, ..., M_K$ by averaging. Hence, by Claim 1 we have
$$
B(M) = B(\tilde{M}) \ge \min_k B(M_k) .
$$
Let $M^*$ be the optimal measurement. Clearly, $\min B(M^*) \ge \min_k B(M_k^*)$ by the above and Claim 2 from which the present claim follows.
$\Box$

Finally, we note that we can restrict ourselves to optimizing over positive semi-definite (and hence Hermitian) matrices only.

**Claim 4** Let $F$ be a measurement operator and $M_F = \{g F g^{\dagger} \mid g \in G\}$ the associated measurement. Then there exists a Hermitian operator $F$ such that $B(M_F) = B(M_{\tilde{F}})$.

**Proof.** Let $F^{\dagger} = F U$ be the polar decomposition of $F^{\dagger}$, where $\tilde{F}$ is positive semi-definite and $U$ is unitary [22, Corollary 7.3.3]. Evidently, since the trace is cyclic, all probabilities remain the same. Using the invariance of the von Neumann entropy and the depolarizing quantum operation under unitaries, the claim follows.
$\Box$

Note that Claim 3 also gives us that we have at most $4$ measurement operators. Wlog, we take the measurement outcomes to be labeled $1, 2, 3, 4$ and measurement operators $F_1 = F, F_2 = X F X, F_3 = Z F Z, F_4 = X Z F Z X$. Our final observation is the following easy claim.

**Claim 5** For any linear operator $F$ on Hilbert space $\mathcal{H}$ and any state $|\phi\rangle \in \mathcal{H}$ such that $F|\phi\rangle \ne 0$, it holds that the operator $P := \frac{F|\phi\rangle\langle \phi|F^{\dagger}}{Tr(F|\phi\rangle\langle \phi|F^{\dagger})}$ is a projector with $\text{rank}(P)=1$.

**Proof.** Notice that $|\phi\rangle\langle \phi|F^{\dagger} F|\phi\rangle\langle \phi| = Tr(F|\phi\rangle\langle \phi|F^{\dagger}) |\phi\rangle\langle \phi|$. Thus
$$
P P = \frac{F|\phi\rangle\langle \phi|F^{\dagger} F|\phi\rangle\langle \phi|F^{\dagger}}{Tr(F|\phi\rangle\langle \phi|F^{\dagger})^2} = \frac{Tr(F|\phi\rangle\langle \phi|F^{\dagger})}{Tr(F|\phi\rangle\langle \phi|F^{\dagger})^2} F|\phi\rangle\langle \phi|F^{\dagger} = P .
$$
As $F|\phi\rangle \ne 0$ we have that $\text{rank}(F|\phi\rangle\langle \phi|F^{\dagger}) = 1$.
$\Box$

Exploiting our observations, we can considerably simplify the expression $B(M)$ to be minimized:

**Lemma B.1** Let $B(M)$ be defined as above. Then
$$
\min_M B(M) = \min_F C(F),
$$
where the minimization is taken over Hermitian operators $F \in \mathbb{C}^{2\times 2}$ and $C(F)$ is defined as
$$
C(F) = \frac{1}{2} \left( h \left( 2 Tr \left( F \rho_{0+} F \right) \right) + h \left( 2 Tr \left( F \rho_{0\times} F \right) \right) \right) + h \left( \frac{1+r}{2} \right) - H \left( \mathcal{N} \left( 2 F^2 \right) \right) . \tag{B.3}
$$

**Proof.** First of all, note that
$$
P_{0|\theta k} = \rho_{0\theta k} + P_{1|\theta k} = \frac{1}{4} Tr \left( F_k (\rho_{0\theta} + \rho_{1\theta}) F_k \right) = \frac{1}{4} Tr(F_k^2),
$$

993 C. Schaffner, B. Terhal, and S. Wehner

which is independent of $k$. Thus we have
$$
\frac{1}{2} = P_{\theta} = \sum_{k=1}^4 P_{k|\theta} = Tr(F^2) ,
$$
and hence $P_{\theta k} = \frac{1}{2}$. Furthermore, as in the proof of Claim 2, there exists a bijection $f: \{0,1\} \to \{0,1\}$ such that
$$
P_{x|\theta k} = \frac{P_{x\theta k}}{P_{\theta k}} = \frac{Tr(F_k \rho_{x\theta} F_k^{\dagger})/4}{1/8} = 2 Tr(F_k \rho_{x\theta} F_k^{\dagger}) = 2 Tr(F P_{f(x)\theta} F) .
$$
Note again that $h(P_{0|\theta k}^M) = h(P_{0|\theta k})$. We then obtain for the first term
$$
H(X|\Theta K) = \sum_{\theta k} P_{\theta k} h(P_{0|\theta k})
$$
$$
= \sum_{\theta k} \frac{1}{8} h(2 Tr(F_k \rho_{0\theta} F_k^{\dagger}))
$$
$$
= \frac{1}{2} \left( h(2 Tr(F \rho_{0+} F)) + h(2 Tr(F \rho_{0\times} F)) \right) .
$$
For the second term, we need to evaluate $H(\mathcal{N}(\rho_{x\theta}^k))$. It follows from Claim 5 that if $P_{x\theta k} > 0$, the normalized post-measurement state $\rho_{x\theta}^k$ has eigenvalues $0$ and $1$. Applying the depolarizing quantum operation to such rank 1 state gives an entropy $H(\mathcal{N}(\rho_{x\theta}^k)) = h((1+r)/2)$, independent of the state. Thus the second term becomes
$$
\sum_{x\theta k} P_{x\theta k} H \left( \mathcal{N} \left( \rho_{x\theta}^k \right) \right) = \sum_{x\theta k} P_{x\theta k} h \left( \frac{1+r}{2} \right) = h \left( \frac{1+r}{2} \right) .
$$
For the third term, we use that for $0 \le \alpha \le 1$, it holds that $\mathcal{N}(\alpha \rho + (1 - \alpha) \sigma) = \alpha \mathcal{N}(\rho) + (1 - \alpha) \mathcal{N}(\sigma)$. Hence,
$$
P_{0|\theta k} \mathcal{N} \left( \rho_{0\theta}^k \right) + P_{1|\theta k} \mathcal{N} \left( \rho_{1\theta}^k \right) = \mathcal{N} \left( P_{0|\theta k} \rho_{0\theta}^k + P_{1|\theta k} \rho_{1\theta}^k \right)
$$
$$
= \mathcal{N} \left( 2 Tr \left( F_k \rho_{0\theta} F_k^{\dagger} \right) \frac{F_k \rho_{0\theta} F_k^{\dagger}}{Tr(F_k \rho_{0\theta} F_k^{\dagger})} + 2 Tr \left( F_k \rho_{1\theta} F_k^{\dagger} \right) \frac{F_k \rho_{1\theta} F_k^{\dagger}}{Tr(F_k \rho_{1\theta} F_k^{\dagger})} \right)
$$
$$
= \mathcal{N} \left( 2 F_k (\rho_{0\theta} + \rho_{1\theta}) F_k^{\dagger} \right)
$$
$$
= U_k \mathcal{N} \left( 2 F^2 \right) U_k^{\dagger},
$$
where $U_k \in G$. The third term then yields
$$
\sum_{\theta k} P_{\theta k} H \left( \sum_x P_{x|\theta k} \mathcal{N} \left( \rho_{x\theta}^k \right) \right) = H \left( \mathcal{N} \left( 2 F^2 \right) \right) .
$$
These arguments prove the Lemma.
$\Box$

994 Robust cryptography in the noisy-quantum-storage model

### F is Diagonal in the Computational Basis

Now that we have simplified our problem considerably, we are ready to perform the actual optimization. We first show that we can take $F$ to be diagonal in the computational (or Hadamard) basis.

**Claim 6** Let $F \in \mathbb{C}^{2\times 2}$ be the Hermitian operator that minimizes $C(F)$ as defined by Eq. (B.3). Then $F = \alpha |\phi\rangle\langle \phi| + \beta (id - |\phi\rangle\langle \phi|)$ for some $\alpha, \beta \in \mathbb{R}$ and pure state $|\phi\rangle$ lying in the $XZ$ plane of the Bloch sphere. (i.e. $Tr(F Y) = 0$).

**Proof.** Since $F$ is a Hermitian on a 2-dimensional space, we may express $F$ as
$$
F = \alpha |\phi\rangle\langle \phi| + \beta |\phi^{\perp}\rangle\langle \phi^{\perp}|,
$$
for some state $|\phi\rangle$ and real numbers $\alpha, \beta$. We first of all note that from $\sum_k F_k F_k = id$, we obtain that
$$
\sum_k Tr(F_k F_k) = \sum_{g \in \{id, X, Z, XZ\}} Tr(g F g^{\dagger} g F g^{\dagger}) = 4 Tr(F^2) = Tr(id) = 2,
$$
and hence $Tr(F^2) = \alpha^2 + \beta^2 = 1/2$. Furthermore, using that $|\phi\rangle\langle \phi| + |\phi^{\perp}\rangle\langle \phi^{\perp}| = id$ gives
$$
F = \alpha |\phi\rangle\langle \phi| + \beta (id - |\phi\rangle\langle \phi|), \tag{B.4}
$$
with $\beta = \sqrt{1/2 - \alpha^2}$. Hence without loss of generality, we can consider $0 \le \alpha \le 1/\sqrt{2}$. The eigenvalues of $2F^2$ are $2\alpha^2$ and $1 - 2\alpha^2$. Hence, the third term of $C(F)$ becomes $H(\mathcal{N}(2F^2)) = h(2r\alpha^2 + (1 - r)/2)$ which does not depend on $|\phi\rangle$. We want to minimize
$$
\min_F \frac{1}{2} \left( h \left( 2 Tr(F \rho_{0+} F) \right) + h \left( 2 Tr(F \rho_{0\times} F) \right) \right) + h((1+r)/2) - h(2r\alpha^2 + (1 - r)/2) . \tag{B.5}
$$
We first parametrize the state $|\phi\rangle$ in terms of its Bloch vector
$$
|\phi\rangle\langle \phi| = \frac{id + \hat{x} X + \hat{y} Y + \hat{z} Z}{2} .
$$
Since $|\phi\rangle$ is pure we can write $\hat{y} = \sqrt{1 - \hat{x}^2 - \hat{z}^2}$. Note that we may wlog assume that $0 \le \hat{x}, \hat{z} \le 1$, since the remaining three measurement operators are given by $X F X, Z F Z$, and $X Z F Z X$. A small calculation shows that for the encoded bit $x \in \{0, 1\}$
$$
2 Tr \left( F \rho_{x+} F \right) = \frac{1}{2} \left( 1 + (-1)^x (4\alpha^2 - 1) \hat{z} \right) ,
$$
and similarly
$$
2 Tr \left( F \rho_{x\times} F \right) = \frac{1}{2} \left( 1 + (-1)^x (4\alpha^2 - 1) \hat{x} \right) .
$$
Our goal is to show that for every $0 \le \alpha \le 1/\sqrt{2}$, the function
$$
f(\hat{z}) := h(2 Tr(F \rho_{0+} F))
$$

995 C. Schaffner, B. Terhal, and S. Wehner

is non-increasing on the interval $0 \le \hat{z} \le 1$. First of all, note that $f(\hat{z}) = 1$ for $\alpha = 1/2$. We now consider the case of $\alpha \ne 1/2$. A simple computation (using Mathematica) shows that when differentiating $f$ with respect to $\hat{z}$ we obtain
$$
f'(\hat{z}) = \frac{\partial}{\partial \hat{z}} f(\hat{z}) = \frac{1}{2} (1 - 4\alpha^2) \log \left( \frac{2}{1 + \hat{z} - 4\alpha^2 \hat{z}} - 1 \right) ,
$$
$$
f''(\hat{z}) = \frac{\partial^2}{\partial \hat{z}^2} f(\hat{z}) = \frac{(1 - 4\alpha^2)^2}{\ln 2 (\hat{z}^2 (1 - 4\alpha^2)^2 - 1)} .
$$
Hence the function has one maximum at $\hat{z}=0$ with $f(0)=1$. Since $0 \le \alpha \le 1/\sqrt{2}$ and $\alpha \ne 1/2$ we also have that $(1 - 4\alpha^2)^2 \le 1$ and hence $f''(\hat{z}) \le 0$ everywhere and $f$ is concave (though not strictly concave). Thus $f(\hat{z})$ is decreasing with $\hat{z}$.

Since we have $\hat{x}^2 + \hat{y}^2 + \hat{z}^2 = 1$ we can thus conclude that in order to minimize $C(F)$, we want to choose $\hat{x}$ and $\hat{z}$ as large as possible and thus let $\hat{y}=0$ from which the claim follows. $\Box$

We can immediately extend this analysis to find

**Claim 7** Let $F$ be the operator that minimizes $C(F)$, and write $F$ as in Eq. B.4. Then
$$
|\phi\rangle = g|0\rangle,
$$
for some $g \in \{id, X, Z, XZ\}$.

**Proof.** By Claim 6, we can rewrite our optimization problem as
$$
\min_{\hat{x}, \hat{z}} \frac{(f(\hat{x}) + f(\hat{z}))}{2} + h((1+r)/2) - h(2r\alpha^2 + (1 - r)/2)
$$
subject to
$$
\hat{x}^2 + \hat{z}^2 = 1
$$
$$
0 \le \hat{x} \le 1
$$
$$
0 \le \hat{z} \le 1.
$$
By using Lagrange multipliers we can see that for an extreme point we must have either $\hat{x}=\hat{z}=1/\sqrt{2}$ or $\hat{x}=0, \hat{z}=1$ or $\hat{z}=0, \hat{x}=1$. From the definition of $f$ above we can see that to minimize the expression, we want to choose the latter, from which the claim follows. $\Box$

### Optimality of the Trivial Strategies

We have shown that without loss of generality $F$ is diagonal in the computational basis. Hence, we have only a single parameter left in our optimization problem. We must optimize over all operators $F$ of the form
$$
F = \alpha |\phi\rangle\langle \phi| + \sqrt{1/2 - \alpha^2} |\phi^{\perp}\rangle\langle \phi^{\perp}|,
$$
where we may take $|\phi\rangle$ to be $|0\rangle$ or $|1\rangle$. Our aim is to show that either $F$ is the identity, or $F = |\phi\rangle\langle \phi|$ depending on the value of $r$.

**Claim 8** Let $F$ be the operator that minimizes $C(F)$, and let $r_0 := 2h^{-1}(\frac{1}{2}) - 1$. Then $F = c id$ (for some $c \in \mathbb{R}$) for $r > r_0$, and $F = |\phi\rangle\langle \phi|$ for $r < r_0$, where
$$
|\phi\rangle = g|0\rangle,
$$
for some $g \in \{id, X, Z, XZ\}$.

996 Robust cryptography in the noisy-quantum-storage model

**Proof.** We can plug $\hat{x}=0$ and $\hat{z}=1$ in the expressions in the proof of our previous claim. Thus our goal is to minimize
$$
t(r, \alpha) := \frac{1}{2} (1 + g(1, \alpha)) + h \left( \frac{1+r}{2} \right) - g(r, \alpha) ,
$$
where
$$
g(r, \alpha) := h \left( \frac{1+r}{2} - 2\alpha^2 r \right) .
$$
Differentiating $g$ with respect to $\alpha$ gives us
$$
\frac{\partial}{\partial \alpha} g(r, \alpha) = 4\alpha r \left( \log \left( \frac{1+r}{2} - 2\alpha^2 r \right) - \log \left( \frac{1-r}{2} + 2\alpha^2 r \right) \right) ,
$$
with which we can easily differentiate $t$ with respect to $\alpha$ as
$$
\frac{\partial}{\partial \alpha} t(r, \alpha) = \frac{1}{2} \frac{\partial}{\partial \alpha} g(1, \alpha) - \frac{\partial}{\partial \alpha} g(r, \alpha).
$$
We can calculate
$$
\lim_{\alpha \to 0} \frac{\partial}{\partial \alpha} t(r, \alpha) = 0
$$
and
$$
\frac{\partial}{\partial \alpha} t(r, 1/\sqrt{2}) = 0 .
$$
We thus have two extremal points. By computing the second derivative which is equal to $8(2r^2 - 1)/\ln 2$ at the point $\alpha=1/2$, we can see that as $r$ grows from $0$ to $1$, the second extreme point switches from a maximum to a minimum at $r = 1/\sqrt{2}$. Our goal is thus to determine for which $r$ we have $t(r, 0) \le t(r, 1/\sqrt{2})$. Note that shortly after the transition point $r=1/\sqrt{2}$, we do obtain two additional maxima, but since we are interested in finding the minimum they do not contribute to our analysis. By plugging in the definition for $t$ from above, we have that $t(r, 0) \le t(r, 1/\sqrt{2})$ iff
$$
\frac{1}{2} \le h \left( \frac{1+r}{2} \right) ,
$$
or in other words iff
$$
2h^{-1} \left( \frac{1}{2} \right) - 1 \le r,
$$
as promised.
$\Box$

We conclude that Bob's optimal strategy, -the one which minimizes $H(X|\Theta K E)$-, is an extremal strategy, that is, he either measures his qubit in the computational basis, or he stores the qubit as is. This is the content of Theorem 5. We believe that a similar analysis can be done for the dephasing quantum operation, by first symmetrizing the noise by applying a rotation over $\pi/4$ to the input states.