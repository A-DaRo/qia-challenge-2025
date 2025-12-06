## Performance of Practical Quantum Oblivious Key Distribution

Mariano Lemus$^{1,2}$, Peter Schiansky$^3$, Manuel Goulão$^4$, Mathieu Bozzio$^3$, David Elkouss$^{4,5}$,
Nikola Paunković$^{1,2}$, Paulo Mateus$^{1,2}$, and Philip Walther$^3$

$^1$ Instituto de Telecomunicações, 1049-001 Lisbon, Portugal
$^2$ Departamento de Matemática, Instituto Superior Técnico, Universidade de Lisboa, Av. Rovisco Pais 1, 1049-001
Lisboa, Portugal
$^3$ Vienna Center for Quantum Science and Technology (VCQ), Faculty of Physics, University of Vienna,
Boltzmanngasse 5, Vienna A-1090, Austria
$^4$ QuTech, Delft University of Technology, Lorentzweg 1, 2628 CJ Delft, Netherlands
$^5$ Networked Quantum Devices Unit, Okinawa Institute of Science and Technology Graduate University, Okinawa,
Japan

September 9, 2025

**Abstract**
Motivated by the applications of secure multiparty computation as a privacy-protecting data analysis
tool, and identifying oblivious transfer as one of its main practical enablers, we propose a practical
realization of randomized quantum oblivious transfer. By using only symmetric cryptography primitives
to implement commitments, we construct computationally-secure randomized oblivious transfer without
the need for public-key cryptography or assumptions imposing limitations on the adversarial devices. We
show that the protocol is secure under an indistinguishability-based notion of security and demonstrate
an experimental implementation to test its real-world performance. Its security and performance are
then compared to both quantum and classical alternatives, showing potential advantages over existing
solutions based on the noisy storage model and public-key cryptography.

# 1 Introduction

Cryptography is a critical tool for data privacy, a task deeply rooted in the functioning of today's digitalized
world. Whether it is in terms of secure communication over the Internet or secure data access through au-
thentication, finding ways of protecting sensitive data is of utmost importance. The one-time pad encryption
scheme allows communication with perfect secrecy [1], at the cost of requiring the exchange of single-use
secret (random) keys of the size of the communicated messages. Distribution of secret keys, therefore, is
considered one of the most important tasks in cryptography. Modern cryptography relies heavily on con-
jectures about the computational hardness of certain mathematical problems to design solutions for the
key distribution problem. However, as quantum computers threaten to make most of the currently used
cryptography techniques obsolete [2], better solutions for data protection are needed. This transition to-
wards quantum-resistant solutions becomes particularly crucial when it comes to protecting data associated
with the government, finance and health sectors, being already susceptible to *intercept-now-decrypt-later*
attacks. Cryptography solutions secure in a post-quantum world, where large-scale quantum computers will
be commercially available, have been explored in two directions. Classical cryptography based solutions,
also referred as post-quantum cryptography [3-5], involve using a family of mathematical problems that are
conjectured to be resilient to quantum computing attacks. On the other hand, quantum cryptography based
solutions [6] using the laws of quantum mechanics can offer information-theoretic security, depending on the
physical properties of quantum systems rather than computational hardness assumptions. Quantum Key
Distribution (QKD) [7] is the most well-studied and developed of these quantum solutions, while other works
beyond QKD have been proposed [8].

It is noteworthy that secure communication is not the only cryptographic task where end-users' private
data may be exposed to an adversary. Cryptography beyond secure communication and key distribution
includes zero-knowledge proofs, secret sharing, contract signing, bit commitment (BC), e-Voting, secure
data mining, etc. [9]. A huge class of such problems can be cast as Multi-Party Computation (MPC), where
distrustful parties can benefit from a joint collaborative computation on their private inputs. It requires
parties' individual inputs to remain hidden from each other during the computation, among other security
guarantees such as correctness, fairness, etc. [10]. Secure MPC is a powerful cryptographic tool with a vast
range of applications as it allows collaborative work with private data. Generic MPC protocols work by
expressing the function to evaluate as an arithmetic or Boolean circuit and then securely evaluating the
individual gates. These protocols are based on one of two main fundamental primitives [11-14]: **Oblivious
Transfer (OT)** and **Homomorphic encryption**, the former of which is the focus of this work.

A **1-out-of-2 OT** [15], is the task of sending two messages, such that the receiver can choose only one
message to receive, while the sender remains oblivious to this choice. The original protocol, now called **all-
or-nothing OT**, was proposed by Rabin in 1981 [16], where a single message is sent and the receiver obtains
it with $1/2$ probability. The two flavours of OT were later shown to be equivalent [17]. Notably, it has been
shown that it is possible to implement secure MPC using only OT as a building block [18,19]. Relevant to
our work is a variation of OT called **Random Oblivious Transfer (ROT)**, which is similar to 1-out-of-2 OT,
except that both the sent messages and the receiver's choice are randomly chosen during the execution of the
protocol. This can be seen as analogous to the key distribution task, in which both parties receive a random
message (the key) as output. By appropriately encrypting messages using the outputs of a ROT protocol as
a shared resource, it is possible to efficiently perform 1-out-of-2 OT. As an important consequence, parties
expecting to engage in MPC in the future can execute many instances of ROT in advance and save the
respective outputs as keys to be later used as a resource to perform fast OTs during an MPC protocol [20].
Because of this, we can think of ROT as a basic primitive for secure MPC.

In the context of quantum cryptography, OT is remarkable because, unlike classically, there exists a
reduction from OT to commitment schemes [21]. This result is somewhat undermined by the existence of
several theorems regarding the impossibility of unconditionally secure commitments both in classical [22] and
quantum [23,24] cryptography, and it was further proven impossible in the more general abstract cryptogra-
phy framework [25]. These results, in turn, imply that unconditionally secure OT itself is impossible. In light
of this, approaches with different technological or physical constraints on the adversarial power have been
proposed. Practical solutions based on hardware limitations, such as **bounded and noisy storage** [26-29],
have the disadvantage that the performance of such protocols decreases as technology improves.

Computationally-secure classical protocols have also been proposed [30-33], which work under the as-
sumptions of post-quantum public-key cryptography. Alternatively, we can take advantage of quantum
reduction from OT to commitments by implementing commitment schemes using (non-trapdoor) **one-way
functions (OWF)** such as **Hash functions** [34] and **pseudo-random generators** [35] which allows us to construct
OT from symmetric cryptography primitives. The existence of general OWFs is a weaker assumption than
public-key cryptography [36, 37], which requires the existence of the more restrictive **trapdoor OWFs**. This
difference is significant, as the latter are defined over mathematically rich structures, such as elliptic curves
and lattices, and the computational hardness of the associated problems is less understood than that of their
private-key counterparts. For an in-depth study of the relation between OT and OWFs see [38].

Having established that there is a theoretical merit in using computationally-secure quantum protocols
to implement secure MPC, it is also important to understand how practical quantum protocols compare with
currently used classical solutions in security, computational and communication complexity, and practical
speed in current setups. This work focuses in studying the **performance of a practical quantum ROT protocol**
and its potential advantages compared to currently used classical solutions for OT during MPC.

The idea of using quantum conjugate coding and commitments for oblivious transfer was originally
proposed by Crépeau and Kilian [17] and then refined by Bennet et al, in [21] with the **BBCS92 protocol**
(shown in Fig. 1). This construction has been extensively studied from the point of view of its theoretical
security [38-43]. However, while practical security analyses and experimental implementations have been
made for quantum OT in the noisy storage model [28, 29], there are **no works analyzing the quantum
resource requirements and the resulting performance of implementing the BBCS92 protocol using existing
computationally-secure commitment schemes based on OWFs**. Such analyses are needed to demonstrate
secure experimental implementations, and provide an important step in bringing quantum OT to real-world
usage.

**BBCS92 Quantum OT protocol**
**Parties:** The sender Alice and the receiver Bob.
1. Alice prepares $N$ entangled states of the form $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ and, for each state prepared, sends one of the
qubits to Bob.
2. Alice randomly chooses a measurement bases string $\Theta^A \in \{+, \times\}^N$ and, for each $i = 1, \ldots, N$ measures her
share of the $i$-th entangled state in the $\theta_i$ basis to obtain outcome $x^A_i$ and the outcome string $x^A = (x^A_1, \ldots, x^A_N)$.
3. Bob uses the same process to obtain the measurement bases and outcome strings $\Theta^B$ and $x^B$, respectively.
4. For each $i$, Bob commits $(\theta^B_i, x^B_i)$ to Alice.
5. Alice chooses randomly a set of indices $T \subset \{1, \ldots, N\}$ of some fixed size and sends $T$ to Bob.
6. For each $j \in T$, Bob opens the commitments associated to $(\theta^B_j, x^B_j)$.
7. Alice checks that $x^A_j = x^B_j$ whenever $\theta^A_j = \theta^B_j$ within the test set. If the test fails Alice aborts the protocol,
otherwise she sends the string $\Theta^A$ to Bob.
8. Bob separates the remaining indices in two sets: $I_0$ - the indices where Bob's measurement bases match Alice's,
and $I_1$ - the set of indices where their bases do not match. Then, he samples randomly $c$ and sends the ordered
pair $(I_c, I_{\bar{c}})$ to Alice.
9. Alice defines the strings $x^A_0, x^A_{\bar{c}}$ using the indices in the respective sets $(I_c, I_{\bar{c}})$. Then, she samples randomly a
function $f$ from a universal hash family, sends $f$ to Bob and outputs $m_c = f(x^A_c)$ and $m_{\bar{c}} = f(x^A_{\bar{c}})$ to Bob.
10. Similarly, Bob defines the string $x^B_{\bar{c}}$ from the set $I_0$ and outputs $m_{\bar{c}} = f(x^B_{\bar{c}})$ and $c$.

**Figure 1: Quantum oblivious transfer protocol based on commitments**

Motivated by practical considerations, we consider Naor-style **statistically binding and computationally
hiding commitments**, as these are well understood and efficient to implement (note that stronger commit-
ments can be considered, such as the quantum-based commitments studied in [38,42], however, implementing
those requires significantly more computational and quantum resources).

The contributions of this work can be summarized as follows:

We introduce the definition for a **quantum ROT protocol**, satisfying a **strong indistinguishability-based
security notion** equivalent to the one presented in [44], which generalizes the security of classical ROT proto-
cols. We present a protocol that realizes said quantum ROT based on the BBCS construction. The protocol
uses a **weakly-interactive string commitment scheme** which is statistically binding and computationally hid-
ing, and can be implemented in practice using current QKD setups.

We present a **formal finite-key security proof** of the proposed protocol accounting for noisy quantum
channels assuming only the existence of quantum-secure OWFs, together with security bounds as functions
of the protocol's parameters. We also present calculations for the maximum usable channel error, as well as
for the key rate as a function of the number of shared signals per instance of the protocol. Additionally, we
study the **composability properties** of said protocol. In particular, we show that there is a family of weakly-
interactive commitments which, when used in the quantum OT protocol, result in **universally composable
quantum OT in the classical access random oracle model**. We experimentally demonstrate our protocol using
current technology with a setup based on polarization-entangled photons. We also present a **security analysis**
which accounts for potential implementation-specific attacks and how they can be circumvented using an
appropriate reporting strategy. Finally, we compare our performance results with the performance of current
ROT solutions and point out the advantages and disadvantages of using quantum ROT in the context of
MPC.

# 2 Quantum Random Oblivious Transfer (ROT)

In this work, the concept of **indistinguishability** will be often used to compare the state of systems in a “real”
run of the protocol versus another “ideal” desired state. These relations are defined over families of quantum
states parametrized by the security parameter of the respective protocol. Hence, indistinguishability relations
are statements on the asymptotic behavior of the protocol as the security parameter is increased. For formal
definitions of both statistical and computational indistinguishability see Appendix A.

When talking about two indistinguishable families $\{\rho^{(k)}_1\}$ and $\{\rho^{(k)}_2\}$, if the parameter $k$ is implicit, we
will just refer to them as $\rho_1$ and $\rho_2$ and use the following notation to denote indistinguishability:

$$
\rho_1 \approx \rho_2 \quad \text{for statistically indistinguishable}; \\
\rho_1 \approx_{(c)} \rho_2 \quad \text{for computationally indistinguishable}.
$$

Additionally, in this work we consider protocols that can abort if certain conditions are satisfied. Math-
ematically, it is useful to consider the state of the aborted protocol as the zero operator. This means that
events that trigger the protocol to abort are described as **trace-decreasing operations**, and hence, the operator
representing the associated system at the end of the protocol is, in general, not normalized. The probability
of the protocol finishing successfully is given then by the trace of the final state of the output registers. Note
that the above definitions of indistinguishability can be naturally extended to non-normalized operators since
the outcomes of a quantum program can always be represented by the outcomes of a POVM $\{F_i\}$, whose
probabilities are given by $\text{Tr}[F_i \rho]$, which is a well defined quantity even for non-normalized $\rho$.

**Definition 2.1.** (Quantum Random Oblivious Transfer)
An $n$-bit **Quantum Random Oblivious Transfer** with security parameter $k$ is a protocol, without external
inputs, between two parties $S$ (the sender) and $R$ (the receiver) which, upon finishing, outputs the joint
quantum state $\rho_{M_0, M_1, C, M_{\bar{c}}}$ satisfying:

1.  (**Correctness**) The final state of the outputs when the protocol is run with both honest parties satisfies

$$
\rho_{M_0, M_1, C, M_{\bar{c}}} \approx \frac{P_{\text{succ}}}{2(2n+1)} \sum_{\substack{m_0, m_1 \in \{0, 1\}^n \\ c \in \{0, 1\}}} |m_0\rangle \langle m_0| M_0 \otimes |m_1\rangle \langle m_1| M_1 \otimes |c\rangle \langle c| C \otimes |m_{\bar{c}}\rangle \langle m_{\bar{c}}| M_{\bar{c}}, \quad (1)
$$

where $P_{\text{succ}} = \text{Tr}[\rho_{M_0, M_1, C, M_{\bar{c}}}]$ is the probability of the protocol finishing successfully.

2.  (**Security against dishonest sender**) Let $H_S$ be the Hilbert space associated to all of the sender's memory
    registers. For the final state after running the protocol with an honest receiver it holds that

$$
\rho_{S, C} \approx \rho_S \otimes U_C. \quad (2)
$$

3.  (**Security against dishonest receiver**) Let $H_R$ be the Hilbert space associated to all of the receiver's
    memory registers. For the final state after running the protocol with an honest sender, there exists a
    binary probability distribution given by $(P_0, P_1)$ such that

$$
\rho_{R, M_0, M_1} \approx \sum_b P_b \rho_{R, M_b} \otimes U_{M_{\bar{b}}}, \quad (3)
$$

The above properties define statistical security for each feature of the ROT protocol. If any of them holds for
the case of a dishonest party being limited to efficient quantum operations and the notion of computational
indistinguishability $\approx_{(c)}$ instead, we say that the ROT protocol is computationally secure in the respective
sense.

We expect the outputs $m_0, m_1, c$ to be uniformly distributed and the receiver always obtaining the
correct corresponding $m_{\bar{c}}$. The first property is typically called **correctness** and it states that, when both
parties follow the protocol, the probability of it not aborting and having incorrect outputs is neglible in
the security parameter. The probability $P_{\text{succ}}$ of the protocol finishing appears explicitly in this expression
as the success of quantum protocols often depends on external conditions, most notably the noise in the
quantum communication channels. For any specific value of $P_{\text{succ}}$ and any $\varepsilon'' < 1 - P_{\text{succ}}$ we say that, under
the associated external conditions, the protocol is $\varepsilon^{(\tau)}$-**robust**.

The second property, called **security against dishonest sender**, states that regardless of how much the
sender deviates from the protocol, their final quantum state (which includes all the information accessible
to them) is uncorrelated to the uniformly distributed value of the receiver's choice bit $c$. Analogously, the
third property, called **security against dishonest receiver**, states that even for a receiver running an arbitrary
program, by the end of the protocol there is always at least one of the two strings $m_0, m_1$ that is completely
unknown to them (denoted by $m_b$).

## 2.1 Additional schemes

In this section, we define the subroutines used inside of our main protocol. We start by defining a weakly-
interactive commitment scheme, which gets its name from the fact that the verifier publishes a single random
message at the start, which defines the operations that the committer performs.

**Definition 2.2.** (String commitment scheme)
Let $k, n \in \mathbb{N}$. A **weakly-interactive $n$-bit string commitment scheme** with security parameter $k$ is a family of
efficient (in $n$, as well as in $k$) programs $\text{com}, \text{open}, \text{ver}$
$$
\text{com}: \{0, 1\}^n \times \{0, 1\}^{n_s(k)} \times \{0, 1\}^{n_r(k)} \rightarrow \{0, 1\}^{n_c(k)}; \\
\text{open} : \{0, 1\}^n \times \{0, 1\}^{n_s(k)} \rightarrow \{0, 1\}^{n_o(k)}; \\
\text{ver}: \{0, 1\}^{n_c(k)} \times \{0, 1\}^{n_o(k)} \times \{0, 1\}^{n_r(k)} \rightarrow \{0, 1\}^n \cup \{\bot\}, \quad (4)
$$
such that
1.  (**correctness**) $\text{ver}(\text{com}(m, s, r), \text{open}(m, s), r) = m$ for all $m \in \{0, 1\}^n$, $s \in \{0, 1\}^{n_s}$, and $r \in \{0, 1\}^{n_r}$.

2.  (**hiding property**) For all $m_1, m_2 \in \{0, 1\}^n$ and $r \in \{0, 1\}^{n_r}$ the distributions for $\text{com}(m_1, s_1, r)$ and
    $\text{com}(m_2, s_2, r)$ are computationally (or statistically) indistinguishable in $k$ whenever $s_1, s_2$ are sampled
    uniformly.

3.  (**binding property**) For uniformly sampled $r$, the probability $\varepsilon_{\text{bind}}(k)$ that there exists a tuple $(\text{com}, \text{open}_1, \text{open}_2)$
    such that $\text{ver}(\text{com}, \text{open}_{1/2}, r) \neq \bot$ and
    $$
    \text{ver}(\text{com}, \text{open}_1, r) \neq \text{ver}(\text{com}, \text{open}_2, r), \quad (5)
    $$
    is negligible in $k$.

Weakly-interactive string commitment schemes can be implemented using common cryptographic prim-
itives like hash functions or pseudo-random generators. Most notably, Naor's commitment protocol [35]
provides a black box construction of weakly-interactive commitments from OWFs.

**Definition 2.3.** (Verifiable information reconciliation scheme)
Let $\mathcal{C} \subseteq \{0, 1\}^n \times \{0, 1\}^n$. A **verifiable one-way Information Reconciliation (IR) scheme** with security
parameter $k$ and leak $\ell$ for $\mathcal{C}$ is a pair of efficient programs $(\text{syn}, \text{dec})$ with
$$
\text{syn}: \{0, 1\}^n \rightarrow \{0, 1\}^{\ell}, \\
\text{dec}: \{0, 1\}^{\ell} \times \{0, 1\}^n \rightarrow \{0, 1\}^n \cup \{\bot\}, \quad (6)
$$
such that,
1.  (**correctness**) Whenever $(x, y) \in \mathcal{C}$ it holds that $\text{dec}(\text{syn}(x), y) = x$ except with negligible probability
    in $k$.
2.  (**verifiability**) For any $(x, y) \in \{0, 1\}^n \times \{0, 1\}^n$ it holds that either $\text{dec}(\text{syn}(x), y) = x$ or $\text{dec}(\text{syn}(x), y) =
    \bot$, except with negligible probability $\varepsilon_{\text{IR}}(k)$.

Due to Shannon's Noisy-channel coding theorem, the size of the leak $\ell$ for any IR scheme over a discrete
memoryless channel is lower bounded by $h(p)$, where $p$ represents the bit-error probability, and $h(\cdot)$ denotes
the binary entropy function. For concrete IR schemes, we can usually describe their efficiency using the ratio
between the scheme's leak and the theoretical optimal: $f = \frac{\ell}{n h(p)}$.

# 3 The protocol

In this section we present the protocol $\pi_{QROT}$ for an $n$-bit quantum ROT based on the primitives described
in the previous section and the use of quantum communication. The protocol's main security parameter is
$N_0$, which corresponds to the number of quantum signals sent during the quantum phase. Additionally, it
has two secondary security parameters $k, k'$, which define the security of the underlying commitment and IR
schemes, respectively.

In order to facilitate the **finite-key security analysis**, the description of $\pi_{QROT}$ features two statistical
tolerance parameters, denoted as $\delta_1, \delta_2$. The role of $\delta_1$ is to account for the error in the estimation of the
**Qubit Error Rate (QBER)**, while the role of $\delta_2$ is to account for the small variations in the frequency of
outcomes of 50/50 events. These parameters can be ignored (set to zero) when considering very large values
of $N_0$.

In the following description of the protocol we use the common **conjugate coding notation** used in BB84-
based protocols. The bit values $0, 1$ denote the **computational and Hadamard bases** for qubit Hilbert
spaces, respectively. For added clarity, we use the superscripts $A$ and $B$ to respectively denote Alice and
Bob. Additionally, we use variable $x$ to denote measurement outcomes and $\theta$ to denote measurement bases
(e.g. the pair $(\theta^A_i, x^A_i)$ denotes that Alice measured her $i$-th subsystem in the $\theta^A$ basis and obtained $x^A_i$ as
the outcome). We use $|\Phi^+\rangle$ to denote the Bell state $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$. Finally, we will use the **relative (or
normalized) Hamming weight function** $r_H: \{0, 1\}^n \rightarrow [0, 1]$ defined for any $x = (x_1, \ldots, x_n)$ as

$$
r_H(x) = \frac{1}{n} \sum_{i=1}^n x_i. \quad (7)
$$

**Parameters:**
*   **Parameter estimation sample ratio** $0 < \alpha < 1$
*   **Statistical tolerance parameters** $\delta_1, \delta_2$
*   **Maximum qubit error rate** $0 \leq P_{\text{max}} \leq 1/2$
*   **Coincidence block size** $N_0 \in \mathbb{N}$, **test set size** $N_{\text{test}} = \alpha N_0$, **minimum check set size** $N_{\text{check}} = (\frac{1}{2} - \delta_2) \alpha N_0$,
    and **raw string block size** $N_{\text{raw}} = (\frac{1}{2} - \delta_2)(1 - \alpha) N_0$
*   **Weakly-interactive 2-bit string commitment scheme** $(\text{com}, \text{open}, \text{ver})$, which is computationally hiding
    and statistically binding, with security parameter $k \in \mathbb{N}$ and associated string lengths $n_s, n_r, n_c, n_o$
*   **Verifiable one-way information reconciliation scheme** $(\text{syn}, \text{dec})$ on the set $\mathcal{C} = \{(x, y) \in \{0, 1\}^{N_{\text{raw}}} \times$
    $\{0, 1\}^{N_{\text{raw}}} : r_H(x \oplus y) < P_{\text{max}} + \delta_1\}$, with security parameter $k' \in \mathbb{N}$ and leak $\ell = f \cdot h(P_{\text{max}} + \delta_1)$
*   **Universal hash family** $\mathcal{F} = \{f_i: \{0, 1\}^{N_{\text{raw}}} \rightarrow \{0, 1\}^n\}_i$

**Parties:** The sender Alice and the receiver Bob.
**Protocol steps:**
**Quantum phase**
1. Alice generates the state $\bigotimes_{i=1}^{N_0} \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ and, for each state prepared, sends one of the
qubits to Bob.
2. Alice randomly chooses a measurement bases string $\Theta^A \in \{0, 1\}^{N_0}$ and, for each $i \in I = \{1, \ldots, N_0\}$ performs a measurement in the basis $\theta^A_i$ on her qubit of $|\Phi^+\rangle_i$ to obtain the
outcome string $x^A$.
3. Bob samples the string $\Theta^B \in \{0, 1\}^{N_0}$ and for each $i \in I$ performs a measurement in the basis $\theta^B_i$ on
his qubit of $|\Phi^+\rangle_i$ to obtain the outcome string $x^B$.
**Commit/open phase**
4. Alice uniformly samples the string $r \in \{0, 1\}^{n_r}$ and sends it to Bob.
5. For each $i \in I$, Bob samples a random string $s_i \in \{0, 1\}^{n_s}$, computes
$$
(\text{com}_i, \text{open}_i) = (\text{com}((\theta^B_i, x^B_i), s_i, r), \text{open}((\theta^B_i, x^B_i), s_i)), \quad (8)
$$
and sends the string $\text{com} = (\text{com}_i)_i$ to Alice.
6. Alice randomly chooses a subset $\text{test } I_t \subset I$ of size $\alpha N_0$ and sends $I_t$ to Bob.
7. For each $j \in I_t$, Bob sends $\text{open}_j$ to Alice.
8. For each $j \in I_t$, Alice checks that $\text{ver}(\text{com}_j, \text{open}_j, r) \neq \bot$. If so, she sets $(\hat{\theta}^B_j, \hat{x}^B_j) = \text{ver}(\text{com}_j, \text{open}_j, r)$.
Then, Alice computes the set $I_{\delta} = \{j \in I_t | \hat{\theta}^B_j = \theta^A_j\}$ and the quantity
$$
p = r_H(x^A_{I_{\delta}} \oplus \hat{x}^B_{I_{\delta}}), \quad (9)
$$
and checks that $|I_{\delta}| \geq N_{\text{check}}$ and $p \leq P_{\text{max}}$. If any of the checks fail Alice aborts the protocol.
**String separation phase**
9. Alice sends $\Theta^A_{I_{\bar{t}}}$ to Bob.
10. Bob constructs the set $I_0$ by randomly selecting $N_{\text{raw}}$ indices $i \in I_{\bar{t}}$ for which $\theta^A_i = \theta^B_i$. Similarly, he
constructs $I_1$ by randomly selecting $N_{\text{raw}}$ indices $i \in I_{\bar{t}}$ for which $\theta^A_i \neq \theta^B_i$. He then samples a random
bit $c$ and sends the ordered pair $(I_c, I_{\bar{c}})$ to Alice. If Bob is not able to construct $I_0$ or $I_1$, he aborts the
protocol.
**Post processing phase**
11. Alice computes the strings $(\text{syn}(x^A_{I_0}), \text{syn}(x^A_{I_1}))$ and sends the result to Bob.
12. Bob computes $\text{dec}(\hat{x}^B_{I_0}, \text{syn}(x^A_{I_0})) = y^B$. If $y^B = \bot$ Bob aborts the protocol.
13. Alice randomly samples $f \in \mathcal{F}$, computes $m^A_0 = f(x^A_{I_0})$ and $m^A_1 = f(x^A_{I_1})$, sends the description of $f$
to Bob and outputs $(m^A_0, m^A_1)$.
14. Bob computes $m^B = f(y^B)$ and outputs $(m^B, c)$.

## 3.1 Security and performance of the main protocol

We start by stating the main theorem regarding security of the proposed $\pi_{QROT}$ protocol.

**Theorem 3.1.** (Security of $\pi_{QROT}$)
*The protocol $\pi_{QROT}$ is a statistically correct, computationally secure against dishonest sender, and statisti-
cally secure against dishonest receiver $n$-bit $\text{ROT}$ protocol.*

A high-level proof of Theorem 3.1, including the derivation of the security bounds from Lemmas 3.1
and 3.2 can be found in Section 4 and further details can be found in Appendix B. The security of $\pi_{QROT}$ is
given by its main security parameter $N_0$, as well as the security parameters of the underlying commitment
and IR schemes $k$ and $k'$, respectively. These values can be computed for the statistical security features of
the protocol and are given by the following lemmas:

**Lemma 3.1.** (Correctness)
*The outputs of $\pi_{QROT}$ when run by honest sender and receiver satisfy*

$$
\rho_{M_0, M_1, C, M_{\bar{c}}} \approx_{\varepsilon} \frac{P_{\text{succ}}}{2(2n+1)} \sum_{\substack{m_0, m_1 \in \{0, 1\}^n \\ c \in \{0, 1\}}} |m_0\rangle \langle m_0| M_0 \otimes |m_1\rangle \langle m_1| M_1 \otimes |c\rangle \langle c| C \otimes |m_{\bar{c}}\rangle \langle m_{\bar{c}}| M_{\bar{c}}, \quad (10)
$$

*with*
$$
\varepsilon = 2^{-\frac{1}{2}(N_{\text{raw}}-n)} + 2 \varepsilon_{\text{IR}}(k'), \quad (11)
$$

*where $\varepsilon_{\text{IR}}$ is a negligible function given by the security of the underlying $\text{IR}$ scheme.*

**Lemma 3.2.** (Security against dishonest receiver)
*For the final state after running the protocol of $\pi_{QROT}$ with an honest sender, there exists a binary probability
distribution given by $(P_0, P_1)$ such that*

$$
\rho_{R, M_0, M_1} \approx_{\varepsilon'} \sum_b P_b \rho_{R, M_b} \otimes U_{M_{\bar{b}}}, \quad (12)
$$

*with*
$$
\varepsilon' = \sqrt{2} \left( e^{-(\frac{1}{2} (1-\alpha)^2 N_{\text{test}} \delta_1^2 + e^{-\frac{1}{2} N_{\text{check}} \delta_2^2})} + e^{-D_{\text{KL}}(\frac{1}{2} - \delta_2 || \frac{1}{2})(1-\alpha) N_0} + \varepsilon_{\text{bind}}(k) \right) \\
+ \frac{1}{2} \cdot 2^{N_{\text{raw}} (\frac{1}{2} - \sqrt{\frac{1}{4} - \delta_2^2} - h(P_{\text{max}} + \delta_1))} \quad (13)
$$

*where $H_R$ denotes the Hilbert space associated to all of the receiver's memory registers and $\varepsilon_{\text{bind}}$ is a negligible
function given by the security of the underlying commitment scheme.*

We can use these results to find the minimum requirements, both in terms of channel losses and number
of shared entangled qubits, necessary to securely realize $\text{ROT}$ for a given security level. We focus on the
quantity

$$
\varepsilon_{\text{max}} = \varepsilon + \varepsilon'. \quad (14)
$$

For the purposes of this analysis, we assume that the commitment and IR schemes, as well as their security
parameters $k, k'$, are appropriately chosen to satisfy the desired security level and we focus on the dependence
of $\varepsilon_{\text{max}}$ on the channel error rate, characterized by the parameter $P_{\text{max}}$, and the number of quantum signals
$N_0$. We are also interested in a quantity known as the **secret key rate** $R_{\text{key}}$. For given values of $N_0, \alpha, \delta_1, \delta_2$,
$P_{\text{max}}$, and $\varepsilon_{\text{max}}$, let $n_{\text{max}}$ be the largest number for which the associated $n_{\text{max}}$-bit $\text{ROT}$ has at least security
$\varepsilon_{\text{max}}$, then

$$
R_{\text{key}} = \frac{n_{\text{max}}}{N_0}, \quad (15)
$$

[IMAGE: Figure 2: Maximum key rate output $\frac{n}{N_0}$ versus error rate $P_{\text{max}}$. The blue line represents the upper bound for the key rate, when $N_0 \rightarrow \infty$, $\alpha, \delta_1, \delta_2$ are taken to be 0 and $f = 1$. The orange line represents a more typical case with $\alpha = 0.35$, $\delta_1 = 0.01$, $\delta_2 = 0.025$, and $f = 1.2$.]

[IMAGE: Figure 3: Maximum key rate behaviour as a function of $N_0$ for different security levels. Parameter values used are $\alpha = 0.35$; $\delta_1 = 9.20 \times 10^{-3}$; $\delta_2 = 3.00 \times 10^{-3}$; $P_{\text{max}} = 0.01$; $f = 1.2$. The horizontal axis is $N_0$ and the vertical axis is $R_{\text{key}}$.]

[IMAGE: Figure 4: Critical value $N_{\text{crit}}$ of the number of shared qubits needed to obtain positive key rates as a function of the security level. The values of $N_{\text{crit}}$ were computed using the parameters $\alpha, \delta_1, \delta_2$ that minimize the value of $N_{\text{crit}}$ for each $\varepsilon_{\text{max}}$. The horizontal axis is $\varepsilon_{\text{max}}$ ranging from $10^{-110}$ to $10^{-10}$ and the vertical axis is $N_{\text{crit}}$ ranging from $5.0 \times 10^6$ to $3.5 \times 10^7$. The curve decreases as $\varepsilon_{\text{max}}$ increases.]

represents the ratio in which the original measurements of the shared qubits “transform” into the oblivious
key. In Figure 2 we can see the behavior of $R_{\text{key}}$ as $P_{\text{max}}$ increases. Note that, similarly to the case of
quantum key distribution, there is a **critical error** $P_{\text{crit}}$ after which $R_{\text{key}}$ becomes negative and no secure key
can be generated. The value of $P_{\text{crit}}$ is upper bounded by $\approx 0.028$, which is achieved when we set $\alpha, \delta_1, \delta_2 \rightarrow 0$
and $N_0 \rightarrow \infty$.

Another important aspect to analyze is the relation between $R_{\text{key}}$ and $N_0$, which is shown in Figure 3.
Fixing the $\alpha, \delta_1, \delta_2, P_{\text{max}}$, there is a clearly marked **phase transition-like behaviour** in which, for each $\varepsilon_{\text{max}}$,
there is a critical value of $N_0 = N_{\text{crit}}$ before which $R_{\text{key}} = 0$, and after which it quickly reaches its maximum
value. This result comes from the fact that the parameter estimation requires relatively big sample sizes
to reach high confidence. It shows that, even for small $n$, there is a minimum amount of entangled qubits
needed to be shared. In some cases, for instance, generating a 1-bit oblivious key or a 128-bit one may
have similar costs in terms of quantum communication. Because the use of resources of the protocol scales
with $N_0$, the parameters $\alpha, \delta_1, \delta_2$ should be chosen such that $N_{\text{crit}}$ is the smallest. Figure 4 exemplifies the
dependency of $N_{\text{crit}}$ on $\varepsilon_{\text{max}}$.

## 3.2 Experimental implementation performance

An experiment was implemented to test the performance of the $\pi_{QROT}$ protocol with contemporary technol-
ogy. Data was acquired using a picosecond pulsed photon source in a Sagnac configuration [45], producing
wavelength degenerate, polarization-entangled photons at $1550\text{nm}$. In this setup, entangled photons were
produced via **spontaneous parametric down conversion (SPDC)** by applying a laser pump beam into a $30\text{mm}$
long **periodically-poled potassium titanyl phosphate (ppKTP) crystal**. The photon pairs were split using a
half-wave plate (HWP) and a polarizing beam splitter (PBS), and then sent to each party where they are
detected using **superconducting nanowire single-photon detectors**.

To test the $\text{OT}$ speed of this implementation, different values for the power $P$ of the laser pump were
tested, as well as the use of **multiplexing**. As the $P$ increases, the amount of coincidences detected per second
$R_c$ increases, but the fidelity of the produced entangled pairs decreases, resulting in larger values for qubit
error rate, which is represented by the protocol parameter $P_{\text{max}}$. The number of maximum potential $\text{OT}$
instances per second is computed as

$$
R_{OT} = \frac{R_c}{N_{\text{crit}}}, \quad (16)
$$

where $N_{\text{crit}}$ is computed using the optimal values of $\alpha, \delta_1, \delta_2$ for the respective error rate $P_{\text{max}}$ and undetected
multi-photon rate $P_{\text{multi}}$ associated to $P$, assuming perfectly efficient information reconciliation, $f = 1$ (see
Section 5 for the details on the implementation and its security). As seen in Figure 5, for this implementation,
the additional coincidence rate gained by increasing $P$ is not enough to compensate for the induced increased
error. This result is not immediately obvious, as $N_{\text{crit}}$ does not depend explicitly on $P_{\text{max}}$. The decrease in
performance comes from the fact that increasing $P_{\text{max}}$ limits the values that $\delta_1$ can have while maintaining
positive key rates. This restriction on the values of $\delta_1$ ultimately results in an increase in $N_{\text{crit}}$ and therefore,
a reduction on $R_{OT}$.

[IMAGE: Figure 5: Maximum potential ROT rates as a function of the pump power P for $\varepsilon_{\text{max}} = 10^{-7}$. The horizontal axis is $P$ (mW) ranging from 50 to 300, and the vertical axis is $OT/s$ ranging from 0.06 to 0.10. The plot shows a peak around $P=170$ mW.]

Table 1 shows an example of the performance of the protocol in a real-world implementation using the
data from the experimental setup. For the commit/open phase, the weakly-interactive string commitment
protocol introduced in [35] was implemented using the **BLAKE3 hash function algorithm** as a one-way
function. For the post-processing phase, a **low density parity check (LDPC) code** was used for $\text{IR}$, and
random binary matrices were used to implement the universal hash family for privacy amplification. We
evaluated the performance by the number of 128-bit $\text{ROT}$ instances able to be completed per second (It is
worth noting that, using a Mac mini M1 2020 16GB computer, the post-processing throughput was enough
to handle all the data from the experiment, the bottleneck being the quantum signal generation rate).

**Table 1: Table of protocols parameters and the resulting performance. The values of $N_0$ and $\delta_1$ and the laser pump power were optimized to yield the highest $\text{ROT}$ rate for an $\text{LDPC}$ code with efficiency $f = 1.61$.**

| Parameter | Symbol | Value |
|:---|:---|:---|
| Message size (bits) | $n$ | 128 |
| Security level | $\varepsilon_{\text{max}}$ | $1.91 \times 10^{-8}$ |
| Cost in quantum signals | $N_0$ | $5.86 \times 10^6$ |
| Max allowed QBER | $P_{\text{max}}$ | $1.14\%$ |
| Testing set ratio | $\alpha$ | $0.35$ |
| Statistical parameter 1 | $\delta_1$ | $9.00 \times 10^{-3}$ |
| Statistical parameter 2 | $\delta_2$ | $3 \times 10^{-3}$ |
| IR verifiability security | $\varepsilon_{\text{IR}}$ | $2^{-32}$ |
| Commitment binding security | $\varepsilon_{\text{bind}}$ | $2^{-32}$ |
| Efficiency of IR | $f$ | $1.64$ |
| Max allowed multi-photon rate | $P_{\text{multi}}$ | $3.67 \times 10^{-3}$ |
| **ROT rate** | $R_{OT}$ | **$0.023 \text{ ROT/s}$** |

# 4 Security Analysis

In this section, we prove the main security result, which relates the overall security of the protocol as a
function of its parameters $N_0, \alpha, \delta_1$, and $\delta_2$ in Theorem 3.1. For clarity of presentation, we have compacted
some of the properties into lemmas, for which detailed proofs can be found in Appendix B. Definitions and
properties of entropic quantities can be found in Appendix A

## 4.1 Correctness

In order to prove correctness we need to show that either the protocol either finishes with Alice outputting
uniformly distributed messages $m_0, m_1$ and Bob outputting a uniformly random bit $c$ and the corresponding
message $m_{\bar{c}}$, or it aborts, except with negligible probability.

Recall that we model the aborted state of the protocol as the zero operator. This way, whenever we have
a mixture of states, some of which trigger aborting and some that do not, the abort operation removes the
events that trigger it from the mixture, effectively reducing its trace by the probability of aborting. There
are three instances where the protocol can abort: first during Step (7) if the estimated qubit error rate is
larger than $P_{\text{max}}$; the second one is during Step (9) if Bob does not have enough (mis)matching bases to
construct the sets $I_0, I_1$; and finally during Step (11) if the IR verification fails. The probability of aborting
in Steps (7) and (11) depends on the particular transformation that the states undergo when being sent
from Alice's to Bob's laboratory, about which we make no assumptions. We can group these three abort
events and denote by $P_{\text{abort}}$ the probability of the protocol aborting by the end of Step (11). The state at
this point can be written as $(1 - P_{\text{abort}}) \rho'$, where $\rho'$ represents the normalized state conditioned that the
protocol has not aborted by this point. As Lemma 4.1 states, the verifiability property of the Information
Reconciliation scheme guarantees that the states that “survive” past Step (11) have the property that Bob's
corrected string $y^B$ is the same as Alice's outcome string $x^A_{I_0}$, which is uniformly distributed.

**Lemma 4.1.** *Let $X^A_{I_0}, X^A_{I_1}, C, Y^B$ denote the systems holding the information of the respective values $x^A_{I_0}, x^A_{I_1}, c,$
*and* $y^B$ *of* $\pi_{QROT}$. *Denote by* $\rho^T$ *the parties' joint state at the end of Step (11) conditioned that Bob con-
structed the sets* $(I_0, I_1)$ *during Step (9) and the protocol has not aborted. Assume both parties follow the
Steps of the protocol, then*

$$
\rho^T_{X^A_{I_0}, X^A_{I_1}, C, Y^B} \approx_{\varepsilon_{\text{IR}}(k')} \rho^T_{X^A_{I_0}, X^A_{I_1}, C, Y^B}, \quad (17)
$$

*where* $\varepsilon_{\text{IR}}(k')$ *is a negligible function given by the security of the underlying Information Reconciliation
scheme, $k'$ its associated security parameter, and*

$$
\rho^T_{X^A_{I_0}, X^A_{I_1}, C, Y^B} = \frac{1}{2(2^{N_{\text{raw}}}+1)} \sum_{x_{I_0}, x_{I_1}} \sum_c |x_{I_0}\rangle \langle x_{I_0}| X^A_{I_0} \otimes |x_{I_1}\rangle \langle x_{I_1}| X^A_{I_1} \otimes |c\rangle \langle c| C \otimes |x_{I_0}\rangle \langle x_{I_0}| Y^B, \quad (18)
$$

## 4.2 Security against dishonest sender

For this scenario we show that, in the case of an honest Bob and Alice running an arbitrary program, the
resulting state after the protocol successfully finishes satisfies Eq. (2). In other words, independently of
what quantum state Alice shares at the beginning of the protocol and which operations she performs on
her systems, her final state is independent of the value of $c$. We assume that Alice's laboratory consists of
everything outside Bob's. In particular, this means that she controls the environment, which includes the
transmission channels. We also assume that Alice is limited to performing efficient computations.

Let $A$ be the system consisting of all of Alice's laboratory after Step (1) of the protocol, that is, $A$ contains
her part of the shared system and every other ancillary system she may have access, but does not contain any
system from Bob's laboratory, including Bob's part of the system shared in Step (1). During the execution of
the protocol, Alice receives external information from Bob exactly three times: the commitment information
shared during Step (4), the opening information $\text{open}_{I_t}$ for the commitments associated to the test set $I_t$ in
Step (6), and the information of the pair of sets $(J_0, J_1) = (I_c, I_{\bar{c}})$ during Step (9). Let $COM = (COM)_I$
and $OPEN = (OPEN)_{I_t}$ be the respective systems used by Bob to store the information of the strings
$\text{com} = (\text{com}_i)_I$ and $\text{open} = (\text{open}_j)_{I_t}$, and let $SEP$ be the system holding the string separation information
$(J_0, J_1)$. We want to show that, by the end of the protocol, the state of the system $A, COM, OPEN_I, SEP, C$
satisfies:

$$
\rho_{A, COM, OPEN_I, SEP, C} \approx_{(c)} \rho_{A, COM, OPEN_I, SEP} \otimes U_C. \quad (22)
$$

To guarantee that Alice will not be able to obtain information about the value of $c$ during the string separation
phase, it is necessary to show that Alice does not have access to the information of Bob's bases choices $\Theta^B_{I_t}$
from the commitments sent by Bob during Step (4) of the protocol. As shown by Lemma 4.2, the shared
state of the parties after the commitment information is sent is computationally indistinguishable from a
state where Alice's information is independent of $\Theta^B_{I_t}$.

**Lemma 4.2.** *Assuming Bob follows the protocol, for any $J \subseteq I$, the state of the system $A, COM, OPEN_J, \Theta^B_J$
after Step (4) satisfies*

$$
\rho_{A, COM, OPEN_J, \Theta^B_J} \approx_{(c)} \rho_{A, COM, OPEN_J} \otimes U_{\Theta^B_J}. \quad (23)
$$

At Step (8) of the protocol, Alice sends Bob the system $\Theta^A_{I_{\bar{t}}}$ intended to have the information of her
measurement bases. Bob then is able to determine the indices for which $\Theta^A$ and $\Theta^B$ coincide. With this
information, he randomly selects sets $I_0, I_1 \in \mathcal{I}_{\bar{t}}$ of size $N_{\text{raw}}$ for which all indices are associated with matching
(for $I_0$) or nonmatching (for $I_1$) bases. Then he computes $(J_0, J_1) = (I_c, I_{\bar{c}})$, by flipping the order if the pair
$(I_0, I_1)$ depending on the value of $c$. Clearly, $(J_0, J_1)$ depend on both $\Theta^A_{I_{\bar{t}}}$ and $c$, but as Lemma 4.3 states,
any correlation between $(J_0, J_1), c$, and Alice's information disappears if one does not have access to $\Theta^B_{I_t}$.

**Lemma 4.3.** *Denote by $A'$ the system representing Alice's laboratory at the start of Step (9). Let $\mathcal{E}(I_t) :$
$\mathcal{D}(H_{A'}, \rho^B_{\Theta^B_{I_t}, C}) \rightarrow \mathcal{D}(H_{A'}, \rho^B_{\Theta^B_{I_{\bar{t}}}}, C, SEP)$ be the quantum operation used by Bob to compute the string sep-
aration information $(J_0, J_1)$ during Step (9) of the protocol. The resulting state after applying $\mathcal{E}(I_t)$ to a
product state of the form*

$$
\mathcal{E}(I_t) (\rho_{A'} \otimes U_{\Theta^B_{I_{\bar{t}}}} \otimes U_C) = \sigma_{A', \Theta^A_{I_{\bar{t}}}, \Theta^B_{I_{\bar{t}}}, C, SEP} \quad (24)
$$

satisfies
$$
\text{Tr}_{\Theta^A_{I_{\bar{t}}}, \Theta^B_{I_{\bar{t}}}} [\sigma_{A', \Theta^A_{I_{\bar{t}}}, \Theta^B_{I_{\bar{t}}}, C, SEP}] = \sigma_{A'} \otimes \sigma_{SEP} \otimes U_C. \quad (25)
$$

## 4.3 Security against dishonest receiver

We consider now the scenario in which Alice runs the protocol honestly and Bob runs an arbitrary program.
For this analysis, note that Alice trusts her quantum state preparation and detection. We want to show
that the state after finishing the protocol successfully satisfies Eq. (3). This means that the state at the
end of the protocol can be described as a mixture of states where Bob's system is uncorrelated with at least
one of the two strings outputted by Alice. Similarly to the dishonest sender's case, we assume that Bob's
laboratory consists of everything outside Alice's, which means that he controls the communication channels
and the environment. However, we do not assume that Bob is restricted to efficient computations.

The values of Alice's output strings depend on several quantities: Alice's measurement outcomes, the
choice of the $I_t, J_0, J_1$ subsets, and the choice of hashing function $f$ during the post-processing phase of the
protocol. From all of these, the only ones that are not made explicitly public during the protocol's execution
are Alice's measurement outcomes. Instead, partial information of these outcomes is revealed at different
steps of the protocol. Let $x^A_{J_0}, x^A_{J_1}, x^A_{\bar{J}_d}$ be the sub-strings of measurement outcomes used to compute Alice's
outputs $m_0, m_1$, respectively, and let $R$ denote Bob's system at the end of the protocol (which includes all
the systems that Alice sent during the execution of the protocol). In order to prove security we need to
show that the joint state of the system $X^A_{J_0}, X^A_{J_1}, R$ can be written as a mixture of states $\rho^b$ (with $b \in \{0, 1\}$)
such that the **conditional min-entropy** $H_{\min}(X^A_b | R)_{\rho^b}$ is high enough, so that we can use the **leftover hash
Lemma A.4** to guarantee that the outcome of the universal hashing $m_{\bar{b}} = f(x^A_{\bar{b}})$ is uncorrelated with $R$.

At the start of the protocol the parties share a completely correlated entangled system. If the parties
make measurements as intended, their outcomes will be only partially correlated, but if Bob was able to
postpone his measurement until after Alice's reveals her measurement bases, Bob could potentially obtain
the whole information of $x^A$ by measuring in the appropriate basis on his system. To prevent this, Bob is
required to commit his measurement bases and results to Alice before knowing which set is going to be tested.
Then a **statistical test** is performed in Step (7) to estimate the correlation of Alice's measurement outcomes
with with the ones that Bob committed. As Lemma 4.4 states, any state passing the aforementioned test
is such that, regardless of how Bob defines the sets $(J_0, J_1)$ during the string separation phase, there is a
minimum of uncertainty that he has with respect to Alice's measurement outcomes. Recall that, when Alice
is honest, the overall state of the protocol before Step (8) will be a partially classical state, which could be
written as a mixture over all of Alice's classical information. Let $\mathcal{T} = (x^A_{I_t}, \Theta^A_{I_t}, r, \text{com}, I_t, I_{\bar{t}}, \text{open}_{I_t})$ denote
the **transcript** of the protocol up to Step (8), and let $\rho_{X^A B}(\mathcal{T}, J_0, J_1)$ be the joint state of Alice's measurement
outcomes and Bob's laboratory conditioned to $\mathcal{T}, J_0, J_1$.

**Lemma 4.4.** *Assuming Alice follows the protocol, let $\mathcal{T}, SEP, B$ denote the systems of the protocols tran-
script, the strings $J_0, J_1$, and Bob's laboratory at the end of Step (9) of the protocol, and let $\rho_{\mathcal{T}, SEP, X^A, B}$ be
the state of the joint system at that point. There exists a state $\rho'_{\mathcal{T}, SEP, X^A, B}$, which is classical in $\mathcal{T}$ and $SEP$*

$$
\rho_{X^A B} = \sum_{J_0, J_1} P(\mathcal{T}, J_0, J_1) \rho_{X^A B}(\mathcal{T}, J_0, J_1), \quad (29)
$$

*where* $P(\mathcal{T}, J_0, J_1)$ *defines a probability distribution which is dependent on Bob's behavior during the
previous steps. We can now separate the $\rho_{X^A B}(\mathcal{T}, J_0, J_1)$ in two categories depending on which of the
$x^A_{J_0}, x^A_{J_1}$ is the least correlated with Bob's system. Consider the function $b(\mathcal{T}, J_0, J_1)$ to be equal to $0$ if
$H_{\min}(X^A_{J_0} | X^A_{J_1} B)_{\rho(\mathcal{T}, J_0, J_1)} \geq H_{\min}(X^A_{J_1} | X^A_{J_0} B)_{\rho(\mathcal{T}, J_0, J_1)}$, and equal to $1$ otherwise. By regrouping the terms
from (29) for which the value of $b$ is the same, we can rewrite the joint state as:*

$$
\rho_{X^A B} = \sum_{b \in \{0, 1\}} P_b \rho_{X^A B}^b, \quad (30)
$$

*where, from Lemma 4.4 and recalling that, as Lemma A.3 (5) states, the min-entropy of a mixture is lower
bounded by that of the term with the least min-entropy, we know that*

$$
H_{\min}(X^A_b | X^A_{\bar{b}} B)_{\rho^b} \geq N_{\text{raw}} \left( \frac{1}{2} - \frac{2 \delta_2}{1 - 2\delta_2} - h\left(\frac{P_{\text{max}} + \delta_1}{\frac{1}{2} - \delta_2}\right) \right). \quad (31)
$$

## 4.4 Composability considerations

Since $\text{OT}$ protocols are mainly used as a subroutine of larger applications it is important to understand
the **composability properties** of $\pi_{ROT}$. In general, this is done through **simulation-based composability
frameworks**. As mentioned in Section 1, this protocol is based on the $\text{BBSC}$ construction, which has been
proven secure in the quantum Universal Composability ($\text{UC}$) framework by Unruh [41] assuming access to an
ideal commitment functionality. This means that we can understand the composability properties of $\pi_{ROT}$
by understanding the respective properties of the underlying weakly-interactive commitment protocol.

It is well known that $\text{UC}$ commitments are impossible to realize in the plain model [22,25]. Because of
this, protocols are often analyzed within a **hybrid model**, where the parties have access to some base external
functionality. We show in Appendix 4.4 that there exists a family of commitment schemes that are both
**weakly-interactive** and **UC-secure in the classical access Random Oracle Model (ROM)** [46]. This, in tandem
with the aforementioned reduction of $\text{OT}$ to commitments, results in the following theorem:

**Theorem 4.1.** *There exists a family of weakly-interactive commitment schemes in relation to which $\pi_{ROT}$
is $\text{UC}$-secure in the classical access $\text{ROM}$.*

In relation to Theorem 4.1, we want to emphasize that, even though limiting the access to the random
oracle to be classical may seem at first strong in the context of a quantum protocol (where the parties are
required access to some quantum capabilities), it has little impact in the resulting security of larger $\text{MPC}$
protocols for which the security is analyzed in the classical setting.

Finally, we would like to stress the merits of Def. 2.1 by itself. In particular, this definition was studied
in [47] and [44] and stated to ensure security when the protocol is executed sequentially. Furthermore,
the indistinguishability properties stated in Def. 2.1 provide a very strong security guarantee and, because
the protocol does not have external inputs and the indistinguishability relations include arbitrary external
systems, these properties will still hold in any environment, which makes it relatively straightforward to
analyze as part of bigger applications.

# 5 Experimental Implementation

## 5.1 Description of the Setup

A schematic representation of the experimental setup can be seen in Fig. 6. Spontaneous parametric down
conversion ($\text{SPDC}$), attributed to Alice, is used to create polarization entangled photon pairs in the state
$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|HH\rangle + |VV\rangle)$, which are coupled into optical fiber. One photon is sent through a $50/50$ fiber
beam splitter, probabilistically routing it to one of two polarization projection stages. There, a quarter-wave
plate ($\text{QWP}$), a half-wave plate ($\text{HWP}$) and a polarizing beam splitter ($\text{PBS}$) are used to project the photons
state onto the linear ($\text{H}/\text{V}$) or diagonal ($\pm / -$) basis, respectively. All photons are sent to superconducting
nanowire single photon detectors ($\text{SNSPDs}$) and their arrival time is recorded using a time tagging module
($\text{TTM}$). The second photon of the state $|\Phi^+\rangle$, attributed to Bob, travels through an equivalent probabilistic
projection setup.

[IMAGE: Figure 6: Experimental setup. Polarization-entangled photon pairs are created using spontaneous parametric down conversion. Alice's and Bob's photons are individually fiber coupled and each sent to 50/50 fiber beam splitters, which probabilistically route them to free-space polarization projection stages - one projecting onto the linear, and one onto the diagonal basis each for Bob and Alice. The schematic shows a Ti:sapph laser pumping a ppKTP crystal, creating entangled photons that are split and routed via fiber beam splitters, QWP/HWP, polarizing beam splitters, mirrors, and dichroic mirrors to SNSPDs for detection by both Alice and Bob.]

Entangled photon pairs are generated using collinear type-II $\text{SPDC}$ in a periodically poled $\text{KTiOPO}_4$-
crystal with a poling period of $46.2 \, \mu\text{m}$ inside of a sagnac interferometer. The pump light is produced by
a pulsed $\text{Ti:Sapphire}$ laser ($\text{Coherent Mira } 900\text{HP}$) with a pulse width of $2.93 \, \text{ps}$ and a central wavelength
of $\lambda_P = 773 \, \text{nm}$, creating degenerate single-photon pairs at $\lambda_S = \lambda_I = 1546 \, \text{nm}$. The laser's inherent pulse
repetition rate of $76 \, \text{MHz}$ is doubled twice to $304 \, \text{MHz}$ using a passive temporal multiplexing scheme [48].
More precisely, for $n$ simultaneously emitted pairs and $k$ multiplexing stages, each doubling the repetition
rate, higher-order pair production events are attenuated by a factor of $1/(2^k)^{n-1}$. In our experiment, $k = 2$,
so this scheme reduces the probability of emitting a double pair $(n = 2)$ by a factor of 4 compared to a source
relying on the pump's inherent repetition rate, while the single-pair emission probability remains constant.
Finally, about $100 \, \text{m}$ of single mode fiber separate the experimental setup from the $1\text{K}$ cryostat housing the
$\text{SNSPDs}$ with a detection efficiency of around $95\%$ and a dark-count rate of around $300 \, \text{Hz}$.

We note that our entanglement-based implementation presents two main technological advantages over
prepare-and-measure configurations:

*   It circumvents the need for a **certified quantum random number generator** or for classical **pseudo-
    randomness** that may compromise the security of the quantum phase: instead of feeding random (or
    pseudorandom) sequences into the active polarization modulator of a prepare-and-measure scheme, the
    choice of $\text{BB84}$ state is performed in a passive and uniformly random way by the beamsplitters present
    in both Alice and Bob's measurement setups (also known as “**remote state preparation**”).
*   In free space, it avoids the need for **active polarization modulation**, which imposes a strict upper limit
    on the protocol's repetition rate governed by the bandwidth of the $\text{Pockels Cell}$ and its high-voltage
    amplifier, typically achieving a few hundred $\text{kHz}$ to a few hundred $\text{MHz}$ [49]. By generating entangled
    photons that are passively projected onto one of the four $\text{BB84}$ states instead, our $\text{OT}$ rate is not
    limited by any active prepare-and-measure encoding routine, but only by our picosecond-pulsed pump
    rate of around $300 \, \text{MHz}$. With other $\text{SPDC}$ sources reaching the $\text{GHz}$ [50] to tens of $\text{GHz}$ regimes [51],
    our passive state preparation routine can perform even better.

## 5.2 Practical protocol

The protocol is identical to that from $\pi_{ROT}$ as described in Section 3, with the following amendments:

*   The parties agree on an additional parameter $P_{\text{multi}}$ — the accepted ratio of multi-photon events.

*   During the quantum phase of the protocol, Alice may observe detection patterns that are incompatible
    with the emission of a single photon pair. Instead of sharing $N_0$ states in Step (1), she continues sharing
    states until, after agreeing on **coincidence time-tags** with Bob, the parties obtain $N_0$ coincidences
    associated with single-photon events on Alice's side. Let $N_{\text{tot}}$ be the number of coincidences obtained
    at this point and $N_{\text{multi}} = N_{\text{tot}} - N_0$. Alice computes the value

$$
P_{\text{multi}} = \frac{N_{\text{multi}}}{3 N_{\text{tot}}}, \quad (37)
$$

    and aborts the protocol if $P'_{\text{multi}} \geq P_{\text{multi}}$.

*   Similarly to Alice, Bob may also observe **multi-click patterns**. While reporting its detection events he
    uses the following rules:
    (a) **1 click:** assign the correct measured bit value and report a successful round
    (b) **2 clicks from the same basis:** assign a random bit value to the measurement result and report a
    successful round
    (c) **any other click pattern:** report an unsuccessful round

## 5.3 Practical security

Any photonic implementation of quantum cryptography presents experimental imperfections, which can be
exploited by dishonest parties to enhance their cheating probability and violate ideal security assumptions.
Important examples of such imperfections include **multiphoton noise, lossy/noisy quantum channels, non-unit
detection efficiency and detector dark counts**.

**Dishonest sender.** In our experiment, threshold detectors cannot resolve the incident photon number,
and unexpected click patterns can occur. For example, several of the four detectors may simultaneously click
for a given round, which leads to an inconclusive measurement outcome that has to be back-reported by the
honest receiver. This in turn allows a dishonest sender to gain a significant amount of information about the
receiver's measurement basis choice. Adopting the reporting strategy presented above makes the protocol
secure against this type of attack. For a complete analysis of both the attack and its countermeasures,
see [52].

**Dishonest receiver.** Due to **Poisson statistics** in the $\text{SPDC}$ process, emission of double pairs can occur
for a given round. When the two photons kept by the sender are projected onto the same state (i.e. only
a single click is recorded in the four detectors), the two photons sent to Bob have the same polarization.
In this case, a dishonest receiver can split the two photons and measure one in each basis. Assuming 4
detectors with equal efficiencies (which can be guaranteed in practice by appropriate attenuation the higher
efficiency ones), and using the fact that for an $\text{SPDC}$ source, whenever multiple pairs are produced, there is
no correlation among them, we know that the number of undetected multi-photon events is approximately
$1/3$ of the number of detected ones. We can then estimate the probability $P'_{\text{multi}}$ of an accepted coincidence to
be associated with a multi-photon event with Eq. (37).

Note that the statistical check performed by Alice in the second step of the amended protocol (Section
5.2) ensures security under the assumption that there is no **coherence in the photon-number basis**. This is
the case in our implementation, since $\text{SPDC}$ produces states of the form $\sum_{n=0}^{100} \sqrt{c_n} |n\rangle_1 |n\rangle_2$ in the number
basis $\{|n\rangle\}$ [53], leaving the individual subsystems in incoherent mixtures of the form $\sum_{n=0} c_n |n\rangle \langle n|$.

To account for the leakage caused by undetected multi-photon emissions to our $\text{OT}$ rate expression, we
effectively grant Bob an amount of information about Alice's measurement outcomes equal to the number of
indices in $I_{\bar{t}}$, associated to multi-photon events, upper bounded by $P_{\text{multi}}(1 - \alpha) N_0$ for large $N_0$. Subtracting
this leak to the total entropy expression in Eq. (32) leads to a version of Lemma 3.2 for security against
dishonest receiver corrected for the experimental implementation, which differs from the theoretical version
by replacing Eq (13) with

$$
\varepsilon_{\exp} = \sqrt{2} \left( e^{-(\frac{1}{2} (1-\alpha)^2 N_{\text{test}} \delta_1^2 + e^{-\frac{1}{2} N_{\text{check}} \delta_2^2})} + e^{-D_{\text{KL}}(\frac{1}{2} - \delta_2 || \frac{1}{2})(1-\alpha) N_0} + \varepsilon_{\text{bind}}(k) \right) \\
+ \frac{1}{2} \cdot 2^{N_{\text{raw}} (\frac{1}{2} - \sqrt{\frac{1}{4} - \delta_2^2} - h(\frac{P_{\text{max}} + \delta_1}{\frac{1}{2} - \delta_2}))} \quad (38)
$$

# 6 Discussion

Using Naor's protocol [35] in conjunction with a linear time $\text{OWF}$ (such as a hash function fron the $\text{SHA3}$ or
$\text{BLAKE}$ family), it is possible to implement the required 2-bit commitment in linear time in $k$. On the other
hand, using an $\text{LDPC}$ code with soft-decision decoding and hash based verification, one can implement an $\text{IR}$
scheme which is linear in both the block size $N_{\text{raw}}$ (and therefore $N_0$) and $k'$. Finally, by taking the universal
hash set $\mathcal{F}$ to be the set of $\text{Toeplitz}$ matrices of size $N_{\text{raw}} \times n$, and using the $\text{FFT}$ algorithm for matrix-vector
multiplication, the computation of the output strings can be done in time $\mathcal{O}(N_{\text{raw}} \log(N_{\text{raw}}))$. Considering
that the protocol requires $N_0$ commitments and all the remaining computations of random subsets and
checks can be done in linear time in $N_0$, the total protocol running time is $\mathcal{O}(N_0(k + k' + \log(N_0))$.

Regarding the **practicality of implementing $\pi_{QROT}$**, the protocol is designed to be compatible with $\text{BB84}$-
based $\text{QKD}$ setups, both from the physical layer up to the post-processing, only requiring the addition of the
commitment scheme. The most important difference to note is that $\pi_{QROT}$ has significantly lower tolerance
for **Qubit Error Rate ($\text{QBER}$)**. While most common $\text{QKD}$ protocols can produce keys through $\text{QBER}$s above
$10\%$, this protocol is limited to a maximum of $2.8\%$. This comparatively reduces the distances at which the
protocol can be successful. However, it is important to note that, as opposed to key distribution between
trusting parties, there are legitimate use-cases for $\text{OT}$ at short range. While being in proximity to each
other can help two trusting parties isolate themselves from a third party eavesdropper, mistrusting parties
do not gain anything (security wise) from being in the same place while attempting to do $\text{MPC}$, making the
protocol useful regardless of the distance between the users.

**Comparisons between classical and quantum protocols** can be difficult because physical/technological
assumptions, such as access to quantum communication or noisy quantum storage, do not straightforwardly
compare with computational hardness assumptions. Furthermore, there is no natural way of quantitatively
comparing statistical versus computational security. We can, however, contrast the (dis-)advantages of using
a computationally-secure quantum $\text{OT}$ protocol as compared to both fully classical computationally-secure
protocols, as well as statistically-secure quantum ones.

Classical $\text{OT}$ protocols based on asymmetric cryptography comprise the overwhelming majority of current
real-world implementations of $\text{OT}$. The obvious main advantage of quantum $\text{OT}$ is the weaker computational
hardness assumption ($\text{OWF}$ vs asymmetric cryptography), while the main advantage of current post-quantum
classical $\text{OT}$ implementations is speed. As shown in Fig. 5, the presented experimental setup is able to
produce up to $0.10 \text{ OT/s}$, which pales in comparison to contemporary classical protocols, such as [30-33],
that can achieve upwards of $10^5 \text{ OT/s}$ (not including latency between parties) with current off-the-shelf
hardware (for more details, see [33]). This difference can be mitigated by the use of $\text{OT}$ extension algorithms,
as the difference in speed would only matter during the generation of the base $\text{OT}$s. Note that in this case
one should use a $\text{OT}$ extension that matches the computational assumption of this work, such as [54].

Quantum protocols, both discrete variable ($\text{DV}$) [27] and continuous variable ($\text{CV}$) [29], have been shown
to achieve statistically-secure $\text{OT}$ in the **Quantum Noisy-Storage model ($\text{QNS}$)**. Their experimental imple-
mentations show comparable values of quantum communication cost in terms of shared signals: $10^8$ (no
memory encoding assumption), and $10^5$ ($\text{Gaussian}$ encoding) for $\text{CV}$, and $10^7$ for $\text{DV}$. As shown in Fig. 4,
our protocol requires $10^6$ quantum signals when matching their security ($\varepsilon = 10^{-7}$), which improves upon
the alternatives when no additional assumption on the memory encoding of the adversary is made. Less
straightforward to compare is the strength of the assumptions of noisy storage and $\text{OWF}$s. We note that the
existence of $\text{OWF}$s is an assumption that permeates modern cryptography, from block cipher encryption and
message authentication up to public-key cryptography protocols [55], which makes $\pi_{QROT}$ more suited to

**Table 2: Comparison of our work with other approaches for OT. $N$ denotes the respective security parameter. Acronyms for assumptions are as follows: OWF - One Way Functions; QNS - Quantum Noisy Storage; DDH - Decisional Diffie-Helmann; RLWE - Ring Learning With Errors - SLS - Space-Like Separation enforced. Protocols marked with * do not have a reference experimental implementation at the time of writing.**

| Protocol | Type | Assumption | Quantum Cost | Security |
|:---|:---|:---|:---|:---|
| This work | Quantum Discrete Variable | OWF | $\mathcal{O}(N)$ | Indistinguishability UC ROM |
| GLSV21 [38]* | Quantum Discrete Variable | OWF | $Poly(N)$ | Stand-alone plain model |
| S10 [27,47] | Quantum Discrete Variable | QNS | $\mathcal{O}(N)$ | Indistinguishability |
| FGSPSW18 [29] | Quantum Continuous Variable | QNS | $\mathcal{O}(N)$ | Indistinguishability |
| MR19 [30] | Classical | DDH | - | Stand-alone ROM |
| BFGMMS21 [33] | Classical | RLWE | - | UC ROM |
| P16 [56]* | Quantum/Relativistic Discrete Variable | SLS | $\mathcal{O}(N)$ | Other |

be introduced in current cipher suites than protocols with alternative assumptions. In particular, as noted
above, $\text{OWF}$s are required for $\text{OT}$ extension algorithms. A summary of comparisons between the different
approaches can be found in Fig.2.

Regarding potential improvements and further work, we can identify two main directions to build upon
this work: **performance** and **security**. Regarding performance, we note that dominant term in the expression
for $\varepsilon_{\text{max}}$ is the one associated with the significance of the parameter estimation (the first term in Eq. 13).
This translates into the relatively large values of $N_0$ needed to achieve adequate security, which was the
bottleneck in the performance of our implementation. One way to reduce the number of signals needed per
$\text{OT}$ is to modify the protocol to perform many concurrent $\text{ROT}$s in a single run. This would mean performing
one single estimation, albeit of a larger sample, that would work for many $\text{OT}$s in such a way that the required
number of signals per $\text{ROT}$ is decreased. On the topic of increasing security two main directions come to
mind. First, we can consider the constructions of **collapsing hash functions** proposed in [57,58] to implement
statistically hiding, computationally collapse binding commitments, which in turn allow for $\text{OT}$ protocols
that feature **forward security** (the $\text{OT}$ remains secure even if the underlying hash function can be attacked
after the commit/open phase of the protocol). The second direction would be a deeper exploration of the
**composable security of the protocol in the ROM**. This can come from generalizing Theorem 4.1 for any
weakly-interactive commitments (currently the proof applies only to the $\text{LRV25}$ construction), or applying
the techniques developed in [59] to prove $\text{UC}$ security of commitments in the quantum $\text{ROM}$ to remove
the adversary's limitation of classical access to the oracle. From the practical implementation perspective,
it seems natural to **integrate quantum $\text{OT}$ into both $\text{QKD}$ setups for a unified physical layer** capable of
providing secure communication and computation powered by $\text{OT}$ extension and $\text{MPC}$ algorithms, bringing
the benefits of quantum $\text{OT}$ closer to real world usage.

# 7 Acknowledgements

M.L., P.M., and N.P. acknowledge Fundação para a Ciência e Tecnologia ($\text{FCT}$), Instituto de Telecomu-
nicações Research Unit, ref. $\text{UID}/50008/2020$, and $\text{PEst-OE}/\text{EEI}/\text{LA0008}/2013$, as well as $\text{FCT}$ projects
$\text{QuantumPrime}$ reference $\text{PTDC}/\text{EEI-TEL}/8017/2020$. M.L. acknowledges $\text{PhD}$ scholarship $\text{PD}/\text{BD}/114334/2016$.
N.P. acknowledges the $\text{FCT Estímulo}$ ao $\text{Emprego Científico}$ grant no. $\text{CEECIND}/04594/2017/\text{CP}1393/\text{CT}000$.
P.S., M.B. and P.W. acknowledge funding from the European Union's $\text{Horizon Europe}$ research and inno-
vation program under $\text{Grant Agreement No. } 101114043 (\text{QSNP})$, along with the $\text{Austrian Science Fund}$
$\text{FWF } 42$ through $[\text{F}7113] (\text{BeyondC})$ and the $\text{AFOSR}$ via $\text{FA}9550-21-1-0355 (\text{QTRUST})$. $\text{D.E.}$ was sup-
ported by the $\text{JST Moonshot R}\& \text{D}$ program under $\text{Grant JPMJMS226C}$. M.G. acknowledges $\text{FCT Portu-
gal}$ financing refs. $\text{UIDB}/50021/2020$ and $\text{UIDP}/50021/2020$ (resp. $\text{DOI } 10.54499/\text{UIDB}/50021/2020$ and
$10.54499/\text{UIDP}/50021/2020$).

# Appendix A Preliminaries

## A.1 Quantum computational efficiency and distinguishability

We model the quantum capabilities of parties through programs running on quantum computers, for which
we adopt a model based on **deterministic-control quantum Turing Machines** [60]. For the purposes of the
following definitions, a quantum computer is a device that has a classical interface and a quantum part, which
contains the quantum memory registers available to the party. The classical interface has the capabilities of
a classical computer augmented with the ability to perform a predefined universal set of quantum operators
on the quantum memory registers and perform measurements in the canonical (computational) basis. Given
a specified type of quantum computer, a quantum program is a classical description of a set of instructions
to be run by the computer, including the quantum operations and measurements to be executed in the
quantum part, as well as any classical computation. Quantum programs can be compared with probabilistic
classical programs as they both have natural numbers as inputs/outputs. When a quantum computer runs
the program $T$ with input $x \in \mathbb{N}$, we assume that the quantum part of the computer starts with some
predefined initial state, performs a sequence of operations on its quantum registers, and upon halting, it
outputs $T(x) \in \mathbb{N}$ on its classical interface by reading the appropriate registers associated with the program's
output. Each execution of a quantum program is then associated to a quantum operation, which is the result
of all the operations performed on the quantum part during the execution of the program.

**Definition A.1.** (*Computational efficiency*)
*Let $T$ be a quantum program. We say that $T$ is computationally efficient (or polynomial-time) if there exists
a polynomial $P$ such that the running time of $T(x)$ is $\mathcal{O}(P(|x|))$.*

**Definition A.2.** (*Distinguishing Advantage*)
*Let $X_1, X_2$ be two random variables with values in $\mathbb{N}$. For any quantum program $T$, the distinguishing
advantage of $X_1, X_2$ using $T$ is defined as*

$$
\text{Adv}_T (X_1, X_2) = | \text{Pr}[T(X_1) = 1] - \text{Pr}[T(X_2) = 1] |, \quad (39)
$$

*Analogously, let $\rho_1, \rho_2 \in \mathcal{D}(H)$. For any quantum program $T$, the distinguishing advantage of $\rho_1, \rho_2$ using $T$
is defined as*

$$
\text{Adv}_T (\rho_1, \rho_2) = | \text{Pr}[T(\rho_1) = 1] - \text{Pr}[T(\rho_2) = 1] |, \quad (40)
$$

*where $T(\rho)$ denotes the classical output of the program starting with the quantum state $\rho$ and zero classical
input.*

**Definition A.3.** (Indistinguishability - Finite)
*Let $\rho_1, \rho_2 \in \mathcal{D}(H)$ and $\varepsilon \geq 0$. We say that $\rho_1$ and $\rho_2$ are $\varepsilon$-indistinguishable, denoted by $\rho_1 \approx_{\varepsilon} \rho_2$, whenever*

$$
\text{Adv}_T (\rho_1, \rho_2) \leq \varepsilon, \quad \text{for all quantum programs } T. \quad (41)
$$

$\varepsilon$-indistinguishability for random variables is defined analogously.

As the following proposition states, to show that two states are $\varepsilon$-indistinguishable, it is enough to upper
bound their trace distance $D$. (for more detail on the relationship of these quantities, see [61,62]).

**Proposition A.1.** *For any pair of quantum states $\rho_1, \rho_2 \in \mathcal{D}(H)$ it holds that*

$$
\rho_1 \approx_{D(\rho_1, \rho_2)} \rho_2. \quad (42)
$$

**Definition A.4.** (Indistinguishability – Asymptotical)
*Let $\{\rho^{(k)}_1 \in \mathcal{D}(H_k)\}$ and $\{\rho^{(k)}_2 \in \mathcal{D}(H_k)\}$ be two families of density operators. We say that the two families
are statistically indistinguishable if there exists a negligible function $\varepsilon(k) \geq 0$ such that*

$$
\rho^{(k)}_1 \approx_{\varepsilon(k)} \rho^{(k)}_2 \quad \text{for all } k \in \mathbb{N}. \quad (43)
$$

*Furthermore, we say the two families are computationally indistinguishable if for every efficient quantum
program $T$, there exists a negligible function $\varepsilon_T(k) \geq 0$ such that*

$$
\text{Adv}_T (\rho^{(k)}_1, \rho^{(k)}_2) \leq \varepsilon_T(k) \quad \text{for all } k \in \mathbb{N}. \quad (44)
$$

Statistical and computational indistinguishability for random variables is defined analogously.

Recall from Section 2 that, when the parameter $k$ is implicit, we may omit the explicit dependence on
$k$ and use $\approx$ and $\approx_{(c)}$ for statistical and computational indistinguishability, respectively. We now turn our
attention to the properties of indistinguishable states. It is worth noting that computational indistinguisha-
bility is only meaningful in terms of information security when the adversary is assumed to have limited
computational capabilities. It is important then to define the type of quantum operations such adversary
can perform:

**Definition A.5.** (*Efficient quantum operation*)
*We say that a family $\{\mathcal{E}^{(k)}\}_{k=1}^\infty$ of quantum operations is efficient if there exists an efficient quantum program
$T$ such that, for each $k$, $\mathcal{E}^{(k)}$ is the associated operation applied to the quantum part of the machine while
running $T$ on input $k$*

The following properties are straightforward to prove from Definitions A.3 and A.4 and the properties of
trace distance:

**Lemma A.1.** (Properties of indistinguishable states I)
*Let $\rho_1, \rho_2, \rho_3 \in \mathcal{D}(H)$:*
1.  *$\rho_1 \approx_{\varepsilon} \rho_2 \land \rho_2 \approx_{\varepsilon'} \rho_3 \Rightarrow \rho_1 \approx_{\varepsilon + \varepsilon'} \rho_3$.*
2.  *$\rho_1 \approx_{\varepsilon} \rho_2 \land \sigma_1 \approx_{\varepsilon'} \sigma_2 \Rightarrow \rho_1 \otimes \sigma_1 \approx_{\varepsilon + \varepsilon'} \rho_2 \otimes \sigma_2$.*
3.  *Let $x \in \mathcal{X}$. For any probability distribution $P_X$, assume that $(\forall x \in \mathcal{X}) \rho^x_1 \approx_{\varepsilon^x} \rho^x_2$. Then*
    $$
    \sum_{x \in \mathcal{X}} P_X \rho^x_1 \approx_{\varepsilon_{\max}} \sum_{x \in \mathcal{X}} P_X \rho^x_2 \quad \text{where } \varepsilon_{\max} = \max_{x \in \mathcal{X}} \{\varepsilon^x\}.
    $$
4.  *$\rho_1 \approx_{\varepsilon} \rho_2 \Rightarrow \mathcal{E}(\rho_1) \approx_{\varepsilon} \mathcal{E}(\rho_2)$, for any completely positive, trace non-increasing map $\mathcal{E}$.*

**Lemma A.2.** (Properties of indistinguishable states II)
*Let $\{\rho_1^{(k)}\}, \{\rho_2^{(k)}\}, \{\rho_3^{(k)}\}$ be families of density operators parameterized by $k = 1, 2, \ldots$. The following
statements hold for asymptotic computational indistinguishability:*
1.  *$\rho_1 \approx_{(c)} \rho_2 \land \rho_2 \approx_{(c)} \rho_3 \Rightarrow \rho_1 \approx_{(c)} \rho_3$.*
2.  *$\rho_1 \approx_{(c)} \rho_2 \land \sigma_1 \approx_{(c)} \sigma_2 \Rightarrow \rho_1 \otimes \sigma_1 \approx_{(c)} \rho_2 \otimes \sigma_2$.*
3.  *Let $x \in \mathcal{X}$. For any probability distribution $P_X$, assume that $(\forall x \in \mathcal{X}) \rho^x_1 \approx_{(c)} \rho^x_2$. Then*
    $$
    \sum_{x \in \mathcal{X}} P_X \rho^x_1 \approx_{(c)} \sum_{x \in \mathcal{X}} P_X \rho^x_2.
    $$
4.  *$\rho_1 \approx_{(c)} \rho_2 \Rightarrow \mathcal{E}(\rho_1) \approx_{(c)} \mathcal{E}(\rho_2)$, where $\{\mathcal{E}^{(k)}\}$ is an efficient family of quantum operations acting on the
    respective $\rho_i^{(k)}$.*

## A.2 Entropic quantities

We start off by defining a useful pair of quantities for measuring information in quantum systems: the **max-
entropy** and the **conditional min-entropy**. The max entropy is a measure of the number of possible different
outcomes that can result from measuring a quantum state, whereas the conditional min-entropy is a way of
measuring the information that a party can infer from a quantum system given access to another correlated
quantum system. This measures will be useful to bound the distance between states based on their internal
correlations.

**Definition A.6.** (Max-entropy)
*Let $\rho \in \mathcal{D}(H)$. The **max-entropy** of $\rho$ is defined as*

$$
H_{\max}(\rho) = \log (\text{dim}(\text{supp}(\rho))), \quad (45)
$$

*where $\text{supp}(\rho)$ denotes the support subspace of $\rho$ and $\text{dim}$ denotes its dimension.*

**Definition A.7.** (Min-entropy and conditional min-entropy)
*Let $\rho \in \mathcal{D}(H)$ and $\lambda_{\max}(\rho)$ denote the maximum eigenvalue of $\rho$. The **min-entropy** of $\rho$ is defined as*

$$
H_{\min}(\rho) = -\log(\lambda_{\max}(\rho)). \quad (46)
$$

*Let $\rho_{AB} \in \mathcal{D}(H_A \otimes H_B)$ and $\sigma_B \in \mathcal{D}(H_B)$. The **conditional min-entropy** of $\rho_{AB}$ given $\sigma_B$ is defined as*

$$
H_{\min}(\rho_{AB} | \sigma_B) = -\log(\lambda_{\sigma_B}), \quad (47)
$$

*where $\lambda_{\sigma_B}$ is the minimum real number such that $\lambda_{\sigma_B} (\mathbb{I}_A \otimes \sigma_B) - \rho_{AB}$ is non-negative. The **conditional
min-entropy** of $\rho_{AB}$ given $H_B$ is defined as*

$$
H_{\min}(A|B)_\rho = \sup_{\sigma_B \in \mathcal{D}(H_B)} H_{\min}(\rho_{AB} | \sigma_B), \quad (48)
$$

*Furthermore, let $\varepsilon \geq 0$. The $\varepsilon$-**smooth conditional min-entropy** is defined as*

$$
H_{\min}^\varepsilon(A|B)_{\rho} = \sup_{\rho'_{AB} \in B^{\varepsilon}(\rho_{AB})} H_{\min}(A|B)_{\rho'}, \quad (49)
$$

*where $B^{\varepsilon}(\rho_{AB}) = \{\rho'_{AB} : D(\rho_{AB}, \rho'_{AB}) < \varepsilon\}$.*

The smooth conditional min-entropy is in general hard to compute. Because of this, it is useful to have
some tools to bound it for states that have some specific forms. In our case we are interested in states that
are partially classical.

**Definition A.8.** (Partially classical states)
*A quantum state described by the density operator $\rho_{AB} \in \mathcal{D}(H_A \otimes H_B)$ is **classical in $H_A$** (or **classical in
$A$**) if it can be written in the form*

$$
\rho_{AB} = \sum_x \lambda_x |x\rangle \langle x|_A \otimes \rho^x_B, \quad (50)
$$

*where the set $\{|x\rangle\}_x$ is an orthonormal basis for $H_A$. A **multipartite state** is said to be classical if it is
classical in all its parts.*

When dealing with partially classical states as shown in Eq. (50), we will refer to the operators $\rho^x_B$ as the
state of the system $B$ conditioned to $x$.

**Lemma A.3.** (Properties of min- and max-entropy)
*Let $\varepsilon, \varepsilon' \geq 0$:*
1.  *$H_{\min}(\rho_{AB} | \rho_B) = -\log(\lambda_{\max}(\rho_A)).$*
2.  *$H_{\min}^{\varepsilon + \varepsilon'}(AA' | BB')_{\rho \otimes \rho'} \geq H_{\min}^\varepsilon(A|B)_\rho + H_{\min}^{\varepsilon'}(A'|B')_{\rho'}$.*
3.  *$H_{\min}(A|BC)_{\rho} \leq H_{\min}(A|B)_{\rho}$.*
4.  *$H_{\min}^{\varepsilon}(A|BC)_{\rho} \leq H_{\min}^\varepsilon(A|BC)_{\rho} + H_{\max}(\rho_B)$.*
5.  *$H_{\min}(\rho_{AB}) \geq \inf_x \{H_{\min}(\rho^x_A)\}$, whenever the state $\rho_{AB}$ is classical on $B$.*

We use universal hashing to implement randomness extraction in the final steps of the protocol. The
proof both Lemmas A.3 and A.4 can be found in [63].

**Definition A.9.** (Universal hashing)
*A set of functions $\mathcal{F} = \{f_i : \{0, 1\}^m \rightarrow \{0, 1\}^n\}$ is a **universal hash family** if, for all $x, y \in \{0, 1\}^m$, such
that $x \neq y$, and $i$ chosen uniformly at random, we have*

$$
\text{Pr}[f_i(x) = f_i(y)] \leq \frac{1}{2^n}. \quad (51)
$$

**Lemma A.4.** (Quantum leftover hash)
*Let $\mathcal{F} = \{f_i : \{0, 1\}^m \rightarrow \{0, 1\}^n\}$ be a universal hash family, let $H_A, H_B, H_{\mathcal{F}}, H_E$ be Hilbert spaces such that
$\{|x\rangle\}_{x \in \{0, 1\}^m}, \{|f_i\rangle\}_{f_i \in \mathcal{F}},$ and $\{|e\rangle\}_{e \in \{0, 1\}^n}$ are orthonormal bases for $H_A, H_{\mathcal{F}},$ and $H_E$ respectively. Then
for any $\varepsilon \geq 0$ and any state of the form*

$$
\rho_{ABF E} = \frac{1}{|\mathcal{F}|} \sum_{f_i \in \mathcal{F}} \sum_{x \in \{0, 1\}^m} |f_i(x)\rangle \langle f_i(x)|_F \otimes |f_i(x)\rangle \langle f_i(x)|_E \\
\otimes |x\rangle \langle x|_A \otimes \rho^x_B, \quad (52)
$$

*it holds that*

$$
\rho_{EBF} \approx_{\varepsilon'} U_E \otimes \rho_{BF}, \quad (53)
$$

*with*

$$
\varepsilon' = \varepsilon + \frac{1}{2} 2^{-\frac{1}{2}(H_{\min}(A|B)_{\rho} - n)}. \quad (54)
$$