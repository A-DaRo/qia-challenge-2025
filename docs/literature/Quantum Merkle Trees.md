arXiv:2112.14317v4 [quant-ph] 14 Jun 2024

# Quantum Merkle Trees

Lijie Chen¹ and Ramis Movassagh²

¹Miller Institute for Basic Research in Science, University of California, Berkeley, CA, 94720, USA
²Google Quantum AI, Venice, CA, 90291, USA
IBM Quantum, MIT-IBM Watson AI Lab, Cambridge, MA, 02142, USA

Committing to information is a central task in cryptography, where a party (typically called a prover) stores a piece of information (e.g., a bit string) with the promise of not changing it. This information can be accessed by another party (typically called the verifier), who can later learn the information and verify that it was not meddled with. Merkle trees [1] are a well-known construction for doing so in a succinct manner, in which the verifier can learn any part of the information by receiving a short proof from the honest prover. Despite its significance in classical cryptography, there was no quantum analog of the Merkle tree. A direct generalization using the Quantum Random Oracle Model (QROM) [2] does not seem to be secure. In this work, we propose the **quantum Merkle tree**. It is based on what we call the **Quantum Haar Random Oracle Model (QHROM)**. In QHROM, both the prover and the verifier have access to a Haar random quantum oracle $G$ and its inverse.

Using the quantum Merkle tree, we propose a **succinct quantum argument** for the Gap-$k$-Local-Hamiltonian problem. Assuming the Quantum PCP conjecture is true, this succinct argument extends to all of QMA. This work raises a number of interesting open research problems.

# 1 Introduction

A commitment scheme [3] is a cryptographic primitive that allows a party (i.e., a prover) to (1) commit to a piece of information such as a bit string while keeping it hidden from others and (2) reveal the information they have committed to later. Commitment schemes are designed to ensure that a party cannot change the information after they have committed to it. Commitment schemes have numerous applications in cryptography, such as the construction of protocols for secure coin flipping, zero-knowledge proofs, and secure computation.

The Merkle tree [1] is an efficient example of commitment schemes, which captures the following scenario: There are two parties, the prover $P$ and the verifier $V$. $P$ first computes a short string called the commitment which is denoted by $\text{commit}(x)$ from a long input string $x$ and sends $\text{commit}(x)$ to $V$. Then $V$ asks $P$ to reveal a subset of bits of $x$ together with a short message that would enable $V$ to verify that the string $x$ has not been altered. The security promise is that after $P$ sends $\text{commit}(x)$ to $V$, then upon $V$'s request of any subset of bits, a computational bounded $P$ can only reveal those bits faithfully. Namely, if $P$ claims that the $i$-th bit of $x$ is the wrong value $1 - x_i$, then her claim will be rejected by $V$ with high probability.

Lijie Chen: `lijiechen@berkeley.edu`
Ramis Movassagh: `q.eigenman@gmail.com`

Accepted in `<xuantum 2023-11-19, click title to verify. Published under CC-BY 4.0.`

1

The Merkle tree has wide applications in cryptography since it allows $P$ to delegate a potentially very long string to $V$ (i.e., a database) while enabling $V$ to maintain an efficient *verifiable random access* to that string (say to any subset of the bits of the string). A well-known application of the Merkle tree is the succinct arguments for NP from probabilistically checkable proofs [4, 5] or interactive oracle proofs [6], where by succinctness one means that the total communication between the prover and verifier constitutes a small number of bits, say $\text{polylog}(n)$ bits of communication.

Despite being very influential in (classical) cryptography, there is no known quantum analog of the Merkle tree that allows committing to quantum states. Such a quantum analog is appealing since it would allow a party to commit to a large quantum state $\sigma$ while maintaining verifiable access to individual qubits.

Protocols based on the classical Merkle tree are often analyzed in the random oracle model. There are also quantum models such as the Quantum Random Oracle Model [2] (QROM) for analyzing the quantum attacks against the classical Merkle tree. There are works showing that classical Merkle-tree-based protocols are secure against quantum attacks [7, 8]. These works showed that commitment to classical bit strings by the Merkle tree cannot be broken by quantum adversaries. Here we hope to obtain a quantum analog of the Merkle tree that can be used to commit to quantum states.

In this work, we propose a new random oracle model which we call the **Quantum Haar Random Oracle Model (QHROM)**. We then use it in our construction of the **Quantum Merkle tree**. We then use it to propose a quantum analog of Kilian's succinct argument for NP and conjecture its security.

## 1.1 The Merkle Tree Algorithm

Our definition of QHROM is motivated by our adaptation of the Merkle tree to the quantum setting, so it is instructive to recall the standard Merkle tree algorithm.

Let $b \in \mathbb{N}$ be the block-length parameter. We assume that both $P$ and $V$ have access to a random oracle function $h: \{0, 1\}^{2b} \to \{0, 1\}^b$. For simplicity of the argument, we will first focus on the simplest non-trivial case of a Merkle tree with two leaves and depth one, and take $n = 2b$ to be the length of the string that $P$ wishes to commit to. Here the string $x$ resides on the leaves and $\text{commit}(x)$ string resides on the root. As we will see shortly, a straightforward adaption of the Merkle tree to the quantum setting is not secure even in this simple setting.

> $P$ sends to $V \xrightarrow{\text{commit}(x)} V$

$$
h(x_{1}, \ldots, x_{2b})
$$
$$
\begin{array}{|c|} \hline x_{1}, \ldots, x_{b} \\ \hline \end{array} \quad \begin{array}{|c|} \hline x_{b+1}, \ldots, x_{2b} \\ \hline \end{array}
$$

**Figure 1:** An illustration of the toy example for the classical Merkle tree

In this simplified setting, the protocol starts by $P$ simply sending the hash value $h = h(x)$ of $x \in \{0, 1\}^{2b}$ as the $\text{commit}(x)$ of length $b$ to $V$ (see Figure 1 for an illustration). Then $V$ requests the values of a subset of bits in $x$, for which the honest $P$ simply responds by revealing the whole string $x$ to $V$. Then $V$ checks that the string has the same hash value $h$. If a (dishonest) $P$ can first commit to $x$ and later convince $V$ that its $i$-th bit is $1 - x_i$, then $P$ has found two strings $x \neq \tilde{x}$ with $h(x) = h(\tilde{x})$. This requires at least $2^{b/2}$ queries to the random oracle $h$ due

2

to the birthday paradox, which is infeasible.

## 1.2 A Failed Attempt to Adapt Merkle Tree in the Quantum Setting

Let us see how one might directly try to adapt the special case above of the Merkle tree algorithm to the quantum setting. An immediate idea is that, given a $2b$-qubit quantum state $|\psi\rangle = \sum_z \alpha_z |z\rangle$ in the register denoted by **data**, $P$ treats $h$ as a quantum oracle $O_h$\footnotemark[1], creates $b$ qubits initialized to $|0^b\rangle$ in register **com**, applies $O_h$ to both **data** and **com** to obtain $\sum_z \alpha_z |z\rangle |h(z)\rangle$, and sends the **com** register to $V$; see Figure 2 for an illustration.

\footnotetext[1]{That is, $O_h |x\rangle |y\rangle = |x\rangle |y \oplus h(x)\rangle$, where $x \in \{0, 1\}^{2b}$, $y \in \{0, 1\}^b$, and $\oplus$ denotes the entry-wise addition over $\text{GF}(2)$.}

> $P$ sends to $V \xrightarrow{\text{com: b qubits}} V$

$$
\begin{array}{|c|} \hline \multicolumn{1}{|c|}{\text{com: b qubits}} \\ \hline |0^b\rangle \\ \hline \multicolumn{1}{|c|}{|\psi\rangle} \\ \hline \multicolumn{1}{|c|}{\text{data: 2b qubits}} \\ \hline \end{array} \quad \text{apply } G \text{ or } O_h
$$

**Figure 2:** An illustration of the toy example for the quantum Merkle tree

To reveal qubits in $|\psi\rangle$, $P$ simply sends the **data** register to $V$ as well, and $V$ applies $O_h$ again to both **data** and **com**, and measures **com** in the computational basis to check if it is $|0^b\rangle$ and rejects immediately otherwise. However, this is not secure against **phase attack**. After sending **com** to $V$, for every Boolean function $f: \{0, 1\}^{2b} \to \{0, 1\}$, $P$ can apply the unitary $|z\rangle \to (-1)^{f(z)} |z\rangle$ to **data**, and then sends it to $V$. One can see that $V$ still accepts this state with probability 1, but $P$ has cheated by changing the state from $\sum_z \alpha_z |z\rangle$ to $\sum_z (-1)^{f(z)} \alpha_z |z\rangle$, which can be entirely a different state for some function $f$.

The issue above is that the mapping $O_h, |x\rangle |y\rangle \to |x\rangle |y \oplus h(x)\rangle$, has too much structure to be exploited by the attacker. This immediately suggests to us to consider a more random choice of quantum oracles which indeed we take to be the most random choice of quantum oracles: a Haar random quantum oracle.

Comment: One way to address the phase attack above is to make $O_h$ more complicated. For example, instead of applying $O_h$ once to the registers **data** and **com**, we can repeatedly apply $O_h H^{\otimes 2b}$ several times ($H$ denotes the Hadamard gate). We found such a construction more cumbersome and even harder to analyze compared to a Haar random unitary. Moreover, it is conjectured [9, Section 6] that similar constructions may already be indistinguishable from a Haar random unitary (see Section 3.3 for more discussions). Hence, it seems more natural to directly work with a Haar random unitary.

## 1.3 The Quantum Haar Random Oracle Model (QHROM) and Quantum Merkle Tree

We are now ready to introduce the **Quantum Haar Random Oracle Model (QHROM)**. In QHROM both $P$ and $V$ have access to a Haar random quantum oracle $G$ and its inverse $G^\dagger$ that act on $3b$ qubits (see Definition 2.1 for a precise definition). The protocol between $P$ and $V$ remains the same for the special case $n = 2b$ except for replacing $O_h$ by $G$. It is easy to see

3

$$
\begin{array}{cccccccc}
& & & 1 & & & \\
& 2 & & & 3 & & \\
4 & & 5 & & 6 & & 7 \\
\dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots \\
\multicolumn{1}{c}{x^{(1)}} & \multicolumn{1}{c}{x^{(2)}} & \dots & \dots & \dots & \multicolumn{1}{c}{x^{(l-1)}} & \multicolumn{1}{c}{x^{(l)}} \\
l+0 & l+1 & & & & 2l-2 & 2l-1
\end{array}
$$

**Figure 3:** An illustration of the quantum Merkle tree with $l = 2^d$ input blocks; when block $x^{(2)}$ is requested by $V$, $P$ sends all the diamond shape nodes.

that since $G$ completely **obfuscates** the original state $|\psi\rangle$, the phase attack described above no longer applies.

Next, we describe the quantum Merkle tree in the general setting in which $n$ can be arbitrarily large and denotes the number of qubits in the state that $P$ wishes to commit to. Given a quantum state $\sigma$ on $n = b \cdot l$ qubits for some $l = 2^d$ and $d \in \mathbb{N}$,² we partition $x$ into $l$ consecutive blocks of length $b$ as $x^{(1)}, x^{(2)}, \ldots, x^{(l)}$. Then, we build a perfect binary tree with $l$ leaves (see Figure 4), where each leaf corresponds to a block of the input. Next, from the leaves to the root, we assign to each node $u$ a $b$-qubit register $\text{com}_u$ as follows: (1) if $u$ is a leaf, then $\text{com}_u$ is simply the qubits of the assigned block and (2) if $u$ is an intermediate node with two children $\beta$ and $\gamma$, then we initialize $\text{com}_u$ to $|0^b\rangle$, and apply $G$ to the three registers $\text{com}_\beta$, $\text{com}_\gamma$, and $\text{com}_u$. Finally, $P$ sends the register $\text{com}_{\text{rt}}$ to $V$, where $\text{rt}$ is the root of the binary tree.

Suppose $V$ requests the state of the $i$-th block $x^{(i)}$ of the quantum state. To reveal the $i$-th block $x^{(i)}$ on a leaf (which we denote $\mu$) of the tree $P$ sends all the $\text{com}_\alpha$ for nodes $\alpha$ that are the (1) ancestors of $\mu$, (2) siblings of an ancestor of $\mu$, or (3) $\mu$ or the sibling of $\mu$. $V$ then "undoes" all the applied $G$ in the exact reverse order by applying $G^\dagger$ to the registers sent by $P$ starting from the register $\text{com}_{\text{rt}}$, and then from the root downwards to the leaves. After that, for every ancestor $\alpha$ of $\mu$, $V$ checks that $\text{com}_\alpha$ is $|0^b\rangle$ by measuring it in the computational basis. To illustrate, if $V$ asks for the block $x^{(2)}$, then $P$ sends the corresponding $\text{com}_\alpha$ registers for all diamond shape nodes in Figure 3.

Comment: So how might one heuristically instantiate a Haar-Random unitary? One might use a random quantum circuit that well approximates the behavior of a Haar unitary. For example, one might use a polynomially deep circuit. One way to formalize the degree of approximation is via the ideas in $k$-design [10].

## 1.4 A Candidate for Succinct Quantum Argument for Gap-$k$-LH in QHROM

Similar to Kilian's succinct argument for NP, the quantum Merkle tree naturally suggests a succinct argument $\Pi_{\text{succinct}}$ for the Gap Local Hamiltonian Problem. We first recall its definition below.

**Definition 1.1.** (Gap-$k$-Local Hamiltonian Problem) Given $\alpha, \beta$ with $0 < \alpha < \beta < 1$ and a $k$-local Hamiltonian with $m$ local terms $\{H_i\}_{i \in [m]}$ such that $0 \le H_i \le I$, decide whether

²We can always pad any quantum state to such length by adding dummy qubits. This at most doubles the number of qubits.

4

$\lambda_{\min} (\sum_{i=1}^m H_i)$ is at most $\alpha m$ or at least $\beta m$. Below we abbreviate this problem by $(\alpha, \beta)$-$k$-LH.

Formally, in $\Pi_{\text{succinct}}$ the honest prover $P$ applies the quantum Merkle tree to a ground state $\sigma$ of $\sum_{i=1}^m H_i$, and sends $\text{com}_{\text{rt}}$ to $V$. Then $V$ draws an integer $i$ from $\{1, 2, \ldots, m\}$ uniformly at random and asks $P$ to reveal the qubits in the support of the term $H_i$. $V$ does the decommitment from the root towards the qubits in the support of $H_i$ as described above. If in this decommitment phase the ancestors of the qubits in the support of $H_i$ all result in $|0^b\rangle$ it proceeds to the last step. In the last step, it measures the POVM $\{H_i, I - H_i\}$ on the qubits in the support of $H_i$ and rejects if it sees $H_i$. Indeed, this is the natural analog of Kilian's succinct argument [4] in the quantum setting.

We prove that if $P$ follows the protocol, then (1) when $\lambda_{\min}(\sum_{i=1}^m H_i) \le \alpha \cdot m$, $P$ can make $V$ accept with probability at least $1 - \alpha$, and (2) when $\lambda_{\min}(\sum_{i=1}^m H_i) \ge \beta \cdot m$, $P$ cannot force $V$ to accept with a probability greater than $1 - \beta < 1 - \alpha$ (See Theorem 3.1 for details). By a sequential repetition argument, the completeness $1 - \alpha$ and the soundness $1 - \beta$ can be boosted to $1 - n^{-\omega(1)}$ and $n^{-\omega(1)}$ respectively where $\omega(1)$ means super constant.

However, a malicious $P$ may not follow the protocol, but instead come up with some arbitrary states for the different nodes that are not the result of the quantum Merkle tree algorithm and send those to $V$ instead. We currently do not know how to analyze such an arbitrary attack, but we conjecture the following:

**Conjecture 1.2.** For the constants $k \in \mathbb{N}$ and $0 < \alpha < \beta < 1$, $\Pi_{\text{succinct}}$ (with sequential repetition) for $(\alpha, \beta)$-$k$-LH has completeness $1 - n^{-\omega(1)}$ and soundness $n^{-\omega(1)}$ in QHROM.

## 1.5 Open Questions and Follow-up Works

We believe our inability to prove Conjecture 1.2 is mainly due to the lack of tools available for analyzing this new QHROM setting. We remark that only two years ago [7] managed to prove that the succinct argument for NP [4, 5] is secure in the QROM model by using the recently proposed **compressed oracles technique** introduced in [11] which gives a nice way to analyze QROM. To prove the security of our succinct argument for Gap-$k$-LH, one likely needs similar advances for analyzing the QHROM. We now list some specific open problems:

**Open Problem 1.** Is there an analog of the compressed oracle technique in [11] for the QHROM?

Above we generalized Kilian's constant-round succinct argument [4] to the quantum setting and conjectured its soundness. A natural open question is whether we can generalize Micali's non-interactive succinct argument for NP [5] to the quantum settings as well.

**Open Problem 2.** Is there an analog of Micali's non-interactive succinct argument for Gap-$k$-LH?

A particularly useful feature of previous succinct arguments for NP [4, 5] is that they can be made **zero-knowledge** with minimal overhead. A natural open question is whether we can make our proposed succinct argument for Gap-$k$-LH zero-knowledge as well.

**Open Problem 3.** Is there a zero-knowledge succinct argument for Gap-$k$-LH in QHROM?

**Subsequent work.** Finally, we remark that this paper formed the basis of the ideas in an exciting subsequent work [12]. They proved the security of a tree commitment similar to what is presented here but from standard (quantum) cryptographic assumptions. Note that it is not a priori clear what "security" even means for commitments to quantum states³, and a major

³Although such commitment scheme has appeared implicitly in several quantum cryptographic protocols [13, 14, 15].

5

contribution of [12] is to formally define the security of commitments to quantum states. We refer readers to [12] for an overview of more prior works on quantum commitment schemes. As far as we know, the security of the precise protocol (in the QHROM) given in this paper remains open.

# 2 Preliminaries

## 2.1 Notation

We always denote by $\log$ the logarithm in base 2. We denote by $[n]$ the set of integers $\{1, 2, \ldots, n\}$. Let $\text{reg}$ be a register of $n$ qubits. For each $i \in [n]$, $\text{reg}(i)$ denotes the $i$-th qubit in $\text{reg}$, and $\text{reg}[l, r]$ denotes the qubits from $\text{reg}(l)$ to $\text{reg}(r)$. The corresponding Hilbert space is denoted by $\mathcal{H}_{\text{reg}}$. For $k$ pairwise-disjoint sets $S_1, \ldots, S_k$, we use $\bigsqcup_{i \in [k]} S_i$ to denote their union. We say a function $a: \mathbb{N} \to [0, 1]$ satisfies $a(n) \le \text{negl}(n)$ (i.e., $a$ is negligible), if for all constant $k \ge 1$, $\lim_{n \to \infty} a(n) \cdot n^k = 0$ (i.e., $a(n) = o(n^{-k})$ for every $k \in \mathbb{N}$).

For a quantum state $\sigma$ on $n$ qubits and a subset $S \subseteq [n]$, $\sigma_S := \text{Tr}_{[n] \setminus S}[\sigma]$ is the reduced density matrix. For a quantum state $|\psi\rangle \in \mathcal{H}_{\text{reg}}$, for simplicity we sometimes use $\psi$ to denote the corresponding density matrix $\psi = |\psi\rangle \langle \psi|$. Given a unitary sequence $U_1, \ldots, U_r$, we write $U_{[l, r]}$ to denote the product $U_r U_{r-1} \cdots U_l$ for ease of notation.

For two quantum states $\sigma$ and $\rho$, we use $||\sigma - \rho||_1$ to denote their trace distance. We also write $x \in_{\text{R}} A$ to mean that $x$ is drawn from the set $A$ uniformly at random.

## 2.2 The Quantum Haar Random Oracle Model

We will consider the **Quantum Haar random oracle model (QHROM)**, in which every agent (prover and verifier) gets access to a Haar random oracle $G$ acting on $\lambda$ qubits and its inverse $G^\dagger$, where $\lambda$ is the so-called **security parameter**.

We denote by $U(N)$ the set of all $N \times N$ unitaries. By $G \in_{\text{R}} U(N)$ we mean that $G$ is an $N \times N$ unitary drawn from the Haar measure.

**Definition 2.1.** An interactive protocol $\Pi$ between the prover $P$ and verifier $V$ is a proof system for a promise problem $L = (L_{\text{yes}}, L_{\text{no}})$ with completeness $c(n, \lambda)$ and soundness $s(n, t, \lambda)$ in QHROM, if the following holds:

$P$ and $V$: $P$ and $V$ are both given an input $x \in L_{\text{yes}} \cup L_{\text{no}}$. $V$ is polynomial-time and outputs a classical bit indicating acceptance or rejection of $x$, and $P$ is unbounded. Both $V$ and $P$ are given access to a Haar random quantum oracle $G$ and its inverse $G^\dagger$ that act on $\lambda$ qubits (that is, $G \in_{\text{R}} U(2^\lambda)$). Let $n = |x|$.

**Completeness:** If $x \in L_{\text{yes}}$,
$$
\underset{G \in_{\text{R}} U(2^\lambda)}{\mathbb{E}} \left[ \text{Pr}[(\mathcal{V}^{G, G^\dagger} \leftrightarrow P^{G, G^\dagger})(x) = 1] \right] \ge c(n, \lambda) .
$$

**Soundness:** If $x \in L_{\text{no}}$, for every $t \in \mathbb{N}$ and any unbounded prover $P^*$ making at most $t$ total queries to $G$ and $G^\dagger$, we have that
$$
\underset{G \in_{\text{R}} U(2^\lambda)}{\mathbb{E}} \left[ \text{Pr}[(\mathcal{V}^{G, G^\dagger} \leftrightarrow (P^*)^{G, G^\dagger})(x) = 1] \right] \le s(n, t, \lambda) .
$$
In the above "$\leftrightarrow$" denotes the interactive nature of the protocol between $P$ and $V$.

6

We remark that in the soundness part, the only restriction on a malicious prover $P^*$ is the number of queries it can make to $G$ and $G^\dagger$. In particular, this means that even if $P^*$ has unbounded computational power, as long as it makes a small number of queries to $G$ and $G^\dagger$, it cannot fool the verifier.

## 2.3 Quantum Local Proofs

Next, we provide formal definitions of LocalQMA. For a reader familiar with QMA in the following definition, one can think of $x$ as the classical description of the local Hamiltonian problem, and $m(n)$ as the number of terms in it (i.e., $H = \sum_{i=1}^{m(n)} H_i$).

**Definition 2.2** (($k, \gamma$)-LocalQMA). For two constants $k, \gamma \in \mathbb{N}$, a promise problem $L = (L_{\text{yes}}, L_{\text{no}})$ is in the complexity class $(k, \gamma)$-LocalQMA with soundness $s(n)$ and completeness $c(n)$ if there are polynomials $m$ and $p$ such that the following hold:

- (**A $k$-local verifier $V_L$**) Let $n = |x|$. There is a verifier $V_L$ that acts as follows:
  1. $V_L$ gets access to a $p(n)$-qubit proof $\sigma$ for $L$ and draws $i \in_{\text{R}} [m(n)]$. $V_L$ then computes in $\text{poly}(n, k, \gamma)$ time a $k$-size subset $S_i \subseteq [p(n)]$ and a $\gamma$-size quantum circuit $C_i$ that is over the Clifford + $T$ gate-set and acts on $k$ qubits. $C_i$ may use $\gamma$ ancilla qubits, with the first ancilla qubit being the output qubit.
  2. $V_L$ next applies $C_i$ to the restriction of $\sigma$ on qubits in $S_i$ and measures the first ancilla qubit. $V_L$ accepts if the outcome is 1 and rejects otherwise.
- (**Completeness**) If $x \in L_{\text{yes}}$, there is a $p(n)$-qubit state $\sigma$ such that $V_L$ accepts $\sigma$ with probability at least $c(n)$.
- (**Soundness**) If $x \in L_{\text{no}}$, $V_L$ accepts every $p(n)$-qubit state $\sigma$ with probability at most $s(n)$.
- (**Strongly explicit**) Moreover, we say that $V_L$ is strongly explicit, if $V_L$ computes $S_i$ and $C_i$ in $\text{poly}(\log n, k, \gamma)$ time instead of $\text{poly}(n, k, \gamma)$ time.

We will use $(k, \gamma)\text{-LocalQMA}_{s, c}$ to denote the class above for notational convenience.

## 2.4 The Quantum PCP Conjecture

We first recall the quantum PCP conjecture [16, 17].

**Conjecture 2.3** (QPCP conjecture). There are constants $k \in \mathbb{N}$ and $\alpha, \beta$ satisfying $0 < \alpha < \beta \le 1$ such that $(\alpha, \beta)$-$k$-LH is QMA-complete.

In particular, the following corollary is immediate from the definition of $(\alpha, \beta)$-$k$-LH.

**Corollary 2.4.** If QPCP holds, then there are constants $k, \gamma \in \mathbb{N}$ and $c, s \in [0, 1]$ satisfying that $s < c$, such that
$$
\text{QMA} \subseteq (k, \gamma)\text{-LocalQMA}_{s, c} .
$$

# 3 A Candidate Succinct Argument for LocalQMA in QHROM

In this section, we present a candidate succinct argument for LocalQMA in QHROM. Assuming QPCP, this succinct argument also works for all of QMA.

7

## 3.1 The Succinct Protocol $\Pi_{\text{succinct}}$

**Notation.** Let $L = (L_{\text{yes}}, L_{\text{no}}) \in (k, \gamma)\text{-LocalQMA}_{s, c}$ for two integers $k, \gamma \in \mathbb{N}$ and two reals $s, c \in [0, 1]$ such that $s < c$. Let $m_L$ and $p_L$ be the polynomials and $V_L$ be the $k$-local verifier in Definition 2.2. Throughout this section, we will always use $n$ to denote the length of the input to $L$, $N = p_L(n)$ to denote the number of qubits in a witness for $V_L$, and $\lambda$ to denote the security parameter.

We set $b = \lambda/3$, and $l = N/b$. We assume that $b$ is an integer and $l$ is a power of 2 for simplicity and without loss of generality since one can always add dummy qubits to the witness.

[IMAGE: Figure 4: An illustration of the labeling of the nodes in the tree $T_l$ with $l$ leaves. Nodes are numbered 1 (root), 2, 3, 4, 5, 6, 7, ... down to the leaves $l+0, l+1, \dots, 2l-2, 2l-1$ on the bottom layer.]

**Figure 4:** An illustration of the labeling of the nodes in the tree $T_l$ with $l$ leaves

**The perfect binary tree $T_l$.** We will consider a perfect binary tree $T_l$ of $l$ leafs (see Figure 4 for an illustration). Note that $T_l$ has $\log l$ layers. We label the nodes in $T_l$ first from root to leaves and then from left to right, starting with 1.

For a node $u$ in $T_l$, we observe that $u$'s parent is $\lfloor u/2 \rfloor$ if $u$ is not the root (i.e., $u \neq 1$) and $u$'s two children are $2u$ and $2u + 1$ if $u$ is not a leaf (i.e., $u < l$). We use $P_u$ to denote the set of nodes consisting of $u$ and all ancestors of $u$. Formally, we have
$$
P_u = \begin{cases} \{u\} & \text{if } u = 1, \\ \{u\} \cup P_{\lfloor u/2 \rfloor} & \text{if } u > 1. \end{cases}
$$
We also define $R_u$ as follows:
$$
R_u = \{v \in P_u \text{ or } \lfloor v/2 \rfloor \in P_u : v \in [2l - 1]\} .
$$
That is, a node $v$ belongs to $R_u$ if either $v$ is in $P_u$ or the parent of $v$ is in $P_u$. Also, for a set of nodes $S \subseteq [2l - 1]$, we set $R_S = \bigcup_{u \in S} R_u$.

Given an $N$-qubit state $\sigma$, we define the following commitment algorithm (Algorithm 1) and the corresponding local decommitment algorithm (Algorithm 2).

8

**Algorithm 1:** Algorithm for committing to an $N$-qubit quantum state

1. **Function** $\text{commit}^{\mathcal{G}}(\sigma, N, \lambda)$
2. **Input:** $\sigma$ is an $N$-qubit quantum state, $\lambda$ is the security parameter (recall that $\lambda = 3b$)
3. **Let** $l = N/b$;
4. **For** each node $u$ in $T_l$, create a $b$-qubit register $\text{state}^{(u)}$;
5. Store $\sigma$ in registers $\text{state}^{(l)}, \text{state}^{(l+1)}, \ldots, \text{state}^{(2l-1)}$;
6. **for** $u$ from $l - 1$ down to $1$ **do**
7. $\quad$ Initialize $\text{state}^{(u)}$ as $|0^b\rangle$;
8. $\quad$ Apply $G$ on $\text{state}^{(2u)}, \text{state}^{(2u+1)}$, and $\text{state}^{(u)}$;
9. **return** all qubits in $\{\text{state}^{(u)}\}_{u \in [2l-1]}$; // Here $\text{state}^{(1)}$ is the commitment to be sent to the verifier, while the prover keeps all other states $\{\text{state}^{(u)}\}_{u \in \{2, \ldots, 2l-1\}}$

**Algorithm 2:** Algorithm for recovering part of the original quantum state

1. **Function** $\text{decommit}^{\mathcal{G}}(N, \lambda, S, \{\eta_u\}_{u \in R_S})$
2. **Input:** $S \subseteq \{l, \ldots, 2l - 1\}$ is a subset of leafs in $T_l$, for each $u \in R_S$, $\eta_u$ is a $b$-qubit quantum state, $\lambda$ is the security parameter. (We remark that $\{\eta_u\}_{u \in (R_S \setminus \{1\})}$ are the states provided by the prover to the verifier.)
3. **Let** $l = N/b$;
4. **For** each node $u$ in $R_S$, create a $b$-qubit register $\text{state}^{(u)}$, and store $\eta_u$ at $\text{state}^{(u)}$;
5. **for** $u \in R_S \cap [l - 1]$, from smallest to the largest **do**
6. $\quad$ Apply $G^\dagger$ on $\text{state}^{(2u)}, \text{state}^{(2u+1)}$, and $\text{state}^{(u)}$;
7. $\quad$ Measure $\text{state}^{(u)}$ in the computational basis to obtain an outcome $z \in \{0, 1\}^b$;
8. $\quad$ **if** $z \neq 0^b$ **then**
9. $\quad \quad$ **return** $\perp$ // $\perp$ means the check fails
10. **return** all qubits in $\{\text{state}^{(u)}\}_{u \in S}$;

Finally, we are ready to specify the following candidate succinct argument for $L \in (k, \gamma)\text{-LocalQMA}_{s, c}$:

**The candidate succinct argument $\Pi_{\text{succinct}}$ for $L \in (k, \gamma)\text{-LocalQMA}_{s, c}$**

- Both prover ($P$) and verifier ($V$) get access to a Haar random quantum unitary $G$ acting on $3b = \lambda$ qubits and its inverse $G^\dagger$. They also both get an input $x \in \{0, 1\}^n$ to $L$. The goal for the prover is to convince the verifier that $x \in L_{\text{yes}}$.
- Let $l = N/b$ and we assume that $l = 2^d$ for $d \in \mathbb{N}$.
- (**First message:** $P \to V$) The honest prover $P$ acts as follows: If $x \in L_{\text{no}}$, $P$ aborts immediately. Otherwise, $P$ finds an $N$-qubit state $\sigma$ such that $V_L$ accepts with probability at least $c$, and runs $\text{commit}^{\mathcal{G}}(\sigma, N, \lambda)$ to obtain qubits $\{\eta_u\}_{u \in [2l-1]}$. $P$ then sends $\eta_1$ to $V$.
- (**Second message:** $V \to P$) $V$ now simulates the local verifier $V_L$: $V$ first draws $i \in_{\text{R}} [m_L(n)]$ and sends $i$ to $P$, and then computes a $k$-size subset $S_i \subseteq [N]$ and a $\gamma$-size circuit $C_i$ acting on $k$ qubits, according to Definition 2.2.

9

Let $W_i$ be the set of leaves in $T_l$ that contain the qubits indexed by $S_i$. That is,
$$
W_i = \left\{ l + \lfloor (u - 1)/b \rfloor : u \in S_i \right\} .
$$
- (**Third message:** $P \to V$) The honest prover $P$ sends $\{\eta_u\}_{u \in R_{W_i}, u \neq 1}$ to $V$. $V$ then runs $\text{decommit}^{\mathcal{G}}(N, \lambda, W_i, \{\eta_u\}_{u \in R_{W_i}})$ (note that $V$ already has $\eta_1$). If $\text{decommit}$ returns $\perp$, $V$ rejects immediately.
Otherwise, $V$ continues the simulation of $V_L$ by running $C_i$ using $\{\eta_u\}_{u \in W_i}$, and $V$ accepts if and only if $V_L$ accepts.

## 3.2 Analysis of $\Pi_{\text{succinct}}$

We say a prover $P$ is **semi-honest**, if $P$ commits to an arbitrary $N$-qubit state (as opposed to the true ground state) $\sigma$ in the first message but indeed follows $\Pi_{\text{succinct}}$. We remark that, unlike an honest prover, a semi-honest prover may not necessarily commit to a state that makes $V_L$ accepts with probability at least $c$.⁴

Now we prove the completeness and succinctness of $\Pi_{\text{succinct}}$. We also show $\Pi_{\text{succinct}}$ is sound against semi-honest provers.⁵

**Theorem 3.1.** Let $\Pi_{\text{succinct}}$ be the protocol between $P$ and $V$ for the promise language $L \in (k, \gamma)\text{-LocalQMA}_{s, c}$. For every $x \in \{0, 1\}^n$, the following hold:

**Completeness:** If $x \in L_{\text{yes}}$, then for every $G \in U(2^\lambda)$,
$$
\text{Pr}[(\mathcal{V}^{G, G^\dagger} \leftrightarrow P^{G, G^\dagger})(x) = 1] \ge c .
$$

**Soundness against semi-honest provers:** If $x \in L_{\text{no}}$, then for every $G \in U(2^\lambda)$ and every semi-honest prover $P$,
$$
\text{Pr}[(\mathcal{V}^{G, G^\dagger} \leftrightarrow (P)^{G, G^\dagger})(x) = 1] \le s .
$$

**Succinctness:** $P$ and $V$ communicate at most $O(k \cdot \lambda \cdot \log n)$ qubits in total.

**Efficiency:** $V$ runs in $\text{poly}(n, k, \gamma)$ time. If $V_L$ is strongly explicit, then $V$ runs in $O(k \cdot \lambda \cdot \log n + \text{poly}(\log n, k, \gamma))$ time.

*Proof.* We first establish the succinctness part. Examining the protocol $\Pi_{\text{succinct}}$, one can see that the first message takes $O(\lambda)$ qubits, the second messages takes $O(\log m_L(n)) = O(\log n)$ classical bits, and the third message takes $O(|R_{W_i}| \cdot b)$ qubits. Note that $|W_i| \le k$. $|R_{W_i}| \le O(\log l) \le k \cdot O(\log N) \le O(k \cdot \log n)$, the total communication complexity is thus bounded by $O(k \cdot \lambda \cdot \log n)$.

For the running time of $V$, one can see that its running time is dominated by the running time of $\text{decommit}^{\mathcal{G}}(N, \lambda, W_i, \{\eta_u\}_{u \in R_{W_i}})$ and the running time of $V_L$ computing $W_i$ and $C_i$, which are at most $O(k \cdot \lambda \cdot \log N)$ and $\text{poly}(n, k, \gamma)$ respectively. The latter becomes $\text{poly}(\log n, k, \gamma)$ if $V_L$ is strongly explicit.

⁴In particular, a semi-honest prover may still commit to some state even when $x \in L_{\text{no}}$, while an honest prover would abort when $x \in L_{\text{no}}$.
⁵We remark that semi-honest prover security is more of a sanity check than a solid contribution since it is not hard to construct a trivial protocol that satisfies this semi-honest prover security.

10

Now we prove the completeness. Let $G^{(u)}$ be a $G$ gate applying on registers $\text{state}^{(2u)}$, $\text{state}^{(2u+1)}$, and $\text{state}^{(u)}$. Then we know for the honest prover $P$, when $x \in L_{\text{yes}}$, it prepares an $N$-qubit state $\sigma$ that makes $V_L$ accept with probability at least $c$, and then applies $U_{\text{com}} := G^{(l-1)} \cdots G^{(1)}$ to $\sigma \otimes |0\rangle \langle 0|_{\text{state}^{(1)}, \ldots, \text{state}^{(l-1)}}$.

Let $U_{\text{decom}} := U_{\text{com}}^\dagger = G^{(1)\dagger} \cdots G^{(l-1)\dagger}$. Recall that verifier $V$ at the end simulates the quantum circuit $C_i$ only on registers in $\{\text{state}^{(u)}\}_{u \in W_i}$. We now argue that $V$ is effectively simulating $C_i$ on
$$
U_{\text{decom}} U_{\text{com}} \sigma \otimes |0\rangle \langle 0|_{\text{state}^{(1)}, \ldots, \text{state}^{(l-1)}} = \sigma \otimes |0\rangle \langle 0|_{\text{state}^{(1)}, \ldots, \text{state}^{(l-1)}} .
$$
The reason is that $\text{decommit}^{\mathcal{G}}(N, \lambda, W_i, \{\eta_u\}_{u \in R_{W_i}})$ performs all gates in $U_{\text{decom}}$ that reside in the lightcone of the registers $\{\text{state}^{(u)}\}_{u \in W_i}$ in the chronological order (see Line 4 of Algorithm 2). Also, since $P$ starts with the state $\sigma |0\rangle \langle 0|_{\text{state}^{(1)}, \ldots, \text{state}^{(l-1)}}$, $\text{decommit}$ never outputs $\perp$. Therefore, $V$ is simulating $V_L$ faithfully on $\sigma$, meaning that it accepts with probability at least $c$.

Finally, we establish the soundness against semi-honest provers. The argument above for completeness indeed established that whenever the prover commits to a state $\sigma$ in the first message and follows $\Pi_{\text{succinct}}$ (i.e., the prover is semi-honest), for every possible $G$, the acceptance probability of $V$ equals the acceptance probability of the simulated $V_L$ on $\sigma$. Hence, when $x \in L_{\text{no}}$, for every semi-honest prover and every possible $G$, the acceptance probability of $V$ is at most $s$.

We conjecture that the soundness also holds more generally.

**Conjecture 3.2** ($\Pi_{\text{succinct}}$ is sound in QHROM). Let $\Pi_{\text{succinct}}$ be the protocol between $P$ and $V$ for the promise language $L \in (k, \gamma)\text{-LocalQMA}_{s, c}$. For every $x \in \{0, 1\}^n$, the following hold:

**Soundness:** If $x \in L_{\text{no}}$, then for every $t \in \mathbb{N}$ and all (potentially malicious) $P^*$ that makes at most $t$ total queries to $G$ and $G^\dagger$, for some $\delta = \delta(t, \lambda) = \text{poly}(t) / 2^{\Omega(\lambda)}$, it holds that
$$
\underset{G \in_{\text{R}} U(2^\lambda)}{\mathbb{E}} \left[ \text{Pr}[(\mathcal{V}^{G, G^\dagger} \leftrightarrow (P^*)^{G, G^\dagger})(x) = 1] \ge s + \delta \right] < \delta .
$$

Comment: The difference between this conjecture and our main theorem (Theorem 3.1) is that in the conjecture we require soundness against all unbounded provers instead of only against semi-honest provers.

## 3.3 Discussions

We remark that (1) the constant soundness in Conjecture 3.2 and the constant completeness in Theorem 3.1 can be easily amplified to $n^{-\omega(1)}$ and $1 - n^{-\omega(1)}$ by repeating the protocols $\log^2 n$ times, and (2) assuming QPCP, the protocol works for all languages in QMA.

**Corollary 3.3.** Assuming Conjecture 3.2, there is a protocol for $L \in (k, \gamma)\text{-LocalQMA}_{s, c}$ with $\lambda \cdot \text{polylog}(n)$ communication complexity, completeness $1 - n^{-\omega(1)}$ and soundness $n^{-\omega(1)}$ in QHROM. Also, if $V_L$ is strongly explicit, then the verifier running time of the protocol is also bounded by $\lambda \cdot \text{polylog}(n)$.

Moreover, if we further assume that QPCP holds, then the aforementioned succinct protocol exists for every $L \in \text{QMA}$.

11

How easy is it for the prover to cheat after having sent the commitment to the verifier in the quantum Merkle tree construction? We believe (but cannot yet prove; see Conjecture 3.2) that a computationally bounded prover will not be able to make the verifier accept. However, a computationally unbounded prover can. We now demonstrate this by the application of Schrödinger-HJW theorem [18] to the simple toy example of a Merkle tree with depth one. Suppose $|\psi\rangle$ is the $2b$-qubit state $P$ initially committed to, and $|\phi\rangle$ is another $2b$-qubit state that $P$ wishes to cheat by switching $|\psi\rangle$ with. Mathematically the process of committing, switching the initial state and lastly decommitting writes
$$
G^\dagger (W \otimes I) G (|\psi\rangle \otimes |0^b\rangle) \approx (|\phi\rangle \otimes |0^b\rangle) ,
$$
where in the above we think of $|0^b\rangle$ as the parent and see that the initially committed $2b$-qubit state $|\psi\rangle$ can be changed to another completely different $2b$-qubit state $|\phi\rangle$ by applying Schrödinger-HJW theorem (i.e., application of a $W \otimes I$) for a suitable unitary $W$ that acts on the first $2b$ qubits. We note that such $W$ exists, because the reduced density matrix of the last $b$ qubits of both $G(|\psi\rangle |0^b\rangle)$ and $G(|\phi\rangle |0^b\rangle)$ are very close to the maximally mixed state, for any two fixed states $|\phi\rangle$ and $|\psi\rangle$. However, we conjecture that finding $W$ requires computationally unbounded prover. For example, in the foregoing equation a direct way to solve for $W$ would require solving a linear system of equations that is exponentially large. Moreover, the oracle $G$ is fully random and does not afford any structure we can utilize to reduce the computation.

This is exacerbated by the fact that finding a unitary $W$ that makes the two sides approximately equal can also make the verifier accept with sufficiently high probability. We leave this resolution as a mathematical challenge.

### Acknowledgments

L.C. would like to thank Jiahui Liu and Qipeng Liu for helpful discussions and pointing out many related works. This work was done while L.C. did an internship at IBM Quantum Research. L. C. is supported by NSF CCF-2127597 and an IBM Fellowship.

### References

[1] Ralph C. Merkle. “A digital signature based on a conventional encryption function”. In *Advances in Cryptology - CRYPTO '87, A Conference on the Theory and Applications of Cryptographic Techniques, Santa Barbara, California, USA, August 16-20, 1987, Proceedings*. Volume 293 of *Lecture Notes in Computer Science*, pages 369–378. Springer (1987).
[2] Dan Boneh, Özgür Dagdelen, Marc Fischlin, Anja Lehmann, Christian Schaffner, and Mark Zhandry. “Random oracles in a quantum world”. In *Advances in Cryptology - ASIACRYPT 2011 - 17th International Conference on the Theory and Application of Cryptology and Information Security, Seoul, South Korea, December 4-8, 2011. Proceedings*. Volume 7073 of *Lecture Notes in Computer Science*, pages 41–69. Springer (2011).
[3] Gilles Brassard, David Chaum, and Claude Crépeau. “Minimum disclosure proofs of knowledge”. *J. Comput. Syst. Sci.* 37, 156–189 (1988).
[4] Joe Kilian. “A note on efficient zero-knowledge proofs and arguments (extended abstract)”. In S. Rao Kosaraju, Mike Fellows, Avi Wigderson, and John A. Ellis, editors, *Proceedings of the 24th Annual ACM Symposium on Theory of Computing, May 4-6, 1992, Victoria, British Columbia, Canada*. Pages 723–732. ACM (1992). url: `https://doi.org/10.1145/129712.129782`.

12

[5] Silvio Micali. “Computationally sound proofs”. *SIAM J. Comput.* 30, 1253–1298 (2000). url: `https://doi.org/10.1137/S0097539795284959`.
[6] Eli Ben-Sasson, Alessandro Chiesa, and Nicholas Spooner. “Interactive oracle proofs”. In Martin Hirt and Adam D. Smith, editors, *Theory of Cryptography - 14th International Conference, TCC 2016-B, Beijing, China, October 31 - November 3, 2016, Proceedings, Part II*. Volume 9986 of *Lecture Notes in Computer Science*, pages 31–60. (2016). url: `https://doi.org/10.1007/978-3-662-53644-5_2`.
[7] Alessandro Chiesa, Peter Manohar, and Nicholas Spooner. “Succinct arguments in the quantum random oracle model”. In Dennis Hofheinz and Alon Rosen, editors, *Theory of Cryptography - 17th International Conference, TCC 2019, Nuremberg, Germany, December 1-5, 2019, Proceedings, Part II*. Volume 11892 of *Lecture Notes in Computer Science*, pages 1–29. Springer (2019). url: `https://doi.org/10.1007/978-3-030-36033-7_1`.
[8] Alessandro Chiesa, Fermi Ma, Nicholas Spooner, and Mark Zhandry. “Post-quantum succinct arguments: Breaking the quantum rewinding barrier”. In *62nd IEEE Annual Symposium on Foundations of Computer Science, FOCS 2021, Denver, CO, USA, February 7-10, 2022*. Pages 49–58. IEEE (2021).
[9] Zhengfeng Ji, Yi-Kai Liu, and Fang Song. “Pseudorandom quantum states”. In Hovav Shacham and Alexandra Boldyreva, editors, *Advances in Cryptology - CRYPTO 2018 - 38th Annual International Cryptology Conference, Santa Barbara, CA, USA, August 19-23, 2018, Proceedings, Part III*. Volume 10993 of *Lecture Notes in Computer Science*, pages 126–152. Springer (2018).
[10] Fernando GSL Brandao, Aram W Harrow, and Michał Horodecki. “Local random quantum circuits are approximate polynomial-designs”. *Communications in Mathematical Physics* 346, 397–434 (2016). url: `https://doi.org/10.1007/s00220-016-2706-8`.
[11] Mark Zhandry. “How to record quantum queries, and applications to quantum indifferentiability”. In *Advances in Cryptology - CRYPTO 2019 - 39th Annual International Cryptology Conference, Santa Barbara, CA, USA, August 18-22, 2019, Proceedings, Part II*. Volume 11693 of *Lecture Notes in Computer Science*, pages 239–268. Springer (2019).
[12] Sam Gunn, Nathan Ju, Fermi Ma, and Mark Zhandry. “Commitments to quantum states”. In Barna Saha and Rocco A. Servedio, editors, *Proceedings of the 55th Annual ACM Symposium on Theory of Computing, STOC 2023, Orlando, FL, USA, June 20-23, 2023*. Pages 1579–1588. ACM (2023).
[13] Anne Broadbent, Zhengfeng Ji, Fang Song, and John Watrous. “Zero-knowledge proof systems for QMA”. *SIAM J. Comput.* 49, 245–283 (2020).
[14] Andrea Coladangelo, Thomas Vidick, and Tina Zhang. “Non-interactive zero-knowledge arguments for qma, with preprocessing”. In Daniele Micciancio and Thomas Ristenpart, editors, *Advances in Cryptology - CRYPTO 2020 - 40th Annual International Cryptology Conference, CRYPTO 2020, Santa Barbara, CA, USA, August 17-21, 2020, Proceedings, Part III*. Volume 12172 of *Lecture Notes in Computer Science*, pages 799–828. Springer (2020).
[15] Anne Broadbent and Alex Bredariol Grilo. “Qma-hardness of consistency of local density matrices with applications to quantum zero-knowledge”. *SIAM J. Comput.* 51, 1400–1450 (2022).
[16] Dorit Aharonov, Itai Arad, Zeph Landau, and Umesh V. Vazirani. “The detectability lemma and quantum gap amplification”. In *Proceedings of the 41st Annual ACM Symposium on Theory of Computing, STOC 2009, Bethesda, MD, USA, May 31 - June 2, 2009*. Pages 417–426. ACM (2009).

13

[17] Dorit Aharonov, Itai Arad, and Thomas Vidick. “Guest column: the quantum PCP conjecture”. *SIGACT News* 44, 47–79 (2013). url: `https://doi.org/10.1145/2491533.2491549`.
[18] Lane P Hughston, Richard Jozsa, and William K Wootters. “A complete classification of quantum ensembles having a given density matrix”. *Physics Letters A* **183**, 14–18 (1993). url: `https://doi.org/10.1016/0375-9601(93)90880-9`.

14