**Session 2C: Secure Computing I**

**CCS ’19, November 11–15, 2019, London, United Kingdom**

**Efficient Two-Round OT Extension
and Silent Non-Interactive Secure Computation**

Elette Boyle
IDC Herzliya
Israel
eboyle@alum.mit.edu

Geoffroy Couteau
Karlsruhe Institute of Technology
Germany
geoffroy.couteau@kit.edu

Niv Gilboa
Ben Gurion University
Israel
gilboan@bgu.ac.il

Yuval Ishai
Technion
Israel
yuvali@cs.technion.ac.il

Lisa Kohl
Karlsruhe Institute of Technology
Germany
lisa.kohl@kit.edu

Peter Rindal
Visa Research
rindalp@oregonstate.edu

Peter Scholl
Aarhus University
Denmark
peter.scholl@cs.au.dk

**ABSTRACT**
We consider the problem of securely generating useful instances
of two-party correlations, such as many independent copies of a
random oblivious transfer (OT) correlation, using a small amount
of communication. This problem is motivated by the goal of secure
computation with silent preprocessing, where a low-communication
input-independent setup, followed by local (“silent”) computation,
enables a lightweight “non-cryptographic" online phase once the
inputs are known.
Recent works of Boyle et al. (CCS 2018, Crypto 2019) achieve this
goal with good concrete efficiency for useful kinds of two-party
correlations, including OT correlations, under different variants
of the Learning Parity with Noise (LPN) assumption, and using a
small number of "base" oblivious transfers. The protocols of Boyle
et al. have several limitations. First, they require a large number
of communication rounds. Second, they are only secure against
semi-honest parties. Finally, their concrete efficiency estimates are
not backed by an actual implementation. In this work we address
these limitations, making three main contributions:
*   **Eliminating interaction.** Under the same assumption, we ob-
    tain the first concretely efficient 2-round protocols for generating
    useful correlations, including OT correlations, in the semi-honest
    security model. This implies the first efficient 2-round OT ex-
    tension protocol of any kind and, more generally, protocols for
    non-interactive secure computation (NISC) that are concretely
    efficient and have the silent preprocessing feature.
*   **Malicious security.** We provide security against malicious par-
    ties without additional interaction and with only a modest over-
    head; prior to our work, no similar protocols were known with
    any number of rounds.
*   **Implementation.** Finally, we implemented, optimized, and bench-
    marked our 2-round OT extension protocol, demonstrating that it
    offers a more attractive alternative to the OT extension protocol
    of Ishai et al. (Crypto 2003) in many realistic settings.

**CCS CONCEPTS**
*   **Security and privacy → Cryptography.**

**KEYWORDS**
secure computation; oblivious transfer extension; pseudorandom
correlation generators; active security

**ACM Reference Format:**
Elette Boyle, Geoffroy Couteau, Niv Gilboa, Yuval Ishai, Lisa Kohl, Peter
Rindal, and Peter Scholl. 2019. Efficient Two-Round OT Extension and Silent
Non-Interactive Secure Computation. In *2019 ACM SIGSAC Conference on
Computer and Communications Security (CCS ’19), November 11–15, 2019,
London, United Kingdom*. ACM, New York, NY, USA, 18 pages. https://doi.
org/10.1145/3319535.3354255

# 1 INTRODUCTION

There is a large body of work on optimizing the concrete efficiency
of secure computation protocols via input-independent preprocess-
ing. By securely generating many instances of simple correlations,
one can dramatically reduce the online communication and com-
putation costs of most existing protocols.

To give just one example, multiple independent instances of a
random oblivious transfer¹ (OT) correlation can be used for se-
cure two-party computation of Boolean circuits in the semi-honest
model, with communication cost of only two bits per party per
(nonlinear) gate, and with computation cost that is comparable to
computing the circuit with no security requirements at all [9, 33, 43].
Thus, assuming a fast communication network, protocols based on
correlated randomness can achieve near-optimal performance.

The main challenge in applying this approach is the high con-
crete cost of securely generating the correlated randomness. Tradi-
tional solutions involve carefully optimized special-purpose secure
computation protocols that have a high communication cost for
each instance of the desired correlation [11, 22]. This holds even for
the case of OT correlations, for which relatively fast OT extension
techniques are known [8, 37, 41]. Moreover, even if offline commu-
nication is cheap, the cost of storing large amounts of correlated
randomness for each party with whom a future interaction might
take place can be significant.

Motivated by the limitations of traditional approaches for gener-
ating and storing correlated randomness, the notion of a **pseudoran-
dom correlation generator (PCG)** was recently proposed and studied
by Boyle et al. [15, 16]. The goal of a PCG is to compress long
sources of correlated randomness without violating security. More
concretely, a (two-party) PCG replaces a target two-party correla-
tion, say many independent OT correlation instances, by a pair of
short correlated keys, which can be “silently" expanded without any
interaction. The process of generating the correlated keys and locally
expanding them should emulate an ideal process for generating the
target correlation not only from the point of view of outsiders, but
also from the point of view of insiders who can observe one of the
two keys. Among other results, the aforementioned works of Boyle
et al. [15, 16] obtain concretely efficient constructions of PCGs for
OT correlations and **vector oblivious linear evaluation (VOLE)** cor-
relations [4, 39, 48] based on variants of the Learning Parity with
Noise assumption [12]. These PCG constructions are motivated by
the goal of secure computation with **silent preprocessing**, where a
low-communication input-independent setup, followed by local
(“silent") computation, enables a lightweight “non-cryptographic"
online phase once the inputs are known.

However, towards realizing this goal, one major challenge re-
mains: how can the pair of keys be securely generated? While
the keys are short, their sampling algorithm is quite complex and
involves multiple invocations of cryptographic primitives. Thus,
even applying the fastest general-purpose protocols (e.g., [40]) for
generating these keys incurs a very significant overhead.

An alternative approach for distributing the PCG key generation,
suggested in [15, 16], relies on a recent special-purpose protocol of
Doerner and shelat [24] for secure key generation of a **distributed
point function (DPF)** [17, 30]. This protocol only makes a black-box
use of symmetric cryptography and a small number of oblivious
transfers, and hence it is also concretely efficient. Using this proto-
col for distributing the key generation of a PCG for OT correlations,
Boyle et al. [16] obtained a **silent OT extension protocol** that gener-
ates (without any trusted setup) a large number of pseudo-random
OTs from a small number of base OTs, using a low-communication
setup followed by silent key expansion [16].

While the silent OT extension protocol from [16] and other pro-
tocols obtained using this approach have good concrete efficiency,
they also have several limitations. First, they require a large num-
ber of communication rounds that grows (at least) logarithmically
with the output length. Second, they are only secure against semi-
honest parties. Both of the above limitations are inherited from
the DPF key generation protocol of [24]. Finally, their concrete
efficiency estimates are not backed by an actual implementation,
and ignore possible cache-misses and other system- and network-
related sources of slowdown.

## 1.1 Our Contribution

In this work, we address the above limitations by making the fol-
lowing contributions.

**Two-Round Silent OT Extension.** We present the first concretely
efficient **two-round OT extension protocol**, based on a variant of
the LPN assumption. The protocol has a **silent preprocessing** fea-
ture, allowing the parties to push the bulk of the computational
work to an offline phase. It can be used in two modes: either a
**random-input mode**, where the communication complexity is sub-
linear in the outputs length, or a **chosen-input mode**, where the
communication is roughly equal to the total input length. This
applies even to the more challenging case of 1-bit OT, for which
standard OT extension techniques that make a black-box use of
symmetric cryptography [8, 37, 41, 44] have a high communication
overhead compared to the input length. A key idea that underlies
this improvement is replacing the DPF primitive in the PCG for
OT from [16] by the simpler **puncturable pseudorandom function
(PPRF)** primitive [14, 18, 42]. We design a parallel version of the
distributed key generation protocol from [24] that applies to a PPRF
instead of a DPF.

Our OT extension protocol bypasses a recent impossibility result
of Garg et al. [28] on 2-round OT extension due to the use of the
LPN assumption. While our construction (inevitably) does not fall
into the standard black-box framework considered [28], it still has
a black-box flavor in that it only invokes a **syndrome computation**
of any error-correcting code for which the LPN assumption holds.
We remark that aside from its concrete efficiency, our 2-round
OT extension protocol can be based on a conservative variant of
(binary) LPN in a noise regime that is not known to imply public-key
encryption, let alone oblivious transfer [3].

The technique we use for generating OT correlations in two
rounds can also be applied to VOLE correlations, as well as general
protocols for **non-interactive secure computation (NISC)** with silent
preprocessing.

**Malicious Security.** We present simple, practical techniques for
secure distributed setup of PPRF keys with a weak form of mali-
cious security. This suffices to upgrade our semi-honest OT and
VOLE protocols to malicious security, at a very low cost. Our main
protocols in this setting have 4 rounds of interaction, but this can be
reduced to 2 rounds using the Fiat-Shamir transform. We can also
use this to obtain **maliciously secure silent NISC** or **two-round OT
extension on chosen inputs**. These protocols are based on slightly
stronger variants of LPN, where the adversary is allowed a single
query to a one-bit leakage function on the error vector.

**Implementation.** We demonstrate the efficiency of our construc-
tions with an implementation of our random OT extension protocol.
The most costly part of the implementation is a large matrix-vector
multiplication, which comes from applying the LPN assumption.
We optimize this using **quasi-cyclic codes**, similarly to several re-
cent, candidate post-quantum secure cryptosystems, and present
different tradeoffs with parameter choices. Our protocols have a
very low communication overhead and perform significantly faster
than previous, state-of-the-art protocols [8, 37, 41] in environments
with restricted bandwidth. For instance, in a 100Mbps WAN setting,
we are around 5x faster, and this improves to 47x in a 10MBps
WAN. This is because, while our computational costs are around an
order of magnitude higher, we need around 1000–2000 times less
communication than the other protocols.

**Applications.** As well as the new application to NISC with silent
preprocessing, our protocols can be applied to a range of traditional
secure computation tasks. Below we mention just a few areas where
we expect silent OT extension to have an impact.
*   **Semi-honest MPC.** In the semi-honest “GMW protocol” [33],
    the correlated randomness needed to evaluate a Boolean cir-
    cuit can be obtained from two random OTs per AND gate.
    Plugging in our random OT extension, we obtain a practical
    2-PC protocol where each party communicates just 2 bits per
    AND gate on average. This is around 30x less communication
    than the state-of-the-art [23].
*   **Malicious secure MPC.** Protocols based on authenticated
    garbling [55, 56] are currently the state-of-the-art in mali-
    ciously secure MPC for binary circuits, particularly in a high-
    latency scenario. The main cost in these protocols comes
    from a preprocessing phase, where the parties use a large
    number of random, correlated oblivious transfers to produce
    correlated randomness. Our protocol can produce the same
    kind of oblivious transfers with almost zero communication,
    and we estimate this could reduce the overall communication
    in authenticated garbling protocols by around an order of
    magnitude.
*   **Private set intersection (PSI).** In circuit-based PSI, a generic
    2-PC protocol is used to first compute a secret-sharing of
    the intersection of two sets, and then perform some useful
    computation on the result [36, 50, 51]. With the improve-
    ments to GMW mentioned above, we can expect to obtain a
    similar reduction in communication for these families of PSI
    protocols.

## 1.2 Technical Overview

We now give an overview of our silent constructions in the semi-
honest and malicious settings. For simplicity, we focus here on the
case of 1-out-of-2 oblivious transfer.

We start by recalling the high-level idea of the pseudorandom
correlation generators for vector-OLE (VOLE) and OT from [15, 16].
These constructions distribute a pair of seeds to a sender and a
receiver, who can then locally expand the seeds to produce many
instances of pseudorandom OT or VOLE. To do so, they use two
main ingredients: a variant of the LPN assumption, and a method
for the two parties to obtain a compressed form of random secret
shares $\mathbf{z}_0, \mathbf{z}_1$, satisfying

$$
\mathbf{z}_1 = \mathbf{z}_0 + \mathbf{e} \cdot \mathbf{x} \in \mathbb{F}_2^{N \times \lambda} \quad (1)
$$

where $\mathbf{e} \in \{0, 1\}^N$ is a random, sparse vector held by one party,
and $\mathbf{x} \in \mathbb{F}_{2^\lambda}$ is a random field element held by the other party.

Given this, the shares can be randomized by taking a public,
binary matrix $\mathbf{H}$ that compresses from $N$ down to $n \ll N$ elements,
and locally multiplying each share with $\mathbf{H}$. This works because
$\mathbf{u} = \mathbf{e} \cdot \mathbf{H}$ is pseudorandom under a suitable variant of LPN. Writing
$\mathbf{v} = \mathbf{z}_0 \cdot \mathbf{H}$ and $\mathbf{w} = \mathbf{z}_1 \cdot \mathbf{H}$, from (1) we then get $\mathbf{w} = \mathbf{v} + \mathbf{u} \mathbf{x}$. This
can be seen as a set of random correlated OTs, where $\mathbf{u}_i \in \{0, 1\}$ are
the receiver's choice bits, and $(\mathbf{v}_i, \mathbf{v}_i + \mathbf{x})$ are the sender's strings,
of which the receiver learns $\mathbf{w}_i$. These can be locally converted into
random string-OTs with a standard hashing technique [37].

To obtain a compressed form of the shares in (1), the construc-
tions of [15, 16] used a **distributed point function (DPF)** [17, 30].
Our first observation is that the distributed point function is an
overkill for this application,² and can be replaced with the simpler
**puncturable pseudorandom function (PPRF)** primitive [14, 18, 42].
We design a parallel version of the distributed key generation protocol from [24] that applies to a PPRF
instead of a DPF.

In the setup procedure, we will give the sender a random key
$\mathbf{k}$ and $\mathbf{x}$, and give to the receiver a random point $\mathbf{a} \in \{1, \ldots, N\}$,
a punctured key $\mathbf{k}_{\{\mathbf{a}\}}$, and the value $\mathbf{z} = \text{PRF}(\mathbf{k}, \mathbf{a}) + \mathbf{x}$. Given these
seeds, the sender and receiver can now define the expanded outputs,
for $i = 1, \ldots, n$:

$$
\mathbf{z}_0^{[i]} = \mathbf{F}(\mathbf{k}, i), \mathbf{z}_1^{[i]} = \begin{cases} \mathbf{F}(\mathbf{k}, i) + \mathbf{x} & i \neq \mathbf{a} \\ \mathbf{z} & i = \mathbf{a} \end{cases}
$$

These immediately satisfy (1), with $\mathbf{e}$ as the $\mathbf{a}$-th unit vector. To
obtain sharings of sparse $\mathbf{e}$ with, say, $t$ non-zero coordinates, as
needed to use LPN, we repeat this $t$ times and XOR together all $t$
sets of outputs.

Conceptually, this construction is simpler than using a DPF, and
moreover, as we now show, it brings several efficiency advantages.

**Two-Round Setup of Puncturable PRF Keys.** We present a sim-
ple, two-round protocol for distributed the above setup with semi-
honest security, inspired by the DPF setup protocol of Doerner and
shelat [24]. The core of our protocol is the following procedure.
For each of $t$ secret LPN noise coordinates $\mathbf{a}_j \in [N]$ known to the
receiver, the sender generates a fresh PRF key $\mathbf{k}_j$, and wishes to
obliviously communicate a punctured key $\mathbf{k}_{j\{\mathbf{a}_j\}}$ and hardcoded
punctured output $\mathbf{z}_j = \text{PRF}(\mathbf{k}_j, \mathbf{a}) + \mathbf{x}$ to the receiver. Combined,
this yields a secret sharing of the vector $\mathbf{x} \mathbf{e}$, as required. To do
so, for each $\mathbf{k}_{\{\mathbf{a}\}}$, the parties made use of $\ell = \log N$ parallel OT
executions: the sender's $\ell$ message pairs correspond to appropriate
sums of partial evaluations from a consistent GGM PRF tree and
his secret value $\mathbf{x}$, and the receiver's $\ell$ selection bits correspond to
the bits of his chosen path $\mathbf{a}$.

Compared with previous works based on distributed point func-
tions [15, 16, 24], the number of rounds of interaction collapses
from $O(\log N)$ to just two, given any two-round OT protocol. This
is possible since the punctured point $\mathbf{a}$ is known to the receiver,
whereas when $\mathbf{a}$ is secret-shared as in a DPF, the OTs in the setup
procedure seem hard to parallelize.

**Two-Round OT Extension and Silent NISC.** We observe that in the
two-round setup, the receiver can already compute part of its output
before sending the first round message. In the case of OT, this part
corresponds to its random vector of choice bits $\mathbf{u}$. This means that
the receiver can already **derandomize** its OT outputs in the first
round, by sending in parallel with its setup message the value $\mathbf{u} + \mathbf{c}$,
where $\mathbf{c}$ is its chosen input vector. Since the sender can compute
its random OT outputs after the first round, this leads to a **two-
round OT extension protocol** that additionally enjoys the “**silent
preprocessing**" feature of pushing the bulk of the computation to an
offline phase, before the inputs are known. This can be generalized
from OT to VOLE and other useful instances of non-interactive
secure computation (NISC) [38], simultaneously inheriting the silent
preprocessing feature from the PCG and the interaction feature
from an underlying NISC protocol. See Section 3 for a more detailed
discussion of our new notion of NISC with silent preprocessing.

**Maliciously Secure Setup.** In the above semi-honest setup proce-
dure, a malicious receiver has no cheating space; altered selection
bits merely correspond to a different choice of noise coordinate
$\mathbf{a}' \in [N]$. However, a malicious sender may generate message pairs
inconsistent with any correct PRF evaluation tree, or use inconsis-
tent inputs $\mathbf{x}$ across the $t$ executions (in which case the outputs are
not valid shares of $\mathbf{x} \mathbf{u}$ for any single $\mathbf{x}$). For example, by injecting
errors into one of the two messages within an OT message pair,
the sender can effectively “guess" and learn a bit of $\mathbf{a}$, and will go
unnoticed if his guess is correct.

We demonstrate that with small overhead, we can restrict a mali-
cious sender to only such **selective-failure attacks**. This is formalized
via an **ideal functionality** where the adversarial sender can send a
guess range $\mathbf{I} \subset [N]$ for $\mathbf{a}$, a “**getting caught**" predicate is tested as
a function of the receiver’s true input, and the functionality either
aborts or delivers the output accordingly. We then show that paired
with an **interactive leakage notion for LPN**, this suffices to give us
PCG setup protocols for VOLE and OT with malicious security.

Our basic maliciously secure protocols have 4 rounds, but this
can be compressed to two rounds with the Fiat-Shamir transform,
in the **random oracle model**. Just as in the semi-honest protocols,
we can convert the setup protocols into NISC protocols, this time
under a slightly stronger variant of LPN with one bit of **adaptive
leakage** on the error vector, obtaining **two-round OT extension
with malicious security**.

# 2 PRELIMINARIES

## 2.1 Puncturable Pseudorandom Function

Pseudorandom functions (PRF) are keyed functions which are in-
distinguishable from truly random functions, have been introduced
in [32]. A **puncturable pseudorandom function (PPRF)** is a PRF $\mathbf{F}$
such that given an input $\mathbf{x}$, and a PRF key $\mathbf{k}$, one can generate a
punctured key, denoted $\mathbf{k}_{\{\mathbf{x}\}}$, which allows evaluating $\mathbf{F}$ at every
point except for $\mathbf{x}$, and does not conceal any information about the
value of $\mathbf{F}$ at $\mathbf{x}$. PPRFs have been introduced in [14, 18, 42].

**Definition 2.1** (*t*-**Puncturable Pseudorandom Function**). A punc-
turable pseudorandom function (PPRF) with key space $\mathcal{K}$, domain
$\mathcal{X}$, and range $\mathcal{Y}$, is a pseudorandom function $\mathbf{F}$ with an additional
punctured key space $\mathcal{K}_p$ and three probabilistic polynomial-time
algorithms ($\mathbf{F}.\text{KeyGen}, \mathbf{F}.\text{Puncture}, \mathbf{F}.\text{Eval}$) such that
*   $\mathbf{F}.\text{KeyGen}(1^\lambda)$ outputs a random key $\mathbf{K} \in \mathcal{K}$,
*   $\mathbf{F}.\text{Puncture} (\mathbf{K}, \mathcal{S})$, on input a key $\mathbf{K} \in \mathcal{K}$, and a subset $\mathcal{S} \subset \mathcal{X}$
    of size $t$, outputs a punctured key $\mathbf{K}_{\{\mathcal{S}\}} \in \mathcal{K}_p$,
*   $\mathbf{F}.\text{Eval}(\mathbf{K}_{\{\mathcal{S}\}}, \mathbf{x})$, on input a key $\mathbf{K}_{\{\mathcal{S}\}}$ punctured at all points
    in $\mathcal{S}$, and a point $\mathbf{x}$, outputs $\mathbf{F}(\mathbf{K}, \mathbf{x})$ if $\mathbf{x} \notin \mathcal{S}$, and $\bot$ otherwise,
such that no probabilistic polynomial-time adversary wins the ex-
periment $\text{Exp-}s\text{-pPRF}$ represented on Figure 1 with non-negligible
advantage over the random guess.
By $\mathbf{F}.\text{FullEval}(\mathbf{K})$ we denote the algorithm that on input a key
$\mathbf{K} \in \mathcal{K}$ evaluates $\mathbf{F}$ on all inputs $\mathcal{X}$ and returns the vector of outputs.

**Experiment $\text{Exp-}s\text{-pPRF}$**
**Setup Phase.** The adversary $\mathcal{A}$ sends a size-$t$ subset $\mathcal{S}^* \in \mathcal{X}$ to
the challenger. When it receives $\mathcal{S}^*$, the challenger picks
$\mathbf{K} \leftarrow \mathbf{F}.\text{KeyGen}(1^\lambda)$ and a random bit $b \leftarrow \{0, 1\}$.
**Challenge Phase.** The challenger sends $\mathbf{K}_{\{\mathcal{S}^*\}} =
\mathbf{F}.\text{Puncture}(\mathbf{K}, \mathcal{S}^*)$ to $\mathcal{A}$. If $b = 0$, the challenger
additionally sends $(\mathbf{F}(\mathbf{K}, \mathbf{x}))_{\mathbf{x} \in \mathcal{S}^*}$ to $\mathcal{A}$; otherwise, if
$b = 1$, the challenger picks $t$ random values $(\mathbf{y}_{\mathbf{x}})_{\mathbf{x} \in \mathcal{S}^*}, \mathbf{y} \in \mathcal{Y}$
for every $\mathbf{x} \in \mathcal{S}^*$ and sends them to $\mathcal{A}$.
**Figure 1:** Selective security game for puncturable pseudo-
random functions. At the end of the experiment, $\mathcal{A}$ sends
a guess $b'$ and wins if $b' = b$.

A PPRF can be constructed from any length-doubling pseudo-
random generator, using the GGM tree-based construction [14,
18, 32, 42]. The construction proceeds as follows: On input a key
$\mathbf{K}$ and a point $\mathbf{x}$, set $\mathbf{K}^{(0)} \leftarrow \mathbf{K}$ and perform the following iter-
ative evaluation procedure: for $i = 1$ to $\ell \leftarrow \log |\mathcal{X}|$, compute
$(\mathbf{K}^{(i)}_0, \mathbf{K}^{(i)}_1) \leftarrow \mathbf{G}(\mathbf{K}^{(i-1)})$, and set $\mathbf{K}^{(i)} \leftarrow \mathbf{K}^{(i)}_{x_i}$. Output $\mathbf{K}^{(\ell)}$. This
procedure creates a complete binary tree with edges labeled by
keys; the output of the PRF on an input $\mathbf{x}$ is the key labeling the
leaf at the end of the path defined by $\mathbf{x}$ from the root of the tree.
*   $\mathbf{F}.\text{KeyGen}(1^\lambda)$: output a random seed for $\mathbf{G}$.
*   $\mathbf{F}.\text{Puncture}(\mathbf{K}, \mathbf{x}) :$ on input a key $\mathbf{K} \in \{0, 1\}^\lambda$ and a point $\mathbf{x}$,
    apply the above procedure and return $\mathbf{K}_{\{\mathbf{x}\}} = (\mathbf{K}^{(1)}_{1-x_1}, \ldots,
    \mathbf{K}^{(\ell)}_{1-x_{\ell}})$.
*   $\mathbf{F}.\text{Eval}(\mathbf{K}_{\{\mathbf{x}\}}, \mathbf{x}')$, on input a punctured key $\mathbf{K}_{\{\mathbf{x}\}}$ and a point
    $\mathbf{x}'$, if $\mathbf{x} = \mathbf{x}'$, output $\bot$. Otherwise, parse $\mathbf{K}_{\{\mathbf{x}\}}$ as $(\mathbf{K}^{(1)}_{1-x_1}, \ldots,
    \mathbf{K}^{(\ell)}_{1-x_{\ell}})$ and start the iterative evaluation procedure from the
    first $\mathbf{K}^{(i)}_{1-x_i}$ such that $x'_i = 1 - x_i$.

To obtain a $t$-puncturable PRF with input domain $[N]$, one can
simply run $t$ instances of the above puncturable PRF and set the
output of the PRF to be the bitwise xor of the output of each instance.
With this construction, the length of a key punctured at $t$ points is
$t \lambda \log N$, where $\lambda$ is the seed size of the PRG.

## 2.2 Learning Parity with Noise

In this work, we rely on variants of the **Learning Parity with Noise
(LPN)** assumption [12] over either $\mathbb{F}_2$ or a large finite field $\mathbb{F}$, where
the noise is assumed to have a small Hamming weight. Similar
assumptions have been used in the context of secure arithmetic
computation [4, 25, 29, 39, 48]; unlike most of these works, the
flavors of LPN on which we rely do not require the underlying
code to have an algebraic structure and are thus not susceptible to
algebraic (list-) decoding attacks.

**Definition 2.2 (LPN).** Let $\mathcal{D}(\mathcal{R}) = \{\mathcal{D}_{k, q}(\mathcal{R})\}_{k, q \in \mathbb{N}}$ denote a
family of distributions over a ring $\mathcal{R}$, such that for any $k, q \in \mathbb{N}$,
$\text{Im}(\mathcal{D}_{k, q}(\mathcal{R})) \subseteq \mathcal{R}^q$. Let $\mathcal{C}$ be a probabilistic code generation algo-
rithm such that $\mathcal{C}(k, q, \mathcal{R})$ outputs a matrix $\mathbf{A} \in \mathcal{R}^{k \times q}$. For dimen-
sion $k = k(\lambda)$, number of samples (or block length) $q = q(\lambda)$, and
ring $\mathcal{R} = \mathcal{R}(\lambda)$, the $(\mathcal{D}, \mathcal{C}, \mathcal{R})\text{-LPN}(k, q)$ assumption states that

$$
\{(\mathbf{A}, \mathbf{b}) \mid \mathbf{A} \leftarrow \mathcal{C}(k, q, \mathcal{R}), \mathbf{e} \leftarrow \mathcal{D}_{k, q}(\mathcal{R}), \mathbf{s} \leftarrow \mathcal{R}^k, \mathbf{b} \leftarrow \mathbf{s} \cdot \mathbf{A} + \mathbf{e} \} \\
\approx_c \{(\mathbf{A}, \mathbf{b}) \mid \mathbf{A} \leftarrow \mathcal{C}(k, q, \mathcal{R}), \mathbf{b} \leftarrow \mathcal{R}^q \}
$$

Here and in the following, all parameters are functions of the
security parameter $\lambda$ and **computational indistinguishability** is de-
fined with respect to $\lambda$. When $\mathcal{R} = \mathbb{F}_2$ and $\mathcal{D}$ is the Bernoulli
distribution over $\mathbb{F}_2^q$, where each coordinate is 1 with probability
$r$ and 0 otherwise, this corresponds to the standard **binary LPN**
assumption. Note that the search LPN problem, of finding the vec-
tor $\mathbf{s}$, can be reduced to the decisional LPN assumption as defined
above above when the code generator $\mathcal{C}$ outputs a uniform matrix
$\mathbf{A}$ [5, 12]. However, this is less relevant for us as we are mainly inter-
ested in efficient variants with more structured codes. See [26] for
further discussion of search-to-decision reductions in the general
case.

**Example: LPN with Fixed Weight Noise.** For a finite field $\mathbb{F}_p$, we
denote by $\text{HW}_r(\mathbb{F}_p^q)$ the distribution of uniform, weight $r$ vectors
over $\mathbb{F}_p^q$; that is, a sample from $\text{HW}_r(\mathbb{F}_p^q)$ is a uniformly random
nonzero field element in $r$ random positions, and zero elsewhere.
The $(\text{Ber}, (\mathbb{F}_p^q), \mathcal{C}, \mathbb{F}_p)\text{-LPN}(k, q)$ assumption corresponds to the
standard (non-binary, fixed-weight) LPN assumption over a field $\mathbb{F}_p$
with code generator $\mathcal{C}$, dimension $k$, number of samples (or block
length) $q$, and noise rate $r$.
When the block length $q$ and noise rate $r$ are such that $k$ random
coordinates will be all noiseless with non-negligible probability
(e.g., when $r$ is constant and $q = \mathcal{O}(k^2)$), LPN can be broken via
Gaussian elimination (cf. [7]). This attack does not apply to our
constructions, which typically have $q = \mathcal{O}(k)$.

**Definition 2.3 (dual LPN).** Let $\mathcal{D}(\mathcal{R})$ and $\mathcal{C}$ be as in Definition 2.2,
$n, N \in \mathbb{N}$ with $N > n$, and define $\mathcal{C}^{\perp}(N, n, \mathcal{R}) = \{\mathbf{B} \in \mathcal{R}^{N \times n} :
\mathbf{A} \cdot \mathbf{B} = 0, \mathbf{A} \in \mathcal{C}(N - n, N, \mathcal{R}), \text{rank}(\mathbf{B}) = n\}$.
For $n = n(\lambda), N = N(\lambda)$ and $\mathcal{R} = \mathcal{R}(\lambda)$, the $(\mathcal{D}, \mathcal{C}, \mathcal{R})\text{-dual-LPN}(N, n)$
assumption states that

$$
\{(\mathbf{H}, \mathbf{b}) \mid \mathbf{H} \leftarrow \mathcal{C}^{\perp}(N, n, \mathcal{R}), \mathbf{e} \leftarrow \mathcal{D}(\mathcal{R}), \mathbf{b} \leftarrow \mathbf{e} \cdot \mathbf{H} \} \\
\approx_c \{(\mathbf{H}, \mathbf{b}) \mid \mathbf{H} \leftarrow \mathcal{C}^{\perp}(N, n, \mathcal{R}), \mathbf{b} \leftarrow \mathcal{R}^n \}
$$

We will slightly abuse our notations by omitting to explicitely men-
tion the code $\mathcal{C}$ and writing $(\mathcal{D}, \mathbf{H}, \mathcal{R})\text{-dual-LPN}(N, n)$ for above
dual-LPN assumption with a matrix $\mathbf{H} \in \mathcal{C}^{\perp}(N, n, \mathcal{R})$.

The search version of the dual LPN problem is also known as
**syndrome decoding**. The decision version defined above is equiva-
lent to primal variant of LPN from Definition 2.2 with dimension
$N - n$ and number of samples $N$. This follows from the simple fact
that $(\mathbf{s} \cdot \mathbf{A} + \mathbf{e}) \cdot \mathbf{H} = \mathbf{s} \cdot \mathbf{A} \cdot \mathbf{H} + \mathbf{e} \cdot \mathbf{H} = \mathbf{e} \cdot \mathbf{H}$, when $\mathbf{H}$ is the parity-check
matrix of $\mathbf{A}$.

**Attacks on LPN.** We recall here the main attacks on LPN, fol-
lowing the analysis of [15]. We refer the reader to [27] for a more
comprehensive overview. We assume that $\mathcal{D}$ is a noise distribution
with Hamming weight bounded by some integer $t$.
*   **Gaussian elimination.** The most natural attack on LPN
    recovers $\mathbf{s}$ from $\mathbf{b} = \mathbf{s} \cdot \mathbf{A} + \mathbf{e}$ by guessing $n$ non-noisy coor-
    dinates of $\mathbf{b}$, and inverting the corresponding subsystem to
    verify whether the guess was correct. This approach recovers
    $\mathbf{s}$ in time at least $(1/(1 - r))^n$ using at least $\mathcal{O}(n/r)$ samples
    ($r = t/N$). For low-noise LPN, with noise rate $1/n^c$ for some
    constant $c \geq 1/2$, this translates to a bound on attacks of
    $\mathcal{O}(e^{n/c})$ time using $\mathcal{O}(n^{1+c})$ samples.
*   **Information Set Decoding (ISD)** [52]. Breaking LPN is
    equivalent to solving its dual variant, which can be inter-
    preted as the task of decoding a random linear code from
    its syndrome. The best algorithms for this task are improve-
    ments of Prange’s ISD algorithm, which attempts to find
    a size-$t$ subset of the rows of $\mathbf{B}$ (the parity-check matrix of the
    code) that spans $\mathbf{e} \cdot \mathbf{B}$, where $t \approx rN$ is the number of noisy
    coordinates. The state of the art variant of Prange’s informa-
    tion set decoding attack is the **BJMM attack** [10], which was
    analyzed in [54], and in the NIST candidate BIKE [6, Section
    5.2], which also take into account the effect of the DOOM
    attack [53] which applies to the specific case of LPN with
    quasi-cyclic codes.
*   **The BKW algorithm** [13]. This algorithm is a variant of
    Gaussian elimination which achieves subexponential com-
    plexity even for high-noise LPN (e.g. constant noise rate),
    but requires a subexponential number of samples: the attack
    solves LPN over $\mathbb{F}_2$ in time $2^{\mathcal{O}(n/\log(n/r))}$ using $2^{\mathcal{O}(n/\log(n/r))}$
    samples.
*   **Combinations of the above** [27]. The authors of [27] con-
    ducted an extended study of the security of LPN, and de-
    scribed combinations and refinements of the previous three
    attacks (called the **well-pooled Gauss attack**, the **hybrid attack**,
    and the **well-pooled MMT attack**). All these attacks achieve
    subexponential time complexity, but require as many sample
    as their time complexity.
*   **Scaled-down BKW** [45]. This algorithm is a variant of the
    BKW algorithm, tailored to LPN with polynomially-many
    samples. It solves LPN in time $2^{\mathcal{O}(n/\log \log(n/r))}$, using $n^{1+\varepsilon}$
    samples (for any constant $\varepsilon > 0$) and has worse performance
    in time and number of samples for larger fields.
*   **Low-Weight Parity Check** [57]. Eventually, all the pre-
    vious attacks recover the secret $\mathbf{s}$. A more efficient attack
    (by a polynomial factor) can be used if one simply wants
    to distinguish $\mathbf{b} = \mathbf{s} \cdot \mathbf{A} + \mathbf{e}$ from random: by the singleton
    bound, the minimal distance of the dual code of $\mathcal{C}$ is at most
    $n + 1$, hence there must be a parity-check equation for $\mathcal{C}$ of
weight $n + 1$. Then, if $\mathbf{b}$ is random, it passes the check with
probability at most $1/|\mathbb{F}|$, whereas if $\mathbf{b}$ is a noisy encoding, it
passes the check with probability at least $((N - n - 1)/N)^{\lambda}$.

## 2.3 Secure Computation and NISC

We use standard definitions of (composable) secure two-party com-
putation. Our protocols can be analyzed and used either in a stan-
dalone setting, as formalized in [19, 31], or in a UC setting [20, 38,
49]. It will be convenient to cast our protocols in a hybrid model
that allows parallel calls to an **ideal oblivious transfer functionality**.
These calls can be instantiated by any composable OT protocol (e.g.,
the “PVW protocol” [49] when considering UC security against
malicious adversaries in the CRS model). We use $\lambda$ to denote a com-
putational security parameter, which we view as a public parameter
that is available to all algorithms even when not explicitly stated.

We will specifically be interested in **2-round protocols** for “**sender-
receiver functionalities**” that take an input $\mathbf{x}$ from a receiver $\mathcal{R}$ and
input $\mathbf{y}$ from a sender $\mathcal{S}$, and deliver an output $f(\mathbf{x}, \mathbf{y})$ to $\mathcal{R}$. The
communication consists of a single message from the receiver to
the sender followed by a single message from the receiver to the
sender. Such protocols can be viewed as being **non-interactive** in
that the receiver can publish its message $\mathbf{x}$ (which depends only
on its input $\mathbf{x}$) and then go offline, before even knowing who the
sender will be. Then $\mathbf{x}$ can be used by any sender $\mathcal{S}$ (in fact, in some
cases even multiple senders) by sending the encrypted output $\mathbf{z}$
to the receiver's mailbox. We use the term **non-interactive secure
computation** from [38] (**NISC** for short) to highlight this qualitative
advantage. When described in the OT-hybrid model, NISC protocols
involve only one round of parallel OT calls. They can additionally
involve a message from $\mathcal{R}$ to $\mathcal{S}$ and a message from $\mathcal{S}$ to $\mathcal{R}$, as long as
these messages (in an honest execution) do not depend on outputs
of the OT oracle. Such NISC protocols in the OT-hybrid model can
be converted into NISC protocols in the plain model (or CRS model
for malicious security) using any 2-round (parallel-)OT protocol.

## 2.4 Pseudorandom Correlation Generators

A (two-party) pseudorandom correlation generator (PCG) securely
generates long correlated pseudo-randomness from a pair of corre-
lated keys. Defining a PCG requires care, since the natural simulation-
based definition is not realizable. Instead, the following relaxed
definition has been proposed in [15, 16].

The ideal output distribution of a PCG is specified by a (long)
**target correlation** $(\mathbf{R}_0, \mathbf{R}_1)$, e.g., $n$ independent instances of an OT
correlation. This target correlation is specified by PPT algorithm
$\mathcal{C}$, called a **correlation generator**, where $\mathcal{C}(1^\lambda)$ outputs a pair of
strings. We furthermore restrict $\mathcal{C}$ to be **reverse-samplable** in the
following sense: there exists a PPT algorithm $\text{RSample}$ such that
for $\sigma \in \{0, 1\}$, the correlation obtained via:

$$
\{(\mathbf{R}_0, \mathbf{R}_1) \mid (\mathbf{R}_0, \mathbf{R}_1) \leftarrow \mathcal{C}(1^\lambda), \mathbf{R}'_{\bar{\sigma}} := \mathbf{R}_{\bar{\sigma}}, \mathbf{R}'_{\sigma} \leftarrow \text{RSample}(\sigma, \mathbf{R}_{\bar{\sigma}})\}
$$

is computationally indistinguishable from $\mathcal{C}(1^\lambda)$.

Examples for standard and useful correlations, all of which are
reverse-samplable, include **Oblivious Transfer (OT) correlation**,
where $\mathbf{R}_0$ includes $n$ independent pairs of bit-strings $(\mathbf{s}_0^i, \mathbf{s}_1^i)$ and $\mathbf{R}_1$
includes $(\mathbf{c}_i, \mathbf{s}_{\mathbf{c}_i}^i)$ for random bits $\mathbf{c}_i$, and **Vector-OLE (VOLE) correla-
tion** over a finite field $\mathbb{F}$, where $\mathbf{R}_0 = (\mathbf{u}, \mathbf{v})$ for random $\mathbf{u}, \mathbf{v} \in \mathbb{F}_p^n$,
and $\mathbf{R}_1 = (\mathbf{x}, \mathbf{u} \mathbf{x} + \mathbf{v})$ for random $\mathbf{x} \in \mathbb{F}$.

**Definition 2.4 (Pseudorandom Correlation Generator (PCG) [16]).**
Let $\mathcal{C}$ be a reverse-samplable correlation generator. A pseudorandom
correlation generator (PCG) for $\mathcal{C}$ is a pair of algorithms ($\text{PCG}.\text{Gen},
\text{PCG}.\text{Expand}$) with the following syntax:
*   $\text{PCG}.\text{Gen}(1^\lambda)$ is a PPT algorithm that given a security pa-
    rameter $\lambda$, outputs a pair of seeds $(\mathbf{k}_0, \mathbf{k}_1)$;
*   $\text{PCG}.\text{Expand}(\sigma, \mathbf{k}_\sigma)$ is a polynomial-time algorithm that given
    party index $\sigma \in \{0, 1\}$ and a seed $\mathbf{k}_\sigma$, outputs a bit string
    $\mathbf{R}_\sigma \in \{0, 1\}^n$.
The algorithms ($\text{PCG}.\text{Gen}, \text{PCG}.\text{Expand}$) should satisfy:
*   **Correctness.** The correlation obtained via:

$$
\{(\mathbf{R}_0, \mathbf{R}_1) \mid (\mathbf{k}_0, \mathbf{k}_1) \leftarrow \text{PCG}.\text{Gen}(1^\lambda), \mathbf{R}_\sigma \leftarrow \text{PCG}.\text{Expand}(\sigma, \mathbf{k}_\sigma)\}
$$

is computationally indistinguishable from $\mathcal{C}(1^\lambda)$.
*   **Security.** For corrupted party $\sigma \in \{0, 1\}$, the following two
    distributions are computationally indistinguishable:

$$
\{(\mathbf{k}_0, \mathbf{R}_{\bar{\sigma}}) \mid (\mathbf{k}_0, \mathbf{k}_1) \leftarrow \text{PCG}.\text{Gen}(1^\lambda), \mathbf{R}_{\bar{\sigma}} \leftarrow \text{PCG}.\text{Expand}(\bar{\sigma}, \mathbf{k}_{\bar{\sigma}})\} \text{ and } \\
\{(\mathbf{k}_0, \mathbf{R}_{\bar{\sigma}}) \mid (\mathbf{k}_0, \mathbf{k}_1) \leftarrow \text{PCG}.\text{Gen}(1^\lambda), \mathbf{R}_{\sigma} \leftarrow \text{PCG}.\text{Expand}(\sigma, \mathbf{k}_\sigma), \\
\mathbf{R}'_{\bar{\sigma}} \leftarrow \text{RSample}(\sigma, \mathbf{R}_\sigma) \}
$$

where $\bar{\sigma} = 1 - \sigma$ and $\text{RSample}$ is the reverse sampling algo-
rithm for $\mathcal{C}$.

As shown in [16], a PCG as defined above can be used as a ‘drop-
in replacement’ for ideal correlated randomness generated by $\mathcal{C}$
in any application that remains secure even when $\mathcal{C}$ is replaced
by the following **corruptible version $\tilde{\mathcal{C}}$**. In $\tilde{\mathcal{C}}$ the corrupted party
can choose its own randomness, and the randomness of the honest
party $\mathbf{R}_{1-\sigma}$ is obtained by applying $\text{RSample}$. It turns out that in
most concretely efficient MPC protocols that consume correlated
randomness, security still holds even with this corruptible variant.
In particular, this holds for the simple protocols that implement
standard (chosen-input) OT or VOLE from the corresponding cor-
relations. However, applying PCGs, the pair of keys $(\mathbf{k}_0, \mathbf{k}_1)$ to be
generated either by a trusted dealer or by a secure protocol realizing
$\text{PCG}.\text{Gen}$.

# 3 PCG PROTOCOLS AND SILENT NISC

We now define two new cryptographic primitives we introduce
in this work: A **pseudorandom correlation generation protocol (PCG
protocol** for short) and a **non-interactive secure computation protocol
with silent preprocessing (silent NISC** for short).

## 3.1 PCG Protocols

The above notion of PCG gives a deterministic procedure for se-
curely generating long sources of correlated randomness from short
but suitably correlated seeds. It does not explicitly address the ques-
tion of generating the seeds. In the following we formalize a nat-
ural generalization of PCGs to a low-communication protocol for
securely generating long sources of correlated randomnness from
scratch. By “**low communication**” we means that the total commu-
nication complexity is sublinear in the output length.³

We take the following natural definition approach: a PCG proto-
col for an ideal correlation $\mathcal{C}$ is a **secure two-party protocol** (in the
usual sense) for the corruptible correlated randomness functionality
$\tilde{\mathcal{C}}$ described below.

**Definition 3.1 (PCG protocol).** Let $\mathcal{C}$ be a reverse-samplable cor-
relation generator. Define a randomized functionality $\tilde{\mathcal{C}}$ that takes
from a corrupted party $\sigma$ a string $\mathbf{r}_\sigma$ as input, and outputs to the
honest party $\bar{\sigma}$ a string $\mathbf{r}_{\bar{\sigma}}$ sampled by $\text{RSample}(\sigma, \mathbf{r}_\sigma)$. If no party
is corrupted, $\tilde{\mathcal{C}}$ outputs to both parties a fresh pair of outputs gen-
erated by $\mathcal{C}$. A (two-party) PCG protocol is a two-party protocol
realizing $\tilde{\mathcal{C}}$ in which the communication complexity grows sub-
linearly with the output length. In the case of security against
semi-honest adversaries, we still allow the ideal-model corrupted
party (if any) to pick its input $\mathbf{r}_\sigma$ for $\tilde{\mathcal{C}}$ arbitrarily, whereas the
real-model adversary must follow the protocol.

As a simple corollary of an MPC composition theorem, a PCG
protocol for $\tilde{\mathcal{C}}$ can serve as a substitute for ideal correlated random-
ness $\mathcal{C}$ in any higher-level application that remains secure even
when $\mathcal{C}$ is replaced by $\tilde{\mathcal{C}}$. Indeed, this is the case for standard MPC
protocols that rely on OT correlations or other types of simple
correlations, both for semi-honest and malicious security.

A general way of obtaining a PCG protocol is by distributing the
randomized key generation functionality $\text{PCG}.\text{Gen}$ of a PCG (as in
Definition 2.4) via a secure two-party computation protocol, and
then locally applying $\text{PCG}.\text{Expand}$. Indeed, this is the approach
suggested in [16] for the purpose of applying PCGs in the context
of “MPC with silent preprocessing.” However, our notion of a PCG
protocol is less stringent than an alternative definition that requires
securely emulating $\text{PCG}.\text{Gen}$ for some PCG, while at the same time
being as good for applications. We make use of this extra degree
of freedom in our PCG protocols for the malicious model.

A central contribution of this work is the construction of **two-
round PCG protocols**, namely ones involving only a message from $\mathcal{R}$
to $\mathcal{S}$ followed by a message from $\mathcal{S}$ to $\mathcal{R}$. We refer to such a protocol
as a **non-interactive PCG protocol**. We use the following syntax
to highlight the fact that the message of $\mathcal{R}$ can be published as a
“**public key**” before the sender(s) are known.

**Definition 3.2 (Non-interactive PCG protocol).** A non-interactive
PCG protocol is defined by 4 algorithms with the following syntax:
*   $\mathcal{R}.\text{Gen}(1^\lambda) \rightarrow (\text{sk}_\mathcal{R}, \text{pk}_\mathcal{R})$
*   $\mathcal{S}.\text{Gen}(\text{pk}_\mathcal{R}) \rightarrow (\text{sk}_\mathcal{S}, m_\mathcal{S})$
*   $\mathcal{R}.\text{Expand}(\text{sk}_\mathcal{R}, m_\mathcal{S}) \rightarrow \mathbf{r}_\mathcal{R}$
*   $\mathcal{S}.\text{Expand}(\text{sk}_\mathcal{S}) \rightarrow \mathbf{r}_\mathcal{S}$
We say that the above algorithms define a **non-interactive PCG
protocol** for a reverse-samplable correlation $\mathcal{C}$ if the two-round
protocol they naturally define (where each party outputs the output
of $\text{Expand}$) is a PCG protocol for $\mathcal{C}$ as in Definition 3.1.

In a non-interactive PCG protocol as above, the two $\text{Gen}$ algo-
rithms can be viewed as defining a cheap setup that results in short,
correlated keys. The two $\text{Expand}$ algorithms are used to locally
perform “**silent preprocessing**” that generates useful correlated ran-
domness (e.g., many instances of an OT correlation, or few instances
of a long VOLE correlation). In the most useful special case of OT
correlations, we will refer to a non-interactive PCG that makes a
small number of parallel OT calls as a **non-interactive (or 2-round)
silent OT extension protocol**.

## 3.2 Silent NISC

In this section we define our new notion of **non-interactive secure
computation with silent preprocessing**, or **silent NISC** for short. A
silent NISC protocol for $f$ can be viewed as a “best-of-both-worlds”
combination of a non-interactive PCG protocol (see Definition 3.2)
and a NISC protocol (see Section 2.3). That is, a 2-round (chosen-
input) secure computation protocol that supports “**silent prepro-
cessing**” followed by a light-weight (and often “non-cryptographic")
online phase, without additional interaction.

Combining non-interactive PCG and NISC protocols in a generic
way does not achieve the above goal, since it involves 4 rounds:
two to generate the correlated randomness, and two to use it. To
collapse these 4 rounds into two, we rely on the following feature
of our concrete non-interactive PCG constructions. For useful NISC
correlations such as OT and VOLE, the receiver’s piece of the corre-
lated randomness $\mathbf{r}_\mathcal{R}$ can be split into two parts: $\mathbf{r}_{\text{in}}$, which is used
to mask its input, and $\mathbf{r}_{\text{out}}$, used to unmask the output. The key
feature is that the construction allows $\mathcal{R}$ to locally generate $\mathbf{r}_{\text{in}}$ from
its public key $\text{pk}_\mathcal{R}$ alone, independently of the sender. This enables
$\mathcal{R}$ to prepare to a future NISC before the sender is even known.

More concretely, let $f(\mathbf{x}, \mathbf{y})$ be a sender-receiver functionality
with receiver input $\mathbf{x}$ and sender input $\mathbf{y}$. Useful examples for which
we get efficient solutions include: (1) $n$ instances of string-OT; (2)
bitwise-AND of two $n$-bit strings; (3) inner product of two length-$n$
vectors over $\mathbb{F}$; (4) a general function $f$ represented by a Boolean
circuit, which can be efficiently and non-interactively reduced to
(1) via garbled circuits (see [1, 38, 46] for such black-box reductions
for the malicious model).

A **NISC protocol with silent preprocessing** (or **silent NISC**) for $f$ is
defined by 8 algorithms:
*   $\mathcal{R}.\text{Gen}(1^\lambda) \rightarrow (\text{sk}_\mathcal{R}, \text{pk}_\mathcal{R})$
*   $\mathcal{R}.\text{Expand}_{\text{in}}(\text{sk}_\mathcal{R}) \rightarrow \mathbf{r}_{\text{in}}$
*   $\mathcal{S}.\text{Gen}(\text{pk}_\mathcal{R}) \rightarrow (\text{sk}_\mathcal{S}, \text{pk}_\mathcal{S})$
*   $\mathcal{R}.\text{Expand}_{\text{out}}(\text{sk}_\mathcal{R}, \text{pk}_\mathcal{S}) \rightarrow \mathbf{r}_{\text{out}}$
*   $\mathcal{S}.\text{Expand}(\text{sk}_\mathcal{S}) \rightarrow \mathbf{r}_\mathcal{S}$
*   $\mathcal{R}.\text{Msg}(\mathbf{r}_{\text{in}}, \mathbf{x}) \rightarrow \hat{\mathbf{x}}$
*   $\mathcal{S}.\text{Msg}(\mathbf{r}_\mathcal{S}, \hat{\mathbf{x}}, \mathbf{y}) \rightarrow \hat{\mathbf{z}}$
*   $\mathcal{R}.\text{Dec}(\mathbf{r}_{\text{out}}, \hat{\mathbf{x}}, \hat{\mathbf{z}}) \rightarrow \mathbf{z}$
The security requirement is that the 2-round protocol obtained by
executing the above algorithms in any consistent order satisfies the
security requirement of a (standard) NISC protocol for $f$.

To clarify the intended use and the features of our model for non-
interactive secure computation protocols with silent preprocessing,
we provide on Figure 2 a pictural representation of the protocol flow,
illustrating the interdependencies between the algorithms, and we
identify the main features of each of the algorithms (whether they
require small communication, or only silent computation; whether
they require cryptographic or non-cryptographic computation).

[IMAGE: Figure 2: Pictural representation of the protocol flow for non-interactive secure computation with silent preprocessing. The receiver input is $y$, the sender input is $x$, and the target output is $z = f(x, y)$. The diagram shows Small Communication (pkR, pkS are short), Silent Computation (R.Expand_in, R.Expand_out, S.Expand), and Non-Cryptographic Computation (R.Msg, S.Msg, R.Dec). The flow involves R.Gen, R.Expand_in, S.Gen, R.Expand_out, S.Expand, R.Msg, S.Msg, and R.Dec, with various inputs and outputs.]

The 3 $\text{Expand}$ algorithms define the "**silent preprocessing**” phase,
that can be executed before the inputs are known. The last 3 algo-
rithms define the **online part** of the NISC protocol, which is carried
out once the inputs are known. Among the four examples given
above, this part is “non-cryptographic” in the first three cases, and
makes a black-box use of symmetric crypto in the last one.

We will be particularly interested in silent NISC realizing many
parallel OTs using few parallel OTs, which can be viewed as a non-
interactive, chosen-input variant of **silent OT extension**. While here
one cannot make the communication complexity sublinear in the
input length, our goal (which we achieve both in theorem and in
practice) to make the communication very close to the total input
length. This is the case even for the more challenging case of 1-bit
OT, for which standard OT extension techniques that make a black-
box use of cryptography [8, 37, 41, 44] have a high communication
overhead compared to the input length.

# 4 OPTIMIZED VOLE/OT CONSTRUCTION

## 4.1 Simplified subfield VOLE generator

We provide a construction of a PCG for subfield-VOLE correlations
on Figure 3. Recall that in **subfield-VOLE**, one party receives random
vectors $\mathbf{u} \in \mathbb{F}_{p^r}^n$ and $\mathbf{v} \in \mathbb{F}_{p^r}^{N_r}$, while the other party gets a random
$\mathbf{x} \in \mathbb{F}_{p^r}$, and $\mathbf{w} = \mathbf{u} \mathbf{x} + \mathbf{v}$. The construction follows the informal
description from Section 1.2 (where we described the special case
$p = 2$, which is equivalent to correlated OT), and is essentially the
same as the construction in [16], with a puncturable PRF instead of
a DPF. Likewise, the security analysis is essentially identical to the
analysis of [16].

**Construction $\mathcal{G}_{sVOLE}$**
**PARAMETERS:** $1^\lambda, n, N, t, p, r \in \mathbb{N}$, where $N > n$. A matrix $\mathbf{H} \in
\mathbb{F}_p^{N \times n}$ and a weight-$t$ error distribution $\mathcal{D}_{t, N}$ over $\mathbb{F}_p^N$.
**CORRELATION:** After expansion, outputs $(\mathbf{u}, \mathbf{v}) \in \mathbb{F}_{p^r}^{n} \times \mathbb{F}_{p^r}^{N_r}$, and
$(\mathbf{x}, \mathbf{w}) \in \mathbb{F}_{p^r} \times \mathbb{F}_{p^r}^n$, where $\mathbf{w} = \mathbf{u} \mathbf{x} + \mathbf{v}$.
We view $\mathbb{F}_p$ as a subfield of $\mathbb{F}_{p^r}$, via some fixed embedding and
representation of field elements.
$\text{PPRF}$ is a puncturable $\text{PRF}$ with domain $[N]$ and range $\mathbb{F}_{p^r}$.
**Gen:** On input $1^\lambda$:
(1) Sample $\mathbf{e} \leftarrow \mathcal{D}_{t, N}$. Let $\mathcal{S} = \{\mathbf{a}_1, \ldots, \mathbf{a}_t\} \in [N]^t$ be the
sorted indices of non-zero entries in $\mathbf{e}$, and $y_i = \mathbf{e}_{\mathbf{a}_i} \in \mathbb{F}_p$.
(2) Sample $\mathbf{x} \in \mathbb{F}_{p^r}$.
(3) Sample $\mathbf{k}_{\text{pprf}} \leftarrow \text{PPRF}.\text{Gen}(1^\lambda)$, and $\mathbf{k}^*_{\text{pprf}} \leftarrow
\text{PPRF}.\text{Puncture}(\mathbf{k}_{\text{pprf}}, \mathcal{S})$.
(4) For $i = 1, \ldots, t$, let $\mathbf{z}_i \leftarrow \mathbf{x} \cdot y_i - \text{PPRF}.\text{Eval}(\mathbf{k}_{\text{pprf}}, \mathbf{a}_i)$
(5) Let $\mathbf{k}_0 \leftarrow (\mathbf{k}^*_{\text{pprf}}, \mathcal{S}, \mathbf{y}, \{\mathbf{z}_i\}_{i \in [t]})$ and $\mathbf{k}_1 \leftarrow (\mathbf{k}_{\text{pprf}}, \mathbf{x})$.
(6) Output $(\mathbf{k}_0, \mathbf{k}_1)$.
**Expand:** On input $(\sigma, \mathbf{k}_\sigma)$:
(1) If $\sigma = 0$, parse $\mathbf{k}_0$ as $(\mathbf{k}^*_{\text{pprf}}, \mathcal{S}, \mathbf{y}, \{\mathbf{z}_i\}_{i})$ and do as follows:
(a) Define $\mathbf{e} \in \mathbb{F}_{p^r}^N$ using $\mathbf{y}, \{\mathbf{z}_i\}_{i}$ as above.
(b) For $j \in [N]$, define the $j$-th entry of vector $\mathbf{z}_0$ as
$$ \mathbf{z}_{0}^{[j]} = \begin{cases} \mathbf{z}_i & \text{if } j = \mathbf{a}_i \in \mathcal{S} \\ -\text{PPRF}.\text{Eval}(\mathbf{k}^*_{\text{pprf}}, j) & \text{if } j \notin \mathcal{S} \end{cases} $$
(c) Output $(\mathbf{u}, \mathbf{v}) \leftarrow (\mathbf{e} \cdot \mathbf{H}, -\mathbf{z}_0 \cdot \mathbf{H})$.
(2) If $\sigma = 1$, parse $\mathbf{k}_1$ as $(\mathbf{k}_{\text{pprf}}, \mathbf{x})$ and do as follows:
(a) Compute $\mathbf{z}_1 \leftarrow \text{PPRF}.\text{FullEval}(\mathbf{k}_{\text{pprf}})$ in $\mathbb{F}_{p^r}^N$.
(b) Output $(\mathbf{x}, \mathbf{w} \leftarrow \mathbf{z}_1 \cdot \mathbf{H})$.

**Figure 3: PPRF-based PCG for subfield vector-OLE**

In our two-round protocols, we actually obtain a slightly different
variant, called **reverse VOLE** [4], where the sender inputs $\mathbf{x}, \mathbf{w}$, while
the receiver inputs $\mathbf{u}$ and learns $\mathbf{v} = \mathbf{w} - \mathbf{u} \mathbf{x}$.

## 4.2 Instantiating the puncturable PRF

We use a simple puncturable PRF based on the GGM approach [32]
(as defined in Section 2). To build a PPRF supporting $t$ punctured
points, we simply create $t$ independent GGM PRFs, each punctured
once. Evaluation of the final PPRF is defined by adding the evalua-
tions of all $t$ GGM-based PRFs.

**More Efficient Puncturing Strategy.** The key size for the above
$t$-puncturable PRF is $t \lambda \log N$. It is possible to reduce this size to
$t \cdot \lambda \log (N/t)$ with a more optimized puncturing strategy; however,
this alternative construction is not compatible with our optimized
distributed generation protocols of Section 5 and Section 6. It is
nonetheless useful in a setting where a trusted dealer is available to
distribute the PCG seeds, or where computation is not a bottleneck
compared to long-term storage. For a formal treatment we refer to
the full version.

# 5 SEMI-HONEST PCG PROTOCOL AND TWO-ROUND OT EXTENSION

In this section, we show how to securely compute the $\text{Gen}$ algo-
rithm from Figure 3, in just 2 rounds (assuming any 2-round OT).
Using the construction of [16], this leads to a distributed protocol
for generating random OT correlations as well, assuming in ad-
dition a correlation-robust hash function. Then, we observe that
our protocols satisfy a specific feature, which allows them to be
derandomized into chosen-input VOLEs and OTs, without increas-
ing their round complexity; this leads to 2-round OT extension and
VOLE extension protocols, with silent preprocessing. Our construc-
tion relies on the GGM puncturable PRF [32] constructed from any
length-doubling pseudorandom generator $\mathbf{G}$.

**On VOLE and reverse VOLE.** In a typical (chosen-input) VOLE, the
sender inputs $(\mathbf{u}, \mathbf{v})$, while the receiver inputs $\mathbf{x}$ and gets $\mathbf{w} = \mathbf{u} \mathbf{x} + \mathbf{v}$.

## 5.1 Distributed GGM-PPRF Correlation

We first consider a functionality where a party $\mathcal{R}$ holds a PPRF
key $\mathbf{k}_{\text{pprf}} \in \{0, 1\}^\lambda$ for the GGM PPRF [32], and a point $\mathbf{a} =
\alpha_1 \cdots \alpha_\ell$ where $\ell = \ell(\lambda)$ is logarithmic in $\lambda$, and a party $\mathcal{S}$ holds a
value $\beta \in \{0, 1\}^\lambda$. The functionality computes and gives $\mathbf{k}_{\{\mathbf{a}\}}, \mathbf{t} = \beta -
\text{PPRF}.\text{Eval}(\mathbf{k}, \mathbf{a})$ to $\mathcal{R}$. The functionality is represented on Figure 4.

**Functionality $\mathcal{F}_{\text{PPRF-GGM}}$:**
**PARAMETERS:** $1^\lambda, \ell, p, r \in \mathbb{N}$. $\text{PPRF}$ is a puncturable $\text{PRF}$ with
domain $\{0, 1\}^\ell$, key space $\{0, 1\}^\lambda$, and range $\mathbb{F}_{p^r}$.
**INPUTS:**
*   $\mathcal{S}$ inputs $\beta \in \mathbb{F}_{p^r}$ and a $\text{PPRF}$ key $\mathbf{k}_{\text{pprf}} \in \{0, 1\}^\lambda$.
*   $\mathcal{R}$ inputs $\mathbf{a} \in \{0, 1\}^\ell$.
**FUNCTIONALITY:**
*   Compute $\mathbf{k}^*_{\text{pprf}} = \text{PPRF}.\text{Puncture}(\mathbf{k}_{\text{pprf}}, \mathbf{a})$.
*   Send $\mathbf{k}^*_{\text{pprf}}$ and $\mathbf{t} = \beta - \text{PPRF}.\text{Eval}(\mathbf{k}_{\text{pprf}}, \mathbf{a})$ to $\mathcal{R}$.

**Figure 4: Functionality for distributing a PPRF correlation**

**THEOREM 5.1.** *Assuming a black-box access to a PRG, there exists a
2-party protocol for $\Pi_{\text{PPRF-GGM}}$, with semi-honest security in the OT-
hybrid model, and the following efficiency features. The computational
complexity is dominated by $\mathcal{O}(2^\ell)$ calls to a length-doubling PRG
$\mathbf{G}: \{0, 1\}^\lambda \leftrightarrow \{0, 1\}^{2\lambda}$. The interaction consists of $\ell$ parallel calls to
$\mathcal{F}_{\text{OT}}$ and has communication complexity $\lambda + (3\lambda + 1)\ell$.*

**Protocol $\Pi_{\text{PPRF-GGM}}$:**
**PARAMETERS:** $1^\lambda, \ell, p, r \in \mathbb{N}$. $\text{PPRF}_{\text{GGM}}$ is the GGM puncturable
$\text{PRF}$ with domain $\{0, 1\}^\ell$, key space $\{0, 1\}^\lambda$, and range $\mathbb{F}_{p^r}$,
constructed from a length-doubling $\text{PRG}$ $\mathbf{G}: \{0, 1\}^\lambda \leftrightarrow \{0, 1\}^{2\lambda}$,
and a second $\text{PRG}$ $\mathbf{G}': \{0, 1\}^\lambda \leftrightarrow (\mathbb{F}_{p^r})^2 \times \{0, 1\}^\lambda$ used to compute the
$\text{PRF}$ outputs on the last level of the tree.
**INPUTS:**
*   $\mathcal{R}$ inputs $\mathbf{a} \in \{0, 1\}^\ell$.
*   $\mathcal{S}$ inputs $\beta \in \mathbb{F}_{p^r}$ and a $\text{PPRF}$ key $\mathbf{k}_{\text{pprf}} \in \{0, 1\}^\lambda$.
**PROTOCOL:**
(1) $\mathcal{R}$ and $\mathcal{S}$ execute in parallel $\ell$ calls to $\mathcal{F}_{\text{OT}}$, where for $i = 1$
to $\ell - 1$:
*   $\mathcal{R}$ uses as input the choice bit $\alpha_i$;
*   $\mathcal{S}$ computes the $2^{i-1}$ partial evaluations at level $i$ of the
    GGM tree defined by $\mathbf{k}$, denoted $\mathbf{s}_0^{i}, \ldots, \mathbf{s}_{2^{i-1}-1}^{i}$ (in left-
    to-right order) and uses the two $\text{OT}$ inputs
    $\mathbf{t}_0^i = \sum_{j \in [0, 2^{i-1})} \mathbf{t}_{j, \mathbf{k}}^i \mathbf{t}_1^i = \sum_{j \in [0, 2^{i-1})} \mathbf{t}_{j, \mathbf{k}}^{i} \mathbf{s}_{2j+1}^{i}$
    and for the last $\text{OT}$,
*   $\mathcal{R}$ uses as input the choice bit $\alpha_\ell$;
*   $\mathcal{S}$ computes the $2^{\ell-1}$ evaluations of the GGM tree defined
    by $\mathbf{k}$, denoted $\mathbf{s}_0^\ell, \ldots, \mathbf{s}_{2^{\ell-1}-1}^\ell \in (\mathbb{F}_{p^r})^2 \times \{0, 1\}^\lambda$ (in left-to-right
    order) and uses the two $\text{OT}$ inputs
    $\mathbf{t}_0^\ell = \sum_{j=0}^{2^{\ell-1}-1} \mathbf{t}_{j, \mathbf{k}}^{\ell} \mathbf{s}_{2j}^{\ell} \mathbf{t}_1^\ell = \sum_{j=0}^{2^{\ell-1}-1} \mathbf{t}_{j, \mathbf{k}}^{\ell} \mathbf{s}_{2j+1}^{\ell}$
(2) In parallel to the $\text{OT}$ calls, $\mathcal{S}$ sends $\mathbf{c} = \beta - (\mathbf{t}_0^\ell + \mathbf{t}_1^\ell)$ to $\mathcal{R}$.
**OUTPUT:** $\mathcal{R}$ computes its output as follows:
(1) Let $\mathbf{t}^1$ be $\mathcal{R}$'s output in the first $\text{OT}$. Define $\mathbf{s}^1 = \mathbf{t}^1$.
(2) For $i = 2, \ldots, \ell - 1$:
(a) Compute $(\mathbf{s}_{2j}^{i}, \mathbf{s}_{2j+1}^{i}) = \mathbf{G}(\mathbf{s}^{i-1}_{j})$, for $j \in$
$[0, \ldots, 2^{i-1}), j \neq \alpha_1 \cdots \alpha_{i-1}$.
(b) Let $\mathbf{t}^i$ be the output from the $i$-th $\text{OT}$.
(c) Define $\mathbf{a}' = \alpha_1 \cdots \alpha_{i-1} \alpha_{i}$. Compute
$$ \mathbf{s}_{\mathbf{a}'}^{i} = \mathbf{t}^i \oplus \bigoplus_{j \in [0, 2^{i-1}), j \neq \alpha_1 \cdots \alpha_{i-1}} \mathbf{s}_{2j+\alpha_i}^{i}, $$
for $j \in [0, \ldots, 2^{i-1}), j \neq \alpha_1 \cdots \alpha_{i-1}$.
(3) Compute $(\mathbf{s}_{2j}^{\ell}, \mathbf{s}_{2j+1}^{\ell}) = \mathbf{G}'(\mathbf{s}_j^{\ell-1})$: for $j \in [0, \ldots, 2^{\ell-1}-1), j \neq
\alpha_1 \cdots \alpha_{\ell-1}$.
(4) $\mathcal{R}$ receives $\mathbf{c}$, and computes
$$ \mathbf{t} = \mathbf{c} + \mathbf{t}^\ell + \sum_{j=0, j \neq \mathbf{a}}^{2^\ell-1} \mathbf{s}_{2j+\alpha_\ell} $$
(5) $\mathcal{R}$ outputs the punctured key $\{\mathbf{s}_j^i\}_{i \in [\ell], j \neq \mathbf{a}'}$ and the final
correction value $\mathbf{t}$.

**Figure 5: Protocol $\Pi_{\text{PPRF-GGM}}$ for distributing a GGM-based
PPRF correlation with semi-honest security in the $\mathcal{F}_{\text{OT}}$-
hybrid model**

Then, Sim simulates the OT sender using input $(\mathbf{t}^1, \mathbf{d}_i)$ as input if
$\alpha_i = 0$, and $(\mathbf{d}_i, \mathbf{t}^1)$ as input otherwise, where $\mathbf{d}_i$ is an arbitrary
dummy value; Sim also sends $\mathbf{c}'$ in parallel to the OTs. The indistin-
guishability of the simulation follows directly from the definition
of $\mathcal{F}_{\text{OT}}$ and by construction of $\mathbf{c}'$.

## 5.2 Semi-Honest Non-Interactive PCG Protocol for Subfield-VOLE Correlations

We now explain how to implement a semi-honest public-key PCG
for the subfield-VOLE correlation in the $\mathcal{F}_{\text{PPRF-GGM}}$-hybrid model,
by describing a 2-message 2-party semi-honest protocol to distribu-
tively execute the procedure $\mathcal{G}_{s\text{VOLE}}.\text{Gen}$. The functionality $\mathcal{F}_{\text{Gen}}$ is
represented on Figure 6. When $p > 2$, the implementation requires
in addition a single (subfield-) reverse vector-OLE on vectors of
length $t$. Reverse vector-OLE can be implemented in two rounds
under an appropriate variant of LPN [4] or using linearly homo-
morphic encryption. We represent the functionality $\mathcal{F}_{\text{rev-VOLE}}$ on
Figure 7. Note that in a **reverse vector-OLE protocol**, the sender is
the one holding the input $\mathbf{x}$ (while in a standard vector-OLE, $\mathbf{x}$ is
held by the receiver).

**Functionality $\mathcal{F}_{s\text{VOLE-Gen}}$:**
**PARAMETERS:** $1^\lambda, N, t, p, r \in \mathbb{N}$. $\text{PPRF}$ is a puncturable $\text{PRF}$ with
domain $[N]$, key space $\{0, 1\}^\lambda$, and range $\mathbb{F}_{p^r}$.
**INPUTS:**
*   $\mathcal{R}$ inputs a weight-$t$ vector $\mathbf{e} \in \mathbb{F}_p^N$. Let $\mathcal{S} =
    \{\mathbf{a}_1, \ldots, \mathbf{a}_t\} \in [N]^t$ be the sorted indices of non-zero
    entries in $\mathbf{e}$, and $y_i = \mathbf{e}_{\mathbf{a}_i} \in \mathbb{F}_p$.
*   $\mathcal{S}$ inputs $\mathbf{x} \in \mathbb{F}_{p^r}$ and a $\text{PPRF}$ key $\mathbf{k}_{\text{pprf}} \in \{0, 1\}^\lambda$.
**FUNCTIONALITY:**
(1) Compute $\mathbf{k}^*_{\text{pprf}} \leftarrow \text{PPRF}.\text{Puncture}(\mathbf{k}_{\text{pprf}}, \mathcal{S})$.
(2) For $i = 1, \ldots, t$, let $\mathbf{z}_i \leftarrow \mathbf{x} \cdot y_i - \text{PPRF}.\text{Eval}(\mathbf{k}_{\text{pprf}}, \mathbf{a}_i)$
(3) Let $\mathbf{k}_0 \leftarrow (\mathbf{k}^*_{\text{pprf}}, \mathcal{S}, \mathbf{y}, \{\mathbf{z}_i\}_{i \in [t]})$ and $\mathbf{k}_1 \leftarrow (\mathbf{k}_{\text{pprf}}, \mathbf{x})$.
(4) Output $\mathbf{k}_0$ to $\mathcal{R}$ and $\mathbf{k}_1$ to $\mathcal{S}$.

**Figure 6: Functionality for the Generation Procedure of the Subfield-VOLE Generator**

**Functionality $\mathcal{F}_{\text{rev-VOLE}}$:**
**PARAMETERS:** $t, p, r \in \mathbb{N}$.
**INPUT:** The sender $\mathcal{S}$ inputs a pair $(\mathbf{b}, \mathbf{x}) \in \mathbb{F}_{p^r}^{t} \times \mathbb{F}_{p^r}$. The receiver
$\mathcal{R}$ inputs a vector $\mathbf{y} \in \mathbb{F}_p^t$.
**FUNCTIONALITY:** Compute $\hat{\mathbf{y}} \leftarrow \mathbf{y} \mathbf{x} - \mathbf{b}$ and output $\hat{\mathbf{y}}$ to $\mathcal{R}$.

**Figure 7: Reverse Vector-OLE Functionality over a Field $\mathbb{F}_p$**

We present the protocol $\Pi_{s\text{VOLE-Gen}}$ in Figure 8. Correctness
follows easily by inspection: for $i = 1$ to $t$, we have $\mathbf{z}_i = w_i +
c_i = (\mathbf{b}_i - \text{PPRF}.\text{Eval}(\mathbf{k}_{\text{pprf}}, \mathbf{a}_i)) + c_i = \mathbf{x} \cdot y_i - \text{PPRF}.\text{Eval}(\mathbf{k}_{\text{pprf}}, \mathbf{a}_i)$.
Security is straightforward. We note that when $p = 2$, since $\mathbf{y}$ is
a weight-$t$ vector, it always hold that $y_i = 1$, hence computing a
share of $\mathbf{x} y_i = \mathbf{x}$ is trivial and does not require a call to the VOLE
functionality.

**Protocol $\Pi_{s\text{VOLE-Gen}}$:**
**PARAMETERS:** $1^\lambda, N = 2^\ell, t, p, r \in \mathbb{N}$. $\text{PPRF}$ is a puncturable $\text{PRF}$
with domain $[N]$ and range $\mathbb{F}_{p^r}$.
**INPUTS:**
*   $\mathcal{R}$ inputs a weight-$t$ vector $\mathbf{e} \in \mathbb{F}_p^N$. Let $\mathcal{S} =
    \{\mathbf{a}_1, \ldots, \mathbf{a}_t\} \in [N]^t$ be the sorted indices of non-zero
    entries in $\mathbf{e}$, and $y_i = \mathbf{e}_{\mathbf{a}_i} \in \mathbb{F}_p$.
*   $\mathcal{S}$ inputs $\mathbf{x} \in \mathbb{F}_{p^r}$ and a $\text{PPRF}$ key $\mathbf{k}_{\text{pprf}} \in \{0, 1\}^\lambda$.
**PROTOCOL:**
(1) $\mathcal{S}$ picks $\beta \in \mathbb{F}_{p^r}^t$ and $\mathbf{x} \in \mathbb{F}_{p^r}$.
(2) $\mathcal{R}$ and $\mathcal{S}$ call $\mathcal{F}_{\text{rev-VOLE}}(t, p)$ on respective inputs $\mathbf{y}$ and
$((\beta, \mathbf{x}), (\mathbf{b}, \mathbf{x}))$. $\mathcal{R}$ receives an output $\hat{\mathbf{y}}$.
(3) For $i = 1$ to $t$, $\mathcal{R}$ and $\mathcal{S}$ call $\mathcal{F}_{\text{PPRF-GGM}}(1^\lambda, \ell, p, r)$ on
respective inputs $\mathbf{a}_i$ and $(\mathbf{b}_i, \mathbf{k}_{\text{pprf}})$. $\mathcal{R}$ receives $\mathbf{k}^*$ and $(\mathbf{w}_i, \mathbf{t}_i)$.
(4) For $i = 1$ to $t$, $\mathcal{R}$ computes $\mathbf{z}_i \leftarrow \mathbf{w}_i + \mathbf{t}_i$. $\mathcal{R}$ outputs
$(\mathbf{k}^*_{\text{pprf}}, \mathcal{S}, \mathbf{y}, \{\mathbf{z}_i\}_{i \in [t]})$ and $\mathcal{S}$ outputs $(\mathbf{k}_{\text{pprf}}, \mathbf{x})$.

**Figure 8: Protocol for the Generation Procedure of the Subfield-VOLE Generator**

Implementing $\mathcal{F}_{\text{PPRF-GGM}}$ with the protocol $\Pi_{\text{PPRF-GGM}}$ and
$\mathcal{F}_{\text{OT}}$ with any 2-round semi-honest OT protocol, this immediately
leads to a semi-honest non-interactive PCG protocol $\Pi_{s\text{VOLE}}(\mathcal{F}_{p^r})$
for the subfield-VOLE correlation:
*   $\mathcal{R}.\text{Gen}(1^\lambda)$: sets $\text{pk}_\mathcal{R}$ to be the first message of $\Pi_{s\text{VOLE-Gen}}$
    and $\text{sk}_\mathcal{R}$ to be the secret state of $\mathcal{R}$.
*   $\mathcal{S}.\text{Gen}(\text{pk}_\mathcal{R})$: sets $m_\mathcal{S}$ to be the second message of $\Pi_{s\text{VOLE-Gen}}$
    on first message $\text{pk}_\mathcal{R}$, and $\text{sk}_\mathcal{S}$ to be the sender output in
    $\Pi_{s\text{VOLE-Gen}}$.
*   $\mathcal{R}.\text{Expand}(\text{sk}_\mathcal{R}, m_\mathcal{S})$: computes the output $\mathbf{k}_0$ of the receiver
    from the state $\text{sk}_\mathcal{R}$ and the second message $m_\mathcal{S}$, and outputs
    $\mathcal{G}_{s\text{VOLE}}.\text{Expand}(0, \mathbf{k}_0)$.
*   $\mathcal{S}.\text{Expand}(\text{sk}_\mathcal{S})$: outputs $\mathcal{G}_{s\text{VOLE}}.\text{Expand}(1, \text{sk}_\mathcal{S})$.

**COROLLARY 5.3.** *Assuming the $(\text{HW}_t, \mathbf{H}, \mathbb{F}_{p^r})\text{-dual-LPN}(n', n)$
assumption, $\Pi_{s\text{VOLE-Gen}}$ is a semi-honest non-interactive PCG proto-
col for subfield-VOLE correlations over an arbitrary extension field $\mathbb{F}_{p^r}$
of $\mathbb{F}_2$, which only makes a black-box use of a 1-out-of-2 semi-honest
2-message $\text{OT}$ and a length-doubling $\text{PRG}$ $\mathbf{G}: \{0, 1\}^\lambda \leftrightarrow \{0, 1\}^{2\lambda}$. By
making additionally a single black-box use of a 2-message length-$t$
semi-honest reverse $\text{VOLE}$, this can be generalized to arbitrary fields.*

In the above corollary, $\Pi_{s\text{VOLE-Gen}}$ makes $t \cdot \lceil \log n' \rceil$ black-box
accesses to the 1-out-of-2 semi-honest 2-message $\text{OT}$, $t \cdot n'$ black-box
accesses to a length-doubling $\text{PRG}$, and additionally computes one
matrix-vector multiplication with $\mathbf{H}$. Regarding communication,
the size of $\text{pk}_\mathcal{R}$ is $t \cdot \lceil \log n' \rceil N_\mathcal{R}$ and the size of $m_\mathcal{S}$ is $t \cdot (1 + \lceil \log n' \rceil
N_\mathcal{S})$, where $N_\mathcal{R}$ (resp. $N_\mathcal{S}$) denote the receiver communication (resp.
the sender communication) in the underlying $\text{OT}$ protocol; over
general fields, there is an additional $\mathbf{t} \cdot M_\mathcal{R}(t, q, r)$ term in the size
of $\text{pk}_\mathcal{R}$ and $+ t \cdot M_\mathcal{S}(t, q, r)$ in the size of $m_\mathcal{S}$, where $M_\mathcal{R}(t, q, r)$ (resp.
$M_\mathcal{S}(t, q, r))$ denote the receiver communication (resp. the sender
communication) in the underlying length-$t$ reverse subfield-VOLE
protocol over $\mathbb{F}_{p^r}$.

## 5.3 Semi-Honest Non-Interactive Secure Computation with Silent Preprocessing

While the non-interactive PCG protocols of the previous section
are interesting in their own right, we observe that they satisfy the
features outlined in Section 3.2, and therefore lead to 2-round proto-
cols, and even silent NISC, for the OT and the VOLE functionalities.

### 5.3.1 Semi-Honest 2-Round OT with Silent Preprocessing.

As ob-
served in [16], a PCG for subfield-VOLE correlation together with
a correlation-robust hash function lead to a PCG $\mathcal{G}_{\text{OT}}$ for the $\text{ROT}$
correlation. For completeness, we recall the construction $\mathcal{G}_{\text{OT}}$ on
Figure 9. Using our distributed generation algorithm (which can
be implemented in two rounds given any 2-round $\text{OT}$ and 2-round
**Construction $\mathcal{G}_{\text{OT}}$**
**PARAMETERS:**
*   Security parameter $1^\lambda$, integers $n, q = p^r$.
*   An $\mathbb{F}_p$-correlation-robust function $\mathbf{H}: \{0, 1\}^\lambda \times \mathbb{F}_q \rightarrow
    \{0, 1\}^\lambda$.
*   The subfield-VOLE PCG ($\mathcal{G}_{s\text{VOLE}}.\text{Gen}, \mathcal{G}_{s\text{VOLE}}.\text{Expand}$)
**CORRELATION:**
Outputs $(\mathbf{R}_0, \mathbf{R}_1) = (\{(\mathbf{u}_i, \mathbf{v}_i, \mathbf{u}_i \mathbf{x})\}_{i \in [n]}, \{\mathbf{w}_{i, j}\}_{i \in [n], j \in [p]})$,
where $\mathbf{w}_{i, j} \in \{0, 1\}^\lambda$ and $\mathbf{u}_i \leftarrow \{1, \ldots, p\}$, for $i \in [n], j \in [p]$.
**GEN:** On input $1^\lambda$, output $(\mathbf{k}_0, \mathbf{k}_1) \leftarrow \mathcal{G}_{s\text{VOLE}}.\text{Gen}(1^\lambda, n, p, q)$.
**EXPAND:** On input $(\sigma, \mathbf{k}_\sigma)$:
(1) If $\sigma = 0$: compute $(\mathbf{u}', \mathbf{v}') \leftarrow \mathcal{G}_{s\text{VOLE}}.\text{Expand}(\sigma, \mathbf{k}_\sigma)$,
where $\mathbf{u}' \in \mathbb{F}_{p^r}^n, \mathbf{v}' \in \mathbb{F}_{p^r}^n$. Compute
$$ \mathbf{v}_i \leftarrow \mathbf{H}(i, \mathbf{v}'_i) \quad \text{for } i = 1, \ldots, n $$
and output $(\mathbf{u}_i, \mathbf{v}_i)$.
(2) If $\sigma = 1$: compute $(\mathbf{x}, \mathbf{w}') \leftarrow \mathcal{G}_{s\text{VOLE}}.\text{Expand}(\sigma, \mathbf{k}_\sigma)$,
where $\mathbf{x} \in \mathbb{F}_q, \mathbf{w}' \in \mathbb{F}_{p^r}^n$. Compute
$$ \mathbf{w}_{i, j} \leftarrow \mathbf{H}(i, \mathbf{w}'_i - j \cdot \mathbf{x}) \quad \text{for } i = 1, \ldots, n, \forall j \in \mathbb{F}_p $$
and output $\{\mathbf{w}_{i, j}\}_{i, j}$.

**Figure 9: PCG for $n$ sets of 1-out-of-$p$ random OT**

subfield-VOLE), together with the standard protocol for chosen-
input OT from ROT, directly leads to a **2-round OT extension pro-
tocol**, which performs $n$ OTs on $s$-bit strings with communication
$(2s + 1) n + o(n)$ (for any $s$).

**Protocol $\Pi_{\text{OT}}$:**
**PARAMETERS:** $1^\lambda, n, N = 2^\ell, t, p, r \in \mathbb{N}$. $\mathbf{H} \in \mathbb{F}_p^{N \times n}$. $\text{PPRF}$ is a
puncturable $\text{PRF}$ with domain $[N]$ and range $\mathbb{F}_{p^r}$. $\mathcal{D}_{t, N}$ is a
weight-$t$ error distribution over $\mathbb{F}_p^N$.
**INPUTS:**
*   $\mathcal{R}$ inputs $n$ field elements $(\mathbf{s}_i)_{i \leq n} \in \mathbb{F}_p$.
*   $\mathcal{S}$ inputs $n$ length-$p$ vectors $(\mathbf{m}_i)_{i \leq n}$ where each $\mathbf{m}_i$ is
    over $(\{0, 1\}^\lambda)^p$.
**PROTOCOL:**
(1) $\mathcal{R}$ picks $\mathbf{e} \leftarrow \mathcal{D}_{t, N}$. Let $\mathcal{S} = \{\mathbf{a}_1, \ldots, \mathbf{a}_t\} \in [N]^t$ be the
sorted indices of non-zero entries in $\mathbf{e}$, and $y_i = \mathbf{e}_{\mathbf{a}_i} \in \mathbb{F}_p$.
$\mathcal{R}$ computes the first part $\mathbf{u}$ of $\mathcal{G}_{\text{OT}}.\text{Expand}(0, \mathbf{k}_0)$ (note
that $\mathbf{u}$ is computed as $\mathbf{e} \cdot \mathbf{H}$ where $\mathbf{e}$ depends solely on $\mathbf{e}$).
$\mathcal{R}$ sets $\mathbf{t} \leftarrow \mathbf{u} + \mathbf{s}$.
(2) $\mathcal{S}$ samples $\mathbf{x} \in \mathbb{F}_{p^r}$ and $\mathbf{k}_{\text{pprf}} \leftarrow \text{PPRF}.\text{Gen}(1^\lambda)$. He
sets $\mathbf{k}_1 \leftarrow (\mathbf{k}_{\text{pprf}}, \mathbf{x})$ and computes $\{\mathbf{w}_{i, j}\}_{i \leq n, j \leq p} \leftarrow$
$\mathcal{G}_{\text{OT}}.\text{Expand}(1, \mathbf{k}_1)$.
(3) $\mathcal{R}$ computes and sends to $\mathcal{S}$ the first round of $\Pi_{s\text{VOLE-Gen}}$
on input $\mathbf{e}$, together with $\mathbf{t}$.
(4) $\mathcal{S}$ computes and sends to $\mathcal{R}$ the second round of
$\Pi_{s\text{VOLE-Gen}}$ on input $(\mathbf{x}, \mathbf{k}_{\text{pprf}})$ together with $\mathbf{m}_{i, \mathbf{s}_i} \leftarrow
\mathbf{m}_{i, \mathbf{s}_i} + \mathbf{w}_{i, \mathbf{s}_i}$ for $i = 1$ to $n$ and $\mathbf{j} = 1$ to $p$; $\mathcal{R}$ gets an
output $\mathbf{k}_0$.
(5) $\mathcal{R}$ computes $(\mathbf{u}, \mathbf{v}) \leftarrow \mathcal{G}_{\text{OT}}.\text{Expand}(0, \mathbf{k}_0)$ and outputs
$(\mathbf{m}_{i, \mathbf{s}_i} - \mathbf{v}_i)_{i \leq n}$

**Figure 10: Two-Round OT Extension**

**THEOREM 5.4.** *Assuming the $(\text{HW}_t, \mathbf{H}, \mathbb{F}_p)\text{-dual-LPN}(n', n)$ as-
sumption, $\Pi_{\text{OT}}$ is a semi-honest 2-round $\text{OT}$ extension with silent
preprocessing for generating $n$ 1-out-of-2 $\text{OT}$s, which makes $o(n)$
black-box uses of a 2-round semi-honest 1-out-of-2 $\text{OT}$, and $\mathcal{O}(n)$
black-box uses to a length-doubling $\text{PRG}$ and an $\mathbb{F}_p$-correlation robust
hash function.*

*Assuming further any 2-round semi-honest reverse $\text{VOLE}$, there is
a 2-round $\text{OT}$ extension with silent preprocessing for 1-out-of-$p$ $\text{OT}$
with comparable costs, using additionnally one black-box execution
of a reverse-$\text{VOLE}$ on length-$o(n)$ inputs.*

*PROOF.* In the above theorem, $\Pi_{\text{OT}}$ additionally requires the
computation of one matrix-vector multiplication with $\mathbf{H}$. It has
total communication $(2s + 1) n + o(n)$, where $s$ is the bit-length
of the sender messages. We represent the protocol for 2-round $\text{OT}$
extension on Figure 10.
Correctness. By the correctness of $\Pi_{s\text{VOLE-Gen}}$ and $\mathcal{G}_{\text{OT}}$, it holds
that $\mathbf{v}_i = \mathbf{m}_{i, \mathbf{s}_i} + \mathbf{w}_{i, \mathbf{s}_i}$ for $i = 1$ to $n$. Therefore, $\mathbf{m}_{i, \mathbf{s}_i} - \mathbf{v}_i = \mathbf{m}_{i, \mathbf{s}_i} -
(\mathbf{m}_{i, \mathbf{s}_i} + \mathbf{w}_{i, \mathbf{s}_i}) = -\mathbf{w}_{i, \mathbf{s}_i}$.

Security. We exhibit a simulator $\text{Sim}$ that generates a view indis-
tinguishable from an honest run of the protocol as long as a single
party is corrupted.

Case 1: $\mathcal{S}$ is corrupted. $\text{Sim}$ simulates $\mathcal{R}$ by constructing $(\mathbf{e}, \mathbf{u})$
honestly, participating to $\Pi_{s\text{VOLE-Gen}}$ as $\mathcal{R}$ does (note that this does
not require any input of $\mathcal{R}$). $\text{Sim}$ simulates $\mathbf{t}$ by sending $\mathbb{F}_{p^r}^n$.
Since $\Pi_{s\text{VOLE-Gen}}$ securely emulates $\mathcal{F}_{\text{Gen}}$, no information about $\mathbf{u}$
leaks to $\mathcal{S}$ during the execution of $\Pi_{s\text{VOLE-Gen}}$. By the security of
$\mathcal{G}_{\text{OT}}$, $\mathbf{u}$ is computationally indistinguishable from random from the
viewpoint of $\mathcal{S}$, hence so is $\mathbf{t} = \mathbf{u} + \mathbf{s}$; therefore, the simulation is
indistinguishable from an honest run of the protocol.

Case 2: $\mathcal{R}$ is corrupted. $\text{Sim}$ receives $\mathcal{R}$’s input $(\mathbf{s}_i)_{i \leq n}$, $\mathcal{R}$’s random
tape, and the corresponding target output $(\mathbf{m}_{i, \mathbf{s}_i})_{i \leq n}$ from the $\text{OT}$
functionality. $\text{Sim}$ simulates $\mathcal{S}$ by sampling $\mathbf{k}_1$ and computing the
$\mathbf{w}_{i, j}$ honestly (this does not require the input of $\mathcal{S}$). $\text{Sim}$ computes
the random noise vector $\mathbf{e}$ of $\mathcal{R}$ using $\mathcal{R}$’s random tape, from which
he can compute $\mathcal{R}$’s output $\mathbf{k}_0 = (\mathbf{u}, \mathbf{v})$. For $i = 1$ to $n$, $\text{Sim}$ computes
$\mathbf{m}_{i, \mathbf{s}_i} \leftarrow \mathbf{m}_{i, \mathbf{s}_i} + \mathbf{v}_i$, and picks $\mathbf{m}_{i, j} \leftarrow \{0, 1\}^\lambda$ for each $j \neq \mathbf{s}_i$. $\text{Sim}$
sends $(\mathbf{m}_{i, j})_{i, j}$ to $\mathcal{R}$. By the security of $\Pi_{s\text{VOLE-Gen}}$ and $\mathcal{G}_{\text{OT}}$, the $\mathbf{m}_{i, j}$
for $j \neq \mathbf{s}_i$ are indistinguishable from random from the viewpoint of
$\mathcal{R}$, hence the simulation is indistinguishable from an honest run of
the protocol.

### 5.3.2 NISC for OT with Silent Preprocessing.

Our 2-round OT ex-
tension protocol does actually directly give rise to a non-interactive
secure computation protocol for the oblivious transfer function-
ality, with **silent preprocessing**, as defined in Section 3.2. For the
sake of concreteness, we frame our OT extension protocol into the
language of NISC with silent preprocessing on Figure 11.

**Protocol $\text{NISCOT}$:**
**PARAMETERS:** $1^\lambda, n, N = 2^\ell, t, p, r \in \mathbb{N}$. $\mathbf{H} \in \mathbb{F}_p^{N \times n}$. $\text{PPRF}$ is a
puncturable $\text{PRF}$ with domain $[N]$ and range $\mathbb{F}_{p^r}$. $\mathcal{D}_{t, N}$ is a
weight-$t$ error distribution over $\mathbb{F}_p^N$.
**par** $\leftarrow (1^\lambda, n, N, t, p, r, \mathbf{H})$.
**INPUTS:**
*   $\mathcal{R}$ inputs $n$ field elements $(\mathbf{s}_i)_{i \leq n} \in \mathbb{F}_p$.
*   $\mathcal{S}$ inputs $n$ length-$p$ vectors $(\mathbf{m}_i)_{i \leq n}$ where each $\mathbf{m}_i$ is
    over $(\{0, 1\}^\lambda)^p$.
**PROTOCOL:**
*   $\mathcal{R}.\text{Gen}(\text{par})$ : pick $\mathbf{e} \leftarrow \mathcal{D}_{t, N}$. Compute the first round
    of $\Pi_{s\text{VOLE-Gen}}$ on input $\mathbf{e}$. Set $\text{sk}_\mathcal{R}$ to be the secret state
    of $\mathcal{R}$ after computing the first round of $\Pi_{s\text{VOLE-Gen}}$, and
    $\text{pk}_\mathcal{R}$ be the message computed by $\mathcal{R}$ in $\Pi_{s\text{VOLE-Gen}}$.
*   $\mathcal{R}.\text{Expand}_{\text{in}}(\text{sk}_\mathcal{R})$ : compute the first part $\mathbf{u}$ of
    $\mathcal{G}_{\text{OT}}.\text{Expand}(0, \mathbf{k}_0)$ (note that $\mathbf{u}$ is computed as $\mathbf{e} \cdot \mathbf{H}$ where
    $\mathbf{e}$ depends solely on $\mathbf{e}$). Output $\mathbf{r}_{\text{in}} \leftarrow \mathbf{u}$.
*   $\mathcal{S}.\text{Gen}(\text{pk}_\mathcal{R})$ : sample $\mathbf{x} \in \mathbb{F}_{p^r}$ and $\mathbf{k}_{\text{pprf}} \leftarrow
    \text{PPRF}.\text{Gen}(1^\lambda)$. Set $\text{sk}_\mathcal{S} \leftarrow (\mathbf{k}_{\text{pprf}}, \mathbf{x})$. Define $\text{pk}_\mathcal{S}$ to be
    the second round message of $\mathcal{S}$ in $\Pi_{s\text{VOLE-Gen}}$ on input
    $(\mathbf{x}, \mathbf{k}_{\text{pprf}})$ after receiving the message $\text{pk}_\mathcal{R}$ from $\mathcal{R}$.
*   $\mathcal{R}.\text{Expand}_{\text{out}}(\text{sk}_\mathcal{R}, \text{pk}_\mathcal{S})$ : compute the output $\mathbf{r}_{\text{out}}$ ob-
    tained by $\mathcal{R}$ with state $\text{sk}_\mathcal{R}$ upon receiving the message
    $\text{pk}_\mathcal{S}$ from $\mathcal{S}$ in $\Pi_{s\text{VOLE-Gen}}$.
*   $\mathcal{S}.\text{Expand}(\text{sk}_\mathcal{S})$ : compute $\mathbf{r}_{\mathcal{S}} = \{\mathbf{w}_{i, j}\}_{i \leq n, j \leq p} \leftarrow$
    $\mathcal{G}_{\text{OT}}.\text{Expand}(1, \text{sk}_\mathcal{S})$.
*   $\mathcal{R}.\text{Msg}(\mathbf{r}_{\text{in}}, \mathbf{s}_i)$: output $\hat{\mathbf{x}} \leftarrow \mathbf{r}_{\text{in}} + \mathbf{s}_i$.
*   $\mathcal{S}.\text{Msg}(\mathbf{r}_\mathcal{S}, \hat{\mathbf{x}}, (\mathbf{m}_i)_i)$ : parse $\mathbf{r}_\mathcal{S}$ as $\{\mathbf{w}_{i, j}\}_{i \leq n, j \leq p}$. Compute
    $\mathbf{m}_{i, \hat{\mathbf{x}}_i} \leftarrow \mathbf{m}_{i, \hat{\mathbf{x}}_i} + \mathbf{w}_{i, \hat{\mathbf{x}}_i}$ for $i = 1$ to $n$ and $j = 1$ to $p$ and
    output $\hat{\mathbf{z}} = (\mathbf{m}_{i, j})_{i, j}$.
*   $\mathcal{R}.\text{Dec}(\mathbf{r}_{\text{out}}, \hat{\mathbf{x}}, \hat{\mathbf{z}})$ : parse $\hat{\mathbf{z}}$ as $(\mathbf{m}_{i, j})_{i, j}$. Compute $(\mathbf{u}, \mathbf{v}) \leftarrow$
    $\mathcal{G}_{\text{OT}}.\text{Expand}(0, \mathbf{r}_{\text{out}})$ and output $\mathbf{z} = (\mathbf{m}_{i, \hat{\mathbf{x}}_i} - \mathbf{v}_i)_{i \leq n}$

**Figure 11: Non-interactive secure computation with silent preprocessing for oblivious transfer**

### 5.3.3 Semi-Honest NISC for Reverse Subfield-VOLE.

The same de-
randomization strategy as above directly implies, starting from the
non-interactive PCG protocol for subfield-VOLE of Section 5.2, a
NISC protocol for **reverse subfield-VOLE with silent preprocessing**,
with features comparable to that of the NISC for OT extension. We
omit the details.

**THEOREM 5.5.** *Suppose the $(\text{HW}_t, \mathbf{H}, \mathbb{F}_p)\text{-dual-LPN}(n', n)$ as-
sumption holds. Then there is a semi-honest $\text{NISC}$ protocol for
reverse subfield-$\text{VOLE}$ with silent preprocessing for generating
length-$n$ reverse subfield-$\text{VOLE}$s over an arbitrary field $\mathbb{F}_p$, which
uses $o(n)$ black-box executions of a 2-message semi-honest 1-out-of-2
$\text{OT}$, $\mathcal{O}(n)$ black-box calls to a length-doubling $\text{PRG}$, one black-box call
to a 2-message semi-honest reverse $\text{VOLE}$, and additionally computes
one matrix-vector multiplication with $\mathbf{H}$. It has total communication
$(2s + 1) n + o(n)$.*

# 6 MALICIOUS DISTRIBUTED SETUP

In this section, we present protocols for VOLE, OT extension and
NISC with security against malicious parties. Our final protocol for
OT extension takes place in four rounds, and can be compressed to
two rounds via Fiat-Shamir.

We begin in Section 6.1 by formalizing and describing an aug-
mented PPRF primitive with a “**malicious key verification**” proce-
dure, corresponding to the event of when a selective-failure attack
will (or not) be identified. The described selective-failure-only se-
curity notion is formalized and achieved for distributed generation
of a single PPRF key in Section 6.2, and for $t$ PPRF keys (with con-
sistent $\mathbf{x}$) in Section 6.3. Then, in Section 6.4, we build atop this
functionality to obtain a PCG protocol for subfield VOLE with stan-
dard malicious security. In the full version we explain how the PCG
protocol for subfield VOLE can be converted into a four round PCG

**Functionality $\mathcal{F}_{\text{mal-PPRF}}$:**
**PARAMETERS:** $1^\lambda, N = 2^\ell, p, r \in \mathbb{N}$. $\text{PPRF}$ is a puncturable $\text{PRF}$
with domain $[N] = \{0, 1\}^\ell$, key space $\{0, 1\}^\lambda$, and range $(\mathbb{F}_{p^r})^2$,
supporting verification of malicious keys.
**INPUTS:**
*   $\mathcal{R}$ inputs $\mathbf{a} \in \{0, 1\}^\ell$.
*   $\mathcal{S}$ inputs $\beta \in (\mathbb{F}_{p^r})^2$ and a $\text{PPRF}$ key $\mathbf{k}_{\text{pprf}} \in \{0, 1\}^\lambda$.
**FUNCTIONALITY:**
*   If $\mathcal{S}$ is honest:
(1) Compute $\mathbf{k}^*_{\text{pprf}} = \text{PPRF}.\text{Puncture}(\mathbf{k}_{\text{pprf}}, \mathbf{a})$.
(2) Send $\mathbf{k}^*_{\text{pprf}}$ and $\mathbf{w} = -\text{PPRF}.\text{Eval}(\mathbf{k}_{\text{pprf}}, \mathbf{a}) + \beta$ to $\mathcal{R}$, and
    $\mathbf{k}_{\text{pprf}}$ to $\mathcal{S}$.
*   If $\mathcal{S}$ is corrupted:
(1) Wait for $\mathcal{A}$ to send a guess $\mathcal{I} \subseteq [N]$ and a key $\mathbf{K}^* \in \mathcal{K}$.
(2) Check that $\mathbf{a} \in \mathcal{I}$ and that $\text{Ver}(\mathbf{K}^*, \mathcal{I}) = 1$. If either
    check fails, send $\text{abort}$ to $\mathcal{R}$ and wait for a response
    from $\mathcal{R}$. When $\mathcal{R}$ responds with $\text{abort}$, forward this to
    $\mathcal{S}$ and halt.
(3) Compute $\mathbf{k}^*_{\text{pprf}} = \text{Puncture}^*(\mathbf{K}^*, \mathbf{a})$ and
    $\mathbf{w} = -\text{Eval}^*(\mathbf{K}^*, \mathcal{I}, \mathbf{a}) + \beta$.
(4) Send $\mathbf{k}^*_{\text{pprf}}$ and $\mathbf{w}$ to $\mathcal{R}$, and $\text{success}$ to $\mathcal{S}$.

**Figure 12: Functionality for malicious distributed setup of single-point $\text{PPRF}$**

## 6.1 Puncturable PRF with Malicious Keys

In the following sections we will realize a relaxed form of distributed
PPRF setup functionality, where a corrupt sender may choose its
own “master key,” defining a PRF evaluation that need not coincide
with any honest GGM tree, provided that it is consistent with the
receiver’s punctured point. The consistency check will serve as the
“getting caught” predicate in our ideal functionality. In this section,
we introduce necessary terminology in order for the consistency
check to be formulated.

Roughly, we will have a **malicious key space** $\mathcal{K}$, such that given
a malicious key $\mathbf{K} \in \mathcal{K}$ and a subset of values $\mathcal{I} \subset \mathcal{X}$ in the domain,
one can check whether puncturing $\mathbf{K}$ at any of the points in the set $\mathcal{I}$
yields a consistent output. For a formal definition and instantiation
for the GGM puncturable $\text{PRF}$, we refer to the full version.

In order to allow for a consistency check we use the GGM con-
struction with domain $[2N]$ and range $(\mathbb{F}_{p^r})^2 \times \{0, 1\}^\lambda$ (where the
former is the range of the left leaves and the latter the range of the
right leaves). We will use an output $((\mathbf{w}, \mathbf{w}'), \gamma) \in (\mathbb{F}_{p^r})^2 \times \{0, 1\}^\lambda$
as follows: The value $\mathbf{w}$ will correspond to the actual output of the
$\text{PPRF}$. The value $\mathbf{w}'$ will be used to ensure consistency within a single
$\text{PPRF}$ evaluation. The value $\gamma$ will be used to ensure consistency
across $t$ $\text{PPRF}$ evaluations.

## 6.2 Malicious Setup for Single-Point PPRF

As mentioned in the previous section, in order to achieve malicious
security of a single $\text{PPRF}$ evaluation, we use the redundancy intro-
duced via the domain extension for checking consistent behaviour,
by letting the sender provide a hash of all right leaves of the fully
evaluated GGM tree. The idea is that a sender computing the cor-
rect hash value (relative to the receiver’s input $\mathbf{a}$), either behaved
honestly, or guessed a set $\mathcal{I}$ such that $\mathbf{a} \in \mathcal{I}$. This is captured in
the functionality in Figure 12. The functionality is similar to the
semi-honest functionality given in Figure 12, but the adversary
is additionally allowed to give a set $\mathcal{I} \subseteq [N]$ as guess. If indeed
$\mathbf{a} \in \mathcal{I}$, the sender will successfully finish the protocol and learn
some partial information about $\mathbf{a}$ (namely, $\mathbf{a} \in \mathcal{I}$). Otherwise, the
functionality will abort.

In order for the right leaves of the GGM tree to fix a unique tree,
we require the $\text{PRG}$ of the last level $\mathbf{G}': \{0, 1\}^\lambda \leftrightarrow (\mathbb{F}_{p^r})^2 \times \{0, 1\}^\lambda$ to
satisfy an additional property we call **right-half injectivity**, formally
defined in the full version.

The protocol we present implements the functionality for the
$\text{PPRF}_1$, which corresponds to the GGM $\text{PPRF}$, but where evaluation
returns a value in $(\mathbb{F}_{p^r})^2$ instead of $(\mathbb{F}_{p^r})^2 \times \{0, 1\}^\lambda$.

We give the protocol for distributed setup of $\text{PPRF}_1$ with security
against malicious adversaries in Figure 13. First, in steps 1–6 the
parties run the semi-honest protocol, such that the receiver holds
a key $\mathbf{k}^*$ punctured at $\mathbf{a} || 0$ and the sender a possibly malicious
key $\mathbf{K}$. As the tree is always punctured at an even value, both
parties can compute all the right leaves of the GGM tree. The sender
additionally sends a hash of all these leaves to the receiver. The
receiver checks if this hash is consistent with his view and aborts
otherwise.

**Protocol $\Pi_{\text{mal-PPRF}}$:**
**PARAMETERS:** $1^\lambda, \ell, N = 2^\ell, p, r \in \mathbb{N}$. $\text{PPRF}_{\text{GGM}}$ is the GGM
puncturable $\text{PRF}$ with domain $\{0, 1\}^{\ell+1} = [2N]$, key space
$\{0, 1\}^\lambda$, and range $(\mathbb{F}_{p^r})^2 \times \{0, 1\}^\lambda$, constructed from a length-
doubling $\text{PRG}$ $\mathbf{G}: \{0, 1\}^\lambda \leftrightarrow \{0, 1\}^{2\lambda}$, and a second $\text{PRG}$
$\mathbf{G}': \{0, 1\}^\lambda \leftrightarrow (\mathbb{F}_{p^r})^2 \times \{0, 1\}^\lambda$ used to compute the $\text{PRF}$ outputs
on the last level of the tree.
**INPUTS:**
*   $\mathcal{R}$ inputs $\mathbf{a} \in \{0, 1\}^\ell$.
*   $\mathcal{S}$ inputs $\beta \in (\mathbb{F}_{p^r})^2$ and a $\text{PPRF}$ key $\mathbf{k}_{\text{pprf}} \in \{0, 1\}^\lambda$.
**PROTOCOL:**
(1) $\mathcal{S}$ samples a random seed $\mathbf{k}_{\text{pprf}} \in \{0, 1\}^\lambda$.
(2) $\mathcal{S}$ computes the $2^{\ell}$ partial evaluations at level $\ell$ of the
    GGM tree:
    (a) $\mathcal{S}$ sets $\mathbf{s}_0^{\ell} = \mathbf{k}_{\text{pprf}}$.
    (b) For $i \in \{1, \ldots, \ell\}, j \in [0, \ldots, 2^{i-1})$: $\mathcal{S}$ computes
    $(\mathbf{s}_{2j}^{i}, \mathbf{s}_{2j+1}^{i}) = \mathbf{G}(\mathbf{s}_{j}^{i-1})$.
    (c) For $j \in \{0, 1\}, \mathcal{S}$ computes $(\mathbf{s}_{2j}^{\ell+1}, \mathbf{s}_{2j+1}^{\ell+1}) = \mathbf{G}'(\mathbf{s}_j^\ell) \in
    (\mathbb{F}_{p^r})^2 \times \{0, 1\}^\lambda$.
(3) $\mathcal{S}$ computes the “left” and “right” halves for $i \in \{1, \ldots, \ell\}$:
    $\mathbf{K}_0^i = \bigoplus_{j \in [0, 2^{i-1})} \mathbf{s}_{2j}^i \mathbf{K}_1^i = \bigoplus_{j \in [0, 2^{i-1})} \mathbf{s}_{2j+1}^i$
(4) $\mathcal{S}$ computes the “right” half for $i = \ell + 1$:
    $\mathbf{K}_R^{\ell+1} = \bigoplus_{j \in \{0, 1\}} \mathbf{s}_{2j+1}^{\ell+1}$
(5) For $i = 1, \ldots, \ell = \log N$ (in parallel) the parties run $\text{OT}$
    where in the $i$-th $\text{OT}$:
    (a) $\mathcal{R}$ inputs the choice bit $\alpha_i$.
    (b) $\mathcal{S}$ inputs the pair $(\mathbf{K}_0^i, \mathbf{K}_1^i)$.
(6) $\mathcal{S}$ sends to $\mathcal{R}$ the key $\mathbf{K}_R^{\ell+1}$ and the correction value
    $\mathbf{c} = \beta - \bigoplus_{j \in [N]} \mathbf{s}_j^{\ell+1}$.
(7) For the consistency check, $\mathcal{S}$ sets $\mathbf{y}_j = \mathbf{s}_{2j+1}^{\ell+1}$ for all $j \in [N]$
    and sends to $\mathcal{R}$ the value $\Gamma = h(\mathbf{y}_0, \ldots, \mathbf{y}_{N-1})$.
(8) Let $\{\mathbf{K}^i\}_{i=1}^{\ell+1}$ denote the $\text{OT}$ outputs received by $\mathcal{R}$ to-
    gether with the key of the $(\ell + 1)$-st level. Then, $\mathcal{R}$ pro-
    ceeds as follows.
    (a) $\mathbf{k}^*_{\text{pprf}} \leftarrow \text{Puncture}^*(\{\mathbf{K}^i\}_{i=1}^{\ell+1}, \mathbf{a})$.
    (b) $\{\mathbf{s}_j\}_{j \neq \mathbf{a} || 0} \leftarrow \text{PPRF}_{\text{GGM}}.\text{FullEval}(\mathbf{k}^*_{\text{pprf}}, \mathbf{a} || 0)$.
    (c) $\mathcal{R}$ receives $\mathbf{c}$, and computes
    $\mathbf{w} = \mathbf{c} - \bigoplus_{j \in [N] \setminus \{\mathbf{a}\}} \mathbf{s}_{2j}$.
    (d) To verify consistency, $\mathcal{R}$ sets $\mathbf{y}_j = \mathbf{s}_{2j+1}^{\ell+1}$ for all $j \in [N]$,
    and computes $\Gamma' = h(\mathbf{y}_0, \ldots, \mathbf{y}_{N-1})$.
(9) If $\Gamma = \Gamma'$, $\mathcal{R}$ outputs the punctured key $\mathbf{k}^*_{\text{pprf}}$ and the
    final correction value $\mathbf{w}$. Otherwise, $\mathcal{R}$ aborts.

**Figure 13: Protocol for distributed setup of single-point $\text{PPRF}$ with consistency check**

## 6.3 Malicious Setup of $t$ PPRFs with Consistent Offset

For the VOLE setup with malicious security, we require a protocol
for distributed setup of $t$ $\text{PPRFs}$, where the inputs $\beta_j$ of the sender
are **consistent** across all evaluations. By consistent, we mean that
each $\beta_j$ is an additive share of $\mathbf{x} y_j$, where the receiver knows the
other share and the noise value $y_j \in \mathbb{F}_p$. To this end, we introduce a
second consistency check, where the sender has to provide a linear
combination of the outputs of each $\text{PPRF}$. We show that a cheating
sender will fail this final check, unless he managed to guess part of
the receiver’s input. This guessing is modelled by the functionality
$\mathcal{F}_{\text{mal-}t\text{-PPRF}}$ (Fig. 15), which is parameterized by a 1-puncturable
$\text{PRF}$ with verification of malicious keys.

To carry out this check, we exploit the extended range of the
$\text{PPRF}$ given by the functionality $\mathcal{F}_{\text{mal-PPRF}}$. The extra $\mathbb{F}_{p^r}$ element
from each evaluation serves to check consistency, by taking a ran-
dom linear combination of all these outputs (for each $\text{PPRF}$), to-
gether with a linear combination of the original outputs, and send-
ing these to the receiver to check. Note that without the extended
range, sending a linear combination of $\text{PPRF}$ outputs to the receiver
would leak the sender’s input $\mathbf{x}$; with the extra outputs, however,
the sender can use a random value $\mathbf{x}$ which serves to mask $\mathbf{x}$.
Since we sacrifice the extended outputs in the consistency check,
the functionality $\mathcal{F}_{\text{mal-}t\text{-PPRF}}$ which we realize gives us a $\text{PPRF}$ with
range $\mathbb{F}_{p^r}$, which is defined by simply ignoring the first element
output from the one with range $\mathbb{F}_{p^r}$.

**Functionality $\mathcal{F}_{p\text{-rev-VOLE}}$:**
**PARAMETERS:** $t, p, r \in \mathbb{N}$.
**INPUT:** The sender $\mathcal{S}$ inputs a pair $((\beta, \mathbf{x}), (\mathbf{b}, \mathbf{x})) \in (\mathbb{F}_{p^r}^{t} \times \mathbb{F}_{p^r})^2$.
The receiver $\mathcal{R}$ inputs a vector $\mathbf{y} \in \mathbb{F}_p^t$.
**FUNCTIONALITY:** Compute $\hat{\mathbf{y}} \leftarrow \mathbf{y} \mathbf{x} - \beta$ and $\hat{\mathbf{c}} \leftarrow \mathbf{y} \mathbf{x} - \mathbf{b}$ and
output $(\hat{\mathbf{y}}, \hat{\mathbf{c}})$ to $\mathcal{R}$.

**Figure 14: Generalized Reverse Vector-OLE Functionality over a Field $\mathbb{F}_p$**

**Functionality $\mathcal{F}_{\text{mal-}t\text{-PPRF}}$:**
**PARAMETERS:** $1^\lambda, N = 2^\ell, t, p, r \in \mathbb{N}$. $\text{PPRF}$ is a puncturable $\text{PRF}$
with domain $[N]$, key space $\{0, 1\}^\lambda$, and range $\mathbb{F}_{p^r}$, supporting
verification of malicious keys.
**INPUTS:**
*   $\mathcal{R}$ inputs indices $\mathbf{a}_1, \ldots, \mathbf{a}_t \in [N]$ and weights
    $y_1, \ldots, y_t \in \mathbb{F}_p$. We define $\mathcal{S} = \{\mathbf{a}_1, \ldots, \mathbf{a}_t\}$.
*   $\mathcal{S}$ inputs $\mathbf{k}_1 = (\{\mathbf{k}_j\}_{j \in [t]}, \mathbf{x})$, where $\mathbf{x} \in \mathbb{F}_{p^r}$ and $\mathbf{k}_j \in
    \{0, 1\}^\lambda$.
**FUNCTIONALITY:**
*   If $\mathcal{S}$ is honest:
(1) Compute $\mathbf{k}^*_j \leftarrow \text{PPRF}.\text{Puncture}(\mathbf{k}_j, \mathbf{a}_j)$, for $j \in$
    $\{1, \ldots, t\}$.
(2) Let $\mathbf{z}_j \leftarrow \mathbf{x} \cdot y_j - \text{PPRF}.\text{Eval}(\mathbf{k}_j, \mathbf{a}_j)$ for $j \in \{1, \ldots, t\}$.
(3) Let $\mathbf{k}_0 \leftarrow (\{\mathbf{k}^*_j, \mathbf{z}_j\}_{j \in [t]}, \mathcal{S}, \mathbf{y})$.
(4) Output $\mathbf{k}_0$ to $\mathcal{R}$.
*   If $\mathcal{S}$ is corrupted:
(1) Receive from $\mathcal{A}$ $t$ subsets $\mathcal{I}_1, \ldots, \mathcal{I}_t \subseteq [N]$ and a set of
    keys $\mathbf{K}_1^*, \ldots, \mathbf{K}_t^* \in \mathcal{K}$.
(2) For each $j \in \{1, \ldots, t\}$ check that $\mathbf{a}_j \in \mathcal{I}_j$ and that
    $\text{Ver}(\mathbf{K}_j^*, \mathcal{I}_j) = 1$. If any check fails, abort.
(3) Compute $\mathbf{k}^*_j = \text{PPRF}.\text{Puncture}^*(\mathbf{K}_j^*, \mathbf{a}_j)$ for each $j \in
    \{1, \ldots, t\}$.
(4) Let $\mathbf{z}_j \leftarrow \mathbf{x} \cdot y_j - \text{PPRF}.\text{Eval}^*(\mathbf{K}_j^*, \mathcal{I}_j, \mathbf{a}_j)$ for $j \in
    \{1, \ldots, t\}$.
(5) Output $\mathbf{k}_0 \leftarrow (\{\mathbf{k}^*_j, \mathbf{z}_j\}_{j \in [t]}, \mathcal{S}, \mathbf{y})$ to $\mathcal{R}$ and $\text{success}$
    to $\mathcal{S}$.

**Figure 15: Functionality for malicious distributed setup of $t$ puncturable $\text{PRFs}$**

To create the shares of $\mathbf{x} \cdot y_j$, when $p > 2$ we again need a
slightly stronger flavor of reverse $\text{VOLE}$, presented in Figure 14.
Here, we require the functionality to take two inputs by the sender
$((\beta, \mathbf{x}), (\mathbf{b}, \mathbf{x}))$, and only one input $\mathbf{y}$ by the receiver, and return to
the sender values $\hat{\mathbf{y}}, \hat{\mathbf{c}}$, such that $(\beta, \hat{\mathbf{y}})$ constitute sharings of $\mathbf{x} \times \mathbf{y}$
(and similar for $\mathbf{c}$). Note that it is not enough for our protocol to
instead call the basic reverse $\text{VOLE}$ functionality twice, as a receiver
providing inconsistent inputs in the two calls can learn the input $\mathbf{x}$
of the sender in the protocol $\Pi_{\text{mal-}t\text{-PPRF}}$ (Figure 16).
For a proof of the following theorem we refer to the full version.

**Protocol $\Pi_{\text{mal-}t\text{-PPRF}}$:**
**PARAMETERS:** $1^\lambda, N = 2^\ell, t, p, r \in \mathbb{N}$. $\text{PPRF}$ is a puncturable $\text{PRF}$
with domain $[N]$, key space $\{0, 1\}^\lambda$ and range $(\mathbb{F}_{p^r})^2$.
**INPUTS:**
*   $\mathcal{R}$ inputs distinct indices $\mathbf{a}_1, \ldots, \mathbf{a}_t \in [N]$ and weights
    $y_1, \ldots, y_t \in \mathbb{F}_p$. We define $\mathcal{S} = \{\mathbf{a}_1, \ldots, \mathbf{a}_t\}$ and $\mathbf{y} =
    (y_1, \ldots, y_t) \in (\mathbb{F}_p)^t$.
*   $\mathcal{S}$ inputs $\mathbf{x} \in \mathbb{F}_{p^r}$.
**PROTOCOL:**
(1) $\mathcal{S}$ picks $\beta, \mathbf{b} \in \mathbb{F}_{p^r}^t$ and $\mathbf{x} \in \mathbb{F}_{p^r}$.
(2) $\mathcal{R}$ and $\mathcal{S}$ call $\mathcal{F}_{p\text{-rev-VOLE}}(t, p)$ on respective inputs $\mathbf{y}$ and
    $((\beta, \mathbf{x}), (\mathbf{b}, \mathbf{x}))$. $\mathcal{R}$ receives $(\hat{\mathbf{y}}, \hat{\mathbf{c}}) \in \mathbb{F}_{p^r}^t \times \mathbb{F}_{p^r}^t$.
(3) $\mathcal{R}$ and $\mathcal{S}$ call $\mathcal{F}_{\text{mal-PPRF}}(1^\lambda, N, p, r)$ $t$ times on respective
    inputs $\mathbf{a}_j$ and $(\beta_j, b_j)$. $\mathcal{R}$ receives $\mathbf{k}^*_j$ and $(\mathbf{w}_j, \mathbf{w}'_j)$ for each
    $j \in \{1, \ldots, t\}$. If any of the runs is not successful, $\mathcal{R}$
    receives $\text{abort}$ from the functionality.
(4) $\mathcal{R}$ samples $\tau, \tau_0, \ldots, \tau_{N-1} \in \mathbb{F}_{p^r}$ and sends these to $\mathcal{S}$.
(5) $\mathcal{S}$ computes $(\mathbf{s}_{i, 2i}, \mathbf{s}_{i, 2i+1}, \mathbf{s}'_{i, 2i}, \mathbf{s}'_{i, 2i+1}) \leftarrow \text{PPRF}.\text{Eval}(\mathbf{k}_j, i)$ for $i \in$
    $[N], j \in \{1, \ldots, t\}$, and sends $\mathbf{X} = \mathbf{x} + \tau \cdot \mathbf{x}$ and $\mathbf{V}_{\mathcal{S}, j} =
    \sum_{i=0}^{N-1} \tau_i (\mathbf{s}_{i, 2i} + \tau_0 \cdot \mathbf{s}'_{i, 2i} + \mathbf{s}_{i, 2i+1} + \tau_0 \cdot \mathbf{s}'_{i, 2i+1})$ for $j \in \{1, \ldots, t\}$ to $\mathcal{R}$.
(6) $\mathcal{R}$ computes $(\mathbf{u}_{\mathcal{R}, 2i}, \mathbf{u}_{\mathcal{R}, 2i+1}, \mathbf{u}'_{\mathcal{R}, 2i}, \mathbf{u}'_{\mathcal{R}, 2i+1}) \leftarrow \text{PPRF}.\text{Eval}'(\mathbf{k}_j, i)$ for
    $i \in [N], j \in \{1, \ldots, t\}$, where $\text{PPRF}.\text{Eval}'$ is an algo-
    rithm that outputs $(\mathbf{u}_j, \mathbf{w}_j) + (\mathbf{y}_j, \mathbf{c}_j)$ on input $(\mathbf{k}_j, \mathbf{a}_j)$,
    and $-\text{PPRF}.\text{Eval}(\mathbf{k}_j, i)$ else.
    $\sum_{i=0}^{N-1} \tau_i (\mathbf{u}_{\mathcal{R}, 2i} + \tau_0 \cdot \mathbf{u}'_{\mathcal{R}, 2i} + \mathbf{u}_{\mathcal{R}, 2i+1} + \tau_0 \cdot \mathbf{u}'_{\mathcal{R}, 2i+1}) = \mathbf{X} \cdot \mathbf{y}_j \cdot \tau + \mathbf{V}_{\mathcal{R}, j}$
(7) $\mathcal{R}$ checks if $\mathbf{V}_{\mathcal{S}, j} + \sum_{i=0}^{N-1} \tau_i (\mathbf{s}_{i, 2i} + \tau_0 \cdot \mathbf{s}'_{i, 2i} + \mathbf{s}_{i, 2i+1} + \tau_0 \cdot \mathbf{s}'_{i, 2i+1}) = \mathbf{X} \cdot \mathbf{y}_j \cdot \tau$
    for $j \in \{1, \ldots, t\}$. If any of these checks fail or $\mathcal{R}$ received
    $\text{abort}$ from $\mathcal{F}_{\text{mal-PPRF}}$ in step 3, $\mathcal{R}$ aborts. Otherwise, $\mathcal{R}$
    sends $\text{ok}$ to $\mathcal{F}_{\text{mal-PPRF}}$ and outputs $\mathbf{k}_0 = (\{\mathbf{k}^*_j, \mathbf{z}_j\}_{j}, \mathcal{S}, \mathbf{y})$.

**Figure 16: Protocol for malicious distributed setup of $t$ puncturable $\text{PRFs}$**

**THEOREM 6.2.** *There exists a 4-message 2-party protocol $\Pi_{\text{mal-}t\text{-PPRF}}$
which securely implements the functionality $\mathcal{F}_{\text{mal-}t\text{-PPRF}}(1^\lambda, N, p, r)$
for the puncturable $\text{PRF}$ $\text{PPRF}$ in the $\mathcal{F}_{p\text{-rev-VOLE}}$-, parallel $\mathcal{F}_{\text{mal-PPRF}}$-
hybrid model, with malicious security, using $t$ parallel calls to $\mathcal{F}_{\text{mal-PPRF}}$,
and only one call to $\mathcal{F}_{p\text{-rev-VOLE}}$, and further communication of $(N +
t + 2)r \log p$ bits. Furthermore, when $p = 2$, the functionality can be
implemented in the parallel $\mathcal{F}_{\text{mal-PPRF}}$-hybrid model, using no call to
$\mathcal{F}_{p\text{-rev-VOLE}}$.*

Note that an additional $\text{PRG}$ with range $\mathbb{F}_{p^r}^{N+t}$, the communica-
tion can be reduced to just $(t + 1) r \log p + \lambda$ bits.

## 6.4 4-Round VOLE and OT Setup with Malicious Security

The $\mathcal{F}_{\text{mal-}t\text{-PPRF}}$ functionality can be immediately used to distribute
the setup of the subfield-VOLE $\text{PCG}$ from Section 4. To prove this
gives secure subfield-VOLE, however, we now need to assume that
the dual-LPN assumption remains secure when an adversary is
allowed to query (on average) one bit of information on the error
vector. This reflects the fact that a malicious sender in $\mathcal{F}_{\text{mal-}t\text{-PPRF}}$
can try to guess subsets containing the receiver’s $\mathbf{a}_j$ inputs, which
correspond to non-zero coordinates of the error vector. This assump-
tion with **leakage** is essentially the same as an assumption recently
used for maliciously secure $\text{MPC}$ based on syndrome decoding [34].
For a formal definition we refer to the full version.

**THEOREM 6.3.** *Suppose that $(\text{HW}_t, \mathcal{C}, \mathbb{F}_p)\text{-dual-LPN}(N, n)$ with
static leakage holds, where $N = \mathcal{O}(n)$ and $t = o(n/\log n)$. Then
there exists a 4-message, maliciously secure $\text{PCG}$ protocol for the sub-
field $\text{VOLE}$ correlation, which makes $o(n)$ parallel calls to an oblivious
transfer functionality, with communication complexity $o(n)$ bits.*

# 7 IMPLEMENTATION

## 7.1 Instantiating the Code and Parameters

The most costly part of our implementation is the syndrome com-
putation with the matrix $\mathbf{H}$ used in the dual-LPN assumption. We
optimize this by instantiating $\mathbf{H}$ using the parity-check matrix of a
**quasi-cyclic code**. Multiplication by $\mathbf{H}$ can then be done with poly-
nomial arithmetic in $\mathbb{Z}_2[X]/(X^n - 1)$, for which we use the library
$\text{bitpolymul}$ [21]. Another optimization that improves efficiency
and reduces the seed size is to use a **regular error distribution**,
where the error vector $\mathbf{e} \in \mathbb{F}_{2^r}^N$ is the concatenation of $t$ random
unit vectors, each of length $N/t$. To choose the code parameters
$N, n$ and the error weight $t$, we analyze security against the best
known attacks, additionally accounting for a $\sqrt{N}$ speedup that can
be obtained from the DOOM attack [53] when using quasi-cyclic
codes. As also observed in [35], we are not aware of any attacks
that exploit regular errors and perform significantly better than
usual.

In the full version, we provide more details on selecting param-
eters and describe some further optimizations for the syndrome
computation.

## 7.2 Results

We implement our semi-honest and malicious secure protocols and
report their performance in several different settings. The source
code can be found at https://github.com/osu-crypto/libOTe. The
benchmark was performed on a single AWS c4.4xLarge instance
with network latency artificially limited to emulate a LAN or WAN
settings. Specifically, we consider a **LAN setting** with bandwidth
of $10\text{Gbps}$ and $0\text{ms}$ latency and two **WAN settings** with $100, 10$
$\text{Mbps} \ \& \ 40\text{ms}$ one-way latency. We compare with the semi-honest
$\text{OT}$ extension protocol of Ishai et al. [37] ($\text{IKNP}$) and the malicious
secure protocol of [41] ($\text{KOS}$) as implemented by a state-of-the-
art library. Both our implementations and that of [37, 41] use the
same three round malicious secure base $\text{OT}$ protocol of Naor \&
Pinkas [47]. We note that our protocols can be composed with a
two round base $\text{OT}$ protocol to give a two round $\text{OT}$ extension. In
the WAN setting this optimization would reduce the running times
by approximately $40\text{ms}$ for all protocols.

The functionality we realize is to produce $n \in \{10^4, 10^5, 10^6, 10^7\}$
uniformly random $\text{OT}$s of length $128$ bits. One distinction between
our protocol and [37, 41] is that the choice bits of the receiver are
uniformly chosen by our protocol, while [37, 41] allows the receiver
to specify them. These random $\text{OT}$s can then be de-randomized
with additional communication.

Figure 17 contains the running time of our protocol. A fuller
table, with alternative choices of parameters (security parameter
$\lambda$, scaling parameter $N/n$, method for computing the base $\text{OT}$s) is
available in the full version. The primary takeaway is that both
of our protocols achieve extremely low communication while the
total running time remains competitive with or superior to $\text{KOS}$
and $\text{IKNP}$. We report running times with each party having 1 or
4 threads, along with a background $\text{IO}$ thread. In the LAN setting
with sub-millisecond latency \& $10\text{Gbps}$ we observe that the $\text{IKNP}$
and $\text{KOS}$ protocols achieve significant performance, requiring just
$0.26$ or $0.33$ seconds to compute $10$ million $\text{OT}$s with a single thread.
While the computational cost of $\text{IKNP}$ and $\text{KOS}$ does outperform our
implementation by roughly one order of magnitude, it also requires
between $1000$ and $2000$ times more communication. This difference
means that for more realistic network settings, such as $100\text{Mbps}$,
our implementation achieves a faster running time. With 4 threads
and a limit of $100\text{Mbps}$ our implementation is up to $5$ times faster
(counting total running time, including both local computation and
communication costs) and remains faster even for small $n$ where
our communication overheads are asymptotically closer together.

For the constrained setting of $10\text{Mbps}$ our protocol truly stands
out with a $47$ times speedup compared to $\text{IKNP}$ with $n = 10^7$ and
$t = 4$. We see a similar $46$ times speedup in the malicious setting
compared to $\text{KOS}$. Moreover, when comparing between the across
the different network settings our protocol incurs minimal to no
perform impact from decreasing bandwidth. For instance, with a
$10\text{Gbps}$ connection our semi-honest protocol processes $n = 10^7 \text{OT}$s
in $2.4$ seconds while with $1000$ times less bandwidth the protocol
still just requires $2.8$ seconds.

This scalability is explained in Figure 18 which contains the com-
munication overhead of our protocol. A fuller table, with alternative
choices of parameters (security parameter $\lambda$, scaling parameter $N/n$,
method for computing the base $\text{OT}$s) is available in the full ver-
sion. We parameterize our protocols by the desired security level
$\lambda \in \{80, 128\}$ and a tunable parameter $s = N/n$. The latter controls
a trade-off between the number of $\text{PPRF}$ evaluations and length of
the resulting vectors. To maintain security level of $\lambda$ bits, increasing
$s$ results in fewer $\text{PPRF}$ evaluations and less communication. How-
ever, it also increases the computational overhead. Our smallest
running times were achieved with $s = 2$. However, we also consider
$s = 4$ which decreases our total communication from $126\text{KB}$ to
$80\text{KB}$ for $n = 10^7$. In contrast, the $\text{IKNP}$ protocol requires $160\text{MB}$
for the same security level. This represents as much as a $2000$ times
reduction in communication. This low communication overhead
results in our protocol requiring as little as $0.038$ bits per $\text{OT}$ for
$n = 10^7$ and $\lambda = 80$. In our worst case of $n = 10^4$ our protocol
still requires between $3$ and $6$ times less communication than $\text{IKNP}$.
Another compelling property of our protocol is that we incur near
constant additive communication overhead when comparing our
malicious and semi-honest protocols.

**Figure 17: The running time in milliseconds of our implementation compared to [8] in both the LAN (0ms latency) and WAN (40ms one-way latency) settings, with security parameter $\lambda = 128$. $\lambda$ is the computational security parameter. We set the scaling $N/n$ to 2. $\tau$ denotes the number of threads. Hybrid refers to doing 128 base OTs followed by IKNP to derive the total required base OTs.**

| Protocol | Base type | $\lambda$ | $\tau$ | $n$ | LAN (10Gbps) times | $n$ | WAN (100Mbps) times | $n$ | WAN (10Mbps) times |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| | | | | $10^7$ | $10^6$ | $10^5$ | $10^4$ | $10^7$ | $10^6$ | $10^5$ | $10^4$ | $10^7$ | $10^6$ | $10^5$ | $10^4$ |
| This (SH) | hybrid | 128 | 4 | 2,441 | 208 | 76 | 67 | 2,726 | 513 | 422 | 425 | 2,756 | 518 | 454 | 422 |
| IKNP | base | 128 | 4 | 268 | 125 | 94 | 91 | 13,728 | 1,850 | 493 | 459 | 128,954 | 13,332 | 1,756 | 445 |
| This (SH) | hybrid | 128 | 1 | 7,990 | 533 | 130 | 100 | 8,252 | 808 | 451 | 422 | 8,291 | 815 | 467 | 422 |
| IKNP | base | 128 | 1 | 573 | 157 | 108 | 98 | 15,622 | 2,030 | 613 | 341 | 129,011 | 13,285 | 1,672 | 429 |
| This (Mal) | hybrid | 128 | 4 | 2,659 | 280 | 84 | 78 | 2,872 | 479 | 457 | 424 | 2,846 | 515 | 438 | 422 |
| KOS | base | 128 | 4 | 333 | 121 | 110 | 111 | 13,722 | 1,933 | 589 | 426 | 129,052 | 13,391 | 1,804 | 536 |
| This (Mal) | hybrid | 128 | 1 | 8,765 | 584 | 141 | 104 | 9,055 | 828 | 460 | 423 | 8,929 | 831 | 467 | 433 |
| KOS | base | 128 | 1 | 674 | 170 | 113 | 106 | 15,741 | 2,088 | 702 | 433 | 129,771 | 13,389 | 1,772 | 518 |

**Figure 18: The communication overhead of our implementation compared to [37, 41], with $N/n = 2$ and $\lambda = 4$. See Figure 17.**

| Protocol | Base type | $n$ | Total Comm. (bytes) | $n$ | Comm./OT (bits) |
|:---|:---|:---|:---|:---|:---|
| | | $10^7$ | $10^6$ | $10^5$ | $10^4$ | $10^7$ | $10^6$ | $10^5$ | $10^4$ |
| This (SH/Mal) | hybrid | 126,658 | 98,754 | 83,394 | 57,806 | 0.101 | 0.790 | 6.672 | 46.245 |
| IKNP/KOS | base | 160,056,360 | 16,011,518 | 1,655,784 | 168,186 | 128.045 | 128.092 | 132.463 | 134.549 |

# 8 ACKNOWLEDGEMENTS

E. Boyle, N. Gilboa, and Y. Ishai supported by ERC Project NTSC
(742754). E. Boyle additionally supported by ISF grant 1861/16 and
AFOSR Award FA9550-17-1-0069. G. Couteau supported by ERC
Project PREP-CRYPTO (724307). N. Gilboa additionally supported
by ISF grant 1638/15, ERC grant 876110, and a grant by the BGU
Cyber Center. Y. Ishai additionally supported by ISF grant 1709/14,
NSF-BSF grant 2015782, DARPA SPAWAR contract N66001-15-C-
4065, and a grant from the Ministry of Science and Technology,
Israel and Department of Science and Technology, Government of
India. L. Kohl supported by ERC Project PREP-CRYPTO (724307)
and by DFG grant HO 4534/2-2. This work was done in part while
visiting the FACT Center at IDC Herzliya, Israel. P. Scholl supported
by the European Union’s Horizon 2020 research and innovation
programme under grant agreement No 731583 (SODA), and the
Danish Independent Research Council under Grant-ID DFF-6108-
00169 (FoCC).

**REFERENCES**
[1] Afshar, A., Mohassel, P., Pinkas, B., Riva, B.: Non-interactive secure computation
based on cut-and-choose. In: Nguyen, P.Q., Oswald, E. (eds.) EUROCRYPT 2014.
LNCS, vol. 8441, pp. 387–404. Springer, Heidelberg (May 2014)
[2] Aguilar Melchor, C., Aragon, N., Bettaieb, S., Bidoux, L., Blazy, O., Deneuville,
J.C., Gaborit, P., Persichetti, E., Zémor, G.: Hamming quasi-cyclic (HQC) (2019),
https://pqc-hqc.org/doc/hqc-specification_2018-12-14.pdf
[3] Alekhnovich, M.: More on average case vs approximation complexity. In: 44th
FOCS. pp. 298–307. IEEE Computer Society Press (Oct 2003)
[4] Applebaum, B., Damgård, I., Ishai, Y., Nielsen, M., Zichron, L.: Secure arithmetic
computation with constant computational overhead. In: Katz, J., Shacham, H.
(eds.) CRYPTO 2017, Part I. LNCS, vol. 10401, pp. 223–254. Springer, Heidelberg
(Aug 2017)
[5] Applebaum, B., Ishai, Y., Kushilevitz, E.: Cryptography with constant input local-
ity. Journal of Cryptology 22(4), 429–469 (Oct 2009)
[6] Aragon, N., Barreto, P., Bettaieb, S., Bidoux, L., Blazy, O., Deneuville, J.C., Gaborit,
P., Gueron, S., Guneysu, T., Melchor, C.A., et al.: Bike: Bit flipping key encapsula-
tion (2019), https://bikesuite.org/files/round2/spec/BIKE-Spec-2019.06.30.1.pdf
[7] Arora, S., Ge, R.: New algorithms for learning in presence of errors. In: Aceto, L.,
Henzinger, M., Sgall, J. (eds.) ICALP 2011, Part I. LNCS, vol. 6755, pp. 403–415.
Springer, Heidelberg (Jul 2011)
[8] Asharov, G., Lindell, Y., Schneider, T., Zohner, M.: More efficient oblivious transfer
and extensions for faster secure computation. In: Sadeghi, A.R., Gligor, V.D., Yung,
M. (eds.) ACM CCS 2013. pp. 535–548. ACM Press (Nov 2013)
[9] Beaver, D.: Efficient multiparty protocols using circuit randomization. In: Ad-
vances in Cryptology - CRYPTO ’91, 11th Annual International Cryptology
Conference, Santa Barbara, California, USA, August 11–15, 1991, Proceedings. pp.
420–432 (1991), https://doi.org/10.1007/3-540-46766-1_34
[10] Becker, A., Joux, A., May, A., Meurer, A.: Decoding random binary linear codes
in 2n/20: How 1 + 1 = 0 improves information set decoding. In: Pointcheval, D.,
Johansson, T. (eds.) EUROCRYPT 2012. LNCS, vol. 7237, pp. 520–536. Springer,
Heidelberg (Apr 2012)
[11] Bendlin, R., Damgård, I., Orlandi, C., Zakarias, S.: Semi-homomorphic encryption
and multiparty computation. In: Paterson, K.G. (ed.) EUROCRYPT 2011. LNCS,
vol. 6632, pp. 169–188. Springer, Heidelberg (May 2011)
[12] Blum, A., Furst, M.L., Kearns, M.J., Lipton, R.J.: Cryptographic primitives based
on hard learning problems. In: Advances in Cryptology - CRYPTO ’93, 13th
Annual International Cryptology Conference, Santa Barbara, California, USA,
August 22–26, 1993, Proceedings. pp. 278–291 (1993), https://doi.org/10.1007/3-
540-48329-2_24
[13] Blum, A., Kalai, A., Wasserman, H.: Noise-tolerant learning, the parity problem,
and the statistical query model. In: 32nd ACM STOC. pp. 435–440. ACM Press
(May 2000)
[14] Boneh, D., Waters, B.: Constrained pseudorandom functions and their applica-
tions. In: Sako, K., Sarkar, P. (eds.) ASIACRYPT 2013, Part II. LNCS, vol. 8270, pp.
280–300. Springer, Heidelberg (Dec 2013)
[15] Boyle, E., Couteau, G., Gilboa, N., Ishai, Y.: Compressing vector OLE. In: Lie, D.,
Mannan, M., Backes, M., Wang, X. (eds.) ACM CCS 2018. pp. 896–912. ACM Press
(Oct 2018)
[16] Boyle, E., Couteau, G., Gilboa, N., Ishai, Y., Kohl, L., Scholl, P.: Efficient pseudo-
random correlation generators: Silent OT extension and more. In: Boldyreva,
A., Micciancio, D. (eds.) CRYPTO 2019, Part III. LNCS, vol. 11694, pp. 489–518.
Springer, Heidelberg (Aug 2019)
[17] Boyle, E., Gilboa, N., Ishai, Y.: Function secret sharing: Improvements and exten-
sions. In: Weippl, E.R., Katzenbeisser, S., Kruegel, C., Myers, A.C., Halevi, S. (eds.)
ACM CCS 2016. pp. 1292–1303. ACM Press (Oct 2016)
[18] Boyle, E., Goldwasser, S., Ivan, I.: Functional signatures and pseudorandom func-
tions. In: Krawczyk, H. (ed.) PKC 2014. LNCS, vol. 8383, pp. 501–519. Springer,
Heidelberg (Mar 2014)
[19] Canetti, R.: Security and composition of multiparty cryptographic protocols.
Journal of Cryptology 13(1), 143–202 (Jan 2000)
[20] Canetti, R.: Universally composable security: A new paradigm for cryptographic
protocols. In: 42nd FOCS. pp. 136–145. IEEE Computer Society Press (Oct 2001)
[21] Chen, M., Cheng, C., Kuo, P., Li, W., Yang, B.: Multiplying boolean polynomials
with frobenius partitions in additive fast fourier transform. CoRR abs/1803.11301
(2018)
[22] Damgård, I., Pastro, V., Smart, N.P., Zakarias, S.: Multiparty computation from
somewhat homomorphic encryption. In: Safavi-Naini, R., Canetti, R. (eds.)
CRYPTO 2012. LNCS, vol. 7417, pp. 643–662. Springer, Heidelberg (Aug 2012)
[23] Dessouky, G., Koushanfar, F., Sadeghi, A.R., Schneider, T., Zeitouni, S., Zohner, M.:
Pushing the communication barrier in secure computation using lookup tables.
In: NDSS 2017. The Internet Society (Feb / Mar 2017)
[24] Doerner, J., shelat, a.: Scaling ORAM for secure computation. In: Thuraisingham,
B.M., Evans, D., Malkin, T., Xu, D. (eds.) ACM CCS 2017. pp. 523–535. ACM Press
(Oct/Nov 2017)
[25] Döttling, N., Ghosh, S., Nielsen, J.B., Nilges, T., Trifiletti, R.: TinyOLE: Efficient
actively secure two-party computation from oblivious linear function evaluation.
In: Thuraisingham, B.M., Evans, D., Malkin, T., Xu, D. (eds.) ACM CCS 2017. pp.
2263–2276. ACM Press (Oct / Nov 2017)
[26] Druk, E., Ishai, Y.: Linear-time encodable codes meeting the gilbert-varshamov
bound and their cryptographic applications. In: Naor, M. (ed.) ITCS 2014. pp.
169–182. ACM (Jan 2014)
[27] Esser, A., Kübler, R., May, A.: LPN decoded. In: Katz, J., Shacham, H. (eds.)
CRYPTO 2017, Part II. LNCS, vol. 10402, pp. 486–514. Springer, Heidelberg (Aug
2017)
[28] Garg, S., Mahmoody, M., Masny, D., Meckler, I.: On the round complexity of OT
extension. In: Shacham, H., Boldyreva, A. (eds.) CRYPTO 2018, Part III. LNCS,
vol. 10993, pp. 545–574. Springer, Heidelberg (Aug 2018)
[29] Ghosh, S., Nielsen, J.B., Nilges, T.: Maliciously secure oblivious linear func-
tion evaluation with constant overhead. In: Takagi, T., Peyrin, T. (eds.) ASI-
ACRYPT 2017, Part I. LNCS, vol. 10624, pp. 629–659. Springer, Heidelberg (Dec
2017)
[30] Gilboa, N., Ishai, Y.: Distributed point functions and their applications. In: Nguyen,
P.Q., Oswald, E. (eds.) EUROCRYPT 2014. LNCS, vol. 8441, pp. 640–658. Springer,
Heidelberg (May 2014)
[31] Goldreich, O.: Foundations of Cryptography: Volume 2, Basic Applications. Cam-
bridge University Press, New York, NY, USA (2004)
[32] Goldreich, O., Goldwasser, S., Micali, S.: How to construct random functions.
Journal of the ACM 33(4), 792–807 (Oct 1986)
[33] Goldreich, O., Micali, S., Wigderson, A.: How to play any mental game or A
completeness theorem for protocols with honest majority. In: Aho, A. (ed.) 19th
ACM STOC. pp. 218–229. ACM Press (May 1987)
[34] Hazay, C., Orsini, E., Scholl, P., Soria-Vazquez, E.: Concretely efficient large-scale
MPC with active security (or, TinyKeys for TinyOT). In: Peyrin, T., Galbraith, S.
(eds.) ASIACRYPT 2018, Part III. LNCS, vol. 11274, pp. 86–117. Springer, Heidel-
berg (Dec 2018)
[35] Hazay, C., Orsini, E., Scholl, P., Soria-Vazquez, E.: TinyKeys: A new approach
to efficient multi-party computation. In: Shacham, H., Boldyreva, A. (eds.)
CRYPTO 2018, Part III. LNCS, vol. 10993, pp. 3–33. Springer, Heidelberg (Aug
2018)
[36] Huang, Y., Evans, D., Katz, J.: Private set intersection: Are garbled circuits better
than custom protocols? In: NDSS 2012. The Internet Society (Feb 2012)
[37] Ishai, Y., Kilian, J., Nissim, K., Petrank, E.: Extending oblivious transfers effi-
ciently. In: Boneh, D. (ed.) CRYPTO 2003. LNCS, vol. 2729, pp. 145–161. Springer,
Heidelberg (Aug 2003)
[38] Ishai, Y., Kushilevitz, E., Ostrovsky, R., Prabhakaran, M., Sahai, A.: Efficient non-
interactive secure computation. In: Paterson, K.G. (ed.) EUROCRYPT 2011. LNCS,
vol. 6632, pp. 406–425. Springer, Heidelberg (May 2011)
[39] Ishai, Y., Prabhakaran, M., Sahai, A.: Secure arithmetic computation with no
honest majority. In: Reingold, O. (ed.) TCC 2009. LNCS, vol. 5444, pp. 294–314.
Springer, Heidelberg (Mar 2009)
[40] Katz, J., Ranellucci, S., Rosulek, M., Wang, X.: Optimizing authenticated garbling
for faster secure two-party computation. In: Advances in Cryptology - CRYPTO
2018 - 38th Annual International Cryptology Conference, Santa Barbara, CA,
USA, August 19–23, 2018, Proceedings, Part III. pp. 365–391 (2018), https://doi.
org/10.1007/978-3-319-96878-0_13
[41] Keller, M., Orsini, E., Scholl, P.: Actively secure OT extension with optimal over-
head. In: Gennaro, R., Robshaw, M.J.B. (eds.) CRYPTO 2015, Part I. LNCS, vol.
9215, pp. 724–741. Springer, Heidelberg (Aug 2015)
[42] Kiayias, A., Papadopoulos, S., Triandopoulos, N., Zacharias, T.: Delegatable pseu-
dorandom functions and applications. In: Sadeghi, A.R., Gligor, V.D., Yung, M.
(eds.) ACM CCS 2013. pp. 669–684. ACM Press (Nov 2013)
[43] Kilian, J.: Founding cryptography on oblivious transfer. In: Proceedings of the
20th Annual ACM Symposium on Theory of Computing, May 2-4, 1988, Chicago,
Illinois, USA. pp. 20–31 (1988), https://doi.org/10.1145/62212.62215
[44] Kolesnikov, V., Kumaresan, R.: Improved OT extension for transferring short
secrets. In: Canetti, R., Garay, J.A. (eds.) CRYPTO 2013, Part II. LNCS, vol. 8043,
pp. 54–70. Springer, Heidelberg (Aug 2013)
[45] Lyubashevsky, V.: The parity problem in the presence of noise, decoding random
linear codes, and the subset sum problem. In: Approximation, randomization and
combinatorial optimization. Algorithms and techniques, pp. 378–389. Springer
(2005)
[46] Mohassel, P., Rosulek, M.: Non-interactive secure 2PC in the offline/online and
batch settings. In: Coron, J., Nielsen, J.B. (eds.) EUROCRYPT 2017, Part III. LNCS,
vol. 10212, pp. 425–455. Springer, Heidelberg (Apr / May 2017)
[47] Naor, M., Pinkas, B.: Computationally secure oblivious transfer. Journal of Cryp-
tology 18(1), 1–35 (Jan 2005)
[48] Naor, M., Pinkas, B.: Oblivious polynomial evaluation. SIAM J. Comput. 35(5),
1254–1281 (2006)
[49] Peikert, C., Vaikuntanathan, V., Waters, B.: A framework for efficient and com-
posable oblivious transfer. In: Wagner, D. (ed.) CRYPTO 2008. LNCS, vol. 5157,
pp. 554–571. Springer, Heidelberg (Aug 2008)
[50] Pinkas, B., Schneider, T., Segev, G., Zohner, M.: Phasing: Private set intersection
using permutation-based hashing. In: Jung, J., Holz, T. (eds.) USENIX Security
2015. pp. 515–530. USENIX Association (Aug 2015)
[51] Pinkas, B., Schneider, T., Tkachenko, O., Yanai, A.: Efficient circuit-based psi with
linear communication (2019), https://eprint.iacr.org/2019/241
[52] Prange, E.: The use of information sets in decoding cyclic codes. IRE Transactions
on Information Theory 8(5), 5–9 (1962)
[53] Sendrier, N.: Decoding one out of many. In: Yang, B.Y. (ed.) Post-Quantum Cryp-
tography 4th International Workshop, PQCrypto 2011. pp. 51–67. Springer,
Heidelberg (Nov / Dec 2011)
[54] Torres, R.C., Sendrier, N.: Analysis of information set decoding for a sub-linear
error weight. In: Takagi, T. (ed.) Post-Quantum Cryptography - 7th International
Workshop, PQCrypto 2016. pp. 144–161. Springer, Heidelberg (2016)
[55] Wang, X., Ranellucci, S., Katz, J.: Authenticated garbling and efficient maliciously
secure two-party computation. In: Thuraisingham, B.M., Evans, D., Malkin, T.,
Xu, D. (eds.) ACM CCS 2017. pp. 21–37. ACM Press (Oct / Nov 2017)
[56] Wang, X., Ranellucci, S., Katz, J.: Global-scale secure multiparty computation.
In: Thuraisingham, B.M., Evans, D., Malkin, T., Xu, D. (eds.) ACM CCS 2017. pp.
39–56. ACM Press (Oct / Nov 2017)
[57] Zichron, L.: Locally computable arithmetic pseudorandom generators. Master’s
thesis, School of Electrical Engineering, Tel Aviv University (2017), http://www.
eng.tau.ac.il/~bennyap/pubs/Zichron.pdf