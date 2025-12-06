1962
IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 58, NO. 3, MARCH 2012

# Unconditional Security From Noisy Quantum Storage
Robert König, Stephanie Wehner, and Jürg Wullschleger

**Abstract**—We consider the implementation of two-party crypto-
graphic primitives based on the sole assumption that no large-scale
reliable quantum storage is available to the cheating party. We con-
struct novel protocols for oblivious transfer and bit commitment,
and prove that realistic noise levels provide security even against
the most general attack. Such unconditional results were previ-
ously only known in the so-called bounded-storage model which
is a special case of our setting. Our protocols can be implemented
with present-day hardware used for quantum key distribution. In
particular, no quantum storage is required for the honest parties.

**Index Terms**—Cryptography, quantum computing.

# I. NOISY-STORAGE MODEL: DEFINITION AND RESULTS

## A. Motivation: Security From Physical Assumptions

The security of most cryptographic systems currently in
use is based on the premise that a certain computational
problem is hard to solve for the adversary. Concretely, this relies
on the assumption that the adversary's computational resources
are limited, and the underlying problem is hard in some pre-
cise complexity-theoretic sense. While the former assumption
may be justified in practice, the latter statement is usually an
unproven mathematical conjecture. In contrast, quantum cryp-
tographic schemes are designed in such a way that they provide
security based solely on the validity of quantum physics. No as-
sumptions on the adversary's computational power nor the va-
lidity of some complexity-theoretic statements are needed.
Unfortunately, not even the laws of quantum physics allow us
to realize all desirable cryptographic functionalities without fur-
ther assumptions [37], [38], [39], [44], [45]. An example of such
a functionality is (fully randomized) oblivious transfer, where
Alice receives two random strings $S_0, S_1$, while Bob receives
one of the strings $S_C$ together with the index $C$. Security for this
primitive means that neither Alice nor Bob can obtain any infor-
mation beyond this specification. A protocol which securely im-
plements oblivious transfer is desirable because any two-party
computation, such as secure identification, can be based on this
building block [23], [29].

In light of this state of affairs, it is natural to consider other
physical assumptions: motivated by similar classical models
[42], [43], the authors in [15], [16] and [54], [59], [61] propose
to assume that the adversary's quantum storage is bounded
and noisy, respectively. The assumption of bounded quantum
storage deals with the noiseless case (but assumes a small
amount of storage), whereas the noisy-storage model deals with
the case of noise (but possibly a large amount of storage). Here,
we introduce a more general point of view which incorporates
both the amount of storage and noise. We refer to this simply
as the **noisy-storage model**. The previously considered settings
are special cases, as we will explain in the following.
Compared to the classical world, the assumption of limited
and noisy quantum storage is particularly realistic in view of
the present state of the art, and the considerable challenges
faced when trying to build scalable quantum memories. Indeed,
it is unknown whether it is physically possible to build noise
free memories. Further motivation for considering noise as a
resource for security over the mere assumption of bounded
storage comes from the fact that the transfer of the state of
a (photonic) qubit used during the execution of the protocol
onto a different carrier used as a quantum memory (such as an
atomic ensemble) is typically already noisy.

## B. Contribution and Methods

We consider the noisy-storage model which was previously
introduced in [54], [59], and [61] where it appeared in a slightly
more specialized form. All previous security proofs in this
model required additional assumptions beyond having noise. In
particular, in the analysis of [61], the adversary was restricted
to performing individual attacks using product measurements
on the qubits received in the protocol. This is a significant
restriction as multiqubit measurements are possible even today,
and can be compared to an analysis of quantum key distribution
[4] where the eavesdropper is restricted to measuring each
qubit individually. We provide a fully general proof of security
against arbitrary attacks that bit commitment and oblivious
transfer can be achieved in the general noisy-storage model.
This encompasses and extends all previously considered set-
tings [15], [16], [54], [59], [61]. As a side effect, we also
obtain significantly improved parameters for the special case
of bounded storage.

In order to obtain this result, we require a number of methods
that have not been used before either in the noisy- or bounded-
quantum-storage setting.

1) We formally relate the security of our protocols to the
problem of sending information through the noisy-storage
channel. This is very intuitive, and much more natural than
previous approaches such as the restriction to individual
attacks in the noisy-storage model [61], or the assump-
tion of bounded storage [16]. More specifically, we show
that a sufficient condition for security is that the number
of classical bits that can be sent through the noisy-storage
channel is limited. We introduce our generalized model in
Section I-C, and state our result in Section I-D.
2) We introduce a novel cryptographic primitive called weak
string erasure (WSE, see Section III) that may be of inde-
pendent interest. We provide a simple quantum protocol
that securely realizes WSE in the noisy-storage model,
in which the honest parties do not require any quantum
memory at all to execute the protocol. Our protocol can be
implemented with present-day technology. In our security
proof, we require information-theoretic tools such as the
recently proven strong converse for channel coding [36].
3) We construct new protocols for bit commitment and obliv-
ious transfer based on WSE, and prove security against ar-
bitrary attacks. Our protocols are purely classical, merely
using the simple quantum primitive of WSE which is a con-
ceptually appealing feature. We make use of various tech-
niques such as error-correcting codes, privacy amplifica-
tion, interactive hashing (IH), and min-entropy sampling
with respect to a quantum adversary.

Our work raises many immediate open questions and has al-
ready led to follow up work which we discuss in Section VI.

## C. Noisy-Storage Model

Let us now describe more formally what we mean by a noisy
quantum memory. We think of a device whose input states are
in some Hilbert space $H_{\text{in}}$. A state $\rho$ stored in the device deco-
heres over time. That is, the content of the memory after some
time $t$ is a state $F_t(\rho)$, where $F_t : B(H_{\text{in}}) \rightarrow B(H_{\text{out}})$ is a com-
pletely positive trace-preserving map (CPTPM) corresponding
to the noise in the memory. Since the amount of noise may of
course depend on the storage time, the behavior of the storage is
completely described by the family of maps $\{F_t\}_{t>0}$. We will
make the minimal assumption that the noise is Markovian, that
is, the family $\{F_t\}_{t>0}$ is a continuous one-parameter semigroup

$$
F_0 = \mathbb{1} \quad \text{and} \quad F_{t_1+t_2} = F_{t_1} \circ F_{t_2} \quad (1)
$$

This tells us that the noise in storage only increases with time,
and is essential to ensure that the adversary cannot gain any in-
formation by delaying the readout¹. This is the only restriction
imposed on the adversary who may otherwise be all-powerful.
In particular, we allow that all his actions are instantaneous, in-
cluding computation, communication, measurement, and state
preparation.

How can we hope to obtain security in such a model? In our
protocol, we will introduce certain time delays $\Delta t$ which force
any adversary to use his storage device for a time at least $\Delta t$.
Our assumptions imply that the best an adversary can do is to
read out the information from the device immediately after time
$\Delta t$, as any further delay will only degrade his information fur-
ther. We can, thus, focus on the channel $F = F_{\Delta t}$ when ana-
lyzing security instead of the family $\{F_t\}_{t>0}$. Note that since
the adversary's actions are assumed to be instantaneous, he can
use any error-correcting code even if the best encoding and de-
coding procedure may be difficult to perform. Summarizing, our
model assumes the following.
1) The adversary has unlimited classical storage and
(quantum) computational resources.
2) Whenever the protocol requires the adversary to wait for a
time $\Delta t$, he has to measure/discard all his quantum infor-
mation except what he can encode (arbitrarily) into $H_{\text{in}}$.
This information then undergoes noise described by $F$.
To see how previously analyzed cases fit into our model, note
that the bounded-storage model (BSM) corresponds to the case
where $H_{\text{in}}$ is of limited input dimension, and $F$ is the identity on
$H_{\text{in}}$. Concretely, Damgård et al. [15] consider protocols with $n$
qubits of communication and $H_{\text{in}} \simeq (\mathbb{C}^2)^{\otimes vn}$ for some param-
eter $v > 0$ which we call the storage rate. Security of certain
protocols was established for $v < 1/4$. Furthermore, the pro-
tocol proposed by Crépeau [11] for oblivious transfer is secure
if the adversary cannot store any quantum information at all,
corresponding to a storage rate of $v = 0$. Previous work on the
noisy-storage model [61] analyzed protocols with $n$ qubits of
communication, where the noise $F = N^{\otimes n}$ is an $n$-fold tensor
product of a noisy single-qubit channel $N : B(\mathbb{C}^2) \rightarrow B(\mathbb{C}^2)$
(i.e., $H_{\text{in}} \simeq (\mathbb{C}^2)^{\otimes n}$ and $v = 1$). Note, however, that in [61],
the adversary was further restricted to performing product mea-
surements on the qubits received in the protocol (albeit other-
wise fully arbitrary).

## D. Main Result

We now state our main result of establishing security in the
noisy-storage model against fully general attacks for arbitrary
channels $F: B(H_{\text{in}}) \rightarrow B(H_{\text{out}})$. As explained, we form
a very natural relation between the security of our protocols
and the problem of transmitting information through the
noisy-storage channel². More specifically, we prove that a suf-
ficient condition for security is that the number of classical bits
that can be sent through the noisy-storage channel is limited.

As usual in cryptography, we would like to compare the ad-
versary's resources to those of the honest parties and/or the com-
plexity of operations used in the protocol. Here, we parame-
terize these by the number $n$ of qubits transmitted during the
protocol. For the adversary's storage, we therefore consider a
family $\{F\}_n$ of storage devices. The quality of the adversary's
storage can then be measured (for a fixed $n$) by the following
operational quantity: the success probability of correctly trans-
mitting a randomly chosen $nR$-bit string $x \in \{0, 1\}^{nR}$ through
the storage device $F$, which can be written as

$$
P_{\text{succ}}^F(nR) := \max_{\{D_x\}, \{\rho_x\}} \frac{1}{2^{nR}} \sum_{x \in \{0, 1\}^{nR}} \text{tr}(D_x F(\rho_x)) \quad (2)
$$

where the maximum is taken over families of code states
$\{\rho_x\}_{x \in \{0, 1\}^{nR}}$ on $H_{\text{in}}$ and decoding positive operator valued
measures (POVMs) $\{D_x\}_{x \in \{0, 1\}^{nR}}$ on $H_{\text{out}}$. We show that
security can be obtained for arbitrary channels with the prop-
erty that the decoding probability decays exponentially above a
certain threshold.

**Theorem I.1 (Informal Statement):** Suppose that for the
family of channels $\{F\}_n$ and the constant $0 < R < 1/2$ there
exist constants $n_0 > 0$ and $\gamma > 0$ such that for all $n \geq n_0$ the
decoding probability satisfies

$$
P_{\text{succ}}^F(nR) \leq 2^{-n \gamma n} \quad (3)
$$

Then, oblivious transfer and bit commitment can be im-
plemented using $O(n)$ qubits of communication against an
adversary whose noisy storage is described by the family $\{F\}_n$.
Moreover, the security is exponential in $n$.

Remarkably, the statement of Theorem I.1 does not require
any knowledge of the channel $F$ beyond its relation to the
coding problem. In particular, we do not need to assume that
$F$ is of tensor product form. This includes, for example, the
practically interesting case where errors are likely to occur
in bursts in the storage device, or the noisy channel itself has
memory. We discuss possible extensions and limitations of our
approach in Section VI. We point out that the length of the
input strings used in oblivious transfer and bit commitment per
communicated qubit depends on the exponent $\gamma$ in (3); this is
hidden in the constant in the $O$-expression in Theorem I.1.
Determining the constant $\gamma$ is of course no easy task for arbi-
trary storage devices. To obtain explicit security parameters, we
thus proceed to consider the special case where the channels are
of the form $F = N^{\otimes vn}$ where $n$ is the number of qubits sent
in the protocol, and $v \geq 0$ is the storage rate. Our proof con-
nects the security of protocols in the noisy-storage model for
such channels to the classical capacity $C_N$ of $N$. This provides
a quantitative expression of our intuition that noisy channels
which are of little use for classical information transmission give
rise to security in the noisy-storage model. First of all, observe
that there can only exist a constant $\gamma > 0$ leading to the expo-
nential decay of (3) if the classical capacity $C_N$ of the channel
is strictly smaller than the rate $R$ at which we send informa-
tion through the channel. This, however, is not sufficient, since
$R > C_N$ is not generally known to imply (3) for $F = N^{\otimes vn}$.
We are, therefore, interested in channels $N$ which satisfy the
following **strong-converse property**. The success probability (2)
decays exponentially for rates $R$ above the capacity, i.e., it takes
the form

$$
P_{\text{succ}}^{N^{\otimes n}}(nR) \leq 2^{-n \gamma_N(R)} \quad \text{where} \quad \gamma_N(R) > 0 \quad \text{for all} \quad R > C_N \quad (4)
$$

In [36], property (4) was shown to hold for a large class of
channels, including the depolarizing channel [see (5) in the fol-
lowing]. It was also shown how to compute $\gamma_N(R)$. Combining
Theorem I.1 with (4), we obtain the following statement.

[IMAGE: Fig. 1. Our results applied to depolarizing noise $F = N_r^{\otimes \nu n}$. The vertical axis represents the noise parameter r, while the horizontal axis represents the storage rate $\nu$. Our protocols are secure when the pair $(r, \nu)$ is in the lower region bounded by the solid blue curve. Security is still possible in the region labeled with '?', but cannot be obtained from our analysis.]

**Corollary I.2 (Informal Statement):** Let $\nu \geq 0$, and suppose
that $N$ satisfies the strong-converse property (4). If

$$
C_N \cdot \nu < \frac{1}{2}
$$

then oblivious transfer and bit commitment can be imple-
mented with polynomial resources (in $n$) and exponential
security against an adversary with noisy storage $F = N^{\otimes \nu n}$.
For the special case of bounded (noise free) qubit storage
($C_N = 1$), this gives security for $\nu < 1/2$.

An important example for which we obtain security is the
$d$-dimensional depolarizing channel $N_r: B(\mathbb{C}^d) \rightarrow B(\mathbb{C}^d)$
defined for $d \geq 2$ as

$$
N_r(\rho) := r \rho + (1 - r) \frac{\mathbb{1}}{d} \quad \text{for some fixed } 0 \leq r \leq 1 \quad (5)
$$

which replaces the input state $\rho$ with the completely mixed state
with probability $1 - r$. For $d = 2$, this means that the adversary
can store $\nu n$ qubits, which are affected by independent and iden-
tically distributed noise. It has been shown that the depolarizing
channel exhibits the strong-converse property [36]. To see for
which values of $r$ we can obtain security, we need to consider
the classical capacity of the depolarizing channel as evaluated
by King [31]. For $d = 2$, i.e., qubits, it is given by

$$
C_N = 1 + \frac{1 + r}{2} \log \frac{1 + r}{2} + \frac{1 - r}{2} \log \frac{1 - r}{2}
$$

Fig. 1 shows the region in the $(r, \nu)$ plane corresponding to
the noise channel $F = N^{\otimes \nu n}$, where we allow $n$ qubits of com-
munication in the protocol. This is obtained from Corollary I.2
(the depolarizing channel $N_r$ satisfies the corresponding condi-
tions).

**Comparison to the BSM: Depolarizing Noise:** It was previ-
ously observed [52] that the case of depolarizing storage noise
(i.e., $r < 1$) can be dealt with using results obtained in the
BSM (i.e., $r = 1$) when the noise is sufficiently strong. More
precisely, the results in [15] can be extended to give nontrivial
statements if the “effective” dimension of the storage system is
less than $n/4$, where $n$ is the number of qubits communicated
in the protocol⁴. We sketch such a simple dimensional analysis
to illustrate that our model offers significant improvements over
the bounded-storage analysis: we obtain security even at lower
noise levels and higher storage rates.

Concretely, consider the noise channel $F = N_r^{\otimes \nu n} :$
$B((\mathbb{C}^2)^{\otimes \nu n}) \rightarrow B((\mathbb{C}^2)^{\otimes \nu n})$ (cf., (5) for $d = 2$). Applying
depolarizing noise to any of the $\nu n$ systems $\mathbb{C}^2$ means that the
state on this system is replaced by the completely mixed state
with probability $1 - r$. We can think of an indicator random
variable $E^{\nu n} = (E_1, \ldots, E_{\nu n}) \in \{0, 1\}^{\nu n}$, where $E_i$ is $1$ if
and only if the $i$-th qubit is replaced by the completely mixed
state. These “erasure” variables are independent and identically
distributed Bernoulli variables with parameter $r = P_E(0)$. In
particular, the number of erasures

$$
|E^{\nu n}| = \sum_{i=1}^{\nu n} E_i
$$

is distributed according to the binomial distribution with $\nu n$
trials, each of which succeeds with probability $1 - r$.
We now assume that the adversary is given the location of the
erasures $E^{\nu n}$ in addition to the output of the channel. Note that
this can only make the adversary more powerful. Conditioned
on the locations $E^{\nu n}$, the “effective dimension” of his channel
is equal to $2^{\nu n - |E^{\nu n}|}$. Hence, we may think of an “effective”
storage rate $v_{\text{eff}}$ given by the random variable

$$
v_{\text{eff}} = \nu \frac{|E^{\nu n}|}{n}
$$

We know from the BSM analysis [15] that for $v_{\text{eff}} < 1/4$, the pre-
viously studied protocols provide security. Overall, we therefore
conclude that security can be obtained from the noisy channel $F$
if $\text{Pr}[v_{\text{eff}} > 1/4]$ is exponentially small. Note that by Chernoff’s
inequality

$$
\text{Pr} [v_{\text{eff}} > 1/4] = \text{Pr} [|E^{\nu n}| < (1 - \delta) \mu] < e^{-\mu \delta^2/2}
$$

if $\delta = \frac{1 - 4 \nu r}{4(1 - r)} > 0$, where $\mu = n \nu (1 - r)$
In particular, we conclude that we obtain security for

$$
\nu r < \frac{1}{4} \quad (6)
$$

[IMAGE: Fig. 2. Security for depolarizing noise parameters $(1, \nu)$ with $\nu < 1/4$ was established in the BSM. Our simple argument and our more refined protocols and analysis give significantly improved parameters of $\nu < 1/2$ for the BSM, for which the same argument extends security to the region bounded by the green dot-dashed curve. However, our study still extends this region even further by considering noisy instead of merely bounded storage (solid blue curve). We stress that such a naïve dimensional analysis does not apply to other channels (such as the two-Pauli channel), while our more refined analysis gives results even in such cases.]

Fig. 2 compares the curve of this equation (6) to the results we
will derive later. We see that for the noiseless case $(r = 1)$, our
analysis provides security for storage rates $v < 1/2$, extending
previous results (i.e., $\nu < 1/4$ in [15]) in the BSM. This im-
provement stems from the fact that (for oblivious transfer) our
protocol uses a different classical postprocessing based on IH
instead of the min-entropy splitting tool in [15]. Note that this
requires additional rounds of classical communication.

One may wonder whether a security proof may alternatively
be obtained based on the idea of simulating the storage noise
$F = N_r^{\otimes \nu n}$ using a limited number of qubits. For channels
without memory, the quantum reverse Shannon theorem [5]
shows us that $F$ can be simulated using a certain number of
(noise-free) qubits when the sender and receiver share entan-
glement. Hence, the total size of the system consisting of the
noise-free qubits and the entanglement is rather large. However,
as explained in [5], the theorem implies an exponential decay
of the decoding probability as in (4), but only for rates $R$
greater than the **entanglement-assisted capacity** of the channel
$N$. Our security results, thus, extend to this regime by our new
analysis. The fact that the entanglement-assisted capacity is
generally greater than the unassisted one suggests that such a
simulation-based approach is suboptimal: we are essentially
overestimating the adversary's capabilities by allowing him to
use (noise-free, time-like) entanglement.

Let us give a simple concrete example that provides some in-
tuition on why bounding the adversary's information by the size
of his storage device is generally undesirable. Imagine that the
adversary's channel $F$ replaces the $n$ input qubits by a fixed
state with overwhelmingly high probability and leaves the input
untouched with negligible probability $2^{-n}$. Clearly, the number
of noise-free qubits required to simulate this channel is equal
to $n$, yet the adversary's decoding probability will be exponen-
tially small. Simply bounding the adversary's information gain
in terms of his storage as in the bounded-storage analysis [15],
therefore, significantly overestimates his abilities.⁵

## E. Techniques: WSE

Before describing our protocols and proving Theorem I.1, we
give a short overview of the techniques involved.

First, we introduce a primitive called WSE, which may be of
independent interest. Our protocols for oblivious transfer and
bit commitment are then based on this primitive. WSE provides
Alice with a random bit string $X^n \in \{0, 1\}^n$, while Bob re-
ceives a randomly chosen substring $X_I = (X_{i_1}, \ldots, X_{i_r})$, to-
gether with the index set $I = \{i_1, \ldots, i_r\}$ specifying the loca-
tion of these bits. Security of WSE roughly means that Bob will
remain ignorant about a significant amount of information about
$X^n$, while security against Alice means that she does not learn
anything about $I$ (for a precise definition, we defer the reader
to Section III).

We provide a protocol for WSE in the noisy-storage model.
This protocol can be implemented with present-day hardware
used for quantum key distribution. In particular, it does not re-
quire the honest parties to have any form of quantum memory.
We prove security of this protocol for channels $F$ as stated in
Theorem I.1. Security against (even an all-powerful) Alice fol-
lows from the fact that the protocol only involves one-way com-
munication from Alice to Bob. The security analysis in the pres-
ence of a malicious Bob limited by storage noise $F$ is more in-
volved. Our proof combines an entropic uncertainty relation in-
volving postmeasurement information [1], [15] with a reformu-
lation of the problem as a coding scheme: essentially, the uncer-
tainty relation implies that with high probability (over measure-
ment outcomes), Bob's classical information about $X^n$ before
using his storage is limited. We then show that this implies that
any successful attacker Bob needs to encode classical informa-
tion at a high rate into his storage device. However, the assumed
noisiness of $F$ precludes this.

Having built a protocol for WSE, we proceed to present pro-
tocols for bit commitment and oblivious transfer. The case of
bit commitment is particularly appealing. It is essentially only
based on WSE and a classical code, and requires little additional
analysis. Our approach to realizing oblivious transfer is some-
what more involved. Here, WSE is combined with a technique
called IH [51]. The output of IH is a pair of substrings of $X$, one
of which is completely known to Bob, while he only has partial
knowledge about the other. Privacy amplification [50] is then
used to extract completely random bits. The security analysis of
this protocol requires the use of entropy sampling with respect
to a quantum adversary [32].

As a side remark, note that Kilian [29] showed that oblivious
transfer is universal for secure two-party computation. In par-
ticular, bit commitment could be built from oblivious transfer,
but this reduction is generally inefficient.

# II. TOOLS

We briefly introduce all necessary notation as well as several
important concepts that we will need throughout this paper.
For WSE, we require the notion of min-entropy (see Sec-
tion II-B1), uncertainty relations (Section II-B3), as well as an
understanding of how storage noise leads to information loss
for the cheating party (Section II-C). In our protocols for bit
commitment and oblivious transfer from WSE, we additionally
require the concepts of smooth min-entropy (see Section II-B2)
and secure keys (see Section II-B4), respectively, and a number
of tools, namely privacy amplification (see Section II-D1),
sampling of min-entropy (see Section II-D2), and finally IH
(see Section II-D4).

## A. Notation

For an integer $n$, let $[n] := \{1, \ldots, n\}$. We use $2^{[n]} :=$
$\{S \mid S \subseteq [n]\}$ to refer to the set of all possible subsets of $[n]$,
including the empty set ($\emptyset$). For an $n$-tuple $x^n = (x_1, \ldots, x_n) \in$
over a set $\mathcal{X}$ and a (non-empty) set $I = \{i_1, \ldots, i_r\} \in 2^{[n]}$,
we write $x_I$ for the subtuple $x_I = (x_{i_1}, \ldots, x_{i_r}) \in \mathcal{X}^{|I|}$.

We use upper case letters to denote a random variable $X$ dis-
tributed according to a distribution $P_X$ over a set $\mathcal{X}$, and use
lower case letters $x$ for elements $x \in \mathcal{X}$. Joint distributions of,
e.g., three random variables $(X, Y, Z)$ on $\mathcal{X} \times \mathcal{Y} \times \mathcal{Z}$ are de-
noted by $P_{XYZ}$. Given a function $f : \mathcal{X} \rightarrow \mathcal{Y}$, any distribu-
tion $P_X$ of a random variable $X$ gives rise to another jointly
distributed random variable $Y = f(X)$. The joint distribution
$P_{XY} = P_{X f(X)}$ is given by

$$
P_{X f(X)}(x, y) = P_X(x) \delta_{f(x), y} \quad (7)
$$

where $\delta_{i, j}$ is the Kronecker symbol. An important example is the
case where $X^n \in \{0, 1\}^n$ is a random bitstring and $I \in 2^{[n]}$ is
a random subset of $[n]$, where $X^n$ and $I$ have joint distribution
$P_{X^n I}$. In this case, the joint distribution $P_{X^n I X_I} = P_{X^n I X_I}$
describes, e.g., a situation where some bits $Z = X_I$ of a string
$X^n$ are given, together with a specification $I$ of where these bits
are located in the original string.

We use $B(\mathcal{H})$ to denote the set of bounded operators on a
Hilbert space $\mathcal{H}$. A (quantum) state is a Hermitian operator $\rho \in$
$B(\mathcal{H})$ satisfying $\text{tr}(\rho) = 1$ and $\rho \geq 0$. Quantum states can be
used to encode classical probability distributions: for a (finite)
set $\mathcal{X}$, we fix a Hilbert space $H_{\mathcal{X}} \simeq \mathbb{C}^{|\mathcal{X}|}$ and an orthonormal
basis $\{|x\rangle \mid x \in \mathcal{X}\}$ of $H_{\mathcal{X}}$. This will be referred to as the
**computational basis**. A probability distribution $P_X$ on $\mathcal{X}$ can
then be encoded into the classical state (c-state)

$$
\rho_X = \sum_{x \in \mathcal{X}} P_X(x) |x\rangle \langle x|
$$

Of particular interest is the uniform distribution over $\mathcal{X}$, which
gives rise to the completely mixed state on $H_{\mathcal{X}}$ denoted by the
shorthand

$$
\tau_{\mathcal{X}} := \frac{1}{|\mathcal{X}|} \sum_{x \in \mathcal{X}} |x\rangle \langle x|
$$

States describing classical information (random variables)
and truly quantum information, simultaneously, are termed
**classical-quantum** or **cq-states**. They are described by bipartite
systems, where the classical part of the state is diagonal with
respect to the computational basis. Concretely, let $H_Q$ be an
additional Hilbert space. A state $\rho_{XQ}$ on $H_{\mathcal{X}} \otimes H_Q$ is a cq-state
if it has the form

$$
\rho_{XQ} = \sum_{x \in \mathcal{X}} P_X(x) |x\rangle \langle x| \otimes \rho_Q^x \quad (8)
$$

In other words, such a state $\rho_{XQ}$ encodes an ensemble of states
$\{P_X(x), \rho_Q^x\}_{x \in \mathcal{X}}$ on $H_Q$, where $\rho_Q^x$ is the conditional state on
$Q$ given $X = x$. The notion of cq-states directly generalizes
to multipartite systems, where classical parts are diagonal with
respect to the computational basis. We often fix an ordering of
the multipartite parts, and indicate by $c$ or $q$ whether a part is
classical or quantum. We can also apply functions to classical
parts as before. For a function $f : \mathcal{X} \rightarrow \mathcal{Y}$

$$
\rho_{X f(X) Q} = \sum_{x \in \mathcal{X}, y \in \mathcal{Y}} P_{X f(X)}(x, y) |x\rangle \langle x| \otimes |y\rangle \langle y| \otimes \rho_Q^{x} \quad (9)
$$

is the **ccq-state** encoding the pair $(X, f(X))$ of classical random
variables (cc) distributed according to (7) as well as the quantum
information $Q$ (q) (which depends only on $X$ in this case). Note
that in (9), the systems on the right-hand side are uniquely de-
termined by the expression on the left-hand side. We will, there-
fore, omit the braces below. Given a state $\rho_{Q_1 Q_2}$ on systems $Q_1$
and $Q_2$, we also use $\rho_{Q_1} = \text{tr}_{Q_2}(\rho_{Q_1 Q_2})$ to denote the state ob-
tained by tracing out $Q_2$.

The Hadamard transform is the unitary described by the ma-
trix

$$
H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
$$

in the computational basis $\{|0\rangle, |1\rangle\}$ of the qubit Hilbert space
$\mathbb{C}^2$. For the $n$-qubit Hilbert space, we let

$$
H^{\Theta^n} |x^n\rangle := H^{\theta_1} |x_1\rangle \otimes \cdots \otimes H^{\theta_n} |x_n\rangle
$$

for $x^n = (x_1, \ldots, x_n), \Theta^n = (\theta_1, \ldots, \theta_n) \in \{0, 1\}^n$
We also call states of this form **BB84-states**.

Finally, we need a distance measure for quantum states on a
Hilbert space $\mathcal{H}$. We use the distance determined by the trace
norm $\|A\|_1 := \text{tr}\sqrt{A^{\dagger} A}$ for bounded operators $A \in B(\mathcal{H})$. We
will say that two states $\rho, \sigma \in B(\mathcal{H})$ are $\varepsilon$-close if $\|\rho - \sigma\|_1 \leq$
$\varepsilon$, which we also write as

$$
\rho \approx_{\varepsilon} \sigma
$$

## B. Quantifying Adversarial Information

**Min-Entropy and Measurements:** One of the main properties
of the WSE primitive is that the adversary's (quantum) infor-
mation $Q$ about the generated bit string $X$ is limited. To make
this statement precise, we first need to introduce an appropriate
measure of information. Throughout, we are interested in the
case where the adversary holds some (possibly quantum) infor-
mation $Q$ about a classical random variable $X$. This situation is
described by a cq-state $\rho_{XQ}$ as in (8). A natural measure for the
amount of information $Q$ gives about $X$ is the maximal average
success probability that a party holding $Q$ has in guessing the
value of $X$. For a given cq-state $\rho_{XQ}$, this guessing probability
can be written as

$$
P_{\text{guess}}(X|Q) := \max_{\{D_x\}_x} \sum_x P_X(x) \text{tr}(D_x \rho_Q^x) \quad (10)
$$

where the maximization is over all POVMs $\{D_x\}_{x \in \mathcal{X}}$ on $H_Q$. It
will be convenient to turn (10) into an conditional entropy-like
quantity, called the **min-entropy**, which is given by⁶

$$
H_{\infty}(X|Q) := -\log P_{\text{guess}}(X|Q) \quad (11)
$$

Note that the min-entropy was originally defined [49] for arbi-
trary bipartite states $\rho_{AB}$, as we will discuss in more detail in
the following.

As an illustrative, yet important, example consider the fol-
lowing ccq-state on $H_X \otimes H_\Theta \otimes H_Q \simeq (\mathbb{C}^2)^{\otimes 3}$

$$
\rho_{X \Theta Q} = \frac{1}{4} \sum_{x, \theta \in \{0, 1\}} |x\rangle \langle x| \otimes |\theta\rangle \langle \theta| \otimes H^\theta |x\rangle \langle x| H^\theta \quad (12)
$$

This state arises when encoding a uniformly random bit $X$ using
either the computational basis ($\Theta = 0$) or the Hadamard basis
($\Theta = 1$) chosen uniformly at random. Clearly, we have

$$
H_\infty(X) = 1 \\
H_\infty(X|\Theta) = 1 \quad \text{and} \\
H_\infty(X|Q\Theta) = 0
$$

where the last identity is a consequence of the fact that given
$\Theta = 0$, the operation $H^\Theta$ can be undone, such that a subsequent
measurement in the computational basis provides $X$ with cer-
tainty. Note that this is a special case of the identity

$$
H_\infty(X|Q\Theta) = -\log E_{P_\Theta} [2^{-H_\infty(X|Q, \Theta = \theta)}] \quad (13)
$$

for a general cq-state $\rho_{XQ\Theta}$ with classical part $\Theta$, where
$E_{P_\Theta}$ denotes the expectation value over the choice of $\Theta$, and
$H_\infty(X|Q, \Theta = \theta)$ is the min-entropy of the conditional state

$$
\rho_{X|Q, \Theta = \theta} = \sum_{x \in \{0, 1\}} P_{X|\Theta = \theta}(x) |x\rangle \langle x| \otimes H^\theta |x\rangle \langle x| H^\theta
$$

Returning to the state (12), it can also be shown [1] that

$$
H_\infty(X|Q) = -\log \left( \frac{1}{2} + \frac{1}{2\sqrt{2}} \right)
$$

**Smooth Min-Entropy:** When building oblivious transfer from
WSE, we will need to employ a more general definition of the
min-entropy given in [49]. For arbitrary (not necessarily unit
trace, or cq) bipartite density operators $\rho_{AB}$, this quantity is
defined as

$$
H_{\min}^\varepsilon(A|B)_{\rho} = -\log \inf_{\sigma \geq 0: \text{tr}(\rho_{AB} - \sigma) \geq 0, \text{tr}(\rho_{AB} - \sigma)^2 \leq \varepsilon^2} \text{tr}(\sigma_{AB}) \quad (14)
$$
where we use the subscript $\rho$ to indicate what state the quantity
refers to. In [34], it was shown via semidefinite programming
duality that for a cq-state $\rho_{XQ}$, definition (14) of $H_{\infty}(X|Q)$ co-
incides with definition (11) in terms of the guessing-probability
$P_{\text{guess}}(X|Q)$. The advantage of (14) is that it allows us to max-
imize over neighborhoods of $\rho_{XQ}$. This leads to the definition
of **smooth entropy** [49], which is defined as

$$
H_{\infty}^\varepsilon(X|Q)_{\rho} := \sup_{\rho_{XQ} \geq 0 : \|\rho_{XQ} - \rho'_{XQ}\|_1 \leq 2\varepsilon} \frac{\text{tr}(\rho_{XQ})}{\text{tr}(\rho'_{XQ})} H_{\infty}(X|Q)_{\rho'} \quad (15)
$$

We will also use the fact that if $\rho_{XQ}$ is a cq-state, the supremum
can be restricted to density operators $\rho'_{XQ}$ where $X$ is classical
and has the same range as the original $X$. Definition (15) will be
convenient for our proof. Roughly, we will construct some state
that has high min-entropy. We then show that the state created
during a real execution of the protocol is $\varepsilon$-close to this state.
By the aforementioned definition, the actual state generated in
the protocol has high smooth min-entropy.

A useful property of the smooth min-entropy is that it obeys
a chain rule [49, Th. 3.2.12], which states that for any ccq-state
$\rho_{XYQ}$, we have

$$
H_{\infty}^\varepsilon(X|YQ)_{\rho} \geq H_{\infty}^\varepsilon(X|Q)_{\rho} - \log |\mathcal{Y}| \quad (16)
$$

where $|\mathcal{Y}|$ is the size of the support of $Y$.

**Uncertainty Relations for Post-Measurement Information:**
When showing the security of WSE, we need to consider a
setting where an adversary can first extract some classical
information $K$ given access to a quantum system $Q$ and later
obtains some additional information $\Theta$. His objective is to
guess the value of a random variable $X$. Suppose he applies a
measurement described by a POVM $\{E_k\}_k$ to $Q$, and retains
only the measurement result $k$. We can think of this as a CPTPM
$\mathcal{K} : B(H_Q) \rightarrow B(H_K)$. When he performs this measurement
on the $Q$-part of a cq-state $\rho_{XQ}$, we get

$$
\rho_{X K(Q)} := (I_X \otimes \mathcal{K}) (\rho_{XQ}) \\
= \sum_k \text{tr}_Q ((I_X \otimes E_k) \rho_{XQ}) \otimes |k\rangle \langle k|
$$

which is a cc-state (i.e., an encoded joint distribution $P_{XK}$) if
$X$ is classical. Due to its definition, the min-entropy $H_{\infty}(X|Q)$
is intimately connected with such measurements, and in fact it
is easy to see that

$$
H_{\infty}(X|Q) = \min_{\mathcal{K}} H_{\infty}(X|K(\mathcal{Q})) \quad (17)
$$

This important identity relates min-entropies given quantum in-
formation $Q$ to min-entropies given classical information $K =
\mathcal{K}(Q)$.

Returning to the example given in (12), let us consider what
happens if the adversary learns the basis information $\Theta$ after
the measurement $\mathcal{K}$. In [1, Th. 4.7], it was shown [62] that the
minimal postmeasurement min-entropy optimized over all mea-
surements $\mathcal{K}$ obeys

$$
\min_{\mathcal{K}} H_{\infty}(X|K(Q)\Theta) = -\log \left( \frac{1}{2} + \frac{1}{2\sqrt{2}} \right)
$$

which in the case of our example matches the min-entropy
$H_{\infty}(X|Q)$ without postmeasurement information $\Theta$. In our
security proof, we will need to consider $n$ repetitions of the
state (12), that is

$$
\rho_{X^n \Theta^n Q^n} = \rho_{X \Theta Q}^{\otimes n}
$$

where $X^n = (X_1, \ldots, X_n)$ and $\Theta^n = (\Theta_1, \ldots, \Theta_n)$ are $n$-bit
strings, and $H_Q = (\mathbb{C}^2)^{\otimes n}$. It follows from [61, Lemma 2] and
[1] that

$$
\min_{\mathcal{K}} H_{\infty}(X^n | \mathcal{K}(Q^n) \Theta^n) = -n \cdot \log \left( \frac{1}{2} + \frac{1}{2\sqrt{2}} \right) \quad (18)
$$

A generalization of this relation to smooth min-entropy is

$$
\min_{\mathcal{K}} H_{\infty}^\varepsilon(X^n | \mathcal{K}(Q^n) \Theta^n)_{\rho} \geq n \left( \frac{1}{2} - 2\delta \right) - \log \frac{n \delta^2}{2} \quad (19)
$$

where $\delta \in ]0, \frac{1}{2}[$ and $\varepsilon = \exp \left( - \frac{n \delta^2}{32(2 + \log \frac{n}{\varepsilon})^2} \right)$

This relation follows from [15, Corollary 3.4] using the standard
purification trick (cf. [64, Lemma 2.3]). Our construction of a
protocol for WSE will make essential use of (18) and (19).

**Secure Keys and What it Means to be Ignorant:** We will often
informally say that an adversary “does not know anything” or
“does not learn anything” or “is ignorant” about some random
variable $X$, even when he holds some (quantum) information
$Q$. In terms of the cq-state $\rho_{XQ}$, this means that $X$ is uniformly
distributed on $\mathcal{X}$, and independent of $Q$, that is

$$
\rho_{XQ} = \tau_{\mathcal{X}} \otimes \rho_Q \quad (20)
$$

Clearly, for such a state, the uncertainty about $X$ given $Q$
is maximal, which in terms of the min-entropy means that
$H_{\infty}(X|Q) = \log |\mathcal{X}|$. For $\rho_{XQ}$ as in (20), $X$ is also referred to
as an **ideal key** with respect to $Q$.

In practice, we are generally forced to work with approxi-
mately ideal keys, where $X$ is called a **$\varepsilon$-secure key** with respect
to $Q$ if $\rho_{XQ}$ is $\varepsilon$-close to the ideal state $\tau_{\mathcal{X}} \otimes \rho_Q$, that is

$$
\rho_{XQ} \approx_{\varepsilon} \tau_{\mathcal{X}} \otimes \rho_Q \quad (21)
$$

This notion of a secure key behaves nicely under composition
[3], [33], [50].

## C. Processes That Increase Uncertainty

To show the security of WSE, we need to capture the
amount of “uncertainty” that an adversary has as a result of
his noisy storage $F$. First, let us consider general processes
which increase uncertainty. Note that from the definitions,
it is immediate [49, Th. 3.1.12] that the min-entropy satis-
fies the following monotonicity property: for every CPTPM
$F: B(H_Q) \rightarrow B(H_{Q'})$, we have

$$
H_{\infty}(X|F(Q)) \geq H_{\infty}(X|Q) \quad (22)
$$

An important case is where $H_Q = H_{Q_1 Q_2}$ is bipartite, and
$F = \text{tr}_{Q_2}$ is the partial trace over the second system $Q_2$. We
then get

$$
H_{\infty}(X|Q_1) \geq H_{\infty}(X|Q_1 Q_2) \quad (23)
$$

reflecting the fact that “forgetting” information makes it harder
to guess $X$.

Inequality (22) is insufficient for our purposes, and we will
need a more quantitative estimate on the increase of entropy due
to a channel $F$ representing the adversary's memory. Clearly,
such an estimate will depend on properties of $F$. Here, we ex-
press the bound in terms of the function $P_{\text{succ}}^F(n)$ introduced in
(2). Intuitively, the following lemma shows that the uncertainty
about $X$ after application of $F$ to $Q$ is related to the problem
of transmitting classical bits through the channel $F$, where the
number of bits is given by the min-entropy of $X$.

**Lemma II.1:** Consider an arbitrary cq-state $\rho_{XQ}$ and a
CPTPM $F : B(H_Q) \rightarrow B(H_{\text{out}})$. Then, $H_{\infty}(X|F(Q)) \geq$
$-\log P_{\text{succ}}^F (\lceil H_{\infty}(X) \rceil)$.

*Proof:* Let $k := \lceil H_{\infty}(X) \rceil$. It is well known (see e.g.,
[55]) that probability distributions $P_X$ with min-entropy at least
$k$ are convex combinations of “flat” distributions, i.e., uniform
distributions over subsets of $\mathcal{X}$ of size $2^k$. In other words, there
is a joint distribution $P_{XT}$, where $T$ is distributed over subsets
of size $2^k$, such that

$$
P_X(x) = \sum_t P_T(t) P_{X|T=t}(x) \quad \text{and} \quad P_{X|T=t} \text{ is uniform on } t \subset \mathcal{X}
$$

The distribution $P_{XT}$ together with $\rho_{XQ}$ gives rise to a state
$\rho_{XQT}$ whose partial trace is equal to $\rho_{XQ}$. Again using (23),
we get

$$
H_{\infty}(X|F(Q)) \geq H_{\infty}(X|F(Q)T)
$$

By property (13) of the min-entropy when conditioning on clas-
sical information, we have

$$
H_{\infty}(X|F(Q)T) = -\log E_{t \leftarrow P_T} [2^{-H_{\infty}(X|F(Q), T=t)}] \quad (24)
$$

where $E_{t \leftarrow P_T}$ denotes the expectation value, and
$H_{\infty}(X|F(Q), T = t)$ is the min-entropy of the conditional
state

$$
\rho_{X F(Q)|T=t} = \sum_x P_{X|T=t}(x) |x\rangle \langle x| \otimes F(\rho_Q^x)
$$

Now, we use the fact that $P_{X|T=t}$ is uniform over a set of size
$2^k$, and the definition of $P_{\text{succ}}^F(n)$. This leads to

$$
H_{\infty}(X|F(Q), T = t) \geq -\log P_{\text{succ}}^F (k) \quad \text{for all } t \text{ in the support of } P_T. \quad (25)
$$

Combining (24) with (25) gives the claim.

We now give a straightforward but important generalization
of this result.

**Lemma II.2:** Consider an arbitrary ccq-state $\rho_{XTQ}$, and let
$\varepsilon, \varepsilon' \geq 0$ be arbitrary. Let $F : B(H_Q) \rightarrow B(H_{Q_{\text{out}}})$ be an
arbitrary CPTPM. Then

$$
H_{\infty}^{\varepsilon+\varepsilon'}(X|TF(Q)) \geq -\log P_{\text{succ}}^F \left( \lceil H_{\infty}^\varepsilon(X|T) \rceil - \log \frac{1}{\varepsilon'} \right)
$$

*Proof:* Clearly, the statement for $\varepsilon = 0$ implies the state-
ment for any $\varepsilon > 0$ because a CPTPM cannot increase distance.
To prove the statement for $\varepsilon = 0$, we consider the quantities
$2^{-H_{\infty}(X|T=t)}$ of the conditional states $\rho_{X|T=t}$, together with
the distribution $P_T$ over $T$ defined by the state $\rho_{XTQ}$. Applying
Markov’s inequality $\text{Pr}[Z > c] < \frac{E[Z]}{c}$ for any real-valued
random variable $Z$ and constant $c > 0$, we obtain

$$
\text{Pr}_{t \leftarrow P_T} [2^{-H_{\infty}(X|T=t)} > 2^{-H_{\infty}(X|T) + \log \frac{1}{\varepsilon'}}] \leq \varepsilon' \\
\varepsilon' (2^{-H_{\infty}(X|T)})^{-1} E_{t \leftarrow P_T} [2^{-H_{\infty}(X|T=t)}] = \varepsilon'
$$

This implies that the distribution $P_T$ has weight at least $1 - \varepsilon'$
on the set

$$
\text{Good} = \{ t \in \mathcal{T} \mid H_{\infty}(X|T = t) \geq \lceil H_{\infty}(X|T) \rceil - \log \frac{1}{\varepsilon'} \} \quad (26)
$$

Accordingly, we can rewrite $\rho_{XTQ}$ as a convex combination

$$
\rho_{XTQ} = (1 - p) \cdot \rho_{XTQ|T \notin \text{Good}} + p \cdot \rho_{XTQ|T \in \text{Good}} \quad \text{where} \quad p = \text{Pr}(\text{Good}) \geq 1 - \varepsilon' \quad (27)
$$

Set $\sigma_{XTQ} := \rho_{XTQ|T \in \text{Good}}$. From (27), we conclude that
$\|\rho_{XTQ} - \sigma_{XTQ}\|_1 \leq \varepsilon'$. By the monotonicity of the distance
under CPTPM, it therefore suffices to show that

$$
H_{\infty}(X|TF(Q))_{\sigma} \geq -\log P_{\text{succ}}^F \left( \lceil H_{\infty}(X|T)_{\sigma} \rceil - \log \frac{1}{\varepsilon'} \right) \quad (28)
$$

For this purpose, note that $\sigma_{XTF(Q)}$ is given by the expression

$$
\sigma_{XTF(Q)} = \sum_{t \in \text{Good}} P_{T|T \in \text{Good}}(t) |t\rangle \langle t| \otimes \rho_{X F(Q)|T=t}
$$

In particular, by using (13) again, we have

$$
2^{-H_{\infty}(X|TF(Q))_{\sigma}} = E_{t \leftarrow P_{T|T \in \text{Good}}} [2^{-H_{\infty}(X|F(Q), T=t)}] \quad (29)
$$

Using Lemma II.1 (applied to the conditional state $\rho_{XQ|T=t}$),
we conclude that

$$
H_{\infty}(X|F(Q), T = t) \geq -\log P_{\text{succ}}^F \left( \lceil H_{\infty}(X|T)_{\rho} \rceil - \log \frac{1}{\varepsilon'} \right) \quad (30)
$$

for all $t \in \text{Good}$
The claim (28) immediately follows from (30) and (29).

## D. Defeating a Quantum Adversary: Essential Building Blocks

In order to build oblivious transfer and bit commitment from
WSE, we will employ three additional tools: first, we require
**privacy amplification** against a quantum adversary [49], [50] as
explained in Section II-D1. For oblivious transfer, we also need
the notion of **min-entropy sampling** outlined in Section II-D2.
In particular, we discuss how min-entropy about classical infor-
mation is approximately preserved when considering randomly
chosen subsystems. We then show in Section II-D4 how random
subsets can be chosen in a cryptographically secure manner with
a protocol called **IH**.

**Privacy Amplification:** Intuitively, privacy amplification al-
lows us to turn a long string $X$, about which the adversary
holds some quantum information $Q$, into a shorter string $Z =
\text{Ext}(X, R)$ about which he is almost entirely ignorant. The max-
imal length of this new string is directly related to the min-en-
tropy $H_{\infty}(X|Q)$ from Section II-B. In order to obtain this new
string, we will need a two-universal hash function. Formally, a
function $\text{Ext} : \{0, 1\}^n \times \mathcal{R} \rightarrow \{0, 1\}^\ell$ is called two-universal
if for all $x \neq x' \in \{0, 1\}^n$ and uniformly chosen $r \in_R \mathcal{R}$, we
have $\text{Pr}[\text{Ext}(x, r) = \text{Ext}(x', r)] \leq 2^{-\ell}$.

**Theorem II.3 (Privacy Amplification [49], [50]):** Consider
a set of two-universal hash functions $\text{Ext} : \{0, 1\}^n \times \mathcal{R} \rightarrow$
$\{0, 1\}^\ell$, and a cq-state $\rho_{X^n Q} \approx_{\varepsilon} \tau_{\{0, 1\}^n} \otimes \rho_Q$, where $X^n$ is an $n$-bit string. De-
fine $\rho_{X^n Q R} = \rho_{X^n Q} \otimes \tau_{\mathcal{R}}$, i.e., $R$ is a random variable uni-
formly distributed on $\mathcal{R}$, and independent of $X^n Q$. Then

$$
\rho_{\text{Ext}(X^n, R) R Q} \approx_{\varepsilon'} \tau_{\{0, 1\}^\ell} \otimes \rho_{RQ}
$$

for all $\varepsilon > 0$
where $\varepsilon' := 2^{-(\frac{1}{2} H_{\infty}(X^n|Q) - \ell) - 1} + 2\varepsilon$

It is important to stress that the extracted key $\text{Ext}(X^n, R)$ is
secure even if the adversary is given $R$ in addition to $Q$. The-
orem II.3 immediately gives rise to a procedure allowing parties
sharing some random variable $X^n$ to extract a key secure against
an adversary holding $Q$. Indeed, one party can simply use inde-
pendent randomness to pick $r \in_R \mathcal{R}$ uniformly at random, and
distribute (publicly) the value of $r$. Because two-universal hash
functions can be efficiently constructed (e.g., using linear func-
tions [10]), this privacy amplification protocol is efficient [7],
[9], [27].

**Min-Entropy Sampling:** For oblivious transfer, we will make
use of the sampling property of min-entropy which was first
established by Vadhan [58] in the classical case, and in [32] for
the classical-quantum case. Consider a cq-state $\rho_{X^n Q}$, where
$X^n = (X_1, \ldots, X_n)$ is an $n$-bit string. An important property of
smooth min-entropy is that the min-entropy rate

$$
\frac{H_{\infty}^\varepsilon(X^n|Q)}{n} \quad (31)
$$

is approximately preserved when considering a randomly
chosen substring $X_S$ of $X^n$. In some sense, we can, therefore,
think of (31) as the (average) min-entropy of an individual bit
$X_i$ given $Q$.

The corresponding technical statement is slightly more in-
volved. In essence, it requires to pick a subset $S$ from a distribu-
tion $\mathcal{P}_S$ over subsets of $[n]$ with certain properties ($\mathcal{P}_S$ needs to
be a so-called *averaging sampler*, see e.g., [22]). For concrete-
ness, we consider the special case where $\mathcal{P}_S$ is uniformly dis-
tributed over subsets of size $s = |S|$. Vadhan’s result for the
classical case [58] then shows that, for sufficiently large $s$, we
have

$$
\frac{H_{\infty}(X_S|C)}{s} \geq \frac{H_{\infty}(X^n|C)}{n} - \delta
$$

with high probability over the choice of $S$, for some small $\varepsilon > 0$
and $\delta > 0$. An analogous statement for the cq-case is given in
[32]. A major difference is that the result in [32] for the quantum
setting requires $X_i$ to be a block, i.e., a $\beta$-bit string instead of a
single bit.

Since our work is mainly a proof of principle, we do not yet
care about optimality or efficiency. We therefore choose $S$ to
be uniform over all subsets of a fixed size $s$. Furthermore, it is
sufficient for our purposes to ensure that the min-entropy rate
decreases by at most a factor of 2. Note that for technical rea-
sons, the results in [32] requires the bit string to be partitioned
into blocks of size $\beta$. A result in [65] shows, however, that the
same bound must also hold for uniform bitwise sampling. This
leads us to the following statement, which we derive by special-
izing the results in [32] and combining it with the result in [65]
(see Appendix A for details).

[IMAGE: Fig. 3. Concept of IH. Honest Bob has input $W^t$. IH creates substrings $W_0^t$ and $W_1^t$ such that there exists $D \in \{0, 1\}$ with $W_D^t = W^t$, where $D$ is unknown to Alice, and Bob has little control over the choice of $W_{1-D}^t$.]

**Lemma II.4 (Min-Entropy Sampling, [32] Combined With
[65]):** Let $\rho_{X^{m\beta} Q}$ be a cq-state, where $X^{m\beta}$ is an $m\beta$-bit
string. Let

$$
\bar{\lambda} = \frac{H_{\infty}^\varepsilon(X^{m\beta}|Q)}{m\beta}
$$

be a lower bound on the smooth min-entropy rate of $X^{m\beta}$ given
$Q$. Let $\omega > 2$ be a constant, and assume $s, \beta \in \mathbb{N}$ are such that

$$
s \geq m/4 \quad \text{and} \quad \beta \geq \max \left\{ 67, \frac{256 \omega^2}{\lambda^2} \right\} \quad (32)
$$

and let $\mathcal{P}_S$ be the uniform distributions over subsets of $[m\beta]$ of
size $s\beta$. Then

$$
\text{Pr}_S \left[ \frac{H_{\infty}^{\varepsilon+4\delta}(X_S|Q)_{\rho_S}}{s\beta} \geq \frac{\bar{\lambda}}{\omega} \right] \geq 1 - \delta^2
$$

where $\delta = 2^{-m\lambda^2/(512\omega^2)}$

**Aborting a Protocol:** As our protocols allow players to be
malicious, they may abort simply by not sending a message.
One way to handle this is to add a special symbol “aborted”
to the definition of each primitive, and to handle this case sepa-
rately in the protocol and the proof. For simplicity, we will take
a different approach here. Whenever a player does not send any
messages (or a message that does not have the right format), the
other player simply assumes that a particular valid message was
sent, for example, the string containing only zeros. Obviously,
the malicious player could have sent this message himself, so
refusing to send a message does not give any advantage to him.
From now on, we will, therefore, assume that all players always
send a message when they are supposed to.

**IH:** A final tool we need is IH [18], [19], [51] first introduced
in [47]. This is a two-party primitive where Bob inputs some
string $W^t$, and Alice has no input. The primitive then gener-
ates two strings $W_0^t, W_1^t$, with the property that one of the two
equals $W^t$. For a protocol implementing this primitive, security
is intuitively specified by the following conditions: Alice does
not learn which of the two strings is indeed equal to $W^t$. Con-
versely, Bob should have very little control over the other string
created by the protocol. Fig. 3 depicts the idealized version of
this primitive.

More formally, the following is achieved in [19, Th. 5.6],
where we refer to [51] for the exact parameters used in the se-
curity condition for Alice.

**Lemma II.5 (IH [19], [51]):** There exists a protocol called
IH between two players, Alice and Bob, where Alice has no
input, Bob has input $W^t \in \{0, 1\}^t$ and both players output
$(W_0^t, W_1^t) \in \{0, 1\}^t \times \{0, 1\}^t$, satisfying the following.
Correctness: If both players are honest, then $W_0^t \neq W_1^t$ and
there exists a $D \in \{0, 1\}$ such that $W_D^t = W^t$. Furthermore,
the distribution of $W_{1-D}^t$ is uniform on $\{0, 1\}^t \setminus \{W^t\}$.
Security for Bob: If Bob is honest, then $W_0^t \neq W_1^t$ and
there exists a $D \in \{0, 1\}$ such that $W_D^t = W^t$. If Bob chooses $W^t$
uniformly at random, then $D$ is uniform and independent of
Alice's view.
Security for Alice: If Alice is honest, then for every subset
$S \subset \{0, 1\}^t$

$$
\text{Pr} [W_{1-D}^t \in S \mid W_D^t \notin S] \leq 16 \cdot \frac{|S|}{2^t}
$$

Note that even though this is not explicitly mentioned in [51],
aborts need to be treated as explained in Section II-B3 to achieve
Lemma II.5.

# III. WSE IN THE NOISY-STORAGE MODEL

We are now ready to introduce our main primitive. After
giving a precise security definition in Section III-A, we present
a protocol for realizing this primitive in the noisy-storage
model. We will subsequently show that the protocol satisfies
the given security definition.

## A. Definition

**“Strong” Versus WSE:** In an ideal world, string erasure
would realize the ideal functionality depicted in Fig. 4. It takes
no inputs, but provides Alice with a uniformly distributed $n$-bit
string $X^n = (X_1, \ldots, X_n) \in \{0, 1\}^n$, while Bob receives a
random subset $I = \{i_1, \ldots, i_r\} \subset 2^{[n]}$ and the substring
$X_I$. The set of indices $I$ would be randomly distributed over
all the set $2^{[n]}$ of all subsets of $[n]$. Intuitively, we think of the
complement of $I$ as the locations of the "erased" bits.

[IMAGE: Fig. 4. Ideal functionality of string erasure. The actual WSE is somewhat weaker. However, a dishonest party cannot gain significantly more information from the protocol than provided by the "box" depicted above.]

Ideally, we would like to realize the functionality in Fig. 4
in such a way that even a dishonest party cannot learn anything
at all beyond what is provided by the box. Unfortunately, this
definition is too stringent to be achieved by our protocol. We,
therefore, relax our functionality to WSE, where the players may
gain a small amount of additional information. More precisely,
we allow a dishonest Bob to learn some information about $X^n$
possibly different from $(I, X_I)$. However, we demand that his
total information about $X^n$ is limited: given a dishonest Bob’s
system $B'$, he still has some residual amount of uncertainty
about $X^n$. For a dishonest Alice, we essentially retain the strong
security property that she does not learn anything about the set
of indices $I$ that Bob receives. In order to obtain bit commitment
and oblivious transfer later on, we also demand one additional
property that may seem superfluous from a classical perspective,
namely that Alice is “committed” to a choice of $X^n$ at the end
of the protocol. This difficulty arises since unlike in a classical
setting, a dishonest Alice may, for example, store some quantum
information and perform measurements only at a later time. This
may allow her to determine parts of $X^n$ after the protocol is
completed. Security against such attacks is subtle to define in
a quantum setting. To address this problem, we define security
in terms of an “ideal” state $\sigma_{A' X^n I X_I}$ that could have been ob-
tained by an honest Alice by preparing some state on $A'$ using
$X^n$ (i.e., by postprocessing). Our security definition then de-
mands that the actual state $\rho_{A'B}$ shared by dishonest Alice and
honest Bob after the execution of the protocol has the same form
as the partial trace of the ideal state, that is, $\rho_{A'B} = \sigma_{A' X^n I X_I}$.

**Formal Definition:** In the following definition of WSE, we
write $\rho_{AB}$ for the resulting state at the end of the protocol if both
parties are honest, $\rho_{A'B}$ if Alice is dishonest and $\rho_{AB'}$ if Bob
is dishonest. Our definition is phrased in terms of ideal states
denoted by $\sigma$ that exhibit all the desired properties of WSE.
We then demand that the actual states $\rho$ created during a real
execution of the protocol are at least $\varepsilon$-close to such ideal states
no matter what kind of attack the dishonest party may perform.

**Definition III.1:** An $(\eta, \lambda, \varepsilon)$ **WSE** scheme is a protocol be-
tween Alice and Bob satisfying the following properties.
Correctness: If both parties are honest, then the ideal state
$\sigma_{X^n I X_I}$ is defined such that
1) The joint distribution of the $n$-bit string $X^n$ and the subset
$I$ is uniform

$$
\sigma_{X^n I} = \tau_{\{0, 1\}^n} \otimes \tau_{2^{[n]}} \quad (33)
$$

2) The joint state $\rho_{AB}$ created by the real protocol is equal to
the ideal state

$$
\rho_{AB} = \sigma_{X^n I X_I} \quad (34)
$$

where we identify $(A, B)$ with $(X^n, I X_I)$.
Security for Alice: If Alice is honest, then there exists an ideal
state $\sigma_{X^n B'}$ such that
1) The amount of information $B'$ gives Bob about $X^n$ is lim-
ited

$$
\frac{1}{n} H_{\infty}^\varepsilon(X^n | B')_{\sigma} \geq \lambda
$$

2) The joint state $\rho_{AB'}$ created by the real protocol is $\varepsilon$-close
to the ideal state

$$
\sigma_{X^n B'} \approx_{\varepsilon} \rho_{AB'}
$$

where we identify $(X^n, B')$ with $(A, B')$.
Security for Bob: If Bob is honest, then there exists an ideal state
$\sigma_{A' \hat{X}^n \hat{I} \hat{X}_{\hat{I}}}$, where $\hat{X}^n \in \{0, 1\}^n$ and $\hat{I} \subseteq [n]$ such that
1) The random variable $\hat{I}$ is independent of $A' \hat{X}^n$ and uni-
formly distributed over $2^{[n]}$

$$
\sigma_{A' \hat{X}^n} = \sigma_{A' \hat{X}^n} \otimes \tau_{2^{[n]}}
$$

2) The joint state $\rho_{A'B}$ created by the real protocol condition
on the event that Alice does not abort is equal to the ideal
state

$$
\rho_{A'B} = \sigma_{A' (\hat{I} \hat{X}_{\hat{I}})}
$$

[IMAGE: Fig. 5. Protocol as a circuit. Alice chooses a random string $x^n = (x_1, \ldots, x_n) \in \{0, 1\}^n$. She then encodes the bits in random bases specified by $\Theta^n = (\theta_1, \ldots, \theta_n) \in \{0, 1\}^n$ and sends the corresponding quantum states to Bob. Bob measures in random bases specified by $\tilde{\Theta}^n = (\tilde{\theta}_1, \ldots, \tilde{\theta}_n) \in \{0, 1\}^n$ obtaining measurement outcomes $\tilde{X}^n = (\tilde{x}_1, \ldots, \tilde{x}_n)$. Upon reception of the basis string $\Theta^n$, Bob determines the locations where he measured in the same basis by computing the bitwise xor $\Theta^n \oplus \tilde{\Theta}^n = (\theta_1 \oplus \tilde{\theta}_1, \ldots, \theta_n \oplus \tilde{\theta}_n)$. He subsequently discards the bits he measured in the wrong bases (indicated by $\ddagger$: this replaces the classical input symbol by an erasure symbol).]

where we identify $(A', B)$ with $(A', \hat{I} \hat{X}_{\hat{I}})$.
Note that we do not require $\hat{X}^n$ to be uniform when Bob is
dishonest. To show security of bit commitment and oblivious
transfer, we will only require that $\hat{X}^n$ has high min-entropy. The
condition that the real state is close to an ideal state having high
min-entropy means that the real state has smooth min-entropy
as outlined in Section II.

## B. Protocol

We now consider a simple protocol achieving WSE in the
noisy-storage model using BB84-states. Other encodings are
certainly possible, and we will discuss some of the implica-
tions of this choice of encoding in Section VI. This protocol
is essentially identical to the first step of known protocols for
quantum key distribution [4], [64]. However, as explained in the
previous section, our security requirements differ greatly as we
are dealing with two mutually distrustful parties.

**Protocol 1: WSE**
Outputs: $x^n \in \{0, 1\}^n$ to Alice, $(I, \tilde{x}^{[I]}) \in 2^{[n]} \times \{0, 1\}^{|I|}$
to Bob.
1: Alice: Chooses a string $x^n \in_R \{0, 1\}^n$ and
bases-specifying string $\Theta^n \in_R \{0, 1\}^n$ uniformly at
random. She encodes each bit $x_i$ in the basis given by $\theta_i$
(i.e., as $H^{\theta_i} |x_i\rangle$) and sends it to Bob.
2: Bob: Chooses a basis string $\tilde{\Theta}^n \in_R \{0, 1\}^n$ uniformly
at random. When receiving the $i$-th qubit, Bob measures
it in the basis given by $\tilde{\theta}_i$ to obtain outcome $\tilde{x}_i$. [Both
parties wait time $\Delta t$.]
3: Alice: Sends the basis information $\Theta^n$ to Bob, and outputs
$x^n$.
4: Bob: Computes $I := \{i \in [n] \mid \theta_i = \tilde{\theta}_i\}$, and outputs
$(I, \tilde{x}^{[I]}) := (I, \tilde{x}_I)$.

Our main claim is the following.

**Theorem III.2 (WSE):** 1) Let $\delta \in ]0, 1/2[$ and let Bob’s storage
be given by $F : B(H_{\text{in}}) \rightarrow B(H_{\text{out}})$. Then, Protocol 1 is an
$(\eta, \lambda(\delta, \eta), \varepsilon(\delta, n))$ WSE protocol with min-entropy rate

$$
\lambda(\delta, n) = - \frac{1}{n} \log P_{\text{succ}}^F \left( \lceil H_{\infty}(\tau) \rceil - \delta \frac{n}{2} \right)
$$

and error

$$
\varepsilon(\delta, n) = 2 \exp \left( - \frac{n \delta^2}{512 (4 + \log \frac{2}{\delta})^2} n \right) \quad (35)
$$

2) Suppose $F = N^{\otimes \nu n}$ for a storage rate $\nu > 0$, $N$ satis-
fying the strong-converse property (4) and having capacity $C_N$
bounded by

$$
C_N \cdot \nu < \frac{1}{2}
$$

Let $\delta \in ]0, \frac{1}{2} - C_N \cdot \nu[$. Then, Protocol 1 is an $(n, \tilde{\lambda}(\delta), \varepsilon(\delta, n))$
WSE protocol for sufficiently large $n$, where

$$
\tilde{\lambda}(\delta) = \nu \cdot \gamma_N \left( \frac{1/2 - \delta}{\nu} \right)
$$

It is easy to see that the protocol is correct if both parties are
honest: if Alice is honest, her string $X^n = x^n$ is chosen uni-
formly at random from $\{0, 1\}^n$ as desired, and if Bob is honest,
he will clearly obtain $\tilde{x}_i = x_i$ whenever $i \in I$ for a random
subset $I \subseteq [n]$. The remainder of Section III is, thus, devoted
to proving security if one of the parties is dishonest. In Sec-
tion III-C, we use the properties of the channel $F$ to show that
the protocol is secure against a dishonest Bob. In Section III-D,
we argue that the protocol satisfies Definition III.1 when Alice
is dishonest.

## C. Security for Honest Alice

We now show that for any cheating strategy of a dishonest
Bob, his min-entropy about the string $X^n = (X_1, \ldots, X_n)$
is large. Before turning to the proof, we first explain in Fig. 6
how our model restricts the actions of Bob in our protocol.
At time $t$, Bob receives an encoding of a classical string
$x^n = (x_1, \ldots, x_n)$ which he would like to reconstruct as
accurately as possible. To this end, he can apply any CPTPM
$\mathcal{E}: B((\mathbb{C}^2)^{\otimes n}) \rightarrow B(H_{\text{in}} \otimes H_K)$ with the following property:
for any input state $\rho$ on $(\mathbb{C}^2)^{\otimes n}$, he obtains an output state
$\rho_{Q_{\text{in}} K} = \mathcal{E}(\rho)$, where $Q_{\text{in}}$ is the quantum information he will
put into his quantum storage, and $K$ is any additional classical
information he retains. Note that we allow an arbitrary amount
of classical storage, that is, $H_K$ may be arbitrarily large. We
call the map $\mathcal{E}$ Bob’s **encoding attack**.

We can think of the encoding attack $\mathcal{E}$ as being composed
of two steps, $\mathcal{E} = (\text{tr}_{Q_{\text{out}}} \otimes \mathcal{K}) \circ \mathcal{E}_1$ where Bob first applies
an arbitrary CPTPM $\mathcal{E}_1 : B((\mathbb{C}^2)^{\otimes n}) \rightarrow B(H_{Q_{\text{in}}} \otimes H_{Q_{\text{out}}})$, and
subsequently performs a measurement $\mathcal{K} : B(H_{Q_{\text{out}}}) \rightarrow B(H_K)$
on $H_{Q_{\text{out}}}$. The outcome of this measurement forms his classical
information $K = \mathcal{K}(Q_{\text{out}})$. For example, Bob can measure some
of the incoming qubits, or encode some information using an

[IMAGE: Fig. 6. Most general structure of a cheating Bob. Bob’s action at time $t$ consists of a CPTPM $\mathcal{E}_1$, followed by a (partial) measurement $\mathcal{K}$, where he may use an arbitrary ancilla $\rho_{\text{aux}}$. At time $t + \Delta t$, Bob can try to reconstruct $x^n = (x_1, \ldots, x_n)$ given the content $F(Q_{\text{in}})$ of the storage device, the classical measurement result $K = \mathcal{K}(Q)$, and the basis information $\Theta^n = (\theta_1, \ldots, \theta_n)$.]

error-correcting code. The joint state before his storage noise is
applied is, hence, given by

$$
\rho_{X^n \tilde{\Theta}^n K Q_{\text{in}}} = \frac{1}{(2^n)^2} \sum_{x^n, \tilde{\theta}^n \in \{0, 1\}^n, k \in \mathcal{K}} P_{K|X^n = x^n, \tilde{\Theta}^n = \tilde{\theta}^n}(k) \pi_{X^n} \otimes \pi_{\tilde{\Theta}^n} \otimes \rho_{Q_{\text{in}}}^k \otimes \pi_K \otimes \rho_{\text{aux}}^{\otimes n} \quad (36)
$$

where $\rho_{Q_{\text{in}}}^k$ is the conditional state on $H_{Q_{\text{in}}}$ conditioned on the
string $X^n = x^n$, the basis choice $\tilde{\Theta}^n = \tilde{\theta}^n$, and Bob’s classical
measurement outcome $K = k$. Here, we used the abbreviation
$\pi_A := |x\rangle \langle x|$. The state (36) is completely determined by Bob’s
encoding attack $\mathcal{E}$ at time $t$.

Bob’s storage $Q_{\text{in}}$ then undergoes noise described by $F :$
$B(H_{\text{in}}) \rightarrow B(H_{\text{out}})$, and the state evolves to $\rho_{X^n \tilde{\Theta}^n K F(Q_{\text{in}})}$.
At time $t + \Delta t$, Bob additionally receives the basis information
$\Theta^n = \theta^n$. The joint state is now given by

$$
\rho_{X^n \Theta^n \tilde{\Theta}^n K F(Q_{\text{in}})} = \frac{1}{(2^n)^2} \sum_{x^n, \theta^n \in \{0, 1\}^n, k \in \mathcal{K}} P_{K|X^n = x^n, \tilde{\Theta}^n = \tilde{\theta}^n}(k) \pi_{X^n} \otimes \pi_{\Theta^n} \otimes \pi_{\tilde{\Theta}^n} \otimes \pi_K \otimes F(\rho_{Q_{\text{in}}}^k) \quad (37)
$$

where Bob holds $B' = \Theta^n K F(Q_{\text{in}})$. We now show that Bob’s
information $B'$ about $X^n$ is limited for large $n$.

**Theorem III.3 (Security for Alice):** Fix $\delta \in ]0, 1/2[$ and let

$$
\varepsilon = 2 \exp \left( - \frac{n \delta^2}{32 (2 + \log \frac{2}{\delta})^2} n \right)
$$

Then, for any attack of a dishonest Bob with storage
$F: B(H_{\text{in}}) \rightarrow B(H_{\text{out}})$, there exists a cq-state $\sigma_{X^n B'}$ such
that
1) $\sigma_{X^n B'} \approx_{\varepsilon} \rho_{X^n B'}$
2) $\frac{1}{n} H_{\infty}^\varepsilon(X^n | B')_{\sigma} \geq -\frac{1}{n} \log P_{\text{succ}}^F \left( \lceil H_{\infty}(\tau) \rceil - \delta \frac{n}{2} \right)$
where $\rho_{X^n B'}$ is given by (37). In particular, if, for some $R < 1/2$, we have $\lim_{n \rightarrow \infty} -\frac{1}{n} \log P_{\text{succ}}^F(nR) > 0$, then $\rho_{X^n B'}$ is exponentially close (in $n$) to a state $\sigma_{X^n B}$ with constant min-entropy rate $\frac{1}{n} H_{\infty}(X^n | B)_{\sigma}$.

*Proof:* We use the notation introduced in (37). By def-
inition (15) of the smooth min-entropy, statements (1) and
(2) follow if we show that the smooth min-entropy rate
$\frac{1}{n} H_{\infty}^\varepsilon(X^n | B')_{\rho}$ is lower bounded by the expression on the
right-hand side in (2). By the uncertainty relation (19), we have

$$
H_{\infty}^\varepsilon(X^n | \mathcal{K}(Q^n) \Theta^n)_{\rho} \geq n \left( \frac{1}{2} - \delta \right) - \log \frac{n \delta^2}{2}
$$

Using Lemma II.2 applied to $T = (\tilde{\Theta}^n, K)$, we conclude that
for $Q_{\text{out}} = F(Q_{\text{in}})$ after the noise $F$ there exists the claimed
ideal state and

$$
H_{\infty}^\varepsilon(X^n | K Q_{\text{out}})_{\rho} \geq -\log P_{\text{succ}}^F \left( \lceil \frac{n}{2} - \frac{n \delta}{2} \rceil - \log \frac{2}{\varepsilon} \right) \\
\geq -\log P_{\text{succ}}^F \left( \frac{n}{2} - \frac{n \delta}{2} - \log \frac{2}{\varepsilon} \right)
$$

where the final inequality follows from the monotonicity
property of the success probability $P_{\text{succ}}^F(m) \leq P_{\text{succ}}^F(m')$
for $m \geq m'$ and the fact that $\log \frac{2}{\varepsilon} \leq \frac{n}{2}$ because
$(n\delta^2)/(32(2 - \log 8/\delta)^2) \leq n/2$ for any $0 < \delta < 1/2$.

Let us specialize Theorem III.3 to the case where $F$ is a tensor
product channel.

**Corollary III.4:** Let Bob’s storage be described by $F =
N^{\otimes \nu n}$ with $\nu > 0$, where $N$ satisfies the strong-converse prop-
erty (4), and

$$
C_N \cdot \nu < \frac{1}{2}
$$

Fix $\delta \in ]0, \frac{1}{2} - C_N \cdot \nu[$, and let $\varepsilon = \varepsilon(\delta, n)$ be defined by (35).
Then, for any attack of a dishonest Bob, there exists a cq-state
$\sigma_{X^n B'}$ such that
1) $\sigma_{X^n B'} \approx_{\varepsilon} \rho_{X^n B'}$
2) $\frac{1}{n} H_{\infty}^\varepsilon(X^n | B')_{\sigma} \geq \nu \cdot \gamma_N \left( \frac{1/2 - \delta}{\nu} \right) > 0$
where $\rho_{X^n B'}$ is given by (37).

*Proof:* Substituting $n$ by $\nu n$ and $R$ by $R/\nu$, the strong-
converse property (4) turns into

$$
-\frac{1}{n} \log P_{\text{succ}}^{N^{\otimes \nu n}} (nR) \geq \nu \cdot \gamma_N (R/\nu)
$$

[IMAGE: Fig. 8. In the security proof, we put an intermediate "simulator" between Alice and Bob to generate the state $\sigma_{A' \hat{X}^n \hat{\Theta}^n T}$. We will show that the security definition III.1 is satisfied with $\sigma_{A' \hat{X}^n \hat{\Theta}^n T}$. The simulator measures the quantum register in the basis specified by the bit string. He then encodes the measurement result $\hat{X}^n = (\hat{X}_1, \ldots, \hat{X}_n)$ into randomly chosen bases.]

for sufficiently large $n$. The claim then follows from Theorem
III.3 by setting $R := \frac{1}{2} - \delta$.

Theorem III.3 and Corollary III.4 establish the first part of
Theorem III.2. It remains to analyze the security against a dis-
honest Alice.

## D. Security for Honest Bob

When Alice is dishonest, it is intuitively obvious that she is
unable to gain any information about the index set $I$, since she
never receives any information from Bob during our protocol.
Yet, in order to obtain bit commitment and oblivious transfer
from WSE, we require a more careful security analysis. Fig. 7
depicts the form of any interaction between a cheating Alice
and an honest Bob. Since Alice takes no input in the protocol,
her actions are completely specified by the state $\rho_{Q_A \Theta^n T}$ she
outputs, where $H_{Q_A} = (\mathbb{C}^2)^{\otimes n}$ is an $n$-qubit register that she
sends to Bob (in the case where Alice is honest, this encodes the
string $X^n$), $\Theta^n$ is some classical $n$-bit string (in the case where
Alice is honest, this encodes the bases), and $H_T$ is an auxiliary
register of Alice corresponding to the (quantum) information
she holds after execution of the protocol. In the actual protocol,
an honest Bob proceeds as shown in Fig. 5.
1) Upon receipt of $Q_A$ at time $t$, an honest Bob mea-
sures in randomly chosen bases specified by the string
$\tilde{\Theta}^n = (\tilde{\theta}_1, \ldots, \tilde{\theta}_n) \in \{0, 1\}^n$, obtaining measurement
outcomes $\tilde{X}^n = (\tilde{X}_1, \ldots, \tilde{X}_n)$.
2) After receiving $\Theta^n = (\Theta_1, \ldots, \Theta_n)$ at time $t + \Delta t$, he
computes the intersecting set $I$ defined by $\Theta^n$ and $\tilde{\Theta}^n$, and
the corresponding substring $\tilde{X}_I$.

The protocol, thus, creates some state $\rho_{A' \tilde{X}_I I}$, where $A' =
(\Theta^n T)$ is Alice's information, and $B = (I \tilde{X}_I)$ is the informa-
tion obtained by Bob. Note that this state can be obtained from
$\rho_{A' X^n \tilde{\Theta}^n \Theta^n}$ because $I$ is a function of $\Theta^n$ and $\tilde{\Theta}^n$, and $\tilde{X}_I$ is a
function of $\tilde{X}^n$ and $I$.

**Theorem III.5 (Security for Bob):** Protocol 1 satisfies security
for honest Bob.

*Proof:* We now construct a state $\sigma_{A' T \hat{X}_I \hat{I}}$ with the required
properties. For simplicity, we give an algorithmic description of
this state. It is obtained by letting Alice and Bob interact with
a simulator which has perfect quantum memory. Note that this
simulator is purely imaginary and is merely used to specify the
desired ideal state $\sigma_{A' \hat{X}^n \hat{\Theta}^n T}$. However, we will later show that
the real state created during the protocol equals this ideal state
on the registers held by Alice and Bob. Fig. 8 summarizes the
actions of the simulator.
1) First, the simulator measures the $n$-qubits $Q_A$ in the bases
specified by the bits $\tilde{\Theta}^n = (\tilde{\Theta}_1, \ldots, \tilde{\Theta}_n)$, obtaining mea-
surement outcomes $\tilde{X}^n = (\tilde{X}_1, \ldots, \tilde{X}_n)$.
2) Second, the simulator re-encodes the measurement out-
comes $\tilde{X}^n$ using randomly chosen bases specified by
$\hat{\Theta}^n = (\hat{\Theta}_1, \ldots, \hat{\Theta}_n) \in_R \{0, 1\}^n$. He then sends the
corresponding qubits to Bob (i.e., the states $H^{\hat{\Theta}_i} |\tilde{X}_i\rangle$). We
call this quantum register $Q_{\hat{A}}$.
3) Finally, the simulator provides Bob with the basis string
$\hat{\Theta}^n = (\hat{\Theta}_1, \ldots, \hat{\Theta}_n)$.

An honest Bob proceeds as before, but with $\tilde{\Theta}^n$ replaced by
the simulator's string $\hat{\Theta}^n$, and $Q_A$ replaced by the simulator's
quantum message $Q_{\hat{A}}$. As before, Alice's information $A' =$
$(\Theta^n T)$ consists of the string $\Theta^n$ and her (quantum) system $T$.
The state $\sigma_{A' \hat{I} \tilde{X}_{\hat{I}}}$ held by Alice and Bob can be obtained from
$\sigma_{A' \tilde{X}^n \hat{\Theta}^n \Theta^n}$, noting that $\tilde{X}_{\hat{I}} = \tilde{X}_{\hat{I}}$.
Let us argue that $\sigma_{A' \hat{I} \tilde{X}_{\hat{I}}}$ has the properties required by Def-
inition III.1. First, observe that

$$
\sigma_{A' \tilde{X}^n \hat{\Theta}^n \Theta^n} = \sigma_{A' \tilde{X}^n} \otimes \tau_{\{0, 1\}^n} \otimes \tau_{\{0, 1\}^n}
$$

since both $\hat{\Theta}^n$ and $\Theta^n$ are chosen uniformly and independently
at random by the simulator and Bob, respectively. Since the set
$\hat{I}$ consists of those indices where $\hat{\Theta}^n$ and $\Theta^n$ agree, we conclude
that $\hat{I}$ is uniform on the set of subsets of $[n]$, and independent
of $A' \tilde{X}^n$. That is, the previous identity implies

$$
\sigma_{A' \tilde{X}^n \hat{I}} = \sigma_{A' \tilde{X}^n} \otimes \tau_{2^{[n]}} \quad (38)
$$

as desired. It remains to prove that the state created during the
real protocol equals this ideal state, that is

$$
\rho_{A'B} = \sigma_{(\Theta^n T) (\hat{I} \tilde{X}_{\hat{I}})} \quad (39)
$$

To produce the state $\sigma_{(\Theta^n T) (\hat{I} \tilde{X}_{\hat{I}})}$, honest Bob (interacting
with the simulator) measures all qubits in the bases $\hat{\Theta}^n$. Since
we are only interested in $\tilde{X}_{\hat{I}}$, we could instead apply the
first measurement and re-encoding (by the simulator) and the
second measurement (by Bob) only on the qubits in $\hat{I}$ without
affecting the output. But since for all $i \in \hat{I}$, we have $\hat{\Theta}_i = \Theta_i$,
the re-encoding and the second measurement are always in
the same basis, and can, therefore, be removed. Therefore, the
state $\sigma_{\Theta^n T \tilde{X}_{\hat{I}}}$ can also be produced in the following way. Let
Alice output registers $(Q_A, \Theta^n, T)$. We first choose $\hat{I} \subseteq [n]$
uniformly at random. Then, we measure all qubits in $\hat{I}$ in
bases $\Theta_i$ to get $\tilde{X}_{\hat{I}}$, and output registers $(\Theta^n, T, \hat{I}, \tilde{X}_{\hat{I}})$. Since
all qubits in the complement $\hat{I}$ are discarded anyway, we
can measure them in $\Theta_i^{\perp}$ without affecting the reduced state
$\sigma_{\Theta^n T \tilde{X}_{\hat{I}}}$. But this is exactly what happens in the real protocol
producing the state $\rho_{A'B}$, which implies (39).

[IMAGE: Fig. 9. Sufficient condition for achieving security (for storage rate $\nu = 1$) is that the noise parameter $r$ lies below the threshold given above. This is equivalent to $C_N \cdot \nu < \frac{1}{2}$.]

## E. Application to Concrete Tensor Product Channels

We examine the security parameters that we can obtain for
several well-known channels. A simple example is the $d$-di-
mensional depolarizing channel defined in (5), which replaces
the input state $\rho$ with the completely mixed state with proba-
bility $1 - r$. Another simple example is the one-qubit two-Pauli
channel [30]

$$
N_{\text{Pauli}}(\rho) := r \rho + \frac{1 - r}{2} X \rho X + \frac{1 - r}{2} Z \rho Z
$$

Both these channels obey the strong-converse property (4) (see
[36]), allowing us to obtain security of WSE by Corollary III.4.
For simplicity, we first consider the case where the storage
rate is $\nu = 1$, that is, Bob’s storage system is $(\mathbb{C}^d)^{\otimes n}$, i.e.,
$n$ copies of a $d$-dimensional system, and his noise channel is
$F = N^{\otimes n}$. We first determine the values of $r$ that allow for a
secure implementation of WSE. By Corollary III.4, the capacity
of the channel $N$ must be bounded by $C_N < 1/2$. The table given
in Fig. 9 summarizes the relevant parameters.

[IMAGE: Fig. 10. Tradeoff between $\nu$ and $r$. Security can be obtained for the qutrit depolarizing channel below the solid blue line and the two-Pauli channel below the dashed red line. Note, however, that for the same storage rate, the dimension of the storage system is larger for the qutrit than for the qubit channel.]

When allowing storage rates other than $\nu = 1$, we may again
consider the regime where our proof provides security. Fig. 10
examines this setting for the qutrit depolarizing channel and the
two-Pauli channel, respectively.

To determine the exact security of the protocol, we need to
compute the min-entropy rate

$$
\tilde{\lambda}(\delta) = \nu \cdot \gamma_N \left( \frac{1/2 - \delta}{\nu} \right)
$$

as stated in Corollary III.4. For the class of channels
$N : B(\mathbb{C}^d) \rightarrow B(\mathbb{C}^d)$ considered in [36], the strong con-
verse property (4) was shown to be satisfied with the function
$\gamma_N$ given by

$$
\gamma^N(R) = \max_{\alpha \geq 1} \frac{\alpha - 1}{\alpha} (R - \log d + S_{\min}^\alpha (N))
$$

[IMAGE: Fig. 11. Value of the min-entropy rate $\tilde{\lambda}$ for the qubit depolarizing channel (dashed red line) and the qutrit depolarizing channel (solid blue line) as a function of the noise parameter $r$, for $\nu = 1$ and $\delta = 0.01$. Using qutrits means that the dimension of the overall storage system is higher, and we expect the resulting higher capacity to lead to a smaller min-entropy rate $\tilde{\lambda}$. Our analysis confirms this intuition.]

where $S_{\min}^\alpha(N)$ is the minimum output $\alpha$-Rényi-entropy of the
channel. For the $d$-dimensional depolarizing channel [see (5)],
we may rewrite this expression [31] as

$$
\gamma^N(R) = \max_{\alpha \geq 1} \frac{\alpha - 1}{\alpha} \left( R - \log d - \log \left[ r + \frac{1 - r}{d} \right] - \frac{1}{\alpha - 1} \log \left[ r + \frac{1 - r}{d} \right] \right)
$$

Fig. 11 shows how the min-entropy rate $\tilde{\lambda}(\delta)$ relates to the noise
parameter $r$ for the qubit and qutrit depolarizing channels for a
storage rate of $\nu = 1$ and error $\delta = 0.01$. The figure shows that
the min-entropy rate we can achieve in our protocol is directly
related to the amount of noise in the storage.

# IV. BIT COMMITMENT FROM WSE

## A. Definition

Informally, a standard commitment scheme consists of a
**Commit** and an **Open** primitive between two parties Alice and
Bob. First, Alice and Bob execute the Commit primitive, where
Alice has input $Y^\ell \in \{0, 1\}^\ell$, and Bob has no input. As output,
Bob receives a notification that Alice has chosen an input $Y^\ell$.
Afterward, they may execute the Open protocol, during which
Bob either accepts or rejects. If both parties are honest, Bob
always accepts and receives the value $Y^\ell$. If Alice is dishonest,
however, we still demand that Bob either outputs the correct
value of $Y^\ell$ or rejects (binding). If Bob is dishonest, he should
not be able to gain any information about $Y^\ell$ before the Open
protocol is executed (hiding).

Here, we make use of a randomized version of a commitment
as depicted in Fig. 12. This simplifies both our definition, as well

[IMAGE: Fig. 12. Randomized string commitment. Alice receives a random $C^\ell \in_R \{0, 1\}^\ell$ from Commit. During the Open phase, Bob outputs $\tilde{C}^\ell$ and $F$. If both parties are honest, then $\tilde{C}^\ell = C^\ell$ and $F = \text{accept}$. If Alice is dishonest, Bob outputs $F \in \{\text{accept, reject}\}$, but $\tilde{C}^\ell = C^\ell$ if $F = \text{accept}$. To obtain a stan-
dard commitment, Alice can send the extra message indicated by the dashed line.]

as the protocol. Instead of inputting her own string $Y^\ell$, Alice
now receives a random string $C^\ell$ from the Commit protocol.
Note that if Alice wants to commit to a value $Y^\ell$ of her choice,
she may simply send the xor of her value with the random com-
mitment $Y^\ell \oplus C^\ell$ to Bob at the end of the Commit protocol.

To give a more formal definition, note that we may write the
Commit and the Open protocol as CPTPMS $\mathcal{C}_{AB}$ and $\mathcal{O}_{AB}$, re-
spectively, consisting of the local actions of honest Alice and Bob,
together with any operations they may perform on messages that
are exchanged. When both parties are honest, the output of the
Commit protocol will be a state $\mathcal{C}_{AB}(\rho_{\text{in}}) = \rho_{C^\ell U V}$ for some
fixed input state $\rho_{\text{in}}$, where $C^\ell \in \{0, 1\}^\ell$ is the classical output
of Alice, and $U$ and $V$ are the internal states of Alice and Bob,
respectively. Clearly, if Alice is dishonest, she may not follow the
protocol, and we use $\mathcal{C}_{A'B}$ to denote the resulting map. Note that
$\mathcal{C}_{A'B}$ may not have output $C^\ell$, and we, hence, simply write $\rho_{A'V}$
for the resulting output state, where $A'$ denotes the register of a
dishonest Alice. Similarly, we use $\mathcal{C}_{AB'}$ to denote the CPTPM cor-
responding to the case where Bob is dishonest, and write $\rho_{C^\ell U B'}$
for the resulting output state, where $B'$ denotes the register of a
dishonest Bob.

The Open protocol can be described similarly. If both parties
are honest, the map $\mathcal{O}_{AB}: B(H_{UV}) \rightarrow B(H_{C^\ell F})$ creates the
state $\eta_{C^\ell \tilde{C}^\ell F} := (\mathbb{I}_{C^\ell} \otimes \mathcal{O}_{AB}) (\rho_{C^\ell U V})$, where $C^\ell \in \{0, 1\}^\ell$
and $F \in \{\text{accept, reject}\}$ is the classical output of Bob. Again, if
Alice is dishonest, we write $\mathcal{O}_{A'B}$ to denote the resulting CPTPM
with output $\eta_{A' \tilde{C}^\ell F}$, and if Bob is dishonest, we write $\mathcal{O}_{AB'}$ for
the resulting CPTPM with output $\eta_{C^\ell \tilde{C}^\ell B'}$. The following defini-
tion is similar to the one given in [15], but slightly more general.

**Definition IV.1:** An $(\ell, \varepsilon)$-**random-
ized string commitment scheme** is a protocol between
Alice and Bob satisfying the following properties.
Correctness: If both parties are honest, then the ideal state
$\sigma_{C^\ell \tilde{C}^\ell F}$ is defined such that
1) The distribution of $C^\ell$ is uniform, and Bob accepts the
commitment

$$
\sigma_{C^\ell \tilde{C}^\ell F} = \tau_{\{0, 1\}^\ell} \otimes |\text{accept}\rangle \langle \text{accept}|
$$

2) The joint state $\eta_{C^\ell \tilde{C}^\ell F}$ created by the real protocol is
$\varepsilon$-close to the ideal state

$$
\eta_{C^\ell \tilde{C}^\ell F} \approx_{\varepsilon} \sigma_{C^\ell \tilde{C}^\ell F}
$$

where we identify $(A, B)$ with $(C^\ell, \tilde{C}^\ell F)$.
Security for Alice ($\varepsilon$-hiding): If Alice is honest, then for any
joint state $\rho_{C^\ell B'}$ created by the Commit protocol, Bob does not
learn $C^\ell$

$$
\rho_{C^\ell B'} \approx_{\varepsilon} \tau_{\{0, 1\}^\ell} \otimes \rho_{B'}
$$

Security for Bob ($\varepsilon$-binding): If Bob is honest, then there exists
an ideal cqq-state $\sigma_{C^\ell \tilde{C}^\ell A' V}$ such that for all $\mathcal{O}_{A'B}$
1) Bob almost never accepts $\tilde{C}^\ell \neq C^\ell$

$$
\text{Pr}[\tilde{C}^\ell \neq C^\ell \mid \mathcal{E}_{C^\ell A'} (\mathbb{I}_{C^\ell} \otimes \mathcal{O}_{A'V}) (\sigma_{C^\ell \tilde{C}^\ell A' V})] < \varepsilon
$$

2) The joint state $\rho_{A'V}$ created by the real protocol is $\varepsilon$-close
to the ideal state

$$
\rho_{A'V} \approx_{\varepsilon} \sigma_{A'V}
$$

## B. Protocol

Let $\varepsilon' > 0$. To construct our protocol based on WSE, we will
need a binary $(n, k, d)$-linear code $C \subseteq \{0, 1\}^n$, i.e., a linear
code with $2^k$ elements and minimal distance $d := 2 \log 1/\varepsilon'$.
Let $\text{Syn}: \{0, 1\}^n \rightarrow \{0, 1\}^{n-k}$ be a function that outputs a
parity-check syndrome for the code $C$. Let $\text{Ext}: \{0, 1\}^n \times$
$\mathcal{R} \rightarrow \{0, 1\}^\ell$ be a two-universal hash function as defined in
Section II-D1

**Protocol 2a: Commit**
Inputs: none. Outputs: $C^\ell \in \{0, 1\}^\ell$ to Alice.
1: Alice and Bob: Execute $(\eta, \lambda, \varepsilon)$-WSE. Alice gets
$x^n \in \{0, 1\}^n$, and Bob gets $I \subseteq [n]$ and $s = X_I$.
2: Alice: Chooses $r \in_R \mathcal{R}$ and sends $r$ and $w := \text{Syn}(x^n)$
to Bob.
3: Alice: Outputs $C^\ell := \text{Ext}(x^n, r)$ and stores $x^n$. Bob stores
$(r, w, I, s)$.

**Protocol 2b: Open**
Inputs: none. Outputs: $\tilde{C}^\ell \in \{0, 1\}^\ell$ and $f \in \{\text{accept, reject}\}$
to Bob.
1: Alice: Sends $x^n$ to Bob.
2: Bob: If $s \neq X_I$ or $w \neq \text{Syn}(x^n)$, then he outputs $\tilde{C}^\ell := 0^\ell$
and $f := \text{reject}$. Otherwise, he outputs $\tilde{C}^\ell := \text{Ext}(x^n, r)$
and $f := \text{accept}$.

Our main claim of this section is the following.

**Theorem IV.2 (String Commitment):** The pair (2a, 2b) of pro-
tocols (Commit,Open) is an $(\ell, 2\varepsilon + \varepsilon')$-randomized string com-
mitment scheme based on one in-
stance of $(\eta, \lambda, \varepsilon)$ WSE.
The length $\ell := \lambda \eta - (n - k) - 2 \log 1/\varepsilon'$ of the com-
mitment depends on our choice of code $C$. Since we require
that $\ell > 0$, we need $n - k$ to be small compared to $n$, which
means that we need codes for which $k/n \rightarrow 1$ for $n \rightarrow \infty$.
A simple construction of codes that satisfy this can be based
on Reed-Solomon codes [48] over the field $GF(2^m)$, which are
$(2^m - 1, 2^m - d, d)$-linear codes. We can convert these codes
into binary $((2^m - 1)m, (2^m - d)m, d)$-linear codes by simply
mapping each field element to $m$ bits. For $n := (2^m - 1)m$,
we have $n - k = (d - 1)m \leq d(\log n - 1)$, since $n \geq 2 \cdot 2^m$
whenever $m \geq 3$. Therefore, with these codes, we can achieve
$\ell \geq \lambda n - 2 \log n \log 1/\varepsilon'$, i.e., our commitment rate is roughly
$\lambda$.

## C. Security Proof

We again show security for Alice and Bob individually. Re-
call that if Bob is dishonest, our goal is to show that his infor-
mation about $C^\ell$ is negligible. The intuition behind this proof is
that WSE ensures that Bob’s information about the string $X^n$
is limited. Via privacy amplification, we then obtain that his in-
formation about $C^\ell$, which is the output of a two-universal hash
function applied to $X^n$, is negligible.

**Lemma IV.3 (Security for Alice):** The pair of protocols
(Commit,Open) is $(2\varepsilon + \varepsilon')$-hiding.

*Proof:* Let $\rho_{X^n B'}$ be the cq-state created by the execution
of WSE. From the properties of WSE, it follows that there exists
a state $\sigma_{X^n B'}$ such that $H_{\infty}^\varepsilon(X^n | B')_{\sigma} \geq \lambda n$ and $\rho_{X^n B'} \approx_{\varepsilon}$
$\sigma_{X^n B'}$. This implies that

$$
H_{\infty}^\varepsilon(X^n | B')_{\rho} \geq \lambda n
$$

By the chain rule [see (16)], we get

$$
H_{\infty}^\varepsilon(X^n | B' \text{Syn}(X^n))_{\rho} \geq \lambda n - (n - k) = \ell + 2 \log 1/\varepsilon'
$$

Using privacy amplification (Theorem II.3), we then get that

$$
\frac{1}{2} \|\rho_{C^\ell B'} - \tau_{\{0, 1\}^\ell} \otimes \rho_{B'} \|_1 \leq 2\varepsilon + 2^{- \frac{1}{2} \log 1/\varepsilon' - 1} < 2\varepsilon + \varepsilon'
$$

as promised.

To show security for honest Bob, we need the following prop-
erty of linear codes. Note that the function $\text{Syn}$ is linear, i.e., for
all codewords $x^n$ and $\tilde{x}^n$, we have $\text{Syn}(x^n \oplus \tilde{x}^n) = \text{Syn}(x^n) \oplus$
$\text{Syn}(\tilde{x}^n)$. Therefore, for any $x^n$ and $\tilde{x}^n$ with $x^n \neq \tilde{x}^n$ and
$\text{Syn}(x^n) = \text{Syn}(\tilde{x}^n)$, we have that the string $\text{Syn}(x^n \oplus \tilde{x}^n) \in$
$\{0, 1\}^{n-k}$ is the all zero string $0^{n-k}$. From this, it follows that
$x^n \oplus \tilde{x}^n$ is a codeword different from $0^n$. Since all codewords
except $0^n$ have weight at least $d$, it follows that $x^n$ and $\tilde{x}^n$ have
distance at least $d$.

The intuition behind the following proof is the observation
that WSEs ensure that Bob knows the substring $X_I$ of a string
$X$. The properties of the error-correcting code limit the set of
strings $X^n$ consistent with this substring and the given syn-
drome $W$; this implies that Alice will be detected with high
probability if she attempts to cheat.

**Lemma IV.4 (Security for Bob):** The pair of protocols
(Commit,Open) is $\varepsilon$-binding.

*Proof:* Let $\rho_{A'B}$ be the state shared by Alice and Bob
after the execution of WSE. From the properties of WSE, it fol-
lows that there exists a state $\sigma_{A' X_I I} = \sigma_{A' X^n} \otimes \tau_{2^{[n]}}$ such
that $\rho_{A'B} = \sigma_{A' (I X_I)}$, where $B = (I X_I)$. Let $X^n$ be the
closest string to $\tilde{X}^n$ that satisfies $\text{Syn}(\hat{X}^n) = W$, and let $C^\ell :=$
$\text{Ext}(\hat{X}^n, R)$. We will now show that the state $\sigma_{C^\ell A' (R W T \tilde{X}_I)}$ cre-
ated during the Commit protocol satisfies the binding condition.

First of all, note that if Alice sends $X^n = \hat{X}^n$, then Bob
outputs $\tilde{C}^\ell = C^\ell$. It thus remains to analyze the case of $X^n \neq$
$\hat{X}^n$. Note that we may write

$$
\text{Pr}[\tilde{C}^\ell \neq C^\ell \mid F = \text{accept}] \\
= \text{Pr}[\text{Ext}(X^n, R) \neq \text{Ext}(\hat{X}^n, R) \mid F = \text{accept}] \\
\quad X^n, \hat{X}^n, R \\
\quad \text{Syn}(X^n) = \text{Syn}(\hat{X}^n) \\
+ \text{Pr}[\text{Ext}(\hat{X}^n, R) \neq \text{Ext}(\hat{X}^n, R) \mid F = \text{accept}] \\
\quad X^n, \hat{X}^n, R \\
\quad \text{Syn}(X^n) = \text{Syn}(\hat{X}^n) \\
= \text{Pr}[\text{Ext}(X^n, R) \neq \text{Ext}(\hat{X}^n, R) \mid F = \text{accept}] \\
\quad X^n, \hat{X}^n, R \\
\quad \text{Syn}(X^n) = \text{Syn}(\hat{X}^n)
$$

where the last equality follows from the fact that Bob always
rejects if $\text{Syn}(X^n) \neq \text{Syn}(\hat{X}^n)$.

We now show that the remaining term is small. Note that if
$\text{Syn}(X^n) = \text{Syn}(\hat{X}^n)$, and $X^n \neq \hat{X}^n$, the distance between
$X^n$ and $\hat{X}^n$ is at least $d$. We also know that for our choice of
$\hat{X}^n$, the distance between $\tilde{X}^n$ and $\hat{X}^n$ is at most $d/2$. Hence,
$X^n$ has distance at least $d/2$ to $\tilde{X}^n$. Since Alice does not know
$I$ and every $i \in [n]$ is in $I$ with probability $1/2$, Bob accepts with
probability at most $\varepsilon = 2^{-d/2}$. Hence, we obtain

$$
\text{Pr}[\tilde{C}^\ell \neq C^\ell \mid F = \text{accept}] \leq \varepsilon'
$$

as promised.

It remains to show that the protocol is correct. This follows
essentially from the properties of WSE. However, we still need
to demonstrate that the state we obtain from WSE has $C^\ell$ close
to uniform.

**Lemma IV.5 (Correctness):** The pair of protocols
(Commit,Open) satisfies correctness with an error of at most
$2\varepsilon + \varepsilon'$.

*Proof:* Let $\eta_{C^\ell \tilde{C}^\ell F}$ be the state at the end of the protocol.
It follows directly from the properties of WSE that $\eta_{C^\ell \tilde{C}^\ell} =
\eta_{C^\ell \tilde{C}^\ell}$. It remains to show that this state is close to the ideal state
$\sigma_{C^\ell \tilde{C}^\ell F}$. By the same arguments as in Lemma IV.3, it follows that
$\|\eta_{C^\ell \tilde{C}^\ell} - \sigma_{C^\ell \tilde{C}^\ell}\|_1 \leq 2\varepsilon + \varepsilon'$. Hence, we also have $\|\eta_{C^\ell \tilde{C}^\ell F} -
\sigma_{C^\ell \tilde{C}^\ell F}\|_1 \leq 2\varepsilon + \varepsilon'$.

# V. 1-2 OBLIVIOUS TRANSFER FROM WSE

## A. Definition

We now show how to obtain 1-2 oblivious transfer given
access to WSE. Usually, one considers a nonrandomized ver-
sion of 1-2 oblivious transfer, in which Alice has two inputs
$Y_0^\ell, Y_1^\ell \in \{0, 1\}^\ell$, and Bob has input a choice bit $D \in \{0, 1\}$.
At the end of the protocol, Bob receives $Y_D^\ell$, and Alice receives
no output. The protocol is considered secure if the parties do
not gain any information beyond this specification, that is, Alice
does not learn $D$ and there exists some input $Y_{1-D}^\ell$ about which
Bob remains ignorant.

Here, we again make use of fully randomized oblivious
transfer (FROT). Fully randomized oblivious transfer takes no
inputs, and outputs two strings $S_0^\ell, S_1^\ell \in \{0, 1\}^\ell$ to Alice, and
a choice bit $C \in \{0, 1\}$ and $S_C^\ell$ to Bob. Security means that

[IMAGE: Fig. 13. Fully randomized 1-2-oblivious transfer when Alice and Bob are honest. Intuitively, if one of the parties is dishonest, he/she should not be able to obtain more information from the primitive as depicted above. The dashed messages are exchanged to obtain nonrandomized oblivious transfer from FROT.]

if Alice is dishonest, she should not learn anything about $C$.
Similar to WSE, we also demand that two strings $S_0^\ell$ and $S_1^\ell$ are
created by the protocol. Intuitively, this ensures that just like
in a classical protocol, we can again think of the protocol as
being completed once Alice and Bob have exchanged their final
message. If Bob is dishonest, we demand that there exists some
random variable $\tilde{C}$ such that Bob is entirely ignorant about
$S_{1-\tilde{C}}^\ell$. That is, he may learn at most one of the two strings
which are generated.

Fully randomized oblivious transfer can easily be converted
into “standard” oblivious transfer as depicted in Fig. 13 using
the protocol presented in [8] (see also [2]). To obtain nonran-
domized 1-2 oblivious transfer, Bob sends Alice a message in-
dicating whether $C = D$. Note that since Alice does not know
$C$, she also does not know anything about $D$. If $C = D$, Alice
sends Bob $Y_0^\ell \oplus S_0^\ell$, and $Y_1^\ell \oplus S_1^\ell$, otherwise she sends $Y_0^\ell \oplus S_1^\ell$
and $Y_1^\ell \oplus S_0^\ell$. Clearly, if Bob does not learn anything about $S_{1-C}^\ell$,
he can learn at most one of $Y_0^\ell$ and $Y_1^\ell$ [2], [8].

We now provide a more formal definition, which is very sim-
ilar to the definitions in [15] and [21].

**Definition V.1:** An $(\ell, \varepsilon)$-**FROT** is a protocol between Alice
and Bob satisfying the following.
1) Correctness: If both parties are honest, then the ideal state
$\sigma_{S_0^\ell S_1^\ell C S_C^\ell}$ is defined such that
1) The distribution over $S_0^\ell, S_1^\ell$ and $C$ is uniform

$$
\sigma_{S_0^\ell S_1^\ell C} = \tau_{\{0, 1\}^\ell} \otimes \tau_{\{0, 1\}^\ell} \otimes \tau_{\{0, 1\}}
$$

2) The real state $\rho_{S_0^\ell S_1^\ell C Y^\ell}$ created during the protocol is
$\varepsilon$-close to the ideal state

$$
\rho_{S_0^\ell S_1^\ell C Y^\ell} \approx_{\varepsilon} \sigma_{S_0^\ell S_1^\ell C S_C^\ell} \quad (40)
$$

where we identify $A = (S_0^\ell, S_1^\ell)$ and $B = (C, Y^\ell)$.
2) Security for Alice: If Alice is honest, then there exists an
ideal state $\sigma_{S_0^\ell S_1^\ell B' C}$, where $\tilde{C}$ is a random variable on
$\{0, 1\}$, such that
1) Bob is ignorant about $S_{1-\tilde{C}}^\ell$

$$
\sigma_{S_{1-\tilde{C}}^\ell B' C} \approx_{\varepsilon} \tau_{\{0, 1\}^\ell} \otimes \sigma_{B' C}
$$

2) The real state $\rho_{S_0^\ell S_1^\ell B' C}$ created during the protocol is
$\varepsilon$-close to the ideal state

$$
\rho_{S_0^\ell S_1^\ell B' C} \approx_{\varepsilon} \sigma_{S_0^\ell S_1^\ell B' C}
$$

3) Security for Bob: If Bob is honest, then there exists an ideal
state $\sigma_{A' S_0^\ell S_1^\ell C}$ such that
1) Alice is ignorant about $C$

$$
\sigma_{A' S_0^\ell S_1^\ell C} \approx_{\varepsilon} \sigma_{A' S_0^\ell S_1^\ell} \otimes \tau_{\{0, 1\}}
$$

2) The real state $\rho_{A' C Y^\ell}$ created during the protocol is
$\varepsilon$-close to the ideal state

$$
\rho_{A' C Y^\ell} \approx_{\varepsilon} \sigma_{A' C S_C^\ell}
$$

where we identify $B = (C, Y^\ell)$.

Again, we allow the protocol implementing this primitive to
abort, but demand that the security conditions are satisfied if the
protocol does not abort.

## B. Protocol

We now show how to obtain an FROT given access to WSE.
As in Section III, honest players never abort the protocol. If
the dishonest player refuses to send correctly formed messages,
the honest player chooses the messages himself. Note that we
require the same also from the IH protocol: If one player aborts
it, the other terminates the interaction and proceeds to simulate
the remainder of the protocol himself. Indeed, this is needed to
satisfy Lemma II.5, which does not deal with aborts. By inspec-
tion of the protocols, it is easy to see that the honest player can
indeed simulate all the other player’s messages in this way.

To obtain some intuition for the actual protocol, consider the
following naïve protocol, which we only state informally. It
makes use of a two-universal hash function $\text{Ext}: \{0, 1\}^{n/4} \times$
$\mathcal{R} \rightarrow \{0, 1\}^\ell$.

**Protocol 3': Naïve Protocol (informal)**
Outputs: $(S_0^\ell, S_1^\ell) \in \{0, 1\}^\ell \times \{0, 1\}^\ell$ to Alice, and
$(C, Y^\ell) \in \{0, 1\} \times \{0, 1\}^\ell$ to Bob
1: Alice and Bob: Execute WSE. Alice gets a string
$X^n \in \{0, 1\}^n$, Bob a set $I \subseteq [n]$ and a string $s = X_I$. If
$|I| < n/4$, Bob randomly adds elements to $I$ and pads
the corresponding positions in $s$ with $0$s. Otherwise,
he randomly truncates $I$ to size $n/4$, and deletes the
corresponding values in $s$.
2: Alice and Bob: Execute IH with Bob’s input $w$ equal to a
description of $I = \text{Enc}(w)$. Interpret the outputs $w_0$ and
$w_1$ as descriptions of subsets $I_0$ and $I_1$ of $[n]$.
3: Alice: Chooses $r_0, r_1 \in_R \mathcal{R}$ and sends them to Bob.
4: Alice: Outputs $(S_0^\ell, S_1^\ell) := (\text{Ext}(X_{I_0}, r_0), \text{Ext}(X_{I_1}, r_1))$.
5: Bob: Computes $C \in \{0, 1\}$ with $I = I_C$, and $Y^\ell$ from $s$.
He outputs $(C, Y^\ell) := (C, \text{Ext}(s, r_C))$.

For now, let us neglect the fact that the outputs of IH are
strings, and assume that the subset $I_{1-C}$ generated by the IH
protocol is uniformly distributed over subsets of size $n/4$ not
equal to $I$. The string $X_{I_{1-C}}$ is then obtained by sampling from
the string $X^n$, which by the definition of WSE has high min-en-
tropy. We, therefore, expect the value $S_{1-C}^\ell$ to be uniform and in-
dependent of Bob’s view. This should imply security for Alice,
whereas security for Bob immediately follows from the proper-
ties of IH.

**Protocol 3: WSE-to-FROT**
Parameters: Integers $n, \beta$ such that $m := n/\beta$ is a multiple of
4. Set $t := \lceil \log \binom{m}{m/4} \rceil$. Outputs: $(S_0^\ell, S_1^\ell) \in \{0, 1\}^\ell \times \{0, 1\}^\ell$
to Alice, and $(C, Y^\ell) \in \{0, 1\} \times \{0, 1\}^\ell$ to Bob
1: Alice and Bob: Execute $(\eta, \lambda, \varepsilon)$-WSE. Alice gets a string
$X^n \in \{0, 1\}^n$, Bob a set $I \subseteq [n]$ and a string $s = X_I$. If
$|I| < n/4$, Bob randomly adds elements to $I$ and pads
the corresponding positions in $s$ with $0$s. Otherwise, he
randomly truncates $I$ to the size $n/4$, and deletes the
corresponding values in $s$.
2: Bob:
1) Randomly chooses a string $W^t \in_R \{0, 1\}^t$
corresponding to an encoding of a subset $\text{Enc}(W^t)$
of $[n]$ with $n/4$ elements.
2) He randomly chooses a permutation $\Pi : [n] \rightarrow [n]$
of the entries of $X^n$ such that he knows $\Pi(X^n)_{\text{Enc}(W^t)}$
(that is, these bits are permutation of the bits of $s$).
Formally, $\Pi$ is uniform over permutations satisfying
the following condition: for all $j \in [n]$ and $j' := \Pi(j)$,
we have $j \in I \Leftrightarrow j' \in \text{Enc}(W^t)$.
3) Bob sends $\Pi$ to Alice.
3: Alice and Bob: Execute IH with Bob’s input equal to $W^t$.
They obtain $W_0^t, W_1^t \in \{0, 1\}^t$ with $W^t \in \{W_0^t, W_1^t\}$.
4: Alice: Chooses $r_0, r_1 \in_R \mathcal{R}$ and sends them to Bob.
5: Alice: Outputs $(S_0^\ell, S_1^\ell) :=$
$(\text{Ext}(\Pi(X^n)_{\text{Enc}(W_0^t)}, r_0), \text{Ext}(\Pi(X^n)_{\text{Enc}(W_1^t)}, r_1))$.
6: Bob: Computes $C$, where $W^t = W_C^t$, and $\Pi(X^n)_{\text{Enc}(W^t)}$ from
$s$. He outputs $(C, Y^\ell) := (C, \text{Ext}(\Pi(X^n)_{\text{Enc}(W^t)}, r_C))$.

To use IH in conjunction with subsets, the actual protocol
needs an encoding of subsets $\text{Enc} : \{0, 1\}^t \rightarrow \mathcal{T}$, where $\mathcal{T}$
is the set of all subsets of $[n]$ of size $n/4$. Here, we choose $t$
such that $2^t \leq \binom{n}{n/4} \leq 2 \cdot 2^t$, and an injective encoding $\text{Enc} :$
$\{0, 1\}^t \rightarrow \mathcal{T}$, i.e., no two strings are mapped to the same subset.
Note that this means that not all possible subsets are encoded,
but at least half of them. We refer to [19] and [51] for details on
how to obtain such an encoding. Note that since not every subset
has an encoding, we cannot simply take $W^t := \text{Enc}^{-1}(I)$. To
solve this problem, we first choose a $W^t$ uniformly at random,
and then choose a random permutation $\Pi$ such that Bob knows
the subset encoded by $W^t$ in $\Pi(X^n)$.

**Theorem V.2 (Oblivious Transfer):** For any constant $\omega >
2$ and $\beta \geq \max \{67, 256 \omega^2/\lambda^2\}$, the protocol WSE-to-FROT
implements an $(\ell, 43 \cdot 2^{- \frac{n \lambda^2}{512 \omega^2 \beta}} + 2 \varepsilon)$-FROT from one instance
of $(\eta, \lambda, \varepsilon)$ WSE, where $\ell := \lceil (\lambda \frac{n}{\beta} - \frac{n}{\beta} - 2 \log \frac{1}{\varepsilon}) \frac{1}{2} \rceil$.
Since this study is a proof of principle, we may choose $\omega = 2$.
However, if we were to look at a more practical setting, choosing
other values of $\omega$ can be beneficial.

## C. Security Proof

We first show that the protocol is secure against a cheating
Alice. Intuitively, the properties of WSE ensure that Alice does
not know which bits $X_I$ of $X^n$ are known to Bob, that is, she
is ignorant about the index set $I$. This implies that essentially
any partition of the bits is consistent with Alice’s view. In par-
ticular, she does not gain much information from the particular
partition chosen by Bob. Finally, the properties of IH ensure that
she cannot gain much information about which of the two final
strings is known to Bob.

**Lemma V.3 (Security for Bob):** Protocol WSE-to-FROT sat-
isfies security for Bob.

*Proof:* Let $\rho_{A'' C Y^\ell}$ denote the joint state at the end of the
protocol, where $A''$ is the quantum output of a malicious Alice
and $(C, Y^\ell)$ is the classical output of an honest Bob. We con-
struct an ideal state $\sigma_{A'' W W_0^t W_1^t C} := \sigma_{A'' W W_0^t W_1^t} \otimes \tau_{\{0, 1\}}$ that sat-
isfies $\rho_{A'' C Y^\ell} \approx \sigma_{A'' C S_C^\ell}$.
First, we divide a malicious Alice into two parts. The first
part interacts with Bob in the WSE protocol, after which the
state shared by Alice and Bob is $\rho_{A' X_I I}$. From the properties
of WSE, it follows that there exists an ideal state $\sigma_{\hat{X}^n I}$
such that the reduced state satisfies $\rho_{A' X_I I} \approx \sigma_{A' X_I I}$.

The second part of Alice takes $A'$ as input and interacts with
Bob in the rest of the protocol. To analyze the resulting joint
output state $\rho_{A'' C Y^\ell}$, we can use the properties of WSE, and let
the second part of Alice interact with honest Bob starting from
the state $\sigma_{A' \hat{X}^n I}$. The protocol outputs a state $\tilde{\sigma}_{A'' \hat{X}^n C Y^\ell M}$,
where $M$ denotes all classical communication during the pro-
tocol. Note that the values $\Pi, W, W_0^t, W_1^t, R_0$, and $R_1$ can be
computed from $M$. Let $S_i := \text{Ext}(\Pi(\hat{X}^n)_{\text{Enc}(W_i^t)}, R_i)$ for $i \in$
$\{0, 1\}$. We obtain the state $\tilde{\sigma}_{A'' S_0^\ell S_1^\ell C Y^\ell}$ by taking the partial
trace of $\tilde{\sigma}_{A'' \hat{X}^n C Y^\ell M}$. From the construction of this state
and the fact that $\rho_{A' X_I I} \approx \sigma_{A' \hat{X}_I I}$, it follows directly that
$\rho_{A'' C Y^\ell} = \tilde{\sigma}_{A'' C Y^\ell}$ and $\tilde{\sigma}_{A'' S_0^\ell S_1^\ell C Y^\ell} = \tilde{\sigma}_{A'' S_0^\ell S_1^\ell C S_C^\ell}$. Hence

$$
\rho_{A'' C Y^\ell} \approx_{\varepsilon} \tilde{\sigma}_{A'' C S_C^\ell}
$$

It remains to be shown that Alice does not learn anything
about $C$, that is, $\tilde{\sigma}_{A'' S_0^\ell S_1^\ell C} \approx \tilde{\sigma}_{A'' S_0^\ell S_1^\ell} \otimes \tau_{\{0, 1\}}$. From the prop-
erties of WSE it follows that $\sigma_{A' \hat{X}^n} = \sigma_{A' \hat{X}^n} \otimes \tau_{2^{[n]}}$. Since
Bob randomly truncates/extends $I$ such that $|I| = n/4$, the re-
sulting set $I$ is also uniformly distributed over all subsets of size
$n/4$ and independent of $A'$. Hence, conditioned on any fixed
$W^t = w^t$, the permutation $\Pi$ is uniform and independent of $A'$.
It follows that the string $\Pi(\hat{X}^n)$ is uniform and independent of
$A'$ and $\Pi$. From the properties of IH, we are guaranteed that $C$ is
uniform and independent of Alice's view afterward, and hence

$$
\tilde{\sigma}_{A'' S_0^\ell S_1^\ell C} \approx \tilde{\sigma}_{A'' S_0^\ell S_1^\ell} \otimes \tau_{\{0, 1\}}
$$

Second, we show that the protocol is secure against a cheating
Bob. We again first give an intuitive argument. We have from
WSE that Bob gains only a limited amount of information about
the string $X^n$. The properties of IH ensure that Bob has very
little control over one of the subsets of bits chosen by the IH.
Therefore, by the results on min-entropy sampling, Bob only
has limited information about these bits of $X^n$. Privacy ampli-
fication can then be used to turn this into almost complete igno-
rance.

**Lemma V.4 (Security for Alice):** Protocol WSE-to-FROT sat-
isfies security for Alice with an error of

$$
41 \cdot 2^{-\frac{n \lambda^2}{512 \omega^2 \beta}} + 2\varepsilon
$$

*Proof:* Let $\rho_{X^n B'}$ be the cq-state created by the execution
of WSE. From the properties of WSE, it follows that there exists
a state $\sigma_{X^n B'}$ such that $H_{\infty}^\varepsilon(X^n | B')_{\sigma} \geq \lambda n$ and $\rho_{X^n B'} \approx_{\varepsilon}$
$\sigma_{X^n B'}$, which implies that

$$
H_{\infty}^\varepsilon(X^n | B')_{\rho} \geq \lambda n
$$

Since the permutation $\Pi$ is chosen by Bob based on his quantum
information $B'$, it follows from (22) that

$$
H_{\infty}^\varepsilon(\Pi(X^n) | \Pi B')_{\rho} \geq H_{\infty}^\varepsilon(X^n | \Pi B')_{\rho} \geq H_{\infty}^\varepsilon(X^n | B')_{\rho}
$$

where $B''$ is Bob’s part of the shared quantum state after he has
sent $\Pi$ to Alice.

Recall that our goal is to show that Bob has high min-entropy
about the string $X^n$ restricted to one of the subsets generated by
the IH protocol. Our first step is to count the subsets which are
bad for Alice in the sense that Bob has a lot of information about
$X^n$. We then show that the probability that both sets chosen via
the IH primitive lie in the bad set is exponentially small in $n$.

With Lemma II.4, we conclude that for the uniform$^{10}$ distri-
bution over subsets $S \subseteq [n] = [\beta m]$ of size $\beta m/4 = |S|$

$$
\text{Pr}_S \left[ H_{\infty}^{\varepsilon+4\delta}(\Pi(X^n)_S | S \Pi B'')_{\rho} < \frac{1}{\omega} (\frac{\lambda n}{\beta m}) \frac{\beta m}{4} \right] < \delta^2 \quad (41)
$$

where $\delta = 2^{-m \lambda^2/(512 \omega^2)}$. Let $\text{Bad}$ be the set of all subsets of
size $\beta m/4$ that result in small min-entropy, i.e.

$$
\text{Bad} = \left\{ S \subseteq [\beta m], |S| = \frac{\beta m}{4} \mid H_{\infty}^{\varepsilon+4\delta}(\Pi(X^n)_S | S \Pi B'')_{\rho} < \frac{1}{\omega} \frac{\lambda n}{4} - 1 \right\}
$$

Since we have considered the uniform distribution over all sub-
sets of $[\beta m]$ of size $\beta m/4$, we conclude from (41) that

$$
|\{W^t \in \{0, 1\}^t \mid \text{Enc}(W^t) \in \text{Bad}\}| \leq |\text{Bad}| \cdot 2^t \frac{\binom{\beta m}{\beta m/4}}{\binom{\beta m}{\beta m/4}} \\
\leq \delta^2 \cdot 2^t \binom{\beta m}{\beta m/4} \\
< 2 \cdot 2^t \delta^2 \quad (42)
$$

In the first inequality, we have used the fact that $\text{Enc}$ is injective,
i.e., every element in the image has exactly one preimage. In the
last inequality, we used the fact that $\binom{\beta m}{\beta m/4} \leq 2 \cdot 2^t$. By the third
property of the IH, we conclude that

$$
\text{Pr}[\text{Enc}(W_0^t) \in \text{Bad} \text{ and } \text{Enc}(W_1^t) \in \text{Bad}] \leq 16 \cdot \frac{2 \cdot 2^t \delta^2}{2^t} \\
\leq 32 \delta^2 \quad (43)
$$

Let $\rho_{X^n W W_0^t W_1^t B'''}$ be the shared quantum state after the IH,
where $B'''$ is Bob’s part of that state. From (43), it follows that
there exists a $C \in \{0, 1\}$, or more precisely, there exists an ideal
state $\sigma_{X^n W W_0^t W_1^t B''' C}$ with $\rho_{X^n W W_0^t W_1^t B'''} \approx \sigma_{X^n W W_0^t W_1^t B'''}$,
such that

$$
\text{Pr}_{\rho} \left[ H_{\infty}^{\varepsilon+4\delta}(\Pi(X^n)_{\text{Enc}(W_{1-C}^t)} | W_{1-C}^t \Pi B''')_{\rho} \geq \frac{1}{\omega} \frac{\lambda n}{4} - 1 \right] \\
\geq 1 - 32 \delta^2 \quad (44)
$$

Note that Bob may use his quantum state during the IH, but he
cannot increase the probability of (43) this way. Furthermore,
any processing may only increase his uncertainty. Let $\mathcal{A}$ be the
event that the inequality in the argument on the left-hand side of
(44) holds. Let

$$
\tilde{\sigma}_{X^n W W_0^t W_1^t B''' C R_0 R_1} = \tilde{\sigma}_{X^n W W_0^t W_1^t B''' C} \otimes \tau_{\mathcal{R}} \otimes \tau_{\mathcal{R}}
$$

and let $S_0^\ell$ and $S_1^\ell$ be calculated as stated in the protocol. Using
the chain rule [see (16)] and the fact that $(R_0, R_1)$ are indepen-
dent, we get

$$
H_{\infty}^{\varepsilon+4\delta}(\Pi(X^n)_{\text{Enc}(W_{1-C}^t)} | S_C^\ell R_{1-C} R_C W^t W_C^t W_{1-C}^t \Pi B''', \mathcal{A}) \\
\geq \frac{1}{\omega} \frac{\lambda n}{4} - \ell - 1
$$

Using privacy amplification (Theorem II.3), we then have con-
ditioned on the event $\mathcal{A}$ that

$$
\|\tilde{\sigma}_{S_{1-C}^\ell, S_C^\ell C R_0 R_1 W^t W_C^t W_{1-C}^t \Pi B'''} - \tau_{\{0, 1\}^\ell} \otimes \tilde{\sigma}_{S_C^\ell C R_0 R_1 W^t W_C^t W_{1-C}^t \Pi B'''}\|_1 \\
\leq 2(\delta + 2\varepsilon + 8\delta) \quad (45)
$$

since

$$
\frac{1}{\omega} \frac{\lambda n}{4} - \ell - 1 \geq 2 \log 1/\delta = 2 \cdot \frac{n \lambda^2}{512 \omega^2 \beta}
$$

which follows from

$$
\ell \leq \frac{1}{\omega} \frac{\lambda n}{8} - \frac{n \lambda^2}{512 \omega^2 \beta} \cdot \frac{1}{2}
$$

Let $B^* := (R_0 R_1 W^t W_0^t W_1^t \Pi B''')$ be Bob’s part in the output
state. Since $\text{Pr}[\mathcal{A}] \geq 1 - 32\delta^2$, we get

$$
\tilde{\sigma}_{S_{1-C}^\ell S_C^\ell C B^*} \approx_{32 \delta^2 + 9\delta + 2\varepsilon} \tau_{\{0, 1\}^\ell} \otimes \tilde{\sigma}_{S_C^\ell C B^*}
$$

and

$$
\tilde{\sigma}_{S_0^\ell S_1^\ell B^*} = \rho_{S_0^\ell S_1^\ell B^*}
$$

Since $\delta^2 < \delta$, this implies the security condition for Alice, with
a total error of at most $41\delta + 2\varepsilon$.

Finally, we show that the protocol is correct when Alice and
Bob are both honest.

**Lemma V.5 (Correctness):** Protocol WSE-to-FROT satisfies
correctness with an error of

$$
43 \cdot 2^{-\frac{n \lambda^2}{512 \omega^2 \beta}} + 2 \varepsilon
$$

*Proof:* Let $\tilde{\varepsilon} := 2^{-n \lambda^2/512 \omega^2 \beta}$. We have to show that the state
$\rho_{S_0^\ell S_1^\ell C Y^\ell}$ at the end of the protocol is close to the given ideal
state $\sigma_{S_0^\ell S_1^\ell C S_C^\ell}$. Using the Hoeffding bound [57], the proba-
bility that a random subset of $[n]$ has less than $n/4$ elements
is at most $\exp(-n/8) \leq \tilde{\varepsilon}$. Hence, the probability that Bob has
to pad $s$ with $0$s (which are likely to be incorrect) when both par-
ties are honest is at most $\tilde{\varepsilon}$. Let $\mathcal{A}$ be the event that this does not
happen. It remains to show that the state $\rho_{S_0^\ell S_1^\ell C Y^\ell | \mathcal{A}}$ is close to
the given ideal state $\sigma_{S_0^\ell S_1^\ell C S_C^\ell}$. Note that the correctness con-
dition of WSE ensures that the state created by WSE is equal to
$\rho_{X^n I X_I} = \sigma_{X^n I X_I}$, where $\sigma_{X^n I} = \tau_{\{0, 1\}^n} \otimes \tau_{2^{[n]}}$. Since $I_0$
and $I_1$ are chosen independently of $X^n$, $X_{I_0}$ and $X_{I_1}$ have a
min-entropy of $n/4$ each. Since $\ell < n/8 \leq n/4 - 2 \log 1/\tilde{\varepsilon}$, it
follows from Theorem II.3 that $S_i^\ell$ is independent and $\tilde{\varepsilon}$-close
to uniform. Since dishonest Bob is only more powerful than
honest Bob, we furthermore have from Lemma V.4 that also
$S_{1-C}^\ell$ is independent and uniform except with an error of at
most $\tilde{\varepsilon}' = 41 \cdot 2^{-n \lambda^2/512 \omega^2 \beta}$, where we used the fact that Bob is also
honest during WSE $(\varepsilon = 0)$. Finally, by the same arguments as
in Lemma V.3, we have that $C$ is uniform and independent of
$S_0^\ell$ and $S_1^\ell$. Hence

$$
\rho_{S_0^\ell S_1^\ell C | \mathcal{A}} \approx_{\tilde{\varepsilon}' + \tilde{\varepsilon}} \sigma_{S_0^\ell S_1^\ell C}
$$

Since the extra condition on the permutation $\Pi$ implies that Bob
can indeed calculate $\Pi(X^n)_{\text{Enc}(W^t)}$ from $X_I$, we have that $Y^\ell =$
$S_C^\ell$. Using $\text{Pr}[\mathcal{A}] \geq 1 - \tilde{\varepsilon}$, we get

$$
\rho_{S_0^\ell S_1^\ell C Y^\ell} \approx_{2\tilde{\varepsilon} + \tilde{\varepsilon}'} \sigma_{S_0^\ell S_1^\ell C S_C^\ell}
$$

Finally, $\lambda < 1, \beta > 1$, and $\omega > 2$ give us $1/16 >$
$\lambda^2 / (512 \omega^2 \beta)$. Adding up all errors and noting that

$$
2 \cdot 2^{-m} < 2 \cdot 2^{-\frac{n \lambda^2}{512 \omega^2 \beta}}
$$

gives our claim.

# VI. CONCLUSIONS AND OPEN PROBLEMS

We have shown that secure bit commitment and oblivious
transfer can be obtained with unconditional security in the
noisy-storage model. We have connected the security of our
protocols to the information-carrying capacity of the noisy
channel describing the malicious party’s storage. We found a
natural tradeoff between the (classical) capacity of the storage
channel and the rates at which oblivious transfer and bit com-
mitment can be performed: higher noise levels lead to stronger
security.

The connection between capacities of channels and security
turns out to be directly applicable to a number of settings of
practical interest. At the same time, our study raises several im-
mediate questions concerning the exact requirements for secu-
rity in the noisy-storage model. It has already led to follow-up
work: our technique of relating security to a coding problem
has been used to construct another, simpler, protocol for obliv-
ious transfer [53], albeit at the expense of requiring significantly
more noise in memory to achieve security. Other channels have
been shown to satisfy the strong converse property, and hence
lead to security in our model [24]. Alternate forms of WSE using
high-dimensional states have been investigated using our tech-
niques to show that in the limit of large $n$, security in BSM holds
as long as a constant fraction of transmitted states is lost (i.e.,
$\nu < 1$) [40], and the security of an eavesdropper with a noisy
memory device in QKD was investigated [6]. A practical imple-
mentation is in progress [20].

**Extending security:** Clearly, it is desirable to extend the secu-
rity guarantee to a wider range of noisy channels. The limiting
factor in obtaining security from a noisy storage described by
$F = N^{\otimes \nu n}$ was the fact that we require the sufficiency condi-
tion $C_N \cdot \nu < 1/2$ to hold (see Corollary I.2), where $\nu$ is the
storage rate and $C_N$ is the classical capacity of $N$. The constant
$1/2$ is a result of using BB84-states, and stems from a corre-
sponding uncertainty relation using postmeasurement informa-
tion [41]. It is a natural question whether we can go beyond this
bound using BB84-encodings.

For channels with small classical capacity, our study reduces
security to proving a strong converse for coding. Of consider-
able practical interest are continuous-variable channels: our re-
sults are also applicable in this case, given a suitable bound on
the information-carrying capacity.

A more challenging question is to extend security to entirely
different classes of channels than considered here. Our results
are currently restricted to channels without memory. Possibly
the most important class of channels to which our results do
not apply are those with high classical capacity. This includes,
for example, the dephasing channel whose classical capacity
is 1. Security tradeoffs for such a channel are known [28] for
the case of individual storage attacks [61]. For the fully general
case considered here, it is not *a priori* clear whether small clas-
sical capacity is a necessary condition for security. Our security
proof overestimates the capabilities of the malicious party by
expressing his power purely by his ability to preserve classical
information. Completely different techniques may be required
to address this question.

Another way to extend our security analysis is to combine
our protocols with computationally secure protocols to achieve
security if the adversary either has noisy quantum storage or is
computationally bounded. This can be achieved by using com-
biners (see [25], [26], and [46]). For oblivious transfer, the same
can be achieved using the techniques in [8], [12], [14], and [66],
which only requires the use of a computationally secure bit com-
mitment scheme.

**Limits for security:** We have found sufficient conditions for
security in the noisy-storage model. For concrete channels,
these conditions give regions in the plane parameterized by
the storage rate and the noise level (cf. Fig. 1) where security
is achievable. Establishing outer bounds on the achievability
region is an interesting open problem. Corresponding necessary
conditions could become practically relevant as technology
advances.

Note that when the adversarial player is restricted to indi-
vidual storage attacks, the optimal attacks are known [54]. It is
an open problem whether the fully general coherent attacks con-
sidered here actually reduce the achievability region. In contrast,
both kinds of attacks are known to be equivalent in QKD [49].

**Efficiency and Robustness:** Our study is merely a proof of principle. For practical realiza-
tions of our protocols, the following issues need to be addressed.
Efficiency: One can reduce the amount of classical compu-
tation and communication needed to execute our protocols by
using techniques from derandomization. In particular, we could
use the constant-round IH protocol and the efficient encoding of
subsets from [19], randomness-efficient samplers (see, for e.g.,
[22]), and extractors (see, for e.g., [35], [55], and [56]) instead
of two-universal hash functions.

In practice, both the security parameter $\varepsilon$ and the number $\ell$ of
bits in the commitment or oblivious transfer are fixed constants.
Savings in communication may then be obtained by using alter-
native uncertainty relations (i.e., generalizations of (18), which
is tight [1] for $\varepsilon = 0$).

Composability: We have shown security of oblivious transfer
and bit commitment with respect to security definitions that
are motivated by composability considerations. This should en-
sure that the protocols remain secure even when executed many
times e.g., sequentially. It is, however, an open problem to show
formal composability in our model as has been done in the set-
ting of bounded-storage [21], [63]. To this end, a composability
framework for our setting needs to be established.

Robustness: We have considered an idealized setting where
the operations of the honest parties are error free. In particular,
the communication channel connecting Alice and Bob was as-
sumed to be noiseless. In real applications, both the BB84-state
preparation by (honest) Alice, the communication, and the mea-
surement of (honest) Bob will be affected by noise. To guarantee
security even in such a setting, we can apply the error-correction
techniques in [54]. However, it remains to determine the exact
tradeoff between the amount of tolerable noise of the communi-
cation channel (parameterized e.g., by the bit error rate) and the
amount of noise in the malicious player’s storage device [13].

We conclude with a few speculative remarks on potential ap-
plications of our study. Note that, in contrast to key distribution,
general two-party computation is also interesting at short (phys-
ical) distances. An example is the problem of secure identifica-
tion [17], where Alice wants to identify herself to Bob (possibly
an ATM machine) without ever giving her password away. Our
approach could be extended to realize this primitive in a sim-
ilar way as in [54]. It would be interesting to find a new and
more efficient protocol based directly on WSE. The setting of
secure identification is especially suitable for our model, since
the short distance between Alice and Bob implies that their com-
munication channel is essentially error free. At such short range,
we could also use visible light for which much better detec-
tors exists than are presently used in quantum key distribution.
Note that Alice only needs to carry a device capable of gen-
erating BB84-states and allowing her to enter her password on
a keypad. This device does not need to store any information
about Alice herself, and hence, each user could carry an iden-
tical device which is completely exchangeable among different
(trusted) users at any time. In particular, this means that Alice’s
password is not compromised even if the device is lost. Finally,
note that Alice’s technological requirements are minimal. She
only needs a device capable of generating BB84-states. This
could potentially be small enough to be carried on a key chain.

# APPENDIX
PARAMETERS FOR SAMPLING—PROOF OF LEMMA II.4:

For the proof of Lemma II.4, we first recall the definition of
a sampler.

**Definition A.1:** An $(m, \xi, \gamma)$-averaging sampler is a prob-
ability distribution over subsets $S \subseteq [m]$ with the property that
for all $(\mu_1, \ldots, \mu_m) \in [0, 1]^m$ we have

$$
\text{Pr}_S \left[ \frac{1}{|S|} \sum_{i \in S} \mu_i \leq \frac{1}{m} \sum_{i = 1}^m \mu_i - \xi \right] \leq \gamma
$$

Choosing subsets of a fixed size at random is a prime example
of a sampler; this is the sampler we will use. The parameters of
this sampler are as follows.

**Lemma A.2:** Let $s < m$ and let $\mathcal{P}_S$ be the uniform distri-
bution over subsets $S \subseteq [m]$ of size $|S| = s$. Then, $\mathcal{P}_S$ is an
$(m, \xi, 2^{-s \xi^2/2})$ sampler for every $s > 0$ and $\xi \in [0, 1]$.

*Proof:* Fix $s > 0$ and $\xi \in [0, 1]$. In [32, Lemma 2.2],
$\mathcal{P}_S$ was shown to be a $(m, \xi, e^{-s \xi^2/2})$-sampler. The claim then
follows from the fact that $e^{-s \xi^2/2} < 2^{-s \xi^2/(2 \ln 2)}$.

Replacing $H_{\min}$ by $H_{\infty}$, the following lemma follows directly
from [32, Lemma 6.15 and Lemma 6.20]. The proof follows the
same step as the proof of Theorem 6.18 in [32].

**Lemma A.3:** Let $\rho_{Z^m Q}$ be a cq-state, where $Z^m =
(Z_1, \ldots, Z_m)$ with $Z_i \in \{0, 1\}^\beta$, and let $\mathcal{P}_S$ be an $(m, \xi, \gamma)$-av-
eraging sampler supported on subsets $S$ of size $s = |S|$. Then

$$
\text{Pr}_S \left[ \frac{H_{\infty}^{\varepsilon+\gamma}(Z_S|SQ)_{\rho_S}}{s \beta} > \frac{H_{\infty}(Z^m|Q)}{m \beta} - \xi - \sqrt{\frac{\gamma}{\beta s}} \right] \geq 1 - \sqrt{\gamma}
$$

where
$C = \frac{\log 1/\gamma}{2 m \beta} + \xi + 2 \kappa \log 1/\kappa$
and $\kappa = \frac{m}{s \beta} \leq 0.15$.

Specializing Lemma A.3 to the sampler defined by Lemma
A.2 gives the following statement.

**Lemma A.4 (Min-Entropy Block Sampling [32]):** Let
$\rho_{Z^m Q}$ be a cq-state as in Lemma A.2, and let

$$
\bar{\lambda} = \frac{H_{\infty}(Z^m|Q)}{m \beta} \geq \lambda
$$

be a lower bound on the smooth min-entropy rate of $Z^m$ given
$Q$. Let $\omega > 2$ be a constant, and assume $s, \beta \in \mathbb{N}$ are such that

$$
s \geq m/4 \quad \text{and} \quad \beta \geq \max \left\{ 67, \frac{256 \omega^2}{\lambda^2} \right\} \quad (46)
$$

Let $\mathcal{P}_S$ be the uniform distributions over subsets of $[m]$ of size
$s$. Then

$$
\text{Pr}_S \left[ \frac{H_{\infty}^{\varepsilon+4\delta}(Z_S|SQ)_{\rho_S}}{s \beta} \geq \frac{\bar{\lambda}}{\omega} \right] \geq 1 - \delta^2
$$

where $\delta = 2^{-m \lambda^2/(512 \omega^2)}$.

*Proof:* Because of the definition of smooth min-entropy
and the fact that partial traces do not increase distance, it suffices
to establish the claim for $\varepsilon = 0$. By Lemma A.2 and Lemma
A.3, we have

$$
\text{Pr}_S \left[ H_{\infty}^{\varepsilon+2^{-s \xi^2/8}}(Z_S|SQ) \geq \frac{H_{\infty}(Z^m|Q)}{m} - \xi - C \right] \geq 1 - 2^{-s \xi^2/4}
$$

where
$C = \frac{\xi^2}{2 s \beta} + \frac{2 \xi^2}{m \beta} + 2 \sqrt{\kappa}$
if $\kappa = \frac{m}{s \beta} \leq 0.06$. Here, we used the inequalities

$$
\kappa \log 1/\kappa \leq \sqrt{\kappa} \quad \text{for } \kappa \leq 0.06, \\
\beta \geq 1, \quad s \leq m, \quad \text{and} \quad \xi \leq 1
$$

Note that the condition $\kappa \leq 0.06$ is satisfied if $s \geq m/4$ and
$\beta \geq 67$. Setting $\xi := \lambda/(4\omega)$ and using $s \geq m/4$ again, we get
for $256 \omega^2/\lambda^2 \leq \beta$ that

$$
C \leq \frac{\xi^2}{64 \omega \beta} + \frac{\lambda}{4 \omega} + \frac{4}{\sqrt{\beta}} \\
\leq \frac{\lambda^2}{214 \omega^2 \beta} + \frac{\lambda}{4 \omega} + \frac{\lambda}{4 \omega} \\
\leq \frac{\lambda}{\omega} \quad \text{In particular, this implies that} \\
\frac{H_{\infty}(Z^m|Q)}{m \beta} - \xi - C \geq \frac{\lambda}{\omega} \left( 1 - \frac{1}{\omega} \right) \quad (48)
$$

Combining (47), (48) with $s \xi^2 = s \lambda^2/(16 \omega^2) \geq m \lambda^2/(64 \omega^2)$
and $\delta = 2^{-m \lambda^2/(512 \omega^2)}$ gives the claim.

Instead of grouping the $m\beta$ bits into $m$ blocks $Z_i \in \{0, 1\}^\beta$,
let us now look at a normal bit string $X^{m\beta} \in \{0, 1\}^{m\beta}$. The
following lemma has been proven in [65] and shows that the
bound in Lemma A.4 can also be achieved if the sample is a
subset of size $s\beta$ chosen bitwise uniformly at random.

**Lemma A.5 (Min-Entropy Block Sampling Implies Bit
Sampling [65]):** The bound of Lemma A.3 also applies if the
sample is chosen bitwise uniformly. More generally, assume
that $s, m, \beta, \lambda, \lambda', \varepsilon, \varepsilon', \delta'$ are such that the following holds. For
all cq-states $\rho_{Z^m Q}$ with $Z^m = (Z_1, \ldots, Z_m), Z_i \in \{0, 1\}^\beta$
and $\frac{1}{m \beta} H_{\infty}^\varepsilon(Z^m|Q) \geq \lambda$, we have

$$
\text{Pr}_T \left[ \frac{H_{\infty}^{\varepsilon'}(Z_T|Q)}{s \beta} \geq \lambda' \right] \geq 1 - \delta' \quad (49)
$$

where $\mathcal{P}_T$ is the uniform distribution over subsets of $[m]$ of size
$s$. Let $\mathcal{P}_S$ be the uniform distribution over subsets of $[m\beta]$ of
size $s\beta$. We then have

$$
\text{Pr}_S \left[ \frac{H_{\infty}^{\varepsilon'}(X_S|Q)}{s \beta} \geq \lambda' \right] \geq 1 - \delta'
$$

for all cq-states $\rho_{X^{m\beta} Q}$ with $X^{m\beta} \in \{0, 1\}^{m\beta}$ and
$H_{\infty}^\varepsilon(X^{m\beta}|Q) \geq \lambda$.

*Proof:* Let $\rho_{X^{m\beta} Q}$ be a cq-state where $X^{m\beta} \in \{0, 1\}^{m\beta}$.
Let $S \subseteq [m\beta]$ be chosen uniformly at random from all subsets
of size $s\beta$ and let $T \subseteq [m\beta]$ be a random subset of size $s\beta$
chosen blockwise as in (49) (that is, after rearranging $X^{m\beta} =$
$Z^m = (Z_1, \ldots, Z_m)$ into a collection of $\beta$-bit strings). Let $\Pi$
be a permutation chosen uniformly at random, but such that it
maps all elements in $S$ into $T$. Strong subadditivity ([49, Th.
3.2.12]) implies

$$
H_{\infty}^\varepsilon(X_S|SQ) \geq H_{\infty}^\varepsilon(X_S|S\Pi Q) \\
= H_{\infty}^\varepsilon(\Pi(X^{m\beta})_T | T \Pi Q)
$$

Note that from $(S, \Pi)$, it is possible to calculate $(T, \Pi)$, and
vice-versa. Furthermore, since $\Pi$ is chosen independently of
$\rho_{X^{m\beta} Q}$, we have

$$
H_{\infty}^\varepsilon(\Pi(X^{m\beta}) | \Pi Q) = H_{\infty}^\varepsilon(X^{m\beta} | \Pi Q) = H_{\infty}^\varepsilon(X^{m\beta}|Q)
$$

Since $S$ was chosen uniformly and independently of $T$ and
$\rho_{X^{m\beta} Q}$, $\Pi$ is independent of $T$ and $\rho_{X^{m\beta} Q}$. Setting $Q' :=$
$(Q, \Pi)$, we can apply (49) to the state $\rho_{\Pi(X^{m\beta}) T Q'}$ and get a
bound on $H_{\infty}^\varepsilon(\Pi(X^{m\beta})_T | T \Pi Q)$, which then directly implies
the same bound for $H_{\infty}^\varepsilon(X_S|SQ)$.

Lemma II.4 now immediately follows by combining Lemma
A.4 with Lemma A.5.

**ACKNOWLEDGMENT**

We thank Marcos Curty, Andrew Doherty, Amir Kalev, Hoi-
Kwong Lo, Oded Regev, John Preskill and Barbara Terhal for
interesting discussions. We also thank Christian Schaffner for
discussions and comments on an earlier draft, and Dominique
Unruh for pointing out a flaw in the proof of Theorem 3.5 in an
earlier version of the paper, as well as for various other useful
suggestions.

**REFERENCES**

[1] M. Ballester, S. Wehner, and A. Winter, “State discrimination with
post-measurement information,” *IEEE Trans. Inf. Theory*, vol. 54, no.
9, pp. 4183–4198, Sep. 2008.
[2] D. Beaver, “Precomputing oblivious transfer,” in *Proc. Adv. Cryptol.
(Lecture Notes Comput. Sci.)*, 1995, vol. 963, pp. 97–109.
[3] M. Ben-Or, M. Horodecki, D. Leung, D. Mayers, and J. Oppenheim,
“The universal composable security of quantum key distribution,” in
*Proc. 2nd Theory Cryptography Conf. (Lecture Notes Comput. Sci.)*,
2005, vol. 3378, pp. 386–406.
[4] C. H. Bennett and G. Brassard, “Quantum cryptography: Public key
distribution and coin tossing,” in *Proc. IEEE Int. Conf. Comput., Syst.,
Signal Process.*, 1984, pp. 175–179.
[5] C. H. Bennett, I. Devetak, A. Harrow, P. Show, and A. Winter, “Quantum
reverse Shannon theorem,” 2009, arXiv:0912.5537.
[6] A. Bocquet, “Workshop on cryptography from storage imperfections,”
presented at the presented at the Inst. Quantum Inf. Caltech, Pasadena,
CA, Mar. 20–22, 2010.
[7] C. H. Bennett, G. Brassard, C. Crépeau, and U. Maurer, “General-
ized privacy amplification,” *IEEE Trans. Inf. Theory*, vol. 41, no. 6,
pp. 1915–1923, Nov. 1995.
[8] C. H. Bennett, G. Brassard, C. Crépeau, and H. Skubiszewska, “Prac-
tical quantum oblivious transfer,” in *Proc. Adv. Cryptol. (Lecture Notes
Comput. Sci.)*, 1992, vol. 576, pp. 351–366.
[9] C. H. Bennett, G. Brassard, and J.-M. Robert, “Privacy amplification
by public discussion,” *SIAM J. Comput.*, vol. 17, no. 2, pp. 210–229,
1988.
[10] J. L. Carter and M. N. Wegman, “Universal classes of hash functions,”
in *Proc. 9th Annu. ACM Symp. Theory Comput.*, New York, NY, 1977,
pp. 106–112.
[11] C. Crépeau, “Quantum oblivious transfer,” *J. Modern Opt.*, vol. 41, no.
12, pp. 2455–2466, 1994.
[12] C. Crépeau, P. Dumais, D. Mayers, and L. Salvail, “Computational col-
lapse of quantum state with application to oblivious transfer,” in *Proc.
Theory Cryptography Conf. (Lecture Notes Comput. Sci.)*, 2004, vol.
2951, pp. 374–393.
[13] M. Curty, H. Lo, C. Schaffner, and S. Wehner, “Implementing two-
party protocols in the noisy-storage model,” *Phys. Rev. A*, vol. 81, pp.
052336–1–052336–26, 2010.
[14] I. Damgård, S. Fehr, C. Lunemann, L. Salvail, and C. Schaffner, “Im-
proving the security of quantum protocols,” in *Proc. Adv. Cryptol. (Lec-
ture Notes Comput. Sci.)*, 2009, vol. 5677, pp. 408–427.
[15] I. B. Damgård, S. Fehr, R. Renner, L. Salvail, and C. Schaffner, “A tight
high-order entropic quantum uncertainty relation with. Applications,”
in *Proc. Adv. Cryptol. (Lecture Notes Comput. Sci.)*, 2007, vol. 4622,
pp. 360–378.
[16] I. B. Damgård, S. Fehr, L. Salvail, and C. Schaffner, “Cryptography in
the bounded-quantum-storage model,” in *Proc. 46th Annu. IEEE Symp.
Found. Comput. Sci.*, 2005, pp. 449–458.
[17] I. B. Damgård, S. Fehr, L. Salvail, and C. Schaffner, “Secure identifica-
tion and QKD in the bounded-quantum-storage model,” in *Proc. Adv.
Cryptol. (Lecture Notes Comput. Sci.)*, 2007, vol. 4622, pp. 342–359.
[18] Y. Z. Ding, “Oblivious transfer in the bounded storage model,” in
*Proc. Adv. Cryptol. (Lecture Notes Comput. Sci.)*, 2001, vol. 2139, pp.
155–170.
[19] Y. Z. Ding, D. Harnik, A. Rosen, and R. Shaltiel, “Constant-round
oblivious transfer in the bounded storage model,” in *Proc. Theory
Cryptography Conf. (Lecture Notes Comput. Sci.)*, 2004, vol. 2951,
pp. 446–472.
[20] C. Ervens, “Workshop on cryptography from storage imperfections,”
presented at the presented at the Inst. Quantum Inf. Caltech, Pasadena,
CA, Mar. 20–22, 2010.
[21] S. Fehr and C. Schaffner, “Composing quantum protocols in a classical
environment,” 2008, arXiv:0804.1059.
[22] O. Goldreich, “A sample of samplers: A computational perspective on
sampling,” 1997 [Online]. Available: http://www.eccc.uni-trier.de/eccc,
TR97-020.
[23] O. Goldreich and R. Vainish, “How to solve any protocol problem—an
efficiency improvement,” in *Proc. Adv. Cryptol. (Lecture Notes
Comput. Sci.)*, 1988, vol. 293, pp. 73–86.
[24] M. Hayashi, Personal communication with Andreas Winter, 2010.
[25] D. Harnik, J. Kilian, M. Naor, O. Reingold, and A. Rosen, “On robust
combiners for oblivious transfer and other primitives,” in *Proc. Adv.
Cryptol. (Lecture Notes Comput. Sci.)*, 2005, vol. 3494, pp. 96–113.
[26] A. Herzberg, “On tolerant cryptographic constructions,” in *Proc.
CT-RSA (Lecture Notes Comput. Sci.)*, 2005, vol. 3376, pp. 172–190.
[27] R. Impagliazzo, L. A. Levin, and M. Luby, “Pseudo-random generation
from one-way functions,” in *Proc. 21st ACM Symp. Theory Comput.*,
1989, pp. 12–24.
[28] A. Kalev and S. Wehner, Personal communication, 2008.
[29] J. Kilian, “Founding cryptography on oblivious transfer,” in *Proc. 20th
ACM Symp. Theory Comput.*, 1988, pp. 20–31.
[30] C. King, “Additivity for unital qubit channels,” *J. Math. Phys.*, vol. 43,
pp. 4641–4653, 2002.
[31] C. King, “The capacity of the quantum depolarizing channel,” *IEEE
Trans. Inf. Theory*, vol. 49, no. 1, pp. 221–229, Jan. 2003.
[32] R. König and R. Renner, “Sampling of min-entropy relative to quantum
knowledge,” *IEEE Trans. Inf. Theory*, vol. 57, no. 7, pp. 4760–4787,
Jul. 2011.
[33] R. König, R. Renner, A. Bariska, and U. Maurer, “Small accessible
quantum information does not imply security,” *Phys. Rev. Lett.*, vol.
98, pp. 140502–1–140502–4, 2007.
[34] R. König, R. Renner, and C. Schaffner, “The operational meaning of
min- and max-entropy,” 2008, arXiv:0807.1338.
[35] R. König and B. Terhal, “The bounded-storage model in the presence
of a quantum adversary,” *IEEE Trans. Inf. Theory*, vol. 54, no. 2, pp.
749–762, Feb. 2008.
[36] R. König and S. Wehner, “A strong converse for classical channel coding
using entangled inputs,” 2009, arXiv: 0903.2338.
[37] H.-K. Lo, “Insecurity of quantum secure computations,” *Phys. Rev. A*,
vol. 56, pp. 1154–1162, 1997.
[38] H.-K. Lo and H. F. Chau, “Is quantum bit commitment really pos-
sible?,” *Phys. Rev. Lett.*, vol. 78, pp. 3410–3413, 1997.
[39] H.-K. Lo and H. F. Chau, “Why quantum bit commitment and ideal
quantum coin tossing are impossible,” *Phys. D: Nonlinear Phenomena*,
vol. 120, pp. 177–187, 1998.
[40] P. Mandayam and S. Wehner, “Achieving the physical limits of the
bounded-storage model,” 2010, arXiv:1009.1596.
[41] H. Maassen and J. Uffink, “Generalized entropic uncertainty relations,”
*Phys. Rev. Lett.*, vol. 60, pp. 1103–1106, 1988.
[42] U. Maurer, “A provably-secure strongly-randomized cipher,” in *Proc.
Adv. Cryptol. (Lecture Notes Comput. Sci.)*, 1990, vol. 473, pp.
361–373.
[43] U. Maurer, “Conditionally-perfect secrecy and a provably-secure ran-
domized cipher,” *J. Cryptol.*, vol. 5, no. 1, pp. 53–66, 1992.
[44] D. Mayers, “The trouble with quantum bit commitment,” 1996, quant-ph/
9603015.
[45] D. Mayers, “Unconditionally secure quantum bit commitment is im-
possible,” *Phys. Rev. Lett.*, vol. 78, pp. 3414–3417, 1997.
[46] R. Meier, B. Przydatek, and J. Wullschleger, “Robuster combiners for
oblivious transfer,” in *Proc. 4th Theory Cryptography Conf.*, 2007, pp.
404–418.
[47] R. Ostrovsky, R. Venkatesan, and M. Yung, “Fair games against an all-
powerful adversary,” in *DIMACS: Series in Discrete Mathematics and
Theoretical. Computer Science*. Providence, RI: Amer. Math. Soc.,
pp. 155–169, 1991.
[48] I. S. Reed and G. Solomon, “Polynomial codes over certain finite
fields,” *J. Soc. Ind. Appl. Math.*, vol. 8, no. 2, pp. 300–304, 1960.
[49] R. Renner, “Security of Quantum Key Distribution,” Ph.D. dissertation,
Dept. of Physics, ETH Zurich, Zurich, Switzerland, 2005.
[50] R. Renner and R. König, “Universally composable privacy amplifica-
tion against quantum adversaries,” in *Proc. Theory Cryptography Conf.
(Lecture Notes Comput. Sci.)*, 2005, vol. 3378, pp. 407–425.
[51] G. Savvides, “Interactive Hashing and Reductions Between Oblivious
Transfer,” Ph.D. dissertation, Dept. of Computer Science, McGill Uni-
versity, Montreal, QC, Canada, 2007.
[52] C. Schaffner, “Cryptography in the Bounded-Quantum-Storage
Model,” Ph.D. dissertation, Dept. of Computer Science, University of
Aarhus, Aarhus, Denmark, 2007.
[53] C. Schaffner, “Simple protocols for oblivious transfer and secure iden-
tification in the noisy-quantum-storage model,” 2010, arXiv:1002.1495.
[54] C. Schaffner, B. Terhal, and S. Wehner, “Robust cryptography in the
noisy-quantum-storage model,” *Quantum Inf. Comput.*, vol. 9, pp.
0963–0996, 2009.
[55] R. Shaltiel, “Recent developments in explicit constructions of extrac-
tors,” *Bull. EATCS*, vol. 77, pp. 67–95, 2002.
[56] A. Ta-Shma, “Short seed extractors against quantum storage,” 2008,
quant-ph/0808.1994.
[57] W. Uhlmann, “Probability inequalities for sums of bounded random
variables,” *J. Amer. Statist. Assoc.*, vol. 58, no. 301, pp. 13–30, Mar.
1963.
[58] S. Vadhan, “On constructing locally computable extractors and cryp-
tosystems in the bounded storage model,” in *Proc. Adv. Cryptol. (Lec-
ture Notes Comput. Sci.)*, 2003, pp. 61–77.
[59] S. Wehner, “Cryptography in a quantum world,” Ph.D. dissertation,
School of Sciences, University of Amsterdam, Amsterdam, The
Netherlands, 2008, arXiv:0806.3483.
[60] S. Wehner, C. Schaffner, and B. Terhal, “Cryptography from noisy pho-
tonic storage,” 2007, arxiv:0711.2895.
[61] S. Wehner, C. Schaffner, and B. M. Terhal, “Cryptography from noisy
storage,” *Phys. Rev. Lett.*, vol. 100, no. 22, pp. 220502–1–220502–4,
2008.
[62] S. Wehner and A. Winter, “Higher entropic uncertainty relations
for anti-commuting observables,” *J. Math. Phys.*, vol. 49, pp.
062105–1–062105–11, 2008.
[63] S. Wehner and J. Wullschleger, “Composable security in the bounded-
quantum-storage model,” in *Proc. Int. Collaq. Automata Languages
Programming*, 2008, pp. 604–615.
[64] S. Wiesner, “Conjugate coding,” *Sigact News*, vol. 15, no. 1, pp. 78–88,
1983.
[65] J. Wullschleger, “Bitwise quantum min-entropy sampling and new lower
bounds for random access codes,” 2010, arXiv:1012.2291.
[66] A. C.-C. Yao, “Security of quantum protocols against coherent mea-
surements,” in *Proc. 20th ACM Symp. Theory Comput.*, 1995, pp.
67–75.

**Robert König** studied theoretical physics at ETH Zurich, Switzerland. He re-
ceived his PhD from DAMTP at Cambridge University, UK in 2007, and subse-
quently worked as a postdoc at the Institute for Quantum Information of the Cal-
ifornia Institute of Technology. He is currently pursuing postdoctoral research
at IBM Watson.

**Stephanie Wehner** is an Assistant Professor at the School of Computing, Na-
tional University of Singapore, and a Principal Investigator at the Centre for
Quantum Technologies. From 2008 to 2010, she was a Postdoctoral Scholar at
the Institute for Quantum Information, Caltech. Stephanie received her Ph.D.
from the University of Amsterdam in 2008.

**Jürg Wullschleger** has been a postdoctoral researcher at the Université de Mon-
tréal since December 2010. He was born July 5, 1975, in Zug, Switzerland. He
received his Ph.D. in Computer Science in 2007 from the ETH Zürich, Switzer-
land, under the supervision of Professor Stefan Wolf. After that, he was a post-
doctoral fellow at the University of Bristol for three years.