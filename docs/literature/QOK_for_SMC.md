arXiv:1909.11701v2 [quant-ph] 17 Jun 2020
Generation and Distribution of Quantum Oblivious Keys for
Secure Multiparty Computation

Mariano Lemus¹,², Mariana F. Ramos³,⁴, Preeti Yadav¹,², Nuno A. Silva³,⁴, Nelson J.
Muga³,⁴, André Souto⁵, Nikola Paunković¹,², Paulo Mateus¹,², and Armando N. Pinto³,⁴

¹Instituto de Telecomunicacões, Lisbon, Portugal
² Departamento de Matemática, Instituto Superior Técnico, Av. Rovisco Pais, Lisbon, Portugal
³ Instituto de Telecomunicacões, University of Aveiro, Campus Universitário de Santiago, 3810-193, Aveiro, Portugal
⁴ Department of Electronics, Telecommunications and Informatics, University of Aveiro, Portugal
⁵ Departamento de Informática, Faculdade de Ciências da Universidade de Lisboa, Lisbon, Portugal

June 18, 2020

Abstract
The oblivious transfer primitive is sufficient to implement secure multiparty computation. However,
secure multiparty computation based on public-key cryptography is limited by the security and efficiency
of the oblivious transfer implementation. We present a method to generate and distribute oblivious keys
by exchanging qubits and by performing commitments using classical hash functions. With the presented
hybrid approach, quantum and classical, we obtain a practical and high-speed oblivious transfer protocol.
We analyse the security and efficiency features of the technique and conclude that it presents advantages
in both areas when compared to public-key based techniques.

# 1 Introduction

In Secure Multiparty Computation (SMC), several agents compute a function that depends on their own
inputs, while maintaining them private [1]. Privacy is critical in the context of an information society, where
data is collected from multiple devices (smartphones, home appliances, computers, street cameras, sensors,
...) and subjected to intensive analysis through data mining. This data collection and exploration paradigm
offers great opportunities, but it also raises serious concerns. A technology able to protect the privacy
of citizens, while simultaneously allowing to profit from extensive data mining, is going to be of utmost
importance. SMC has the potential to be that technology if it can be made practical, secure and ubiquitous.
Current SMC protocols rely on the use of asymmetric cryptography algorithms [2], which are considered
significantly more computationally complex compared with symmetric cryptography algorithms [3]. Besides
being more computationally intensive, in its current standards, asymmetric cryptography cannot be con-
sidered secure anymore due to the expected increase of computational power that a large-scale quantum
computer will bring [4]. Identifying these shortcomings in efficiency and security motivates the search for
alternative techniques for implementing SMC without the need of public key cryptography.

## 1.1 Secure Multiparty Computation and Oblivious Transfer

Consider a set of N agents and $f(x_1, x_2, ..., x_N) = (y_1, y_2, ..., y_N)$ a multivariate function. For $i \in \{1, ..., N\}$,
a SMC service (see Figure 1) receives the input $x_i$ from the $i$-th agent and outputs back the value $y_i$ in such
a way that no additional information is revealed about the remaining $x_j, y_j$, for $j \neq i$. Additionally, this
definition can be strengthened by requiring that for some number $M < N$ of corrupt agents working together,

[IMAGE: Figure 1: Diagram showing N parties computing a function (Y₁, Y₂, ..., YN)=f(X₁, X₂, ..., XN)]

Figure 1: In secure multiparty computation, N parties compute a function preserving the privacy of their
own input. Each party only has access to their own input-output pair.

no information about the remaining agents gets revealed (secrecy). It can also be imposed that if at most
$M' < N$ agents do not compute the function correctly, the protocol identifies it and aborts (authenticity).
Some of the most promising approaches towards implementing SMC are based on oblivious circuit eval-
uation techniques such as Yao's garbled circuits for the two party case [5] and the GMW or BMR protocols
for the general case [2,6-8]. It has been shown that to achieve SMC it is enough to implement the Oblivious
Transfer (OT) primitive and, without additional assumptions, the security of the resulting SMC depends
only on that of the OT [9]. In the worst case, this requires each party to perform one OT with every other
party for each gate of the circuit being evaluated. This number can be reduced by weakening the security
or by increasing the amount of exchanged data [10]. Either way, the OT cost of SMC represents a major
bottleneck for its practical implementation. Finding fast and secure OT protocols, hence, is a very relevant
task in the context of implementing SMC.
Let Alice and Bob be two agents. A 1-out-of-2 OT service receives bits $b_0, b_1$ as input from Alice and a
bit $c$ as input from Bob, then outputs $b_c$ to Bob. This is done in a way that Bob gets no information about
the other message, i.e. $b_{1-c}$, and Alice gets no information about Bob's choice, i.e. the value of $c$ [11].

## 1.2 State of the art

Classical OT implementations are based on the use of asymmetric keys, and suffer from two types of prob-
lems. The first one is the efficiency: asymmetric cryptography relies on relatively complex key generation,
encryption, and decryption algorithms [12, Chapter 1] [13, Chapter 6]. This limits achievable rates of
OTs, and since implementations of SMC require a very large number of OTs [10] [3], this has hindered
the development of SMC-based applications. The other serious drawback is that asymmetric cryptography,
based on integer number factorization or discrete-logarithm problems, is insecure in the presence of quantum
computers, and therefore, it has to be progressively abandoned. There are strong research efforts in order
to find other hard problems that can support asymmetric cryptography [4]. However, the security of these
novel solutions is still not fully understood.
A possible way to circumvent this problem is by using quantum cryptography to improve the efficiency
and security of current techniques. Quantum solutions for secure key distribution, Bit Commitment (BC)
and OT have been already proposed [14]. The former was proved to be unconditionally secure (assuming an
authenticated channel) and realizable using current technology. Although, it was shown to be impossible to
achieve unconditionally secure quantum BC and OT [15] [16] [17], one can impose restrictions on the power
of adversaries in order to obtain practically secure versions of these protocols [18,19]. These assumptions
include physical limitations on the apparatuses, such as noisy or bounded quantum memories [20-22]. For
instance, quantum OT and BC protocols have been developed and implemented (see [23-25]) under the

noisy storage model. Nevertheless, solutions based on hardware limitations may not last for long, because as
quantum technology improves the rate of secure OT instances will decrease. Other solutions include exploring
relativistic scenarios using the fact that no information can travel faster than light [26-28]. However, at the
moment, these solutions do not seem to be practical enough to allow the large dissemination of SMC.
In this work, we explore the resulting security and efficiency features of implementing oblivious transfer
using a well known quantum protocol [29] supported by using a cryptographic hash based commitment
scheme [30]. We call it a hybrid approach, since it mixes both classical and quantum cryptography. We
analyse the protocol stand alone security, as well as its composable security in the random oracle model.
Additionally, we study its computational complexity and compare it with the complexity of alternative public
key based protocols. Furthermore, we show that, while unconditional information-theoretic security cannot
be achieved, there is an advantage (both in terms of security and efficiency) of using quantum resources in
computationally secure protocols, and as such, they are worth consideration for practical tasks in the near
future.
This paper is organized as follows. In Section II, we present a quantum protocol to produce OT given
access to a collision resistant hash function, define the concept of oblivious keys, and explain how having
pre-shared oblivious keys can significantly decrease the computational cost of OT during SMC. The security
and efficiency of the protocol is discussed in Section III. Finally, in Section IV we summarize the main
conclusions of this work.

# 2 Methods

## 2.1 Generating the OTs

In this section, we describe how to perform oblivious transfer by exchanging qubits. The protocol $\pi_{QOT}$
shown in Figure 2 is the well known quantum oblivious transfer protocol first introduced by Yao, which
assumes access to secure commitments. The two logical qubit states $|0\rangle$ and $|1\rangle$ represent the computational
basis, and the states $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$, $|-\rangle = (|0\rangle - |1\rangle)/\sqrt{2}$ represent the Hadamard basis. We also
define the states $|(s_i, a_i)\rangle$ for $s_i, a_i \in \{0,1\}$ according to the following rule:

$|(0,0)\rangle = |0\rangle \quad |(0,1)\rangle = |+\rangle$
$|(1,0)\rangle = |1\rangle \quad |(1,1)\rangle = |-\rangle$.

Note that these states can be physically instantiated using, for instance, a polarization encoding fiber op-
tic quantum communication system, provided that a fast polarization encoding/decoding process and an
algorithm to control random polarization drifts in optical fibers are available [31,32].
Intuitively, this protocol works because the computational and the Hadamard are conjugate bases. Per-
forming a measurement in the preparation basis of a state, given by $a_i$, yields a deterministic outcome,
whereas measuring in the conjugate basis, given by $\bar{a}_i$, results in a completely random outcome. By prepar-
ing and measuring in random bases, as shown in steps 1 and 2, approximately half of the measurement
outcomes will be equal to the prepared states, and half of them will have no correlation. As Alice sends
the information of preparation bases to Bob in step 6, he gets to know which of his bits are correlated with
Alice's. During steps 3 to 6, Bob commits the information of his measurement basis and outcomes to Alice,
who then chooses a random subset of them to test for correlations. Passing this test (statistically) ensures
that Bob measured his qubits as stated in the protocol as opposed to performing a different (potentially
joint) measurement. Such strategy may extract additional information from Alice's strings, but would fail to
pass the specific correlation check in step 6. At step 8, Bob separates his non-tested measurement outcomes
in two groups: $I_0$ where he measured in the same basis as the preparation one, and $I_1$, in which he measured
in the different basis. He then inputs his bit choice $c$ by selecting the order in which he sends the two sets
to Alice. During step 9, Alice encrypts her first and second input bits with the preparation bits associated
with the first and second second sets sent by Bob respectively. This effectively hides Bob's input bit because
she is ignorant about the measurements that were not opened by Bob (by the security of the commitment
scheme). Finally, Bob can decrypt only the bit encrypted with the preparation bits associated to $I_c$.

Protocol $\pi_{QOT}$
Parameters: Integers $n, m < n$.
Parties: The sender Alice and the receiver Bob.
Inputs: Alice gets two bits $b_0, b_1$ and Bob gets a bit $c$.
(Oblivious key distribution phase)
1. Alice samples $s, a \in \{0,1\}^{n+m}$. For each $i < n + m$ she prepares the state $|\phi_i\rangle = |(s_i, a_i)\rangle$ and sends $|\phi\rangle =
$|\phi_1\phi_2 ... \phi_{n+m}\rangle$ to Bob.
2. Bob samples $\bar{a} \in \{0,1\}^{n+m}$ and, for each $i$, measures $|\phi_i\rangle$ in the computational basis if $\bar{a}_i = 0$, otherwise
measures it in the Hadamard basis. Then, he computes the string $\bar{s} = \bar{s}_1\bar{s}_2 ... \bar{s}_{n+m}$, where $\bar{s}_i = 0$ if the
outcome of measuring $|\phi_i\rangle$ was $0$ or $+$, and $\bar{s}_i = 1$ if it was $1$ or $-$.
3. For each $i$, Bob commits $(\bar{s}_i, \bar{a}_i)$ to Alice.
4. Alice chooses randomly a set of indices $T \subset \{1, ..., n + m\}$ of size $m$ and sends $T$ to Bob.
5. For each $j \in T$, Bob opens the commitments associated to $(\bar{s}_j, \bar{a}_j)$.
6. Alice checks if $s_j = \bar{s}_j$ whenever $a_j = \bar{a}_j$ for all $j \in T$. If the test fails Alice aborts the protocol, otherwise she
sends $a^* = a|_T$ to Bob and sets $k = s|_{\bar{T}}$.
7. Bob computes $x = a^* \oplus \bar{a}|_{\bar{T}}$ and $\bar{k} = \bar{s}|_{\bar{T}}$.
(Oblivious transfer phase)
8. Bob defines the two sets $I_0 = \{i | x_i = 0\}$ and $I_1 = \{i | x_i = 1\}$. Then, he sends to Alice the ordered pair
$(I_c, I_{c \oplus 1})$.
9. Alice computes $(e_0, e_1)$, where $e_i = b_i \bigoplus_{j \in I_c} k_j$, and sends it to Bob.
10. Bob outputs $b_c = e_c \bigoplus_{j \in I_0} \bar{k}_j$.

Figure 2: Quantum OT protocol based on secure commitments. The $\bigoplus$ denotes the bit XOR of all the
elements in the family.

In real implementations of the protocol one should consider imperfect sources, noisy channels, and mea-
surement errors. Thus, in step 6 Alice should perform parameter estimation for the statistics of the measure-
ments, and pass whenever the error parameter $e_s$ below some previously fixed value. Following this, Alice
and Bob perform standard post-processing techniques of information reconciliation and privacy amplifica-
tion before continuing to step 7. These techniques indeed work even in the presence of a dishonest Bob. As
long as he has some minimal amount of uncertainty about Alice's preparation string $s$, an adequate privacy
amplification scheme can be used to maximize Bob's uncertainty of one of Alice's input bits. This comes
at the cost of increasing the amount of qubits shared per OT [33]. An example of these techniques applied
in the context of the noisy storage model (where the commitment based check is replaced by a time delay
under noisy memories) can be found in [19].

## 2.2 Oblivious key distribution

In order to make the quantum implementation of OT more practical during SMC we introduce the concept
of oblivious keys. The protocol $\pi_{QOT}$ can be separated in two phases: the *Oblivious Key Distribution* phase
which consists of steps 1 to 7 and forms the $\pi_{OKD}$ subprotocol, and the *Oblivious Transfer* phase which
takes steps 8 to 10 and we denote as the $\pi_{OK \to OT}$ subprotocol. Note that after step 7 of $\pi_{QOT}$ the subsets
$I_0, I_1$ have not been revealed to Alice, so she has no information yet on how the correlated and uncorrelated
bits between $k$ and $\bar{k}$ are distributed (recall that $k$ and $\bar{k}$ are the result of removing the tested bits from

[IMAGE: Figure 3: Diagram illustrating the shared oblivious key pair between Alice (key k) and Bob (key $\bar{k}$ and bit string x). Shows correlated (x=0, left box) and uncorrelated (x=1, right box) bits.]

Figure 3: Oblivious keys. Alice has the string $k$ and Bob the string $\bar{k}$. For each party, the boxes in the left
and right represent the bits of their string associated to the indices $i$ for which $x_i$ equals 0 (left box) or 1
(right box). Alice knows the entire key, Bob only knows half of the key, but Alice does not know which half
Bob knows.

the strings $s$ and $\bar{s}$ respectively). On the other hand, after receiving Alice's preparation bases, Bob does
know the distribution of correlated and uncorrelated bits between $k$ and $\bar{k}$, which is recorded in the string $x$
($x_i = 0$ if $a_i = \bar{a}_i$, otherwise $x_i = 1$). Note that until step 7 of the protocol all computation is independent
of the input bits $e_0, e_1, c$. Furthermore, from step 8, only the strings $k, \bar{k}$, and $x$ are needed to finish the
protocol (in addition to the input bits). We call these three strings collectively an *oblivious key*, depicted
in Figure 3. Formally, let Alice and Bob be two agents. Oblivious Key Distribution (OKD) is a service
that outputs to Alice the string $k = k_1k_2 ... k_{\ell}$ and to Bob the string $\bar{k} = \bar{k}_1\bar{k}_2 ... \bar{k}_{\ell}$ together with the bit
string $x = x_1x_2 ... x_{\ell}$, such that $k_i = \bar{k}_i$ whenever $x_i = 0$ and $k_i$ does not give any information about $\bar{k}_i$
whenever $x_i = 1$. All of the strings are chosen at random for every invocation of the service. A pair $(k, (\bar{k}, x))$
distributed as above is what we call an *oblivious key pair*. Alice, who knows $k$, is referred to as the sender,
and Bob, who holds $\bar{k}$ and $x$, is the receiver. In other words, when two parties share an oblivious key, the
sender holds a string $k$, while the receiver has only approximately half of the bits of $k$, but knows exactly
which of those bits he has.
When two parties have previously shared an oblivious key pair, they can securely produce OT by perform-
ing the steps $\pi_{OK \to OT}$ of $\pi_{QOT}$. This is significantly faster than current implementations of OT without any
previous shared resource and does not require quantum communication during SMC. Note that the agents
can perform, previously or concurrently, an OKD protocol to share a sufficiently large oblivious key, which
can be then partitioned and used to perform as many instances of OT as needed for SMC.
Fortunately, it is possible to achieve fast oblivious key exchange if the parties have access to fast and
reliable quantum communications and classical commitments. In order to use this QOT protocol, the com-
mitment scheme must be instantiated. Consider the commitment protocol $\pi_{COMH}$ shown in Figure 4, first
introduced by Halevi and Micali. It uses a combination of universal and cryptographic hashing, the former
to ensure statistical uniformity on the commitments, and the latter to hide the committed message. The mo-
tivation for the choice of this protocol for this task will become more apparent during the following sections
as we discuss the security and efficiency characteristics of the composition of $\pi_{QOT}$ with $\pi_{COMH}$, henceforth
referred as the $\pi_{HOK}$ (for Hybrid Oblivious Key) protocol for OT.
The existence of a reduction from OT to commitments, while proven within quantum cryptography
through the $\pi_{QOT}$ protocol, is an open problem in classical cryptography. The existence of commitment
schemes such as $\pi_{COMH}$, which do not rely on asymmetric cryptography, provides a way to obtain OT in
the quantum setting while circumventing the disadvantages of asymmetric cryptography.

Protocol $\pi_{COMH}$
Parameters: Message length $\bar{n}$ and security parameter $k$. A universal hash family $F = \{f : \{0,1\}^l \to \{0,1\}^{\bar{n}}\}$,
with $l = 4k + 2\bar{n} + 4$. A collision resistant hash function $H$.
Parties: The verifier Alice and the committer Bob.
Inputs: Bob gets a string $m$ of length $\bar{n}$.
(Commit phase)
1. Bob samples $r \in \{0,1\}^l$, computes $y = H(r)$, and chooses $f \in F$, such that $f(r) = m$. Then, he sends $(f, y)$ to
Alice.
(Open phase)
2. Bob sends $r$ to Alice.
3. Alice checks that $H(r) = y$. If this test fails she aborts the protocol. Otherwise, she outputs $f(r)$.

Figure 4: Commitment protocol based on collision resistant hash functions

# 3 Results and discussion

## 3.1 Security

In this section, we analyse the security of the proposed composition of protocols. The main result is encap-
sulated in the following theorem.

**Theorem 3.1.** The protocol $\pi_{HOK}$ is secure as long as the hash function is collision resistant. Moreover,
if the hash function models a Random Oracle, a simple modification of the protocol can make it universally-
composable secure.

Proof. The security proof relies on several well-established results in cryptography. First, notice that the
$\pi_{HOK}$ protocol is closely related to the standard Quantum OT protocol $\pi_{QOT}$, which is proven statisti-
cally secure in Yao's original paper [34] and later universally composable in the quantum composability
framework [35]. The difference between the two is that $\pi_{QOT}$ uses ideal commitments, as opposed to the
hash-based commitments in $\pi_{HOK}$. We start by showing that the protocol $\pi_{HOK}$ is standalone secure. For
this case, we only need to replace the ideal commitment of $\pi_{QOT}$ with a standalone secure commitment
protocol, such as the Halevi and Micali [30], which is depicted in $\pi_{COMH}$. Since the latter is secure when-
ever the hash function is collision resistant, we conclude that $\pi_{HOK}$ is secure whenever the hash function is
collision resistant.
Finally, we provide the simple modification of $\pi_{HOK}$ that makes it universally-composable secure when the
hash function models a Random Oracle. The modification is only required to improve upon the commitment
protocol, as Yao's protocol with ideal commitments is universally-composable [35]. Indeed, we need to
consider universally composable commitment scheme instead of $\pi_{COMH}$. This is achieved by the HMQ
construction [36] which, given a standalone secure commitment scheme and a Random Oracle, outputs a
universally-composable commitment scheme, which is perfectly hiding and computationally binding, that is,
secure as far as collisions cannot be found. So we just need to replace $\pi_{COMH}$ with the output of the HMQ
construction, when $\pi_{COMH}$ and $H$ are given as inputs and $H$ models a Random Oracle. $\blacksquare$
Regarding the above theorem we note that, for the composable security, the HMQ construction mentioned
in the proof formally requires access to a random oracle, which is an abstract object used for studying
security and cannot be realized in the real world. Hence, we leave it as an additional security property, as
hash functions are traditionally modelled as random oracles. Stand alone security of the $\pi_{HOK}$ protocol does
*not* require the hash function to be a random oracle.
The use of collision resistant hash functions is acceptable in the quantum setting, as it has been shown
that there exist functions for which a quantum computer does not have any significant advantage in finding
collisions when compared with a classical one [37]. One point to note about the security of $\pi_{OKD}$ is that

it is not susceptible to intercept now-decrypt later style of attacks. Bob can attempt an attack in which he
does not properly measure the qubits sent by Alice at step 2, and instead waits until Alice reveals the test
subset in step 4 to measure honestly only those qubits. For that he must be able to control the openings of
the commitment scheme such that Alice opens the values of his measurement outcomes for those qubits. In
order to do this, he must be able find collisions for $H$ before step 5. This means that attacking the protocol
by finding collisions of the hash function is only effective if it is done in real time, that is, between steps 3
and 5 of the protocol. This is in contrast to asymmetric cryptography based OT, in which Bob can obtain
both bits if he is able to overcome the computational security at a later stage.
Finally, we point out that the OT extension algorithms that are used during SMC often rely only on
collision resistant hash functions [38] anyway. If those protocols are used to extend the base OTs produced
by $\pi_{HOK}$, we can effectively speed up the OT rates without introducing any additional computational
complexity assumption.

## 3.2 Efficiency

Complexity-wise, the main problem with public-key based OT protocols is that they require a public/private
key generation, encryption, and decryption per transfer. In the case of RSA and ElGamal based algorithms,
this has complexity $O(n^{2.58})$ (where $N = 2^n$ is the size of the group), using Karatsuba multiplication and
Berett reduction for Euclidian division [39]. Post-quantum protocols are still ongoing optimization, but
recent results show RLWE key genereration and encryption in time $O(n^2 \log(n))$ [40].
To study the time complexity of the $\pi_{HOK}$ protocol, consider first the complexity of $\pi_{COMH}$. It requires
two calls of $H$ and one call of the universal hash family $F$, $\bar{n}$ bit comparisons (if using the technique proposed
in [30] to find the required $f$), and one additional evaluation of $f$. Cryptographic hash functions are designed
so that their time complexity is linear on the size of the input, which in this case is $l = 4k + 2\bar{n} + 4$. To
compute the universal hashing, the construction in [30] requires $\bar{n}k$ binary multiplications. Thus, the running
time of $\pi_{COMH}$ is linear on the security parameter $k$. On the other hand, $\pi_{QOT}$ has two security parameters:
$n$, associated to the size of the keys used to encrypt the transferred bits, and $m$, associated to the security
of the measurement test done by Alice. The protocol requires $n + m$ qubit preparations and measurements,
$n + m$ calls of the commitment scheme, and $n$ bit comparisons. This leads to an overall time complexity of
$O(k(n + m))$ for the $\pi_{HOK}$ protocol, which is linear in all of its security parameters.
In realistic scenarios, however, error correction and privacy amplification must be implemented during
the $\pi_{OK \to OT}$. For the former, LDPC codes [41] or the cascade algorithm [42] can be used, and the latter can
be done with universal hashing. For a given channel error parameter, these algorithms have time complexity
linear in the size of the input string, which in our case is $n$. Hence, $\pi_{HOK}$ stays efficient when considering
channel losses and preparation/measurement errors.
One of the major bottlenecks in the GMW protocol for SMC is the number of instances of OT required
(it is worth noting that GMW uses 1-out-of-4 OT, which can efficiently be obtained from two instances of
the 1-out-of-2 OT [43]). A single Advanced Encryption Standard (AES) circuit can be obtained with the
order of $10^6$ instances of OT. However, with current solutions, i.e., with computational implementations of
OT based on asymmetric classical cryptography, one can generate $\sim 10^3$ secure OTs per second in standard
devices [44]. It is possible to use OT extension algorithms to increase its size up to rates of the order of $10^6$
OT per second [3]. Several of such techniques are based on symmetric cryptography primitives [44], such as
hash functions, and could also be used to extend the OTs generated by $\pi_{HOK}$.
Due to the popularity of crypto-currencies, fast and efficient hashing machines have recently become more
accessible. Dedicated hashing devices are able to compute SHA-256 at rates of $10^{12}$ hashes per second (see
Bitfury, Ebit, and WhatsMiner, for example). In addition, existent standard Quantum Key Distribution
(QKD) setups can be adapted to implement OKD, since both protocols share the same requirements for the
generation and measurement of photons. Notably, QKD setups have already demonstrated secret key rates
of the order of $10^6$ bits per second [45-49]. It is also worth mentioning that, as opposed to QKD, OKD
is useful even in the case when Alice and Bob are at the same location. This is because in standard key
distribution the parties trust each other and, if at the same location, they can just exchange hard drives
with the shared key, whereas when sharing oblivious keys, the parties do not trust each other and need a

protocol that enforces security. Thus, for the cases in which both parties being at the same location is not
an inconvenience, the oblivious key rates can be further raised, as the effects of channel noise are minimized.
Direct comparisons of OT generation speed between asymmetric cryptography techniques and quantum
techniques are difficult because the algorithms run on different hardware. Nevertheless, as quantum technolo-
gies keep improving, the size and cost of devices capable of implementing quantum protocols will decrease
and their use can result in significant improvements of OT efficiency, in the short-to-medium term future.

# 4 Conclusions

Motivated by the usefulness of SMC as a privacy-protecting data mining tool, and identifying its OT cost
as its main implementation challenge, we have proposed a potential solution for practical implementation
of OT as a subroutine SMC. The scheme consists on pre-sharing an oblivious key pair and then using it
to compute fast OT during the execution of the SMC protocol. We call this approach hybrid because it
uses resources traditionally associated with classical symmetric cryptography (cryptographic hash functions),
as well as quantum state communication and measurements on conjugate observables, resources associated
with quantum cryptography. The scheme is secure as far as the chosen hash function is secure against
quantum attacks. In addition, we showed that the overall time complexity of $\pi_{HOK}$ is linear on all its
security parameters, as opposed to the public-key based alternatives, whose time complexities are at least
quadratic on their respective parameters. Finally, by comparing the state of current technology with the
protocol requirements, we concluded that it has the potential to surpass current asymmetric cryptography
based techniques.
It was also noted that current experimental implementations of standard discrete-variable QKD can be
adapted to perform $\pi_{HOK}$. The same post-processing techniques of error correction and privacy amplification
apply, however, fast hashing subroutines should be added for commitments during the parameter estimation
step. Future work includes designing an experimental setup, meeting the implementation challenges, and
experimentally testing the speed, correctness, and security of the resulting oblivious key pairs. This includes
computing oblivious key rate bounds for realistic scenarios and comparing them with current alternative
technologies. Real world key rate comparisons can help us understand better the position of quantum
technologies in the modern cryptographic landscape.
Regarding the use of quantum cryptography during the commitment phase; because of the impossibility
theorem for unconditionally secure commitments in the quantum setting [17], one must always work with an
additional assumption on top of needing quantum resources. The noisy storage model provides an example in
which the commitments are achieved by noisy quantum memories [21,50,51]. The drawback of this particular
assumption is the fact that advances in quantum storage technology work against the performance of the
protocol, which is not a desired feature. The added cost of using quantum communication is a disadvantage.
So far, to the knowledge of the authors, there are no additional practical quantum bit commitment protocols
that provide advantages in security or efficiency compared to classical ones once additional assumptions (such
as random oracles, common reference strings, computational hardness, etc.,) are introduced. Nevertheless,
we are optimistic that such protocols can be found in the future, perhaps by clever design, or by considering
a different a kind of assumption outside of the standard ones.

## Acknowledgments

This work is supported by the FundaÃ§Ã£o para a CiÃªncia e a Tecnologia (FCT) through national funds,
by FEDER, COMPETE 2020, and by Regional Operational Program of Lisbon, under UIDB/50008/2020,
UIDP/50008/2020, UID/CEC/00408/2013, POCI-01-0145-FEDER-031826, POCI-01-0247-FEDER-039728,
PTDC/CCI-CIF/29877/2017, PD/BD/114334/2016, PD/BD/113648/2015, and CEECIND/04594/2017.

# References

[1] Y. Lindell and B. Pinkas, “Secure multiparty computation for privacy-preserving data mining," Journal
of Privacy and Confidentiality, no. 1, pp. 59-98, 2009.
[2] P. Laud and L. Kamm, Applications of secure multiparty computation. Ios Press, 2015, vol. 13.
[3] G. Asharov, Y. Lindell, T. Schneider, and M. Zohner, "More efficient oblivious transfer extensions,"
Journal of Cryptology, vol. 30, no. 3, pp. 805-858, 2017.
[4] D. J. Bernstein and T. Lange, “Post-quantum cryptography,” Nature, vol. 549, pp. 188 EP –, Sep 2017.
[Online]. Available: https://doi.org/10.1038/nature23461
[5] A. C. C. Yao, “How to generate and exchange secrets,” in 27th Annual Symposium on Foundations of
Computer Science (sfcs 1986), Oct 1986, pp. 162-167.
[6] O. Goldreich, S. Micali, and A. Wigderson, "How to play any mental game," in Proceedings of the
Nineteenth Annual ACM Symposium on Theory of Computing, ser. STOC '87. New York, NY, USA:
ACM, 1987, pp. 218-229. [Online]. Available: http://doi.acm.org/10.1145/28395.28420
[7] T. Schneider and M. Zohner, GMW vs. Yao? Efficient Secure Two-Party Computation with Low Depth
Circuits. Berlin, Heidelberg: Springer Berlin Heidelberg, 2013, pp. 275-292.
[8] D. Beaver, S. Micali, and P. Rogaway, “The round complexity of secure protocols," in Proceedings of the
twenty-second annual ACM symposium on Theory of computing, 1990, pp. 503-513.
[9] J. Kilian, "Founding cryptography on oblivious transfer," in Proceedings of the Twentieth Annual ACM
Symposium on Theory of Computing, ser. STOC '88. New York, NY, USA: ACM, 1988, pp. 20–31.
[10] D. Harnik, Y. Ishai, and E. Kushilevitz, “How many oblivious transfers are needed for secure multiparty
computation?" in Annual International Cryptology Conference. Springer, 2007, pp. 284-302.
[11] M. O. Rabin, "How to exchange secrets,” in Technical Report TR-81, Aiken Computation Laboratory,
Harvad University, 1981.
[12] O. Goldreich, Foundations of Cryptography, Volume I Basic Techniques. Cambridge University Press,
2001.
[13] C. Paar and J. Pelzl, Understanding Cryptography. Springer, 2010.
[14] A. Broadbent and C. Schaffner, "Quantum cryptography beyond quantum key distribution,”
Designs, Codes and Cryptography, vol. 78, no. 1, pp. 351-382, Jan 2016. [Online]. Available:
https://doi.org/10.1007/s10623-015-0157-4
[15] A. Shenoy-Hejamadi, A. Pathak, and S. Radhakrishna, "Quantum cryptography: key distribution and
beyond," Quanta, vol. 6, no. 1, pp. 1-47, 2017.
[16] H. K. Lo and H. F. Chau, "Is quantum bit commitment really possible?” Phys. Rev. Lett., vol. 78, pp.
3410-3413, 1997.
[17] D. Mayers, “Unconditionally secure quantum bit commitment is impossible," Phys. Rev. Lett., vol. 78,
pp. 3414-3417, 1997.
[18] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from noisy storage," Phys. Rev. Lett., vol.
100, p. 220502, 2008.
[19] S. Wehner, M. Curty, C. Schaffner, and H.-K. Lo, "Implementation of two-party protocols in the noisy-
storage model," Phys. Rev. A, vol. 81, p. 052336, 2010.
[20] R. Konig, S. Wehner, and J. Wullschleger, “Unconditional security from noisy quantum storage," IEEE
Transactions on Information Theory, vol. 58, no. 3, pp. 1962-1984, 2012.
[21] R. Loura, Á. J. Almeida, P. André, A. Pinto, P. Mateus, and N. Paunković, “Noise and measurement
errors in a practical two-state quantum bit commitment protocol," Phys. Rev. A, vol. 89, p. 052336,
2014. [Online]. Available: http://link.aps.org/doi/10.1103/PhysRevA.89.052336
[22] Á. J. Almeida, A. D. Stojanovic, N. Paunković, R. Loura, N. J. Muga, N. A. Silva, P. Mateus, P. S.
André, and A. N. Pinto, "Implementation of a two-state quantum bit commitment protocol in optical
fibers," Journal of Optics, vol. 18, no. 1, p. 015202, 2015.
[23] C. Erven, N. Ng, N. Gigov, R. Laflamme, S. Wehner, and G. Weihs, "An experimental implementation
of oblivious transfer in the noisy storage model," Nature Communications, vol. 5, p. 3418, 2014.
[24] F. Furrer, T. Gehring, C. Schaffner, C. Pacher, R. Schnabel, and S. Wehner, "Continuous-variable
protocol for oblivious transfer in the noisy-storage model," Nature Communications, vol. 9, p. 1450,
2018.
[25] N. H. Y. Ng, S. K. Joshi, C. Chen Ming, C. Kurtsiefer, and S. Wehner, "Experimental implementation
of bit commitment in the noisy-storage model," Nature Communications, vol. 3, p. 1326, Dec 2012.
[26] T. Lunghi, J. Kaniewski, F. Bussières, R. Houlmann, M. Tomamichel, S. Wehner, and H. Zbinden,
"Practical relativistic bit commitment," Phys. Rev. Lett., vol. 115, p. 030502, 2015.
[27] E. Verbanis, A. Martin, R. Houlmann, G. Boso, F. Bussières, and H. Zbinden, "24-hour relativistic bit
commitment," Phys. Rev. Lett., vol. 117, p. 140506, 2016.
[28] D. Pitalúa-García and I. Kerenidis, “Practical and unconditionally secure spacetime-constrained oblivi-
ous transfer," Phys. Rev. A, vol. 98, p. 032327, 2018.
[29] A. Yao, "How to generate and exchange secrets," in Proceedings of the 27th Annual Symposium on
Foundations of Computer Science, ser. SFCS '86. Washington, DC, USA: IEEE Computer Society,
1986, pp. 162-167. [Online]. Available: http://dx.doi.org/10.1109/SFCS.1986.25
[30] S. Halevi and S. Micali, Practical and Provably-Secure Commitment Schemes from Collision-Free Hash-
ing. Berlin, Heidelberg: Springer Berlin Heidelberg, 1996, pp. 201-215.
[31] A. N. Pinto, M. F. Ramos, N. A. Silva, and N. J. Muga, "Generation and distribution of oblivious
keys through quantum communications," in 2018 20th International Conference on Transparent Optical
Networks (ICTON), July 2018, pp. 1-3.
[32] M. F. Ramos, N. A. Silva, N. J. Muga, and A. N. Pinto, "Reversal operator to compensate polarization
random drifts in quantum communications," Optics Express, vol. 28, no. 4, pp. 5035-5035, February
2020.
[33] Y. Lindell and B. Pinkas, “An efficient protocol for secure two-party computation in the presence of mali-
cious adversaries," in Annual International Conference on the Theory and Applications of Cryptographic
Techniques. Springer, 2007, pp. 52-78.
[34] A. C.-C. Yao, "Security of quantum protocols against coherent measurements," in Proceedings of the
Twenty-seventh Annual ACM Symposium on Theory of Computing, ser. STOC '95. New York, NY,
USA: ACM, 1995, pp. 67-75.
[35] D. Unruh, “Universally composable quantum multi-party computation,” in Annual International Con-
ference on the Theory and Applications of Cryptographic Techniques. Springer, 2010, pp. 486-505.
[36] D. Hofheinz and J. Müller-Quade, “Universally composable commitments using random oracles," in
Theory of Cryptography, M. Naor, Ed. Springer, 2004, pp. 58-76.
[37] S. Aaronson and Y. Shi, “Quantum lower bounds for the collision and the element distinctness problems,”
J. ACM, vol. 51, no. 4, pp. 595-605, 2004.
[38] G. Asharov, Y. Lindell, T. Schneider, and M. Zohner, "More efficient oblivious transfer and extensions
for faster secure computation," in Proceedings of the 2013 ACM SIGSAC Conference on Computer
& Communications Security, ser. CCS '13. New York, NY, USA: ACM, 2013, pp. 535-548.
[39] A. J. Menezes, J. Katz, P. C. Van Oorschot, and S. A. Vanstone, Handbook of applied cryptography.
CRC press, 1996.
[40] J. Ding, X. Xie, and X. Lin, "A simple provably secure key exchange scheme based on the learning with
errors problem." IACR Cryptology EPrint Archive, vol. 2012, p. 688, 2012.
[41] J. Martinez-Mateo, D. Elkouss, and V. Martin, “Key reconciliation for high performance quantum key
distribution," Scientific reports, vol. 3, p. 1576, 2013.
[42] G. Brassard and L. Salvail, “Secret-key reconciliation by public discussion," in Workshop on the Theory
and Application of of Cryptographic Techniques. Springer, 1993, pp. 410-423.
[43] M. Naor and B. Pinkas, "Computationally secure oblivious transfer," Journal of Cryptology, vol. 18,
no. 1, pp. 1-35, 2005.
[44] T. Chou and C. Orlandi, "The simplest protocol for oblivious transfer," in International Conference on
Cryptology and Information Security in Latin America. Springer, 2015, pp. 40-58.
[45] L. Comandar, B. Fröhlich, M. Lucamarini, K. Patel, A. Sharpe, J. Dynes, Z. Yuan, R. Penty, and
A. Shields, "Room temperature single-photon detectors for high bit rate quantum key distribution,"
Applied Physics Letters, vol. 104, no. 2, p. 021101, 2014.
[46] N. T. Islam, C. C. W. Lim, C. Cahall, J. Kim, and D. J. Gauthier, “Provably secure and high-rate
quantum key distribution with time-bin qudits," Science advances, vol. 3, no. 11, p. e1701491, 2017.
[47] H. Ko, B.-S. Choi, J.-S. Choe, K.-J. Kim, J.-H. Kim, and C. J. Youn, “High-speed and high-performance
polarization-based quantum key distribution system without side channel effects caused by multiple
lasers," Photonics Research, vol. 6, no. 3, pp. 214-219, 2018.
[48] T. Wang, P. Huang, Y. Zhou, W. Liu, H. Ma, S. Wang, and G. Zeng, "High key rate continuous-variable
quantum key distribution with a real local oscillator," Optics express, vol. 26, no. 3, pp. 2794-2806,
2018.
[49] S. Pirandola, U. Andersen, L. Banchi, M. Berta, D. Bunandar, R. Colbeck, D. Englund, T. Gehring,
C. Lupo, C. Ottaviani et al., "Advances in quantum cryptography," arXiv preprint arXiv:1906.01645,
2019.
[50] A. Almeida, A. Stojanovic, N. Paunković, R. Loura, N. J. Muga, N. Silva, P. Mateus,
P. André, and A. Pinto, "Implementation of a two-state quantum bit commitment protocol
in optical fibers," Journal of Optics, vol. 18, no. 1, p. 015202, 2016. [Online]. Available:
http://stacks.iop.org/2040-8986/18/i=1/a=015202
[51] R. Loura, D. Arsenović, N. Paunković, D. B. Popović, and S. Prvanović, "Security of two-state and
four-state practical quantum bit-commitment protocols," Phys. Rev. A, vol. 94, p. 062335, Dec 2016.
[Online]. Available: http://link.aps.org/doi/10.1103/PhysRevA.94.062335