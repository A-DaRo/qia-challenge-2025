PHYSICAL REVIEW A **81**, 052336 (2010)

# Implementation of two-party protocols in the noisy-storage model

Stephanie Wehner,$^{1,*}$ Marcos Curty,$^{2,\dagger}$ Christian Schaffner,$^{3,\ddagger}$ and Hoi-Kwong Lo$^{4,\S}$
$^1$Institute for Quantum Information, Caltech, Pasadena, California 91125, USA
$^2$ETSI Telecomunicación, Department of Signal Theory and Communications, University of Vigo, Campus Universitario, E-36310 Vigo (Pontevedra), Spain
$^3$Centrum Wiskunde & Informatica (CWI), P. O. Box 94079, 1090 GB Amsterdam, the Netherlands
$^4$Center for Quantum Information and Quantum Control (CQIQC), Department of Physics and Department of Electrical & Computer Engineering, University of Toronto, Toronto, Ontario, M5S 3G4, Canada
(Received 26 November 2009; published 25 May 2010)

The noisy-storage model allows the implementation of secure two-party protocols under the sole assumption that no large-scale reliable quantum storage is available to the cheating party. No quantum storage is thereby required for the honest parties. Examples of such protocols include bit commitment, oblivious transfer, and secure identification. Here, we provide a guideline for the practical implementation of such protocols. In particular, we analyze security in a practical setting where the honest parties themselves are unable to perform perfect operations and need to deal with practical problems such as errors during transmission and detector inefficiencies. We provide explicit security parameters for two different experimental setups using weak coherent, and parametric down-conversion sources. In addition, we analyze a modification of the protocols based on decoy states.

DOI: 10.1103/PhysRevA.81.052336

PACS number(s): 03.67.Dd

## I. INTRODUCTION

Quantum cryptography allows us to solve cryptographic tasks without resorting to unproven computational assump- tions. One example is quantum key distribution (QKD) which is well studied within quantum information [1,2]. In QKD, the sender (Alice) and the receiver (Bob) trust each other but want to shield their communication from the prying eyes of an eavesdropper. In many other cryptographic problems, however, Alice and Bob themselves do not trust each other but nevertheless want to cooperate to solve a certain task. An important example of such a task is secure identification. Here, Alice wants to identify herself to Bob (possibly an ATM machine) without revealing her password. More generally, Alice and Bob wish to perform secure function evaluation as depicted in Fig. 1.

[IMAGE: Figure 1 illustration showing Alice inputs X, Bob inputs Y, and they evaluate and share f(x, y).]

In this scenario, security means that the legitimate users should not learn anything beyond this specification. That is, Alice should not learn anything about $y$ and Bob should not learn anything about $x$, other than what they may be able to in- fer from the value of $f(x,y)$. Classically, it is possible to solve this task if one is willing to make computational assumptions, such as that factoring of large integers is difficult. Sadly, these assumptions remain unproven. Unfortunately, even quantum mechanics does not allow us to implement such interesting cryptographic primitives without further assumptions [3–7].

### A. The noisy-storage model

The noisy-storage model (NSM) allows us to obtain secure two-party protocols under the *physical assumption* that any cheating party does not possess a large reliable quantum storage. First introduced in Refs. [8,9], the NSM has recently [10] been shown to encompass both the case where the adversary has a bounded amount of noise-free storage [11,12] (also known as the bounded-storage model), as well as the case where the adversary has access to a potentially large amount of noisy storage. This last assumption is well justified given the state of present-day technology and the fact that merely transferring the state of a photonic qubit onto a different carrier (such as an atomic ensemble) is typically already noisy, even if the resulting quantum memory is perfect. In the protocols considered, the honest parties themselves do not require any quantum storage at all. We briefly review the NSM here for completeness. Without loss of generality, noisy quantum storage is described by a family of completely positive trace-preserving maps $\{\mathcal{F}_t : \mathcal{B}(\mathcal{H}_{\text{in}}) \to \mathcal{B}(\mathcal{H}_{\text{out}})\}_{t>0}$, where $t$ is the time that the adversary uses his storage device. An input state $\rho$ on $\mathcal{H}_{\text{in}}$ stored at time $t_0 = 0$ decoheres over time, resulting in a state $\mathcal{F}_t(\rho)$ of the memory at time $t$. We make the minimal assumption that the noise is Markovian, meaning that the adversary does not gain any advantage by delaying the readout whenever he wants to retrieve encoded information: waiting longer only degrades the information further. The only assumption underlying the noisy-storage model consists in demanding that the adversary can keep only quantum informa- tion in this noisy-storage device. In particular, he is otherwise completely unrestricted; for example, he can perform arbitrary (instantaneous) quantum computations using information from the storage device and additional ancillas. In particular, he is able to perform perfect, noise-free quantum computation and communication. However, after his computation he needs to discard all quantum information except what is contained in the storage device, where he may prepare an arbitrary encoded state on $\mathcal{H}_{\text{in}}$. This scenario is illustrated in Fig. 2.

How can we obtain security from such a physical as- sumption? We consider protocols which force the adversary to store quantum information for extended periods to gain information: This is achieved by using certain time delays $\Delta t$ at specific points in the protocol (e.g., before starting a round of communication). This forces the adversary to use his device for a time at least $\Delta t$ if he wants to preserve quantum information. Due to the Markovian assumption, it suffices to analyze security for the channel $\mathcal{F} = \mathcal{F}_{\Delta t}$. Hence the security model can be summarized as follows:
* The adversary has unlimited classical storage and (quan- tum) computational resources. He is able to perform any operations noise-free and has access to a noise-free quantum channel.
* Whenever the protocol requires the adversary to wait for a time $\Delta t$, he has to measure or discard all his quantum information except what he can encode (arbitrarily) into $\mathcal{H}_{\text{in}}$. This information then undergoes noise described by $\mathcal{F}$.

We stress that in contrast to the adversary's potential resources allowed in this model, the technological demands on honest parties are minimal: in our protocol, honest parties merely need to prepare and measure BB84-encoded qubits$^1$ and do not require any quantum storage.

[IMAGE: Figure 2. During waiting times $\Delta t$, the adversary must use his noisy-quantum storage described by the completely positive trace- preserving (CPTP) map $\mathcal{F}$. Before using his quantum storage, he performs any (error-free) “encoding attack” of his choosing, which consists of a measurement or an encoding into an error-correcting code. After time $\Delta t$, he receives some additional information that he can use for decoding.]

### B. Challenges in a practical implementation

In this work we focus on how to put the protocols of [10] into practice. Unfortunately, the theoretical analysis of Ref. [10] assumes perfect single-photon sources that are not available yet [13,14]. Here, we remove this assumption leading to a slightly modified protocol that can be implemented immediately using today's technology. At first glance, it may appear that the secu- rity analysis for a practical implementation differs little from the problems encountered in practical realizations of QKD. After all, the quantum communication part of the protocols in [10] consists of Alice sending BB84 states to Bob. Yet, since now the legitimate users do not trust each other, the analysis differs from QKD in several fundamental aspects. Intuitively, these differences arise because Alice and Bob do not cooperate to check on an outside eavesdropper. Quite on the contrary, Alice can never rely on anything that Bob says. A second important aspect that differentiates the setting in Ref. [10] from QKD lies in the task the cryptographic protocols aim to solve. For instance, secure identification is particularly interesting at extremely short distances, for which Alice would ideally use a small, low-power portable device. Bob, on the other hand, may use more bulky detectors. At such short distances, we could furthermore use visible light for which much better detectors exist than those typically used in QKD at telecom wavelengths. It is an interesting experimental challenge to come up with suitable devices. Small handheld setups have been proposed to perform QKD at short distance [15], which we can also hope to use here. The QKD devices of Ref. [15] have been devised to distribute nonreusable authentication keys which could also be employed for identification. At such short distance, this could also be achieved by for example loading keys onto a USB stick at a trusted loading station at a bank for instance. We emphasize that our work is in spirit very different in that we allow authentication keys to be reused over and over again, just as traditional passwords [16].

We first analyze a generic experimental setup in Sec. II. More specifically, we present a source-independent characterization of such a setup and discuss all parameters that are necessary to evaluate security in the NSM. Especially important is that in any real-world setting even the honest parties do not have access to perfect quantum operations, and the channel connecting Alice and Bob is usually noisy. The challenge we face is to enable the honest parties to execute the protocol successfully in the presence of errors, while ensuring that the protocol remains secure against any cheating party. We shall always assume a worst-case scenario where a cheating party is able to perform perfect quantum operations and does not experience channel noise; its only restriction is its noisy quantum storage.

The primary source of errors at short distances lies in the low detector efficiencies of present-day single-photon detectors. For telecom wavelengths these detector efficiencies $\eta_D$ lie at roughly 10%, where at visible wavelengths one can use detectors of about 70% efficiency. Hence, a considerable part of the transmissions will be lost. In Sec. III, we augment the protocol for weak string erasure presented in Ref. [10] to deal with such erasure errors. This protocol is the main ingredient to realize the primitive of oblivious transfer, which can be used to solve the problem of computing a function $f(x,y)$. The second source of errors lies in bit errors which result from noise on the channel itself or imperfections in Alice and Bob's measurement apparatus. At short distances, such errors will typically be quite small. In Sec. IV, we show how to augment the protocol for oblivious transfer to deal with bit

---
$^1$ That is, qubits encoded in one of two conjugate bases, such as the computational and Hadamard basis.

---

errors. It should be noted that we treat these errors in the classical communication part of the protocols, independently of erasure errors, and similar techniques may be used in other schemes based on weak string erasure in the future.

To obtain security, we have to make a reasonable estimation of the errors that the honest parties expect to occur. We state the necessary parameters in Sec. II and provide concrete estimates for two experimental setups in Sec. V. In particular, we present explicit security parameters for a source of weak coherent pulses and a parametric down-conversion (PDC) source. Throughout we assume that the reader is familiar with commonly used entropic quantities also relevant for QKD and quantum information. An introduction to all concepts relevant for security in the NSM is given in Ref. [10].

## II. GENERAL SETUP

Before turning to the actual protocols, we need to inves- tigate the parameters involved in an experimental setup. The quantum communication part of all the protocols in the NSM is a simple scheme for weak string erasure which we will describe in detail in the next section. In each round of this protocol, Alice chooses one of the four possible BB84 states [17] at random and sends this state to Bob. Bob now measures randomly the state received either in the computational or in the Hadamard basis. Such a setup is characterized by a source held by Alice and a measurement apparatus held by Bob as depicted in Fig. 3. The source can as well include a measurement device, depending on the actual state preparation process (e.g., when a PDC source acts as a triggered single-photon source). If Alice is honest, we can trust the source entirely, which means that, in principle, we have full knowledge of its parameters. Note, however, that in any practical setting the parameters of the source will undergo small fluctuations. For clarity of exposition, we do not take these fluctuations into account ex- plicitly but assume that all the parameters below are worst-case estimates of what we can reasonably expect from our source.

### A. Source parameters

Unfortunately, we do not have access to a perfect single- photon source in a practical setting [13,14] but can only arrange the source to emit a certain number of photons with a certain probability. To approximate a single-photon source, we will later let Alice perform some measurements herself to exclude multiphoton events in the case of a PDC source. The following table summarizes the two relevant probabilities we need to know in any implementation. When using decoy states, we will frequently add an index $s$ to all parameters to specify a particular source $s$ that is used.

| Probability | Description |
| :--- | :--- |
| $P_{\text{src}}^{n}$ | The source emits $n$ photons. |
| $P_{\text{sent}}^{n|1}$ | The source emits $n$ photons conditioned on the event that Alice concludes that *one* photon has been emitted. |

[IMAGE: Figure 3. A general setup for weak string erasure.]

In our analysis, we will be interested in bounding the number of single-photon emissions in $M$ rounds of the protocol, which can be achieved using the well-known Chernoff's inequality (see, e.g., Ref. [18]): Suppose we have a source that emits a single photon with probability $P_{\text{src}}^1$ and a different number of photons otherwise. How many single-photon emissions do we expect? Intuitively, it is clear that in $M$ rounds we have roughly $P_{\text{src}}^1 M$ many. Yet, in the following we need to consider a small interval around $P_{\text{src}}^1 M$, such that the probability that we do not fall into this interval is extremely small. More precisely, we want that
$$
\text{Prob}[|S - P_{\text{src}}^1 M| \ge \zeta_{\text{src}}^1 M] < \varepsilon,
\tag{1}
$$
where $S$ is the number of single-photon emissions. To apply Chernoff's inequality, let $X_j = 1$ denote the event where a single-photon emission occurred and let $X_j = 0$ otherwise, giving us $S = \sum_j X_j$. We then demand that
$$
2e^{-2(\zeta_{\text{src}}^1)^2 M} < \varepsilon,
\tag{2}
$$
which can be achieved by choosing $\zeta_{\text{src}}^1 = \sqrt{\ln(2/\varepsilon)/(2M)}$. Operationally this means that the number of single-photon emissions lies in the interval $[(P_{\text{src}}^1 - \zeta_{\text{src}}^1)M, (P_{\text{src}}^1 + \zeta_{\text{src}}^1)M]$, except with probability $\varepsilon$. Note that for $M$ being very large we indeed have $\zeta_{\text{src}}^1 \approx 0$, leaving us with approximately $P_{\text{src}}^1 M$ many single-photon emissions. By exactly the same argument, if now $M$ refers to the number of rounds in the protocol where Alice concluded the source emitted single photons, the actual number of single-photon rounds within these postselected events lies in the interval $[(P_{\text{sent}}^{1|1} - \zeta_{\text{sent}}^{1|1})M, (P_{\text{sent}}^{1|1} + \zeta_{\text{sent}}^{1|1})M]$ for $\zeta_{\text{sent}}^{1|1} = \sqrt{\ln(2/\varepsilon)/(2M)}$, except with probability $\varepsilon$. We will make use of this argument repeatedly and use $\zeta$ to denote the interval when considering an event that occurs with probabiity $P$.

We emphasize that for our security proof to work, we only need a conservative lower bound on the number of single- photon emissions. Should there be some intensity fluctuations in Alice's laser provided that we know the worst case (i.e., a conservative lower bound $P_{\text{src}}^1$) in the asymptotic case of large $M$, then the discussion for the finite-size case will go through if we consider a one-sided bound in Eq. (1). i.e., $\text{Prob}[S < (P_{\text{src}}^1 - \zeta_{\text{src}}^1)M] < \varepsilon$.

### B. Error parameters

For any setup, we need to determine the following error parameters. These parameters should be a reasonable estimate that is made once for a particular experimental implementation and fixed during subsequent executions of the protocol. For instance, for a given device meant to be used for identification, these estimates would be fixed during construction.

#### 1. Losses

As mentioned above, the primary restriction in a practical setting arises from the loss of signals. These losses can occur on the channel or be caused by detector inefficiencies. The following table summarizes all the probabilities we need. Throughout, we use the superscripts $h$ and $d$ to indicate that these parameters apply to an honest or dishonest party respectively.

| Probability | Description |
| :--- | :--- |
| $P_{\text{erase}}^n$ | $n$ photons are erased on the channel |
| $P_{\text{B,click}}^h$ | Honest Bob observes a click in his detection apparatus |
| $P_{\text{B,no click}}^h$ | Honest Bob observes no click in his detection apparatus |
| $P_{\text{B,click}}^{h|n}$ | Honest Bob observes a click in his detection apparatus, conditioned on the event that Alice sent $n$ photons. |
| $P_{\text{B,S,no click}}^h$ | Honest Bob observes no click from the signal alone |
| $P_{\text{dark}}$ | An honest player obtains a click when the signal was a vacuum state (dark count) |

Note that we have
$$
P_{\text{B, no click}}^h = \sum_{n=0}^{\infty} P_{\text{src}}^n P_{\text{B, no click}}^{h|n}
\tag{3}
$$
and again the number of rounds we expect to be lost can be bounded to lie in the interval $[(P_{\text{B,no click}}^h - \zeta_{\text{B,no click}}^h)M, (P_{\text{B,no click}}^h + \zeta_{\text{B,no click}}^h)M]$ with $\zeta_{\text{B,no click}}^h = \sqrt{\ln(2/\varepsilon)/(2M)}$, except with probability $\varepsilon$.

#### 2. Bit errors

The second source of errors are bit-flip errors that can occur due to imperfections in Alice's or Bob's measurement apparatus or due to noise on the channel. We use the following notation for the probability of such an event in the case that Bob is honest. This probability depends on the detection error $e_{\text{det}}$ in our experimental setup, i.e., on the probability that a signal sent by Alice produces a click in the erroneous detector on Bob's side and on $P_{\text{dark}}$. The quantity $e_{\text{det}}$ characterizes the alignment and stability of the optical system.

| Parameter | Description |
| :--- | :--- |
| $e_{\text{det}}$ | Detection error |
| $P_{\text{B,err}}^h$ | Honest Bob outputs the wrong bit |

For a single bit $b \in \{0,1\}$, a bit-flip error is described by the classical binary symmetric channel with error parameter $P_{\text{err}}$
$$
\mathcal{S}_{P_{\text{err}}}(b) = \begin{cases} b & \text{with probability } 1 - P_{\text{err}} \\ 1-b & \text{with probability } P_{\text{err}}. \end{cases}
\tag{4}
$$
When each bit of a $k$-bit string is independently affected by bit-flip errors, the noise can be described by the channel
$$
\mathcal{E}_{P_{\text{err}}} = \mathcal{S}_{P_{\text{err}}}^{\otimes k}
\tag{5}
$$
where we omit the explicit reference to $k$ on the left-hand side when it is clear from the context.

### C. Parameters for dishonest Bob

Recall our conservative assumption that a dishonest party is restricted only by its noisy quantum storage but can otherwise perform perfect quantum operations and has access to a perfect channel. Yet, even for a dishonest Bob there are some errors he cannot avoid, caused by the imperfections in Alice's apparatus. If Alice's source simply outputs no photon for example, then even a dishonest Bob cannot detect the transmission which is captured by the following parameter.

| Probability | Description |
| :--- | :--- |
| $P_{\text{B, no click}}^d$ | Dishonest Bob observes no click in his detection apparatus |

Generally, we have $P_{\text{B, no click}}^d = P_{\text{sent}}^0$. In the protocols that follow, we will ask an honest Bob to report any round as missing that has not resulted in a click. Without loss of generality, we can assume that even a dishonest Bob will report a particular round as lost when he does not observe a click. Of course, if Bob is dishonest he potentially chooses to report additional rounds as missing.

[IMAGE: Figure 4. Weak string erasure with errors when both parties are honest. $\mathcal{E}_{P_{\text{err}}}$ denotes the bit-error channel defined in (5).]

In our analysis, we also have to evaluate the following probability which depends on the experimental setup, as well as on our choice of protocol parameters.

| Probability | Description |
| :--- | :--- |
| $P_{\text{B,err}}^{d,n}$ | Dishonest Bob outputs the wrong bit if Alice sent $n$ photons, and he gets the basis information for free |

## III. WEAK STRING ERASURE WITH ERRORS

The basic quantum primitive on which all other protocols in Ref. [10] are based is called weak string erasure. Intuitively, weak string erasure provides Alice with a random $m$-bit string $X^m$ and Bob with a random set of indices $\mathcal{I} \in 2^{[m]}$ and the substring $X_{\mathcal{I}}$ of $X^m$ restricted to the elements in $\mathcal{I}$.$^{2}$ If Bob is honest, then we demand that whatever attack dishonest Alice mounts, she cannot gain any information about which bits Bob has learned. That is, she cannot gain any information about $\mathcal{I}$. If Alice herself is honest, we demand that the amount of information that Bob can gain about the string $X^m$ is limited.

We now present an augmented version of the weak string erasure protocol proposed in Ref. [10] that allows us to deal with the inevitable errors encountered during a practical implementation. We thereby address the two possible errors separately: losses are dealt with directly in weak string erasure. Bit-flip errors, however, are not corrected in weak string erasure itself, but in subsequent protocols.$^{3}$ We will thus implement weak string erasure with errors where the substring $X_{\mathcal{I}}$ is allowed to be affected by bit-flip errors. That is, honest Bob actually receives $\mathcal{E}_{P_{\text{err}}}(X_{\mathcal{I}})$, where $\mathcal{E}_{P_{\text{err}}}$ is the classical channel corresponding to the bit errors as given in (5), with $k = |\mathcal{I}|$ being the length of the string $X_{\mathcal{I}}$. Figure 4 provides an intuitive description of this task.

We now provide an informal definition of weak string erasure with errors. A formal definition can be found in Appendix A. Even in this informal definition we need to quantify the knowledge that a cheating Bob has about the string $X^m$ given access to his entire system $B'$.$^{4}$ This quantity has a simple interpretation in terms of the min-entropy as $H_{\infty}(X^m|B') = -\log_2 P_{\text{guess}}(X^m|B')$, where $P_{\text{guess}}(X^m|B')$ represents the probability that Bob guesses $X^m$, maximized over all measurements of the quantum part $B'$. The quantity $H(X^m|B')$ thereby behaves like $H_{\infty}(X^m|B')$, except with probability $\varepsilon$. We refer to Ref. [10] for an introduction to these quantities and their use in the NSM.

**Definition III.1 (informal).** An $(m, \lambda, \varepsilon, P_{\text{err}})$ weak string erasure protocol with errors (WSEE) is a protocol between Alice and Bob satisfying the following properties, where $\mathcal{E}_{P_{\text{err}}}$ is defined as in (5):

*   **Correctness:** If both parties are honest, then Alice obtains a randomly chosen $m$-bit string $X^m \in \{0,1\}^m$, and Bob obtains a randomly chosen subset $\mathcal{I} \subseteq [m]$, as well as the string $\mathcal{E}_{P_{\text{err}}}(X_{\mathcal{I}})$.
*   **Security for Alice:** If Alice is honest, then the amount of information Bob has about $X^m$ is limited to
$$
\frac{1}{m} H_{\infty}^{\varepsilon}(X^m | B') \ge \lambda,
\tag{6}
$$
where $B'$ denotes the total state of Bob's system.
*   **Security for Bob:** If Bob is honest, then Alice learns nothing about $\mathcal{I}$.

We are now ready to state a simple protocol for WSEE. We thereby introduce explicit time slots into the protocol. If Alice herself concludes that no photon or a multi-photon has been emitted in a particular time slot, she simply discards this round and tells Bob to discard this round as well. Since this action represents no security problem for us, we will for simplicity omit these rounds altogether when stating the protocol below. This means that the number of rounds $M$ in the protocol below actually refers to the set of postselected pulses that Alice did count as a valid round.

In addition, introducing time slots enables Bob to report a particular bit as missing if he has obtained no click in a particular time slot. Alice and Bob will subsequently discard all missing rounds. This does pose a potential security risk, which we need to analyze and hence we explicitly include this step in the protocol below.

**Protocol 1: Weak String Erasure with Errors**
Outputs: $x \in \{0,1\}^m$ to Alice, $(\mathcal{I}, z^{|\mathcal{I}|}) \in 2^{[m]} \times \{0,1\}^{|\mathcal{I}|}$ to Bob.

1.  **Alice:** Chooses a string $X^M \in_R \{0,1\}^M$ and basis-specifying string $\Theta^M \in_R \{0,1\}^M$ uniformly at random.
2.  **Bob:** Chooses a basis string $\tilde{\Theta}^M \in_R \{0,1\}^M$ uniformly at random.
3.  In time slot $i = 1,...,M$ (considered a valid round by Alice):
    1.  **Alice:** Encodes bit $x_i$ in the basis given by $\theta_i$ (i.e., as $H^{\theta_i}(x_i)$), and sends the resulting state to Bob.
    2.  **Bob:** Measures in the basis given by $\tilde{\theta}_i$ to obtain outcome $\tilde{x}_i$. If Bob obtains no click in this time slot, he records round $i$ as missing.
4.  **Bob:** Reports to Alice which rounds were missing.
5.  **Alice:** If the number of rounds that Bob reported missing does not lie in the interval $[(P_{\text{B,no click}}^h - \zeta_{\text{B,no click}}^h)M, (P_{\text{B,no click}}^h + \zeta_{\text{B,no click}}^h)M]$, then Alice aborts the protocol. Otherwise, she deletes all bits from $X^M$ that Bob reported missing. Let $X^m \in \{0,1\}^m$ denote the remaining bit string, and let $\Theta^m$ be the basis-specifying string for the remaining rounds. Let $\tilde{\Theta}^m$, and $\tilde{X}^m$ be the corresponding strings for Bob.
    *Both parties wait time $\Delta t$.*
6.  **Alice:** Sends the basis information $\Theta^m$ to Bob, and outputs $X^m$.
7.  **Bob:** Computes $\mathcal{I} := \{i \in [m] | \theta_i = \tilde{\theta}_i\}$, and outputs $(\mathcal{I}, Z^{|\mathcal{I}|}) := (\mathcal{I}, \tilde{X}_{\mathcal{I}})$.

### A. Security analysis

#### 1. Parameters

We prove the security of Protocol 1 in Appendix A, where our analysis forms an extension of the proof presented in Ref. [10]. The security proof for dishonest Alice is analogous to Ref. [10]. The only novelty is to ensure that allowing Bob to report rounds as missing does not compromise the security. Here, we focus on weak string erasure with errors, when the adversary's storage is of the form $\mathcal{F} = \mathcal{N}^{\otimes vM_{\text{store}}}$, and $\mathcal{N}$ obeys the strong converse property [19]. An important example is the $d$-dimensional depolarizing channel. For this case, we can give explicit security parameters in terms of the amount of noise generated by $\mathcal{N}$. The quantity $v$ denotes the storage rate, and $M_{\text{store}}$ is the number of single-photon emissions that we expect an honest Bob to receive for large $M$. That is
$$
M_{\text{store}} := P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1} M.
\tag{7}
$$
We hence allow Bob's storage size to be determined as in the idealized setting of Ref. [10], where we have only single- photon emissions. Throughout, we let $M^{(n)}$ denote the number of $n$ photon emissions in $M$ valid rounds and use $r^{(n)}$ to denote the fraction of these $n$-photon pulses that Bob decides to report as missing. Clearly, $r^{(n)}$ is not a parameter we can evaluate but depends on the strategy of dishonest Bob. Finally, we use $M_{\text{left}}^{(n)} = (1 - r^{(n)})M^{(n)}$ to denote the number of $n$-photon pulses that are left. Note that $M_{\text{left}}^{(n)}$ is a function of $r^{(n)}$ chosen by Bob according to certain constraints which we investigate later. A proof of Theorem III.2, as well as a generalization to other channels $\mathcal{F}$ not necessarily of the form $\mathcal{F} = \mathcal{N}^{\otimes vM_{\text{store}}}$, can be found in Appendix A. Here, we state the theorem for a worst-case setting which can be obtained using (A29). This result is independent of the actual choice of signals that Bob chooses to report as missing. For simplicity, we present the theorem omitting terms that vanish for large $M$. These terms are, however, considered in Appendix A.

**Theorem III.2 (WSEE).** Let Bob's storage be given by $\mathcal{F} = \mathcal{N}^{\otimes vM_{\text{store}}}$ for a storage rate $v > 0$, $\mathcal{N}$ satisfying the strong converse property [19] and having capacity $C_{\mathcal{N}}$ bounded by
$$
C_{\mathcal{N}} v < \frac{1}{2} \left( \frac{P_{\text{sent}}^1 - P_{\text{B,no click}}^h + P_{\text{B,no click}}^d}{P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1}} \right)
\tag{8}
$$
Then Protocol 1 is an $[m, \lambda(\delta), \varepsilon(\delta), P_{\text{err}}]$ weak string erasure protocol with errors with the following parameters: Let $\delta \in ]0, \frac{1}{2} - C_{\mathcal{N}}v[$. Then the min-entropy rate $\lambda(\delta)$ is given by
$$
\lambda(\delta) = \min_{\{r^{(n)}\}_n} \frac{1}{m} \left[ \frac{1}{\nu} \gamma_{\mathcal{N}}^* \left(\frac{R}{v}\right) M_{\text{store}} - \sum_{n=2}^{\infty} M_{\text{left}}^{(n)} \log_2 (1 - P_{\text{B,err}}^{d,n}) \right]
\tag{9}
$$
where $\gamma_{\mathcal{N}}^*$ is the strong converse parameter of $\mathcal{N}$ (see (15)) and the minimization is taken over all $\{r^{(n)}\}_n$ such that $\sum_{n=1}^{\infty} r^{(n)} M^{(n)} < M_{\text{report}}^d$, $M_{\text{store}}$ is given by (7), and
$$
m = \sum_{n=1}^{\infty} M_{\text{left}}^{(n)}
\quad \text{(the number of remaining rounds)}
$$
$$
M_{\text{report}}^d = (P_{\text{B,no click}}^h - P_{\text{B,no click}}^d) M
\quad \text{(the number of rounds dishonest Bob can report missing)}
$$
$$
R = \left(\frac{1}{2} - \delta\right) \frac{M_{\text{left}}^{(1)}}{P_{\text{B,click}}^{h|1} M}
\quad \text{(the rate at which dishonest Bob has to send information through storage)}
$$
for sufficiently large $M$. The error has the form
$$
\varepsilon(\delta) < 4 \exp \left[ - \frac{\delta^2}{512(4 + \log_2 \frac{1}{\delta})^2} \times \left(P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1} + P_{\text{B, no click}}^d \right) M \right].
\tag{10}
$$

What kind of channels $\mathcal{N} : \mathcal{B}(\mathcal{H}_{\text{in}}) \to \mathcal{B}(\mathcal{H}_{\text{out}})$ satisfy the strong converse property? It was recently shown in [19] that all channels for which the maximum $\alpha$-norm is multiplicative, and which are group covariant, that is $\mathcal{N}(g \rho g^{\dagger}) = g \mathcal{N}(\rho) g^{\dagger}$ for all $g \in G$ where $g$ acts irreducibly on the output space $\mathcal{H}_{\text{out}}$, satisfy this property. An important example of such a channel is the $d$-dimensional depolarizing channel given as
$$
\mathcal{N}_r(\rho) := r \rho + (1 - r) \frac{\mathcal{I}}{d}
\tag{11}
$$
which replaces the input state $\rho$ with the completely mixed state $\mathcal{I}/d$ with probability $1 - r$. Security parameters for this channel can be found in Ref. [10] for the case of a perfect setup with a single-photon source, assuming no errors nor detection inefficiencies.

#### 2. Limits to security

Before analyzing in detail concrete practical implementa- tions based on a weak coherent source, and a PDC source, we investigate when security can be obtained at all for the $d$-dimensional depolarizing channel as a function of $P_{\text{sent}}^{1|1}$, $P_{\text{B,no click}}^{h|1}$, $P_{\text{B, no click}}^d$, and $P_{\text{B,click}}^{h|1}$ in comparison to the storage parameters $r$ and $v$. Note that for the security parameter $\varepsilon(\delta)$ to vanish we need
$$
P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1} + P_{\text{B, no click}}^d > 0.
\tag{12}
$$
Second, we require (in the limit of large $M$ where we may choose $\delta \to 0$) that
$$
C_{\mathcal{N}} v < \frac{1}{2} \frac{P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1} + P_{\text{B, no click}}^d}{P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1}}
\tag{13}
$$
where $C_{\mathcal{N}, r}$ is given by [20]
$$
C_{\mathcal{N},r} = \log_2 d + \frac{r + \frac{1-r}{d}}{2} \log_2 \left(\frac{r + \frac{1-r}{d}}{\frac{1-r}{d}}\right) + \frac{r + d-1 \frac{1-r}{d}}{2} \log_2 \left(\frac{r + d-1 \frac{1-r}{d}}{1-r}\right)
\tag{14}
$$
In subsections V A and V B we provide sample trade-offs between $r$ and $v$ for some typical values of the source parameters and the losses.

To determine the magnitude of the actual security pa- rameters, we need to evaluate the strong converse parameter $\gamma_{\mathcal{N}}^*(R)$. In the case of the $d$-dimensional depolarizing channel it can be expressed as [10]
$$
\gamma_{\mathcal{N}, r}^*(R) := \max_{\alpha \ge 1} \frac{\alpha}{\alpha-1} \left[ R - \log_2 d + \frac{1}{\alpha} \log_2 \left( \frac{r + \frac{1-r}{d}}{r + \frac{1-r}{d} + (d-1) \frac{1-r}{d}} \right) \right]
\tag{15}
$$
For a general definition and discussion on how to evaluate this parameter for other channels see Refs. [10,19]. For simplicity, we consider here a setup where Bob always gains full information from a multiphoton emission, that is $P_{\text{B,err}}^{d,n} = 0$ for $n > 1$. This means that he will never report any such rounds as missing, that is, $r^{(n)} = 0$ for $n > 1$. From (A29) it follows that
$$
\lambda(\delta) \ge \frac{1}{m} \left[ \frac{1}{\nu} \gamma_{\mathcal{N}}^* \left(\frac{R}{v}\right) \frac{P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1}}{P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1}} M_{\text{store}} \right],
\tag{16}
$$
providing the security conditions (12) and (13) are satisfied. In subsections V A and V B we plot $\lambda(\delta)$ for a variety of parameter choices for a weak coherent and a PDC source, respectively.

### B. Using decoy states

We now consider a slight modification of the protocol above, where we make use of so-called decoy states as they are also used in QKD [21-23]. The main idea consists of Alice randomly choosing a particular setting of her photon source according to a distribution $P_{\mathcal{S}}$ over some set of settings $\mathcal{S}$ for each state she sends to Bob. One of these settings (signal setting) corresponds to the configuration of the source she would normally use to execute the weak string erasure protocol above, all others (decoy settings) are used to test the behavior of dishonest Bob. In our setting, the effect of using decoy states is that dishonest Bob needs to behave roughly the same as honest Bob when it comes to choosing which rounds to report as missing. This enables us to place a better bound on the parameter $r^{(1)}$, which can lead to a significant increase in the set of detection efficiencies for which we can hope to show security (e.g., for a weak coherent source see subsection V A3), and translates into an enhancement of the rate $R$ given by (A26) and (A29) at which the adversary needs to transmit information through his storage, if he wants to break the security of the protocol.

We briefly describe how we make use of decoy states, before turning to the actual protocol. For each source setting, Alice can compute the *gain*, that is, the probability that Bob observes a click. Here we consider only the number of rounds $M$ which Alice determines to be valid and all probabilities are as explained in Sec. II conditioned on the event that Alice declared the round to be valid. We can then write the gain of honest Bob when Alice uses setting $s$, averaged over all possible numbers of photons, as
$$
Q_s^h = P_{\text{B,click}, s}^h = \sum_{n=0}^{\infty} P_{\text{sent}}^{n|s} P_{\text{B,click}}^{h|n}
\tag{17}
$$
Note that $P_{\text{B,click}}^{h|n}$ thereby does not depend on the source setting $s$, even though Bob can gain information about the setting $s$ by making a photon number measurement, since not all photon numbers are equally likely to occur for the different settings. Yet, since the photon number is the only information that Bob obtains, we can without loss of generality assume that his strategy is deterministic and depends only on the observed photon number. By counting the number of rounds that Bob reports missing, Alice obtains an estimate of this gain as
$$
Q_s^{\text{meas}} = \frac{M_{\text{left}, s}}{M_s}
\tag{18}
$$
The parameter $M_s$ denotes the number of valid rounds in which Alice uses setting $s$, and $M_{\text{left}, s}$ represents the number of such rounds that Bob did not report as missing. For an honest Bob, we have $Q_s^{\text{meas}} \approx Q_s^h$ in the limit of large $M_s$. For finite $M_s$, we conclude that $M_{\text{left}, s}$ lies in the interval $[(Q_s^h - \zeta_s^h)M_s, (Q_s^h + \zeta_s^h)M_s]$, except with probability $\varepsilon$. In the protocol below, Alice will hence abort if $M_{\text{left}, s}$ lies outside this interval for any setting $s \in \mathcal{S}$.

From the observed quantities $Q_s^{\text{meas}}$ for different settings, Alice can obtain a lower bound on the yield of the single- photon emissions following standard techniques used in decoy state QKD [21-24]. Let us denote this lower bound as $\tau$. For honest Bob, the yield of single photons is of course just $P_{\text{B,click}}^{h|1}$ as honest Bob always reports a round as missing if he did not observe a click. For dishonest Bob, placing a bound on this yield corresponds to placing a bound on $1 - r^{(1)}$, which in the limit of large $M$ can be seen as the probability that dishonest Bob does not choose to report a round as missing. Hence, we can use decoy states to obtain an estimate for the parameter $r^{(1)}$ as
$$
r^{(1)} < 1 - \tau
\tag{19}
$$
even if Bob is dishonest. In Sec. V we provide an explicit expression for $\tau$ for the case of a source emitting phase- randomized coherent states.

**Protocol 2: Weak String Erasure with Errors using decoy states**
Outputs: $X^m \in \{0,1\}^m$ to Alice, $(\mathcal{I}, Z^{|\mathcal{I}|}) \in 2^{[m]} \times \{0,1\}^{|\mathcal{I}|}$ to Bob.

1.  **Alice:** Chooses a string $X^M \in_R \{0,1\}^M$ and basis-specifying string $\Theta^M \in_R \{0,1\}^M$ uniformly at random.
2.  **Bob:** Chooses a basis string $\tilde{\Theta}^M \in_R \{0,1\}^M$ uniformly at random. He initializes $\mathcal{M} \leftarrow \emptyset$.
3.  In time slot $i = 1, ..., M$:
    1.  **Alice:** Chooses a source setting $s_i \in \mathcal{S}$ with probability $P_{\mathcal{S}}(s_i)$. Encodes bit $x_i$ in the basis given by $\theta_i$ (i.e., as $H^{\theta_i}(x_i)$) and sends the resulting state to Bob.
    2.  **Bob:** Measures in the basis given by $\tilde{\theta}_i$ to obtain outcome $\tilde{x}_i$. If Bob obtains no click in this time slot, he records round $i$ as missing by letting $\mathcal{M} \leftarrow \mathcal{M} \cup \{i\}$.
4.  **Bob:** Reports to Alice which rounds were missing by sending $\mathcal{M}$.
5.  **Alice:** For each possible source setting $s \in \mathcal{S}$, Alice computes the set of missing rounds $\mathcal{M}_s = \{i \in \mathcal{M} \mid s_i = s\}$. Let $M_s = |\{j \in [M] \mid s_j = s\}|$ be the number of rounds sent using setting $s$.
6.  **Alice:** For each source setting $s \in \mathcal{S}$: if the number of rounds that Bob reported missing does not lie in the interval $[(P_{\text{B, no click}, s}^h - \zeta_{\text{B, no click}, s}^h) M_s, (P_{\text{B, no click}, s}^h + \zeta_{\text{B, no click}, s}^h) M_s]$, then Alice aborts the protocol. Otherwise, she deletes all bits from $X^M$ that Bob reported missing, and all bits that correspond to decoy state settings $s \in \mathcal{S}$. Let $X^m \in \{0,1\}^m$ denote the remaining bit string, and let $\Theta^m$ be the basis-specifying string for the remaining rounds. Let $\tilde{\Theta}^m$, and $\tilde{X}^m$ be the corresponding strings for Bob.
    *Both parties wait time $\Delta t$.*
7.  **Alice:** Informs Bob which rounds remain and sends the basis information $\Theta^m$ to Bob, and outputs $X^m$.
8.  **Bob:** Computes $\mathcal{I} := \{i \in [m] \mid \theta_i = \tilde{\theta}_i\}$, and outputs $(\mathcal{I}, Z^{|\mathcal{I}|}) := (\mathcal{I}, \tilde{X}_{\mathcal{I}})$.

We now state the security parameters for this protocol for the case of large $M_s = M_s$ for each possible source. The only difference to the previous statement is that we replace the bound on the rate (A29) with the bound obtained by bounding $r^{(1)}$ as in (19). The parameter $M$ refers to the number of valid pulses coming from the signal setting. The decoy pulses are merely used as an estimate and play no further role in the protocol. However, the probability $\varepsilon$ to make a correctness or security error is increased by $\varepsilon$ for every interval check Alice does. As she does one check per source setting, we get a factor of $1 + |\mathcal{S}|$ increase in the error probability.

**Theorem III.3 (WSEE with decoy states).** Let $M = M_{\text{signal}}$. When Bob's storage is given by $\mathcal{F} = \mathcal{N}^{\otimes vM_{\text{store}}}$ for a storage rate $v > 0$, with $\mathcal{N}$ satisfying the strong converse property [19] and having capacity $C_{\mathcal{N}}$ bounded by
$$
C_{\mathcal{N}} v < \left( \frac{1}{2} - \delta \right) \frac{\tau}{P_{\text{B,click}}^{h|1}}
\tag{20}
$$
with $\tau \le 1 - r^{(1)}$, then Protocol 1 is an $(m, \lambda(\delta), \varepsilon(\delta), P_{\text{B,err}}^h)$ weak string erasure protocol with errors with the following parameters: Let $\delta \in ]0, \frac{1}{2} - C_{\mathcal{N}}v[$. Then the min-entropy rate $\lambda(\delta)$ is given by
$$
\lambda(\delta) = \min_{\{r^{(n)}\}_n} \frac{1}{m} \left[ v \gamma_{\mathcal{N}}^* \left(\frac{R}{v}\right) M_{\text{store}} - \sum_{n=2}^{\infty} M_{\text{left}}^{(n)} \log_2 (1 - P_{\text{B,err}}^{d,n}) \right]
\tag{21}
$$
where $\gamma_{\mathcal{N}}^*$ is the strong converse parameter of $\mathcal{N}$ [see (15)] and the minimization is taken over all $\{r^{(n)}\}_n$ with $1 - r^{(1)} \ge \tau$ such that $\sum_{n=1}^{\infty} r^{(n)} M^{(n)} < M_{\text{report}}^d$ and
$$
m = \sum_{n=1}^{\infty} M_{\text{left}}^{(n)} \quad M_{\text{store}} = P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1} M
\tag{22}
$$
$$
R = \left(\frac{1}{2} - \delta\right) \frac{1 - r^{(1)}}{P_{\text{B,click}}^{h|1}} M
\quad M_{\text{report}}^d = (P_{\text{B,no click}}^h - P_{\text{B,no click}}^d) M
\tag{23}
$$

[IMAGE: Figure 5. 1-2 oblivious transfer from fully randomized transfer by sending additional messages given by the dashed lines.]

for sufficiently large $M$. The error has the form
$$
\varepsilon(\delta) < (1 + |\mathcal{S}|) 2 \exp \left( - \frac{\delta^2}{512(4 + \log_2 \frac{1}{\delta})^2} \frac{\tau}{P_{\text{sent}}^{1|1}} M \right).
\tag{24}
$$

## IV. OBLIVIOUS TRANSFER FROM WSEE

We now show how to obtain oblivious transfer from WSEE. Here we implement a fully randomized oblivious transfer protocol (FROT), which can easily be converted into 1-2 oblivious transfer as shown in Fig. 5. We now give an informal description of this task and refer to Ref. [10] for a formal definition.

**Definition IV.1 (informal).** An $(\ell, \varepsilon)$ fully randomized oblivious transfer protocol (FROT) is a protocol between two parties, Alice and Bob, satisfying the following properties:

*   **Correctness:** If both parties are honest, then Alice obtains two random strings $S_0, S_1 \in \{0,1\}^{\ell}$, and Bob obtains a random choice bit $C \in \{0,1\}$ as well as $S_C$.
*   **Security for Alice:** If Alice is honest, then there exists $C \in \{0,1\}$ such that given $S_C$, Bob cannot learn anything about $S_{1-C}$, except with probability $\varepsilon$.
*   **Security for Bob:** If Bob is honest, then Alice learns nothing about $C$.

### A. Ingredients

#### 1. Suitable error-correcting codes

To deal with the bit-flip errors in the weak string erasure we need to augment the protocol of Ref. [10] with an additional error-correction step as in Ref. [25]. That is, Alice has to send some small amount of error-correcting information to Bob. The challenge we face is to ensure that security is preserved: Recall that if Bob is dishonest, we assume a worst-case scenario where he does not experience any transmission errors and he can perform perfect quantum operations. Hence, he could use this additional error-correcting information to correct some of the errors caused by his noisy quantum storage. On the other hand, if Alice is dishonest, we have to guarantee that

[IMAGE: Figure 6. Interactive hashing.]

the error-correcting process does not allow her to gain any information about the choice bit $C$. This last requirement can be achieved by using a *one-way* (or *forward*) error-correcting code in which only Alice sends information to Bob. Let $\{\mathcal{C}_n\}$ be a family of linear error-correcting codes of length $n$ capable of efficiently correcting $P_{\text{err}} n$ errors. For a $k$-bit string $X^k$, error correction is done by sending the syndrome information $\text{syn}(X^k)$ to Bob who can then efficiently recover $X^k$ from his noisy string $\mathcal{E}_{P_{\text{err}}}(X^k)$. For instance, low-density parity-check (LDPC) codes can correct a $k$-bit string, where each bit flipped with probability $P_{\text{err}}$, by sending at most $1.2 h(P_{\text{err}}) k$ bits of error-correcting information [27].

#### 2. Interactive hashing

Apart from an error-correcting code, the protocol below re- quires three classical ingredients that need to be implemented: First, we need to use the primitive of interactive hashing of subsets. This is a classical protocol in which Bob holds as input a subset $W^{\mathcal{I}} \subseteq [a]$ (where $a$ is some natural number) and Alice has no input. Both Alice and Bob receive two subsets $W_0, W_1 \subseteq [a]$ as outputs, where there exists some $C \in \{0,1\}$ such that $W_C = W^{\mathcal{I}}$ as depicted in Fig. 6. Informally, security means that Alice does not learn $C$, and $W_{1-C}$ is chosen almost at random from the set of all possible subsets of $[a]$. That is, Bob has very little control over the choice of $W_{1-C}$. Here we restrict ourselves to this definition and refer to [10] for a formal definition. In order to perform interactive hashing, we describe below how to encode the input subsets into a $t$-bit string. Intuitively, interactive hashing can be done by Alice asking Bob for random parities of his $t$-bit string $W$. After $t - 1$ linearly independent queries, there are only two possible strings left: one of which is Bob's original input, the other one is pretty much out of his control. A concrete protocol for interactive hashing can be found, for instance, in Ref. [28].

#### 3. Encoding of subsets

The second ingredient we need is thus an encoding of subsets as bit strings. More precisely, we map $t$-bit strings to subsets using $\text{Enc}: \{0,1\}^t \to \mathcal{T}$, where $\mathcal{T}$ is the set of all subsets of $[a]$ of size $a/4$. Here we assume without loss of generality that $a$ is a multiple of 4. The encoding $\text{Enc}$ is injective, that is, no two strings are mapped to the same subset. Below, we furthermore choose $t$ such that $2^t \le \binom{a}{a/4} \le 2 \cdot 2^t$. This means that not all possible subsets are encoded, but at least half of them. We refer to Refs. [28,29] for details on how to obtain such an encoding.

#### 4. Two-universal hashing

Finally, we require the use of two-universal hash functions for privacy amplification as they are also used in QKD [30]. Any implementation used for QKD may be used here. Below, we use $\mathcal{R}$ to denote the set of possible hash functions and use $\text{Ext}(X, R)$ to represent the output of the hash function given by $R$ when applied to the string $X$.

### B. Protocol

Before providing a detailed description of the protocol, we first give a description of the different steps involved in Fig. 7.

**Protocol 3: WSEE to FROT**
Parameters: Integers $m, \beta$ such that $\alpha := m/\beta$ is a multiple of 4. Set $t := \alpha/2$. Outputs: $(S_0^{\ell}, S_1^{\ell}) \in \{0,1\}^{\ell} \times \{0,1\}^{\ell}$ to Alice, and $(C, Y^{\ell}) \in \{0,1\} \times \{0,1\}^{\ell}$ to Bob.

1.  **Alice and Bob:** Execute $(m, \lambda, \varepsilon, P_{\text{err}}) \text{WSEE}$. Alice obtains a string $X^m \in \{0,1\}^m$, Bob a set $\mathcal{I} \subseteq [m]$ and a string $\tilde{S} = \mathcal{E}_{P_{\text{err}}}(X_{\mathcal{I}})$. If $|\mathcal{I}| < m/4$, Bob aborts. Otherwise, he randomly truncates $\mathcal{I}$ to the size $m/4$, and deletes the corresponding values in $\tilde{S}$.
    We arrange $X^m$ into a matrix $Z \in \mathcal{M}_{\alpha \times \beta}(\{0,1\})$, by $Z_{j,k} := X_{(j-1)\beta+k}$ for $(j,k) \in [\alpha] \times [\beta]$.
2.  **Bob:**
    1.  Randomly chooses a string $W^{\mathcal{I}} \in_R \{0,1\}^t$ corresponding to an encoding of a subset $\text{Enc}(W^{\mathcal{I}})$ of $[\alpha]$ with $\alpha/4$ elements.
    2.  Randomly partitions the $m$ bits of $X^m$ into $\alpha$ blocks of $\beta$ bits each: He randomly chooses a permutation $\pi: [\alpha] \times [\beta] \to [\alpha] \times [\beta]$ of the entries of $Z$ such that he knows $\pi(Z)_{\text{Enc}(W^{\mathcal{I}})}$ (that is, these bits are permutation of the bits of $\tilde{S}$). Formally, $\pi$ is uniform over permutations satisfying the following condition: for all $(j,k) \in [\alpha] \times [\beta]$ and $(j',k') := \pi(j,k)$, we have $(j-1)\beta+k \in \mathcal{I}$ if and only if $j' \in \text{Enc}(W^{\mathcal{I}})$.
    3.  Bob sends $\pi$ to Alice.
3.  **Alice and Bob:** Execute interactive hashing with Bob's input equal to $W^{\mathcal{I}}$. They obtain $W_0, W_1 \in \{0,1\}^t$ with $W_C = W^{\mathcal{I}}$.
4.  **Alice:** Sends error-correcting information for every block in $\text{Enc}(W_0)$ and $\text{Enc}(W_1)$, i.e., $\forall j \in \text{Enc}(W_0) \cup \text{Enc}(W_1)$, Alice sends $\text{Syn}(\pi(Z)_j)$ to Bob.
5.  **Alice:** Chooses $R_0, R_1 \in_R \mathcal{R}$ and sends them to Bob.
6.  **Alice:** Outputs $(S_0^{\ell}, S_1^{\ell}) := [\text{Ext}(\pi(Z)_{\text{Enc}(W_0)}, R_0), \text{Ext}(\pi(Z)_{\text{Enc}(W_1)}, R_1)]$.
7.  **Bob:** Computes $C$, where $W^C = W^{\mathcal{I}}$, and $\pi(Z)_{\text{Enc}(W^C)}$ from $\tilde{S}$. Performs error correction on the blocks of $\pi(Z)_{\text{Enc}(W^C)}$. He outputs $(C, Y^{\ell}) := [C, \text{Ext}(\pi(Z)_{\text{Enc}(W^C)}, R_C)]$.

[IMAGE: Figure 7. Conceptual steps in the protocol for FROT from WSEE.]

When using WSEE to obtain FROT, Protocol 3 achieves the following parameters. The proof of this statement can be found in Appendix B.

**Theorem IV.1 (oblivious transfer).** For any constant $\omega \ge 2$ and $\beta \ge \max\{67, 256\omega^2/\lambda^2\}$, the protocol WSEE-to-FROT implements an $(\ell, 41\cdot 2^{-5\lambda^2/(512\omega^2 \beta)} + 2\varepsilon)$-FROT from one in- stance of $(m, \lambda, \varepsilon, P_{\text{err}}) \text{WSEE}$, where
$$
\ell := \left\lfloor \left[ \left(\frac{\omega-1}{\omega} \frac{\lambda}{2} - \frac{\lambda^2}{512\omega^2 \beta} - \frac{1.2h(P_{\text{err}})}{2} \right) \frac{m}{\beta} - 1 \right] \right\rfloor.
$$

The parameter $\omega$ appearing in the theorem above is an additional parameter that we can tune to trade off a higher rate of OT against an error that decays more slowly. Our choice of $\omega$ will thereby depend on the error $P_{\text{err}}$: Note that for large values of $\omega$, we can essentially achieve security as long as $\lambda > h(P_{\text{err}})$ (see Fig. 8). Of course, this requires us to use many more rounds to be able to achieve the desired block size $\beta$, as well as to make the error sufficiently small again. Using more rounds, however, may be much easier than to decrease the bit error rate of the channel.

[IMAGE: Figure 8. (Color online) Security can be achieved if $(P_{\text{err}}, \lambda)$ lies in the shaded region, where we chose a very large value of $\omega = 100\,000$.]

## V. SECURITY FOR TWO CONCRETE IMPLEMENTATIONS

We now show how our security analysis applies to two particular experimental setups using weak coherent pulses or a parametric down-conversion source. Unlike in QKD, our protocols are particularly interesting at short distance, where one may use visible light for which better detectors exist.

### A. Phase-randomized weak coherent pulses

#### 1. Experimental setup and loss model

We first consider a phase-randomized weak coherent source. The basic setup for Alice and Bob is illustrated in Fig. 9. The signal states sent by Alice can be described as
$$
\rho_k = e^{-\mu} \sum_{n=0}^{\infty} \frac{\mu^n}{n!} |n k\rangle \langle n k|,
\tag{25}
$$
where the signals $|nk\rangle$ denote Fock states with $n$ photons in one of the four possible polarization states of the BB84 scheme, which are labeled with the index $k$.

On the receiving side, we shall assume that honest Bob uses an active-basis-choice measurement setup. It consists of a polarization analyzer and a polarization shifter which effectively changes the polarization basis of the subsequent measurement. The polarization analyzer has two threshold detectors, each monitoring the output of a polarizing beam splitter. These detectors are characterized by their detection efficiency $\eta$ and their dark-count probability $P_{\text{dark}}$. Note that we include all sources of loss in the system (including channel loss, coupling loss in Alice's and Bob's laboratory, etc.) in the definition of the detection efficiency $\eta$. For the case of honest Alice and Bob, the overall transmittance, $\eta$, is a product, i.e., $\eta = \eta_A \eta_{\text{channel}} \eta_B \eta_D$, where $\eta_A$ is the transmittance on Alice's side, $\eta_{\text{channel}}$ is the channel transmittance, $\eta_B$ is the transmittance on Bob's side (excluding detection inefficiency), and $\eta_D$ is the detector efficiency defined previously in the introductory section. Recall, from the introductory section, that $\eta_D$ is about 10% for telecom wavelengths and 70% for visible wavelengths.

Now, for some practical setups (such as short-distance free-space with visible wavelength), it is probably technologically feasible to achieve $\eta_A \eta_{\text{channel}} \eta_B$ of order 1, say 50%. In more detail, in some setups (e.g., with a weak coherent state source), Alice may compensate for her internal loss by characterizing it and then simply turning up the intensity of her laser. In those cases, she may effectively set $\eta_A = 1$. Now, for short-distance applications, $\eta_{\text{channel}}$ can be made of order 1. All that is required to achieve $\eta_A \eta_{\text{channel}} \eta_B$ of order 1 is to reduce Bob's internal loss, thus boosting $\eta$ to order 1. For simplicity, we consider that both detectors have equal parameters. Since we absorb all terms into the detector inefficiency, we simply refer to this as $\eta$.

As in QKD [36], the fact that each signal state is phase randomized is an important element for our security analysis. It allows us to argue that, without loss of generality, a dis- honest Bob always performs a quantum nondemolition (QND) measurement of the total number of photons contained in each pulse sent by Alice. Hence, we can analyze the single-photon pulses separately from the multiphoton pulses, which makes an important difference for Bob's cheating capabilities. In Appendix C, we compute all relevant probabilities to evaluate security in this scenario. These probabilities are summarized in Table I. For completeness, we explicitly state some parameters which we need in order to evaluate the error probability $P_{\text{B,err}}^h$. These parameters are as follows: the probability that Bob makes an error due to dark counts alone $(P_{\text{B,D,err}})$, the signal alone $(P_{\text{B,S,err}}^h)$, and the probability that he makes an error due to dark counts and the signal $(P_{\text{B,DS, err}}^h)$, as well as the probability that a signal alone produces no click in Bob's side $(P_{\text{B,S,no click}}^h)$.

[IMAGE: Figure 9. Experimental setup with phase-randomized weak coherent pulses. The Encoder codifies the BB84 signal information. The polarization shifter (PS) allows to change the polarization basis (computational basis + or Hadamard basis x) of the measurement as desired. The polarization analyzer consists of a polarizing beam splitter (PB) and two threshold detectors. The PB discriminates the two orthogonal polarized modes.]

---
$^6$ In this work, we are not considering the detection efficiency mismatch problem and detector-related attacks such as time-shift attacks [31,32] or faked-state attacks [33,34]. We remark that security proofs for QKD schemes that take into account such detection efficiency mismatch do exist, see, e.g., Ref. [35].

---

**TABLE I. Summary of the probabilities for phase-randomized weak coherent pulses.**

| Parameter | Value |
| :--- | :--- |
| $P_{\text{src}}^n$ | $e^{-\mu} \frac{\mu^n}{n!}$ |
| $P_{\text{sent}}^{n|1}$ | 0 |
| $P_{\text{B,click}}^{h|1}$ | $\eta + (1-\eta) P_{\text{dark}} (2 - P_{\text{dark}})$ |
| $P_{\text{B,err}}^{d,n}$ | 0 |
| $P_{\text{B,no click}}^h$ | $e^{-\mu} \sum_{n=0}^{\infty} \frac{\mu^n}{n!} (1 - \eta)^n = e^{-\mu \eta}$ |
| $P_{\text{B,S,no click}}^h$ | $P_{\text{B,S,no click}}^h + P_{\text{dark}} (2 - P_{\text{dark}})$ |
| $P_{\text{B,D,err}}$ | $P_{\text{dark}} (1 - P_{\text{dark}}) + P_{\text{dark}}^2/2$ |
| $P_{\text{B,S,err}}^h$ | $(1 - P_{\text{B,S,no click}}^h) [(1 - e_{\text{det}}) P_{\text{dark}} + e_{\text{det}} P_{\text{dark}} (\frac{3}{2} - P_{\text{dark}})]$ |
| $P_{\text{B,DS,err}}^h$ | $e_{\text{det}} (1 - P_{\text{B,S,no click}}^h)$ |
| $P_{\text{B,err}}^h$ | $P_{\text{B,S,err}}^h [1 - P_{\text{dark}} (2 - P_{\text{dark}})] + P_{\text{B,S,no click}}^h P_{\text{B,D,err}} + P_{\text{B,DS, err}}^h$ |

#### 2. Security parameters

To evaluate the probabilities above we assume that $P_{\text{dark}} = 0.85 \times 10^{-6}$ and use $e_{\text{det}} = 0.033$ as a very conservative number on a distance of 122 km [37].

##### a. Weak string erasure
We now investigate the security of $(m, \lambda, \varepsilon, P_{\text{err}})$ weak string erasure, when using a weak coherent source. Before examining the weak string erasure rate $\lambda$ that one can obtain for some set of source parameters, we first consider when security can be obtained in principle [i.e., when (12) and (13) are satisfied] as a function of the mean photon number $\mu$, the detection efficiency $\eta$, the storage rate $v$ and the amount of storage noise. Our examples here focus on the depolarizing channel with parameter $r$ as defined in (11). First, Fig. 10 tells us when security is possible at all, independently of the amount of storage noise. We then examine a particular example of storage noise and storage rates in

[IMAGE: Figure 10. (Color online) Security possible for $(\eta, \mu)$ in the shaded region where (12) is fulfilled. Our proof does not apply to parameters in the region below the curve. For the shaded region above the curve, additional conditions such as (13) are checked in Fig. 11.]

[IMAGE: Figure 12. (Color online) Security for $(r, v)$ below the lines for $\mu = 0.3$ and detection efficiencies $\eta$: 0.7 (solid black line), 0.5 (large dashed magenta line), 0.4 (dot-dashed blue line), 0.3 (dotted green line), and 0.2 (dashed red line).]

Fig. 11. This shows that even for low storage noise, we can hope to achieve security for many source settings. Note that this plot is merely an example, and, of course, does not rule out security of other forms of storage noise or other storage rates. The following plots have been made using MATHEMATICA, and the corresponding files used are available on request.

We now consider when conditions (12) and (13) can be satisfied in terms of the amount of noise in storage given by $r$, and the storage rate $v$ for some typical parameters in an experimental setup. Figure 12 shows us that there is a clear trade-off between $r$ and $v$ dictating when weak string erasure can be obtained from our analysis, but typical parameters of the source move us well within a possible region.

Now that we have established that secure weak string erasure can be obtained for a reasonable choice of parameters, it remains to establish the weak string erasure rate $\lambda$. This parameter cannot be read off explicitly but is determined by the optimization problem given in (16). To gain some intuition about the magnitude of this parameter we plot it in Fig. 13 for various choices of experimental settings and a storage rate of $v = 1$. This shows that even for a very high storage rate, there is a positive rate of $\lambda$ for many reasonable settings. Of course $\lambda$ can be larger if we were to consider a lower storage rate.

[IMAGE: Figure 11. (Color online) Security possible for $(\eta, \mu)$ in the upper enclosed regions for a low storage noise of $r = 0.9$ and storage rates $v$ of 1/2 (dashed red line), 0.45 (dotted green line), 0.35 (dot-dashed blue line), 0.25 (large dashed magenta line), and 0.15 (solid black line) [satisfying (12) and (13)].]

[IMAGE: Figure 13. (Color online) The WSE rate $\lambda$ in terms of the amount of depolarizing noise $r$ where $\mu = 0.3$, and a variety of detection efficiencies $\eta$: 0.7 (solid black line), 0.6 (dashed red line), 0.5 (dotted blue line), 0.4 (dot-dashed yellow line), 0.3 (large dashed magenta line), and 0.2 (larger dashed turquoise line).]

[IMAGE: Figure 14. (Color online) The WSE rate $\lambda$ in terms of the detection efficiency $\eta$ for $r = 0.8$ and storage rates $v$: 1/5 (solid blue line), 1/4 (dashed red line), 1/2 (dotted green line), and 2/3 (dot-dashed magenta line).]

To gain further intuition into the role that the different parameters play in determining the rate $\lambda$, we investigate the trade-off between $\lambda$ and the detection efficiency $\eta$ in Fig. 14, and the trade-off between $\lambda$ and the mean photon number $\mu$ in Fig. 15 for some choices of storage noise $r$ and storage rate $v$.

##### b. 1-2 oblivious transfer
We can now consider the security of $(\ell, \varepsilon)$ oblivious transfer based on weak string erasure implemented using a weak coherent source. The parameter which is of most concern to us here is the bit error rate $P_{\text{err}}^h = P_{\text{B,err}}^h / (1 - P_{\text{B, no click}}^h)$. As we already saw in Fig. 8, this error cannot be arbitrarily large for a fixed value of the WSE rate $\lambda$. In a practical implementation, this translates into a trade-off between the bit error $P_{\text{err}}^h$ and the efficiency $\eta$ as shown in Fig. 16, where for now we treat $P_{\text{err}}^h$ as an independent parameter to get an intuition for its contribution.

Of course $P_{\text{err}}^h$ is not an independent parameter but depends on $\mu$, $\eta$ and most crucially on $e_{\text{det}}$. Figure 17 shows how many bits $\ell$ of 1-2 oblivious transfer we can hope to obtain per valid pulse $M$ for very large $M$. The parameter $\mu$ has thereby been chosen to obtain a high rate when all other parameters were fixed. We will also refer to $\ell/M$ as the oblivious transfer rate. As expected, we can see that this rate does of course depend greatly on the efficiency $\eta$ but also on the storage noise and on the storage rate.

[IMAGE: Figure 15. (Color online) The WSE rate $\lambda$ in terms of the mean photon number $\mu$ for $r = 0.8$ and $v$ as in Figure 14.]

[IMAGE: Figure 16. (Color online) Security for $(P_{\text{err}}, \eta)$ in the shaded region, for example, parameters $r = 0.4, v = 1/5$ and large $\omega = 100\,000$.]

#### 3. Parameters using decoy states

We now analyze the scenario where Alice sends decoy states. In particular, let us consider a simple system with only two decoy states: vacuum and a weak decoy state with mean photon number $\hat{\mu}$. The mean photon number of the signal states will be denoted as $\mu$. Moreover, we select $\hat{\mu} < \mu$. Without loss of generality, we hence use labels $\mathcal{S} = \{\text{vac}, \hat{\mu}, \mu\}$ for the possible settings of the source. Furthermore, we assume that Alice chooses one of these settings uniformly at random, that is $P_{\mathcal{S}}(s) = 1/3$ for all $s \in \mathcal{S}$. This may not be optimal, but due to the large number of parameters we will limit ourselves to this choice. Since $P_{\text{src}}^n = P_{\text{sent}}^n$ for the case of a phase-randomized weak coherent source, we can write for honest Bob
$$
Q_{\text{vac}}^h = P_{\text{B,click}}^{h|0}
\tag{26}
$$
$$
Q_{\hat{\mu}}^h = e^{-\hat{\mu}} \sum_{n=0}^{\infty} \frac{\hat{\mu}^n}{n!} P_{\text{B,click}}^{h|n}
\tag{27}
$$

[IMAGE: Figure 17. (Color online) The rate $\ell/M$ of oblivious transfer for a large number of valid pulses $M$ for parameters $\omega = 1000$ and $\mu = 0.15, \eta = 0.3, r = 0.1, v = 1/10$ (solid blue line); $\mu = 0.4, \eta = 0.7, r = 0.1, v = 1/10$ (dashed red line); $\mu = 0.15, \eta = 0.7, r = 0.7, v = 1/4$ (dotted magenta line); and $\mu = 0.2, \eta = 0.7, r = 0.4, v = 1/3$ (light blue line).]

$$
Q_{\mu}^h = e^{-\mu} \sum_{n=0}^{\infty} \frac{\mu^n}{n!} P_{\text{B,click}}^{h|n}
\tag{28}
$$
For the typical channel model, that is, if Bob were honest, we furthermore have
$$
P_{\text{B,click}}^{h|0} = 2 P_{\text{dark}} (1 - P_{\text{dark}}) + P_{\text{dark}}^2
\tag{29}
$$
$$
P_{\text{B,click}}^{h|n} = 1 - (1 - P_{\text{B,click}}^{h|1}) (1 - \eta)^n.
\tag{30}
$$
For simplicity, when calculating the value of the parameter $P_{\text{B,click}}^{h|0}$ we have only considered the noise arising from dark counts in the detectors. In a practical situation, however, there might be other effects like stray light that also contribute to the final value of $P_{\text{B,click}}^{h|0}$. Still, from her knowledge of the experimental setup, Alice can always make a reasonable estimate of the maximum tolerable value of $P_{\text{B,click}}^{h|0}$ such that the protocol is not aborted and the analysis is completely analogous. Furthermore, we have assumed that the losses come mainly from the finite detection efficiency of the detectors, since the communication distance will be typically quite short.

To estimate a lower bound on the yield of single photons we follow the procedure proposed in Ref. [24]. Note, however, that many other estimation techniques are also available, like, for instance, linear programming tools [38]. In the asymptotic case we obtain [24]
$$
(1 - r^{(1)}) \ge \tau
$$
with
$$
\tau := \frac{\hat{\mu}}{\mu (\mu - \hat{\mu})} \left[ \left( Q_{\mu}^{\text{vac}} - \frac{\mu^2 - \hat{\mu}^2}{\mu^2} Q_{\text{vac}}^h \right) e^{\hat{\mu}} - \left( Q_{\hat{\mu}}^{\text{vac}} - \frac{\hat{\mu}^2 - \mu^2}{\mu^2} Q_{\text{vac}}^h \right) e^{\mu} \right]
\tag{31}
$$
where we used the fact that for honest Bob $P_{\text{B,click}}^{h|1} = 1 - r^{(1)}$ in the limit of large $M$, as Bob will decide to report any round as missing that he did not receive. In Protocol 2 we have that conditioned on the event that Alice does not abort the protocol
$$
Q_{\text{vac}}^{\text{meas}} \in [(Q_{\text{vac}}^h - \zeta_0^h), (Q_{\text{vac}}^h + \zeta_0^h)]
\tag{32}
$$
$$
Q_{\hat{\mu}}^{\text{meas}} \in [(Q_{\hat{\mu}}^h - \zeta_{\hat{\mu}}^h), (Q_{\hat{\mu}}^h + \zeta_{\hat{\mu}}^h)]
\tag{33}
$$
$$
Q_{\mu}^{\text{meas}} \in [(Q_{\mu}^h - \zeta_{\mu}^h), (Q_{\mu}^h + \zeta_{\mu}^h)]
\tag{34}
$$
where $\zeta_0 = \sqrt{\ln(2/\varepsilon)/(2M_0)}$, $\zeta_{\hat{\mu}} = \sqrt{\ln(2/\varepsilon)/(2M_{\hat{\mu}})}$, and $\zeta_{\mu} = \sqrt{\ln(2/\varepsilon)/(2M_{\mu})}$. We can hence bound
$$
(1 - r^{(1)}) \ge \tau
$$
with
$$
\tau := \frac{\hat{\mu}}{\mu (\mu - \hat{\mu})} \left[ \left( Q_{\mu}^{\text{meas}} - 2\zeta_{\mu} - \frac{\mu^2 - \hat{\mu}^2}{\mu^2} (Q_{\text{vac}}^{\text{meas}} + 2\zeta_0) \right) e^{\hat{\mu}} - \left( Q_{\hat{\mu}}^{\text{meas}} - 2\zeta_{\hat{\mu}} - \frac{\hat{\mu}^2 - \mu^2}{\mu^2} (Q_{\text{vac}}^{\text{meas}} + 2\zeta_0) \right) e^{\mu} \right]
\tag{35}
$$
which in the limit of large $M_0, M_{\mu}$, and $M_{\hat{\mu}}$ gives us (31). The factor 2 in Eq. (35) above stems from the fact that Alice still accepts a value at the upper (or lower) edge of the interval such as $Q_{\mu}^h + \zeta_{\mu}$. In this case however, the real parameter $Q_{\mu}^h$ is possibly as high as $Q_{\mu}^h + 2\zeta_{\mu}$.

[IMAGE: Figure 18. (Color online) Security possible for $(\eta, \mu)$ with decoy states in the shaded region where (12) is fulfilled. Additional conditions such as (13) are checked in Fig. 19.]

#### 4. Weak string erasure

For direct comparison, we now provide the same plots as given in subsection V A2a, where for simplicity we will always choose $\hat{\mu} = 0.05$. Of course, this may not be optimal, but serves as a good comparison. As expected using decoy states limits dishonest Bob from reporting too many single-photon rounds as missing, thereby allowing us to place a better bound on $r^{(1)}$. This fact greatly increases the range of parameters $\eta$ and $\mu$ for which we can hope to show security as shown in Figs. 18 and 19. We also observe in Fig. 20 that the detection efficiency $\eta$ plays almost no role in determining for which values of storage noise $r$ and storage rate $v$ we can obtain

[IMAGE: Figure 19. (Color online) Security possible for $(\eta, \mu)$ with decoy states in the upper enclosed regions for a low storage noise of $r = 0.9$ and storage rates 1/2 (dashed red line), 0.45 (dotted green line), 0.35 (dot-dashed blue line), 0.25 (large dashed magenta line), and 0.15 (solid black line). [satisfying (12) and (13)].]

[IMAGE: Figure 20. (Color online) Security for $(r, v)$ with decoy states below the lines for $\mu = 0.3$ and detection efficiencies $\eta$: 0.7 (solid black line), 0.5 (large dashed magenta line), 0.4 (dot-dashed blue line), 0.3 (dotted green line), and 0.2 (dashed red line).]

security. This is true for all values of $\mu \le 0.4$ we have chosen to examine.

It is, however, interesting to observe that the magnitude of the final weak string erasure rate $\lambda$ changes only slightly when we use decoy states. This is due to the strong converse parameter (15) which determines $\lambda$ as given in (16) and which is not necessarily large for larger values of $R$. This is shown by Fig. 21. Still, we again observe that we may use much lower values of $\eta$ as shown in Fig. 22 and a much higher mean photon number $\mu$ as shown in Fig. 23.

[IMAGE: Figure 21. (Color online) The WSE rate $\lambda$ for decoy states in terms of the amount of depolarizing noise $r$ where $\mu = 0.3$ and a variety of detection efficiencies $\eta$: 0.7 (solid black line), 0.6 (dashed green line), 0.5 (dotted blue line), 0.4 (dot-dashed yellow line), 0.3 (large dashed magenta line), and 0.2 (larger dashed turquoise line).]

[IMAGE: Figure 22. (Color online) The WSE rate $\lambda$ for decoy states in terms of the detection efficiency $\eta$ for $r = 0.8$ and storage rates $v$: 1/5 (solid blue line), 1/4 (dashed red line).]

[IMAGE: Figure 23. (Color online) The WSE rate $\lambda$ for decoy states in terms of the mean photon number $\mu$ for $r = 0.8$ and $v$ as in Fig. 22.]

QKD, it may be possible to use the remaining pulses which one could incorporate in our analysis given in the Appendix A. However, for clarity of exposition, we have chosen not to make use of such pulses in this work.

#### 5. 1-2 oblivious transfer

Again, we also consider the security of $(\ell, \varepsilon)$ oblivious transfer based on weak string erasure implemented using a weak coherent source and decoy states as above. We first observe that decoy states soften the trade-off between the bit error $P_{\text{err}}^h$ and the efficiency $\eta$ as shown in Fig. 24, where we for now treat $P_{\text{err}}^h$ as an independent parameter to get an intuition for its contribution. Figure 25 now shows how many bits $\ell$ of 1-2 oblivious transfer we can hope to obtain per valid pulse $M$ for very large $M$, when using decoy states. Again, we see that using decoy states softens the effects of $\eta$. Note that we again count only the valid pulses, which here corresponds to all pulses sent with the signal setting. As in

[IMAGE: Figure 24. (Color online) Security for $(P_{\text{err}}, \eta)$ with decoy states in the shaded region for example parameters $r = 0.4, v = 1/5$ and large $\omega = 100\,000$.]

[IMAGE: Figure 25. (Color online) The rate $\ell/M$ of oblivious transfer with decoy states for a large number of valid pulses $M$ for parameters $\omega = 1000$ and $\mu = 0.2, \eta = 0.3, r = 0.1, v = 1/10$ (solid blue line); $\mu = 0.3, \eta = 0.7, r = 0.1, v = 1/10$ (dashed red line); $\mu = 0.3, \eta = 0.7, r = 0.7, v = 1/4$ (dotted magenta line); and $\mu = 0.3, \eta = 0.7, r = 0.4, v = 1/3$ (light blue line).]

[IMAGE: Figure 26. Experimental setup with a PDC source. Alice and Bob measure each output signal by means of an active BB84 measurement setup, like the one described in subsection V A1.]

depicted in Fig. 26. We furthermore choose $\eta$ as in the case of a weak coherent source. That is, since $\eta_A \eta_{\text{channel}} \eta_B = 1$ we simply write $\eta = \eta_D$ for both parties. The dark count rate is again denoted by $P_{\text{dark}}$.

An important difference between the setup using a PDC source and the one using a weak coherent pulse source, is that Alice herself can (with some probability) discard a round if she concludes no photon or too many photons have been emitted. These rounds can be safely discarded by herself, and thus do not contribute to the protocol any further. We will refer to the remaining pulses as *valid*. To compare the two approaches more easily, we will assume that in the case of a PDC source, we consider only the valid pulses. That is, the parameter $M$ in the WSE protocol corresponds to the valid pulses and not to all pulses emitted by Alice. It is certainly debatable whether this is a fair comparison, but since $M$ is the parameter which is relevant to the security of the protocol, we choose to consider the final rates as a function of $M$.

The setting of a PDC source is slightly more difficult to analyze but can lead to better rates $\ell/M$ than those arising from a weak coherent source, where, like before, $\ell$ is the number of bits of oblivious transfer we obtain and $M$ is the number of valid pulses. The reason for this improvement is twofold: First, from her measurement results, Alice can (with some probability) estimate how many photons have been emitted each given time. This means that we are no longer restricted to tuning the source such that the number of multiphoton emissions is too low but can permit for a larger variation by relying on Alice to filter out the unwanted events. Second, a multiphoton emission does not provide dishonest Bob with full information about the signal state sent by Alice. In this scenario we need to consider the probability of success for dishonest Bob when a certain number of photons have been emitted which is given by Claim C2 in the Appendix C. Table II again summarizes the probabilities we need to know in order to evaluate security. Since some expressions can be rather unwieldy for the case of a PDC source, we will sometimes refer to the corresponding equation in the Appendix C.

#### 2. Security parameters

##### a. Weak string erasure
We now investigate the security of $(m, \lambda, \varepsilon, P_{\text{err}})$ weak string erasure, when using a PDC source. For easy comparison, we will consider exactly the same plots as before, where, however, we sometimes choose a different value for the mean photon number which seemed more useful

**TABLE II. Summary of probabilities for parametric down- conversion source.**

| Parameter | Value |
| :--- | :--- |
| $P_{\text{src}}^n$ | $\mu/[1 + (\mu/2)]^3$ |
| $P_{\text{sent}}^{n|1}$ | (C21) |
| $P_{\text{B,click}}^{h|1}$ | $\eta + (1-\eta) P_{\text{dark}} (2 - P_{\text{dark}})$ |
| $P_{\text{B,err}}^{d,n}$ | (C27) |
| $P_{\text{B,S,no click}}^h$ | $P_{\text{sent}}^{n|0}$, see (C21) |
| $P_{\text{B,no click}}^h$ | (C23) |
| $P_{\text{B,S,no click}}^h$ | $P_{\text{B,S,no click}}^h + P_{\text{dark}} (2 - P_{\text{dark}})$ |
| $P_{\text{B,D,err}}$ | $P_{\text{dark}} (1 - P_{\text{dark}}) + P_{\text{dark}}^2/2$ |
| $P_{\text{B,DS,err}}^h$ | (C26) |
| $P_{\text{B,S,err}}^h$ | (C24) |
| $P_{\text{B,err}}^h$ | $P_{\text{B,S,err}}^h [1 - P_{\text{dark}} (2 - P_{\text{dark}})] + P_{\text{B,S,no click}}^h P_{\text{B,D,err}} + P_{\text{B,DS, err}}^h$ |

[IMAGE: Figure 27. (Color online) Security possible for $(\eta, \mu)$ in the shaded region where (12) is fulfilled. Our proof does not apply to parameters in the region below the curve. For the shaded region above the curve, additional conditions such as (13) are checked in Fig. 28.]

for this source. For simplicity, we will also consider a setting where we give all the information encoded in multiphotons to dishonest Bob for free, i.e., we consider $P_{\text{B,err}}^{d,n} = 0$, which clearly overestimates his capabilities as we see in Claim C2. Again, we first consider when security can be obtained in principle [i.e., when (12) and (13) are satisfied] as a function of the mean photon number $\mu$, the detection efficiency $\eta$, the storage rate $v$, and the amount of storage noise, where our examples here focus on the depolarizing channel with parameter $r$ as defined in (11). Figure 27 thereby tells us again when security is possible at all, independently of the amount of

[IMAGE: Figure 28. (Color online) Security possible for $(\eta, \mu)$ in the upper enclosed regions for a low storage noise of $r = 0.9$ and storage rates 1/2 (dashed red line), 0.45 (dotted green line), 0.35 (dot-dashed blue line), 0.25 (large dashed magenta line), and 0.15 (solid black line) [satisfying (12) and (13)].]

[IMAGE: Figure 29. (Color online) Security for $(r, v)$ below the lines for $\mu = 0.3$ and detection efficiencies $\eta$: 0.7 (solid black line), 0.5 (large dashed magenta line), 0.4 (dot-dashed blue line), 0.3 (dotted green line), and 0.2 (dashed red line).]

storage noise. As before, we then examine a particular example of storage noise and storage rates in Fig. 28, that even for low storage noise, we can hope to achieve security for many source settings.

Second, we consider again when conditions (12) and (13) can be satisfied in terms of the amount of noise in storage given by $r$, and the storage rate $v$ for some typical parameters in an experimental setup in Fig. 29. It is interesting to note that the efficiency $\eta$ plays a much more prominent role when using a PDC source. This comes from the fact that Alice herself also uses a detector of efficiency $\eta$ to postselect some of the pulses.

Yet, we conclude that secure weak string erasure can be obtained for a reasonable choice of parameters, so it remains to establish the weak string erasure rate $\lambda$ by solving the optimization problem given by (16). Figure 30 gives us $\lambda$ for various choices of experimental settings, and a storage rate of $v = 1$. This demonstrates that even for a very high storage rate, there is a positive rate of $\lambda$ for many reasonable settings.

The trade-off between $\lambda$ and the detection efficiency $\eta$ given in Fig. 31 is quite similar to what we observed in the case of a weak coherent source. On the other hand, the trade-off between $\lambda$ and the mean photon number $\mu$ in Fig. 32 shows that having a low mean photon number seems more significant.

[IMAGE: Figure 30. (Color online) The WSE rate $\lambda$ in terms of the amount of depolarizing noise $r$ where $\mu = 0.3$ and a variety of detection efficiencies $\eta$: 0.7 (solid black line), 0.6 (dashed red line), 0.5 (dotted blue line), 0.4 (dot-dashed yellow line), 0.3 (large dashed magenta line), and 0.2 (larger dashed turquoise line).]

[IMAGE: Figure 31. (Color online) The WSE rate $\lambda$ in terms of the detection efficiency $\eta$ for $r = 0.8$ and storage rates $v$: 1/5 (solid blue line), 1/4 (dashed red line), 1/2 (dotted green line), and 2/3 (dot-dashed magenta line).]

[IMAGE: Figure 32. (Color online) The WSE rate $\lambda$ in terms of the mean photon number $\mu$ for $r = 0.8$ and $v$ as in Fig. 31.]

Recall, however, that we have for simplicity assumed that we give all multiphotons to Bob for free which greatly overestimates his capabilities when using a PDC source. These parameters could thus be improved when including multiphotons.

##### b. 1-2 oblivious transfer
We can now consider the security of $(\ell, \varepsilon)$ oblivious transfer based on weak string erasure implemented using a PDC source. In Fig. 33, we first examine the trade-off between an independently chosen bit error rate $P_{\text{err}}^h$ and the efficiency $\eta$, which is similar to what we observe for the case of a weak coherent source.

Figure 34 now shows how many bits $\ell$ of 1-2 oblivious transfer we can hope to obtain per valid pulse $M$ for very large $M$. This is much higher than what we observe for the case of a weak coherent source, but note that in all plots we only consider the *valid* pulses $M$. For a weak coherent source, this is equal to the actual number of pulses emitted as Alice does not postselect. However, for the case of a PDC source, Alice can (with some probability) discard rounds in which no photon has been emitted. This comparison is arguably unfair, but since $M$ is the parameter that is relevant to the security of our protocol, we chose to use the number of valid pulses, instead of the number of all pulses.

[IMAGE: Figure 33. (Color online) Security for $(P_{\text{err}}, \eta)$ in the shaded region, for example, parameters $r = 0.4, v = 1/5$ and large $\omega = 100\,000$.]

[IMAGE: Figure 34. (Color online) The rate $\ell/M$ of oblivious transfer for a large number of valid pulses $M$ for parameters $\mu = 0.05, \omega = 1000$ and $\eta = 0.3, r = 0.1, v = 1/10$ (solid blue line); $\eta = 0.7, r = 0.1, v = 1/10$ (dashed red line); $\eta = 0.7, r = 0.7, v = 1/4$ (dotted magenta line); $\eta = 0.7, r = 0.4, v = 1/3$ (light blue line). Note that the scaling of this plot differs for the WCP source with and without decoy states.]

## VI. CONCLUSIONS AND OPEN QUESTIONS

We have shown that security in the noisy-storage model [8,10] can in principle be obtained in a practical setting and provided explicit security parameters for two possible experimental setups. Our analysis shows that the protocols of Ref. [10] are well within reach of today's technology.

We have been mostly focusing our attention on short- distance (in the order of a few meters) applications. For this range, it is an interesting experimental challenge to construct small hand-held devices which can be used to implement these protocols. Nonetheless, in the future it might be interesting to study the curve between the rate and the distance of secure WSEE (in a similar way as the key rate versus distance curve in QKD). Such a curve will allow us to see if our protocols can be applied in a local area network (LAN) or metropolitan area network (MAN). Note that for medium-distance (say order 10 km) applications, our protocol may still work. For instance, standard telecom fiber has a channel loss of about 0.2 dB/km at telecom wavelength (i.e., 1550 nm) So, 10 km translates to only 2 dB channel loss, which seems quite manageable!

Many important theoretical (see Ref. [10]) as well as practical issues remain to be addressed. As in QKD, we have assumed that all experimental components behave as we expect them to. Hence, we have not considered any practical attacks such as exploiting detectors that are blind above a certain threshold [40], which is outside the scope of this work. Most importantly, however, it is certaintly possible to improve the parameters obtained here. These improvements can come from theoretical advances [10], as well as an exact optimization of all parameters for a particular experimental setup. Furthermore, in the case of parametric down-conversion, for example, we have not made use of the fact that Bob cannot gain full information from multiphoton emissions, which leads to an increase in rates. Similarly, when using decoy states, one could make use of pulses emitted using a decoy setting in the protocol. This requires a careful analysis of weak string erasure for different photon sources analogous to the one presented in the Appendix C. Nevertheless, we hope that this analysis paves the way for a practical implementation of protocols in the noisy-storage model.

## ACKNOWLEDGMENTS

We thank Matthias Christandl, Andrew Doherty, Chris Ervens, Robert König, Prabha Mandayam, John Preskill, Joe Renes, Gregor Weihs, and Jürg Wullschleger for interesting discussions on various aspects of the noisy-storage model. S.W. is supported by NSF grants PHY-04056720 and PHY- 0803371. M.C. is supported by Xunta de Galicia (Spain, Grant No. INCITE08PXIB322257PR). C.S. is supported by the EU fifth framework project QAP IST 015848 and the NWO VICI project 2004-2009. H.L. is supported by funding agencies CFI, CIPI, the CRC program, CIFAR, MITACS, NSERC, OIT, and QuantumWorks. S.W. and H.L. also thank the KITP program on quantum information science funded by NSF grant PHY-0551164. Part of this work was carried out during visits to the University of Toronto, Caltech, and KITP. We thank these institutions for their kind hospitality.

## APPENDIX A: PROOF OF SECURITY: WSEE

Here we show how the security proof of Ref. [10] can be modified to apply in the practical settings considered in this article. To this end, we first provide a more formal definition of WSEE.

**Definition A.1.** An $(m, \lambda, \varepsilon, P_{\text{err}})$ weak string erasure proto- col with errors (WSEE) is a protocol between Alice and Bob satisfying the following properties where $\mathcal{E}_{P_{\text{err}}}$ is defined as in (5):

*   **Correctness:** If both parties are honest, then the ideal state $\sigma_{X^m \mathcal{I}} \mathcal{E}_{P_{\text{err}}}(X_{\mathcal{I}})$ is defined such that
    1.  The joint distribution of the $m$-bit string $X^m$ and subset $\mathcal{I}$ is uniform:
$$
\sigma_{X^m \mathcal{I}} = \mathcal{T}_{\{0,1\}^m} \otimes \mathcal{T}_{2^{[m]}}
\tag{A1}
$$
    2.  The joint state $\rho_{\text{AB}}$ created by the real protocol is $\varepsilon$-close to the ideal state:
$$
\rho_{\text{AB}} \approx_{\varepsilon} \sigma_{X^m \mathcal{I}} \mathcal{E}_{P_{\text{err}}}(X_{\mathcal{I}}),
\tag{A2}
$$
where we identify $(A, B)$ with $[X^m, \mathcal{I} \mathcal{E}_{P_{\text{err}}}(X_{\mathcal{I}})]$.
*   **Security for Alice:** If Alice is honest, then there exists an ideal state $\sigma_{X^m B'}$ such that
    1.  The amount of information $B'$ gives Bob about $X^m$ is limited:
$$
\frac{1}{m} H_{\infty}^{\varepsilon}(X^m | B') \ge \lambda
\tag{A3}
$$
    2.  The joint state $\rho_{\text{AB}'}$ created by the real protocol is $\varepsilon$-close to the ideal state:
$$
\sigma_{X^m B'} \approx_{\varepsilon} \rho_{\text{AB}'}
\tag{A4}
$$
where we identify $(X^m, B')$ with $(A, B')$.
*   **Security for Bob:** If Bob is honest, then there exists an ideal state $\sigma_{A' \mathcal{I}}$ where $A' \mathcal{I}^m \in \{0,1\}^m$ and $\mathcal{I} \subseteq [m]$ such that
    1.  The random variable $\mathcal{I}$ is independent of $A' X^m$ and uniformly distributed over $2^{[m]}$:
$$
\sigma_{A' \mathcal{I}} = \sigma_{A' X^m} \otimes \mathcal{T}_{2^{[m]}}.
\tag{A5}
$$
    2.  The joint state $\rho_{\text{AB}}$ created by the real protocol is $\varepsilon$-close to the ideal state:
$$
\rho_{A' B} \approx_{\varepsilon} \sigma_{A' \mathcal{I}} \mathcal{E}_{P_{\text{err}}}(X_{\mathcal{I}})
\tag{A6}
$$
where we identify $(A', B)$ with $[A' \mathcal{I}^m, \mathcal{I} \mathcal{E}_{P_{\text{err}}}(X_{\mathcal{I}})]$.

We study Protocol 1, i.e., without the use of decoy states. The case of decoy states is analogous, where we obtain a different bound in (A19), as discussed in subsection III B. The analysis is essentially the same in both cases, only we bound certain parameters in a different way. The general security evaluation of correctness and the case when Bob is honest follows the same arguments as in Ref. [10]. It is clear by construction that an honest Bob reports enough rounds so that Alice does not abort except with probability $\varepsilon$ and hence the real output states are at most $\varepsilon$-far from the ideal states.

From now on we concentrate on the situation where Alice is honest, but Bob might try to cheat. Our analysis contains two steps. We first consider single-photon emissions, which we analyze as in Ref. [10], taking into account that Bob may report some additional single-photon rounds as missing. Second, we consider multiphoton rounds. The main difficulty arises from the fact that Bob may report up to
$$
M_{\max}^d = (P_{\text{B,no click}}^h + \zeta_{\text{B,no click}}^h + P_{\text{B,no click}}^d + \zeta_{\text{B,no click}}^d) M
\tag{A7}
$$
of the $M$ rounds as missing, where he himself can choose which rounds to report. First, note that we can assume that even a dishonest Bob always reports a round as missing if he receives a vacuum state. By the same arguments as in Sec. II we have that the number of rounds where Bob observes no click lies in the interval $[(P_{\text{B, no click}}^d - \zeta_{\text{B, no click}}^d)M, (P_{\text{B, no click}}^d + \zeta_{\text{B, no click}}^d)M]$ for $\zeta_{\text{B, no click}}^d = \sqrt{\ln(2/\varepsilon)/(2M)}$, except with probability $\varepsilon$. Here, we make a worst-case assumption that the number of rounds where dishonest Bob observes no click is given by
$$
M_{\text{nc}}^d = (P_{\text{B,no click}}^d + \zeta_{\text{B,no click}}^d)M
\tag{A8}
$$
and he can thus report up to
$$
M_{\text{report}}^d = M_{\max}^d - M_{\text{nc}}^d = (P_{\text{B,no click}}^h - P_{\text{B,no click}}^d + \zeta_{\text{B,no click}}^h + \zeta_{\text{B,no click}}^d) M
\tag{A9}
$$
rounds of his choice to be missing. Let $M^{(n)}$ denote the number of rounds corresponding to an $n$-photon emission, let $r^{(n)}$ denote the fraction of $n$ photon rounds that dishonest Bob chooses to report as missing, and let $M_{\text{left}}^{(n)} = (1 - r^{(n)})M^{(n)}$ denote the number of $n$ photon rounds that dishonest Bob has left. Note that in the limit of large $M$ we have $M^{(n)} = P_{\text{src}}^n M$. Clearly, we must have that
$$
\sum_{n=1}^{\infty} r^{(n)} M^{(n)} < M_{\text{report}}^d
\tag{A10}
$$
or Alice will abort the protocol.

#### 1. Single-photon emissions

Single photons are desirable, since they correspond to the idealized setting analyzed in Ref. [10] where Alice does indeed send BB84 states. Clearly, in the limit of large $M$, we expect roughly $P_{\text{sent}}^1 M$ single-photon rounds. However, since Bob may choose to report single-photon rounds as missing, we have to analyze how many rounds still contribute to our security analysis. The analysis of Ref. [10] links the security to the rate at which Bob has to send classical information through his noisy storage channel. In order to determine this rate, we first investigate the setting where he is not allowed to keep any quantum state.

Let $X^{(1)}$ denote the substring of $X^M$ that corresponds to single-photon emissions. In Ref. [10] the rate at which Bob needs to send information through his noisy-storage channel depends on an uncertainty relation using postmeasurement information. This uncertainty relation provides a bound on the min-entropy that Bob has about $X^{(1)}$ given a classical measurement outcome $K$, and the basis information he obtains later on. We are thus interested in the min-entropy
$$
H_{\infty}(X^{(1)} | K^{(1)} \Theta^{(1)}) = -\log_2 P_{\text{guess}}(X^{(1)} | K^{(1)} \Theta^{(1)}),
\tag{A11}
$$
where we use $K^{(1)}$ and $\Theta^{(1)}$ to denote Bob's classical information and the basis information corresponding to the single-photon rounds respectively and $P_{\text{guess}}$ is the probability that Bob guesses the string $X^{(1)}$ maximized over all choices of measurements anticipating his post-measurement information $\Theta^{(1)}$ [41]. Important for us is the fact that since Alice picks one of the four BB84 encodings uniformly at random in each time slot, the initial state
$$
\rho_{X^{(1)} \Theta^{(1)} K^{(1)}} = \bigotimes_{j=1}^{M^{(1)}} \rho_{x_j \theta_j k_j}
\tag{A12}
$$
has tensor-product form, and it follows from Ref. [41] together with Ref. [8] that also the state
$$
\rho_{X^{(1)} K^{(1)} \Theta^{(1)}} = \bigotimes_{j=1}^{M^{(1)}} \rho_{x_j k_j \theta_j}
\tag{A13}
$$
is a tensor product, that is, Bob's best strategy to guess $X^{(1)}$ purely with the help of classical information $K^{(1)}$ has tensor- product form. It is important to note that this does not mean that Bob does indeed perform a tensor-product attack in general. It merely states that with respect to the uncertainty he has about $X^{(1)}$ given only his classical information and the basis information if he kept no quantum computation, his best attack would be a tensor-product attack. And hence for any other classical information that he may obtain from his actual attack in the protocol, this uncertainty is only going to be greater.

We can now use the fact that the min-entropy of a tensor- product state is additive [8], to conclude that the min-entropy that Bob has about $X^{(1)}$ given $K^{(1)}$ and $\Theta^{(1)}$ is thus a min- entropy per bit, which allows us to compute the remaining min-entropy if Bob reports some of the single-photon rounds as missing. More precisely, if $X_{\text{left}}^{(1)}$ is the substring of $X^M$ corresponding to the single-photon rounds that Bob does not report as missing, we know from the uncertainty relation of Ref. [12] and a purification argument that
$$
H_{\infty}^{\varepsilon}(X_{\text{left}}^{(1)} | K_{\text{left}}^{(1)} \Theta_{\text{left}}^{(1)}) \ge \left( \frac{1}{2} - 2\delta \right) M_{\text{left}}^{(1)}
\tag{A14}
$$
where $K_{\text{left}}^{(1)}$ and $\Theta_{\text{left}}^{(1)}$ correspond to the classical and basis information respectively for the remaining single-photon rounds, and
$$
\varepsilon \le \exp \left[ - \frac{\delta^2 M_{\text{left}}^{(1)}}{32(2 + \log_2 \frac{1}{\delta})^2} \right]
\tag{A15}
$$
To determine the security as a whole, we of course need to take into account that dishonest Bob also holds some quantum information about $X_{\text{left}}^{(1)}$, besides his classical information. We adopt the notation of Ref. [10] and write
$$
\rho_{X_{\text{left}}^{(1)} K_{\text{left}}^{(1)} \Theta_{\text{left}}^{(1)} F(\mathcal{Q}_{\text{in}})} = \frac{1}{2^{|\mathcal{I}|}} \sum_{X, \Theta \in \mathcal{K}} P_{K|X=\mathbf{x}, \Theta=\mathbf{\theta}}(k) |\mathbf{x} \rangle \langle \mathbf{x}|_{\text{Alice}} \otimes |\mathbf{\theta} \rangle \langle \mathbf{\theta}| \otimes |k \rangle \langle k| \otimes \mathcal{F}(\zeta_{X \Theta k}),
\tag{A16}
$$
where Bob holds $B^{(1)} = \mathcal{O}_{K_{\text{left}}^{(1)} \Theta_{\text{left}}^{(1)}} \mathcal{F}(\mathcal{Q}_{\text{in}})$, and $\zeta_{X \Theta k} \in \mathcal{B}(\mathcal{Q}_{\text{in}})$ is the state entering Bob's quantum storage when Alice chose $X$ and $\Theta$, and Bob already extracted some classical information $k$. Here, $K^{(1)}$ includes all of Bob's classical information and depending on Bob's attack may not have tensor-product form. Nevertheless, we know from Ref. [10] that (A14) tells us at which rate cheating Bob has to send information through his storage channel $\mathcal{F}$ for any attack he conceives.

##### a. General storage noise

In particular, we can now make use of the uncertainty relation (A14) together with the analysis of Ref. [10], Lemma 2.2 and Theorem 3.3] to obtain that for single-photon rounds we have that for any attack of dishonest Bob
$$
H_{\infty}^{\varepsilon}(X_{\text{left}}^{(1)} | K_{\text{left}}^{(1)} \Theta_{\text{left}}^{(1)} \mathcal{F}(\mathcal{Q}_{\text{in}}^{(1)})) \ge -\log_2 P_{\text{succ}}^{\text{cert}}\left( \left(\frac{1}{2} - \delta\right) M_{\text{left}}^{(1)} \right)
\tag{A17}
$$
for
$$
\varepsilon \approx 2 \exp \left\{ - \frac{(\delta/4)^2}{32[2 + \log_2(4/\delta)]^2} M_{\text{left}}^{(1)} \right\}
\tag{A18}
$$
We note that we have from (A10) that
$$
r^{(1)} \le \min \left[ P_{\text{B,no click}}^h - P_{\text{B,no click}}^d + \zeta_{\text{B,no click}}^h + \zeta_{\text{B,no click}}^d, 1 \right]
\tag{A19}
$$
and hence
$$
M_{\text{left}}^{(1)} = (1 - r^{(1)}) M^{(1)}
\tag{A20}
$$
$$
\ge M^{(1)} \left( 1 - \frac{P_{\text{B,no click}}^h - P_{\text{B,no click}}^d + \zeta_{\text{B,no click}}^h + \zeta_{\text{B,no click}}^d}{P_{\text{sent}}^1 - \zeta_{\text{sent}}^1} \right)
\tag{A21}
$$
which in the limit of large $M$ gives us
$$
M_{\text{left}}^{(1)} \ge M(P_{\text{sent}}^1 + P_{\text{B,no click}}^d - P_{\text{B,no click}}^h)
\tag{A22}
$$
Since $r^{(1)}$ is chosen by dishonest Bob and hence is unknown to Alice, we bound $\varepsilon$ for any strategy of dishonest Bob as
$$
\varepsilon < 2 \exp \left\{ - \frac{(\delta/4)^2}{32[2 + \log_2(4/\delta)]^2} \left[ 1 - \left( \frac{P_{\text{B,no click}}^h - P_{\text{B,no click}}^d + \zeta_{\text{B,no click}}^h + \zeta_{\text{B,no click}}^d}{P_{\text{sent}}^1 - \zeta_{\text{sent}}^1} \right) \right] P_{\text{sent}}^1 M \right\}
\tag{A23}
$$
In the case of decoy states, we just obtain a better bound in (A19), where the remaining security analysis is analogous.

##### b. Tensor-product channels

Of particular interest is the case where Bob's storage noise is of the form $\mathcal{F} = \mathcal{N}^{\otimes vM_{\text{store}}}$, where $v$ is the storage rate, $M_{\text{store}}$ is the number of bits we count to determine Bob's storage, and $\mathcal{N}$ obeys the strong converse property [19]. As outlined earlier, we assume that the number of qubits that determines Bob's storage size is as in the idealistic setting of Ref. [10] given by the number of single-photon emissions that we expect an honest Bob to receive for large $M$, i.e., $M_{\text{store}}$.

From the strong converse property of $\mathcal{N}$ follows that
$$
-\log_2 P_{\text{succ}}^{\mathcal{N}^{\otimes vM_{\text{store}}}}(M_{\text{store}} R) \ge v \cdot \gamma_{\mathcal{N}}^*(R/v) M_{\text{store}},
\tag{A24}
$$
where $\gamma_{\mathcal{N}}^*(R/v) > 0$ for $C_{\mathcal{N}} \cdot v < R$ and $C_{\mathcal{N}}$ is the classical capacity of the channel $\mathcal{N}$ [19]. To achieve security in this setting we hence want to determine $R$ such that
$$
\left(\frac{1}{2} - \delta\right) M_{\text{left}}^{(1)} = R M_{\text{store}},
\tag{A25}
$$
which gives us
$$
R = \left(\frac{1}{2} - \delta\right) \frac{(1 - r^{(1)}) P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1}}{P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1}}
\quad \text{for } P_{\text{B,click}}^{h|1} > 0
\tag{A26}
$$
and $R = 0$ otherwise, which for large $M$ becomes
$$
R = \left(\frac{1}{2} - \delta\right) \frac{1 - r^{(1)}}{P_{\text{B,click}}^{h|1}}
\tag{A27}
$$
Whenever $P_{\text{B,click}}^{h|1} > 0$, note that $R$ can be significantly larger than $1/2$ due the difference between $M^{(1)}$ and $M_{\text{store}}$. We can now use (A19) to bound $R$ as
$$
R \ge \left(\frac{1}{2} - \delta\right) \max \left[ 0, \frac{1}{P_{\text{B,click}}^{h|1}} \frac{P_{\text{B,no click}}^h - P_{\text{B,no click}}^d + \zeta_{\text{B,no click}}^h + \zeta_{\text{B,no click}}^d}{P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1}} \right]
\tag{A28}
$$
which for large $M$ is just
$$
R \ge \left(\frac{1}{2} - \delta\right) \max \left[ 0, \frac{P_{\text{sent}}^1 - P_{\text{B,no click}}^h + P_{\text{B,no click}}^d}{P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1}} \right]
\tag{A29}
$$
Summarizing, we have that for any strategy of dishonest Bob
$$
H_{\infty}^{\varepsilon}(X_{\text{left}}^{(1)} | K_{\text{left}}^{(1)} \Theta_{\text{left}}^{(1)} \mathcal{F}(\mathcal{Q}_{\text{in}}^{(1)})) \ge v \gamma_{\mathcal{N}}^* \left( \frac{R}{v} \right) M_{\text{store}}.
\tag{A30}
$$

#### 2. Multiphoton emissions

It remains to address the case of multi-photon emissions. We analyze here a conservative scenario where dishonest Bob obtains the basis information for free whenever a multi-photon emission occurred. This situation can only make dishonest Bob more powerful. Note that this also means that Bob will never attempt to store such emissions, since he will never obtain more information about them as he already has. We thus assume that Bob keeps no quantum knowledge about the rounds corresponding to multiphoton emissions. We will see below that for the case of a PDC source, Bob nevertheless does not obtain full information about a bit in the case of a multiphoton emission.

For an $n$-photon emission, the probability that Bob performs a correct decoding is given by $(1 - P_{\text{B,err}}^{d,n})$. If bit $j$ of $X^M$ was generated by an $N=n$-photon emission, we thus have
$$
H_{\infty}(X_j | \Theta_j K_j, N = n) = -\log_2 (1 - P_{\text{B,err}}^{d,n}).
\tag{A31}
$$
Since we assume that Bob keeps no quantum information about the multiphoton rounds we may write his state corresponding to the rounds in which $n > 1$ photons have been emitted as
$$
\rho_{X_{\text{left}}^{(n)} B^{(n)}} = \bigotimes_j \rho_{x_j b_j}^{(n)},
\tag{A32}
$$
where $B^{(n)}$ is a classical register. Using the fact that the min-entropy is additive for a tensor-product state [8], we have that Bob's min-entropy about the substring $X_{\text{left}}^{(n)}$ of $X^M$ (belonging to $N=n$ photon emissions that Bob does not report as missing) is given by
$$
\frac{1}{M_{\text{left}}^{(n)}} H_{\infty}(X_{\text{left}}^{(n)} | K_{\text{left}}^{(n)} \Theta_{\text{left}}^{(n)}, N = n) = -\log_2 (1 - P_{\text{B,err}}^{d,n}).
\tag{A33}
$$

#### 3. Putting things together

Let $X^m$ be the substring of bits of $X^M$ that Bob does not report as missing. In order to determine the overall security parameters, we need to determine how much min-entropy dishonest Bob has about
$$
X^m = \bigcup_{n=1}^{\infty} X_{\text{left}}^{(n)}
\tag{A34}
$$
Since we assume that Bob keeps no quantum information about the multiphoton rounds we may write the state of the system if Bob is dishonest as
$$
\rho_{X^m B'} = \bigotimes_{n=1}^{\infty} \rho_{X_{\text{left}}^{(n)} B^{(n)}},
\tag{A35}
$$
where $B^{(n)}$ contains a copy of all classical information available to Bob and where we have reordered the systems into parts belonging to different photon number $n$. The following theorem comes from Ref. [10], Theorem 3.3], together with the discussion given above.

**Theorem A.2 (Security against Bob).** Fix $\delta \in ]0, \frac{1}{2}[$ and let
$$
\varepsilon \approx 2 \exp \left\{ - \frac{(\delta/4)^2}{32[2 + \log_2(4/\delta)]^2} M_{\text{left}}^{(1)} \right\}
\tag{A36}
$$
Then for any attack of a dishonest Bob with storage $\mathcal{F}: \mathcal{B}(\mathcal{H}_{\text{in}}) \to \mathcal{B}(\mathcal{H}_{\text{out}})$, there exists a cq-state $\sigma_{X^m B'}$ such that
1.  $\sigma_{X^m B'} \approx_{2\varepsilon} \rho_{X^m B'}$
2.  $H_{\infty}^{\varepsilon}(X^m | B') \ge \frac{1}{m} [-\log_2 P_{\text{succ}}^{\mathcal{F}}(R M_{\text{store}}) + \sum_{n=2}^{\infty} M_{\text{left}}^{(n)} \log_2 (1 - P_{\text{B,err}}^{d,n})]$
where $\rho_{X^m B'}$ is given by (A35).

*Proof.* Let $\sigma_{X_{\text{left}}^{(1)} B^{(1)}}$ be defined as in the analysis of single- photon emissions in Ref. [10]. Following the same arguments as in Ref. [10] and adding another $\varepsilon$ for the probability that the number of rounds in which Bob observes no click lies outside the interval $[(P_{\text{B,no click}}^d - \zeta_{\text{B,no click}}^d)M, (P_{\text{B,no click}}^d + \zeta_{\text{B,no click}}^d)M]$, we get $||\rho_{X_{\text{left}}^{(1)} B^{(1)}} - \sigma_{X_{\text{left}}^{(1)} B^{(1)}}||_1 \le 2\varepsilon$. Further- more, let $\sigma_{X_{\text{left}}^{(n)} B^{(n)}} = \rho_{X_{\text{left}}^{(n)} B^{(n)}}$ for $n > 1$ and let
$$
\sigma_{X^m B'} = \bigotimes_{n=1}^{\infty} \sigma_{X_{\text{left}}^{(n)} B^{(n)}}
\tag{A37}
$$
Note that by the subadditivity of the trace distance, we have
$$
\frac{1}{2} || \rho_{X^m B'} - \sigma_{X^m B'} ||_1 \le 2\varepsilon.
\tag{A38}
$$
It remains to show that $\sigma_{X^m B'}$ has high min-entropy. Note that
$$
H_{\infty}(X^m | B') = H_{\infty}(X_{\text{left}}^{(1)} | B^{(1)}) + \sum_{n=2}^{\infty} H_{\infty}(X_{\text{left}}^{(n)} | B^{(n)})
\tag{A39}
$$
where we have used the additivity of the min-entropy for tensor-product states [8], and that conditioning on independent information does not change the min-entropy. Our claim now follows immediately from subsections A 1 and A 2.
We can again specialize this result to the case of tensor- product channels.

**Corollary A.3 (Security against Bob).** Let Bob's storage be described by $\mathcal{F} = \mathcal{N}^{\otimes vM_{\text{store}}}$ with $v > 0$, $\mathcal{N}$ satisfying the strong converse property [19], and
$$
C_{\mathcal{N}} v < \min_{r^{(1)}} R,
\tag{A40}
$$
where $R$ is defined in (A26). Fix $\delta \in ]0, \min_{r^{(1)}} R - C_{\mathcal{N}} v[$. Then, for any attack of dishonest Bob there exists a cq-state $\sigma_{X^m B'}$ such that
1.  $\sigma_{X^m B'} \approx_{2\varepsilon} \rho_{X^m B'}$
2.  $H_{\infty}^{\varepsilon}(X^m | B') \ge \frac{1}{m} [M_{\text{store}} v \cdot \gamma_{\mathcal{N}}^*(R/v) - \sum_{n=2}^{\infty} M_{\text{left}}^{(n)} \log_2(1 - P_{\text{B,err}}^{d,n})]$
with $\rho_{X^m B'}$ and $\varepsilon$ given by (A35) and (A36) respectively.

Our main theorem now follows by allowing Bob to choose $\{r^{(n)}\}$ minimizing his total min-entropy. To be able to give an exact security guarantee we bound the parameter $\varepsilon$ which may depend on dishonest Bob's choice of $r^{(1)}$ using (A23).

**Theorem A.4 (Weak string erasure).** Protocol 1 is an $(m, \lambda(\delta), \varepsilon(\delta), P_{\text{B,err}}^h)$ weak string erasure protocol for the following two settings:

1.  Let Bob's storage be given by $\mathcal{F}: \mathcal{B}(\mathcal{H}_{\text{in}}) \to \mathcal{B}(\mathcal{H}_{\text{out}})$, and let $\delta \in ]0, \frac{1}{2}[$. Then we obtain a min-entropy rate
$$
\lambda(\delta) = \min_{\{r^{(n)}\}_n} \lim_{M \to \infty} \frac{1}{m} \left[ -\log_2 P_{\text{succ}}^{\mathcal{F}}(R M_{\text{store}}) - \sum_{n=2}^{\infty} M_{\text{left}}^{(n)} \log_2 (1 - P_{\text{B,err}}^{d,n}) \right]
\tag{A41}
$$
where the minimization is taken over all $\{r^{(n)}\}_n$ such that $\sum_{n=1}^{\infty} r^{(n)} M^{(n)} < M_{\text{report}}^d$ and
$$
m = \sum_{n=1}^{\infty} M_{\text{left}}^{(n)} \quad M_{\text{store}} = P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1} M
\tag{A42}
$$
$$
R = \left(\frac{1}{2} - \delta\right) \frac{1 - r^{(1)}}{P_{\text{B,click}}^{h|1}}
\tag{A43}
$$
$$
\varepsilon(\delta) < 4 \exp \left\{ - \frac{\delta^2}{512(4 + \log_2 \frac{1}{\delta})^2} \left[ P_{\text{sent}}^{1|1} P_{\text{B,click}}^{h|1} - (P_{\text{B,no click}}^h - P_{\text{B,no click}}^d + \zeta_{\text{B,no click}}^h + \zeta_{\text{B,no click}}^d) \right] M \right\}
\tag{A44}
$$

2.  Suppose $\mathcal{F} = \mathcal{N}^{\otimes vM_{\text{store}}}$ for a storage rate $v > 0$, $\mathcal{N}$ satisfying the strong converse property [19] and having capacity $C_{\mathcal{N}}$ bounded by
$$
C_{\mathcal{N}} v < \min_{r^{(1)}} R.
\tag{A45}
$$
Let $\delta \in ]0, \frac{1}{2} - C_{\mathcal{N}} v[$. Then we obtain a min-entropy rate of
$$
\lambda(\delta) = \min_{\{r^{(n)}\}_n} \frac{1}{m} \left[ v \gamma_{\mathcal{N}}^* \left(\frac{R}{v}\right) M_{\text{store}} - \sum_{n=2}^{\infty} M_{\text{left}}^{(n)} \log_2 (1 - P_{\text{B,err}}^{d,n}) \right]
\tag{A46}
$$
for sufficiently large $M$.

## APPENDIX B: PROOF OF SECURITY: FROT FROM WSEE

We show that our augmented protocol implements fully randomized oblivious transfer, as defined in Ref. [10]. The proofs of correctness and security for honest Bob are analogous to the ones given in Ref. [10], using the fact that the properties of the error-correcting code ensure that Bob obtains $S_C$ except with probability $\varepsilon$. Furthermore, note that a dishonest Alice cannot gain any information about $C$ from a one-way error-correction scheme. We therefore concentrate on proving security for an honest Alice when Bob is dishonest. The proof proceeds as in Ref. [10], except for a small variation which we state below.

**Lemma B.1 (Security for Alice).** Let $\ell := \lfloor [(\frac{\omega-1}{\omega} \frac{\lambda}{2} - \frac{\lambda^2}{512\omega^2 \beta} - \frac{1.2h(P_{\text{err}})}{2}) \frac{m}{\beta} - 1] \rfloor$. Then, Protocol WSEE to FROT satisfies security for Alice with an error of
$$
41 \times 2^{-\frac{\lambda^2}{512\omega^2 \beta}} + 2\varepsilon.
$$
*Proof.* We know from the analysis in Ref. [10] that
$$
H_{\infty}^{\varepsilon+4\delta} (\Pi(Z)_{\text{Enc}(W_C)} | S_C R_0 R_1 W_0 W_1 \Pi B^{\text{enc}}, A) \ge \left( \frac{\omega-1}{\omega} \frac{\lambda}{4} - \ell - 1 \right),
\tag{B1}
$$
where $B^{\text{enc}}$ is the system of dishonest Bob after the interactive hashing protocol and $A$ is the event that the interactive hashing protocol provides us with a set $W_{1-C}$ of high min-entropy. $\mathcal{A}$ has probability $\text{Prob}[\mathcal{A}] \ge 1 - 32\delta^2$, where $\delta = 2^{-\alpha \lambda^2 / (512\omega^2)}$. Here, Bob has some additional information given by the syn- dromes $\text{Syn}(\Pi(Z)_j)$ of the blocks $j \in \text{Enc}(W_0) \cup \text{Enc}(W_1)$. Let us denote the total of this error-correcting information by $\text{Syn} := \{\text{Syn}[\Pi(Z)_j]\}_{j \in \text{Enc}(W_0) \cup \text{Enc}(W_1)}$. Notice that even if the encodings overlap in some blocks, only the syndromes of the $\alpha/4$ blocks in $\text{Enc}(W_{1-C})$ lower Bob's min-entropy on $\Pi(Z)_{\text{Enc}(W_{1-C})}$. We can hence bound
$$
H_{\infty}^{\varepsilon+4\delta} (\Pi(Z)_{\text{Enc}(W_{1-C})} | S_C R_0 R_1 W_0 W_1 \Pi B^{\text{enc}}, \mathcal{A})
\ge H_{\infty}^{\varepsilon+4\delta} (\Pi(Z)_{\text{Enc}(W_{1-C})} | S_C R_0 R_1 W_0 W_1 \Pi B^{\text{enc}}, \mathcal{A})
- 1.2 \cdot h(P_{\text{err}}) \frac{m}{\alpha}
\tag{B2}
$$
$$
\ge \left[ \left(\frac{\omega-1}{\omega} \frac{\lambda}{4} - \frac{1.2h(P_{\text{err}})}{4} \right) m - \ell - 1 \right],
\tag{B3}
$$
$$
\ge \left( \frac{\omega-1}{\omega} \frac{\lambda}{4} - \frac{1.2h(P_{\text{err}})}{4} \right) m - \ell - 1
\tag{B4}
$$
where the first inequality follows from the chain rule, the monotonicity of the smooth min-entropy [30], and the fact that error-correction information needs to be sent for $\beta \alpha/4 = m/4$ bits. Using privacy amplification [30], we then have that, conditioned on the event $\mathcal{A}$,
$$
\frac{1}{2} || \tilde{\sigma}_{S_{1-C} S_C R_0 R_1 W_0 W_1 \Pi \text{Syn} B^{\text{enc}} - \mathcal{T}_{\{0,1\}^\ell} \otimes \tilde{\sigma}_{S_C R_0 R_1 W_0 W_1 \Pi \text{Syn} B^{\text{enc}}} ||_1 \le \delta + 2\varepsilon + 8\delta^2,
\tag{B5}
$$
since
$$
\ell < \left( \frac{\omega-1}{\omega} \frac{\lambda m}{4} - \frac{1.2h(P_{\text{err}}) m}{\alpha} - 2\ell - 1 \right)
\ge 2 \log_2 1/\delta = \frac{\lambda^2}{2 \cdot 512\omega^2}
$$
which follows from
$$
\ell < \left[ \left(\frac{\omega-1}{\omega} \frac{\lambda}{8} - \frac{1.2h(P_{\text{err}})}{8} \right) m - \frac{\lambda^2 \alpha}{512\omega^2} \frac{1}{2} \right]
$$
Let $B^* := (R_0 R_1 W_0 W_1 \Pi \text{Syn} B^{\text{enc}})$ be Bob's part in the output state. Since $\text{Prob}[\mathcal{A}] \ge 1 - 32\delta^2$, we get
$$
\tilde{\sigma}_{S_{1-C} S_C B^* C} \approx_{32\delta^2+9\delta+2\varepsilon} \mathcal{T}_{\{0,1\}^\ell} \otimes \tilde{\sigma}_{S_C B^* C}
$$
and
$$
\tilde{\sigma}_{S_0 S_1 B^*} = \rho_{S_0 S_1 B^*}.
$$
Since $\delta^2 < \delta$, this implies the security condition for Alice, with a total error of at most $41\delta + 2\varepsilon$.

## APPENDIX C: DERIVATION OF PARAMETERS

In this section, we show how to compute the parameters for both experimental setups.

### 1. Weak coherent source

The case of phase-randomized weak coherent pulses is particularly easy to analyze, since here we can assume that Bob always gains full knowledge of the encoded bit from a multiphoton emission. That is, $P_{\text{B,err}}^{d,n} = 0$ for all $n > 1$. In particular, this yields
$$
P_{\text{B,no click}}^{d,n} = P_{\text{src}}^n = e^{-\mu} \frac{\mu^n}{n!}
\tag{C1}
$$
and
$$
P_{\text{src}}^1 = e^{-\mu} \mu.
\tag{C2}
$$
The action of Bob's detection device can be described by two positive-operator valued measures (POVM), one for each of the two polarization bases $\beta \in \{+, \times\}$ used in the BB84 protocol. Each POVM contains four elements: $F_{\text{vac}}^{\beta}$, $F_0^{\beta}$, $F_1^{\beta}$, and $F_{\text{D}}^{\beta}$. The outcome of the first operator, $F_{\text{vac}}^{\beta}$, corresponds to no click in the detectors, the following two POVM operators, $F_0^{\beta}$ and $F_1^{\beta}$, give precisely one detection click, and the last one, $F_{\text{D}}^{\beta}$, gives rise to both detectors being triggered. If we denote by $|n,m\rangle_{\beta}$ the state which has $n$ photons in one mode and $m$ photons in the orthogonal polarization mode with respect to the polarization basis $\beta$, the elements of the POVM for this basis are given by
$$
F_{\text{vac}}^{\beta} = \sum_{n,m=0}^{\infty} \tilde{\eta}^n \tilde{\eta}^m |n,m\rangle_{\beta} \langle n,m|,
$$
$$
F_0^{\beta} = \sum_{n,m=0}^{\infty} \eta^n \tilde{\eta}^m |n,m\rangle_{\beta} \langle n,m|,
$$
$$
F_1^{\beta} = \sum_{n,m=0}^{\infty} \tilde{\eta}^n \eta^m |n,m\rangle_{\beta} \langle n,m|,
$$
$$
F_{\text{D}}^{\beta} = \sum_{n,m=0}^{\infty} \eta^n \eta^m |n,m\rangle_{\beta} \langle n,m|,
\tag{C3}
$$
where $\eta$ is the detection efficiency of a detector as introduced in subsection V A 1 and $\tilde{\eta} = (1-\eta)$. Furthermore, we take into account that the detectors show noise in the form of dark counts which are, to a good approximation, independent of the incoming signals. As in subsection V A 1, the dark count probability of each detector is denoted by $P_{\text{dark}}$.

First, since Alice does not verify how many photons have actually been emitted, we have
$$
P_{\text{sent}}^n = P_{\text{src}}^n
\tag{C4}
$$
To determine the other parameters, we start by computing the probability that an honest Bob does not observe a click due to a signal being sent which can be expressed as
$$
P_{\text{B,S,no click}}^h = \text{Tr}(F_{\text{vac}}^{\beta} \rho_k) = e^{-\mu} \sum_{n=0}^{\infty} \frac{\mu^n}{n!} (1 - \eta)^n
\tag{C5}
$$
with $\rho_k$ given by (25). Conversely, the probability that Bob does see a click due to a signal being sent is
$$
P_{\text{B,S,click}}^h = 1 - P_{\text{B,S,no click}}^h
\tag{C6}
$$
To calculate the total probability of Bob observing a click in his detection apparatus, we have to take dark counts into account. We now write the probability of Bob observing no-click due to a dark count as
$$
P_{\text{B,D,no click}} = (1 - P_{\text{dark}})^2,
\tag{C7}
$$
and the probability that at least one of his two detectors clicks becomes
$$
P_{\text{B,D,click}} = P_{\text{dark}} (2 - P_{\text{dark}}).
\tag{C8}
$$
The total probability that honest Bob observes a click is thus
$$
P_{\text{B,click}}^h = P_{\text{B,S,click}}^h P_{\text{B,D,no click}} + P_{\text{B,S,no click}}^h P_{\text{B,D,click}} + P_{\text{B,S,click}}^h P_{\text{B,D,click}}
$$
$$
= P_{\text{B,S,click}}^h + P_{\text{B,S,no click}}^h P_{\text{B,D,click}}.
\tag{C9}
$$
Note that
$$
P_{\text{B,no click}}^h = 1 - P_{\text{B,click}}^h
\tag{C10}
$$
To finish our analysis, it remains to evaluate the error probability for honest Bob, which determines how much error-correcting information Alice will send him. First, an error may occur from the signal itself, for example, due to misalignment in the channel. We have
$$
P_{\text{B,S,err}}^h = e_{\text{det}} P_{\text{B,S,click}}^h
\tag{C11}
$$
The second source of errors are dark counts. If the signal has been lost, the probability of making an error due to a dark count is given by the probability that Bob experiences a click in the wrong detector or both his detectors click. Hence, we have
$$
P_{\text{B,D,err}} = P_{\text{dark}} (1 - P_{\text{dark}}) + P_{\text{dark}}^2/2,
\tag{C12}
$$
where the second term stems from letting Bob flip a coin to determine the outcome bit when both of his detectors click. We can also have a combination of errors from the signal and the dark counts. Considering all different possibilities we obtain
$$
P_{\text{B,DS,err}}^h = P_{\text{B,S,click}}^h \left[ (1 - e_{\text{det}}) \frac{P_{\text{dark}}}{2} + e_{\text{det}} P_{\text{dark}} \left( \frac{3}{2} - P_{\text{dark}} \right) \right].
\tag{C13}
$$
Putting everything together we have
$$
P_{\text{B,err}}^h = P_{\text{B,S,err}}^h P_{\text{B,D,no click}} + P_{\text{B,S,no click}}^h P_{\text{B,D,err}} + P_{\text{B,DS,err}}^h.
\tag{C14}
$$

### 2. Parametric down-conversion source

In this section, we show how to compute all relevant parameters for a PDC source. Recall that at each time slot, the source itself emits an entangled state given by (36). The state $|\Phi_n\rangle_{\text{AB}}$ which appears in (38) can be written as
$$
|\Phi_n\rangle_{\text{AB}} = \sum_{m=0}^{n} \frac{(-1)^m \sqrt{n+1}}{\sqrt{(n-m)!} \sqrt{m!}} |m, n-m\rangle_{\text{A}} |m, n-m\rangle_{\text{B}}.
\tag{C15}
$$
We shall consider that both detectors on Alice's side are equal. In this situation, it is possible to attribute their losses to a single-loss beam splitter of transmittance $\eta$ as illustrated in Fig. 35. The creation operators $a_1^{\dagger}$ and $a_2^{\dagger}$ can be expressed as

[IMAGE: Figure 35. $a$ and $b$ denote the input modes to a beam splitter (BS) of transmittance $\eta$, while $c$ and $d$ are the output modes.]

for the two orthogonal polarization modes. Tracing out the modes $d_1$ and $d_2$ we obtain that the state shared by Alice and Bob, after accounting for Alice's losses, is given by
$$
\rho_{\text{AB}} = \sum_{n, n'} \sqrt{P_{\text{src}}^n P_{\text{src}}^{n'}} \sum_{m=0}^{\min(n, n')} \sum_{m'=0}^{\min(n', n)} \sum_{j=0}^{\min(n-m, n'-m')} \sum_{l=0}^{\min(m, m')} \dots
$$
$$
\dots \frac{(n-m)! m!}{(n-m-j)! j! (m-l)! l!} \dots
$$
Even though we again have two bases of course, we will only consider one of the two, the other one merely differs in a prior transform by Alice and does not change the resulting probabilities. For perfect threshold detectors, the probability that Alice sees a click in her first detector (concluding an encoding of "0") is given by
$$
P_{\text{A,S,click}}^0 = \text{Tr}[(\tilde{C}_1^A \otimes \mathcal{I}_B) \rho_{\text{AB}}]
$$
where
$$
\tilde{C}_1^A = \sum_{n=1}^{\infty} \frac{P_{\text{src}}^n}{n+1} \sum_{m=0}^{n-1} [(1 - \eta)^m - (1 - \eta)^n].
\tag{C16}
$$
$$
C_1^A = \sum_{n=1}^{\infty} |n\rangle \langle n|_{c_1} \otimes |0\rangle \langle 0|_{c_2}.
\tag{C17}
$$
The probability that she observes a click in the second detector is similarly determined by $P_{\text{A,S,click}}^1 = \text{Tr}[(\tilde{C}_2^A \otimes \mathcal{I}_B) \rho_{\text{AB}}]$ with
$$
C_2^A = |0\rangle \langle 0|_{c_1} \otimes \sum_{n=1}^{\infty} |n\rangle \langle n|_{c_2}.
\tag{C18}
$$
If Alice sees no click in a given round, or both her detectors click, she simply discards this round all together and it no longer contributes to the protocol. We have that $P_{\text{A,S,click}} = P_{\text{A,S,click}}^0$.

As discussed previously, we consider that the noise in the form of dark counts shown by the detectors is, to a good approximation, independent of the incoming signals. Then, to include this effect, we have to consider the probability of observing a click due to a dark count alone. This is given by the probability that we detect no photons
$$
P_{\text{vac}} = \text{Tr}[(|0,0\rangle \langle 0,0|_{c_1, c_2} \otimes \mathcal{I}_B) \rho_{\text{AB}}],
\tag{C19}
$$
but the detector clicks because of a dark count. We can obtain the probability that Alice observes only one click due to a signal or a dark count by considering operators of the form
$$
\tilde{C}_{\text{A}}^0 = (1 - P_{\text{dark}}) C_{\text{A}}^0 + (1 - P_{\text{dark}}) P_{\text{dark}} |0,0\rangle \langle 0,0|_{c_1, c_2},
$$
$$
\tilde{C}_{\text{A}}^1 = (1 - P_{\text{dark}}) C_{\text{A}}^1 + (1 - P_{\text{dark}}) P_{\text{dark}} |0,0\rangle \langle 0,0|_{c_1, c_2},
$$
which gives us
$$
P_{\text{A,click}}^0 = P_{\text{A,click}}^1 = (1 - P_{\text{dark}}) P_{\text{A,S,click}}^0
$$
$$
+ (1 - P_{\text{dark}}) P_{\text{dark}} \sum_{n=0}^{\infty} P_{\text{src}}^n (1 - \eta)^n.
\tag{C20}
$$
Combining everything, and tracing out Alice's regis- ter we obtain that Bob's unnormalized states are given by
$$
\tilde{\rho}_B^{0} = (1 - P_{\text{dark}}) \tilde{\rho}_B^{0, \text{sig}} + (1 - P_{\text{dark}}) P_{\text{dark}} \tilde{\rho}_B^{\text{vac}},
$$
$$
\tilde{\rho}_B^{1} = (1 - P_{\text{dark}}) \tilde{\rho}_B^{1, \text{sig}} + (1 - P_{\text{dark}}) P_{\text{dark}} \tilde{\rho}_B^{\text{vac}},
$$
with
$$
\tilde{\rho}_B^{0, \text{sig}} = \sum_{n=1}^{\infty} \frac{P_{\text{src}}^n}{n+1} \sum_{m=0}^{n-1} [(1 - \eta)^m - (1 - \eta)^n] |m, n-m\rangle \langle m, n-m|_B,
$$
$$
\tilde{\rho}_B^{1, \text{sig}} = \sum_{n=1}^{\infty} \frac{P_{\text{src}}^n}{n+1} \sum_{m=0}^{n-1} [(1 - \eta)^m - (1 - \eta)^n] |n-m, m\rangle \langle n-m, m|_B,
$$
$$
\tilde{\rho}_B^{\text{vac}} = \sum_{n=0}^{\infty} \frac{P_{\text{src}}^n}{n+1} \sum_{m=0}^{n} (1 - \eta)^n |m, n-m\rangle \langle m, n-m|_B.
$$
In the following, we use $\rho = \tilde{\rho} / \text{Tr}(\tilde{\rho})$ to refer to the normal- ized versions of these states. Note that these normalization factors are the same for an encoding of a "0" or a "1" and are given by $C_n = P_{\text{A,click}}$.
We can now write the probability that the source emits $n$ photons given that Alice obtained one single click in her measurement apparatus as
$$
P_{\text{sent}}^{n|1} := \frac{1}{C_n} P_{\text{src}}^n (1 - P_{\text{dark}}) \left[ P_{\text{dark}} (1 - \eta)^n + \frac{1}{n+1} \sum_{m=0}^{n-1} [(1 - \eta)^m - (1 - \eta)^n] \right]
\tag{C21}
$$
We are now ready to compute the probabilities relevant to the security analysis. First, we need to know the probability that honest Bob observes a click for the pulses where Alice has obtained one single click,
$$
P_{\text{B,click}}^{h|1} = P_{\text{B,S,click}}^h P_{\text{B,D,no click}} + P_{\text{B,S,no click}}^h P_{\text{B,D,click}} + P_{\text{B,S,click}}^h P_{\text{B,D,click}}.
\tag{C22}
$$
The probability that honest Bob does not observe a click at all, due to the signal is given by
$$
P_{\text{B,S,no click}}^h = \text{Tr}(\mathcal{F}_{\text{vac}} \rho_B^{1})
$$
$$
= \frac{1}{C} P_{\text{dark}} (1 - P_{\text{dark}}) \sum_{n=0}^{\infty} P_{\text{src}}^n (1 - \eta)^{2n}
$$
$$
+ (1 - P_{\text{dark}}) \sum_{n=1}^{\infty} \frac{P_{\text{src}}^n}{n+1} \sum_{m=0}^{n-1} [(1 - \eta)^m - (1 - \eta)^n] (1 - \eta)^n.
\tag{C23}
$$
and
$$
P_{\text{B,S,click}}^h = 1 - P_{\text{B,S,no click}}^h,
\tag{C24}
$$
where the probabilities $P_{\text{B,D,no click}}$ and $P_{\text{B,D,click}}$ are defined in the same way as in the previous section. We also need to determine the probability of an error for honest Bob. This is calculated analogous to the case of a weak coherent source, where we consider the probabilities of an error due to the signal itself, dark counts, and both combined. In our setting an honest Bob has two detectors to decide what bit Alice has encoded. If both detectors click, we shall consider again that honest Bob flips a coin to determine the outcome. It is enough to analyze the case of a "0" encoding; the "1" encoding provides the same result. The probability that Bob makes an error due to the signal is given by
$$
P_{\text{B,S,err}}^h = \text{Tr}(F_{\text{err}}^0 \rho_B^0),
\tag{C25}
$$
where
$$
F_{\text{err}}^0 = F_{\text{err}}^{0, 1} + F_{\text{err}}^{0, 0}
$$
$$
F_{\text{err}}^{0, 1} = (1 - e_{\text{det}}) F_0^1 + e_{\text{det}} F_1^1,
$$
and $F_{\text{err}}^{0, 0}, F_0^1$, and $F_1^1$ are given by (C3). Note that
$$
P_{\text{B,S,click}}^h = P_{\text{B,S,err}}^h + P_{\text{B,S,no err}}^h.
\tag{C26}
$$
Then, using that
$$
P_{\text{B,DS,err}}^h = P_{\text{B,S,err}}^h P_{\text{dark}} \left(\frac{3}{2} - P_{\text{dark}}\right) + P_{\text{B,S,no err}}^h P_{\text{dark}}^2
\tag{C27}
$$
we can now compute the combined error of Bob as in Eq. (C14).

In the case of PDC source we also need to compute Bob's success probability of decoding a bit from a multiphoton emission, if he is given the basis information for free. First, note that since $\tilde{\rho}_B^0$ and $\tilde{\rho}_B^1$ are Fock diagonal states, without loss of generality we can always assume that dishonest Bob first measures the photon number of each pulse sent by Alice, and afterward he performs his attack. For $n \ge 1$, we have
$$
\tilde{\rho}_B^{0, n, \text{sig}} = \frac{P_{\text{src}}^n}{n+1} \sum_{m=0}^{n-1} [(1 - \eta)^m - (1 - \eta)^n] |m, n-m\rangle \langle m, n-m|_B,
$$
$$
\tilde{\rho}_B^{1, n, \text{sig}} = \frac{P_{\text{src}}^n}{n+1} \sum_{m=0}^{n-1} [(1 - \eta)^m - (1 - \eta)^n] |n-m, m\rangle \langle n-m, m|_B,
$$
$$
\tilde{\rho}_B^{\text{vac}, n} = \frac{P_{\text{src}}^n}{n+1} \sum_{m=0}^{n} (1 - \eta)^n |m, n-m\rangle \langle m, n-m|_B.
$$
The unnormalized states of Bob containing $n$ photons and corresponding to an encoding of a "0" or "1" respectively can then be written as
$$
\tilde{\rho}_B^{0, n} = (1 - P_{\text{dark}}) \tilde{\rho}_B^{0, n, \text{sig}} + (1 - P_{\text{dark}}) P_{\text{dark}} \tilde{\rho}_B^{\text{vac}, n},
$$
$$
\tilde{\rho}_B^{1, n} = (1 - P_{\text{dark}}) \tilde{\rho}_B^{1, n, \text{sig}} + (1 - P_{\text{dark}}) P_{\text{dark}} \tilde{\rho}_B^{\text{vac}, n}.
$$
The normalization factor for both states is
$$
c_n = \text{Tr}(\tilde{\rho}_B^{0, n}) = (1 - P_{\text{dark}}) \text{Tr}(\tilde{\rho}_B^{0, n, \text{sig}}) + (1 - P_{\text{dark}}) P_{\text{dark}} \text{Tr}(\tilde{\rho}_B^{\text{vac}, n}),
$$
$$
= (1 - P_{\text{dark}}) \frac{P_{\text{src}}^n}{n+1} \sum_{m=0}^{n-1} [(1 - \eta)^m - (1 - \eta)^n] + (1 - P_{\text{dark}}) P_{\text{dark}} P_{\text{src}}^n (1 - \eta)^n.
$$

**Claim C.1.** The probability that Bob makes an error in decoding if Alice sent an $n$-photon signal and he is given the basis information for free is given by
$$
P_{\text{B,err}}^{d, n} = \frac{1}{2} - \frac{1}{4} \frac{P_{\text{dark}} P_{\text{src}}^n}{c_n} \sum_{m=0}^{n} (1 - \eta)^m - \frac{1}{2c_n} P_{\text{src}}^n \sum_{m=0}^{n-1} [(1 - \eta)^m - (1 - \eta)^{n-m}].
\tag{C28}
$$
*Proof.* This is an immediate consequence of Helstrom's theorem [42] using the fact that an encoding of "0" and "1" are a priori equally probable for Bob. Furthermore, note that $\tilde{\rho}_B^{0, n}$ and $\tilde{\rho}_B^{1, n}$ are both Fock diagonal, and hence their trace distance is simply given by the classical statistic distance on the right-hand side of
$$
\frac{1}{2} || \tilde{\rho}_B^{0, n} - \tilde{\rho}_B^{1, n} ||_1 = \frac{1}{2 c_n} P_{\text{dark}} P_{\text{src}}^n \frac{1}{n+1} \sum_{m=0}^{n} |(1 - \eta)^m - (1 - \eta)^{n-m}|.
\tag{C29}
$$