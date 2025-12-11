# Non-Interactive Oblivious Transfer and One-Time Programs from Noisy Quantum Storage

Ricardo Faleiro\*, Manuel Goulão\textsuperscript{†}, Leonardo Novo\textsuperscript{#}, and Emmanuel Zambrini Cruzeiro\textsuperscript{§}
\*Instituto de Telecomunicações, University of Aveiro, Portugal
\textsuperscript{†}INESC-ID, Instituto Superior Técnico, Universidade de Lisboa, Portugal;
Okinawa Institute of Science and Technology Graduate University, Okinawa, Japan
\textsuperscript{#}International Iberian Nanotechnology Laboratory (INL), Portugal
\textsuperscript{§}Instituto de Telecomunicações, Lisbon, Portugal;
Departamento de Engenharia Electrotécnica e de Computadores, Instituto Superior Técnico, Portugal

## Abstract

Few primitives are as intertwined with the foundations of cryptography as Oblivious Transfer (OT). Not surprisingly, with the advent of quantum information processing, a major research path has emerged, aiming to minimize the requirements necessary to achieve OT by leveraging quantum resources, while also exploring the implications for secure computation. Indeed, OT has been the target of renewed focus regarding its newfound quantum possibilities (and impossibilities), both towards its computation and communication complexity. For instance, non-interactive OT, known to be impossible classically, has been strongly pursued. In its most extreme form, non-interactive chosen-input OT (one-shot OT) is equivalent to a One-Time Memory (OTM). OTMs have been proposed as tamper-proof hardware solutions for constructing One-Time Programs – single-use programs that execute on an arbitrary input without revealing anything about their internal workings. In this work, we leverage quantum resources in the Noisy-Quantum-Storage Model to achieve:

1.  **Unconditionally-secure two-message non-interactive OT** — the smallest number of messages known to date for unconditionally-secure chosen-input OT.
2.  **Computationally-secure one-shot OT/OTM, with everlasting security**, assuming only one-way functions and sequential functions – without requiring trusted hardware, QROM, or pre-shared entanglement.
3.  **One-Time Programs without the need for hardware-based solutions or QROM**, by compiling our OTM construction with the [GKR08, GIS+10] compiler.

\*ricardofaleiro@tecnico.ulisboa.pt
\textsuperscript{†}manuel.goulao@inesc-id.pt
\textsuperscript{#}leonardo.novo@inl.int
\textsuperscript{§}emmanuel.cruzeiro@lx.it.pt
The authors are listed in alphabetical order.

1

## 1 Introduction

The field of quantum cryptography had its genesis with the concept of “Conjugate Coding” [Wie83]. The same primitive would later be published as Oblivious Transfer (OT) [Rab81], and would expand to become one of the most relevant primitives in cryptography. OT has different but equivalent formulations [Cré88], with the most prominent one being 1-out-of-2 OT [EGL85]. 1-out-of-2 OT is a simple protocol between two parties, the Sender and the Receiver, where the Sender has two input messages ($x_0, x_1$) and the Receiver has an input choice-bit $y$ and outputs the message $x_y$. This happens while the Sender remains oblivious to $y$ and the Receiver remains oblivious to $x_{1-y}$. In this work, the Sender messages are considered to be bits. Notably, a series of works established the impossibility of constructing unconditionally-secure OT and Bit Commitment (BC) without any assumption [LC97, May97, Lo97]. This ignited a research line focused on finding the minimal requirements to implement these primitives.

Given the impossibility to construct unconditionally-secure OT, some restriction must be introduced to its execution environment. Often, limitations to the computing power (e.g., computational hardness assumptions), or a restricted physical model (e.g., bounded/noisy memory, shared randomness) are introduced in the system to enable the desired functionality. Moreover, relevant results show that OT may be built from quantum computation and communication and (quantum-secure) One-Way Functions (OWFs) [GLSV21, BCKM21], or even weaker EFI pairs [BCQ23], thus relaxing the classical-world requirements of Public-Key Cryptography (PKC) [IR89]. This opens up a series of new possibilities for potential OT constructions, in particular, constructions that achieve otherwise unattainable security or efficiency levels. Indeed, low communication complexity is a highly desirable property in secure computation, and following this research line, this work proposes to answer the question:

What is the minimal number of communication rounds required
to construct 1-out-of-2 Oblivious Transfer?

Without further analysis, given that no restrictions are known on the minimal number of messages, achieving one-shot OT would be the best one could aim for. However, classically, two-message OT (one message each way) is optimal, as the messages of the Sender must somehow depend on the choice of the Receiver, or otherwise it could recover both messages, i.e., the protocol must be interactive. The pursuit of this two-message optimality has led to an extended research road (e.g., [NP01, PVW08, DGH+20]). Additionally, prior work has also explored delegating the OT functionality to a trusted device, known as the One-Time Memory (OTM) [GKR08], which can be transmitted between the parties, thus enabling non-interactive OT, assuming the device remains secure. Therefore, it is pertinent to study what happens when quantum computation and communication and quantum-secure computational assumptions are introduced.

We remark that, for the purposes of this work, a message means a single package of information sent from one party to the other. Thus, one-shot means that just one message is sent from one party to the other, as a single event. On the other hand, non-interactivity is used to state that communication is unidirectional, with one party sending possibly multiple messages to the other, which does not reply.\footnote{Note that non-interactivity is used with various different meanings in the literature, such as parties exchanging messages but the messages not depending on each other, e.g., non-interactive key-exchange.}

Remarkably, such a simple primitive as OT, by itself, is complete for general secure computation (Two-Party Computation (2PC) and Multi-Party Computation (MPC)) [Yao86, GMW87, Kil88]. Consequently, a further line of investigation that analyzes how to relate the complexity of OT

2

with the complexity of MPC has been pursued (e.g., [BL18, GS18]). Since these only account for OT built from classical resources, such works only aim for optimality as two messages of interactive communication (albeit sometimes with limited interaction, e.g., [IKO+11]). In [GKR08], the concept of One-Time Program (OTP) is introduced as program that can be run on an arbitrary single input, but only once, without revealing anything about itself (besides the output). The possibility to copy the program and running it again makes this primitive impossible to achieve, and to circumvent this issue, [GKR08] considers a secure memory device called OTM. Interestingly, an OTM is simply a rename of OT with the added restriction of being a non-interactive, 1-message primitive (i.e., one-shot OT), envisioned not as a cryptographic protocol (like OT), but as a physical tamper-proof hardware device. (This distinction has since been blurred in the literature.) Then, as OT yields 2PC, so does OTM yield OTP, with a compiler being able to achieve non-interactive malicious-secure OTP from OTM. The concept of OTP was further studied and generalized in [GIS+10], where it is shown that by allowing the parties to exchange tamper-proof hardware it is possible to directly build non-interactive secure 2PC. This leads to the question:

How to design One-Time Programs under weaker requirements and without trusted hardware?

### 1.1 Contributions

Three major conceptual contributions are established in this work, related to the proposed questions above. These are in the form of two 1-out-of-2 OT protocols secure against malicious adversaries, and, as a corollary, a construction for OTPs. As far as we are aware, these are the first evidences of such protocols in the scientific literature.

The first conceptual contribution answers that

**Two-message non-interactive 1-out-of-2 Oblivious Transfer is possible in the Noisy-Quantum-Storage Model, unconditionally.**

This solution exploits the Noisy-Quantum-Storage Model (NQSM), a model where the quantum memory of the parties performing the protocol, in particular the adversarial parties, is imperfect, and subject to noise. Thus, it prevents the indefinite (time) storage of quantum states, while no restrictions are made to the computing power or classical memory of the parties. Meanwhile, to execute the protocol, the honest parties require no quantum memory whatsoever. This is usually considered a general and weak assumption, as it is a realistic model that replicates the physical limitations of the present and near-future technology. Another construction is also provided by replacing the NQSM by the stronger assumption of the Bounded-Quantum-Storage Model (BQSM) to substantially improve efficiency and remove artificially introduced time-delays.

The second conceptual contribution evidences a

**One-shot 1-out-of-2 Oblivious Transfer (or, One-Time Memory) in the Noisy-Quantum-Storage Model assuming the existence of a One-Way Function and a Sequential Function.**

This construction again relies on the NQSM, but also depends on the existence of a quantum-secure OWF and the existence of a Sequential Function (SF), as the construction relies on the primitive of Time-Lock Puzzle (TLP). The existence of SFs (also called non-parallelizing languages [BGJ+16]), and their relation to the construction of TLPs have been previously studied [BGJ+16, JMRR21], while candidates for SFs ranging from hash functions (Quantum Random Oracle Model (QROM)) [CFHL21] to lattice-based assumptions [LM23, AMZ24] have been recently proposed in the literature. Again, no restrictions are made to the classical memory of the

3

parties, and no quantum memory is required to honestly complete the protocol. But now, it must be assumed that the parties are probabilistic-polynomial-time quantum machines and have limited computing power. Another technical contribution from this construction is the introduction of the use of the TLP primitive when proving security in the NQSM, such that the time it takes to solve the TLP enforces quantum decoherence of the memories of an adversary.

The third conceptual contribution exhibits a

**One-Time Program, secure against malicious adversaries, assuming the existence of a one-shot Oblivious Transfer/One-Time Memory in the Noisy-Quantum-Storage Model.**

Given that OTM is impossible in the plain model, even under computational assumptions [BGS13], we construct OTPs in the NQSM, achieving a solution in a weak trusted model — arguably the best one can hope for. We leverage our previous contribution, the one-shot OT/OTM, combined with the compiler from [GKR08] which transforms a malicious-secure OTM into a malicious-secure OTP for arbitrary functions. For this corollary, we need the same assumptions as before of a quantum-secure OWF and SF, and work in the NQSM. OTPs are an extremely powerful concept that is impossible to realize in most settings, with previous proposals relying on trusted hardware assumptions, which have often proven unreliable in practice. Here, we construct OTPs under the weakest assumptions to date. OTPs have applications such as software protection and one-time proofs [GKR08].

Finally, in terms of technical novelty, a key contribution is the explicit handling of post-measurement classical information in non-interactive OT, OTM, and consequently OTP. This approach enables security guarantees against the most general class of attacks conceivable by a malicious quantum adversary. Although this framework [GW10] has been proposed for some time, to our knowledge, this is the first instance of its application to OT, OTM, and OTP. The techniques employed are non-trivial and may have broader applications in quantum cryptography. Our approach relies on deriving tight upper bounds for the eigenvalues of qubit register states, which, interestingly, are connected to specific Hamiltonians of spin-chain systems, widely studied in condensed-matter physics for entirely different reasons. Additionally, a notable feature of our protocol is the potential use of entanglement to employ self-testing techniques, introducing an extra layer of (semi-)device-independent security.

### 1.2 Related Work

**OT in Restricted Settings:** The impossibility of unconditional OT from exclusively informational theoretical considerations demands extra assumptions, either physical or computational, beyond the validity of quantum mechanics [Lo97]. The first proposals for quantum-based OT protocols were constructed in the same setup of the original Conjugate Coding [Wie83], and were secure only in certain restricted settings [CK88, BBCS92]. In fact, the authors of [BBCS92] even described possible measurement attacks compromising Sender-security, wherein the Receiver would delay the measurements and implement multi-qubit measurements later on. Thus, in order to establish security, one should still require that the Receiver implements the measurements at the desired time, by any means necessary, e.g., computational or physical limitations. One alternative they propose is assuming the existence of commitment schemes secure against limited computing power, say using OWFs [BBCS92]. In fact, a recent line of work has confirmed the belief that the weaker assumption of OWFs suffices for secure OT in the quantum world [GLSV21, BCKM21], as opposed to the classical setting where PKC is known to be a requirement [IR89]. As an alternate approach, one may consider physically motivated restrictions, like bounding the memory of the adversaries, the BQSM. This type of restriction had already been invoked in the classical setting [Mau92, CM97],

4

with explicit OT constructions presented therein [CCM98], before being considered in the quantum setting [DFSS05] (only bounding the total quantum storage), and further generalized to a more realistic scenario [WST08] (unbounded quantum storage, but noisy). Precisely, in [Sch10] the NQSM was explicitly leveraged in order to prove the security of [BBCS92]. Recently, constructions leveraging physical restricted models (BQSM) together with computational assumptions (Learning-With-Errors) have been proposed, opening up a wide range of new applications and enabling device-independent OT [BY23].

**Non-Interactive OT and MPC:** The OT protocols proposed in this work are non-interactive, in the sense that communication is always one-way, from Sender to Receiver. For these kinds of OT protocols, perfect Receiver-security can be immediately established from reasonable physical principles, like the *no-signalling-from-the-future* [CDP10]. In fact, Wiesner's original proposal of Conjugate Coding [Wie83], even if not proven to be secure for the Sender, was non-interactive, and thus perfectly Receiver-secure. It follows naturally that physically constrained models precluding unbounded quantum storage, such as the BQSM and NQSM, would be prime candidates for constructing such non-interactive unconditionally-secure OT protocols. Indeed, in [DFSS05], where the BQSM was first introduced, a construction for non-interactive All-or-Nothing OT based on the original Conjugate Coding setup was introduced. This was followed by a non-interactive two-message 1-out-of-2 Random OT [DFR+07], also in the BQSM, which was further generalized to the NQSM [WST08].\footnote{Translating Random OT to chosen-input OT requires one extra message [DFSS06].} Lastly, different attempts to achieve secure computation non-interactively have been developed, e.g.: the subject of “Non-Interactive Secure Computation” [IKO+11, BL20], a 2PC scenario that can be computed in two steps, but one step is delegated to a pre-processing publishing phase (which in practice makes it interactive); or similarly, the “Private Simultaneous Messages" protocols [FKN94, BGI+14, HIJ+17, HIKR18], where some parties communicate to a different entity a message that depends on their input, but requires that they share some randomness source (again delegating interaction to a pre-processing phase).

**Round-Optimal OT and MPC:** Generally, the usefulness of round-optimal OT is drawn from trying to improve the communication round complexity of MPC. As its most costly primitive, and often reliant on PKC primitives, minimizing the round complexity of OT is paramount. In spite of this fact, while round-optimal OT has been a pursued goal for a long time, research on the topic has mostly been restricted to classical solutions for OT. Therefore, round-optimality of OT is largely and explicitly been considered to be two messages, and necessarily interactive [NP01, Kal05, PVW08, DGH+20, CSW20]. Furthermore, the round complexity of secure computation protocols has also been extensively studied in the literature. In particular, analyzing black-box constructions of MPC in the plain model is known to require at least four messages, and interactivity [KO04, GMPP16, ACJ17, BHP17, HHPV18, RCCG+20]. However, relaxations to the number of corrupted parties [IKP10, ACGJ18], or to the security [KO04, QWW18, ABJ+19, COWZ22] allow for more efficient protocols to be achieved that only take two messages. Also, assuming shared randomness, it is possible to compile $n$-message OT into $n$-message MPC, for $n \ge 2$ [BL18, GS18].

**Hardware and software OTM and OTP:** Considering OT and secure computation, together with non-interactivity, the notions of OTM and OTP naturally arouse [GKR08]. The pursuit of these constructions has continued since, originally focusing on secure hardware but later seeking to eliminate this strong requirement. In [DDKZ14], it was shown how to implement OTP without assuming the security of hardware devices, by relying on two trusted models that assume bounded

5

leakage and bounded storage. In [EGG+22], another hardware functionality (albeit more practical and available, named the counter lockbox) is used to devise OTMs and OTPs. A clear direction to approach the issue of copying the program and running it again is to explore the no-cloning property of quantum information. However, in [BGS13], it is shown that OTPs for all programs cannot exist in the plain model (even under computational assumptions), requiring some trusted setup. Still, [BGS13] generalized the concept of OTPs to quantum programs. In [BGZ21], secure hardware (but stateless) is once again used to achieve quantum OTPs assuming a bounded number of adversarial queries.

Recently, in [Liu23] an independent proposal for a OTM was introduced. The crucial distinction with our approach lies in the way one limits said time-frame. Our OTM leverages the NQSM to limit the adversary's ability to maintain coherent quantum states over time due to noise. In [Liu23], instead, an artificial bound on circuit depthness for a pre-fixed polynomial depth is considered (motivated by NISQ computers). On the other hand, our setup in the NQSM is based on the physical difficulty of obtaining high-fidelity coherent quantum memories, a weak and appealing security model often considered nowadays for OT and secure computation, e.g., [LPAK23]. Moreover, in [Liu23] the security proof requires the QROM, hinging on idealized oracle behavior and leading to the usual pitfalls of heuristic oracle modeling. In terms of efficiency, both protocols scale polynomially in the computational security parameter, i.e., as the adversarial power grows (polynomially), the communication also grows (polynomially). Another critical difference, regarding the adversarial model, [Liu23] imposes the strong assumption that adversaries only may do full quantum measurements over the entire qubit register (meaning they do not need to consider post-measurement information). We consider arbitrary measurements and with classical post-measurement information, which constitutes the broadest class of attacks possible, coherent attacks, safeguarding even against cleverly chosen arbitrary partial measurements.

Also recently, in [ABKK23, BKS23], the authors have tackled similar questions relating to non-interactive OT (without tackling OTM and OTP). In [ABKK23], three constructions for OT are presented. While the first solution claims to be one-shot, it assumes shared maximally-entangled pairs before the execution of the protocol, in a setup phase that is not accounted for as a round. This means that this protocol needs, effectively, two messages. This fact is explicitly acknowledged by the authors in their Section 2.2 [ABKK23], where it is mentioned that to construct a protocol without assuming the setup phase, one more message must be introduced in an interactive manner (i.e., in the other direction). Also, their construction is for Random OT in the QROM, in opposition to this work, where a chosen-input OT in the NQSM is proposed. In [BKS23], another construction for a one-shot OT in the shared EPR pairs model is proposed, under the sub-exponential LWE assumption, together with a proposal for an MPC protocol also requiring the QROM. In contrast, the contributions of our work rely on the NQSM to show that not only an unconditionally-secure two-message non-interactive chosen-input OT exists, but even a one-shot OT from OWFs and SFs and without the need to have pre-shared entanglement; furthermore, OTPs is built from these constructions.

### 1.3 Overview

Here, an overview of the main results is provided. The objective is to give intuition about the contributions of this work in a simple manner. As such, most of the arguments reasoning is built-up from well-known principles of quantum information and adapting them to the desired setting.

6

#### 1.3.1 Non-Interactive Oblivious Transfer and One-Time Memory

Constructing non-interactive OT and OTM has been an elusive task, known to be impossible classically, without any further computational assumptions and trusted models. Moreover, not only unconditionally-secure OT, but even OT from OWFs were widely held to be impossible. From recent results, it is now known that OT and MPC are possible from quantum computation and information and OWFs, without the need for PKC. Also, unconditionally-secure OT can be enabled by restricting the physical setting of its execution, specially interesting for realistic physical models. For the particular case of OTM (i.e., one-shot OT), its realization is impossible in the plain model even under computational assumptions [BGS13].

Two relevant OT constructions are provided, based on the realistic modeling of imperfect quantum memories, the NQSM:

*   **Non-interactive two-message unconditionally-secure** (chosen-input) 1-out-of-2 OT, **secure against malicious adversaries**.
*   **OTM, i.e., one-shot** (chosen-input) 1-out-of-2 OT, **assuming the existence of a OWF and a SF, secure against malicious adversaries.**

The first proposed OTs attains unconditional security, and is conceptualized in the NQSM so as to avoid the usual impossibility results. The NQSM is a highly appealing model, as physical quantum memories are imperfect and suffer from quantum decoherence relatively fast, and is specially relevant as the protocol does not require any memory to honestly run. The protocol works as follows:

1.  The Sender prepares two maximally-entangled qubits in which it encodes its inputs.
2.  The Sender hides these two qubits in a large set of uniformly random qubits, such that the Receiver cannot tell which qubits encode the information.
3.  The Receiver measures each qubit, in a basis defined by its input-bit, and stores the measurement results.
4.  After waiting some pre-defined time, the Sender communicates the encoding, which allows the Receiver to compute the desired OT output.

To ensure security, the NQSM establishes that, after some time, the quantum memory of the parties becomes irretrievable. So, this model is leveraged by making an adversary trying to break the protocol wait a predefined amount of time, such that it cannot make joint measurements on only the information qubits unless it guesses them correctly (as from separate measurements cheating is impossible). This can be made to be unfeasible by appropriately choosing the amount of hiding qubits that the Sender sends to the Receiver. Here, the distance between the statistical distributions become closer at a linear rate, and can be made sufficiently close by appropriately setting the statistical security parameter. Also, state preparation is very efficient and current technology already enables high-rate sending of qubits. Moreover, from the non-interactivity of the protocol, the Sender cannot do anything, as it is unable to extract information from future events. Evidently, in this construction, no quantum memory whatsoever is required for the honest parties to engage in the protocol.

A variation of the first protocol is also proposed, where time efficiency is increased in exchange for replacing the weaker NQSM with the stronger assumption of the BQSM. Here, the BQSM is exploited, as it allows for an instant to be chosen when the adversary can only store a subset of its total quantum memory. If this instant is chosen to be exactly between the Sender sending the

7

qubits and sending the encoding, then, no waiting time is required to achieve security, given that a large enough number of qubits are sent to mask the legitimate ones.

The second proposed OT achieves the captivating goal of being one-shot, i.e., constitutes a OTM. Here, for the first time, the NQSM is connected with the concept of TLP. Conveniently, a TLP is a primitive that allows for a party to send a hidden message to another, such that the recipient must spend some time (via computation) to recover the concealed information. So, from the NQSM, by requiring that an adversary must spend some physical time to gain information that would enable an attack, its memory storage suffers from the phenomenon of quantum decoherence, and the attack becomes unfeasible. In this particular construction, the same rationale from the previous one is used, where the information qubits are hidden among random qubits, such that an adversary cannot perform joint measurements. But here, the encoding is hidden inside the TLP and sent together with the full state, and the parameters of the TLP, i.e., the time it takes to solve it is chosen such that quantum decoherence would happen in the meantime. Thus, a malicious Receiver cannot store the qubits until it knows the encoding of which two to measure jointly and break security. This is essentially the same situation as in the previous two-message construction, but delegates the time-keeping from the Sender to a computational cryptographic primitive to achieve the OTM. Moreover, using a TLP does not give any advantage to a malicious Sender to learn the input of the Receiver. Hence, this OT protocol immediately achieves everlasting security, meaning that the non-chosen message becomes perfectly irretrievable after the execution of the protocol, since the computational assumption only needs to hold while the TLP is being solved. Clearly, also in this construction, no quantum memory whatsoever is required for the honest parties to engage in the protocol.

**Remark.** Our OT proposal is secure both regarding indistinguishability of statistical distributions (via the hiding of the information in the qubits — quantum communication), and the computational unfeasibility of a polynomially-bounded quantum adversary to prematurely extract a secret encoding (via the TLP building block — classical communication). First, the computational security of the protocol must account for a growing computational power of any adversary (represented by a computational security parameter), meaning that the communicated TLP size grows polynomially to the computational security parameter. In opposition, the number of qubits that must be sent to ensure security does not grow with the power of the adversary (it is constant), and must be set to the desirable level of indistinguishability of the distributions according to the statistical security parameter. We instantiate this statistical distance to $2^{-40}$, meaning that the constant number of qubits sent must be $2^{40}$ independently of the power of the adversary. Therefore, in our first proposal (non-interactive OT), the classical and quantum communication is constant in relation to the adversarial power (statistical security); and, in our second proposal (OTM), the quantum communication is constant and the classical communication grows polynomially with the computational security parameter, as does the size of our proposal for the OTP construction.

#### 1.3.2 One-Time Program

Influenced by the seminal work of Yao's garbled circuits [Yao86], [GKR08] introduced the concept of OTP. An OTP implements the functionality of a black-box that can only be evaluated once on an arbitrary input $x$, and returns the value of a function $f$ on this input. Security of the OTM requires that no adversary can learn more about $f$ than what can be learned from a single tuple $(x, f(x))$.

In [GKR08], a compiler is introduced that constructs a malicious-secure OTP for an arbitrary function from a parallel-OTM. We show that our OTM may be run in parallel without hindering

8

security, particularly not allowing an adversary to query the OTMs adaptively, fulfilling the requirements of a parallel-OTM. Thus, as a direct corollary of the compiler of [GKR08], we achieve an OTP from the constructed OTM in the NQSM, assuming the existence of a OWF and a SF. We stress that our OTP construction has the limitation that it must be run directly after being received due to working in the NQSM, as qubits holding the information will suffer decoherence, making it impossible to run the OTP after a long enough period of time.

### 1.4 Open Questions

The statistical security component of our proposed constructions require a large number of qubits (exponential in the statistical security parameter) to ensure security. This happens as the security of the protocols rely on hiding information in a combinatorial manner with the number of sent qubits. While constant for a fixed statistical distance of the distributions and independent of any adversary power (even unbounded), this large number of qubits required hinders efficiency of the protocols, given the current available technology. Hence, removing this requirement would be of relevance, even if replacing it by computational assumptions. Following the same approach that was used for this construction, one could try to leverage a relationship between higher-dimensional qudits and string OT.

Moreover, device-independent security extends the standard notion of security, such that even the devices or laboratories used by the parties do not need to be trusted. Although this is a highly appealing security model, demonstrated for OT [KW16, BY23] and other cryptographic primitives [PAM+10, VV14, AMPS16, FG21], it is also extremely demanding, as it relies on the violation of Bell inequalities. To address this challenge, semi-device-independence relaxes the model by allowing certain assumptions to be made, while still preserving the essential properties of the quantum systems that ensure security. The use of entanglement in our construction makes it an attractive candidate for analysis within the device-independence framework. For example, introducing self-testing as a subroutine in some rounds of the protocol could partially verify the resources used, adding a layer of (semi-)device-independence to the security.

## 2 Background

### 2.1 Quantum Systems, States, and Processes

A finite $d$-dimensional quantum system is represented by a Hilbert space $\mathcal{H} \simeq \mathbb{C}^d$. Of fundamental importance in quantum information is the 2-dimensional quantum system $\mathcal{H} \simeq \mathbb{C}^2$, the **qubit**. Composition of quantum systems is given by the tensor product of individual Hilbert spaces, such that a system of $n$-qubits, often called a $n$-qubit register, is represented by $\mathcal{H} \simeq \mathbb{C}_{(1)}^2 \otimes \cdots \otimes \mathbb{C}_{(n)}^2 \simeq \mathbb{C}^{2^n}$.

The state-space of a quantum system is given by the set of all trace one, Hermitian, positive semi-definite operators acting on the corresponding Hilbert space, i.e., $\rho \in \mathcal{L}(\mathcal{H}) \simeq \mathbb{C}^{d \times d}$. **Pure states** can be described by outer products of vectors of the Hilbert space $\rho = |\psi\rangle\langle \psi|$ and, in that case, it is customary to represent the state of the system by the vector itself, $|\psi\rangle \in \mathcal{H} \simeq \mathbb{C}^d$. Pure states in composite systems are said to be **entangled**, if they cannot be factorized into vectors of the product Hilbert spaces. Also important are the four different two-qubit ($\mathbb{C}^2 \otimes \mathbb{C}^2$) maximally entangled states, known as **Bell states**,
$$|B_{xy}\rangle_{SR} = \frac{1}{\sqrt{2}} (|0\bar{y}\rangle + (-1)^x |1y\rangle)_{SR}, \quad (2.1)$$

9

for $x, y \in \{0, 1\}$, and $\bar{y}$ being the negation of $y$. A two-qubit pair in any of the Bell states is said to form an **Einstein-Podolsky-Rosen (EPR) pair** (or Bell pair).

In quantum-information processing, it is useful to adopt an operational perspective when describing the evolution of quantum systems throughout protocols. From that perspective, one considers different types of idealized black-box processes that can be implemented on quantum systems, changing their states at different stages. Fundamentally, three processes are noteworthy:

*   **Preparation** (classical-to-quantum process): Process with non-trivial classical input $x$, which outputs a corresponding quantum state $\rho_x$ obeying the usual normalization $\text{Tr}(\rho_x) = 1$.
*   **Transformation** (quantum-to-quantum process): Process taking as input a state $\rho_{in}$ and outputting $\rho_{out} = \Phi(\rho_{in}) = \sum_k E_k \rho E_k^\dagger$, for $\Phi \in \{\mathcal{L}(\mathcal{H}_{in}) \to \mathcal{L}(\mathcal{H}_{out})\}$ a **Completely Positive Trace Preserving (CPTP) map**, and $\{E_k\}$ the corresponding **Kraus operators** satisfying $\sum_k E_k^\dagger E_k = \mathbb{I}$. For a unitary transformation $U$ ($U^\dagger U = U U^\dagger = \mathbb{I}$), it simplifies to $\rho_{out} = U \rho_{in} U^\dagger$. Transformations can also be considered to have a classical control-input whose value dictates the fixed transformation applied.

Especially important in this work are the $X, Y, Z$ Pauli unitaries and the Hadamard transform, given in matrix form, respectively, as
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}, \quad H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \quad (2.2)$$

*   **Measurement** (quantum-to-classical process): Process with non-trivial input tuple $(y, \rho)$ (classical $y$ and quantum $\rho$), and a classical output $m$. It is modelled by a **Positive Operator-Valued Measure (POVM)** $\{M_{m|y}\}_m$, such that, for input $(y, \rho)$, it outputs $m$ with probability given by the **Born rule**, $\text{Tr}(\rho M_{m|y})$.

Finally, and since transformations can be absorbed either by measurements and/or preparations, the overall probabilities predicted in previous scenarios (often called Prepare-and-Measure (PM) scenarios) are given by
$$P[m|x, y] = \text{Tr}(\rho_x M_{m|y}), \quad (2.3)$$
where the classical inputs $(x, y)$ unambiguously specify the preparation and measurement for the given protocol setup.

### 2.2 Quantum State Discrimination with Post-Measurement Information

In this section, the formalism of [GW10] is introduced, which will be required to analyze the security of the proposed protocols. Quantum state discrimination is a specific task in the PM scenario. Therein, Bob has no input and tries to decode Alice's classical input with the highest probability by optimally distinguishing between the quantum states which encode her message. In [GW10], the state discrimination task is analyzed when classical information related to the preparation is revealed by Alice (who prepares the state according to some information string $x$ and some encoding $e$, where the latter is then revealed) to Bob (who measures the state and tries to guess $x$). But, this reveal is conditioned on the fact that Bob did measure the state and holds no quantum information when receiving this information.

An upper bound is shown to hold when the revealed post-measurement information by Alice ($e \in \mathcal{E}$ with probability $p_e$, where $\mathcal{E}$ is the set of all possible encodings) and the previously measured

10

information by Bob ($x$ with probability $p_x$) form a product distribution ($p_{x, e} = p_x p_e$), and for the preparation of the state $x$ is sampled from the uniform distribution, i.e., $p_x = 1/|\mathcal{X}|$.

Moreover, without loss of generality, it is assumed that Bob performs a measurement whose outcomes are vectors $m = (x^{(1)}, \dots, x^{(|\mathcal{E}|)}) \in \mathcal{X}^{|\mathcal{E}|}$. And depending on the encoding $e \in \mathcal{E}$ that Bob learns (given to them by Alice) after measuring, Bob will output the guess $x^{(e)}$.

**Lemma 2.1** ([GW10]). *Let $|\mathcal{X}|$ be the number of possible strings, and suppose that the joint distribution over strings and encodings satisfies $P_{x,e} = P_e/|\mathcal{X}|$, where the distribution $\{P_e\}_e$ is arbitrary. Then*
$$P_{\text{guess}}^{P}[\mathcal{X}|E, P] \le \frac{1}{|\mathcal{X}|} \text{Tr}\left[\left(\sum_{m \in \mathcal{X}^{|\mathcal{E}|}} \rho_m^\alpha \right)^{1/\alpha}\right]$$
*for all $\alpha > 1$, where $\mathcal{E} = \{\rho_{x^{(e)}, e}\}_{x \in \mathcal{X}, e \in \mathcal{E}}$ is the ensemble of all possible states of messages and encodings, $\mathcal{P} = \{P_{x, e}\}_{x \in \mathcal{X}, e \in \mathcal{E}}$ its associated probability distribution and $\rho_m = \sum_{e=1}^{|\mathcal{E}|} P_e \rho_{x^{(e)}, e}$, the state that corresponds to some outcome vector $m$.*

### 2.3 Oblivious Transfer

Oblivious Transfer is a protocol between two parties, a **Sender** and a **Receiver**, and can be formulated in different but equivalent functionalities [Cré88]. The most common and perhaps most useful formulation is the **1-out-of-2 OT** [EGL85], where two messages are sent by a Sender to a Receiver, and the Receiver is only able to recover one message of its choice with the Sender remaining oblivious to which message was received. This intuition is made precise in Definition 2.1 by bounding the distance (as given by the trace-norm $\|\mathbb{A}\|_1 = \text{Tr}\sqrt{A^\dagger A}$) of the ideal state containing no information useful for cheating, and the actual state produced from a cheating strategy.

**Definition 2.1** (1-out-of-2 Oblivious Transfer). *A 1-out-of-2 Oblivious Transfer protocol is a protocol between two parties, a Sender and a Receiver, where the Sender has inputs $x_0, x_1 \in \{0, 1\}$ and no output, and the Receiver has input $y \in \{0, 1\}$ and output $m$, such that the following properties hold:*

*   *($\epsilon$-Correctness) For an honest Sender and Receiver, $P[m = x_y | x_0, x_1, y] \ge 1 - \epsilon$.*
*   *($\epsilon$-Receiver-security): Let $\rho_{y, x_0, x_1; \tilde{S}}$ be the state at the end of the protocol with an honest Receiver and in the presence of a malicious Sender, $\tilde{S}$. Then, for all algorithms $\tilde{S}$, there exists $(x_0, x_1) \in \{0, 1\}^2$, such that $P[m = x_y] \ge 1 - \epsilon$ and*
$$\|\rho_{y, x_0, x_1; \tilde{S}} - \rho_y \otimes \rho_{x_0, x_1; \tilde{S}}\|_1 \le \epsilon.$$
*   *($\epsilon$-Sender-security) Let $\rho_{y, x_0, x_1; \tilde{R}}$ be the state at the end of the protocol with an honest sender and in the presence of a malicious Receiver, $\tilde{R}$. Then, for all algorithms $\tilde{R}$, exists $y \in \{0, 1\}$, such that*
$$\left\|\rho_{\bar{y}, x_y, C; \tilde{R}} - \frac{1}{2}\rho_{x_{\bar{y}}, y; \tilde{R}}\right\|_1 \le \epsilon.$$

*If these properties only hold when restricting the algorithms $\tilde{S}$ or $\tilde{R}$ to run in probabilistic polynomial time, then the protocol is said to be computationally-secure.*

11

Despite its simplicity, OT is a fundamental primitive in cryptography, and it was shown to be sufficient to construct MPC [Kil88]. However, no black-box construction of OT can exist given only OWFs in the classical world [IR89], meaning that PKC was compulsory. Nevertheless, by also accounting for quantum computation and communication and quantum-secure OWFs, OT can be achieved without any PKC requirement [BCKM21, GLSV21]. This means that introducing quantum computation and communication substantially relaxes the requirements to construct OT, as candidates for quantum-secure OWFs are simpler and more frequent.

### 2.4 One-Time Memory and One-Time Programs

**One-Time Memory** was introduced in [GKR08] as a secure hardware device inspired by the cryptographic protocol of OT. It executes exactly the same functionality as the 1-out-of-2 (chosen-input) OT (as presented in Section 2.3), defined in [GKR08, GIS+10]. However, since it consists of a physical tamper-proof memory with restricted read and write access, it effectively enforces a one-shot OT execution, where the Sender transmits a single message to the Receiver by physically transferring the memory object (the OTM). The reliance of OTM on secure hardware presents a significant limitation, as designing tamper-proof hardware is notoriously difficult. Consequently, this requirement has since been relaxed, with cryptographic alternatives introduced in the literature, albeit under certain trusted models [DDKZ14, Liu23].

Hence, we choose to formulate the definition of OTM in terms of the standard OT definition, as it will integrate with our other contributions in this paper (non-interactive OT).

**Definition 2.2** (One-Time Memory). *Let $\mathcal{M}$ be a 1-out-of-2 OT protocol as in Definition 2.1. If $\mathcal{M}$ is one-shot, i.e., if the protocol executes with a single message from the Sender to the Receiver, then $\mathcal{M}$ is a One-Time Memory.*

An important extension of the OTM is the **parallel-OTM**, which involves the parallel and non-adaptive execution of multiple OTMs. A parallel-OTM consists of multiple independent OTMs whose inputs must be chosen concurrently and non-adaptively. That is, the input to one OTM cannot be chosen based on the outcome of any other, as it would in a sequential execution.

In [GKR08], the concept of **One-Time Program** was also introduced, and subsequently further studied in [GIS+10]. A OTP is a program that models a black-box, which computes some function on a given input a single time, preventing any further interaction with the function. The OTM as a concept is particularly interesting as the means to design OTPs.

**Theorem 2.2** ([GKR08, GIS+10] (Informal)). *Assuming the existence of one-way functions, there is a polynomial-time compiler that takes any polynomial-time program and a parallel-OTM, and returns an OTP with the same functionality.*

The original compiler from OTM to OTP of [GKR08] is heavily based on the technique of Yao's garbled circuits [Yao86] (analogous to using OT for 2PC). However, Yao's garbled circuits only provide semi-honest security, and in order to guarantee security of the OTP against malicious adversaries in a non-interactive manner, [GKR08] presents a solution where the OTMs output is XORed to mask the output of the garbed circuit.

### 2.5 Restricted Quantum-Storage Models

Again, it is impossible to achieve unconditional security of OT and BC without any imposed assumption. Therefore, to avoid supporting the security of a protocol on conjectures on computationally-hard problems (e.g., OWFs or PKC), restrictions to the computation model based on physical

12

phenomenons (motivated by current technology limitations) were introduced for BC and OT. Two main restrictions to the quantum-storage capability of the parties have been introduced. First, restrictions to the *storage-space*, either in the dimension of the quantum states that a party can coherently measure [Sal98], or on the total size of the storage available [DFSS05]. Second, restrictions to the *storage-time* (duration) that a quantum state can be stored before being subjected to quantum decoherence [WST08].

#### 2.5.1 Bounded-Quantum-Storage Model

The **Bounded-Quantum-Storage Model** [DFSS05] establishes that there is a point during the protocol, called the **memory bound**, when all but $M$ qubits of the (otherwise unbounded) memory register of the parties are measured. Besides this transient limitation during the execution of the protocol, no restrictions are applied to the classical memory and computing power, which are still considered unbounded.

The functionality of the BQSM is described in Definition 2.3. In this work, it will be assumed that the time instant $t$ and memory size $M$ of Definition 2.3 are set in advance, when designing a protocol in the BQSM.

**Definition 2.3** (Bounded-Quantum-Storage Model). *The Bounded-Quantum-Storage Model consists of two identically modeled computation phases $\mathcal{P}_{\text{pre}}, \mathcal{P}_{\text{post}}$, discontinued by a partial measurement of the memory register of the parties $\mathcal{M}_{t, M}$, where (in chronological order):*

1.  *$\mathcal{P}_{\text{pre}}$: the state of a party may have an arbitrary number of qubits ($N$), and arbitrary computations are allowed over this system.*
2.  *$\mathcal{M}_{t, M}$: at a certain point in time $t$, the memory bound applies, i.e., all but $M \le N$ qubits are measured.*
3.  *$\mathcal{P}_{\text{post}}$: the party is again unbounded in quantum memory and computing power.*

#### 2.5.2 Noisy-Quantum-Storage Model

Generalizing the BQSM to a more realistic noisy-memory model is left as an open question in [DFSS05]. The **Noisy-Quantum-Storage Model** [WST08, KWW12] addresses this weaker assumption by considering the quantum memory of the parties performing the protocol to be imperfect due to the presence of noise. This model represents a more realistic setting given the current available technology, and does not require an arbitrary estimation of the total memory available to an all-powerful adversary. In opposition, any qubit that is stored experiences noise that leads to quantum decoherence.

The functionality of the NQSM is given in Definition 2.4. Again, in this work, it will be assumed that the family $\{\mathcal{F}_t\}$ of Definition 2.4 is known in advance when designing a protocol in the NQSM. Note that the BQSM is a particular case of the NQSM, where $\mathcal{F}_t = \mathbb{I}$ for all $t$ but the dimension of $\mathcal{H}_{in}$ is bounded.

**Definition 2.4** (Noisy-Quantum-Storage Model). *Let $\rho \in \mathcal{L}(\mathcal{H}_{in})$ be a quantum state stored in a quantum memory. The Noisy-Quantum-Storage Model prescribes a family of completely positive trace-preserving functions $\{\mathcal{F}_t\}_{t \ge 0}$, such that the content of the memory after a certain time $t$ is a state $\mathcal{F}_t(\rho)$, where $\mathcal{F}_t: \mathcal{L}(\mathcal{H}_{in}) \to \mathcal{L}(\mathcal{H}_{out})$, and*
$$\mathcal{F}_0 = \mathbb{I} \quad \text{and} \quad \mathcal{F}_{t_1 + t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2},$$
*i.e., noise in storage only increases with time.*

13

To enable an analysis of the relation between the storage size and the probability of successfully decoding stored states, it is often considered that the memory is composed by $N$ different cells and that noise affects these cells separately, i.e., $\mathcal{F} = \mathcal{N}^{\otimes N}$. Then for a large enough $N$, the probability that a party can decode some rate $R$ (above the classical capacity of the channel, $\mathcal{C}_N$) of its quantum memory decays exponentially with $N$ [KWW12]:
$$P_{\text{succ}}^{\mathcal{N}^{\otimes N}}[NR] \le 2^{-N\gamma_{\mathcal{N}}(R)}, \quad \gamma_{\mathcal{N}}(R) > 0 \text{ for all } R > \mathcal{C}_{\mathcal{N}}. \quad (2.4)$$
An example of noisy channel is the $d$-dimension depolarizing channel $\mathcal{N}_r: \mathcal{L}(\mathcal{H}) \to \mathcal{L}(\mathcal{H})$, for $d \ge 2, 0 \le r \le 1$,
$$\mathcal{N}_r(\rho) \to r \rho + (1-r)\frac{1}{d}\mathbb{I}, \quad (2.5)$$
which gradually converts a stored state $\rho$ to a maximally mixed state with probability $1-r$.

Note that the assumption that $\mathcal{F} = \mathcal{N}^{\otimes N}$ considers storing each qubit independently. This means that even if two qubits are entangled, the entanglement is not affected by more than the independent noise that each qubit undergoes by itself, which still leads to the degradation of the entanglement.

## 3 Non-Interactive OT

In this section, our first contribution of an unconditionally-secure two-message non-interactive 1-out-of-2 OT is presented. A construction is given in the NQSM and its security is proved in this model. Also, an alternative construction for a non-interactive 1-out-of-2 OT is presented, which removes the time-delay constraint of the previous NQSM construction in exchange for adopting the BQSM.

Regarding the NQSM (Definition 2.4), we will make a simplification by parameterizing our protocol by a time bound $\tau$ that enforces total decoherence of the memories of the parties. This model may, for instance, be interpreted as a depolarizing channel (Equation (2.5)) that after $\tau$ time steps erases all information about state $\rho$, i.e., $\mathcal{N}_\tau(\rho) = \mathbb{I}/d$. One could instead study different noise models and the dependence of the security of the protocol with the noise level at any point in time $t < \tau$. We explicitly choose to parameterize our protocol directly by the time to total decoherence $\tau$, as it represents the worst case scenario for an adversary. Also, this closely relates the BQSM as the limit of the NQSM.

### 3.1 Preliminaries

We first introduce some basic definitions and notation for key elements of the protocol. Let $[N] := \{n | n \in \mathbb{N} \text{ and } n < N\}$, and $x = x_0 x_1 \in \mathcal{X} = \{00, 01, 10, 11\}$ be the **message**.

**Definition 3.1** ($N$-qubit register). *We refer to a set of $N$ qubits, $\mathcal{R} = \{q_1, \dots, q_i, \dots, q_N\}$, indexed by $i \in [N]$, as an $N$-qubit register. An element, $q_i$, of the register is interpreted as the physical system at the $i$-th site, rather than the operational description of its quantum system.*

**Definition 3.2** (Index-encoding set). *Let the index-encoding set be a set of tuples $\mathcal{E} := \{(k, l) \mid k, l \in [N] \text{ and } k < l\}$, where $|\mathcal{E}| = \binom{N}{2} = N(N-1)/2$. Then, the set $\mathcal{E}$ is the set of ordered tuples $(k, l)$ where $k < l$, such that an element of the index-encoding set selects a pair of distinct sites ($q_k, q_l$) of the register.*

14

**Definition 3.3** (Sub-Register). *Let $\mathcal{R} \setminus \{q_{i_1}, \dots, q_{i_n}\}$ be an $(N-n)$-qubit sub-register of $\mathcal{R}$, indexed by $[N] \setminus \{i_1, \dots, i_n\}$. We write $\rho_{[N] \setminus \{i_1, \dots, i_n\}} := \rho_1 \otimes \cdots \otimes \rho_{i_1-1} \otimes \rho_{i_1+1} \otimes \cdots \otimes \rho_{i_n-1} \otimes \rho_{i_n+1} \otimes \cdots \otimes \rho_N$, to denote that the quantum state in each site $j$ of the sub-register is equal to $\rho$, i.e., $\rho_j = \rho$ for all $j$.*

**Definition 3.4** (Message encoding vector). *Given the set of all possible assignments from the index-encodings to the messages $\mathcal{X}^{|\mathcal{E}|} = \{m_1, \dots, m_{4^{|\mathcal{E}|}}\}$, let the **message encoding vector** be the specific assignment $\mathbf{m}_i$, which is explicitly denoted as $\mathbf{m}_i = \left(\langle x_0^{(k)} x_1^{(l)} \rangle_i \mid (k, l) \in \mathcal{E}\right)$.*

Note that there are $4^{|\mathcal{E}|}$ possible message encoding vectors, and each vector $\mathbf{m}_i$ has $|\mathcal{E}|$ entries. One may assume that the index $i$ gives the placement of the vector in lexicographical order, for example, $\mathbf{m}_1 = \left(\langle 0^{(k)} 0^{(l)} \rangle_1 \mid (k, l) \in \mathcal{E}\right)$ and $\mathbf{m}_{4^{|\mathcal{E}|}} = \left(\langle 1^{(k)} 1^{(l)} \rangle_1 \mid (k, l) \in \mathcal{E}\right)$. While the previous definition assumes a level of generality where the message content could be correlated with the index-encoding, this is not something we consider in the proposed protocol. We assume that the index-encodings $(k, l)$ are randomly sampled and independent of the chosen message $x_0, x_1$. Nevertheless, we adopt this level of generality as it will be required when proving security, namely, when using the discrimination framework with post-measurement classical information of [GW10] (see Section 2.2).

**Definition 3.5** (Message encoding state). *Let $\mathbf{m}_i = \left(\langle x_0^{(k)} x_1^{(l)} \rangle_i \mid (k, l) \in \mathcal{E}\right)$ be a message encoding vector as in Definition 3.4, then, its associate **message encoding state** is given by*
$$\rho_{\mathbf{m}_i} = \rho_{\left(\langle x_0^{(k)} x_1^{(l)} \rangle_i\right)} = \frac{1}{|\mathcal{E}| \cdot 2^{N-2}} \sum_{k<l} \left|B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right\rangle\left\langle B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right|_{k, l} \otimes \mathbb{I}_{[N] \setminus \{k, l\}},$$
*which describes the density matrix for the $N$-qubit register $\mathcal{R}$ in full generality, allowing the message to depend on the uniformly sample index-encodings $\mathcal{E}$.*

It will also be useful to consider the unnormalized version of the state $\sigma_{\mathbf{m}_i} = \rho_{\mathbf{m}_i} \cdot |\mathcal{E}| \cdot 2^{N-2}$.

**Lemma 3.1.** *Let $A, B \in \text{M}_n$ be Hermitian positive semi-definite matrices. Then, $\lambda_{\max}(A+B) \le \lambda_{\max}(A) + \lambda_{\max}(B)$.*

*Proof.* The spectral norm of a Hermitian matrix $M$, denoted $\|M\|_2$, is equal to the largest eigenvalue in magnitude, i.e., $\|M\|_2 = \max_i \{|\lambda_i|\}$, where $\lambda_i$ are the eigenvalues of $M$. Since $A$ and $B$ are also positive semi-definite, all their eigenvalues are non-negative. Therefore, the spectral norm of $A$ and $B$ becomes $\|A\|_2 = \lambda_{\max}(A)$, $\|B\|_2 = \lambda_{\max}(B)$, respectively.

The triangle inequality for the spectral norm states that $\|A+B\|_2 \le \|A\|_2 + \|B\|_2$. Since $A+B$ is also Hermitian and positive semi-definite we have $\|A+B\|_2 = \lambda_{\max}(A+B)$ and by direct substitution we get $\lambda_{\max}(A+B) \le \lambda_{\max}(A) + \lambda_{\max}(B)$.

Finally, in Lemma 3.2, we introduce an important lemma giving a maximal eigenvalue upper bound, which will be essential for the security proof.

**Lemma 3.2.** *Let $\sigma_{\mathbf{m}_i} = \sum_{k<l} \left|B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right\rangle\left\langle B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right|_{k, l} \otimes \mathbb{I}_{[N] \setminus \{k, l\}}$ be a message encoding with unnormalized associated message encoding state*
$$\sigma_{\mathbf{m}_i} = \sum_{k<l} \left|B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right\rangle\left\langle B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right|_{k, l} \otimes \mathbb{I}_{[N] \setminus \{k, l\}}$$
*Then, the largest eigenvalue, $\lambda_{\max}(\sigma_{\mathbf{m}_i})$, is upper bounded by*
$$\lambda_{\max}(\sigma_{\mathbf{m}_i}) \le \frac{N^2}{4} + \frac{N}{4} - \frac{1}{2}$$

15

*Proof.* Let us start by defining a shorthand notation, where we also make explicit the terms $\langle x_0^{(k)} x_1^{(l)} \rangle_i$ of the message encoding in the state and the size of the register $N$,
$$\sigma_{N}^{\langle x_0 x_1 \rangle_i} = \sum_{k<l} \left|B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right\rangle\left\langle B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right|_{k, l} \otimes \mathbb{I}_{[N] \setminus \{k, l\}}, \quad (3.1)$$
with
$$\left|B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right\rangle\left\langle B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right|_{k, l} \otimes \mathbb{I}_{[N] \setminus \{k, l\}} = B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\otimes B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}^\dagger \otimes \mathbb{I}_{[N] \setminus \{k, l\}} \quad (3.2)$$
For an $N$-qubit register, the previous state $\sigma_{N}^{\langle x_0 x_1 \rangle_i}$ can be interpreted as a sum over the $|\mathcal{E}| = \binom{N}{2}$ edges of the complete graph $K_N$, where each vertex represents a qubit and each edge connects qubits $k$ and $l$, and is given by state $\left|B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right\rangle\left\langle B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right|_{k, l}$ for a specific message encoding $\mathbf{m}_i$. Noticing this, we can rewrite Equation (3.1) by separating the summation domain over the edges into two disjoint subsets as follows
$$\sigma_{N}^{\langle x_0 x_1 \rangle_i} = \sigma_{N-1}^{\langle x_0 x_1 \rangle_i} + \sigma_{\text{star}(N)}^{\langle x_0 x_1 \rangle_i}, \quad (3.3)$$
where
$$\sigma_{\text{star}(N)}^{\langle x_0 x_1 \rangle_i} = \sum_{j=1}^{N-1} \left|B_{\langle x_0^{(j)} x_1^{(N)} \rangle_i}\right\rangle\left\langle B_{\langle x_0^{(j)} x_1^{(N)} \rangle_i}\right|_{j, N} \quad (3.4)$$
is the unnormalized mixture of all Bell pairs involving the $N$th qubit. Using the graph interpretation described above, such state can be seen as a star graph with its center at the $N$th vertex, the latter being connected to all other $N-1$ vertices. This relation can be applied recursively, allowing the expression of the $\sigma_N^{\langle x_0 x_1 \rangle_i}$ state as a sum of $\sigma_{\text{star}(n)}^{\langle x_0 x_1 \rangle_i}$ states, for $n \in \{2, \dots, N\}$.
Since the states in Equation (3.3) correspond to Hermitian positive semi-definite matrices, applying Lemma 3.1 we get the following upper bound for the maximum eigenvalue,
$$\lambda_{\max}\left(\sigma_{N}^{\langle x_0 x_1 \rangle_i}\right) \le \lambda_{\max}\left(\sigma_{N-1}^{\langle x_0 x_1 \rangle_i}\right) + \lambda_{\max}\left(\sigma_{\text{star}(N)}^{\langle x_0 x_1 \rangle_i}\right). \quad (3.5)$$
Next, notice that we can apply local unitary transformations at each $j$ qubit to transform it into any Bell pair of our choosing, and since the spectrum is invariant under unitary transformations we have that, for all $i$,
$$\lambda_{\max}\left(\sigma_{\text{star}(N)}^{\langle x_0 x_1 \rangle_i}\right) = \lambda_{\max}(\sigma_{\text{star}(N)}). \quad (3.6)$$
Without loss of generality, let us consider $|B_{11}\rangle\langle B_{11}|$, obtained by applying $(Z^{x_0} X^{x_1})^{\otimes 1} \otimes \mathbb{I}_{\mathbb{C}^2}$ to $B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\otimes B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}^\dagger$. Thus, with the foresight that our attention will lie only in the spectrum of the operators, we can write
$$\sigma_{\text{star}(N)} = \sum_{j=1}^{N-1} |B_{11}\rangle\langle B_{11}|_{j, c} \otimes \mathbb{I}_{[N] \setminus \{j, c\}}.$$
Rewriting the Bell state in terms of the Pauli matrices (Equation (2.2)) we have
$$\sigma_{\text{star}(N)} = \frac{1}{4} \sum_{j=1}^{N-1} \left(\mathbb{I}_j \otimes \mathbb{I}_c - X_j \otimes X_c - Z_j \otimes Z_c - Y_j \otimes Y_c\right) \otimes \mathbb{I}_{[N] \setminus \{j, c\}}$$
$$= \frac{N-1}{4} \mathbb{I}_{[N]} - \frac{1}{4} \sum_{j=1}^{N-1} (X_j \otimes X_c + Z_j \otimes Z_c + Y_j \otimes Y_c). \quad (3.8)$$
where $X_i = \mathbb{I}_1 \otimes \cdots \otimes X_i \otimes \cdots \otimes \mathbb{I}_N$ such that $X_j \cdot X_c = X_j \otimes X_c \otimes \mathbb{I}_{[N] \setminus \{j, c\}}$, and similarly for $Y_i$ and $Z_i$.

Finally, let us rewrite the previous expression as
$$\sigma_{\text{star}(N)} = \frac{N-1}{4} \mathbb{I}_{[N]} - \mathcal{H}_{\text{star}}^{(N)}, \quad (3.9)$$
where
$$\mathcal{H}_{\text{star}}^{(N)} = \frac{1}{4} \sum_{j=1}^{N-1} (X_j \cdot X_c + Z_j \cdot Z_c + Y_j \cdot Y_c) \quad (3.10)$$
is known as the **Heisenberg-star spin model** in many-body physics [RVK95]. Focusing on the largest eigenvalue for $\sigma_{\text{star}(N)}$, we have the following relation,
$$\lambda_{\max}(\sigma_{\text{star}(N)}) = \frac{N-1}{4} + \lambda_{\max}\left(-\mathcal{H}_{\text{star}}^{(N)}\right)$$
$$= \frac{N-1}{4} - \lambda_{\min}\left(\mathcal{H}_{\text{star}}^{(N)}\right), \quad (3.11)$$
where we have rewritten the equation in terms of the **minimum eigenvalue** for $\mathcal{H}_{\text{star}}^{(N)}$, which corresponds to the ground-state energy of the Heisenberg-star spin system, calculated analytically in [RVK95] to be
$$\lambda_{\min}\left(\mathcal{H}_{\text{star}}^{(N)}\right) = - \frac{1+N}{4}. \quad (3.12)$$
From Equations (3.11) and (3.12), we obtain that
$$\lambda_{\max}(\sigma_{\text{star}(N)}) = \frac{N}{2} \quad (3.13)$$
Finally, taking Equation (3.5) and using it recursively (until there is only one Bell pair left), we achieve the desired result
$$\lambda_{\max}\left(\sigma_{N}^{\langle x_0 x_1 \rangle_i}\right) \le \sum_{n=2}^N \lambda_{\max}(\sigma_{\text{star}(n)})$$
$$\le \sum_{n=2}^N \frac{n}{2}$$
$$\le \frac{N^2}{4} + \frac{N}{4} - \frac{1}{2} \quad (3.14)$$

### 3.2 Non-Interactive OT Protocol

Intuitively, to implement the OT protocol, the **Sender** will hide an EPR-pair encoding its two bits $x_0, x_1$, masked among many “decoy” qubits of the $N$-qubit register $\mathcal{R} = \{q_1, \dots, q_N\}$, such that the **Receiver** cannot know which qubits are encoding the information without the Sender revealing them. A detailed operational description of the protocol is given in Figure 1. Furthermore, an informational perspective from the view of the Sender and Receiver is introduced below.

*   **Step 0:** The **Sender** chooses $x_0 \in \{0, 1\}, x_1 \in \{0, 1\}$ and sets up an $N$-sized qubit register $\mathcal{R} = \{q_1, \dots, q_N\}$, where $N$ depends on the security parameter $\sigma$, initialized in the state $\bigotimes_{i=1}^N |0\rangle\langle 0|_i$. The **Receiver** chooses $y \in \{0, 1\}$.

17

**Figure 1: Schematic representation of the proposed two-message non-interactive OT protocol parameterized by $N(\sigma), \tau$. The “Wait $\tau$” procedure by the Sender may be disregarded in exchange for a larger $N$ (Section 3.2.2).**

| Sender | Receiver |
| :--- | :--- |
| **Choose** $x_0 \in \{0, 1\}, x_1 \in \{0, 1\}$ | **Choose** $y \in \{0, 1\}$ |
| **Given** $\mathcal{R} = \{q_1, q_2, \dots, q_N\}$ initialized at $\bigotimes_{i=1}^N |0\rangle\langle 0|_i$ and the index-encoding $\mathcal{E}$ | |
| **(Step 1)** Randomly sample $(k, l)$ from $\mathcal{E}$: $(k, l) \leftarrow\$ \mathcal{E}$ | |
| **(Step 2)** Generate EPR pair between $q_k$ and $q_l$ and encode $x_0, x_1$ in the state as follows: | |
| (a) Apply Hadamard gate $H$ to $q_k$ | |
| Apply CNOT to $q_k$ (control) and $q_l$ (target) | |
| Apply $Z^{x_0}X^{x_1}$ to $q_k$ | |
| (b) Generate maximally mixed states: | |
| **for** $i$ from $1$ to $N$ **do** | |
| **if** $i \ne k$ and $i \ne l$ **then** | |
| $r_i \leftarrow\$ \{0, 1\}$ | |
| apply $X^{r_i}$ to $q_i$ | |
| **end if** | |
| **end for** | |
| **(Step 3)** $\xrightarrow{\quad \text{Send register } \mathcal{R} \quad}$ | Measure all $q_i$ and store $m^y_i$ as follows: |
| | **if** $y=0$ measure in computational-basis |
| | **if** $y=1$ measure in diagonal-basis |
| **(Step 4)** Wait $\tau$ | |
| **(Step 5)** $\xrightarrow{\quad \text{Send indices } k, l \quad}$ | $x_y \leftarrow m^y_k \oplus m^y_l$ |

*   **Step 1:** The **Sender** uniformly samples indices $k, l$ from the index encoding set $\mathcal{E}$ (with $k < l$, without loss of generality), selecting qubits $\{q_k, q_l\} \subset \mathcal{R}$.
*   **Step 2:**
    *   (a) The **Sender** maximally entangles qubits $\{q_k, q_l\}$, $|B_{00}\rangle_{k, l} = (\text{CNOT})_{k, l} \cdot (H_k \otimes \mathbb{I}_l) |00\rangle_{k, l}$. Furthermore, it encodes $x_0, x_1$ in the entangled pair of qubits $q_k, q_l$ accordingly, $|B_{x_0 x_1}\rangle_{k, l} = ((Z^{x_0} X^{x_1})_k \otimes \mathbb{I}_l) |B_{00}\rangle_{k, l}$, leading to the state
$$\left|B_{x_0 x_1}\right\rangle\left\langle B_{x_0 x_1}\right|_{k, l} \otimes |0\rangle\langle 0|_{[N] \setminus \{k, l\}}.$$
    *   (b) The **Sender** generates maximally-mixed states for the remainder of the register, by implementing $X^{r_i}$ for random bit $r_i$ to $|0\rangle\langle 0|_i$ for $i \in [N] \setminus \{k, l\}$
$$\frac{1}{2^{N-2}}\left|B_{x_0 x_1}\right\rangle\left\langle B_{x_0 x_1}\right|_{k, l} \otimes \mathbb{I}_{[N] \setminus \{k, l\}}.$$
*   **Step 3:** The **Sender** sends the entire register $\mathcal{R}$ to the **Receiver**. For each of the four possible $x_0, x_1$ choices there is a corresponding state
$$\rho_{(x_0, x_1)} = \frac{1}{|\mathcal{E}| \cdot 2^{N-2}} \sum_{k<l} \left|B_{x_0 x_1}\right\rangle\left\langle B_{x_0 x_1}\right|_{k, l} \otimes \mathbb{I}_{[N] \setminus \{k, l\}}.$$
Notice that the previous states correspond to the message encoding state (Definition 3.5) for each of the four constant message-encoding vectors. Indeed, the **Sender** will choose the message independently of the particular index-encoded sampled.

18

*   **Step 4:** The **Sender** waits for a pre-determined time $\tau$, specified by the NQSM, for the memory to completely decohere. The **Receiver** measures each individual qubit $q_i$, either in computational basis if $y=0$ or in the diagonal basis if $y=1$, and stores all classical measurement results $m^y_i \in \{0, 1\}$.
*   **Step 5:** Finally, the **Sender** sends the encoding indices $k, l$, to the **Receiver**. The **Receiver** computes the parity of the stored measurement outputs for $\{q_k, q_l\}$, that is, $m^y_k \oplus m^y_l = x_y$.

In this protocol, the honest Receiver will measure individually each qubit in the register, for which no quantum memory is needed. As such, a necessary aspect for the security is that the Receiver be forced to measure the qubits separately, otherwise, a straightforward attack is to perform Superdense Coding (SDC) [BW92] and recover both the inputs of the Sender. One way to mitigate this, as we did, is by imposing the constraints offered by the NQSM, wherein the Sender will need to wait a fixed amount of time ($\tau$) in order for the memory of any malicious Receiver to decohere. Therefore, either the Receiver proceeds honestly and according to the protocol prescription measures every qubit separately, or it acts maliciously and tries to implement a general measurement over the register before losing the encoded message to decoherence. The size of the register $N$ must be set to ensure unconditional security, which defines the success of a malicious actor in a statistical experiment of running the protocol. We will show that the success probability of any possible attack (i.e., the statistical distance between distributions) goes to zero linearly with $N$. Thus, we will set $N$ to be a large enough constant (independent of the adversarial power), such that the statistical distance between distributions is negligible.

Regarding the waiting time $\tau$, we remark that one instance of waiting $\tau$ can be “reused” for many parallel executions of the protocol. And since OT is often used as a building block for other primitives, and these often require many OT executions, this delay can be amortized over all the parallel processes. Nevertheless, in some scenarios it could be perceived as undesirable the need to have an explicit time delay embedded in the design of the protocol, specially when such a delay is substantial when comparing with the generating and transmission of the required messages (qubits and indices) that can be as fast as the speed of light. As an alternative, one can remove the delay without affecting the unconditional security, by changing the NQSM with the BQSM. We analyze this approach in more detail in Section 3.2.2, where the Sender does not wait any time but the number of qubits that it sends ($N$) before revealing the indices $k, l$ is chosen to be large enough, such that the Receiver cannot store all of them (from the BQSM assumption). Thus, it must guess which subset of $M$ qubits to store.

#### 3.2.1 Correctness and Security

To establish that the protocol of Figure 1 implements a secure 1-out-of-2 OT, it must be proved, according to the requirements of Definition 2.1, that: the honest execution of the protocol is correct; the Sender does not acquire any information regarding the input of the Receiver; and, the Receiver remains oblivious to the input of the Sender that was not retrieved.

To accomplish such requirements, start by noticing that all communication in the protocol flows from the Sender to the Receiver, i.e., it is a non-interactive protocol. So first, the Sender must not be able to keep any (arbitrary-dimension) entangled system with the system it sends to the Receiver that would allow the Sender to somehow gather any information about the input of the Receiver later. And second, the Receiver must not be able to design any arbitrary-dimension POVM over the $N$-dimensional state it received from the Sender that would allow the Receiver to extract more information than one of the messages of the Sender. These two properties will be formally proved below, but intuitively, they follow from the inability to extract information from future events for

19

the first case, and from combining the NQSM (by introducing long-enough delay that imposes decoherence of memories) with the hiding of the qubits encoding for the second case.

The OT protocol is parameterized by the statistical security parameter $\sigma$, and by the time $\tau$ to quantum decoherence of memories (up to an exponentially low probability $2^{-\sigma}$) predefined by the NQSM where the protocol is resolved.

**Remark.** *We remark that the number of transmitted qubits, $N$, must be set as $N = 2^\sigma$, meaning that the communication depends on the statistical security parameter, for the statistical distance to be at least $2^{-\sigma}$. However, the circuit to prepare each of the states is constant-size and no memory is required. Moreover, as this is a statistical security parameter that enforces the indistinguishability between two distributions in a single experiment, it is fixed and does not scale with the power of an adversary (that may even be all-powerful). This contrasts with computational security, where the advantage of the adversary must go to zero faster than any polynomial, because an adversary is allowed polynomial-many tries to distinguish two distributions. Indeed, our parameter $N$ is a constant in the protocol parametrization for any desired statistical distance, being independent of any computational security parameter, i.e., does not grow with the adversarial power.*

**Theorem 3.3.** *The protocol from Figure 1 implements a 1-out-of-2 Oblivious Transfer protocol secure against computationally unbounded adversaries (unconditional security parameterized by $\sigma$) in the Noisy-Quantum-Storage Model with time to total decoherence $\tau$.*

*Proof.* The protocol from Figure 1 is a two-party protocol, where the **Sender** has two inputs ($x_0, x_1$) and the **Receiver** has one input $y$ and outputs $x_y$, which performs precisely the functionality of OT (Definition 2.1). We will now show the three necessary properties of correctness, Receiver-security and Sender-security.

**Correctness:** The correctness of the honest strategy for the protocol can be immediately established since it will correspond to a “stochastic dense coding" [PPCT22] applied to qubits $\{q_k, q_l\}$. Therein, both bits are encoded into the Bell state, namely, $x_0$ is encoded in the phase, and $x_1$ in the parity of the Bell state (just as in SDC), but only one bit may be deterministically extracted when using separable measurements. Accordingly, the Receiver can either extract the first or second bit by measuring, respectively, the phase or the parity observables. That is, measuring in the computational or the diagonal basis individually for all qubits of the register $\mathcal{R}$, and deterministically extract the desired bit out of the Bell state shared between $\{q_k, q_l\}$ by computing the parity of the individual measurement outputs after receiving the indices. This shows the protocol to have **perfect correctness**, since an honest strategy will deterministically return $x_y$. We further remark that no quantum memory is required to correctly execute the protocol, and thus, no analysis of the NQSM is required.

**Receiver-security:** To prove that the protocol is secure for an honest Receiver, i.e., against a malicious Sender, it must be guaranteed that no matter what the Sender does, it cannot recover the input of the Receiver ($y$). In this case, it must be noted that the Receiver exclusively performs measurements on its part of the system, and does not explicitly communicate anything to the Sender, i.e, communication is one-way. Thus, any correlated event that the Sender can exhibit ($Z$) must be constrained by the *no-signalling from the future* [CDP10] (also called *no-backward-in-time signaling* [GSS+19]), i.e.,
$$P [Z|X = x_0 x_1, Y = y] = P[Z|X = x_0 x_1]. \quad (3.16)$$

20

This, in turn, implies that any correlation that the Sender holds ($Z$) in its state ($\rho_{y, x_0, x_1; \tilde{S}}$) is conditionally independent of the input of the Receiver ($y$), $\rho_{y, x_0, x_1; \tilde{S}} = \rho_y \otimes \rho_{x_0, x_1; \tilde{S}}$, as required by Definition 2.1. So,
$$\|\rho_{y, x_0, x_1; \tilde{S}} - \rho_y \otimes \rho_{x_0, x_1; \tilde{S}}\|_1 = 0. \quad (3.17)$$
Therefore, the **Sender** cannot obtain any information about the input of the **Receiver**, meaning that the protocol has perfect security, in this case.

**Sender-security:** To prove that the protocol is secure for an honest **Sender**, i.e., against a malicious **Receiver**, it must be unfeasible for the Receiver to recover more than one of the messages of the Sender. For this, the proof will require enforcing the Receiver to measure before receiving the encoding, and then using the formalism of post-measurement information (Section 2.2) to analyze the implications (or lack thereof) of sending the encoding.

Recall that for each message encoding vector $\mathbf{m}_i$ (Definition 3.4) there is a corresponding message encoding state $\rho_{\mathbf{m}_i}$ (Definition 3.5),
$$\rho_{\mathbf{m}_i} = \frac{1}{|\mathcal{E}| \cdot 2^{N-2}} \sum_{k<l} \left|B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right\rangle\left\langle B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right|_{k, l} \otimes \mathbb{I}_{[N] \setminus \{k, l\}} \quad (3.18)$$
Now, we consider the post-measurement information formalism introduced in Section 2.2, and from Lemma 2.1 we have that if $p_{x, e} = P_e/|\mathcal{X}|$, then
$$P_{\text{guess}}^{P}(x|\mathcal{R}) \le \frac{1}{|\mathcal{X}|} \text{Tr}\left[\left(\sum_{m \in \mathcal{X}^{|\mathcal{E}|}} \rho_m^\alpha \right)^{1/\alpha}\right] \quad (3.19)$$
for any $\alpha > 1$. Thus, applied to our scenario where $|\mathcal{X}| = 4$ and $p_{x, e} = \frac{1}{4|\mathcal{E}|}$, then
$$P_{\text{guess}}^{P}(x|\mathcal{R}) \le \mathcal{I}_\alpha(N) \quad (3.20)$$
for
$$\mathcal{I}_\alpha(N) := \frac{1}{4} \text{Tr}\left[\left(\sum_{\mathbf{m}_i \in \mathcal{X}^{|\mathcal{E}|}} \rho_{\mathbf{m}_i}^\alpha \right)^{1/\alpha}\right] \quad (3.21)$$
where
$$\sigma_{\mathbf{m}_i} = \sum_{k<l} \left|B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right\rangle\left\langle B_{\langle x_0^{(k)} x_1^{(l)} \rangle_i}\right|_{k, l} \otimes \mathbb{I}_{[N] \setminus \{k, l\}}. \quad (3.22)$$
Let $\text{Tr}\left[\left(\sum_{\mathbf{m}_i \in \mathcal{X}^{|\mathcal{E}|}} \sigma_{\mathbf{m}_i}^\alpha \right)^{1/\alpha}\right] = \text{Tr}\left[(\mathbf{A}_\alpha)^{1/\alpha}\right]$, where $\mathbf{A}_\alpha = \sum_{\mathbf{m}_i \in \mathcal{X}^{|\mathcal{E}|}} \sigma_{\mathbf{m}_i}^\alpha$. Since $\mathbf{A}_\alpha$ is Hermitian (sum of Hermitian matrices) it can be diagonalized, thus,
$$\text{Tr}\left[(\mathbf{A}_\alpha)^{1/\alpha}\right] = \sum_{i=1}^{2^N} \lambda_i(\mathbf{A}_\alpha)^{1/\alpha} = \sum_{i=1}^{2^N} [\lambda_i(\mathbf{A}_\alpha)]^{1/\alpha} \le 2^N [\lambda_{\max}(\mathbf{A}_\alpha)]^{1/\alpha}. \quad (3.23)$$

21

Then, the maximum eigenvalue of $\mathbf{A}_\alpha$ may be decomposed as
$$\lambda_{\max}(\mathbf{A}_\alpha) = \lambda_{\max}\left(\sum_{\mathbf{m}_i \in \mathcal{X}^{|\mathcal{E}|}} \sigma_{\mathbf{m}_i}^\alpha\right). \quad (3.24)$$
Using Lemma 3.1 we have
$$\lambda_{\max}(\mathbf{A}_\alpha) \le \sum_{\mathbf{m}_i \in \mathcal{X}^{|\mathcal{E}|}} \lambda_{\max}(\sigma_{\mathbf{m}_i}^\alpha) = \sum_{\mathbf{m}_i \in \mathcal{X}^{|\mathcal{E}|}} [\lambda_{\max}(\sigma_{\mathbf{m}_i})]^\alpha. \quad (3.25)$$
Now, let $\sigma_{\mathbf{m}_*^{\alpha}}$ be a state whose largest eigenvalue is the maximum over all $\sigma_{\mathbf{m}_i}$, that is, $\lambda_{\max}(\sigma_{\mathbf{m}_*^{\alpha}}) \ge \lambda_{\max}(\sigma_{\mathbf{m}_i})$ for any other state $\sigma_{\mathbf{m}_i}$. As such,
$$\lambda_{\max}(\mathbf{A}_\alpha) \le 4^{|\mathcal{E}|} [\lambda_{\max}(\sigma_{\mathbf{m}_*^{\alpha}})]^\alpha. \quad (3.26)$$
Considering again Equation (3.23), in turn, means that
$$\text{Tr}\left[(\mathbf{A}_\alpha)^{1/\alpha}\right] \le 2^N [\lambda_{\max}(\mathbf{A}_\alpha)]^{1/\alpha}$$
$$\le 2^N \left(4^{|\mathcal{E}|} [\lambda_{\max}(\sigma_{\mathbf{m}_*^{\alpha}})]^\alpha\right)^{1/\alpha}$$
$$\le 2^N 4^{|\mathcal{E}|/\alpha} \lambda_{\max}(\sigma_{\mathbf{m}_*^{\alpha}}). \quad (3.27)$$
Then, for Equation (3.21) we get
$$\mathcal{I}_\alpha(N) = \frac{1}{|\mathcal{E}| \cdot 2^{N-2}} \text{Tr}\left[(\mathbf{A}_\alpha)^{1/\alpha}\right]$$
$$\le \frac{1}{|\mathcal{E}| \cdot 2^{N-2}} 2^N 4^{|\mathcal{E}|/\alpha} \lambda_{\max}(\sigma_{\mathbf{m}_*^{\alpha}}). \quad (3.28)$$
For $\alpha \gg |\mathcal{E}|$, we have that $\mathcal{I}_{\alpha \gg |\mathcal{E}|}(N) \le (\lambda_{\max}(\sigma_{\mathbf{m}_*^{\alpha}})/|\mathcal{E}|) 4^{\approx 0}$, which with Equation (3.20) yields that
$$P_{\text{guess}}^{P}(x|\mathcal{R}) \le \frac{\lambda_{\max}(\sigma_{\mathbf{m}_*^{\alpha}})}{|\mathcal{E}|}. \quad (3.29)$$
Finally, Lemma 3.2 establishes that $\lambda_{\max}(\sigma_{\mathbf{m}_*^{\alpha}}) \le N^2/4 + N/4 - 1/2$, and, by direct substitution, we have
$$P_{\text{guess}}^{P}(x|\mathcal{R}) \le \frac{1}{2} + \frac{1}{N}. \quad (3.30)$$
Hence, setting $N = 2^\sigma$ makes the OT protocol implementation of Figure 1 both Sender-secure and Receiver-secure, which concludes the proof.

#### 3.2.2 Relinquishing the $\tau$ Constraint

The construction from Figure 1 requires that, at one point of the execution, the **Sender** waits for a time interval $\tau$, such that, given the NQSM, the **Receiver** must measure the qubits before receiving the indices $k, l$. This constraint might be questioned, as it introduces a substantial delay in the system, specially comparing with the generating and transmission of the required messages (qubits and indices) that can be as fast as the speed of light. If the trade-off between the waited time $\tau$ and the time required to generate and send qubits favors the latter, then this waiting can be removed without affecting the unconditional security, but relaxing the NQSM to the BQSM instead. Indeed,

22

by considering that the BQSM forces a limitation on the amount of qubits stored (maximum size of the memory), estimated given some specific limitation of the technology, $\tau$ can be set to zero. Still, a malicious **Receiver** would not be able to cheat and recover more than one of the inputs of the **Sender**, even by measuring its stored system after receiving the indices $k, l$ from the **Sender**.

Note that setting $\tau = 0$ means that the indices $(k, l)$ are sent immediately after the register $\mathcal{R}$. This effectively merges the two messages into an arbitrarily small time period, approaching what could be considered a one-shot protocol. However, we still consider this a two-message procedure, as the messages cannot happen simultaneously (i.e., cannot be permuted), and are inherently sequential with a fixed order (first qubits, then indices), as in the phases of Definition 2.3.

**Theorem 3.4.** *The protocol from Figure 1 implements a 1-out-of-2 Oblivious Transfer protocol secure against computationally unbounded adversaries (unconditional security parameterized by $\sigma$) in the Bounded-Quantum-Storage Model with time bound $t=\tau=0$ and memory bound $M$.*

*Proof.* As in Theorem 3.3, start by perceiving that the protocol from Figure 1 implements a 1-out-of-2 OT. Then, note that the Receiver-security (against a malicious Sender) does not rely on the BQSM, and so this does not alter this part of the security proof. Thus, all that requires proving is the Sender-security of the protocol, i.e., against a malicious **Receiver**.

In this modified setting, besides the general measurements described in the proof of Theorem 3.3, there is an added possibility that the **Receiver** performs joint measurements on the system, by storing some of its qubits until after knowing $k, l$. From the BQSM (Definition 2.3), let $M$ be a parameter representing the maximum size of the memory of a party in the transient phase $\mathcal{M}_{t, M}$. Note that, from the Shannon's source coding theorem [Sha48], no unitary can be applied that compresses the $N$ transmitted qubits into a smaller number, since these are independent and uniformly random prepared states. Then, let $Z$ be the event of sampling $M$ indices from $\{1, \dots, N\}$ without replacement (the qubits stored in memory by the **Receiver**), for a security parameter $\sigma$, set $N > M$ such that
$$P[k, l \in Z] = 2 \frac{M}{N} \frac{M-1}{N-1} \cdot 2^{-\sigma}. \quad (3.31)$$
Therefore, as long as the phase $\mathcal{M}_{t, M}$ of the BQSM happens to the memory of the **Receiver** between receiving register $\mathcal{R}$ and the indices $k, l$, the receiver can only get one of the inputs of the **Sender**, up to an exponentially low probability $2^{-\sigma}$, for a large enough $N$, assuring the security of the OT.

## 4 One-Time Memory and One-Time Programs

In this section, the two-message unconditionally-secure OT protocol from Section 3 is expanded upon to make it a one-shot OT, achieving a OTM. This is accomplished by relaxing the security of the OT protocol to rely on computational assumptions (namely, TLPs built from OWFs and SFs), thus enforcing restrictions on the computing capabilities of adversarial parties, and by still working in the NQSM. Still, even though a computational assumption is introduced, the protocol is everlasting secure, as the non-chosen message cannot be retrieved after the execution of the protocol. Then, by using the compiler from [GKR08], we achieve OTPs from this OTM, under the assumptions of existence of a OWF and a SF, and the NQSM.

We start by introducing the concept of TLP, a cryptographic primitive whose security relies on computational hardness assumptions (Section 4.1). Then, we leverage this primitive together with the previous construction of Section 3.2 to achieve the desired one-shot OT protocol, i.e., the OTM (Section 4.2). Lastly, as a corollary, we use the compiler from [GKR08] to get OTPs from the OTM construction (Section 4.3).

23

### 4.1 Time-Lock Puzzles

A TLP [RSW96] is a non-interactive cryptographic primitive that allows for a party to send a hidden message, such that this message can only be read after some time has elapsed. It is required that a puzzle can be efficiently generated, i.e., the time to generate the puzzle must be much less than the time to solve it; and that the secret can only be read after some pre-defined time, even for parallel algorithms. Definitions 4.1 and 4.2 formally state this idea. The minimal assumptions required to realize a TLP have been studied in [JMRR21].

TLPs have a wide variety of applications, but in this work they will be integrated in the NQSM to introduce a delay in the protocol, such that the quantum memory of a party will decohere before it is able to access the information hidden by the TLP.

**Definition 4.1** (Puzzle [BGJ+16]). *Let $\lambda \in \mathbb{N}$ be the security parameter. A puzzle is a pair of algorithms ($\text{Puzzle.Gen}, \text{Puzzle.Sol}$) with*

*   *$\mathcal{Z} \leftarrow \text{Puzzle.Gen}(\tau, s)$ takes as input a time parameter $\tau$ and a solution $s \in \{0, 1\}^\lambda$, and outputs a puzzle $\mathcal{Z}$. $\text{Puzzle.Gen}(\tau, s)$ takes $\text{poly}(\log \tau, \lambda)$ time.*
*   *$s \leftarrow \text{Puzzle.Sol}(\mathcal{Z})$ takes as input a puzzle $\mathcal{Z}$ and outputs a solution $s$. $\text{Puzzle.Sol}(\mathcal{Z})$ takes $\tau \cdot \text{poly}(\lambda)$ time.*

*Then, for all $\lambda$, time parameter $\tau$, solution $s \in \{0, 1\}^\lambda$, and puzzle $\mathcal{Z}$ in the support of $\text{Puzzle.Gen}(\tau, s)$, $\text{Puzzle.Sol}(\mathcal{Z})$ outputs $s$.*

**Definition 4.2** (Time-Lock Puzzle [BGJ+16]). *A puzzle ($\text{Puzzle.Gen}, \text{Puzzle.Sol}$) is a **time-lock puzzle with gap $\epsilon < 1$** if there exists a polynomial $t(\cdot)$, such that for every polynomial $\tau(\cdot) \ge t(\cdot)$ and adversary $\mathcal{A} = \{\mathcal{A}_\lambda\}_{\lambda \in \mathbb{N}}$ of depth smaller than $\tau \cdot t(\lambda)$, there exists a negligible function $\mu$, such that for all $\lambda \in \mathbb{N}$ and $s_0, s_1 \in \{0, 1\}^\lambda$:*
$$P\left[b \leftarrow \mathcal{A}_\lambda(\mathcal{Z}): \begin{matrix} b \leftarrow \{0, 1\} \\ \mathcal{Z} \leftarrow \text{Puzzle.Gen}(\tau(\lambda), s_b) \end{matrix}\right] \le \frac{1}{2} + \mu(\lambda).$$

In this work, minimal requirements for the TLPs are needed. In particular, it is enough to consider **weak Time-Lock Puzzles** [BGJ+16] that can be build directly from OWFs (assuming the existence of a non-parallelizing language\footnote{A non-parallelizing language is equivalent to a sequential function [JMRR21].}). This relaxed formulation of TLPs only requires that the puzzle can be generated in fast parallel time (circuit computing $\text{Puzzle.Gen}$ of size $\text{poly}(\tau, \lambda)$ has depth $\text{poly}(\log \tau, \lambda)$), while it still takes time $\tau$ to solve ($\text{Puzzle.Sol}$ takes time $\tau \cdot \text{poly}(\lambda)$).

**Lemma 4.1** ([BGJ+16, JMRR21]). *There exists a weak Time-Lock Puzzle, assuming the existence of a One-Way Function and a Sequential Function, which fulfills the security definition of Definition 4.2.*

In addition, for our purpose, since the time intervals that are considered in the NQSM are often short enough (e.g., $0.25$ms [VAVD+22]), the requirements on the puzzle generation can even be further relaxed, such that the time to generate the puzzle may be the same as the time to solve it. This enables very simple and diverse constructions, such as repeated hashing of a shared seed. Nevertheless, to be as general as possible and limit the setup assumptions to the NQSM, without imposing conditions on its parameters (time to quantum decoherence), weak TLPs are considered from here onwards.

24

### 4.2 One-Time Memory (or, one-shot Oblivious Transfer)

Here, a construction for a one-shot chosen-bit 1-out-of-2 OT is given, yielding the desired OTM. First, in Section 3, a two-message non-interactive unconditionally-secure 1-out-of-2 OT in the NQSM was described. Now, by using a OWF and a SF via a TLP in the protocol, relaxing the security requirements to hold on computationally-hard problems, a one-shot 1-out-of-2 OT, i.e., a OTM, is constructed.

**Figure 2: Schematic representation of the proposed one-shot OT protocol parameterized by $N(\sigma), \lambda, \tau$.**

| Sender | Receiver |
| :--- | :--- |
| **Choose** $x_0 \in \{0, 1\}, x_1 \in \{0, 1\}$ | **Choose** $y \in \{0, 1\}$ |
| **Given** $\mathcal{R} = \{q_1, q_2, \dots, q_N\}$ initialized at $\bigotimes_{i=1}^N |0\rangle\langle 0|_i$ and the index-encoding $\mathcal{E}$ | |
| **(Step 1)** Randomly sample $(k, l)$ from $\mathcal{E}$: $(k, l) \leftarrow\$ \mathcal{E}$ | |
| Randomly sample $r \leftarrow\$ \{0, \dots, 2^\lambda - 1\}$ | |
| **(Step 2)** Construct TLP for $(k, l)$ as $\mathcal{Z} \leftarrow \text{Puzzle.Gen}(\tau, s = (k, l, r))$ | |
| Generate EPR pair between $q_k$ and $q_l$ and encode $x_0, x_1$ in the state as follows: | |
| (a) Apply Hadamard gate $H$ to $q_k$ | |
| Apply CNOT to $q_k$ (control) and $q_l$ (target) | |
| Apply $Z^{x_0}X^{x_1}$ to $q_k$ | |
| (b) Generate maximally mixed states: | |
| **for** $i$ from $1$ to $N$ **do** | |
| **if** $i \ne k$ and $i \ne l$ **then** | |
| $r_i \leftarrow\$ \{0, 1\}$ | |
| apply $X^{r_i}$ to $q_i$ | |
| **end if** | |
| **end for** | |
| **(Step 3)** $\xrightarrow{\quad \text{Send register } \mathcal{R} \text{ and puzzle } \mathcal{Z} \quad}$ | Measure all $q_i$ and store $m^y_i$ as follows: |
| | **if** $y=0$ measure in computational-basis |
| | **if** $y=1$ measure in diagonal-basis |
| | Solve puzzle $\mathcal{Z}$ to get $(k, l)$ |
| | $(k, l, \cdot) \leftarrow \text{Puzzle.Sol}(\mathcal{Z})$ |
| | $x_y \leftarrow m^y_k \oplus m^y_l$ |

The operational description of the OTM is given in Figure 2, and executes analogously to the protocol from Section 3. Below, we detail the differences in the various steps when compared to the previous one. Step 0 and Step 2, which are not explicitly mentioned, are identical to Figure 1.

*   **Step 1:** The **Sender** uniformly samples indices $k, l$ from the index encoding set $\mathcal{E}$ (with $k < l$, without loss of generality), selecting qubits $\{q_k, q_l\} \subset \mathcal{R}$. The **Sender** hides the $k, l$, as the solution of the TLP ($\mathcal{Z}$), parametrized by $\tau$ whose lower bound is established by the NQSM.
*   **Step 3:** The **Sender** sends the entire register $\mathcal{R}$ and the TLP ($\mathcal{Z}$) to the **Receiver**. The **Receiver** measures each individual qubit $q_i$, either in computational basis if $y=0$ or in the diagonal basis $y=1$, and stores all classical measurement results $m^y_i \in \{0, 1\}$. Concurrently, the **Receiver** solves the TLP ($\mathcal{Z}$), which will reveal the indices $k, l$ as the solution. Finally, once

25

the puzzle is solved, the **Receiver** computes the parity of the stored measurement outputs for $\{q_k, q_l\}$, that is, $m^y_k \oplus m^y_l = x_y$.

The protocol still works in the NQSM, but instead of relying on an explicit time-delay introduced by the **Sender** in the execution of the protocol, it relies on a TLP to enforce it. This has several advantages (besides proving that such a construction is possible), as it delegates the responsibility of time-keeping from the sender to a cryptographic primitive. But, perhaps as important, it allows for a single TLP to hide the secret information of many OTs/OTMs, effectively amortizing the time lag and computation required to perform many executions that are performed in parallel, greatly boosting performance.

For the protocol to be secure, the TLP is designed such that it explores the quantum decoherence of imperfect quantum memories, here embodied by the NQSM. Setting the time it takes to solve the TLP ($\tau$) such that it is larger than the decoherence time modeled by the NQSM, again, enforces the **Receiver** to measures the two entangled qubits without knowing the encoding, as required to achieve security.

#### 4.2.1 Security

Again, to guarantee security, it must be proved that the **Sender** cannot obtain any information regarding the input of the **Receiver**; and, that the **Receiver** can recover at most one of the inputs of the **Sender**. Since this is a one-shot protocol, security requires that: the **Sender** cannot construct a message (e.g., by keeping correlated ancillas) that allows it to extract any information on the input of the **Receiver**; and that a (single) honestly-crafted message does not reveal more than one of the inputs of the **Sender** regardless of any POVM on the overall register that the **Receiver** can perform, and assuming the security of the underlying assumptions of the TLP.

The protocol is parameterized by the statistical security parameter $\sigma$, computational security parameter $\lambda$, and the time $\tau$ to quantum decoherence of memories established from the NQSM.

**Theorem 4.2.** *The protocol from Figure 2 implements a computationally-secure 1-out-of-2 Oblivious Transfer protocol, assuming the existence of a One-Way Function and a Sequential Function, in the Noisy-Quantum-Storage Model (parameterized by $\sigma, \lambda, \tau$). It is a One-Time Memory, since it is also one-shot.*

*Proof.* Assuredly, the protocol of Figure 2 implements a 1-out-of-2 OT functionality. It is also one-shot, thus if it implements a secure (chosen-input) 1-out-of-2 OT, then it is a OTM (Definition 2.2). So, it remains to prove that it fulfills the security requirements of OT in the NQSM.

From Lemma 4.1, there exists a secure weak TLP assuming the existence of a OWF and a SF, which can be generated in parallel in time $\log \tau$ and that takes time $\tau$ to solve. Then, from the NQSM, let $\tau$ be the time that a quantum memory takes to completely decohere, up to probability $2^{-\sigma}$. Again, the NQSM can be applied to the setting of this protocol as the memory of a malicious **Receiver** must linearly increase with $N$, the number of sent qubits by the **Sender**, which exponentially decreases its memory storage capabilities, as in Equation (2.4).

**Receiver-security:** Same as in Section 3.2.1. All the sender does is send the same ($N$) qubits as before, and instead of sending the indices $k, l$ after, it sends a TLP hiding $k, l$ together with the qubits. Clearly, from the security of the weak TLP, there is nothing the **Sender** can do that allow it to gain any information on the input of the **Receiver**.

26

**Sender-security:** From the security of the TLP, the puzzle does not reveal any information about the indices $k, l$ before time $\tau$, up to negligible probability in $\lambda$. Assuming the NQSM, this means that a malicious **Receiver** cannot store the $N$ qubits more time than the one it takes to solve the puzzle, as they would completely decohere. Then, before time $\tau$, the view of the **Receiver** is indistinguishable (it is the same) of its view in the previous setting of Section 3.2.1 (where all the **Receiver** sees is the $N$ qubits, before receiving $k, l$), up to a negligible probability in $\lambda$, assuming the hardness of the weak TLP. And, after time $\tau$, the view is also indistinguishable, as in both cases the **Receiver** gets total information on the indices $k, l$. Thus, all a malicious **Receiver** can do in this setting, it could also do in the secure setting of Section 3.2, which is proved to be secure. Moreover, since after the execution of the protocol (after time $\tau$) all information in the system is lost, there is no possibility of the malicious **Receiver** recovering the other message in the future, making the OT protocol everlasting secure.

Therefore, by reduction, assuming the existence of a OWF and a SF, and working in the NQSM, no malicious **Sender** or malicious **Receiver** can do anything more when engaging in the protocol of Figure 2 than they could have done in the secure protocol established in Section 3.2. The proof that the protocol of Figure 2 implements a secure chosen-input 1-out-of-2 OT, together with the property of being one-shot (consists of a single message transmitted from the **Sender** to the **Receiver**) means that this protocol is a OTM.

### 4.3 One-Time Program

We construct OTPs in the NQSM, by using the designed OTM of Section 4.2, and thus assuming the existence of a OWF and a SF. This result comes as a direct corollary of the one-time compiler from [GKR08, GIS+10], where from parallel-OTM and OWFs, it is possible to construct OTPs from any standard program (Theorem 2.2). This compiler is directly applicable to the polynomial parallel executions of the OTM achieved in Section 4.2. As such, its size is exponential (but constant) in the statistical security parameter, but still only grows polynomially with the power of the adversary (i.e., with the computational security parameter).

Indeed, our OTM construction, given the essence of the NQSM, has the property that it must be evaluated right away. Unfortunately this is unavoidable in our setting of the NQSM, but still allows for OTPs that are evaluated as they are received. Critically, our construction of OTM implies that, once received, all its qubits must be measured before the TLP may be solved (see Theorem 4.2). From this fact, it is straightforward to notice that our construction of OTM immediately yields **parallel-OTM**: Suppose an adversary receives two OTMs, implemented as in Section 4.2. If it waits to finish solving the TLP corresponding to the first before measuring the states corresponding to the second, these would have completely decohered by then (up to exponentially low probability). Moreover, the OTM protocol of Figure 2 may also be trivially extended to encompass the hiding of all encoding information of the multiple OTMs in a single TLP, making the proof of parallel-OTM follow by direct application of the proof of Theorem 4.2. Note that, even though we propose OTMs with bit outputs, these may be compiled to OTM with string outputs, and then directly applied to obtain OTP [GIS+10].

We remark that the definitions introduced in [GKR08, GIS+10] are simulation-based definitions, while in our work we prove security by demonstrating the properties of Sender and Receiver security of OT, together with the property of being one-shot, according to Definitions 2.1 and 2.2. Nevertheless, assuming our proposal described in Figure 2 implements an OTM, the simulation based security of the compiled OTP is independent of the underlying OTM implementation.

27

## Acknowledgements

The authors thank David Elkouss for insightful discussions.

RF acknowledges the support of the QuantaGenomics project funded within the QuantERA II Programme that has received funding from the European Union's Horizon 2020 research and innovation programme under Grant Agreement No 101017733, and with funding organisations, The Foundation for Science and Technology — FCT (QuantERA/0001/2021), Agence Nationale de la Recherche — ANR, and State Research Agency — AEI. This work was supported in part by the European Union under the programs Horizon Europe R&I, through the project QSNP (GA 101114043). MG acknowledges FCT - Fundação para a Ciência e a Tecnologia (Portugal) financing refs. UIDB/50021/2020 and UIDP/50021/2020 (resp. DOI 10.54499/UIDB/50021/2020 and 10.54499/UIDP/50021/2020). LN acknowledges support from FCT - Fundação para a Ciência e a Tecnologia (Portugal) via the Project No. CEECINST/00062/2018. EZC acknowledges funding by FCT/MCTES - Fundação para a Ciência e a Tecnologia (Portugal) - through national funds and when applicable co-funding by EU funds under the project UIDB/50008/2020, and funding by FCT through project 2021.03707.CEECIND/CP1653/CT0002.

## References

[ABJ+19] Prabhanjan Ananth, Saikrishna Badrinarayanan, Aayush Jain, Nathan Manohar, and Amit Sahai. From fe combiners to secure mpc and back. In Dennis Hofheinz and Alon Rosen, editors, *Theory of Cryptography*, pages 199–228, Cham, 2019. Springer International Publishing.
[ABKK23] Amit Agarwal, James Bartusek, Dakshita Khurana, and Nishant Kumar. A new framework for quantum oblivious transfer. In Carmit Hazay and Martijn Stam, editors, *Advances in Cryptology – EUROCRYPT 2023*, pages 363–394, Cham, 2023. Springer Nature Switzerland.
[ACGJ18] Prabhanjan Ananth, Arka Rai Choudhuri, Aarushi Goel, and Abhishek Jain. Round-optimal secure multiparty computation with honest majority. In Hovav Shacham and Alexandra Boldyreva, editors, *Advances in Cryptology – CRYPTO 2018*, pages 395–424, Cham, 2018. Springer International Publishing.
[ACJ17] Prabhanjan Ananth, Arka Rai Choudhuri, and Abhishek Jain. A new approach to round-optimal secure multiparty computation. In Jonathan Katz and Hovav Shacham, editors, *Advances in Cryptology – CRYPTO 2017*, pages 468–499, Cham, 2017. Springer International Publishing.
[AMPS16] N Aharon, S Massar, S Pironio, and J Silman. Device-independent bit commitment based on the chsh inequality. *New Journal of Physics*, 18(2):025014, feb 2016.
[AMZ24] Shweta Agrawalr, Giulio Malavolta, and Tianwei Zhang. Time-lock puzzles from lattices. In Leonid Reyzin and Douglas Stebila, editors, *Advances in Cryptology – CRYPTO 2024*, pages 425–456, Cham, 2024. Springer Nature Switzerland.
[BBCS92] Charles H. Bennett, Gilles Brassard, Claude Crépeau, and Marie-Hélène Skubiszewska. Practical quantum oblivious transfer. In Joan Feigenbaum, editor, *Advances in Cryptology – CRYPTO '91*, pages 351–366, Berlin, Heidelberg, 1992. Springer Berlin Heidelberg.

28

[BCKM21] James Bartusek, Andrea Coladangelo, Dakshita Khurana, and Fermi Ma. One-Way Functions Imply Secure Computation in a Quantum World. In Tal Malkin and Chris Peikert, editors, *Advances in Cryptology – CRYPTO 2021*, pages 467–496, Cham, 2021. Springer International Publishing.
[BCQ23] Zvika Brakerski, Ran Canetti, and Luowen Qian. On the Computational Hardness Needed for Quantum Cryptography. In Yael Tauman Kalai, editor, *14th Innovations in Theoretical Computer Science Conference (ITCS 2023)*, volume 251 of *Leibniz International Proceedings in Informatics (LIPIcs)*, pages 24:1–24:21, Dagstuhl, Germany, 2023. Schloss Dagstuhl – Leibniz-Zentrum für Informatik.
[BGI+14] Amos Beimel, Ariel Gabizon, Yuval Ishai, Eyal Kushilevitz, Sigurd Meldgaard, and Anat Paskin-Cherniavsky. Non-interactive secure multiparty computation. In Juan A. Garay and Rosario Gennaro, editors, *Advances in Cryptology – CRYPTO 2014*, pages 387–404, Berlin, Heidelberg, 2014. Springer Berlin Heidelberg.
[BGJ+16] Nir Bitansky, Shafi Goldwasser, Abhishek Jain, Omer Paneth, Vinod Vaikuntanathan, and Brent Waters. Time-Lock Puzzles from Randomized Encodings. In *Proceedings of the 2016 ACM Conference on Innovations in Theoretical Computer Science, ITCS '16*, page 345–356, New York, NY, USA, 2016. Association for Computing Machinery.
[BGS13] Anne Broadbent, Gus Gutoski, and Douglas Stebila. Quantum one-time programs. In Ran Canetti and Juan A. Garay, editors, *Advances in Cryptology – CRYPTO 2013*, pages 344–360, Berlin, Heidelberg, 2013. Springer Berlin Heidelberg.
[BGZ21] Anne Broadbent, Sevag Gharibian, and Hong-Sheng Zhou. Towards Quantum One-Time Memories from Stateless Hardware. *Quantum*, 5:429, April 2021.
[BHP17] Zvika Brakerski, Shai Halevi, and Antigoni Polychroniadou. Four round secure computation without setup. In Yael Kalai and Leonid Reyzin, editors, *Theory of Cryptography*, pages 645–677, Cham, 2017. Springer International Publishing.
[BKS23] James Bartusek, Dakshita Khurana, and Akshayaram Srinivasan. Secure computation with shared epr pairs (or: How to teleport in zero-knowledge). In Helena Handschuh and Anna Lysyanskaya, editors, *Advances in Cryptology – CRYPTO 2023*, pages 224–257, Cham, 2023. Springer Nature Switzerland.
[BL18] Fabrice Benhamouda and Huijia Lin. k-round multiparty computation from k-round oblivious transfer via garbled interactive circuits. In Jesper Buus Nielsen and Vincent Rijmen, editors, *Advances in Cryptology – EUROCRYPT 2018*, pages 500–532, Cham, 2018. Springer International Publishing.
[BL20] Fabrice Benhamouda and Huijia Lin. Mr nisc: Multiparty reusable non-interactive secure computation. In Rafael Pass and Krzysztof Pietrzak, editors, *Theory of Cryptography*, pages 349–378, Cham, 2020. Springer International Publishing.
[BW92] Charles H. Bennett and Stephen J. Wiesner. Communication via one- and two-particle operators on Einstein-Podolsky-Rosen states. *Phys. Rev. Lett.*, 69:2881–2884, Nov 1992.
[BY23] Anne Broadbent and Peter Yuen. Device-independent oblivious transfer from the bounded-quantum-storage-model and computational assumptions. *New Journal of Physics*, 25(5):053019, may 2023.

29

[CCM98] Christian Cachin, Claude Crepeau, and Julien Marcil. Oblivious transfer with a memory-bounded receiver. In *Proceedings of the 39th Annual Symposium on Foundations of Computer Science, FOCS '98*, page 493, USA, 1998. IEEE Computer Society.
[CDP10] Giulio Chiribella, Giacomo Mauro D'Ariano, and Paolo Perinotti. Probabilistic theories with purification. *Phys. Rev. A*, 81:062348, Jun 2010.
[CFHL21] Kai-Min Chung, Serge Fehr, Yu-Hsuan Huang, and Tai-Ning Liao. On the compressed-oracle technique, and post-quantum security of proofs of sequential work. In Anne Canteaut and François-Xavier Standaert, editors, *Advances in Cryptology – EUROCRYPT 2021*, pages 598–629, Cham, 2021. Springer International Publishing.
[CK88] Claude Crépeau and Joe Kilian. Achieving oblivious transfer using weakened security assumptions. In [*Proceedings 1988*] *29th Annual Symposium on Foundations of Computer Science*, pages 42–52. IEEE Computer Society, 1988.
[CM97] Christian Cachin and Ueli Maurer. Unconditional security against memory-bounded adversaries. In Burton S. Kaliski, editor, *Advances in Cryptology – CRYPTO '97*, pages 292–306, Berlin, Heidelberg, 1997. Springer Berlin Heidelberg.
[COWZ22] Michele Ciampi, Rafail Ostrovsky, Hendrik Waldner, and Vassilis Zikas. Round-optimal and communication-efficient multiparty computation. In Orr Dunkelman and Stefan Dziembowski, editors, *Advances in Cryptology – EUROCRYPT 2022*, pages 65–95, Cham, 2022. Springer International Publishing.
[Cré88] Claude Crépeau. Equivalence Between Two Flavours of Oblivious Transfers. In Carl Pomerance, editor, *Advances in Cryptology – CRYPTO 1987*, pages 350–354, Berlin, Heidelberg, 1988. Springer Berlin Heidelberg.
[CSW20] Ran Canetti, Pratik Sarkar, and Xiao Wang. Efficient and round-optimal oblivious transfer and commitment with adaptive security. In Shiho Moriai and Huaxiong Wang, editors, *Advances in Cryptology – ASIACRYPT 2020*, pages 277–308, Cham, 2020. Springer International Publishing.
[DDKZ14] Konrad Durnoga, Stefan Dziembowski, Tomasz Kazana, and Michal Zajac. One-time programs with limited memory. In Dongdai Lin, Shouhuai Xu, and Moti Yung, editors, *Information Security and Cryptology*, pages 377–394, Cham, 2014. Springer International Publishing.
[DFR+07] Ivan B. Damgård, Serge Fehr, Renato Renner, Louis Salvail, and Christian Schaffner. A tight high-order entropic quantum uncertainty relation with applications. In Alfred Menezes, editor, *Advances in Cryptology – CRYPTO 2007*, pages 360–378, Berlin, Heidelberg, 2007. Springer Berlin Heidelberg.
[DFSS05] Ivan B. Damgård, Serge Fehr, Louis Salvail, and Christian Schaffner. Cryptography In the Bounded Quantum-Storage Model. In *Proceedings of the 46th Annual IEEE Symposium on Foundations of Computer Science, FOCS '05*, page 449–458, USA, 2005. IEEE Computer Society.
[DFSS06] Ivan B. Damgård, Serge Fehr, Louis Salvail, and Christian Schaffner. Oblivious transfer and linear functions. In Cynthia Dwork, editor, *Advances in Cryptology – CRYPTO 2006*, pages 427–444, Berlin, Heidelberg, 2006. Springer Berlin Heidelberg.

30

[DGH+20] Nico Döttling, Sanjam Garg, Mohammad Hajiabadi, Daniel Masny, and Daniel Wichs. Two-Round Oblivious Transfer from CDH or LPN. In Anne Canteaut and Yuval Ishai, editors, *Advances in Cryptology – EUROCRYPT 2020*, pages 768–797, Cham, 2020. Springer International Publishing.
[EGG+22] Harry Eldridge, Aarushi Goel, Matthew Green, Abhishek Jain, and Maximilian Zinkus. One-time programs from commodity hardware. In Eike Kiltz and Vinod Vaikuntanathan, editors, *Theory of Cryptography*, pages 121–150, Cham, 2022. Springer Nature Switzerland.
[EGL85] Shimon Even, Oded Goldreich, and Abraham Lempel. A Randomized Protocol for Signing Contracts. *Communications of the ACM*, 28(6):637–647, 1985.
[FG21] Ricardo Faleiro and Manuel Goulão. Device-independent quantum authorization based on the Clauser-Horne-Shimony-Holt game. *Phys. Rev. A*, 103:022430, Feb 2021.
[FKN94] Uri Feige, Joe Killian, and Moni Naor. A minimal model for secure computation (extended abstract). In *Proceedings of the Twenty-Sixth Annual ACM Symposium on Theory of Computing, STOC '94*, page 554–563, New York, NY, USA, 1994. Association for Computing Machinery.
[GIS+10] Vipul Goyal, Yuval Ishai, Amit Sahai, Ramarathnam Venkatesan, and Akshay Wadia. Founding cryptography on tamper-proof hardware tokens. In Daniele Micciancio, editor, *Theory of Cryptography*, pages 308–326, Berlin, Heidelberg, 2010. Springer Berlin Heidelberg.
[GKR08] Shafi Goldwasser, Yael Tauman Kalai, and Guy N. Rothblum. One-time programs. In David Wagner, editor, *Advances in Cryptology – CRYPTO 2008*, pages 39–56, Berlin, Heidelberg, 2008. Springer Berlin Heidelberg.
[GLSV21] Alex B. Grilo, Huijia Lin, Fang Song, and Vinod Vaikuntanathan. Oblivious Transfer Is in MiniQCrypt. In Anne Canteaut and François-Xavier Standaert, editors, *Advances in Cryptology – EUROCRYPT 2021*, pages 531–561, Cham, 2021. Springer International Publishing.
[GMPP16] Sanjam Garg, Pratyay Mukherjee, Omkant Pandey, and Antigoni Polychroniadou. The exact round complexity of secure computation. In Marc Fischlin and Jean-Sébastien Coron, editors, *Advances in Cryptology – EUROCRYPT 2016*, pages 448–476, Berlin, Heidelberg, 2016. Springer Berlin Heidelberg.
[GMW87] O. Goldreich, S. Micali, and A. Wigderson. How to play any mental game. In *Proceedings of the Nineteenth Annual ACM Symposium on Theory of Computing, STOC '87*, page 218–229, New York, NY, USA, 1987. Association for Computing Machinery.
[GS18] Sanjam Garg and Akshayaram Srinivasan. Two-round multiparty secure computation from minimal assumptions. In Jesper Buus Nielsen and Vincent Rijmen, editors, *Advances in Cryptology – EUROCRYPT 2018*, pages 468–499, Cham, 2018. Springer International Publishing.
[GSS+19] Yelena Guryanova, Ralph Silva, Anthony J. Short, Paul Skrzypczyk, Nicolas Brunner, and Sandu Popescu. Exploring the limits of no backwards in time signalling. *Quantum*, 3:211, December 2019.

31

[GW10] Deepthi Gopal and Stephanie Wehner. Using postmeasurement information in state discrimination. *Phys. Rev. A*, 82:022326, Aug 2010.
[HHPV18] Shai Halevi, Carmit Hazay, Antigoni Polychroniadou, and Muthuramakrishnan Venkitasubramaniam. Round-optimal secure multi-party computation. In Hovav Shacham and Alexandra Boldyreva, editors, *Advances in Cryptology – CRYPTO 2018*, pages 488–520, Cham, 2018. Springer International Publishing.
[HIJ+17] Shai Halevi, Yuval Ishai, Abhishek Jain, Ilan Komargodski, Amit Sahai, and Eylon Yogev. Non-interactive multiparty computation without correlated randomness. In Tsuyoshi Takagi and Thomas Peyrin, editors, *Advances in Cryptology – ASIACRYPT 2017*, pages 181–211, Cham, 2017. Springer International Publishing.
[HIKR18] Shai Halevi, Yuval Ishai, Eyal Kushilevitz, and Tal Rabin. Best possible information-theoretic mpc. In *Theory of Cryptography: 16th International Conference, TCC 2018, Panaji, India, November 11–14, 2018, Proceedings, Part II*, page 255–281, Berlin, Heidelberg, 2018. Springer-Verlag.
[IKO+11] Yuval Ishai, Eyal Kushilevitz, Rafail Ostrovsky, Manoj Prabhakaran, and Amit Sahai. Efficient non-interactive secure computation. In Kenneth G. Paterson, editor, *Advances in Cryptology – EUROCRYPT 2011*, pages 406–425, Berlin, Heidelberg, 2011. Springer Berlin Heidelberg.
[IKP10] Yuval Ishai, Eyal Kushilevitz, and Anat Paskin. Secure multiparty computation with minimal interaction. In Tal Rabin, editor, *Advances in Cryptology – CRYPTO 2010*, pages 577–594, Berlin, Heidelberg, 2010. Springer Berlin Heidelberg.
[IR89] R. Impagliazzo and S. Rudich. Limits on the Provable Consequences of One-Way Permutations. In *Proceedings of the Twenty-First Annual ACM Symposium on Theory of Computing, STOC '89*, page 44–61, New York, NY, USA, 1989. Association for Computing Machinery.
[JMRR21] Samuel Jaques, Hart Montgomery, Razvan Rosie, and Arnab Roy. Time-release cryptography from minimal circuit assumptions. In Avishek Adhikari, Ralf Küsters, and Bart Preneel, editors, *Progress in Cryptology – INDOCRYPT 2021*, pages 584–606, Cham, 2021. Springer International Publishing.
[Kal05] Yael Tauman Kalai. Smooth projective hashing and two-message oblivious transfer. In Ronald Cramer, editor, *Advances in Cryptology – EUROCRYPT 2005*, pages 78–95, Berlin, Heidelberg, 2005. Springer Berlin Heidelberg.
[Kil88] Joe Kilian. Founding Crytpography on Oblivious Transfer. In *Proceedings of the Twentieth Annual ACM Symposium on Theory of Computing, STOC '88*, page 20–31, New York, NY, USA, 1988. Association for Computing Machinery.
[KO04] Jonathan Katz and Rafail Ostrovsky. Round-optimal secure two-party computation. In Matt Franklin, editor, *Advances in Cryptology – CRYPTO 2004*, pages 335–354, Berlin, Heidelberg, 2004. Springer Berlin Heidelberg.
[KW16] Jędrzej Kaniewski and Stephanie Wehner. Device-independent two-party cryptography secure against sequential attacks. *New Journal of Physics*, 18(5):055004, may 2016.

32

[KWW12] Robert Konig, Stephanie Wehner, and Jürg Wullschleger. Unconditional Security From Noisy Quantum Storage. *IEEE Transactions on Information Theory*, 58(3):1962–1984, 2012.
[LC97] Hoi-Kwong Lo and H. F. Chau. Is quantum bit commitment really possible? *Phys. Rev. Lett.*, 78:3410–3413, Apr 1997.
[Liu23] Qipeng Liu. Depth-Bounded Quantum Cryptography with Applications to One-Time Memory and More. In Yael Tauman Kalai, editor, *14th Innovations in Theoretical Computer Science Conference (ITCS 2023)*, volume 251 of *Leibniz International Proceedings in Informatics (LIPIcs)*, pages 82:1–82:18, Dagstuhl, Germany, 2023. Schloss Dagstuhl – Leibniz-Zentrum für Informatik.
[LM23] Russell W. F. Lai and Giulio Malavolta. Lattice-based timed cryptography. In Helena Handschuh and Anna Lysyanskaya, editors, *Advances in Cryptology – CRYPTO 2023*, pages 782–804, Cham, 2023. Springer Nature Switzerland.
[Lo97] Hoi-Kwong Lo. Insecurity of quantum secure computations. *Phys. Rev. A*, 56:1154–1162, Aug 1997.
[LPAK23] Cosmo Lupo, James T. Peat, Erika Andersson, and Pieter Kok. Error-tolerant oblivious transfer in the noisy-storage model. *Phys. Rev. Res.*, 5:033163, Sep 2023.
[Mau92] Ueli M. Maurer. Conditionally-Perfect Secrecy and a Provably-Secure Randomized Cipher. *Journal of Cryptology*, 5(1):53–66, 1992.
[May97] Dominic Mayers. Unconditionally secure quantum bit commitment is impossible. *Phys. Rev. Lett.*, 78:3414–3417, Apr 1997.
[NP01] Moni Naor and Benny Pinkas. Efficient oblivious transfer protocols. In *Proceedings of the Twelfth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA '01*, page 448–457, USA, 2001. Society for Industrial and Applied Mathematics.
[PAM+10] S. Pironio, A. Acín, S. Massar, A. Boyer de la Giroday, D. N. Matsukevich, P. Maunz, S. Olmschenk, D. Hayes, L. Luo, T. A. Manning, and C. Monroe. Random numbers certified by bell's theorem. *Nature*, 464(7291):1021–1024, Apr 2010.
[PPCT22] Jef Pauwels, Stefano Pironio, Emmanuel Zambrini Cruzeiro, and Armin Tavakoli. Adaptive Advantage in Entanglement-Assisted Communications. *Phys. Rev. Lett.*, 129:120504, Sep 2022.
[PVW08] Chris Peikert, Vinod Vaikuntanathan, and Brent Waters. A framework for efficient and composable oblivious transfer. In David Wagner, editor, *Advances in Cryptology – CRYPTO 2008*, pages 554–571, Berlin, Heidelberg, 2008. Springer Berlin Heidelberg.
[QWW18] Willy Quach, Hoeteck Wee, and Daniel Wichs. Laconic function evaluation and applications. In *2018 IEEE 59th Annual Symposium on Foundations of Computer Science (FOCS)*, pages 859–870, 2018.
[Rab81] Michael O. Rabin. How To Exchange Secrets with Oblivious Transfer. Harvard University Technical Report 81, 1981. https://eprint.iacr.org/2005/187.

33

[RCCG+20] Arka Rai Choudhuri, Michele Ciampi, Vipul Goyal, Abhishek Jain, and Rafail Ostrovsky. Round optimal secure multiparty computation from minimal assumptions. In Rafael Pass and Krzysztof Pietrzak, editors, *Theory of Cryptography*, pages 291–319, Cham, 2020. Springer International Publishing.
[RSW96] R. L. Rivest, A. Shamir, and D. A. Wagner. Time-Lock Puzzles and Timed-Release Crypto. MIT Laboratory for Computer Science, 1996. https://dl.acm.org/doi/book/10.5555/888615.
[RVK95] J. Richter, A. Voigt, and S. Krüger. A solvable quantum spin model: the frustrated heisenberg star. *Journal of Magnetism and Magnetic Materials*, 140-144:1497–1498, 1995. International Conference on Magnetism.
[Sal98] Louis Salvail. Quantum bit commitment from a physical assumption. In Hugo Krawczyk, editor, *Advances in Cryptology – CRYPTO '98*, pages 338–353, Berlin, Heidelberg, 1998. Springer Berlin Heidelberg.
[Sch10] Christian Schaffner. Simple protocols for oblivious transfer and secure identification in the noisy-quantum-storage model. *Phys. Rev. A*, 82:032308, Sep 2010.
[Sha48] C. E. Shannon. A mathematical theory of communication. *The Bell System Technical Journal*, 27(3):379–423, 1948.
[VAVD+22] J Verjauw, R Acharya, J Van Damme, Ts Ivanov, D Perez Lozano, FA Mohiyaddin, D Wan, J Jussot, AM Vadiraj, M Mongillo, et al. Path toward manufacturable superconducting qubits with relaxation times exceeding 0.1 ms. *npj Quantum Information*, 8(1):93, 2022.
[VV14] Umesh Vazirani and Thomas Vidick. Fully device-independent quantum key distribution. *Phys. Rev. Lett.*, 113:140501, Sep 2014.
[Wie83] Stephen Wiesner. Conjugate coding. *SIGACT News*, 15(1):78–88, jan 1983.
[WST08] Stephanie Wehner, Christian Schaffner, and Barbara M. Terhal. Cryptography from Noisy Storage. *Phys. Rev. Lett.*, 100:220502, Jun 2008.
[Yao86] Andrew Chi-Chih Yao. How to generate and exchange secrets. In *27th Annual Symposium on Foundations of Computer Science, FOCS' 86*, pages 162–167, 1986.

34