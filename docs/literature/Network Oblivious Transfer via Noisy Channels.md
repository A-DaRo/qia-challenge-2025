# Network Oblivious Transfer via Noisy Channels: Limits and Capacities

Hadi Aghaee†, Bahareh Akhbari*, Member, IEEE, Christian Deppe†, Senior Member, IEEE
†Institute for Communications Technology, Technische Universität Braunschweig, Braunschweig, Germany
*Faculty of Electrical Engineering, K. N. Toosi University of Technology, Tehran, Iran
Email: (hadi.aghaee, christian.deppe)@tu-bs.de, akhbari@kntu.ac.ir

## Abstract

In this paper, we aim to study the information-theoretical limits of oblivious transfer. This work also investigates the problem of oblivious transfer over a noisy multiple access channel involving two non-colluding senders and a single receiver. The channel model is characterized by correlations among the parties, with the parties assumed to be either honest-but-curious or, in the receiver's case, potentially malicious. At first, we study the information-theoretical limits of oblivious transfer between two parties and extend it to the multiple access channel model. We propose a multiparty protocol for honest-but-curious parties where the general multiple access channel is reduced to a certain correlation. In scenarios where the receiver is malicious, the protocol achieves an achievable rate region.

## Index Terms

Oblivious transfer, Multiple access channel, Bounds for OT capacity.

## I. INTRODUCTION

OBLIVIOUS Transfer (OT), a fundamental primitive in secure multiparty computation (MPC), plays a central role in the design of cryptographic protocols. OT is complete in the sense that, given access to an OT protocol between two parties, any function can be securely computed between them. More precisely, if OT is assumed to be a trusted primitive available between two parties, then it suffices to achieve general secure computation, even in the presence of malicious adversaries [1]. The most basic form of OT is the 1-out-of-2 OT, where a sender (Alice) holds two distinct messages and a receiver (Bob) selects one of them to receive, without revealing his choice to Alice. At the same time, Bob learns nothing about the unchosen message. A natural extension is the 1-out-of-m OT, in which Alice holds $m$ messages (e.g., bit strings), and Bob retrieves exactly one, without gaining information about the others or revealing his selection to Alice.

OT was first introduced by Rabin [2]. In Rabin’s form, Alice sends a message to Bob with the probability of $\frac{1}{2}$ while she remains oblivious to whether or not Bob received the message. This model is called the “Erasure Channel” with the erasure probability equal $\frac{1}{2}$. After that, a basic OT protocol was introduced by Even, Goldreich, and Lempel (EGL) [3]. It is well-known that achieving multi-party security (as a basic model) in noise-free communication is impossible [4]. It has been shown that achieving two-party secure communication over a noiseless channel is possible by randomness sharing [5, 6]. Shared randomness consists of random variables known to all communicating parties but independent of the message being transmitted [7].

Up to now, some primary channels have been studied for OT purposes obtained from noise. The binary symmetric channel (BSC) has made a more outstanding contribution [8–10]. However, all the cryptogates and channels that can be used for obtaining OT are characterized by Kilian in the case of passive adversaries [11]. It should be mentioned that most of the research works in this field have been done from the perspective of the basics of cryptography, and there are a few sources that study the problem of OT from the information theory point of view. However, the OT capacity of noisy channels is generally unknown. It is known to be non-negligible if the players (sometimes we call senders and receivers as players/parties) are committed to the protocol and implement it faithfully, not turning away from additional information (honest-but-curious players). Still, in the case of fully malicious players (active adversaries), non-zero rates have not ever been achieved [12].

As the first step, Nascimento and Winter study the OT capacity of noisy correlations [13], wherein they characterized which noisy channels and distributions are useful for obtaining OT. In [6], they showed that for honest-but-curious players, the OT capacity of noisy resources is positive by achieving a lower bound that coincides with the upper bound of [5]. The OT capacity of the binary erasure channel (BEC) is studied in [14], in which the OT capacity is $C_{OT} = \frac{1}{2}$ with erasure probability $\frac{1}{2}$ that is a property of the channel/system model in the case of honest-but-curious players and a lower bound is calculated in the case of fully malicious players. Ahlswede and Csiszar achieved a lower and upper bound on the OT capacity of noisy channels [5]. The upper bound is general and valid for every noisy channel with honest-but-curious players, while the lower bound is just

> This paper was presented in part at the 2025 IEEE International Symposium on Information Theory (ISIT 2025), and was presented in 5th Workshop on Enabling Security, Trust, and Privacy in 6G Wireless Systems at the 2025 IEEE Global Communications Conference (GLOBECOM).

valid for a special reduced version of a DMC, wherein the channel outputs are separable into two distinct sets: fully erased bits and fully received bits. It can be easily seen that the upper bound in [5] and the lower bound in [6] for a special reduced version coincide. An improved upper bound compared to [5] is proved in [15] based on a monotonicity property of the tension region in the channel model.

The limits of OT are another important bottleneck in secure two-party computation, so that is addressed in various cryptographic-based papers [16, 17]. Some limitations relate to the capability of cheating parties, and some of them relate to the power of noise over the communication channels [18]. The information-theoretical limits of OT between two parties over a shared channel between three nodes are first addressed in [19] from the same authors. In this paper, we have a more general contribution. As a motivation, we would like to draw attention to the following statement by Kurt Gödel [20]:

> "Any consistent formal system within which a certain amount of arithmetic can be carried out is incomplete; there are statements of arithmetic which are true, but not provable within the system."
> — K. Gödel, *On Formally Undecidable Propositions of Principia Mathematica and Related Systems*, 1931.

As Gödel showed that no formal system can be both complete and consistent, one can say no cryptographic primitive, even as complete as OT, can escape the limits imposed by the underlying communication models and assumptions.

In this work, we also aim to study bounds for the OT capacity of the two-user Discrete Memoryless Multiple Access Channel (DM-MAC) as one of the primary network models from the perspective of network information theory. The MAC refers to a communication scenario where multiple users send information over a shared channel to a receiver. This setup is typical in communication systems, such as cellular networks, where several devices need to communicate with a base station or an access point. We consider the following system model: Two senders, both send two independent messages (two strings) over a noisy channel to a receiver. The receiver then has to choose only one string from each sender, and the senders are assumed to be legitimate relative to each other (non-colluding senders). This means that there is no criterion of secrecy between them.

This paper is organized as follows: Some seminal definitions are presented in Section II. Section III is dedicated to related works and known results. The system model and main results are presented in Sections IV and V, respectively. We provide some examples in Section VI, and a brief discussion in Section VII.

## II. PRELIMINARIES

We use the well-known notation of information theory in addition to the following notations: We use capital letters (e.g., $X$) to denote random variables, with the specific alphabet $\mathcal{X}$ defined by the context in which $X$ is used. Lowercase letters (e.g., $x$) represent realizations of the corresponding random variables. Bold uppercase letters (e.g., $\mathbf{X}$) denote random $n$-tuples.

### 1. Notation for Tuples

*   Suppose $A \subset \mathcal{N}$. Then $(A)$ represents the tuple formed by arranging the elements of $A$ in increasing order:
    $$(A) = (a_i | a_i \in A, i = 1, 2, \ldots, |A|), \quad \text{with } a_i < a_{i+1} \text{ for } i \ge 1$$
    Example: If $A = \{1, 3, 2, 9, 4\}$, then $(A) = (1, 2, 3, 4, 9)$.
*   For two sets $A \subset \{1, 2, \ldots, k\}$, and $\mathbf{X}$, we have:
    $$\mathbf{X}|_A = \mathbf{X}|_{(A)} = \{x_i | i \in A\}.$$
    Example: If $\mathbf{X} = (a, b, c, d, e, f, g)$ and $A = \{6, 3, 1\}$, then $\mathbf{X}_A = (a, c, f)$.
*   When a member $i$ is removed from a set $F$, we denote the case by:
    $$F \setminus \{i\}.$$
    Example: Given $F = \{a, b, c, d, e\}$, we have: $\{a, b, d, e\} = F \setminus \{c\}$.

### 2. Markov Chains

Random variables $X, Y, Z$ form a Markov chain $X - Y - Z$ when $X$ and $Z$ are conditionally independent given $Y$. That is, if $X \in \mathcal{X}, Y \in \mathcal{Y}$, and $Z \in \mathcal{Z}$, then $X - Y - Z$ implies:
$$\forall x \in \mathcal{X}, \forall y \in \mathcal{Y}, \forall z \in \mathcal{Z} : P_{X, Z|Y}(x, z|y) = P_{X|Y}(x|y) \cdot P_{Z|Y}(z|y)$$

### 3. Erasure Count Function

Given a sequence $\mathbf{y} \in \{0, 1, e\}^n$, where $e$ indicates an erasure. We denote the erasure count function by:
$$\Delta(\mathbf{y}^n) = |\{i \in \{1, 2, \ldots, n\} : y_i = e\}|,$$
$$\Delta(\mathbf{y}^n) = |\{i \in \{1, 2, \ldots, n\} : y_i \ne e\}|,$$
where $y_i$ is a realization of $Y$.

### 4. Information Theoretic Definitions

The min-entropy of a discrete random variable $X$ is
$$H_{\infty}(X) = \min_{x} \log \left(\frac{1}{P_X(x)}\right).$$
Its conditional version is
$$H_{\infty}(X|Y) = \min_{y} H_{\infty}(X|Y = y).$$
The zero-entropy and its conditional version are defined as
$$H_{0}(X) = \log |\mathcal{X} \in \mathcal{X} : P_X(x) > 0\}|,$$
and
$$H_{0}(X|Y) = \max_{y} H_{0}(X|Y = y).$$
The statistical distance over two probability distributions $P_X$ and $P_Y$, defined over the same domain $\mathcal{X}$, is
$$\|P_X - P_Y\| = \frac{1}{2} \sum_{x \in \mathcal{X}} |P_X(x) - P_Y(x)|.$$
For $\epsilon \ge 0$, the $\epsilon$-smooth min entropy is
$$H^{\epsilon}_{\infty}(X) = \max_{X': \|P_X - P_{X'}\| \le \epsilon} H_{\infty}(X').$$
Similarly,
$$H^{\epsilon}_{\infty}(X|Y) = \max_{X'Y': \|P_{X'Y} - P_{XY}\| \le \epsilon} H_{\infty}(X'|Y').$$
Let $P_{UVW}$ be a probability distribution over $\mathcal{U} \times \mathcal{V} \times \mathcal{W}$. For any $\epsilon > 0$ and $\epsilon' > 0$ it holds that [21]:
$$H^{\epsilon+\epsilon'}_{\infty}(U|V, W) \ge H_{\infty}(U|W) + H^{\epsilon}_{\infty}(V|U, W) - H_{0}(V|W) - \log \left(\frac{1}{1-\epsilon'}\right) \quad (1)$$
Also, $H^{\epsilon}_{\infty}(U, V|W)$ can be bounded from below and above as [21]:
$$H^{\epsilon+\epsilon'}_{\infty}(U|V, W) + H_{0}(V|W) + \log \left(\frac{1}{\epsilon'}\right) \le H^{\epsilon}_{\infty}(U, V|W) \le H_{\infty}(U|W) + H^{\epsilon}_{\infty}(V|U, W). \quad (2)$$
Combining (1) and (2), concludes:
$$H^{\epsilon}_{\infty}(U, V|W) \ge H_{\infty}(U|W) + H^{\epsilon}_{\infty}(V|U, W). \quad (3)$$

**Lemma 1.** For any random variable such $U$ and $V$, we have:
$$H^{\epsilon}_{\infty}(U|V) - \log \left(\frac{1}{\epsilon}\right) \le H_{\infty}(U|V) \le H^{\epsilon}_{\infty}(U|V).$$
**Proof.** In Appendix A.

**Definition 1.** Given a random variable $X$ with alphabet $\mathcal{X}$ and probability distribution $p_x$, the Rényi entropy of order two of a random variable $X$ is given by:
$$H_2(X) = \log_2 \left(\frac{1}{P_c(X)}\right),$$
where the collision probability $P_c(X)$ is the probability that two independent trials of $X$ produce the same outcome. It is defined as:
$$P_c(X) = \sum_{x \in \mathcal{X}} p_X(x)^2.$$
For a given event $\mathcal{E}$, the conditional distribution $p_{X|\mathcal{E}}$ is employed to define the conditional collision probability $P_c(X|\mathcal{E})$ and the conditional Rényi entropy of order 2, $H_2(X|\mathcal{E})$.

**Lemma 2.** [21, Corollary 2.12] Let $P_{X^n Y^n}$ be independent and identically distributed (i.i.d.) according to $P_{XY}$ over the alphabet $\mathcal{X}^n \times \mathcal{Y}^n$. For any $\epsilon > 0$, we have
$$H^{\epsilon}_{\infty}(X^n|Y^n) \ge nH(X|Y) - 4\sqrt{n}\log(1/\epsilon)\log |\mathcal{X}|.$$
**Proof.** In Appendix B.

**Definition 2.** A function $h : \mathcal{R} \times \mathcal{X} \rightarrow \{0, 1\}^n$ is a two-universal hash function [22] if, for any $x_0 \ne x_1 \in \mathcal{X}$ and for $R$ uniformly distributed over $\mathcal{R}$, it holds that
$$\text{Pr}(h(R, x_0) = h(R, x_1)) \le 2^{-n}. \quad (4)$$
Similarly, given two independent hash functions $h_1 : \mathcal{R} \times \mathcal{X} \rightarrow \{0, 1\}^n$ and $h_2 : \mathcal{T} \times \mathcal{Y} \rightarrow \{0, 1\}^m$, for any $x_0 \ne x_1 \in \mathcal{X}$, $y_0 \ne y_1 \in \mathcal{Y}$, and for $R, T$ uniformly distributed over $\mathcal{R}$ and $\mathcal{T}$, respectively, it holds that
$$\text{Pr}(h_1(R, x_0) = h_1(R, x_1) \cap h_2(T, y_0) = h_2(T, y_1)) \le 2^{-(n+m)}. \quad (5)$$
An example of a two-universal class is the set of all linear mappings from $\{0, 1\}^k$ to $\{0, 1\}^l$.
A random variable $X$ over $\mathcal{X}$ is said to be $\epsilon$-close to uniform with respect to $Z$ over $\mathcal{Z}$ if
$$\|P_{XZ} - (P_U \times P_Z)\| \le \epsilon,$$
where $U$ is uniformly distributed over $\mathcal{X}$.

**Lemma 3.** [6, 23] (Distributed leftover hash lemma) Let $\epsilon > 0$, $\epsilon' \ge 0$, and let $g_i : \mathcal{T}_i \times \mathcal{X}_i \rightarrow \{0, 1\}^{n_i}$ for $1 \le i \le m$ be two-universal hash functions. Assume random variables $X_i$ over $\mathcal{X}_i$, $1 \le i \le m$, where for any subset $\mathcal{S} \subset \{1, 2, \ldots, m\}$, and $\mathbf{X}|_{\mathcal{S}} = X_{S(1)}, X_{S(2)}, \ldots, X_{S(|\mathcal{S}|)}$, we have
$$H^{\epsilon}_{\infty}(\mathbf{X}|_{\mathcal{S}}|Z) \ge \sum_{i \in \mathcal{S}} n_i + 2\log(1/\epsilon),$$
where $T_1, \ldots, T_m$ are uniformly distributed over $\mathcal{T}_1 \times \cdots \times \mathcal{T}_m$, and are independent of $X_1, \ldots, X_m$, and $Z$. Then, it holds that the tuple $(g_1(T_1, X_1), \ldots, g_m(T_m, X_m))$ is $(2^{m}\epsilon/2 + 2^{m}\epsilon')$-close to uniform with respect to $T_1, \ldots, T_m, Z$.
We also briefly note that following from [24, Th. 17.3.3], it directly follows from the previously defined $X_i, T_i, g_i(T_i, X_i)$, and $Z$ that
$$I(g_i(T_i, X_i); T_i Z) \le -\epsilon'' \log \frac{\epsilon''}{2^{n_i} |\mathcal{Z}| |\mathcal{T}_i|}$$
where $\epsilon'' = 2^m\epsilon/2 + 2^m\epsilon'$ and $I(g_i(T_i, X_i); T_i Z)$ is defined as
$$I(g_i(T_i, X_i); T_i Z) = H(g_i(T_i, X_i)) - H(g_i(T_i, X_i)|T_i Z),$$
which $I$ and $H$ represent the Shannon mutual information and the Shannon entropy, respectively.
Consider a Discrete Memoryless Channel (DMC) with a transition matrix $W = \{W(y|x), x \in \mathcal{X}, y \in \mathcal{Y}\}$. There are two assumptions:

1) **Free Resources:** Alice and Bob have unlimited computing power, independent local randomness, and access to a noiseless public communication channel for unlimited rounds.
2) **Honest-but-Curious Model:** Both parties follow the protocol honestly but may use all available information to infer what they should remain ignorant about.

The general two-party protocol (Figure 1):
*   **Initial Views:** Alice and Bob start with initial knowledge or views $U'$ and $V'$, respectively.
*   **Random Experiments:** Alice generates random variable $M$, and Bob generates random variable $N$ independently of each other and $(U', V')$.
*   **Message Exchange:** Alice sends Bob a message $C_1$ as a function of $U'$ and $M$. Bob responds with $C_2$, a function of $V', N$ and $C_1$.
*   **Alternating Messages:** In subsequent rounds, they alternately send messages $C_3, C_4, \ldots, C_{2t}$, which are functions of their instantaneous views.
*   **Final Views:** At the end of the protocol, Alice's view $U$ is $(U', M, C)$ and Bob's view $V$ is $(V', N, C)$, where $C = C_1, \ldots, C_{2t}$.

There are two models: The channel model and the source model. In the source model, Alice's initial view is $U' = (M_0, M_1, X^n)$ and Bob's initial view is $V' = (Z, Y^n)$ where $(M_0, M_1)$ are binary strings and uniformly distributed on $\{0, 1\}^k$, and $Z \in \{0, 1\}$ is a binary bit.

In the channel model, Alice starts with her initial view $U' = (M_0, M_1)$ and Bob with his initial view $V' = Z$. In this case, Alice and Bob may perform any noisy protocol with $n$ access to the DMC with their initial views, where $M_0, M_1$ and $Z$ are independent, and $M_0, M_1$ are uniformly distributed on $\{0, 1\}^k$.

$$
\begin{array}{rcl}
\text{Alice} & & \text{Bob} \\
U' = (M_0, M_1, X^n) & \xrightarrow{C_1 = (U', M)} & V' = (Z, Y^n) \\
& \xleftarrow{C_2 = (V', N, C_1)} & \\
& \xrightarrow{C_3 = (U', M, C_1, C_2)} & \\
& \xleftarrow{C_4 = (V', N, C_1, C_2, C_3)} & \\
& \vdots & \\
& \xleftarrow{C_{2t} = (V', N, C)} & \\
U = (U', M, C) & & V = (V', N, C)
\end{array}
$$
**[IMAGE: Fig. 1: The general two-party protocol between Alice and Bob from the perspective of source model (Flowchart showing message exchange C1 through C2t between Alice and Bob, with initial views U' and V' and final views U and V)]**

**Lemma 4.** [25, Corollary 4] Let $p_{X, Y}$ be any probability distribution, where $X \in \mathcal{X}, Y \in \mathcal{Y}$, and $y$ is a specific realization of $Y$. Assume that $H_2(X|Y = y) \ge c$ for some constant $c$. Let $\mathcal{K}$ be a universal class of functions mapping $\mathcal{X}$ to $\{0, 1\}^l$, and let $\kappa$ be sampled uniformly from $\mathcal{K}$. Then:
$$H(\kappa(X)|\kappa, Y = y) \ge l - \log(1 + 2^{l-c})$$
$$\ge l - \frac{2^{l-c}}{\ln 2}$$

**Lemma 5.** [5, Lemma 3] Let $X, Y$, and $Z$ be random variables defined on the finite sets $\mathcal{X}, \mathcal{Y}$, and $\mathcal{Z}$, respectively. For any $z_1, z_2 \in \mathcal{Z}$ with $p \equiv \text{Pr}[Z = z_1] > 0$ and $q \equiv \text{Pr}[Z = z_2] > 0$, the following inequality holds:
$$|H(X|Y, Z = z_1) - H(X|Y, Z = z_2)| \le 1 + 3\log |\mathcal{X}| \sqrt{\frac{(p+q) \ln 2}{2pq} I(X, Y; Z)}.$$

**Lemma 6.** Consider a DM-MAC with two senders and one receiver defined by transition matrix $W(y|x_1, x_2)$. For pair words $(\mathbf{x}_1^n, \mathbf{x}_2^n)$ and $(\mathbf{x}'_1^n, \mathbf{x}'_2^n)$ with Hamming distances $d_H(\mathbf{x}_1^n, \mathbf{x}'_1^n) \ge \delta n$, $d_H(\mathbf{x}_2^n, \mathbf{x}'_2^n) \ge \delta n$, such that
$$\forall x_1 \in \mathcal{X}_1, x_2 \in \mathcal{X}_2, \forall P \text{ p.d. with } P(x_1, x_2) = 0, \left\|\mathbf{W}_{\mathbf{x}_1, \mathbf{x}_2}^n - \sum_{\mathbf{x}'_1, \mathbf{x}'_2} P(\mathbf{x}'_1, \mathbf{x}'_2) \mathbf{W}_{\mathbf{x}'_1, \mathbf{x}'_2}^n \right\| \ge \eta,$$
one has, with $\epsilon = \frac{\delta^2 \eta^2}{2 |\mathcal{X}_1|^2 |\mathcal{X}_2|^2 |\mathcal{Z}|}$,
$$\mathbf{W}_{\mathbf{x}'_1, \mathbf{x}'_2}^n (\mathbf{T}_{\mathbf{W}, \epsilon}^n(\mathbf{x}_1^n, \mathbf{x}_2^n)) \le 2 \exp\left(-\frac{n \epsilon^2}{4}\right)$$
where $\mathbf{T}_{\mathbf{W}, \epsilon}^n(\mathbf{x}_1^n, \mathbf{x}_2^n)$ is the set of joint typical sequences and $\mathbf{W}_{\mathbf{x}_1, \mathbf{x}_2}^n = W_{x_{1, 1} x_{2, 1}} W_{x_{1, 2} x_{2, 2}} \cdots W_{x_{1, n} x_{2, n}}$.

**Proof.** In Appendix C.

**Definition 3.** (Reduction) Throughout this paper, we consider two types of reduction: (i) **Cryptographic reduction:** A cryptographic reduction can be seen as a black-box that reduces one transfer mechanism $Q$ to $R$. If we reduce $Q$ to any noisy transfer, we assume the existence of a black-box that takes a bit from Alice, complements it with a certain probability, and sends the result to Bob. We also use some computational cryptographic reductions in the paper so that the cheating capability of a malicious party can be bounded within a certain range, mostly known as **slightly unfair behavior**. (ii) **Structural reduction:** A structural reduction means that we bound a general interaction between all parties in a communication system or a network to a weaker to limited task.

## III. RELATED WORKS AND KNOWN RESULTS

### A. Limits of secure two-party computations

1) **Yao’s Millionaires’ Problem:** One of the foundational problems that introduced the concept of secure two-party computation is Yao’s Millionaires’ Problem. In this thought experiment, proposed by Andrew Yao in 1982, two millionaires wish to determine who is richer without revealing their actual wealth to each other. Formally, each party holds a private input—Alice has $x$ and Bob has $y$—and they aim to compute the function $f(x, y) = (x > y)$ without disclosing any other information about $x$ or $y$. To solve this problem, Yao introduced the garbled circuit technique, where one party constructs a Boolean circuit representing the function and encrypts (or “garbles”) its components. The other party evaluates the garbled circuit using encrypted inputs obtained via oblivious transfer, ultimately learning only the output. This protocol demonstrated the feasibility of general secure computation between two parties and laid the groundwork for modern cryptographic protocols in this domain [26].

2) **Blum’s Fair Coin Tossing:** Blum’s seminal fair coin tossing (FCT) protocol [17] represents one of the earliest cryptographic solutions for enabling two distrustful parties to agree on a random bit in a fair manner. Consider a scenario where Alice and Bob must decide who gets to use their shared home office during a scheduling conflict. Since neither party trusts the other to flip a coin honestly, they employ a cryptographic mechanism to simulate a fair coin toss. Each party independently selects a random value and commits to it using a secure hash function, effectively locking in their choice without revealing it. Once both commitments are exchanged (e.g., over a public channel), they disclose their original values. The XOR or sum of the two values determines the winner: for example, if the result is even, Alice wins; if odd, Bob does. Crucially, the commitment phase prevents either party from altering their input based on the other’s choice, thereby ensuring fairness. If one party refuses to open their commitment or submits an invalid reveal, they automatically lose the toss, which discourages dishonest behavior. This mechanism exemplifies rational fairness in distributed systems, where the optimal strategy for both participants is to follow the protocol faithfully.

Although this game-theoretic interpretation of fairness aligns naturally with rational behavior, it has received comparatively little attention in the mainstream literature on secure MPC [26–28]. Instead, the field has gravitated toward a more rigorous and adversarially robust notion known as *unbiasability*. Under this definition, the protocol must guarantee that no coalition of malicious participants can skew the outcome of the computation (in particular, a coin toss) regardless of their strategy. Blum’s original coin-tossing scheme does not meet this stronger requirement; while it prevents a player from biasing the outcome in their favor, it does not fully eliminate the possibility of adversarial influence. The concept of unbiasability has been extensively investigated in cryptographic research, where it is well-established that achieving it requires an honest majority among participants [28, 29]. Notably, Cleve’s impossibility result [16] proves that in the presence of a dishonest minority comprising half or more of the parties, unbiasable coin tossing becomes fundamentally unachievable.

3) **Cleve’s Impossibility Result for Fair Coin Tossing:** Consider the task of two parties, Alice and Bob, jointly generating a uniformly random bit $b \in \{0, 1\}$ without relying on a trusted third party. The desired properties of such a protocol are: (i) if both participants behave honestly, they agree on the same random bit $b$; and (ii) if one participant is dishonest, they cannot significantly bias the outcome in their favor. The challenge lies in designing a protocol that upholds these guarantees under adversarial conditions. Cleve’s seminal impossibility result demonstrates that in the two-party setting, achieving strong fairness is impossible. The core intuition is based on the observation that in any interactive protocol, one party can abort at an advantageous point in the execution. For example, a malicious Bob may monitor the progress of the protocol, and at each step, estimate the conditional probability distribution of the final output. If he determines that aborting the protocol at a particular round yields a more favorable distribution—e.g., increasing the likelihood that the output is a certain bit—he can halt execution strategically. This potential for selective aborting allows a dishonest party to introduce bias, even if only indirectly. Cleve formalized this insight by proving that in any two-party protocol for coin tossing, one party can skew the distribution of the final bit beyond what would be possible in an ideal, unbiased setting. Thus, perfect fairness, where no party can gain an advantage through deviation is fundamentally unachievable when one participant may behave maliciously.

Cleve’s FCT protocol: Bob begins the protocol by generating $r$ pairs of public and private keys: $(K_1, T_1), (K_2, T_2), \ldots, (K_r, T_r)$ where $r$ denotes the number of rounds and serves as a security parameter. Each $K_i$ is a public key associated with round $i$, and $T_i$ is the corresponding private (trapdoor) key. A trapdoor function $F$ is employed (a one-way function that is easy to evaluate but computationally difficult to invert without knowledge of the trapdoor). Additionally, Bob selects $r$ random bits $x_1, x_2, \ldots, x_r$, and sends the following to Alice: (i) the sequence of public keys $K_1, K_2, \ldots, K_r$, and (ii) the encrypted random bits $F_{K_1}(x_1), F_{K_2}(x_2), \ldots, F_{K_r}(x_r)$, where $F_{K_i}(x_i)$ denotes the application of $F$ using public key $K_i$ to encrypt bit $x_i$. For each round $i = 1, 2, \ldots, r$, the protocol proceeds as follows: Alice selects a random bit $y_i \in \{0, 1\}$ and sends it to Bob. In response, Bob reveals the corresponding private key $T_i$. Alice then verifies that $T_i$ is indeed the correct trapdoor for $K_i$. If the verification fails, Alice replaces the unrevealed bits $x_i, x_{i+1}, \ldots, x_r$ with random values and proceeds with the rest of the protocol. Once all $r$ rounds are completed, both parties compute the XORs $x_1 \oplus y_1, x_2 \oplus y_2, \ldots, x_r \oplus y_r$, and determine the majority value among them. This majority bit serves as the final output of the protocol—a jointly generated random bit agreed upon by both parties.

As Cleve’s analysis demonstrates, a malicious Alice has limited capacity to influence the outcome, since she lacks access to Bob’s random bits $x_i$ during the protocol execution. Due to the one-way nature of the trapdoor function $F$, and her inability to invert $F_{K_i}(x_i)$ without the corresponding trapdoor $T_i$, she cannot predict or bias the XOR outcomes before the trapdoors are revealed. On the other hand, a malicious Bob has more leverage: he may abort the protocol prematurely by withholding the private key $T_i$ at some round $i$. In response, Alice replaces the unknown values $x_i, x_{i+1}, \ldots, x_r$ with random bits, thereby introducing uncertainty into the final majority computation. However, Cleve shows that even in the worst-case scenario, this form of selective aborting can only shift the outcome of the majority vote by a small amount—specifically, the bias introduced is upper bounded by $O(\frac{1}{\sqrt{r}})$. Moreover, he provides a lower bound of $\frac{1}{2^r}$, establishing that some nonzero bias is unavoidable in any finite-round protocol. This result implies that perfect fairness—defined as zero bias—is unattainable for two-party coin-tossing protocols with a finite number of rounds. As the number of rounds $r$ increases, the bias diminishes but never vanishes entirely.

4) **From OT to FCT:** To illustrate the connection between OT and FCT, we present a simple protocol that uses OT to generate a shared random bit $b \in \{0, 1\}$ in a manner that limits the ability of either party to bias the outcome, even in the presence of malicious behavior. The protocol proceeds as follows: Alice begins by choosing two independent random bits $m_0, m_1 \in \{0, 1\}$. Bob selects a random bit $z \in \{0, 1\}$, which serves as his choice in the OT protocol. Through the execution of the OT, Bob learns only the bit $m_z$, while gaining no information about $m_{1-z}$. Simultaneously, Alice remains unaware of Bob’s selection bit $z$. After the OT phase, both parties locally compute values that contribute to the shared random bit. Alice calculates $b_A = m_0 \oplus m_1$, and Bob computes $b_B = m_z \oplus z$. The final output is defined as $b = b_A \oplus b_B$, which both parties can agree upon after exchanging their local values. Specifically, they verify the consistency of their computations by checking whether $b_A \oplus b_B = 0$. If the check fails, the protocol is aborted. This construction ensures that neither Alice nor Bob can significantly influence the outcome. Alice cannot bias the bit $b$ since she has no knowledge of Bob’s choice $z$, while Bob cannot bias $b$ because he learns only one of the two bits selected by Alice, and remains ignorant of the other. Nevertheless, the protocol does not achieve perfect fairness. As established by Cleve’s impossibility result, any finite-round protocol can be susceptible to slight bias: a malicious party may abort at a strategic point or manipulate their input to influence the final result. To mitigate this, the protocol can be repeated multiple times, with the final bit determined by the majority outcome. While this repetition reduces the overall bias, it cannot eliminate it entirely.

### B. OT from the perspective of information theory

1) **In praise of noise:** As widely recognized in information theory, noise refers to any random or unpredictable interference that disrupts signal transmission. This interference may take the form of physical disturbances (such as static on a radio) or conceptual ambiguities (such as misinterpretation in language). Fundamentally, noise embodies uncertainty or entropy—the unpredictable component of a system that contributes to its complexity. It represents a manifestation of disorder or randomness within communication systems, highlighting the inherent imperfections of real-world transmission processes.

Claude Shannon’s Noisy Channel Coding Theorem [30] demonstrates that reliable communication is achievable even in the presence of noise, provided the information transmission rate remains below a critical limit known as *channel capacity*. This foundational result reveals that noise, while disruptive, is not entirely detrimental; rather, it can be mitigated through appropriate encoding schemes and the strategic use of redundancy. Shannon’s insight challenges the notion that randomness and disorder are purely negative phenomena, illustrating that they can be harnessed within structured frameworks to preserve the integrity of information.

Within the framework of Shannon’s theory, noise is intimately linked to entropy, which quantifies the uncertainty or unpredictability in a system. A system with high entropy exhibits greater disorder and reduced predictability. Noise elevates entropy by introducing ambiguity into the transmitted message, thereby increasing the uncertainty faced by the receiver.

In a broader conceptual sense, entropy serves as a metaphor for disorder or chaos in the universe, with far-reaching implications for how we understand complexity and the emergence of order. Information theory’s treatment of noise thus intersects with deeper philosophical considerations about the relationship between chaos and structure. Michel Serres, in particular, explores this theme through the metaphor of the *parasite* [31], suggesting that noise is not merely disruptive but can serve as a generative force. According to Serres, noise introduces feedback and transformation within communicative systems, acting as a catalyst for innovation and the emergence of new meanings. Importantly, one of the most profound insights drawn from noise is its role in enabling the possibility of secure communication. In the sense of OT, now we know why OT can not be obtained from scratch and the existence of noisy resources is a crucial condition for OT, as well as other cryptographic primitives. When no noisy resource is available, the system lacks the necessary randomness to obfuscate the client’s actions from the servers. This absence of noise leads to a situation of complete determinism, where all parties can eventually deduce the hidden information, violating the principles of OT.

2) **OT capacity:** As we stated before, a few research works consider the problem of OT from the information theory perspective. Nascimento and Winter [6, 13] simplified the problem of a general noisy correlation (a point-to-point channel) by reducing it to a Slightly Unfair Noisy Symmetric Basic Correlation (SU-SBC). They demonstrated that any non-perfect noisy

point-to-point channel or correlation can be transformed into a Slightly Unfair Noisy Channel/Correlation (SUNC/SUCO), and that any SUNC/SUCO can be used to implement a specific SU-SBC. Ultimately, they showed that in this reduced framework, it is possible to achieve $(\frac{1}{n})-OT^k$ (1-out-of-$m$ OT with strings’ length equal $k$) at a positive rate, assuming that the sender behaves in an honest-but-curious manner. These papers are significant for several reasons:

1) They introduce the concept of the oblivious transfer capacity of a DMC, defined as the supremum of all achievable rates $R$ such that $\frac{k}{n} \ge R - \gamma$, where $\gamma > 0$ and $n$ is the number of channel uses. A positive number $R$ is an achievable OT rate for a given DMC if for $n \rightarrow \infty$ there exist $(n, k)$ protocols with $\frac{k}{n} \rightarrow R$ such that protocols are correct and secure.
2) They also address a malicious model where a malicious player can deviate arbitrarily from the channel statistics in up to $\delta n$ instances. In such cases, the deviating player will be detected by the other party with a certain probability.

Ahlswede and Csiszár analyzed the problem under the honest-but-curious model [5]. They derived a general upper bound for the OT capacity of a point-to-point DMC, which aligns with the lower bound established by Nascimento and Winter [6]. Furthermore, when reduced to a specific erasure channel, they demonstrated a lower bound on the OT capacity. Thus, it remains uncertain what the exact OT capacity of a noisy DMC is in general. In practice, for general channels, a potential way is to first convert the channel into a Generalized Erasure Channel (GEC) via alphabet extension and erasure emulation, followed by the application of a general construction for GEC. As a starting point, we begin with a protocol by Ahlswede and Csiszár originated from the general two-party protocol of Section II:

**Two-Party OT Protocol [5]:** Consider the following two-party secure computation in the sense of OT: Alice has two strings $M_0$ and $M_1$ and aims to send them over the noisy point-to-point channel $W : \mathcal{X} \rightarrow \mathcal{Y}$ to Bob. Bob has to choose one of them by inputting a bit of $Z \in \{0, 1\}$ to the channel. Alice should be unaware of the unselected string, while Bob has only one string at the end of the protocol ($M_Z$). Suppose the main channel is an erasure channel assisted by a noiseless channel with unlimited capacity. The OT capacity in this setup is given by $\min(p, 1 - p)$, where $p$ is the erasure probability [5, 14]. Let $r < \min(p, 1 - p)$. The protocol by Ahlswede and Csiszár [5], based on a technique originally introduced for a BSC in [9, Sec. 6.4], proceeds as follows: Alice starts by transmitting a sequence $X^n = X^n \sim \text{Bernoulli}(\frac{1}{2})$ of i.i.d. bits over the channel. Bob observes the channel output $Y$. Let $E$ denote the set of indices at which $Y$ is erased, and let $\bar{E}$ represent the set of indices where $Y$ is not erased. If $|E| < nr$ or $|\bar{E}| < nr$, Bob aborts the protocol, as there are not enough erased or unerased bits to complete the protocol. From $E$, Bob randomly selects a subset $S_Z$ of cardinality $nr$. From $\bar{E}$, Bob randomly picks a subset $S_{\bar{Z}}$, also of size $nr$. Bob then shares the sets $S_0$ and $S_1$ with Alice via the public channel, where $S_0$ and $S_1$ are either $S_Z$ and $S_{\bar{Z}}$, respectively or vice versa. Alice cannot determine which of $S_0$ and $S_1$ corresponds to $E$ (erased positions) and which to $\bar{E}$ (non-erased positions) due to the independent nature of channel erasures. Using $S_0$ and $S_1$, Alice computes keys $X|_{S_0}$ and $X|_{S_1}$ and employs these keys to encrypt her strings, which she sends to Bob over the public channel as $M_0 \oplus X|_{S_0}$ and $M_1 \oplus X|_{S_1}$. Bob, who only knows the sequence corresponding to $E$, can decrypt only one of these encrypted strings, depending on whether $S_Z = S_0$ or $S_Z = S_1$. This enables Bob to retrieve one of the two keys, $M_Z$, while he learns nothing about the other key, $M_{\bar{Z}}$. If $X$ is not uniformly distributed over $\{0, 1\}$, the strings $X|_{S_j}, j \in \{0, 1\}$ are not directly suitable as encryption keys. They need to be transformed into binary strings of length $k < nr$ with a distribution approximately uniform over $\{0, 1\}^k$. It is well-known that for any $\delta > 0$, when $n$ is large, there exists a mapping $\kappa : \{0, 1\}^{nr} \rightarrow \{0, 1\}^k$ with $k = n(H(X) - \delta)$ such that $k - H(\kappa(X^n))$ is exponentially small.

In [32], the author extends the above protocol to pairwise oblivious transfer over a noiseless binary adder channel involving two senders and one receiver, assuming they are non-colluding and honest-but-curious. Each sender has two strings, and Bob has to choose one string from each sender while the unselected strings are hidden from his view. In this system, the output is defined as the sum of the inputs, $Y = X_1 + X_2$, commonly referred to as the Binary Erasure Multiple Access Channel (BE-MAC). This channel uniquely determines the inputs except when they differ, in which case one can not identify the inputs with certainty, effectively resulting in an erasure. Specifically, erasures occur in two out of four possible input scenarios. The OT capacity of this channel is shown to be $R_1 + R_2 \le \max_{P_{X_1} P_{X_2}} H(X_1, X_2|Y) = \frac{1}{2}$.

## IV. SYSTEM MODEL

We assume that the availability of noise is provided in two main forms. Also, we consider the main OT channel with two senders and one receiver:

1) **Discrete Memoryless MAC:** A two-user MAC $W : \mathcal{X}_1 \times \mathcal{X}_2 \rightarrow \mathcal{Y} : (X_1 \times X_2, p(y|x_1, x_2), \mathcal{Y})$, connecting three parties, Alice-1, Alice-2 and Bob, which can be used $n$ times. For an input sequence $\mathbf{x}^n = x_{i, 1} x_{i, 2} \ldots x_{i, n}$, the output distribution over $\mathcal{Y}^n$ is given by:
$$W_{\mathbf{x}_1^n \mathbf{x}_2^n}^n = W_{x_{1, 1} x_{2, 1}} W_{x_{1, 2} x_{2, 2}} \ldots W_{x_{1, n} x_{2, n}}$$

2) **i.i.d. Realizations:** A tuple of random variables $(X_1, X_2, Y)$, where Alice-$i$ sends $X_i$ and Bob receives $Y$. The distribution of these variables is given by $P_{X_1 X_2 Y}$, defined over the finite sets $\mathcal{X}_i$ and $\mathcal{Y}$.

In both cases, the alphabets $\mathcal{X}_1, \mathcal{X}_2$, and $\mathcal{Y}$ are finite.

A key concept when analyzing noisy channels is the idea of redundant symbols [33]. We have the following definition for DM-MACs, presented in [6] for the point-to-point channel.

**Definition 4.** A two-sender DM-MAC $W(y|x_1, x_2)$, characterized by its conditional probability distribution $W(y|x_1, x_2)$ of the output $y$ given inputs $x_1$ from Alice-1 and $x_2$ from Alice-2, is said to be nonredundant if none of its output distributions $W_{x_1, x_2}(y)$ (induced by fixed inputs $(x_1, x_2)$) can be expressed as a convex combination of the other output distributions. Formally, this means:
$$\forall i \in \mathcal{T} \setminus \{(x_1, x_2)\}, \forall P(x_1, x_2) \text{ such that } P\{i \in \mathcal{T} \setminus \{(x_1, x_2)\}\} = 0, \quad W_{i \in \mathcal{T} \setminus \{(x_1, x_2)\}} \ne \sum_{x_1, x_2} P(x_1, x_2) W_{x_1, x_2},$$
for any possible distinct input pairs $\mathcal{T} = \{(x_1, x_2), (x'_1, x'_2), (x''_1, x''_2), (x'''_1, x'''_2)\} \in \mathcal{X}_1 \times \mathcal{X}_2$.

*   **Geometric Interpretation:** In geometric terms, each output distribution $W_{x_1, x_2}$ is a distinct extremal point of the polytope $\mathcal{W} = \text{conv}\{W_{x_1, x_2} : (x_1, x_2) \in \mathcal{X}_1 \times \mathcal{X}_2\}$, where $\mathcal{X}_1$ and $\mathcal{X}_2$ are the input alphabets of Senders 1 and 2, respectively. The polytope $\mathcal{W}$ represents the convex hull of all output distributions over the probability simplex on the output alphabet $\mathcal{Y}$.
*   **Constructing a Nonredundant MAC:** To construct a nonredundant version of the MAC, $W(y|x_1, x_2)$, we can remove all input pairs $(x_1, x_2)$ for which the output distribution $W_{x_1, x_2}$ is not extremal. This results in a reduced set of input pairs for which $W_{x_1, x_2}$ forms the set of extremal points of $\mathcal{W}$. The original MAC can still be simulated using the reduced MAC by reconstructing the removed distributions $W_{x_1, x_2}$ as convex combinations of the extremal distributions from $\mathcal{W}$. This process ensures that the MAC retains its original operational capacity while simplifying its representation by eliminating redundancy in its input space.

A more intuitive definition based on the correlations is presented below.

**Definition 5.** Consider a two-user DM-MAC characterized by random variables $X_1, X_2$ (inputs from the two senders) and $Y$ (output), with joint distribution/correlation $P(X_1, X_2, Y)$. The correlation is said to be nonredundant if:
*   For any possible distinct input pairs $\mathcal{T} = \{(x_1, x_2), (x'_1, x'_2), (x''_1, x''_2), (x'''_1, x'''_2)\} \in \mathcal{X}_1 \times \mathcal{X}_2$:
    $$\text{Pr} \{Y | (X_1, X_2) = i \in \mathcal{T}\} \ne \text{Pr} \{Y | (X_1, X_2) = j \in \mathcal{T} \setminus \{i\}\}.$$
*   Symmetrically, the above condition also applies to redundancy in $X_1$ (for fixed $X_2$) or $X_2$ (for fixed $X_1$), similarly to $Y$.

**Resolving Redundancy:** If there is redundancy, the MAC can be made nonredundant by collapsing indistinguishable input pairs $(x_1, x_2)$ that fail the above inequality into a single equivalent pair. Similarly, redundant output symbols $y_1, y_2$ can be merged into one.

**Geometric Interpretation:** In the MAC context, redundancy occurs when the joint distribution $P(Y | X_1, X_2)$ does not map injectively over distinct input combinations $(x_1, x_2)$. This can be resolved by projecting to the set of unique conditional distributions $\text{Pr}(Y|X_1, X_2)$, thereby defining an equivalent nonredundant MAC.

**Definition 6.** For a DM-MAC with input random variables $X_1$ and $X_2$, and output $Y$, we define **perfect correlation** as follows:
The MAC is perfectly correlated if, given $Y$, both $X_1$ and $X_2$ can be determined with certainty. This implies:
$$H(X_1, X_2|Y) = 0.$$
Similarly, a MAC can be called **perfect** if its joint output distributions (conditioned on input pairs) have mutually disjoint support. Specifically:
Given the output $Y$, the pair $(X_1, X_2)$ is uniquely determined. Formally, this means that for all $y \in \mathcal{Y}$, there exists at most one pair $(x_1, x_2)$ such that $P_{Y|X_1, X_2}(y|x_1, x_2) > 0$.

As proved in [6] for point-to-point channels, a perfect DM-MAC (even after removing the redundancy) cannot be used for oblivious transfer, even against passive adversaries. This relates to the concept of noise and the emergence of noisy resources for cryptographic intents. We know that the noise produces uncertainty or entropy. It is the unpredictable aspect of a system that adds complexity. Conceptually, it can be seen as the manifestation of disorder or randomness in communication systems, emphasizing the non-perfect nature of the real world. Noise plays a crucial role in securing communication by hindering an eavesdropper’s ability to extract meaningful information from the transmitted message. In a noisy channel, the inherent noise limits the amount of information an unauthorized party can access, regardless of their computational power. This concept is central to Wyner’s wiretap channel model [34], which demonstrates how noise can be leveraged to ensure that the legitimate receiver decodes the message accurately. In contrast, experiencing additional noise, the eavesdropper cannot gather sufficient information to reconstruct the message. As is clear, a perfect channel can be simulated by a noiseless channel where the input(s) can be obtained with certainty from the channel output(s). Obviously, such channels cannot be used for cryptographic intents with unconditional security (information-theoretic secrecy).

As is proved in [5, 14], the OT capacity of the point-to-point erasure channel (BEC) with erasure probability $\frac{1}{2}$ is equal $\frac{1}{2}$. Here, we want to investigate whether the OT is possible over a special BE-MAC. We introduce the channel as a correlation

between the senders (Alice-1, Alice-2) and the receiver (Bob). Before that, we delve deeper into the *unfairness* in the channel/correlation model. Damgård et. al. [18], introduced unfairness so that an unfair player could change the communication channel parameters within a certain range. In [6], this concept is limited so that an unfair player who deviates from the channel statistics in $\delta n$ positions will be caught by the other party with probability $> 1 - C_1 \exp(-C_2 \delta^2 n)$, where $C_1$ and $C_2$ are two small positive numbers. There are two senders in our channel model. The concept of unfairness can also be extended to the MAC model. To control the fairness of other players in the last $n$ rounds, all players have access to a test unit. When both senders send a pair symbol to Bob over the channel, Bob will ask for the input symbols with probability $\frac{1}{2}$. He will tell both senders his output when he has received the senders’ response. If Alice-$i$ is unfair, she tells her input wrong; if Bob is unfair, he tells his output wrong. The test between players is to check out the samples after $n$ uses of the channel for joint typicality relative to $P_{X_1 X_2 Y}$.

**Definition 7.** Consider a DM-MAC characterized by random variables $X_1, X_2$ uniformly distributed over $\{0, 1\}$ (inputs from the two senders) and $Y$ (output), with joint distribution $P(X_1, X_2, Y)$. Let $p = \frac{1}{2}$ and $\mathcal{Y}$ of $Y$ be partitioned into two disjoint sets: $\mathcal{Y} = \mathcal{E}_{10} \cup \mathcal{E}_{01}$ of non-zero probability under the distribution of $Y$. The channel/correlation has the following properties:

*   For all $y_{10} \in \mathcal{E}_{10}, y_{01} \in \mathcal{E}_{01}$ and $x_i \in \{0, 1\}, i \in \{1, 2\}$,
    $$\text{Pr}\{Y = y_{10}|X_1 = x_1, X_2 = x_2\} = \text{Pr}\{Y = y_{01}|X_1 = x_1, X_2 = x_2\} = \frac{1}{2}$$
*   $\mathcal{E}_{10}$ is the set of received pairs $(X_1 = x_1, X_2 = e)$, and $\mathcal{E}_{01}$ is the set of received pairs $(X_1 = e, X_2 = x_2)$, where $e$ is an erased bit.

Now, we present a more general SU-SBC over DM-MAC.

**Definition 8.** [SU-SBC$_{p, W, W'}$] Consider a DM-MAC characterized by random variables $X_1, X_2$ uniformly distributed over $\{0, 1\}$ (inputs from the two senders) and $Y$ (output), with joint distribution $P(X_1, X_2, Y)$. Let $0 < p < 1$ and $\mathcal{Y}$ of $Y$ be partitioned into five disjoint sets: $\mathcal{Y} = \mathcal{U}_{11} \cup \mathcal{U}_{10} \cup \mathcal{E} \cup \mathcal{U}_{01} \cup \mathcal{U}_{00}$ of non-zero probability under the distribution of $Y$. Note that $\mathcal{E} = \mathcal{E}_{00} \cup \mathcal{E}_{10} \cup \mathcal{E}_{01}$. A symmetric basic correlation (SBC) over this channel can be defined as follows:

*   For all $y \in \mathcal{E}_{00}, \text{Pr}\{Y = y|X_1 = 1, X_2 = 1\} = \text{Pr}\{Y = y|X_1 = 1, X_2 = 0\} = \text{Pr}\{Y = y|X_1 = 0, X_2 = 1\} = \text{Pr}\{Y = y|X_1 = 0, X_2 = 0\} = (1 - p)^2$.
*   For all $y \in \mathcal{E}_{10} \cup \mathcal{E}_{01}, \text{Pr}\{Y = y|X_1 = 1, X_2 = 1\} = \text{Pr}\{Y = y|X_1 = 1, X_2 = 0\} = \text{Pr}\{Y = y|X_1 = 0, X_2 = 1\} = \text{Pr}\{Y = y|X_1 = 0, X_2 = 0\} = 2p(1 - p)$.
*   (Symmetry) For all $y_{11} \in \mathcal{U}_{11}, y_{10} \in \mathcal{U}_{10}, y_{01} \in \mathcal{U}_{01}, y_{00} \in \mathcal{U}_{00}$, and $x_i \in \{0, 1\}, i \in \{1, 2\}$
    $$\text{Pr}\{Y = y_{i \in \mathcal{T}'}|(X_1 X_2) = j \in \mathcal{T}'\} = \text{Pr}\{Y = y_{i' \in \mathcal{T}' \setminus \{i\}}|(X_1 X_2) = j' \in \mathcal{T}' \setminus \{j\}\},$$
    for $\mathcal{T}' = \{00, 10, 01, 11\}$.
*   (Non-redundancy) For all $y_{11} \in \mathcal{U}_{11}, y_{10} \in \mathcal{U}_{10}, y_{01} \in \mathcal{U}_{01}, y_{00} \in \mathcal{U}_{00}$, and $x_i \in \{0, 1\}, i \in \{1, 2\}$
    $$\text{Pr}\{Y = y_{i \in \mathcal{T}'}|(X_1 X_2) = i\} > \text{Pr}\{Y = y_{i}|(X_1 X_2) = j \in \mathcal{T}' \setminus \{i\}\},$$
    for $\mathcal{T}' = \{00, 10, 01, 11\}$.
*   $\text{Pr}\{Y \in \mathcal{E}\} = 1 - p^2$.

From the senders’ point of view, it looks like uniform inputs to a DM-MAC, while for Bob, it looks like the output of a distinguishable mixture of three channels: a complete erasure MAC $W''(y|x_1, x_2): \{0, 1\} \times \{0, 1\} \rightarrow \mathcal{E}_{00}$, a partial erasure channel $W'(y|x_1, x_2): \{0, 1\} \times \{0, 1\} \rightarrow \mathcal{E}_{01} \cup \mathcal{E}_{10}$ (Definition 7) which erases either $x_1$ or $x_2$, and a channel $W : \{0, 1\} \times \{0, 1\} \rightarrow \mathcal{U}_{11} \cup \mathcal{U}_{10} \cup \mathcal{U}_{01} \cup \mathcal{U}_{11}$, with conditional probabilities $W(y|x_1, x_2) = \text{Pr}\{Y = y|X_1 = x_1, X_2 = x_2\}$.
If Bob finds $y \in \mathcal{E}_{00}$, he has no information at all about the inputs. If Bob finds $y \in \mathcal{E}_{01}$, he has no information at all about Alice-1’s input. Similarly, if he finds $y \in \mathcal{E}_{10}$, he has no information at all about Alice-2’s input. For $y \in \mathcal{U}_i$, where $i \in \mathcal{T}' = \{00, 01, 10, 11\}$, he has a (more or less weak) indication that $x_1 x_2 = i$, as the likelihood for $x_1 x_2 = j \in \mathcal{T}' \setminus \{i\}$ is smaller.

The correlation is clearly fully characterized by $p, W$, and $W'$. Hence, we denote this distribution as $\text{SBC}_{p, W, W'}$. If it is used slightly unfairly, we denote it as $\text{SU-SBC}_{p, W, W'}$. We reduce the $\text{SU-SBC}_{p, W, W'}$ to the case in which both inputs are erased or both of them are decoded. The new SU-SBC is demonstrated by $\text{SU-SBC}_{p, W}$ since the sub-channel $W'$ defined in Definition 7 is removed.

**Definition 9.** [SU-SBC$_{p, W}$] Consider a DM-MAC characterized by random variables $X_1, X_2$ uniformly distributed over $\{0, 1\}$ (inputs from the two senders) and $Y$ (output), with joint distribution $P(X_1, X_2, Y)$. Let $0 < p < 1$ and $\mathcal{Y}$ of $Y$ be partitioned into five disjoint sets: $\mathcal{Y} = \mathcal{U}_{11} \cup \mathcal{U}_{10} \cup \mathcal{E} \cup \mathcal{U}_{01} \cup \mathcal{U}_{00}$ of non-zero probability under the distribution of $Y$. A symmetric basic correlation (SBC) over this channel can be defined as follows:

*   For all $y \in \mathcal{E}, \text{Pr}\{Y = y|X_1 = 1, X_2 = 1\} = \text{Pr}\{Y = y|X_1 = 1, X_2 = 0\} = \text{Pr}\{Y = y|X_1 = 0, X_2 = 1\} = \text{Pr}\{Y = y|X_1 = 0, X_2 = 0\} = 1 - p$.
*   (Symmetry) For all $y_{11} \in \mathcal{U}_{11}, y_{10} \in \mathcal{U}_{10}, y_{01} \in \mathcal{U}_{01}, y_{00} \in \mathcal{U}_{00}$, and $x_i \in \{0, 1\}, i \in \{1, 2\}$,
    $$\text{Pr}\{Y = y_{i \in \mathcal{T}'}|X_1 X_2 = j \in \mathcal{T}'\} = \text{Pr}\{Y = y_{i' \in \mathcal{T}' \setminus \{i\}}|X_1 X_2 = j' \in \mathcal{T}' \setminus \{j\}\},$$
    for $\mathcal{T}' = \{00, 10, 01, 11\}$.
*   (Non-redundancy) For all $y_{11} \in \mathcal{U}_{11}, y_{10} \in \mathcal{U}_{10}, y_{01} \in \mathcal{U}_{01}, y_{00} \in \mathcal{U}_{00}$, and $x_i \in \{0, 1\}, i \in \{1, 2\}$,
    $$\text{Pr}\{Y = y_{i \in \mathcal{T}'}|X_1 X_2 = i\} > \text{Pr} \{Y = y_{i}|X_1 X_2 = j \in \mathcal{T}' \setminus \{i\}\},$$
    for $\mathcal{T}' = \{00, 10, 01, 11\}$.
*   $\text{Pr}\{Y \in \mathcal{E}\} = 1 - p$.

From now on, we work only with $\text{SU-SBC}_{p, W}$. We demonstrate how to reduce the general case of non-perfect MACs to $\text{SU-SBC}_{p, W, W'}$. Since redundant channels or correlations can always be transformed into nonredundant ones, we will henceforth assume that all noisy resources under consideration are nonredundant.

**Proposition 7.** Given a non-perfect noisy DM-MAC $W : \mathcal{X}_1 \times \mathcal{X}_2 \rightarrow \mathcal{Y}$, it can be utilized to obtain a certain SUCO. Similarly, a noisy correlation shared among Alice-1, Alice-2, and Bob can be used to implement an SUCO.

**Proof.** For any given distribution $P_{X_i}$ possessed by Alice-$i$, $i \in \{1, 2\}$, the channel generates a joint distribution $P_{X_1 X_2 Y}$. Alice-$i$ sends independent realizations of $X_i$ according to her probability distribution over the channel. Note that the input joint distribution is $P_{X_1 X_2}(x_1, x_2) = P_{X_1}(x_1) P_{X_2}(x_2)$ due to independence of senders’ probability distributions. For each received pair message, Bob will ask, with probability $\frac{1}{2}$, for the input symbols, and after receiving, he will tell the senders his received message. If one or both senders are cheaters, then they will give wrong information to Bob, and if Bob is a cheater, then he will give wrong information to the senders. Lemma 6 shows that the probability of cheating (in more than $\delta n$ positions) tends to zero as $n \rightarrow \infty$. In other words, a cheater can not deviate from the channel statistics in $\delta n$ positions without being detected because the joint typicality test fails. This shows that any non-perfect noisy MAC $W : \mathcal{X}_1 \times \mathcal{X}_2 \rightarrow \mathcal{Y}$ can be used to obtain a certain SUCO/SUNC. $\square$

**Proposition 8.** Given a non-perfect SUCO $P_{X_1 X_2 Y}$, one can use it to implement a certain $\text{SU-SBC}_{p, W}$.

**Proof.** Given that the slightly unfair correlation is non-perfect, it follows that after removing all redundancy, $H(Y|X_1, X_2) > 0$. Therefore, there exist at least four distinct pairs of symbols $((a, a) \ne (b, b) \ne (a, b) \ne (b, a)) \in \mathcal{X}_1 \times \mathcal{X}_2$ such that:
$$\{y : \text{Pr}(Y = y|(X_1, X_2) = (a, a)) > 0 \text{ and}$$
$$\text{Pr}(Y = y|(X_1, X_2) = (a, b)) > 0 \text{ and}$$
$$\text{Pr}(Y = y|(X_1, X_2) = (b, a)) > 0 \text{ and}$$
$$\text{Pr}(Y = y|(X_1, X_2) = (b, b)) > 0\} \ne \emptyset,$$
since otherwise, the distributions' equivocation (equivocation about the inputs given the output) would be zero.
Our protocol to achieve $\text{SU-SBC}_{p, W}$ closely follows the approach outlined in [6, 8]. It uses the source coding protocol $P_{X_1 X_2 Y}$ twice for two independent realizations. Let:
$$\text{Pr} \left( (\mathbf{X}_1, \mathbf{X}_2)(\mathbf{X}'_1, \mathbf{X}'_2) \cup (\mathbf{X}'_1, \mathbf{X}_2)(\mathbf{X}_1, \mathbf{X}'_2) \cup (\mathbf{X}_1, \mathbf{X}'_2)(\mathbf{X}'_1, \mathbf{X}_2) \in \mathcal{L} \right) = \frac{\alpha}{3},$$
where $\mathcal{L} = \{(a, a)(b, b), (a, a)(a, b), (a, a)(b, a), (b, b)(a, a), (b, b)(a, b), (b, b)(b, a), (a, b)(b, b), (a, b)(a, a), (a, b)(b, a), (b, a) (a, a), (b, a)(a, b), (b, a)(b, b)\}$. As the protocol progresses, if $\{(\mathbf{X}_1, \mathbf{X}_2)(\mathbf{X}'_1, \mathbf{X}'_2), (\mathbf{X}'_1, \mathbf{X}_2)(\mathbf{X}_1, \mathbf{X}'_2), (\mathbf{X}_1, \mathbf{X}'_2)(\mathbf{X}'_1, \mathbf{X}_2)\} \notin \mathcal{L}$, the senders inform Bob of the value and discard the pair. By the law of large numbers and the i.i.d. distribution of $\mathcal{L}$, we know that the probability of $i \in \mathcal{L}$ is equal $\frac{1}{2}$, then fewer than $\frac{1}{2} (n + \epsilon n)$ (for some $\epsilon > 0$) realizations of $P_{X_1 X_2 Y}$ are necessary to achieve $n$ realizations of $\text{SBC}_{p, W}$. This constitutes a symmetric basic correlation because the probability of occurrence is equal for each $i \in \mathcal{L}$. Furthermore, Bob’s alphabet $\mathcal{Y} \times \mathcal{Y}$ is partitioned into $\mathcal{E}, \mathcal{U}_{00}, \mathcal{U}_{10}, \mathcal{U}_{01}, \mathcal{U}_{11}$, satisfying the definition:

*   $\mathcal{E}$ includes all pairs $(z, z')(z, z')$, where $(z, z')(z, z')$ has a positive probability.
*   $\mathcal{U}_{00}, \mathcal{U}_{10}, \mathcal{U}_{01}$ and $\mathcal{U}_{11}$ consist of transposes of one another (i.e., swapping the entries) and cannot be empty. If they were, at least one of the members in the above set would be redundant.

Bob cannot behave unfairly beyond what is provided by the SUCO $P_{X_1 X_2 Y}$. However, the senders can introduce bias by attempting actions such as repeating unfair pairs $(\mathbf{x}_1, \mathbf{x}_2)(\mathbf{x}'_1, \mathbf{x}'_2), (\mathbf{x}'_1, \mathbf{x}_2)(\mathbf{x}_1, \mathbf{x}'_2)$ or $(\mathbf{x}_1, \mathbf{x}'_2)(\mathbf{x}'_1, \mathbf{x}_2)$ for a good pair $i \in \mathcal{L}$. Suppose the senders persist with such strategies for $\delta n$ times (where $\delta > 0$). In that case, Bob can detect their bias by approximating a typicality test, using Lemma 6 and an extended version of [6, Lemma 6]. This aligns with the definition of $\text{SBC}_{p, W}$. $\square$

## V. MAIN RESULTS AND PROOFS

### A. OT limits

1) **Information-Theoretic Formulation of OT:** In the foundational works [35, 36], the information-theoretic security definition for the receiver’s privacy requires that the sender’s final view, denoted by $U = (U', R_A, C)$, be statistically independent of the receiver’s choice bit $z \in \{0, 1\}$. Here, $U'$ represents the sender’s initial state (including inputs), $R_A$ denotes the sender’s private randomness, and $C$ captures the complete public communication between the sender and receiver. However, as highlighted by Crépeau in [37], such a strong condition is generally unachievable in many practical settings. In particular, when there is a known dependency between the parties’ inputs, the sender’s view will unavoidably correlate with the receiver’s input. This is because the sender’s input $X$, which is inherently part of her view, may be statistically dependent on the receiver’s input $Z$. To address this, a more appropriate formulation requires that the sender’s view be conditionally independent of the receiver’s input, given the sender’s input. Formally, this is expressed via the Markov chain $(U', R_A, C) \rightarrow X \rightarrow Z$, which implies the conditional mutual information satisfies $I(U', R_A, C; Z | X) = 0$. Nevertheless, Ahlswede and Csiszár [5] adopt a slightly relaxed criterion, aligning with the interpretation in [35, 36], where the privacy condition is written as $I(M_0, M_1, X^n, R_A, C; Z) \rightarrow 0$. This formulation is weaker, as it allows for some residual dependence under the assumption that it diminishes asymptotically.

Formulating the security definition for the sender presents additional challenges beyond those encountered in defining receiver security [35, 36]. Several issues commonly arise in this context:

Notably, in [38, 39], the model constrains a malicious receiver to alter their input in a purely deterministic manner. That is, the effective input $Z'$—representing the bit ultimately learned by the receiver—is required to be a deterministic function of their original input $Z$. This stands in contrast to the ideal functionality of oblivious transfer, where a malicious receiver is permitted to choose $Z'$ probabilistically, potentially based on arbitrary strategies. Such a deterministic restriction limits the adversary’s capabilities in the simulation, potentially weakening the security guarantee for the sender in realistic adversarial settings.

In scenarios where a malicious receiver may alter their effective input from $z$ to $z'$ (with $z, z' \in \{0, 1\}$), [40] allows $Z'$ to depend on the sender’s input $X$. Such dependency is not permitted in the ideal model, where parties’ inputs and outputs are generated independently of each other through a trusted functionality. Furthermore, the security definition requires that the view of the malicious receiver—denoted by $(V', R_B, Y^n, C)$—be conditionally independent of the sender’s input $X$, given the receiver’s original input $Z$ and their output $M_{Z'}$. This corresponds to the Markov chain: $X \rightarrow (Z, M_{Z'}) \rightarrow (V', R_B, Y^n, C)$, which implies: $I(V', R_B, Y^n, C; X | Z, M_{Z'}) = 0$.

However, a more accurate formulation would account for the potential dependence of the receiver’s strategy on $Z'$, leading to the stronger Markov condition: $X \rightarrow (Z, Z', M_{Z'}) \rightarrow (V', R_B, Y^n, C)$. Ahlswede and Csiszár [5] propose an alternative criterion under the honest-but-curious model—where Alice and Bob follow the protocol faithfully—that requires:
$$I(Z, R_B, Y^n, C; X) = I(Z, R_B, Y^n, C; M_{\bar{Z}}),$$
and study its asymptotic behavior in the i.i.d. setting. Here, Alice’s effective input is the message pair $(M_0, M_1)$, while Bob’s input is $Z$. Although this formulation simplifies analysis under passive adversaries, it is overly permissive and can allow certain protocols that are insecure under stronger, malicious models.

In the context of OT, a malicious receiver may alter their effective input from $z$ to $z'$ for $z, z' \in \{0, 1\}$. In [40], the variable $Z'$—representing the receiver’s effective choice—may depend on the honest sender’s input $X$, which deviates from the ideal functionality where such a dependency is disallowed. Moreover, the receiver’s view, denoted by $(V', R_B, Y^n, C)$, is required to be conditionally independent of the sender’s input $X$, given the receiver’s original input $Z$ and the corresponding output $M_{Z'}$. This condition corresponds to the Markov chain: $X \rightarrow (Z, M_{Z'}) \rightarrow (V', R_B, Y^n, C)$, or equivalently, $I(V', R_B, Y^n, C; X | Z, M_{Z'}) = 0$. However, a more accurate and robust formulation accounts for the dependency of the adversary’s strategy on the full pair $(Z, Z')$, implying the stronger Markov chain: $X \rightarrow (Z, Z', M_{Z'}) \rightarrow (V', R_B, Y^n, C)$. Ahlswede and Csiszár [5] propose a different formulation in the honest-but-curious setting, where both parties follow the protocol faithfully. In this model, Alice’s input is the message pair $(M_0, M_1)$, and Bob’s input is $Z$. Their security criterion requires that: $I(Z, R_B, Y^n, C; X) \approx I(Z, R_B, Y^n, C; M_{\bar{Z}})$, and they study its asymptotic behavior under the i.i.d. assumption. While this approach simplifies the analysis for semi-honest participants, it is too permissive under malicious adversaries and may inadvertently allow protocols with security vulnerabilities.

2) **OT over a point-to-point noisy channel:**

**Definition 10.** [37] A protocol $\Pi$ securely computes $(\frac{1}{2})-OT^k$ perfectly if and only if for every pair of algorithms $\bar{A} = (A_1, A_2)$ that is admissible for protocol $\Pi$ and for all inputs $(X(m_0, m_1), Z)$ and auxiliary input $C$ (representing the total information transmitted over the noiseless public channel), $A$ produces outputs $(U, V)$ such that the following conditions are satisfied:
*   (Correctness) If both players are honest, then $(U, V) = (\Delta, M_Z)$.
*   (Security for Alice) If Alice is honest, then we have $U = \Delta$, and there exists a random variable $Z'$, such that
    $$I(U'; Z'|C, Z) = 0, \quad \text{and} \quad I(U'; V|C, Z, Z', M_{Z'}) = 0.$$
*   (Security for Bob) If Bob is honest, then we have
    $$I(U; \bar{Z}|C, U') = 0,$$
    where $X, U, V, C$ are random variables and $C$ is an additional auxiliary input (total public transmission) available to both players but assumed to be ignored by honest players. Note that $U$ (Alice's final view after completing OT protocol) and $V$ (Bob's final view after completing OT protocol) are $(U', R_A, X^n, C)$ and $(V', R_B, Y^n, C)$, respectively, where $U'$ and $V'$ in the latter form are Alice's and Bob's initial view, respectively.

**Theorem 9.** Consider a noisy DMC between Alice and Bob. OT with perfect secrecy (unconditional security) is impossible over a DM-MAC if one of the players is an unbounded cheater.

**Proof.** In Appendix D.

3) **OT over a DM-MAC:**

**Definition 11.** A protocol $\Pi$ securely computes $(\frac{1}{2})-OT^{k_1, k_2}$ between non-colluding players perfectly if and only if for every algorithm $\bar{A} = (A_1, A_2, A_3)$ that is admissible for protocol $\Pi$ and for all inputs $(\mathbf{X}_i(M_{i0}, M_{i1}), Z_i)$ and auxiliary input $C = (C_1, C_2)$ (total public transmission), $\bar{A}$ produces outputs $(U_1, U_2, V)$ such that the following conditions are satisfied:

*   (Correctness) If all players are honest, then $(U_1, U_2, V) = (\Delta, \Delta, (M_{1 Z_1}, M_{2 Z_2}))$.
*   (Security for Alice-$i$) If both senders are honest; then we have $U_i = \Delta, i \in \{1, 2\}$, and there exist two random variables (malicious Bob's effective inputs) $Z'_i, \bar{Z'}_i \in \{1, 2\}$, such that,
    $$I(X_i; Z'_i|C, Z_i)_{i \in \{1, 2\}} = 0,$$
    $$I(X_i; V|C, Z_i, Z'_i, M_{\bar{Z}'_i})_{i \in \{1, 2\}} = 0.$$
*   (Security for Bob) If Bob is honest, then we have:
    $$I(U_i; Z_i|C, X_i)_{i \in \{1, 2\}} = 0,$$
    where Alice-$i$'s final view and Bob's final view are $U_i = (U'_i, R_{A_i}, C_i)$ and $V = (V', R_B, Y^n, C_i)$, respectively, and $U'_i$ and $V'$ in the latter form are Alice-$i$'s and Bob's initial views, respectively.

**Remark 1** (Structural reduction). We have assumed two reductions from the general OT over MAC: 1- Both senders act independently, and there is no dependency between their chosen messages. 2- Both senders act honestly or maliciously.¹
Now, we aim to demonstrate that the reduced version cannot be realized from an information-theoretical perspective.

**Theorem 10.** Consider the OT setting over the DM-MAC described in Definition 11 and Remark 1. OT with perfect secrecy (unconditional security) is impossible over the DM-MAC if Bob (Or both senders) is an unbounded cheater.

**Proof.** In Appendix E.

### B. OT over the DM-MAC

Consider Alice-1, Alice-2, and Bob connected by a DM-MAC. Alice-$i, i \in \{1, 2\}$ possesses two strings of bits $m_{i 0}, m_{i 1}$, each of which with length $k_i$. They could have OT between themselves (Alice-1 $\leftrightarrow$ Bob, Alice-2 $\leftrightarrow$ Bob) as follows: They send their strings over a noisy DM-MAC to Bob, and Bob has to choose one string from the Alice-1's strings ($m_{1 z_1}$) and one string from Alice-2's strings ($m_{2 z_2}$) by inputting two bits to the channel $Z_1, Z_2$. The unselected messages should be kept hidden from Bob's view, while the selected messages should be kept hidden from Alice's view. After completing a protocol, Bob gets $M_{1 Z_1}, M_{2 Z_2}$ based on his choices, while Alice-$i$ gets nothing $\Delta$.

Now we consider a reduction from the general non-perfect MAC (Figure 2-(a)) to a MAC made up of $\text{SBC}_{p, W}$ with non-independently distributed noise².

---
$^1$ If we assume that Alice-$i$ is honest while Alice-$j$ is malicious, then OT with perfect secrecy is possible between Alice-$i$ and honest Bob due to Cleve's impossibility.
$^2$ The concept of noise could be different in OT over multi-user channels compared to a point-to-point channel. Noise can act independently over the users' links to the receiver in a MAC so that the described $\text{SU-SBC}_{p, W}$ can be defined as $\text{SU-SBC}_{p_1, p_2, W}$, where the channel $W$ in the second one has the conditional probabilities $W(y|x_1, x_2) = \frac{1}{p_1 p_2} \text{Pr}\{Y = y|X_1 = x_1, X_2 = x_2\}$. Also, the sets $\mathcal{E}_{01}$ and $\mathcal{E}_{10}$ could not be unified since their probabilities are not the same. ($1 - p_i$ is the erasure probability for Alice-$i$ messages)

$$
\begin{array}{c}
\text{Noiseless public channel} \\
C_1, C_2 \\
\uparrow \\
\text{Alice-1} \quad X_1^n \quad \xrightarrow{p(y|x_1, x_2)} \quad Y^n (M_{1Z_1}, M_{2Z_2}) \quad \text{Bob} \\
(M_{10}, M_{11}) \quad Z_1 \in \{0, 1\} \\
X_2^n \\
\text{Alice-2} \quad Z_2 \in \{0, 1\} \\
(M_{20}, M_{21}) \\
\text{(a)}
\end{array}
\qquad
\begin{array}{ccc}
\text{Alice-1} & X_1 & \xrightarrow{p} 0 \\
& & \xrightarrow{1-p} e \\
& & \xrightarrow{p} 1 \\
\text{Alice-2} & X_2 & \xrightarrow{1-p} 0 \\
& & \xrightarrow{p} e \\
& & \xrightarrow{1-p} 1 \\
\text{(b)}
\end{array}
$$
**[IMAGE: Fig. 2 (a): OT over a general non-perfect MAC (Flowchart showing C1/C2, X1/X2, Y^n exchange between Alice-1, Alice-2, and Bob). (b): The non-perfect MAC model reduced to the SU-SBCp,w,w' described in Definition 8 (Diagram showing inputs X1, X2 from Alice-1, Alice-2 leading to output Y for Bob, with probabilities p and 1-p)]**

**Definition 12.** Let $n, k_1, k_2 \in \mathbb{N}$. An $(n, k_1, k_2)$ protocol involves interaction among Alice-1, Alice-2, and Bob via the setup illustrated in Figure 2-(b). At each time step $l = 1, 2, \ldots, n$, Alice-$i$ transmits a bit $X_{i, l}$ through the MAC. Users alternately exchange messages over a noiseless public channel in multiple rounds, both prior to each transmission and after the final transmission at $l = n$. While the number of rounds may vary, it is finite. Each user’s transmission is determined by a function of their input, private randomness, and all prior messages, channel inputs, or channel outputs observed. A positive rate pair $(R_1, R_2)$ is said to be an achievable OT rate pair for the DM-MAC if for $n \rightarrow \infty$ there exist $(n, k_1, k_2)$ protocols satisfying $\frac{k_i}{n} \rightarrow R_i$ such that for non-colluding parties, the asymptotic conditions (6)–(8) hold:
$$\lim_{n \rightarrow \infty} \text{Pr} \left[ (M_{1 Z_1}, M_{2 Z_2}) \ne (M_{1 Z_1}, M_{2 Z_2}) \right] = 0, \quad (6)$$
$$\lim_{n \rightarrow \infty} I(M_{1 Z_1}, M_{2 Z_2}; V) = 0, \quad (7)$$
$$\lim_{n \rightarrow \infty} I(Z_i; U_i)_{i \in \{1, 2\}} = 0, \quad (8)$$
where the final view of Alice-$i$ and Bob are $U_i = (M_{i 0}, M_{i 1}, R_{A_i}, X^n, C)$ and $V = (Z_1, Z_2, R_B, Y^n, C)$, respectively, and $C = (C_1, C_2)$. The closure of all achievable $OT^{k_1, k_2}$ rate pairs is called the OT capacity region of the MAC. Condition (6) says that Bob correctly learns both $M_{i Z_i}, i \in \{1, 2\}$ with negligible probability of error. Condition (7) says that Bob gains negligible information about the unselected messages, and conditions (8) says that Alice-$i$ gains negligible information about Bob’s choices $Z_i$.

In this section, we present a protocol of symmetric basic correlations (SU-SBC) in the case of honest-but-curious players. For a malicious receiver, the protocol achieves an achievable rate region.

As previously discussed, from Alice’s perspective, SU-SBC resembles the uniform input to a binary MAC. On the other hand, from Bob’s perspective, it corresponds to the output of a distinguishable mixture of three channels: the complete erasure channel with probability $(1 - p)^2$, a partial erasure channel with probability $2p(1 - p)$ and a DM-MAC with probability $p^2$.

Now, we present an OT protocol over noisy DM-MAC reduced to the noisy correlation (Figure 2-(b)) with non-colluding honest-but-curious parties. We assume that $p < \frac{1}{2}$ and that Alice-1, Alice-2 and Bob are in possession of $n$ realizations $\text{SU-SBC}_{p, W}$. Alice-$i$’s data and Bob’s data are denoted by $X_{i, 1}, \ldots, X_{i, n}$ for $i \in \{1, 2\}$ and $Y_1, \ldots, Y_n$, respectively. The output of the channel $W$ is denoted by $Z_1, \ldots, Z_n$ on inputs $X_{i, 1}, \ldots, X_{i, n}$ for $i \in \{1, 2\}$. The output of the partial erasure channel $W' : \mathcal{X}_1 \mathcal{X}_2 \rightarrow \mathcal{E}_{10} \cup \mathcal{E}_{01}$ is denoted by $Z'_1, \ldots, Z'_n$ on inputs $X_{i, 1}, \ldots, X_{i, n}$ for $i \in \{1, 2\}$.

Let $s_i$ and $r_i$, for $i \in \{1, 2\}$ be four parameters and $h_{i j} : \mathcal{R}_{i j} \times \mathcal{X}^n \rightarrow \{0, 1\}^{s_i n}$, and $k_{i j} : \mathcal{T}_{i j} \times \mathcal{X}^n \rightarrow \{0, 1\}^{r_i n}$, $i \in \{1, 2\}, j \in \{0, 1\}$ be two-universal hash functions. The $(\frac{1}{2})-OT^{k_1, k_2}$ rate pair is defined as $(\frac{k_1}{n}, \frac{k_2}{n})$. Note that the strings are ciphered by $k_{i j} : \mathcal{T}_{i j} \times \mathcal{X}^n \rightarrow \{0, 1\}^{r_i n}, i \in \{1, 2\}, j \in \{0, 1\}$ and $h_{i j} : \mathcal{R}_{i j} \times \mathcal{X}^n \rightarrow \{0, 1\}^{s_i n}$ are used as privacy amplification.

We now prove a lemma that says the above protocol without the hashes $h_{i j} : \mathcal{R}_{i j} \times \mathcal{X}^n \rightarrow \{0, 1\}^{s_i n}$ is still correct and secure.

**Lemma 11.** Protocol 1 satisfies the conditions (6)–(8) even without privacy amplification.

**Proof.** In Appendix F. $\square$

As a central block of proving the upper bound on OT rate, we want to consider the problem of pre-generated secret key agreement over MAC by public discussion. We use the results in the proof of Theorem 15.

**Protocol 1** OT over noisy DM-MAC (Definition 9)

**Parameters:**
*   $p < \frac{1}{2}$.
*   $\gamma > 0, \eta > 0$.
*   The rate of the protocol is $r_i - \gamma = \frac{k_i}{n}$, and $k_i$ is the length of Alice-$i$'s strings.

**Goal.** Alice-$i$ sends two strings $m_{i 0}$ and $m_{i 1}, i \in \{1, 2\}$ to Bob. At the end of the protocol, Bob gets $m_{i Z_i}, Z_i \in \{0, 1\}, i \in \{1, 2\}$ based on his choices while Alice-$i$ gets nothing $\Delta$.

**The protocol:**

1) Alice-$i$ transmits an $n$-tuple $\mathbf{X}_i = X_i^n$ of i.i.d. $\text{Bernoulli}(\frac{1}{2})$ bits over the reduced noisy DM-MAC.
2) Bob receives the $n$-tuple $\mathbf{Y} = Y^n$. Bob forms the sets
    $$\mathcal{E}_i := \{j \in \{1, 2, \ldots, n\} : Y_j = (\bar{x}_i \ne e, x_{\bar{i}})\}$$
    $$\bar{\mathcal{E}}_i := \{j \in \{1, 2, \ldots, n\} : Y_j = (x_i = e, x_{\bar{i}})\}$$
    If $|\mathcal{E}_i| < r_i n$ or $|\bar{\mathcal{E}}_i| < r_i n$, Bob aborts the protocol.
3) Bob creates the following sets:
    $$S_{i Z_i} \sim \text{Unif} \{A \subset \mathcal{E}_i : |A| = (p - \eta)n\}$$
    $$S_{i \bar{Z}_i} \sim \text{Unif} \{A \subset \bar{\mathcal{E}}_i : |A| = (p - \eta)n\}$$
    Bob reveals $S_{i 0}$ and $S_{i 1}$ to Alice-$i$ over the noiseless public channel (only the description of the sets).
4) Alice-$i$ randomly and independently chooses functions $\kappa_{i 0}, \kappa_{i 1}, h_{i j} : \mathcal{R}_{i j} \times \mathcal{X}^n \rightarrow \{0, 1\}^{s_i n}, i \in \{1, 2\}, j \in \{0, 1\}$ from a family $\mathcal{K}$ of two-universal hash functions:
    $$\kappa_{i 0}, \kappa_{i 1}: \{0, 1\}^{r_i n} \rightarrow \{0, 1\}^{k_i}$$
    Alice-$i$ finally sends the following information to Bob over the noiseless public channel:
    $$h_{i 0}, h_{i 1}, \kappa_{i 0}, \kappa_{i 1}, M_{i 0} \oplus \kappa_{i 0}(\mathbf{X}_i|_{S_{i 0}}), M_{i 1} \oplus \kappa_{i 1}(\mathbf{X}_i|_{S_{i 1}})$$
    and the total randomness in her possession.
5) Bob knows $h_{i Z_i}, \mathbf{X}_i|_{S_{i Z_i}}$, one can decode $M_{i Z_i}$. At first, he computes $\mathbf{X}_i|_{S_{i Z_i}}$, so that:
*   $(\mathbf{X}_1|_{S_{1 Z_1}}, \mathbf{X}_2|_{S_{2 Z_2}})$ and $\mathbf{Y}|_{S_{1 Z_1}, S_{2 Z_2}}$ are $\epsilon$-conditional typical according to $W$, that is, $\mathbf{Y}|_{S_{1 Z_1}, S_{2 Z_2}} \in \mathcal{T}_{W, \epsilon}^n(\mathbf{X}_1|_{S_{1 Z_1}}, \mathbf{X}_2|_{S_{2 Z_2}})$;
*   $h_i(R_{i Z_i}, \mathbf{X}_i|_{S_{i Z_i}}) = h_i(R_{i Z_i}, \mathbf{X}_i|_{S_{i Z_i}}), i \in \{1, 2\}$;
If there is more than one such $(\mathbf{X}_1|_{S_{1 Z_1}}, \mathbf{X}_2|_{S_{2 Z_2}})$ or none, Bob outputs an error.

### C. Secret Key Agreement over MAC by Public Discussion

Secret key agreement over a simple point-to-point channel is initially addressed in [42] and [7]. In [42], the author considers that Alice aims to share a secret key with Bob in the presence of a passive wiretapper. A noiseless one-way channel assists the main noisy channel with unlimited capacity. Also, it is assumed that Eve can receive all messages sent over the public channel without error, but the wiretapper can not change the messages without being detected.

A generalization of the above setting can be found in [43, 44]. Also, the problem of secret key agreement over MAC is studied in [45], in which the authors assumed that each transmitter plays the role of wiretapper for the other transmitter. This is the MAC with confidential message [46]. Consider the following settings: Alice-1 and Alice-2 aim to share secret keys with Bob in the presence of a passive wiretapper. Alice-$i$, Bob, and Eve govern the input $X_i$, the output $Y$, and $E$, respectively. Furthermore, a public channel with unlimited capacity is available for one-way communication from the senders to Bob. Both resources (MAC and the public channel) are available for communication, but not at the same time. First, the senders are allowed to use the MAC $n$ times, and then they can use the public channel only once.

Alice-$i$ and Bob use a protocol in which, at each step, Alice-$i$ sends a message to Bob depending on $X_i$ and all the messages previously received by Bob and Bob sends a message to Alice-$i$ depending on $Y$ and all the messages previously received by Alice-$i$. Let $C_{i, k}, i \in \{1, 2\}, k \in \{\text{odd}\}$ the use of public channel from Alice-$i$ to Bob, and $C_{i, k}, i \in \{1, 2\}, k \in \{\text{even}\}$ the use of public channel from Bob to the senders. Also, all legitimate parties can benefit from randomness statistically independent of $X, Y$, and $E$.

At the end of round-$l$, Alice-$i$ computes a key $S_i$ as a function of $X_i$ and $\mathcal{C}_i^l = [C_{i, 1}, \ldots, C_{i, l}]$, $l$ is even, and Bob computes keys $S'_i$ as a function of $Y$ and $\mathcal{C}^l$. The goal is to maximize $H(S_1, S_2)$ while $S_i$ and $S'_i$ agree with very high probability and Eve has negligible information about $S_i$ and $S'_i, i \in \{1, 2\}$:

$$H(C_{i, k}| \mathcal{C}_{i, k-1}, X_1, X_2) = 0, \quad \text{For odd } k \quad (9)$$
$$H(C_{i, k}| \mathcal{C}_{i, k-1}, Y) = 0, \quad \text{For even } k \quad (10)$$
$$H(S_i| \mathcal{C}^l, X_i) = 0, \quad (11)$$
$$H(S_1, S_2| \mathcal{C}_1^l, \mathcal{C}_2^l, X_1, X_2) = 0, \quad (12)$$
$$H(S'_1, S'_2| \mathcal{C}_1^l, \mathcal{C}_2^l, Y) = 0, \quad (13)$$
$$\text{Pr}\{(S_1, S_2) \ne (S'_1, S'_2)\} \le \epsilon, \quad (14)$$
$$I(S_i; \mathcal{C}^l, E) \le \delta_i l, \quad (15)$$
$$I(S_1, S_2; \mathcal{C}_1^l, \mathcal{C}_2^l, E) \le \delta' l, \quad (16)$$
for $i \in \{1, 2\}$, where $\epsilon, \delta_i$, and $\delta'$ are specified small numbers.

**Lemma 12.**
$$H(S_1, S_2| S'_1, S'_2) \le h(\epsilon_1) + h(\epsilon_2) + \log_2(|\mathcal{S}_1| - 1) + \log_2(|\mathcal{S}_2| - 1).$$

**Proof.**
$$H(S_1, S_2| S'_1, S'_2) = H(S_1| S'_1, S'_2) + H(S_2| S_1, S'_1, S'_2) \le H(S_1| S'_1) + H(S_2| S'_2),$$
where the last inequality is due to the fact that $S_i$ is independent of $S_{\bar{i}}$ and $S'_{\bar{i}}$. According to Fano’s lemma, conditions (14), (15) implies that $H(S_i| S'_i) \le h(\epsilon_i) + \log_2(|\mathcal{S}_i| - 1)$. $|\mathcal{S}_i|$ is the number of distinct values that $S_i$ takes on with non-zero probability. Note that $H(S_1, S_2| S'_1, S'_2) \rightarrow 0$ as $\epsilon_i, \epsilon' \rightarrow 0$. $\square$

**Theorem 13.** For every key agreement protocol satisfying (9)-(16), we have:
$$H(S_1) \le I(X_1; Y|\mathcal{C}_1^l, X_2, E) + H(S_1| S'_1) + I(S_1; \mathcal{C}_1^l, E),$$
$$H(S_2) \le I(X_2; Y|\mathcal{C}_2^l, X_1, E) + H(S_2| S'_2) + I(S_2; \mathcal{C}_2^l, E),$$
$$H(S_1, S_2) \le I(X_1, X_2; Y|\mathcal{C}_1^l, \mathcal{C}_2^l, E) + H(S_1, S_2| S'_1, S'_2) + I(S_1, S_2; \mathcal{C}_1^l, \mathcal{C}_2^l, E),$$
and for the case with constant $E$:
$$H(S_1) \le I(X_1; Y|\mathcal{C}_1^l, X_2) + H(S_1| S'_1) + I(S_1; \mathcal{C}_1^l),$$
$$H(S_2) \le I(X_2; Y|\mathcal{C}_2^l, X_1) + H(S_2| S'_2) + I(S_2; \mathcal{C}_2^l),$$
$$H(S_1, S_2) \le I(X_1, X_2; Y|\mathcal{C}_1^l, \mathcal{C}_2^l) + H(S_1, S_2| S'_1, S'_2) + I(S_1, S_2; \mathcal{C}_1^l, \mathcal{C}_2^l).$$

**Proof.** In Appendix G. $\square$

**Corollary 14.** The secret key rate region $R_{SK}$ of $X_1, X_2$, and $Y$ with respect to the constant random variable $E$ is upper bounded as:
$$
\mathcal{R}_{SK} \subseteq
\left\{
\begin{array}{l l}
(R_1, R_2): & R_1 \le \max_{P_{X_1} P_{X_2}} I(X_1; Y|X_2) \\
& R_2 \le \max_{P_{X_1} P_{X_2}} I(X_2; Y|X_1) \\
& R_1 + R_2 \le \max_{P_{X_1} P_{X_2}} I(X_1, X_2; Y)
\end{array}
\right\}, \quad (18)
$$
for some distribution $p(x_1)p(x_2)$ on $\mathcal{X}_1 \times \mathcal{X}_2$.

**Proof.** In Appendix H. $\square$

Now, we present the main theorems of the paper. The OT achievable rate pairs and OT capacity are defined in Definition 12. The OT capacity region is denoted by $C_{OT}$.

**Theorem 15** (honest-but-curious players). The OT capacity region of DM-MAC for honest-but-curious parties, is such that (upper bound):
$$
\mathcal{C}_{OT} \subseteq
\left\{
\begin{array}{l l}
(R_1, R_2): & R_1 \le \max_{P_{X_1} P_{X_2}} \min\{I(X_1; Y|X_2), H(X_1|Y)\} \\
& R_2 \le \max_{P_{X_1} P_{X_2}} \min\{I(X_2; Y|X_1), H(X_2|Y)\} \\
& R_1 + R_2 \le \max_{P_{X_1} P_{X_2}} \min\{I(X_1, X_2; Y), H(X_1, X_2|Y)\}
\end{array}
\right\}, \quad (19)
$$
for some distribution $p(x_1)p(x_2)$ on $\mathcal{X}_1 \times \mathcal{X}_2$.

**Proof.** In Appendix I. $\square$

**Theorem 16** (honest-but-curious players). The OT capacity region of DM-MAC reduced to the noisy $\text{SU-SBC}_{p, W}$ for honest-but-curious parties, is:
$$
\mathcal{C}_{OT} =
\left\{
\begin{array}{l l}
(R_1, R_2): & R_1 \le \max_{P_{X_1} P_{X_2}} I(X_1; Y|X_2) \\
& R_2 \le \max_{P_{X_1} P_{X_2}} I(X_2; Y|X_1) \\
& R_1 + R_2 \le \max_{P_{X_1} P_{X_2}} I(X_1, X_2; Y)
\end{array}
\right\} \quad (20)
$$
for some distribution $p(x_1)p(x_2)$ on $\mathcal{X}_1 \times \mathcal{X}_2$.

**Proof.** In Appendix J. $\square$

A rate pair $(R_1, R_2)$ is achievable if there exists a sequence of $(\frac{1}{2})-OT^{k_1, k_2}$ protocols satisfying (6)-(8) such that $\lim_{n \rightarrow \infty}(\frac{k_1}{n}, \frac{k_2}{n}) = (R_1, R_2)$.

**Theorem 17** (malicious Bob). An achievable rate region for OT ($\mathcal{R}_{OT}$) over the DM-MAC reduced to the noisy $\text{SU-SBC}_{p, W}$ with malicious Bob is:
$$
\mathcal{R}_{OT}(P_{Y|X_1, X_2}) =
\bigcup_{P_{X_1} P_{X_2}}
\left\{
\begin{array}{l l}
(R_1, R_2): & R_1 < \frac{1}{2} \max_{P_{X_1} P_{X_2}} \{I(X_1; Y|X_2) + I(X_1; X_2|Y)\} \\
& R_2 < \frac{1}{2} \max_{P_{X_1} P_{X_2}} \{I(X_2; Y|X_1) + I(X_1; X_2|Y)\} \\
& R_1 + R_2 < \frac{1}{2} \max_{P_{X_1} P_{X_2}} I(X_1, X_2; Y)
\end{array}
\right\} \quad (21)
$$
for some distribution $p(x_1)p(x_2)$ on $\mathcal{X}_1 \times \mathcal{X}_2$.

If $X_1 \rightarrow Y \rightarrow X_2$ is a Markov chain, then the above achievable rate region is simplified to:
$$
\mathcal{R}_{OT}(P_{Y|X_1, X_2}) =
\bigcup_{P_{X_1} P_{X_2}}
\left\{
\begin{array}{l l}
(R_1, R_2): & R_1 < \frac{1}{2} \max_{P_{X_1} P_{X_2}} I(X_1; Y|X_2) \\
& R_2 < \frac{1}{2} \max_{P_{X_1} P_{X_2}} I(X_2; Y|X_1) \\
& R_1 + R_2 < \frac{1}{2} \max_{P_{X_1} P_{X_2}} I(X_1, X_2; Y)
\end{array}
\right\} \quad (22)
$$

**Proof.** In Appendix K. $\square$

## VI. EXAMPLES

### A. Noiseless Binary Adder Channel (BAC)

Consider the following OT setting over the noiseless BAC [32]: two senders, Alice-1 and Alice-2, each hold two binary strings. Bob must select one string from each sender, with the unchosen strings remaining hidden. Communication occurs via the BAC, where the channel output is given by the sum of the inputs, $Y = X_1 + X_2$. This channel deterministically reveals the inputs when they are equal (0 or 1), but produces an erasure symbol ($e$) when they differ, making the output ambiguous. Erasures thus occur in two of the four possible input pairs.

For the non-colluding, honest-but-curious case, the OT capacity satisfies
$$R_1 + R_2 \le \max_{P_{X_1} P_{X_2}} H(X_1, X_2 | Y) = \frac{1}{2}.$$
If Bob cheats, for example, by altering non-erased outputs into the erasure set $\mathcal{E}$ to extract information about the unselected strings, Alice-1 and Alice-2 can detect such manipulation via a typicality test and abort the protocol. However, any such biasing capability would violate perfect security, as per Cleve’s impossibility result (FCT).

### B. A more general BE-MAC

Consider a general noisy DM-MAC with two senders $(X_1 \times X_2 \rightarrow Y, X_1, X_2 \sim \text{Bern}(\frac{1}{2}))$ as described in Definition 11. Now, suppose that this channel is reducible to the $\text{SU-SBC}_{p, W}$. Let the probability of erasure $1 - p = 0.6$. Then, the OT capacity region is $\{R_1 \le 0.4, R_2 \le 0.4, R_1 + R_2 \le 0.8\}$ if the players are honest but curious. The OT capacity for the case of bounded malicious Bob is unknown, but an inner bound on the OT capacity is $\{R_1 \le 0.2, R_2 \le 0.2, R_1 + R_2 \le 0.4\}$. Since the cheating capability of Bob is bounded within a certain range, imperfect OT is still possible, but the region is strictly smaller than the case of honest Bob.

As is clear, the sum rate of the OT capacity of $\text{SU-SBC}_{p, W}$ and $\text{SU-SBC}_{p, W, W'}$ is redundant, independent of the probability of erasure. Now consider the following example.

### C. Noisy BAC

Let $X_1, X_2 \sim \text{Bern}(\frac{1}{2})$ be independent. The channel output $Y$ is given by
$$Y = \left\{
\begin{array}{l l}
X_1 + X_2, & \text{with probability } p \\
e, & \text{with probability } 1 - p'
\end{array}
\right.$$
where $e$ denotes an erasure symbol. The OT capacity region for the honest-but-curious case is as follows:
$$
\mathcal{C}_{OT} =
\left\{
\begin{array}{l l}
(R_1, R_2): & R_1 \le p \\
& R_2 \le p \\
& R_1 + R_2 \le 3p/2
\end{array}
\right\}
$$
and for the case of malicious Bob, the lower bound on OT capacity is:
$$
\mathcal{R}_{OT} =
\left\{
\begin{array}{l l}
(R_1, R_2): & R_1 < 3p/4 \\
& R_2 < 3p/4
\end{array}
\right\}
$$
The key distinction between this example and example VI-A is that independent OT is not feasible over the noiseless BAC, as the channel lacks sufficient intrinsic uncertainty to support independent OT. In contrast, example VI-C presents a different scenario in which the channel exhibits two distinct sources of uncertainty: (i) the first arises from the noisy correlation $\text{SU-SBC}_{p, W}$, and (ii) the second stems from the inherent ambiguity induced by the channel structure itself.

## VII. CONCLUSION

We investigated bounds on the oblivious transfer capacity of noisy MAC, focusing on secure multiparty computations involving two non-colluding senders and a single receiver. A protocol was proposed that is both correct and secure against honest-but-curious parties. Furthermore, we demonstrated that the protocol remains correct and secure when the receiver is dishonest, although the exact capacity in such scenarios remains undetermined. The OT capacity region for honest-but-curious players was established for a special reduced version with optimal rates, while for a malicious receiver, the study delineates a feasible rate region. The work delves into noisy MACs comprehensively, introducing reductions to symmetric basic correlations (SBCs) as defined in [6] and extending OT protocols to multi-sender configurations. Key contributions include precisely characterizing OT capacity for honest-but-curious settings, leveraging information-theoretic bounds, and significant progress in addressing adversarial behavior.

An intriguing future direction is the study of OT over MACs involving colluding parties. If the senders can collude with the receiver—a scenario closer to practical settings—the secrecy criteria become substantially stricter. Also, In [6], the authors offer a protocol secure and correct against a malicious sender when the correlation is slightly unfair, albeit no longer achieving positive rates. In the MAC setting, if we assume both senders are cheating, it can be conjectured that a similar protocol does not provide positive rates. But what about if just one of the senders is cheating? These challenging cases form the basis of our next research endeavors.

## ACKNOWLEDGMENT

The authors hereby acknowledge Andreas Winter, Rémi A. Chou, and Pin-Hsun Lin for valuable discussions during this research project.

## APPENDIX A
### PROOF OF LEMMA 1

A. Upper bound

By definition, $H^{\epsilon}_{\infty}(X|Y)$ considers the min-entropy over all distributions $P_{X'Y'}$ that are $\epsilon$-close to the original distribution $P_{XY}$. This means that:
$$H^{\epsilon}_{\infty}(X|Y) \ge H_{\infty}(P_{X'|Y'}) \quad \text{for any such } P_{X'Y'}.$$
Since the original distribution $P_{XY}$ satisfies $\|P_{XY} - P_{X'Y'}\|_1 = 0 \le \epsilon$, it is included in the set of distributions over which the supremum is taken. Therefore:
$$H^{\epsilon}_{\infty}(X|Y) \ge H_{\infty}(P_{X|Y}),$$
where $P_{X|Y}$ is the conditional distribution derived from $P_{XY}$. From the definition of $H_{\infty}(X|Y)$:
$$H_{\infty}(P_{X|Y}) = H_{\infty}(X|Y).$$
Thus, combining the inequalities:
$$H^{\epsilon}_{\infty}(X|Y) \ge H_{\infty}(X|Y).$$

B. Lower bound

The relaxation ensures that the min-entropy $H^{\epsilon}_{\infty}(X|Y)$ can only increase compared to $H_{\infty}(X|Y)$, as it maximizes over a larger set of distributions. For any $P_{X'Y'}$, it holds that:
$$H^{\epsilon}_{\infty}(X|Y) = -\log \max_{P_{X'Y'}} \max_{y} \sum_{x} P_{X'|Y'=y}(x).$$
This is an optimized version of the standard $H_{\infty}(X|Y)$. However, we consider the $\epsilon$-closeness in terms of probabilities to connect these values. Using standard smoothing arguments, the worst-case difference in probabilities for any outcome is bounded by $\epsilon$. Specifically:
$$\frac{1}{1-\epsilon} P_{X|Y=y}(x) \le P_{X'|Y'=y}(x).$$
taking logs:
$$-\log P_{X|Y=y}(x) + \log(1 - \epsilon) \ge - \log (P_{X'|Y'=y}(x)),$$
maximizing over all $P_{X'Y'}$:
$$H^{\epsilon}_{\infty}(X|Y) \ge H_{\infty}(X|Y) - \log(1/\epsilon).$$
This accounts for the worst-case adjustment introduced by smoothing. $\square$

## APPENDIX B
### PROOF OF LEMMA 2

Let $P_{XY}$ be a probability distribution over $\mathcal{X} \times \mathcal{Y}$, $P_{X^n Y^n} := (P_{XY})^n$ the $n$-wise direct product. Then, for any $\delta \in [0, \log(|\mathcal{X}|)]$ and $(\mathbf{x}^n, \mathbf{y}^n)$ chosen according to $P_{X^n Y^n}$, let $(X_i, Y_i) \sim P_{XY}$ i.i.d. for $i = 0, \ldots, n - 1$. Define:
$$Z_i := \log \left( \frac{1}{P_{X|Y}(X_i|Y_i)} \right) \implies \sum_{i=0}^{n-1} Z_i = \log \left( \frac{1}{P_{X^n|Y^n}(\mathbf{X}^n|\mathbf{Y}^n)} \right)$$
We know that $\mathbb{E}[Z_i] = H(X|Y)$, so the total expectation is:
$$\mathbb{E} \left[ \sum_{i=0}^{n-1} Z_i \right] = nH(X|Y).$$
Each $Z_i \in [0, \log(|\mathcal{X}|)]$ because: (i) For fixed $y$, $P_{X|Y}(x|y) \ge 0$, and (ii) $P_{X|Y}(x|y) \le |\mathcal{X}|$. So:
$$0 \le \log \left( \frac{1}{P_{X|Y}(x|y)} \right) \le \log(|\mathcal{X}|).$$
Let $Z := \sum_{i=0}^{n-1} Z_i$. Then $Z$ is a sum of $n$ i.i.d. bounded random variables in $[0, \log(|\mathcal{X}|)]$. Hoeffding’s inequality [41] states that for independent $Z_i \in [a, b]$:
$$\text{Pr} \left[ \left| \sum_{i=0}^{n-1} Z_i - \mathbb{E}[Z_i] \right| \ge \delta \right] \le \exp \left( - \frac{2n \delta^2}{\sum_{i=0}^{n-1} (b - a)^2} \right) = \exp \left( - \frac{2n \delta^2}{(b - a)^2} \right)$$
Here, $a = 0, b = \log(|\mathcal{X}|)$, so:
$$\text{Pr} [Z \ge n(H(X|Y) + \delta)] \le \exp \left( - \frac{2n \delta^2}{\log^2(|\mathcal{X}|)} \right)$$
$$= 2^{\frac{-2n \delta^2}{\log^2(|\mathcal{X}|) \log_2 e}}$$
$$\le 2^{- \frac{n \delta^2}{16 \log^2(|\mathcal{X}|)}}. \quad (23)$$
In the last step, we used only a slightly looser bound.
Now, set $\delta := 4\sqrt{\frac{\log(1/\epsilon)}{n}} \log(|\mathcal{X}|)$. Then, from (23):
$$\text{Pr}_{(U, V) \sim P_{X^n Y^n}} \left[ \log \left( \frac{1}{P_{X^n|Y^n}(\mathbf{U}|\mathbf{V})} \right) \ge n(H(X|Y) + \delta) \right] \le \epsilon.$$
Thus, with probability at least $1 - \epsilon$, we have:
$$P_{X^n|Y^n}(\mathbf{x}^n|\mathbf{y}^n) \ge 2^{-n H(X|Y) - \delta}.$$
Therefore, the $\epsilon$-smooth conditional min-entropy satisfies:
$$H^{\epsilon}_{\infty}(X^n|Y^n) \ge nH(X|Y) - \delta = nH(X|Y) - 4\sqrt{n}\log(1/\epsilon)\log(|\mathcal{X}|).$$
This completes the proof. $\square$

## APPENDIX C
### PROOF OF LEMMA 6

This lemma is an extended version of [6, Lemma 5]. It evaluates the likelihood of passing a “typicality test” when input sequences are fed into a noisy MAC and output sequences are generated. The lemma bounds the probability that the channel's output remains consistent with a typical input-output relationship, even under deviations in the input strings (e.g., due to cheating). In simpler terms, this lemma demonstrates that if someone attempts to manipulate the inputs or outputs, the probability of such manipulations going undetected is exponentially small under specific conditions.

Divide $\mathbf{X}_1^n$ and $\mathbf{X}_2^n$ into smaller “blocks” where manipulations occurred. Each block contains at least one incorrect symbol. Use the law of large numbers: Since $\delta n$ positions were changed, the overall likelihood of $(\mathbf{X}_1^n, \mathbf{X}_2^n, \mathbf{Z})$ remaining jointly typical decreases exponentially:

Define the sets $\mathcal{I}_{x_1}$ and $\mathcal{I}_{x_2}$:
$$\pi_{x_1} := \pi(\mathbf{x}_1|\mathbf{x}'_1) = |\mathcal{I}_{x_1}| \quad \text{and} \quad \pi_{x_2} := \pi(\mathbf{x}_2|\mathbf{x}'_2) = |\mathcal{I}_{x_2}|.$$
These sets identify positions where the input symbols $x_1$ and $x_2$ appear in $\mathbf{X}_1^n$ and $\mathbf{X}_2^n$, respectively. Assume that the sequences $\mathbf{x}'_1^n$ and $\mathbf{x}'_2^n$ are manipulated. By Hamming distance properties, the cardinalities satisfy:
$$\pi_{x_1} \ge \frac{1}{|\mathcal{X}_1|} \delta n, \quad \pi_{x_2} \ge \frac{1}{|\mathcal{X}_2|} \delta n.$$

For the MAC output, we analyze the empirical distributions $W_{(x_1, x_2)|k}$ over positions $\mathcal{I}_{x_1} \cap \mathcal{I}_{x_2}$. For the joint input-output behavior at positions $k$:
$$\left| \frac{1}{\pi_{x_1} \pi_{x_2}} \sum_{k \in \mathcal{I}_{x_1} \cap \mathcal{I}_{x_2}} W_{(x_1, x_2)|k} - W_{x_1, x_2} \right| \ge \frac{1}{|\mathcal{X}_1| |\mathcal{X}_2|} \delta^2 \eta.$$
Here, $W_{(x_1, x_2)|k}$ is the empirical output distribution, and $W_{x_1, x_2}$ is the expected output distribution given inputs $(x_1, x_2)$. The deviation occurs because the joint input behavior $(x_1, x_2)$ does not match the expected channel behavior. By the pigeonhole principle, there exists at least one output symbol $z \in \mathcal{Z}$ such that:
$$\left| \frac{1}{\pi_{x_1} \pi_{x_2}} \sum_{k \in \mathcal{I}_{x_1} \cap \mathcal{I}_{x_2}} W_{(x_1, x_2)|k}(z) - W_{x_1, x_2}(z) \right| \ge \frac{1}{|\mathcal{X}_1| |\mathcal{X}_2| |\mathcal{Z}|} \delta^2 \eta.$$
$W_{(x_1, x_2)|k}(z)$ is the probability of output $z$ at position $k$. This inequality ensures at least one output symbol $z$ where the deviation is significant.
Now, consider the number of positions where $z$ occurs in the output sequence $\mathbf{Z}^n$. Define:
$$\pi(z|\mathbf{z}|\mathcal{I}_{x_1}, \mathcal{I}_{x_2}) := \text{count of } z \text{ in } \mathbf{Z}^n \text{ over positions } \mathcal{I}_{x_1} \cap \mathcal{I}_{x_2}.$$
Then:
$$\left| \frac{1}{\pi_{x_1} \pi_{x_2}} \sum_{k \in \mathcal{I}_{x_1} \cap \mathcal{I}_{x_2}} W_{(x_1, x_2)|k}(z) \right| \ge \frac{1}{2 |\mathcal{X}_1| |\mathcal{X}_2| |\mathcal{Z}|} \delta^2 \eta \pi_{x_1} \pi_{x_2}.$$
This bounds the deviation in counts of $z$ compared to the expected behavior.
Refine the bound further by introducing sets:
$$\mathcal{I}_{x_1 x_2 y} = \{k \in \mathcal{I}_{x_1} \cap \mathcal{I}_{x_2} : Y_k = y\}, \quad \pi_{x_1 x_2 y} := |\mathcal{I}_{x_1 x_2 y}|.$$
Then:
$$\frac{1}{\pi_{x_1 x_2 y}} \pi_{x_1 x_2 y} \ge \frac{1}{4 |\mathcal{X}_1|^2 |\mathcal{X}_2|^2 |\mathcal{Z}|^2} \delta^2 \eta^2 \pi_{x_1} \pi_{x_2} \ge \frac{1}{4 |\mathcal{X}_1|^3 |\mathcal{X}_2|^3 |\mathcal{Z}|^3} \delta^2 \eta^2 n.$$
Using the Chernoff bound [6, Lemma 4] completes the proof. $\square$

## APPENDIX D
### PROOF OF THEOREM 9

Consider the general case in Definition 10, then we have (Omitting the random variable $C$, since it is also considered in the final views and is assumed to be ignored by honest players):

Bob's privacy: Alice should not learn the receiver's effective choice $Z'$:
$$I(X; Z'|Z) = 0, \quad (24)$$
$$I(X; V|Z, Z', M_{Z'}) = 0. \quad (25)$$
This means that:
$$H(M_{\bar{Z}'}|V, Z, Z', M_{Z'}) = H(M_{\bar{Z}'}). \quad (26)$$
This implies that given the information $V$, the choice $Z$, the effective choice $Z'$ and $M_{Z'}$, Bob has no additional information about $M_{\bar{Z}'}$. Consider the mutual information between Alice's bits and the information received by Bob $I(X; V) \ge I(M_0, M_1; V)$. Since Bob should learn $M_{Z'}$ and nothing about $M_{\bar{Z}'}$, we have:
$$I(M_{Z'}, M_{\bar{Z}'}; V|Z, Z', M_{Z'}) = I(M_{Z'}; V|Z, Z', M_{Z'}) + I(M_{\bar{Z}'}; V|Z, Z', M_{Z'}). \quad (27)$$
The first term should not be equal to zero because $V$ must also convey the value of $M_{Z'}$ to Bob. The second term should be zero due to the perfect secrecy of OT. So consider the second term:
$$I(M_{\bar{Z}'}; V|Z, Z', M_{Z'}) = I(M_{\bar{Z}'}; V|Z, Z')$$
$$= H(M_{\bar{Z}'}|Z, Z') - H(M_{\bar{Z}'}|V, Z, Z')$$
$$= H(M_{\bar{Z}'}|Z, Z') - H(M_{\bar{Z}'}) \quad (28)$$
$$= -I(M_{\bar{Z}'}; Z, Z'),$$
where the first equality is happened due the fact that $M_{Z'}$ and $M_{\bar{Z}'}$ are independent, and the third equality follows from (26).
As $I(M_{\bar{Z}'}; Z, Z')$ is a non-negative quantity, then we proved that $I(M_{\bar{Z}'}; Y|M_{Z'}, Z, Z')$ is not a positive quantity. It is impossible unless $I(M_{\bar{Z}'}; Z, Z') = 0$, which implies that all inputs $(X, Z)$ are independent (We already clarified that this is often unattainable and in the most general case, we assume there is a known dependency between the inputs). Combining (27) and (28), we have:
$$I(X; V|Z, Z', M_{Z'}) \ge I(M_{Z'}, M_{\bar{Z}'}; V|Z, Z', M_{Z'})$$
$$\overset{(a)}{=} I(M_{Z'}; V|Z, Z', M_{Z'}) + I(M_{\bar{Z}'}; V|Z, Z', M_{Z'})$$
$$\overset{(b)}{=} H(M_{\bar{Z}'}|Z, Z') - H(M_{\bar{Z}'}|V, Z, Z', M_{Z'})$$
$$= 0, \quad (29)$$
where $(a)$ is due to the Markov chain $(M_0, M_1) \rightarrow X \rightarrow V$ and data processing inequality (DPI), and $(b)$ is since given $V$ and $Z'$, the uncertainty about $M_{\bar{Z}'}$ is zero (correctness criterion). Also assume the usual OT correctness: given $V$ and the (effective) index $Z'$, Bob recovers $M_{Z'}$ with certainty, so
$$H(M_{Z'} | V, Z, Z') = 0. \quad (30)$$
Finally assume the messages are independent *a priori* and non-degenerate:
$$M_0 \perp M_1, \quad H(M_0) > 0, \quad H(M_1) > 0.$$
From the causal structure of the protocol we have the Markov chain $(M_0, M_1) \rightarrow X \rightarrow V$. Now consider (24):
$$I(X; Z' | Z) = 0.$$
Again use the Markov chain $(M_0, M_1) \rightarrow X \rightarrow Z'$ (the variable $Z'$ depends only on the transcript / view which in turn depends on $X$). By conditional DPI we get
$$I(M_0, M_1; Z' | Z) \le I(X; Z' | Z) = 0,$$
hence
$$I(M_0, M_1; Z' | Z) = 0. \quad (31)$$
Chain-rule on (31) gives
$$0 = I(M_{Z'}; Z' | Z) + I(M_{\bar{Z}'}; Z' | Z, M_{Z'}).$$

Both terms are nonnegative, so each must be zero:
$$I(M_{Z'}; Z' | Z) = 0, \quad I(M_{\bar{Z}'}; Z' | Z, M_{Z'}) = 0.$$
From the first equality we deduce
$$H(M_{Z'} | Z, Z') = H(M_{Z'} | Z). \quad (6)$$
(Conditional independence of $M_{Z'}$ and $Z'$ given $Z$ means conditioning on $Z'$ does not further reduce entropy beyond conditioning on $Z$ alone.) By correctness (30) we have
$$H(M_{Z'} | V, Z, Z') = 0.$$
Combine this with (27)
$$I(M_0, M_1; V | Z, Z', M_{Z'}) = 0. \quad (32)$$
The latter says that modulo $(Z, Z', M_{Z'})$, $V$ is independent of $(M_0, M_1)$ — equivalently, all dependence of $V$ on the message space is already captured by the conditioned $M_{Z'}$. Concretely, we can rewrite (27) as
$$H(V | Z, Z', M_{Z'}) = H(V | Z, Z', M_{Z'}, M_0, M_1).$$
But given $M_0, M_1$, and the protocol, $V$ is determined stochastically through the protocol; the important observation is that (32) implies that, after conditioning on $(Z, Z', M_{Z'})$, $V$ carries no additional information about the messages. Together with (30) this implies that the uncertainty of $M_{Z'}$ given $Z, Z'$ must be no larger than its uncertainty after seeing $V$:
$$H(M_{Z'} | Z, Z') = H(M_{Z'} | V, Z, Z') = 0, \quad (33)$$
where the last equality is due to (30).
This means that the message $M_{Z'}$ must be determined (with probability 1) by $(Z, Z')$. Under the usual OT input model (two independent non-degenerate messages), this is impossible: $M_{Z'}$ has positive entropy conditioned on any small amount of side information unless the messages are degenerate (deterministic functions of the choices). Thus we get a contradiction to the assumption that $H(M_0) > 0$ or $H(M_1) > 0$. Therefore the perfect-information equalities (24) and (25) cannot hold simultaneously for nontrivial messages. $\square$

## APPENDIX E
### PROOF OF THEOREM 10

The proof follows from an extended version presented in [37]. Consider the intuitive security criterion for Bob: if Alice-$i$ acts maliciously, she sends $X'_i$ instead of $X_i$ (This means that Alice-$i$ can replace some bits with any deterministic or random bits in her flavor so that the Hamming distance between the original sequence and the changed sequence could be in the range $0 \le d_H \le n$). In such a case, security constraint implies that the mutual information between Bob and Alice-$i$'s effective input given $X_i$ and $C$ should be zero ($X \rightarrow (C, X_i) \rightarrow Y$). On the other hand, the mutual information between Bob's final view $(V, Y)$ and Alice-$i$'s final view given $(C, X_i, X'_i)$ should be zero ($U_i \rightarrow (C, X_i, X'_i) \rightarrow (V, Y)$):
$$I(Y; X'_i | C, X_i) = 0, \quad \text{and} \quad I(V, Y; U_i | C, X_i, X'_i) = 0,$$
where $Y = Z_i$ and $V = M_{\bar{Z}'_i}$. Combining both, we have:
$$I(Z_i; X'_i | C, X_i) + I(M_{\bar{Z}'_i}, Z_i; U_i | C, X_i, X'_i)$$
$$\overset{(a)}{=} I(Z_i; X'_i | C, X_i) + I(Z_i; U_i | C, X_i, X'_i) + I(M_{\bar{Z}'_i}; U_i | C, Z_i, X_i, X'_i)$$
$$\overset{(b)}{=} I(Z_i; X'_i | C, X_i) + I(Z_i; U_i | C, X_i, X'_i)$$
$$= I(Z_i; X'_i, U_i | C, X_i)$$
$$= I(Z_i; U_i | C, X_i) + I(Z_i; X'_i | C, X_i, U_i)$$
$$\overset{(c)}{=} I(Z_i; U_i | C, X_i)$$
$$= 0,$$
where $(a)$ follows from the fact that $I(M_{\bar{Z}'_i}; U_i | C, Z_i, X_i, X'_i) = 0$, and $(b)$ is due to the following argument: choose $M_{\bar{Z}'_i} = (M_{i 0}, M_{i 1})$ and for the values $l \in \{0, 1\}$, $M_{i l}$ is chosen according to the distribution $P_{V|C, X_i, U_i, Z_i=l}$ except for $M_{Z'_i} = V$ which is Bob's final view. Note that $P_{V|C, X_1, X_2, U_1, U_2, Z_1, Z_2} = P_{V|C, X_1, U_1, Z_1} P_{V|C, X_2, U_2, Z_2}$ due to independence of transmitters. So, both $M_{i, l} \in \{0, 1\}$, have distribution $P_{V|C, X_i, U_i, Z_i=l}$. This makes the Markov chain $X_i \rightarrow (C, X_i, U_i) \rightarrow Z_i$, then we have $I(X_i; Z_i | \bar{C}, X_i, U_i) = 0$, for $i \in \{1, 2\}$.
Without considering the random variable $C$, in the case of malicious Bob, we have:

Bob's privacy: Alice-$i$ should not learn the receiver's effective choice $Z'_i$:
$$I(X_i; Z'_i|Z_i) = 0, \quad (34)$$
$$I(X_i; V | Z_i, Z'_i, M_{Z'_i}) = 0. \quad (35)$$
This means that:
$$H(M_{\bar{Z}'_i} | V, Z_i, Z'_i, M_{Z'_i}) = H(M_{\bar{Z}'_i}). \quad (36)$$
This implies that given the information $V$, the choice $Z_i$, the effective choice $Z'_i$ and $M_{Z'_i}$, Bob has no additional information about $M_{\bar{Z}'_i}$. Consider the mutual information between Alice's bits and the information received by Bob $I(X_i; V) \ge I(M_{i 0}, M_{i 1}; V)$. Since Bob should learn $M_{Z'_i}$ and nothing about $M_{\bar{Z}'_i}$, we have:
$$I(M_{Z'_i}, M_{\bar{Z}'_i}; V | Z_i, Z'_i, M_{Z'_i}) = I(M_{Z'_i}; V | Z_i, Z'_i, M_{Z'_i}) + I(M_{\bar{Z}'_i}; V | Z_i, Z'_i, M_{Z'_i}). \quad (37)$$
The first term should not equal zero because $V$ must also convey the value of $M_{Z'_i}$ to Bob. The second term should be zero due to the perfect secrecy of OT. So consider the second term:
$$I(M_{\bar{Z}'_i}; V | Z_i, Z'_i, M_{Z'_i}) = I(M_{\bar{Z}'_i}; V | Z_i, Z'_i)$$
$$= H(M_{\bar{Z}'_i} | Z_i, Z'_i) - H(M_{\bar{Z}'_i} | V, Z_i, Z'_i)$$
$$= H(M_{\bar{Z}'_i} | Z_i, Z'_i) - H(M_{\bar{Z}'_i}) \quad (38)$$
$$= -I(M_{\bar{Z}'_i}; Z_i, Z'_i),$$
where the first equality is due the fact that $M_{Z'_i}$ and $M_{\bar{Z}'_i}$ are independent, and the third equality follows from (36).
As $I(M_{\bar{Z}'_i}; Z_i, Z'_i)$ is a non-negative quantity, then we proved that $I(M_{\bar{Z}'_i}; Y | M_{Z'_i}, Z_i, Z'_i)$ is not a positive quantity, unless $I(M_{\bar{Z}'_i}; Z_i, Z'_i) = 0$, which implies that all inputs $(X_i, Z_i)$ are independent. Combining (37) and (38), we have:
$$I(X_i; V | Z_i, Z'_i, M_{Z'_i}) \ge I(M_{Z'_i}, M_{\bar{Z}'_i}; V | Z_i, Z'_i, M_{Z'_i})$$
$$= I(M_{Z'_i}; V | Z_i, Z'_i)$$
$$= 0, \quad (39)$$
where the first equality is because given $V$ and $Z'_i$, the uncertainty about $M_{\bar{Z}'_i}$ is zero. $\square$

## APPENDIX F
### PROOF OF LEMMA 11

*   Due to the Chernoff bound, we know that the probability of aborting the protocol by Bob in step 2 tends to zero as $n \rightarrow \infty$. When $|\mathcal{E}_i| < r_i n$ and $|\bar{\mathcal{E}}_i| < r_i n$, then Bob knows $\mathbf{X}_i|_{S_{i Z_i}}$. Since Bob also knows $\kappa_{i Z_i}$, Bob can compute the key $\kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}})$. Then Bob can recover $M_{i Z_i}$ from $M_{i Z_i} \oplus \kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}})$ sent by Alice-$i$. Then,
$$\lim_{n \rightarrow \infty} \text{Pr} \left[ (M_{1 Z_1}, M_{2 Z_2}) \ne (M_{1 Z_1}, M_{2 Z_2}) \right] = 0.$$

*   $$I(M_{1 Z_1}, M_{2 Z_2}; V) = I(M_{1 Z_1}, M_{2 Z_2}; Z_1, Z_2, Y^n, C)$$
$$= I(M_{1 Z_1}, M_{2 Z_2}; Z_1, Z_2, Y^n, C_1, C_2)$$
$$\overset{(a)}{=} I(M_{1 Z_1}; Z_1, Z_2, Y^n, C_1, C_2) + I(M_{2 Z_2}; Z_1, Z_2, Y^n, C_1, C_2 | M_{1 Z_1})$$
$$\overset{(b)}{\le} I(M_{1 Z_1}; Z_1, Y^n, C_1) + I(M_{2 Z_2}; Z_2, Y^n, C_2 | M_{1 Z_1})$$
$$\overset{(c)}{\le} I(M_{1 Z_1}; Z_1, Y^n, C_1) + I(M_{2 Z_2}; Z_2, Y^n, C_2, M_{1 Z_1})$$
$$\overset{(d)}{\le} I(M_{1 Z_1}; Z_1, Y^n, C_1) + I(M_{2 Z_2}; Z_2, Y^n, C_2), \quad (40)$$
where $(a)$ follows from the fact that $M_{1 Z_1} - (Z_1, Y^n, C_1) - (Z_2, C_2)$ is a Markov chain, $(b)$ is due to the independency of $M_{1 Z_1}$ from $M_{2 Z_2}$, and $(c)$ is due the Markov chain $M_{2 Z_2} - (Z_2, Y^n, C_2) - M_{1 Z_1}$. Now, it suffices to show that $I(M_{i Z_i}; Z_i, Y^n, C_i)_{i \in \{1, 2\}}$ tends to zero as $n \rightarrow \infty$.

$$I(M_{i Z_i}; Z_i, Y^n, C_i)_{i \in \{1, 2\}} = I(M_{i Z_i}; Z_i, Y^n, S_{i 0}, S_{i 1}, \kappa_{i 0}, \kappa_{i 1}, M_{i 0} \oplus \kappa_{i 0}(\mathbf{X}_i|_{S_{i 0}}), M_{i 1} \oplus \kappa_{i 1}(\mathbf{X}_i|_{S_{i 1}}))$$
$$= I(M_{i Z_i}; Z_i, Y^n, S_{i Z_i}, S_{i \bar{Z}_i}, \kappa_{i Z_i}, \kappa_{i \bar{Z}_i}, M_{i Z_i} \oplus \kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}}), M_{i \bar{Z}_i} \oplus \kappa_{i \bar{Z}_i}(\mathbf{X}_i|_{S_{i \bar{Z}_i}}))$$
$$= I(M_{i Z_i}; M_{i Z_i} \oplus \kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}}) | Z_i, Y^n, S_{i Z_i}, S_{i \bar{Z}_i}, \kappa_{i Z_i}, \kappa_{i \bar{Z}_i}, M_{i \bar{Z}_i} \oplus \kappa_{i \bar{Z}_i}(\mathbf{X}_i|_{S_{i \bar{Z}_i}}))$$
$$+ I(M_{i Z_i}; Z_i, Y^n, S_{i Z_i}, S_{i \bar{Z}_i}, \kappa_{i Z_i}, \kappa_{i \bar{Z}_i}, M_{i \bar{Z}_i} \oplus \kappa_{i \bar{Z}_i}(\mathbf{X}_i|_{S_{i \bar{Z}_i}}) | M_{i Z_i} \oplus \kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}}))$$
$$\overset{(a)}{=} I(M_{i \bar{Z}_i}; M_{i \bar{Z}_i} \oplus \kappa_{i \bar{Z}_i}(\mathbf{X}_i|_{S_{i \bar{Z}_i}}) | Z_i, Y^n, S_{i Z_i}, S_{i \bar{Z}_i}, \kappa_{i Z_i}, \kappa_{i \bar{Z}_i}, M_{i Z_i} \oplus \kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}}))$$
$$= H(M_{i \bar{Z}_i} \oplus \kappa_{i \bar{Z}_i}(\mathbf{X}_i|_{S_{i \bar{Z}_i}}) | Z_i, Y^n, S_{i Z_i}, S_{i \bar{Z}_i}, \kappa_{i Z_i}, \kappa_{i \bar{Z}_i}, M_{i Z_i} \oplus \kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}}))$$
$$- H(M_{i \bar{Z}_i} \oplus \kappa_{i \bar{Z}_i}(\mathbf{X}_i|_{S_{i \bar{Z}_i}}) | M_{i \bar{Z}_i}, Z_i, Y^n, S_{i Z_i}, S_{i \bar{Z}_i}, \kappa_{i Z_i}, \kappa_{i \bar{Z}_i}, M_{i Z_i} \oplus \kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}}))$$
$$\overset{(b)}{\le} n(r_i - \lambda') $$
$$- H(\kappa_{i \bar{Z}_i}(\mathbf{X}_i|_{S_{i \bar{Z}_i}}) | M_{i \bar{Z}_i}, Z_i, (Y^n|_{S_{i Z_i}}, Y^n|_{S_{i \bar{Z}_i}}), S_{i Z_i}, S_{i \bar{Z}_i}, \kappa_{i Z_i}, \kappa_{i \bar{Z}_i}, M_{i Z_i} \oplus \kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}}))$$
$$\overset{(c)}{\le} n(r_i - \lambda') - H(\kappa_{i \bar{Z}_i}(\mathbf{X}_i|_{S_{i \bar{Z}_i}}) | Y^n|_{S_{i \bar{Z}_i}}, \kappa_{i \bar{Z}_i}), \quad (41)$$
where $(a)$ follows from the independency of $M_{i \bar{Z}_i}$ from $(Z_i, Y^n, S_{i Z_i}, S_{i \bar{Z}_i}, \kappa_{i Z_i}, \kappa_{i \bar{Z}_i}, M_{i Z_i} \oplus \kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}})), i \in \{1, 2\}$, $(b)$ follows since $\kappa_{i \bar{Z}_i}(\mathbf{X}_i|_{S_{i \bar{Z}_i}})$ is $n(r_i - \lambda')$, $i \in \{1, 2\}$ bits long and $(c)$ follows since $\kappa_{i \bar{Z}_i}(\mathbf{X}_i|_{S_{i \bar{Z}_i}}) - (Y^n|_{S_{i \bar{Z}_i}}, \kappa_{i \bar{Z}_i}) - M_{i Z_i}, Z_i, Y^n, S_{i Z_i}, S_{i \bar{Z}_i}, \kappa_{i Z_i}, M_{i Z_i} \oplus \kappa_{i Z_i}(\mathbf{X}_i|_{S_{i Z_i}})), i \in \{1, 2\}$ is a Markov chain. We know that,
$$H_2(\mathbf{X}_i|_{S_{i Z_i}} | Y^n|_{S_{i Z_i}}) = H(\mathbf{X}_i|_{S_{i Z_i}} | \mathbf{Y}^n|_{S_{i Z_i}}) \ge \Delta(Y^n|_{S_{i Z_i}}) \ge n r_i, i \in \{1, 2\},$$
since the size of the set $S_{i Z_i}$ is at least $n r_i$. Then, from Lemma 4 we have:
$$H(\kappa(\mathbf{X}_i|_{S_{i Z_i}}) | \kappa, Y^n|_{S_{i Z_i}} = \mathbf{y}^n|_{S_{i Z_i}}) \ge n(r_i - \lambda') - \frac{2^{n(r_i - \lambda') - n r_i}}{\ln 2}$$
$$= n(r_i - \lambda') - \frac{2^{-n \lambda'}}{\ln 2}.$$
Then, (41) tends to:
$$\lim_{n \rightarrow \infty} I(M_{i Z_i}; V)_{i \in \{1, 2\}} \le \lim_{n \rightarrow \infty} \left[ n(r_i - \lambda') - n(r_i - \lambda') + \frac{2^{-n \lambda'}}{\ln 2} \right]$$
$$= \lim_{n \rightarrow \infty} \frac{2^{-n \lambda'}}{\ln 2} = 0.$$
Then, from (40), we have $\lim_{n \rightarrow \infty} I(M_{1 Z_1}, M_{2 Z_2}; V) = 0$.

*   $$I(Z_i; U_i) = I(Z_i; M_{i 0}, M_{i 1}, X^n, R_{A_i}, C_i)$$
$$= I(Z_i; M_{i 0}, M_{i 1}, X^n, S_{i 0}, S_{i 1}, \kappa_{i 0}, \kappa_{i 1}, M_{i 0} \oplus \kappa_{i 0}(\mathbf{X}_i|_{S_{i 0}}), M_{i 1} \oplus \kappa_{i 1}(\mathbf{X}_i|_{S_{i 1}}), R_{A_i})$$
$$\overset{(a)}{=} I(Z_i; M_{i 0}, M_{i 1}, X^n, S_{i 0}, S_{i 1}, \kappa_{i 0}, \kappa_{i 1}, \kappa_{i 0}(\mathbf{X}_i|_{S_{i 0}}), \kappa_{i 1}(\mathbf{X}_i|_{S_{i 1}}), R_{A_i})$$
$$\overset{(b)}{\le} I(Z_i; X^n, S_{i 0}, S_{i 1})$$
$$\overset{(c)}{\le} I(Z_i; S_{i 0}, S_{i 1})$$
$$= 0,$$
where $R_{A_i} = (\mathcal{R}^{(i)} = (R_{i 0}, R_{i 1}), \mathcal{T}^{(i)} = (T_{i 0}, T_{i 1}))$, $(a)$ follows since $M_{i 0}, M_{i 1}, \kappa_{i 0}, \kappa_{i 1} \perp (Z_i, X^n, S_{i 0}, S_{i 1}, R_{A_i})$, $(b)$ follows since $X_i^n \perp (Z_i, S_{i 0}, S_{i 1})$, and $(c)$ follows since the channel acts independently on each input bit and $|S_{i 0}| = |S_{i 1}|$. $\square$

## APPENDIX G
### PROOF OF THEOREM 13

$$H(S_i) = H(S_i|X_i)$$
$$= I(S_i; \mathcal{C}^l, E|X_i) + H(S_i|\mathcal{C}^l, E, X_i).$$

Consider the last term of the above expression,
$$H(S_i|\mathcal{C}_i^l, E, X_i) = H(S_i, X_i|\mathcal{C}_i^l, E, X_i) - H(X_i|S_i, \mathcal{C}_i^l, E, X_i)$$
$$= H(X_i|\mathcal{C}_i^l, E, X_i) + H(S_i|\mathcal{C}_i^l, E, X_i, X_i) - H(X_i|S_i, \mathcal{C}_i^l, E, X_i)$$
$$\overset{(a)}{=} H(X_i|\mathcal{C}_i^l, E, X_{\bar{i}}) - H(X_i|S_i, \mathcal{C}_i^l, E, X_{\bar{i}})$$
$$\overset{(b)}{\le} H(X_i|\mathcal{C}_i^l, E, X_{\bar{i}}) - H(X_i|S_i, \mathcal{C}_i^l, E, X_{\bar{i}}, Y)$$
$$\overset{(c)}{=} H(X_i|\mathcal{C}_i^l, E, X_{\bar{i}}) - H(X_i, S_i|\mathcal{C}_i^l, E, Y, X_{\bar{i}}) + H(S_i|\mathcal{C}_i^l, E, Y, X_{\bar{i}})$$
$$\overset{(c)}{=} H(X_i|\mathcal{C}_i^l, E, X_{\bar{i}}) - H(X_i|\mathcal{C}_i^l, E, Y, X_{\bar{i}}) + H(S_i|\mathcal{C}_i^l, E, Y, X_{\bar{i}})$$
$$\overset{(d)}{\le} I(X_i; Y|X_{\bar{i}}, \mathcal{C}_i^l, E) + H(S_i|\mathcal{C}_i^l, E, Y)$$
$$\overset{(e)}{\le} I(X_i; Y|X_{\bar{i}}, \mathcal{C}_i^l, E) + H(S_i|S'_i),$$
for $i \in \{1, 2\}$, where $(a)$ and $(c)$ follow from the condition (11), $(b)$ is due to the fact that conditioning does not increase the entropy, $(d)$ is due to the fact that $S_i \perp X_{\bar{i}}$, and $(e)$ follows from:
$$H(S_i, S_{\bar{i}}|\mathcal{C}^l, Y, E) = H(S_i|\mathcal{C}^l, Y, E) + H(S_{\bar{i}}|S_i, \mathcal{C}^l, Y, E)$$
$$= H(S_i|\mathcal{C}^l, Y, E) + H(S_{\bar{i}}|S'_i, \mathcal{C}^l, Y, E).$$
Considering the condition (13), implies that:
$$H(S_i|\mathcal{C}^l, Y, E) = H(S_i|S'_i, \mathcal{C}^l, Y, E) \le H(S_i|S'_i).$$
Putting everything together completes the proof for the individual rates.
For the joint entropy, by the same reasoning as above, we have:
$$H(S_1, S_2) = I(S_1, S_2; \mathcal{C}_1^l, \mathcal{C}_2^l, E) + H(S_1, S_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E). \quad (42)$$
Consider the last term of the above expression,
$$H(S_1, S_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E) = H(S_1, S_2, X_1, X_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E) - H(X_1, X_2|S_1, S_2, \mathcal{C}_1^l, \mathcal{C}_2^l, E)$$
$$= H(X_1, X_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E) + H(S_1, S_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E, X_1, X_2) - H(X_1, X_2|S_1, S_2, \mathcal{C}_1^l, \mathcal{C}_2^l, E)$$
$$= H(X_1, X_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E) - H(X_1, X_2|S_1, S_2, \mathcal{C}_1^l, \mathcal{C}_2^l, E)$$
$$\le H(X_1, X_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E) - H(X_1, X_2|S_1, S_2, \mathcal{C}_1^l, \mathcal{C}_2^l, E, Y)$$
$$= H(X_1, X_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E) - H(X_1, X_2, S_1, S_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E, Y) + H(S_1, S_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E, Y)$$
$$= H(X_1, X_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E) - H(X_1, X_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E, Y) + H(S_1, S_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E, Y)$$
$$\overset{(a)}{\le} I(X_1, X_2; Y|\mathcal{C}_1^l, \mathcal{C}_2^l, E) + H(S_1, S_2|\mathcal{C}_1^l, \mathcal{C}_2^l, E, Y)$$
$$\le I(X_1, X_2; Y|\mathcal{C}_1^l, \mathcal{C}_2^l, E) + H(S_1, S_2|S'_1, S'_2), \quad (43)$$
where $(a)$ follows from (12). $\square$

## APPENDIX H
### PROOF OF COROLLARY 14

Alice-1 and Alice-2 each independently generate uniformly distributed keys $S_1$ and $S_2$, respectively. They then produce channel inputs as stochastic functions of these keys, resulting in $X_1^n = f_1(S_1)$ and $X_2^n = f_2(S_2)$. These inputs are sent over the DM-MAC. The outputs $Y^n$ and $E^n$ are subsequently received by Bob and Eve, respectively. Following this, Alice-$i$ generates $C_i = f_i(S_i, E^n)$. These functions ($C_i, i \in \{1, 2\}$) are then transmitted over the public channel so that the receiver can reconstruct the keys. It is important to note that all the functions mentioned above are stochastic. According to Fano’s inequality, for any arbitrarily small $\epsilon \ge 0$, we have:
$$H(S_1, S_2| Y^n, C_1, C_2) \le h(\epsilon) + \epsilon(H(S_1) + H(S_2))$$
$$\le h(\epsilon) + \epsilon(n R_1 + n R_2 + 2 n \epsilon)$$
$$\le n \left( \frac{h(\epsilon)}{n} + \epsilon(R_1 + R_2 + 2 n \epsilon) \right) \approx n \epsilon'.$$

It is clear that $\epsilon' \rightarrow 0$ if $\epsilon \rightarrow 0$. Also, two security criteria should be fulfilled for arbitrarily small $\epsilon \ge 0$: $I(S_1; E^n C_1) \le n\epsilon$, $I(S_2; E^n C_2) \le n\epsilon$ (Condition (15)):
$$R_i \le \frac{1}{n} H(S_i) + \epsilon$$
$$\overset{(a)}{\le} \frac{1}{n} H(S_i | E^n, C_i) + 2\epsilon$$
$$= \frac{1}{n} H(S_i | X_{\bar{i}}^n, E^n, C_i) + 2\epsilon$$
$$\overset{(b)}{\le} \frac{1}{n} H(S_i | X_{\bar{i}}^n, E^n, C_i) - \frac{1}{n} H(S_i | Y^n, C_{\bar{i}}, C_i) + 2\epsilon + \epsilon'$$
$$\overset{(c)}{\le} \frac{1}{n} H(S_i | X_{\bar{i}}^n, E^n, C_i) - \frac{1}{n} H(S_i | Y^n, X_{\bar{i}}^n, E^n, C_{\bar{i}}, C_i) + 2\epsilon + \epsilon'$$
$$\overset{(d)}{=} \frac{1}{n} H(S_i | X_{\bar{i}}^n, E^n, C_i) - \frac{1}{n} H(S_i | Y^n, X_{\bar{i}}^n, E^n, C_i) + 2\epsilon + \epsilon'$$
$$= \frac{1}{n} I(S_i; Y^n | X_{\bar{i}}^n, E^n, C_i) + 2\epsilon + \epsilon'$$
$$= \frac{1}{n} H(Y^n | X_{\bar{i}}^n, E^n, C_i) - \frac{1}{n} H(Y^n | X_{\bar{i}}^n, E^n, C_i, S_i) + 2\epsilon + \epsilon'$$
$$\overset{(e)}{\le} \frac{1}{n} H(Y^n | X_{\bar{i}}^n, E^n) - \frac{1}{n} H(Y^n | X_i^n, X_{\bar{i}}^n, C_{\bar{i}}, C_i, E^n, S_i) + 2\epsilon + \epsilon'$$
$$\overset{(f)}{=} \frac{1}{n} H(Y^n | X_{\bar{i}}^n, E^n) - \frac{1}{n} H(Y^n | X_i^n, X_{\bar{i}}^n, E^n) + 2\epsilon + \epsilon'$$
$$\le \max_{P_{X_1} P_{X_2}} I(X_i; Y | X_{\bar{i}}, E) + 2\epsilon + \epsilon', \quad (44)$$
for $i \in \{1, 2\}$, where $(a)$ follows from the security criteria, $(b)$ follows from Fano's lemma, $(c)$ and $(d)$ follow from the fact that $C_i = f_i(S_i, E^n)$, $(e)$ follows from the Markov chain $Y^n - (X_i^n, X_{\bar{i}}^n) - (S_1, S_2)$, and $(f)$ follows from the memoryless property of the channel and an argument similar to [42, Theorem 4]. As is mentioned before and proved in [42] for the case of constant random variable $E$, we can remove the impact of $E$ from the above mutual information quantity. The whole above process can similarly be repeated for the joint entropy $H(S_1, S_2)$. This completes the proof. $\square$

## APPENDIX I
### PROOF OF THEOREM 15

A. The first upper bound on OT capacity

Consider the system model illustrated in Figure 2-(a). To prove the upper bound for $(\frac{1}{2})-OT^{k_1, k_2}$ capacity, consider that $(n, k_1, k_2)$ protocols fulfilling conditions (6)-(8). Condition (7) can be rewritten as:
$$\lim_{n \rightarrow \infty} I(M_{1 Z_1}, M_{2 Z_2}; V) = \lim_{n \rightarrow \infty} I(M_{1 Z_1}; V) + \lim_{n \rightarrow \infty} I(M_{2 Z_2}; V | M_{1 Z_1}) \quad (45)$$
$$= \lim_{n \rightarrow \infty} I(M_{1 Z_1}; V) + \lim_{n \rightarrow \infty} I(M_{2 Z_2}; V) \quad (46)$$
$$= 0,$$
This means that $\lim_{n \rightarrow \infty} I(M_{i Z_i}; V)_{i \in \{1, 2\}} = 0$ and its relaxed version: $I(M_{i Z_i}; V)_{i \in \{1, 2\}} = o(n)$:
$$I(M_{i Z_i}; V)_{i \in \{1, 2\}} = I(M_{i Z_i}; Z_i, \bar{Z}_i, R_B, Y^n, C)_{i \in \{1, 2\}} = o(n). \quad (47)$$
$$I(Z_i, \bar{Z}_i, R_B, Y^n, C; M_{i Z_i})_{i \in \{1, 2\}}$$
$$= I(Z_i, \bar{Z}_i; M_{i Z_i})_{i \in \{1, 2\}} + I(R_B, Y^n, C; M_{i Z_i} | Z_i, \bar{Z}_i)_{i \in \{1, 2\}}$$
$$\overset{(a)}{\le} I(R_B, Y^n, C; M_{i Z_i} | Z_i, \bar{Z}_i)_{i \in \{1, 2\}},$$
where $(a)$ follows from the fact that $I(Z_i, \bar{Z}_i; M_{i Z_i})_{i \in \{1, 2\}} = 0$. So, Condition (47) implies that $I(R_B, Y^n, C; M_{i Z_i}, Z_i, \bar{Z}_i)_{i \in \{1, 2\}} \rightarrow 0$. Instead of using Condition (47), we have:
$$I(R_B, Y^n, C; M_{\bar{Z}_i} | Z_i, \bar{Z}_i)_{i \in \{1, 2\}} = o(n). \quad (48)$$

Given a DM-MAC $\{W : \mathcal{X}_1 \mathcal{X}_2 \rightarrow \mathcal{Y}\}$, consider $(\frac{1}{2})-OT^{k_1, k_2}$ protocols that satisfy (6), (8), and (48). According to Lemma 5, Condition (8) implies:

$$H(M_{i Z_i} | X_{\bar{i}}^n, C_i, Z_i = z_i)_{i \in \{1, 2\}} - H(M_{i Z_i} | X_{\bar{i}}^n, C_i, Z_i = z_i)_{i \in \{1, 2\}}$$
$$= H(M_{i Z_i} | C_i, Z_i = z_i)_{i \in \{1, 2\}} - H(M_{i Z_i} | C_i, Z_i = z_i)_{i \in \{1, 2\}}$$
$$\overset{(a)}{\le} H(M_{i Z_i} | C_i, Z_i = z_i)_{i \in \{1, 2\}} - H(M_{i Z_i} | C_i, Z_i = z_i)_{i \in \{1, 2\}}$$
$$- H(M_{i Z_i} | Z_i = z_i)_{i \in \{1, 2\}}$$
$$+ H(M_{i Z_i} | Z_i = z_i)_{i \in \{1, 2\}}$$
$$= I(M_{i Z_i}; C_i | Z_i = z_i)_{i \in \{1, 2\}} - I(M_{i Z_i}; C_i | Z_i = z_i)_{i \in \{1, 2\}}$$
$$= o(n), \quad (50)$$
where $(a)$ follows from the fact that $H(M_{i Z_i} | Z_i = z_i) = H(M_{i Z_i} | Z_i = z_i) = k_i$.
Suppose $Z_i = z_i$ and $\bar{Z}_i = \bar{z}_i$ in (48), then we have $I(R_B, Y^n, C_i; M_{\bar{Z}_i} | Z_i = z_i, \bar{Z}_i = \bar{z}_i)_{i \in \{1, 2\}} = o(n)$ combined with (50) and (51) concludes:
$$I(M_{i Z_i}; C_i | Z_i = z_i, \bar{Z}_i = \bar{z}_i)_{i \in \{1, 2\}} = o(n). \quad (52)$$
Conditions (6) and (52) without conditioning on $(Z_i = z_i, \bar{Z}_i = \bar{z}_i)_{i \in \{1, 2\}}$ are akin to those defining a secret key for Alice-$i$ and Bob with weak secrecy (conditions (14) and (15), respectively), ensuring security from an eavesdropper who observes their public communication $C_i$. So, $k_i$ would constitute such a secret key by definition, as demonstrated in Corollary 14. Thus, we get:
$$k_i = H(M_{i Z_i}) \le \sum_{l=1}^n I(X_{i, l}; Y_l | X_{\bar{i}, l}) + o(n). \quad (53)$$
From the memoryless property of the channel we have: $\frac{k_i}{n} = R_i \le I(X_i; Y | X_{\bar{i}})$. By a similar calculation for the joint entropy where $k_1 + k_2 = H(M_{1 Z_1}, M_{2 Z_2}) \le \sum_{l=1}^n I(X_{1, l}, X_{2, l}; Y_l) + o(n)$ and from the memoryless property of the channel we have: $\frac{k_1+k_2}{n} = R_1 + R_2 \le I(X_1, X_2; Y)$.

B. The second upper bound on OT capacity

At the first step, we prove that $I(M_{\bar{Z}_i}; Y^n, R_B | X^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) = 0$ for $i \in \{1, 2\}$. This means that there is no mutual information between Bob's received messages and his received sequence given his chosen bits, Alice's encoded strings, and the total public transmission. Define for $i \in \{1, 2\}$, $l \in [1, n]$, $t \in [1, r_l]$, $C_{i, l, 1:t} \equiv (C_{0, i, l}(j), C_{i, l}(j))_{j \in [1, t]}$ as the messages exchanged between Alice-$i$ and Bob between the first and the $t$-th communication rounds occurring after the $l$-th channel usage. Let $C^l \equiv (C_{i, j, 1:r_j})_{j \in [1, l]}$ represent all messages exchanged by Server 1 with the client before the $l+1$-th channel use. Let $l \in [1, n]$ and $j \in [1, r_l]$. Then, we have:
$$I(M_{i 0}, M_{i 1}, R_{A_i}; Y^l, R_B | X_i^l, C^{l-1}, C_{i, l, 1:j}, Z_i, \bar{Z}_i)$$
$$\overset{(a)}{\le} I(M_{i 0}, M_{i 1}, R_{A_i}; Y^l, R_B, C_{0, i, l}(j) | X_i^l, C^{l-1}, C_{i, l, 1:(j-1)}, C_{i, l}(j), Z_i, \bar{Z}_i) \quad (54)$$
$$\le I(M_{i 0}, M_{i 1}, R_{A_i}; Y^l, R_B | X_i^l, C^{l-1}, C_{i, l, 1:(j-1)}, Z_i, \bar{Z}_i)$$
$$\overset{(c)}{\le} I(M_{i 0}, M_{i 1}, R_{A_i}; Y^l, R_B | X_i^l, C^{l-1}, C_{i, l, 1:(j-1)}, Z_i, \bar{Z}_i)$$
$$\overset{(d)}{\le} I(M_{i 0}, M_{i 1}, R_{A_i}; Y^l, R_B | X_i^l, C^{l-1}, Z_i, \bar{Z}_i)$$
$$\overset{(e)}{\le} I(M_{i 0}, M_{i 1}, R_{A_i}; Y^{l-1}, R_B | X_i^{l-1}, C^{l-1}, Z_i, \bar{Z}_i)$$
$$\le I(M_{i 0}, M_{i 1}, R_{A_i}, X_{i, l}; Y^{l-1}, R_B | X_i^{l-1}, C^{l-1}, Z_i, \bar{Z}_i)$$
$$\overset{(f)}{\le} I(M_{i 0}, M_{i 1}, R_{A_i}; Y^{l-1}, R_B | X_i^{l-1}, C^{l-1}, Z_i, \bar{Z}_i)$$
$$\overset{(g)}{\le} 0, \quad (57)$$
where the steps are justified as follows:
*   $(a)$ follows by the definition of $C_{i, l, 1:j} = (C_{i, l, 1:(j-1)}, C_{0, i, l}(j), C_{1, i, l}(j))$ and by the chain rule.
*   $(b)$ follows by the chain rule and because $C_{0, i, l}(j)$ depends on $(Z_i, \bar{Z}_i, R_B, Y^l, C_{i, l}(j), C_{i, l, 1:(j-1)})$.
*   $(c)$ follows by the chain rule and because $C_{i, l}(j)$ depends on $(M_{i 0}, M_{i 1}, R_{A_i}, C_{i, l, 1:(j-1)}, C^{l-1})$.
*   $(d)$ follows by previous iterations ($j - 1$) of Equations (54) to (55).
*   $(e)$ follows from the Markov chain: $(M_{i 0}, M_{i 1}, R_{A_i}) - (Y^{l-1}, R_B, X_i^l, C^{l-1}, Z_i, \bar{Z}_i) - Y_l$.
*   $(f)$ follows the chain rule and the fact that $X_{i, l}$ is a function of $(M_{i 0}, M_{i 1}, R_{A_i}, C^{l-1})$.
*   $(g)$ follows from the following calculation:
$$I(M_{i 0}, M_{i 1}, R_{A_i}; Y^l, R_B | X_i^l, C^l, Z_i, \bar{Z}_i) \le I(M_{i 0}, M_{i 1}, R_{A_i}; Y^{l-1}, R_B | X_i^{l-1}, C^{l-1}, Z_i, \bar{Z}_i) \quad (58)$$
$$\le I(M_{i 0}, M_{i 1}, R_{A_i}; R_B | Z_i, \bar{Z}_i) \quad (59)$$
$$= 0. \quad (60)$$
Then, for any $z_i \in \{0, 1\}, i \in \{1, 2\}$ we have:
$$I(M_{\bar{Z}_i}; Y^n, R_B | X^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) \le I(M_{i 0}, M_{i 1}, R_{A_i}; Y^n, R_B | X^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i)$$
$$= 0. \quad (61)$$
Now we can prove that for any $z_i \in \{0, 1\}$, we have $H(M_{\bar{Z}_i}|X^n, C_i, Z_i, \bar{Z}_i) = o(n), i \in \{1, 2\}$ which means that the uncertainty about the unchosen messages given the encoded inputs, Bob's inputs and the total public communication is negligible as $n \rightarrow \infty$:
$$H(M_{\bar{Z}_i}|X^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) \le H(M_{\bar{Z}_i}|X^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i)$$
$$\overset{(a)}{\le} H(M_{\bar{Z}_i}|X^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$\overset{(b)}{\le} H(M_{\bar{Z}_i}|Y^n, R_B, X^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$\overset{(c)}{\le} H(M_{\bar{Z}_i}|Y^n, R_B, M_{Z_i}, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$\overset{(d)}{\le} o(n).$$
Finally,
$$H(M_{\bar{Z}_i}|X^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) = \sum_{z_i, \bar{z}_i} P[(Z_i, \bar{Z}_i) = (z_i, \bar{z}_i)] \times H(M_{\bar{Z}_i}|X^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i)$$
$$= o(n), \quad (62)$$
where $(a)$ follows from (51), $(b)$ follows from (61), $(c)$ holds because $M_{Z_i}$ is a function of $(Y^n, R_B, C_i)$, and $(d)$ holds by Fano's inequality and (6).
Now, we have: we have:
$$k_i = H(M_{i Z_i} | Z_i = z_i, \bar{Z}_i = \bar{z}_i) = H(M_{i Z_i} | R_B, Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$= H(M_{i Z_i}, R_B | Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i)$$
$$+ H(X_i^n | M_{i Z_i}, R_B, Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i)$$
$$- H(X_i^n | M_{i Z_i}, R_B, Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$= H(M_{i Z_i}, X_i^n | R_B, Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i)$$
$$- H(X_i^n | M_{i Z_i}, R_B, Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$\le H(M_{i Z_i}, X_i^n | R_B, Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$= H(X_i^n | R_B, Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i)$$
$$+ H(M_{i Z_i} | X_i^n, R_B, Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$\le H(X_i^n | R_B, Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i)$$
$$+ H(M_{i Z_i} | X_i^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$\overset{(a)}{\le} H(X_i^n | R_B, Y^n, C_i, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$\overset{(b)}{\le} H(X_i^n | Y^n, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$\le \sum_{l=[1:n]} H(X_{i, l} | Y_l, Z_i = z_i, \bar{Z}_i = \bar{z}_i) + o(n)$$
$$\overset{(c)}{=} \sum_{l=1}^n H(X_{i, l} | Y_l) + o(n),$$
where $(a)$ follows from (62), $(b)$ is because conditioning does not increase the entropy, and $(c)$ follows from the arguments presented in [5]. Similarly, we can prove that $k_1 + k_2 = H(M_{1 Z_1}, M_{2 Z_2} | Z_1 = z_1, Z_2 = z_2) \le \sum_{l=1}^n H(X_{1, l}, X_{2, l} | Y_l) + o(n)$.
Then, the final upper bound presented in Theorem 15 is proved. $\square$

## APPENDIX J
### PROOF OF THEOREM 16

We must bound three conditional min-entropies to get three bounds on the lower bound:
$$H_{\infty}^{\epsilon}(X_1|_{S_{1 Z_1}} | h_{1 0}(R_{1 0}, X_1|_{S_{1 0}}), h_{1 1}(R_{1 1}, X_1|_{S_{1 1}}), Y^n, R^{(1)}, \mathcal{T}^{(1)}), \quad (63)$$
$$H_{\infty}^{\epsilon}(X_2|_{S_{2 Z_2}} | h_{2 0}(R_{2 0}, X_2|_{S_{2 0}}), h_{2 1}(R_{2 1}, X_2|_{S_{2 1}}), Y^n, R^{(2)}, \mathcal{T}^{(2)}), \quad (64)$$
$$H_{\infty}^{\epsilon}(X_1|_{S_{1 Z_1}}, X_2|_{S_{2 Z_2}} | h_{1 0}(R_{1 0}, X_1|_{S_{1 0}}), h_{1 1}(R_{1 1}, X_1|_{S_{1 1}}), h_{2 0}(R_{2 0}, X_2|_{S_{2 0}}), h_{2 1}(R_{2 1}, X_2|_{S_{2 1}}), Y^n, R_{A_1}, R_{A_2}). \quad (65)$$

Consider (63). Since the $\text{SU-SBC}_{p, W}$ is i.i.d., we have:
$$H_{\infty}^{\epsilon}(X_1|_{S_{1 Z_1}} | h_{1 0}(R_{1 0}, X_1|_{S_{1 0}}), h_{1 1}(R_{1 1}, X_1|_{S_{1 1}}), Y^n, R^{(1)}, \mathcal{T}^{(1)})$$
$$= H_{\infty}^{\epsilon}(X_1|_{S_{1 Z_1}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)}).$$
Applying (1) for $\epsilon, \epsilon' > 0$, we have:
$$H^{\epsilon + \epsilon'}_{\infty}(X_1|_{S_{1 Z_1}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)})$$
$$\ge H_{\infty}(X_1|_{S_{1 Z_1}} | Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)}) + H^{\epsilon}_{\infty}(h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}) | X_1|_{S_{1 Z_1}}, Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)})$$
$$- H_{0}(h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}) | Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)}) - \log \left(\frac{1}{1-\epsilon'}\right). \quad (66)$$
Note that $H_{0}(h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}) | Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)})$ limits the number of distinct possible outputs, restricting the amount of information Alice-1 can gain about Bob's choice:
$$H_{0}(h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}) | Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)}) \le s_1 n.$$
Knowing that $H_{0}(h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}) | X_1|_{S_{1 Z_1}}, Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)}) = 0$, (66) is simplified to:
$$H^{\epsilon + \epsilon'}_{\infty}(X_1|_{S_{1 Z_1}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)}) \ge H_{\infty}(X_1|_{S_{1 Z_1}} | Y|_{S_{1 Z_1}}^n) - s_1 n - \log \left(\frac{1}{1-\epsilon'}\right)$$
$$\overset{(a)}{>} H^{\epsilon}_{\infty}(X_1|_{S_{1 Z_1}} | Y|_{S_{1 Z_1}}^n) - s_1 n - \log \left(\frac{1}{1-\epsilon'}\right) - \log \left(\frac{1}{\epsilon}\right), \quad (67)$$
where $(a)$ follows from Lemma 1. Let $V$ be an i.i.d. random variable so that $V = e$ (erasure) with probability $\frac{1}{2} - \eta$ and $V = Z$ (the output of channel $W$ on input $(X_1, X_2)$). With negligible error probability and $\text{SU-SBC}_{p, W}$ being i.i.d., for $S_{1 Z_1}$, we have:
$$H^{\epsilon}_{\infty}(X_1|_{S_{1 Z_1}} | Y|_{S_{1 Z_1}}^n) \ge H^{\epsilon}_{\infty}(X_1|_{S_{1 Z_1}} | |V|_{S_{1 Z_1}}|)$$
$$\overset{(a)}{\ge} |S_{1 Z_1}| H(X_1 | V) - 4 \sqrt{|S_{1 Z_1}|} \log(1/\epsilon) \log|\mathcal{X}_1|$$
$$\ge (p - \eta)n H(X_1 | V) - 4 \sqrt{(p - \eta)n} \log(1/\epsilon)$$
$$\ge p n H(X_1 | V) - \eta n H(X_1 | V) - 4 \sqrt{(p - \eta)n} \log(1/\epsilon)$$
$$\ge p n H(X_1 | V) - \eta n - 4 \sqrt{(p - \eta)n} \log(1/\epsilon)$$
$$\overset{(b)}{\ge} p \eta (1 - 2\eta) H(X_1) + 2 \eta n H(X_1 | Z) - \eta n - 4 \sqrt{(p - \eta)n} \log(1/\epsilon)$$
$$\ge p \eta H(X_1) - 2 \eta n - 4 \sqrt{n} \log(1/\epsilon), \quad (68)$$
where $(a)$ follows from Lemma 2, $|\mathcal{X}_1| = 2$, and $(b)$ follows from this fact that honest Bob doesn't split the erasures received from Alice-1 between $S_{1 0}$ and $S_{1 1}$, with probability exponentially close to one, the total number of non-erased symbols Bob receives from each sender will not exceed $(p + \eta)n$, so the number of non-erasures in $Y|_{S_{1 Z_1}}$ is at most $|(p - \eta)n - (p + \eta)n| = 2 \eta n$.
Putting $\epsilon = \epsilon + \epsilon'$ in (67), then putting (68) to (67), we have:
$$H^{\epsilon + 2\epsilon'}_{\infty}(X_1|_{S_{1 Z_1}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)}) \ge p n(H(X_1) + H(X_1 | Z)) - 2 \eta n - 4 \sqrt{n} \log(1/\epsilon) - s_1 n - \log \left(\frac{1}{1-\epsilon'}\right) - \log \left(\frac{1}{\epsilon}\right)$$
$$\overset{(a)}{\approx} p \eta H(X_1) - s_1 n - 2 \eta n - 4 \sqrt{n} \alpha - n(\alpha + \alpha'), \quad (69)$$
where $(a)$ follows from setting $\epsilon = 2^{- \alpha n}$ and $\epsilon' = 2^{- \alpha' n}$ ($\epsilon$ and $\epsilon'$ are negligible in $n$). For any $\delta \ge (\alpha + \alpha' + 2\eta + 4\sqrt{\alpha}) > 0$, we have:
$$H^{\epsilon + 2\epsilon'}_{\infty}(X_1|_{S_{1 Z_1}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)}) \ge p n H(X_1) - s_1 n - \delta n.$$
From Lemma 3, we know that if we set $r_1 < p H(X_1) - s_1$ and appropriately choose the constant $\delta, \eta, \alpha$ and $\alpha'$, Bob can not obtain non-trivial information about the unselected string. The proof for Alice-2 is the same.
Now we consider (65). Since the $\text{SU-SBC}_{p, W}$ is i.i.d., (65) can be written as:
$$H_{\infty}^{\epsilon}(X_1|_{S_{1 Z_1}}, X_2|_{S_{2 Z_2}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), h_{2 j}(R_{2 j}, X_2|_{S_{2 Z_2}}), Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2})$$
$$\overset{(a)}{\ge} H_{\infty}^{\epsilon}(X_1|_{S_{1 Z_1}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), h_{2 j}(R_{2 j}, X_2|_{S_{2 Z_2}}), Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2})$$
$$+ H_{\infty}^{\epsilon}(X_2|_{S_{2 Z_2}} | X_1|_{S_{1 Z_1}}, h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), h_{2 j}(R_{2 j}, X_2|_{S_{2 Z_2}}), Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2})$$
$$\overset{(b)}{\ge} H_{\infty}^{\epsilon}(X_1|_{S_{1 Z_1}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), h_{2 j}(R_{2 j}, X_2|_{S_{2 Z_2}}), Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2}) - \log \left(\frac{1}{\epsilon'}\right)$$
$$+ H_{\infty}^{\epsilon}(X_2|_{S_{2 Z_2}} | X_1|_{S_{1 Z_1}}, h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), h_{2 j}(R_{2 j}, X_2|_{S_{2 Z_2}}), Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2}) - \log \left(\frac{1}{\epsilon}\right)$$
where $(a)$ is due to (3), and $(b)$ is due to Lemma 1. Applying (1) $\epsilon, \epsilon' > 0$ for each terms, we have:
$$H^{\epsilon + \epsilon'}_{\infty}(X_1|_{S_{1 Z_1}}, X_2|_{S_{2 Z_2}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), h_{2 j}(R_{2 j}, X_2|_{S_{2 Z_2}}), Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2})$$
$$\overset{(a)}{\ge} H_{\infty}^{\epsilon}(X_1|_{S_{1 Z_1}} | Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2})$$
$$+ H_{\infty}^{\epsilon}(h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}) | X_1|_{S_{1 Z_1}}, Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2})$$
$$\overset{(b)}{\ge} - H_{0}(h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}) | Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2}) - \log \left(\frac{1}{1-\epsilon'}\right)$$
$$+ H_{\infty}^{\epsilon}(X_2|_{S_{2 Z_2}} | Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2})$$
$$+ H_{\infty}^{\epsilon}(h_{2 j}(R_{2 j}, X_2|_{S_{2 Z_2}}) | X_1|_{S_{1 Z_1}}, Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2})$$
$$- H_{0}(h_{2 j}(R_{2 j}, X_2|_{S_{2 Z_2}}) | Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2}) - \log \left(\frac{1}{1-\epsilon'}\right)$$
$$\overset{(c)}{\ge} H_{\infty}^{\epsilon}(X_1|_{S_{1 Z_1}} | Y|_{S_{1 Z_1}, S_{2 Z_2}}^n) + H_{\infty}^{\epsilon}(X_2|_{S_{2 Z_2}} | X_1|_{S_{1 Z_1}}, Y|_{S_{1 Z_1}, S_{2 Z_2}}^n)$$
$$- s_1 n - s_2 n - 2 \log \left(\frac{1}{1-\epsilon'}\right) - 2 \log \left(\frac{1}{\epsilon}\right) \quad (70)$$
where $(a)$ is due to (1) and the independence of $X_i$ from $h_{i j}$, and $(b)$ is due to the similar simplification as (66)-(67). The first term of (70) can be bounded similarly to (68). For the second term, we have:
$$H^{\epsilon + \epsilon'}_{\infty}(X_2|_{S_{2 Z_2}} | X_1|_{S_{1 Z_1}}, Y|_{S_{1 Z_1}, S_{2 Z_2}}^n) \ge H_{\infty}^{\epsilon}(X_2|_{S_{2 Z_2}} | |X_1|_{S_{1 Z_1}}, V||_{S_{1 Z_1}, S_{2 Z_2}}|)$$
$$\overset{(a)}{\ge} |S_{2 Z_2}| H(X_2 | X_1, V) - 4 \sqrt{|S_{2 Z_2}|} \log(1/\epsilon) \log|\mathcal{X}_2|$$
$$\ge (p - \eta)n H(X_2 | X_1, V) - 4 \sqrt{(p - \eta)n} \log(1/\epsilon)$$
$$\ge p n H(X_2 | X_1, V) - \eta n H(X_2 | X_1, V) - 4 \sqrt{(p - \eta)n} \log(1/\epsilon)$$
$$\ge p \eta (1 - 2 \eta) H(X_2 | X_1) + 2 \eta n H(X_2 | X_1, Z) - \eta n - 4 \sqrt{(p - \eta)n} \log(1/\epsilon)$$
$$\ge p \eta H(X_2) - 2 \eta n - 4 \sqrt{n} \log(1/\epsilon), \quad (71)$$
where $(a)$ is due to Lemma 2 and this fact that in Protocol 1, $|S_{1 Z_1}| = |S_{2 Z_2}| = (p - \eta)n$.

Then by putting $\epsilon = \epsilon + \epsilon'$, (70) is as follows:
$$H^{\epsilon + 2\epsilon'}_{\infty}(X_1|_{S_{1 Z_1}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), h_{2 j}(R_{2 j}, X_2|_{S_{2 Z_2}}), Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2})$$
$$+ H^{\epsilon + \epsilon'}_{\infty}(X_2|_{S_{2 Z_2}} | X_1|_{S_{1 Z_1}}, h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), h_{2 j}(R_{2 j}, X_2|_{S_{2 Z_2}}), Y|_{S_{1 Z_1}, S_{2 Z_2}}^n, R_{A_1}, R_{A_2}) - \log \left(\frac{1}{\epsilon}\right)$$
$$\overset{(a)}{\ge} p n(H(X_1) + H(X_2)) - 4 \eta n - 8 \sqrt{n} \log(1/\epsilon) - s_1 n - s_2 n - 2 \log \left(\frac{1}{1-\epsilon'}\right) - 2 \log \left(\frac{1}{\epsilon}\right)$$
$$\approx p n(H(X_1) + H(X_2)) - s_1 n - s_2 n - 2 \delta n, \quad (72)$$
for any $\delta \ge (\alpha + \alpha' + 2\eta + 4\sqrt{\alpha}) > 0$, $(a)$ follows from setting $\epsilon = 2^{- \alpha n}$ and $\epsilon' = 2^{- \alpha' n}$ ($\epsilon$ and $\epsilon'$ are negligible in $n$).
From Lemma 3, we know that if we set $r_1 + r_2 < p(H(X_1) + H(X_2)) - s_1 - s_2$ and appropriately choose the constant $\delta, \eta, \alpha$ and $\alpha'$, Bob can not obtain non-trivial information about the unselected strings.

Now, we find the appropriate $s_1$ and $s_2$ under which the protocol remains correct and secure. Due to the Chernoff bound, we know that the probability of aborting the protocol by Bob in step (2) tends to zero as $n \rightarrow \infty$. The protocol fails in step (5), if there is more than one pair, such as $(x_1|_{S_{1 Z_1}}, x_2|_{S_{2 Z_2}})$ where $h_i(\mathbf{X}_i|_{S_{i Z_i}}) = h_i(\mathbf{X}'_i|_{S_{i Z_i}})$. We know that if all players are honest, then $Z|_{S_{1 Z_1}, S_{2 Z_2}} = Y|_{S_{1 Z_1}, S_{2 Z_2}}$ with probability exponentially close to one and the number of paired sequences $(x_1|_{S_{1 Z_1}}, x_2|_{S_{2 Z_2}})$ jointly typical with $z|_{S_{1 Z_1}, S_{2 Z_2}}$ can be upper bounded as follows:

*   If one of the sequences $x_i|_{S_{i Z_i}}$ is not typical with $(X_{\bar{i}}|_{S_{\bar{i} Z_{\bar{i}}}}, Z|_{S_{1 Z_1}, S_{2 Z_2}}): 2^{|S_{i Z_i}| (H(X_i | X_{\bar{i}}, Z) - H(X_{\bar{i}}, Z) + \delta')}$, $2^{n p (H(X_i | X_{\bar{i}}, Z) + \delta')}, \delta' > 0$. From (4), we know that $p \le 2^{- s_i n} 2^{n p (H(X_i | X_{\bar{i}}, Z) + \delta')}$, then $s_i \ge p H(X_i | X_{\bar{i}}, Z)$.

*   If both of the sequences $(x_1|_{S_{1 Z_1}}, x_2|_{S_{2 Z_2}})$ are not typical with $z|_{S_{1 Z_1}, S_{2 Z_2}}: 2^{|S_{1 Z_1}| (H(X_1, X_2, Z) - H(Z) + \delta')}$, $2^{n p (H(X_1, X_2, Z) - H(Z) + \delta')}, \delta' > 0$. From (5), we know that $p \le 2^{-(s_1 + s_2) n} 2^{n p (H(X_1, X_2, Z) - H(Z) + \delta')}$, then $s_1 + s_2 \ge p(H(X_1, X_2, Z) - H(Z) + \delta')$.

The final inner bound is as follows:
$$r_1 < p \max_{P_{X_1} P_{X_2}} (H(X_1) - H(X_1 | X_2, Z)) = \max_{P_{X_1} P_{X_2}} I(X_1; Y | X_2),$$
$$r_2 < p \max_{P_{X_1} P_{X_2}} (H(X_2) - H(X_2 | X_1, Z)) = \max_{P_{X_1} P_{X_2}} I(X_2; Y | X_1),$$
$$r_1 + r_2 < p \max_{P_{X_1} P_{X_2}} (H(X_1) + H(X_2) + H(Z) - H(X_1, X_2, Z)) = \max_{P_{X_1} P_{X_2}} I(X_1, X_2; Y).$$
The lower and upper bounds coincide, then the capacity is proved. $\square$

## APPENDIX K
### PROOF OF THEOREM 17

The overall structure of the proof is the same as Theorem 15 wherein all parties are honest. Malicious Bob can benefit from the unfairness of the channel and deviate from the channel statistics in $\delta n$ positions without being detected. He tries to find the unselected strings from both senders. Thus, he could compute sets $S_{i 0}$ and $S_{i 1}$ so that non-erasures are distributed in both sets. With probability exponentially close to one, the total number of non-erasures Bob receives from each sender will be no larger than $(p + \eta)n$. Thus, for any strategy Bob distributes these non-erasures between two sets $S_{i 0}$ and $S_{i 1}$, the number of erasures in $S_{i Z_i}$ is no less than $(p - \eta)n - \frac{\eta n}{2} = (p - 3\eta)n$.

Again, we must bound (63)-(65), to get three bounds on the lower bound. For (63) and (64), all steps are the same until (67). Let $V$ be an i.i.d. random variable so that $V = e$ (erasure) with probability $1 - \eta$ and $V = Z$ (the output of channel $W$ on input $(X_1, X_2)$). As the number of erasures in $S_{i Z_i}$ is no less than $\frac{1}{2}(p - 3\eta)n$ with negligible error probability and $\text{SU-SBC}_{p, W}$ being i.i.d., by taking the same steps as (68) for $S_{i Z_i}$, we have:
$$H^{\epsilon}_{\infty}(X_i|_{S_{i Z_i}} | Y|_{S_{i Z_i}}^n) \ge H^{\epsilon}_{\infty}(X_i|_{S_{i Z_i}} | |V|_{S_{i Z_i}}|)$$
$$\ge \frac{1}{2} n(H(X_i) + H(X_i | Z)) - 2 \eta n - 4 \sqrt{n} \log(1/\epsilon). \quad (73)$$
Putting $\epsilon = \epsilon + \epsilon'$ in (67), then putting (73) to (67), we have:
$$H^{\epsilon + 2\epsilon'}_{\infty}(X_1|_{S_{1 Z_1}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)}) \ge \frac{1}{2} p n(H(X_1) + H(X_1 | Z)) - 2 \eta n - 4 \sqrt{n} \log(1/\epsilon)$$
$$- s_1 n - \log \left(\frac{1}{1-\epsilon'}\right) - \log \left(\frac{1}{\epsilon}\right)$$
$$\overset{(a)}{\approx} \frac{1}{2} p n(H(X_1) + H(X_1 | Z)) - s_1 n - 2 \eta n - 4 \sqrt{n} \alpha - n(\alpha + \alpha'), \quad (74)$$
where $(a)$ follows from setting $\epsilon = 2^{- \alpha n}$ and $\epsilon' = 2^{- \alpha' n}$ ($\epsilon$ and $\epsilon'$ are negligible in $n$). For any $\delta \ge (\alpha + \alpha' + 2\eta + 4\sqrt{\alpha}) > 0$, we have:
$$H^{\epsilon + 2\epsilon'}_{\infty}(X_1|_{S_{1 Z_1}} | h_{1 j}(R_{1 j}, X_1|_{S_{1 Z_1}}), Y|_{S_{1 Z_1}}^n, R^{(1)}, \mathcal{T}^{(1)}) \ge \frac{1}{2} n(H(X_1) + H(X_1 | Z)) - s_1 n - \delta n.$$
From Lemma 3, we know that if we set $r_1 < \frac{1}{2} n(H(X_1) + H(X_1 | Z)) - s_1$ and appropriately choose the constant $\delta, \eta, \alpha$ and $\alpha'$, Bob can not obtain non-trivial information about the unselected string. The proof for Alice-2 is the same.
Now consider (70) for the sum-rate. Similarly, it can be bounded from below:
$$H_{\infty}^{\epsilon}(X_1|_{S_{1 Z_1}} | Y|_{S_{1 Z_1}, S_{2 Z_2}}^n) + H_{\infty}^{\epsilon}(X_2|_{S_{2 Z_2}} | X_1|_{S_{1 Z_1}}, Y|_{S_{1 Z_1}, S_{2 Z_2}}^n) - s_1 n - s_2 n - 2 \log \left(\frac{1}{1-\epsilon'}\right) - 2 \log \left(\frac{1}{\epsilon}\right)$$
$$\ge \frac{1}{2} n(H(X_1) + H(X_2) + H(X_1 | Z) + H(X_2 | X_1, Z)) - 4 \eta n - 8 \sqrt{n} \log(1/\epsilon) - s_1 n - s_2 n$$
$$- 2 \log \left(\frac{1}{1-\epsilon'}\right) - 2 \log \left(\frac{1}{\epsilon}\right)$$
$$\overset{(a)}{\approx} \frac{1}{2} n(H(X_1) + H(X_2) + H(X_1 | Z) + H(X_2 | X_1, Z)) - s_1 n - s_2 n - 2 \delta n,$$
for any $\delta \ge (\alpha + \alpha' + 2\eta + 4\sqrt{\alpha}) > 0$, $(a)$ follows from setting $\epsilon = 2^{- \alpha n}$ and $\epsilon' = 2^{- \alpha' n}$ ($\epsilon$ and $\epsilon'$ are negligible in $n$).
Up to now, we proved that if $r_i > \frac{1}{2} n(H(X_i) + H(X_i | Z)) - s_i n - \delta n$ and $r_1 + r_2 > \frac{1}{2} n(H(X_1) + H(X_1 | Z) + H(X_2) + H(X_2 | X_1, Z)) - s_1 n - s_2 n - 2 \delta n$, then Protocol 1 is private against malicious Bob. Since we proved for $s_i \ge p H(X_i | X_{\bar{i}}, Z)$ and $s_1 + s_2 \ge p(H(X_1, X_2, Z) - H(Z) + \delta')$, Protocol 1 is correct for honest players, Then the above region can be written as:
$$R_1 < \frac{1}{2} \max_{P_{X_1} P_{X_2}} \{I(X_1; Y | X_2) + I(X_1; X_2 | Y)\},$$
$$R_2 < \frac{1}{2} \max_{P_{X_1} P_{X_2}} \{I(X_2; Y | X_1) + I(X_1; X_2 | Y)\},$$
$$R_1 + R_2 < \frac{1}{2} \max_{P_{X_1} P_{X_2}} I(X_1, X_2; Y),$$
for some distribution $p(x_1)p(x_2)$ on $\mathcal{X}_1 \times \mathcal{X}_2$. This completes the proof. $\square$

## REFERENCES

[1] O. Goldreich, S. Micali, and A. Wigderson, “How to play any mental game,” in Proceedings of the 19th Annual ACM Symposium on Theory of Computing (STOC), 1987, pp. 218–229.
[2] M. O. Rabin, "How to Exchange Secrets by Oblivious Transfer," Aiken Comput. Lab., Harvard Univ., Cambridge, MA, Tech. Memo TR-81 (1981).
[3] S. Even, O. Goldreich, and A. Lempel, "A Randomized Protocol for Signing Contracts," *Communications Of The ACM*. 28, 637–647 (1985).
[4] J. Kilian, “Founding Cryptography on Oblivious Transfer,” in Proc. 20th Annu. ACM Symp. Theory of Computing (STOC), Chicago, IL, 20–31 (1988).
[5] R. Ahlswede and I. Csiszár "On Oblivious Transfer Capacity,” 2009 IEEE Information Theory Workshop On Networking And Information Theory. pp. 1–3 (2009).
[6] A. C. A. Nascimento and A. Winter, "On the Oblivious-Transfer Capacity of Noisy Resources," *IEEE Transactions On Information Theory*. 54, 2572–2581 (2008).
[7] R. Ahlswede and I. Csiszár, "Common Randomness in Information Theory and Cryptography. I. Secret sharing,” *IEEE Transactions On Information Theory*. 39, 1121–1132 (1993).
[8] C. Crépeau, "Efficient Cryptographic Protocols Based on Noisy Channels,” *Advances In Cryptology — EUROCRYPT '97*. pp. 306–317 (1997).
[9] C. Crépeau and J. Kilian, "Achieving Oblivious Transfer Using Weakened Security Assumptions,” [Proceedings 1988] 29th Annual Symposium On Foundations Of Computer Science. pp. 42–52 (1988).
[10] D. Stebila and S. Wolf, "Efficient Oblivious Transfer From any Non-trivial Binary-Symmetric Channel," Proceedings *IEEE International Symposium On Information Theory*, pp. 293- (2002).
[11] J. Kilian, "More General Completeness Theorems for Secure Two-party Computation," in Proc. 32nd Annu. ACM Symp. Theory of Computing (STOC), Portland, OR, 316–324 (May 2000).
[12] C. Crépeau, K. Morozov and S. Wolf, "Efficient Unconditional Oblivious Transfer from Almost any Noisy Channel,” *Proc. SCN '04, LNCS*. vol. 3352, pp. 47–59 (2005).
[13] A. C. A. Nascimento, and A. Winter, "On the Oblivious Transfer Capacity of Noisy Correlations," 2006 *IEEE International Symposium On Information Theory*. pp. 1871–1875 (2006).
[14] H. Imai, K. Morozov and A. C. A. Nascimento, "On the Oblivious Transfer Capacity of the Erasure Channel," 2006 *IEEE International Symposium On Information Theory*. pp. 1428–1431 (2006).
[15] K. Rao, and V. Prabhakaran, "A New Upper Bound for the Oblivious Transfer Capacity of Discrete Memoryless Channels," 2014 *IEEE Information Theory Workshop (ITW 2014)*. pp. 35–39 (2014).
[16] R. Cleve, “Limits on the Security of Coin Flips When Half the Processors Are Faulty," Proc. STOC, 1986.
[17] M. Blum, "Coin Flipping by Telephone,” Proc. CRYPTO, 1981.
[18] I. B. Damgård, J. Kilian, and L. Salvail, “On the (Im)possibility of Basing Oblivious Transfer and Bit Commitment on Weakened Security Assumptions,” in *Lecture Notes in Computer Science*, vol. 1561, Springer-Verlag, 1999, pp. 56–73.
[19] H. Aghaee, B. Akhbari, and C. Deppe, “Network Oblivious Transfer: On Information Theoretical Limits,” Submitted to the *IEEE Globecom*, 2025.
[20] K. Gödel, "On formally undecidable propositions of Principia Mathematica and related systems I," *Monatshefte für Mathematik und Physik*, vol. 38, pp. 173–198, 1931. [Reprinted and translated in: *Kurt Gödel: Collected Works, vol. 1*, Oxford University Press, 1986.]
[21] T. Holenstein, “Strengthening Key Agreement Using Hard-core Sets,” Ph.D. dissertation, Dept. Comp. Sci., Swiss Federal Institute of Technology (ETH), Zurich, Switzerland, 2006.
[22] J. L. Carter and M. N. Wegman, “Universal Classes of Hash Functions," *Journal of Computer and System Sciences*, vol. 18, pp. 143–154, 1979.
[23] J. Wullschleger, “Oblivious-Transfer Amplification," in *Lecture Notes in Computer Science*. Berlin, Germany: Springer-Verlag, 2007.
[24] T. M. Cover and J. A. Thomas, *Elements of Information Theory*. New York: Wiley, 1991.
[25] C. H. Bennett, G. Brassard, C. Crépeau, and U. M. Maurer, “Generalized Privacy Amplification,” *IEEE Transactions on Information Theory*, vol. 41, no. 6, pp. 1915–1923, Nov. 1995.
[26] A. C. Yao, "Protocols for secure computations,” in *Proceedings of the 23rd Annual Symposium on Foundations of Computer Science (FOCS)*, pp. 160–164, 1982.
[27] A. Yao, "How to Generate and Exchange Secrets," Proc. 27th Annu. Symp. Found. Comput. Sci. (FOCS), 1986.
[28] O. Goldreich, S. Micali, and A. Wigderson, “How to Play Any Mental Game," Proc. 19th Annu. ACM Symp. Theory of Computing (STOC), 1987.
[29] T. Rabin and M. Ben-Or, “Verifiable Secret Sharing and Multiparty Protocols with Honest Majority," Proc. 21st Annu. ACM Symp. Theory of Computing (STOC), 1989.
[30] C. E. Shannon, "A mathematical theory of communication," *The Bell System Technical Journal*, vol. 27, no. 3, pp. 379–423, 1948, doi: 10.1002/j.1538-7305.1948.tb01338.x.
[31] M. Serres, *The Parasite*, Johns Hopkins University Press, Baltimore, 1982. Translated by Lawrence R. Schehr.
[32] R. A. Chou, "Pairwise Oblivious Transfer,” in *2020 IEEE Information Theory Workshop (ITW)*, 2021, pp. 1–5.
[33] A. Winter, A. C. A. Nascimento, and H. Imai, “Commitment Capacity of Discrete Memoryless Channels," in *Lecture Notes in Computer Science*, ser. 2898. Berlin, Germany: Springer-Verlag, 2003, pp. 35–51.
[34] A. D. Wyner, "The Wire-tap Channel," *The Bell System Technical Journal*, vol. 54, no. 8, pp. 1355–1387, 1975, doi: 10.1002/j.1538-7305.1975.tb02040.x.
[35] C. Blundo, P. D’Arco, A. De Santis, and D. R. Stinson, "New Results on Unconditionally Secure Distributed Oblivious Transfer," Proc. 9th Annu. Int. Workshop on Selected Areas in Cryptography (SAC), pp. 291–309, 2003.
[36] V. Nikov, S. Nikova, B. Preneel, and J. Vandewalle, "On Unconditionally Secure Distributed Oblivious Transfer," Proc. INDOCRYPT, pp. 395–408, 2002.
[37] C. Crépeau, G. Savvides, C. Schaffner, and J. Wullschleger, “Information-Theoretic Conditions for Two-Party Secure Function Evaluation,” *Proc. Advances in Cryptology—EUROCRYPT*, pp. 538–554, 2006.
[38] P. D’Arco and D. R. Stinson, “Generalized Zig-Zag Functions and Oblivious Transfer Reductions,” Proc. 8th Annu. Int. Workshop on Selected Areas in Cryptography (SAC), pp. 87–102, 2001.
[39] G. Brassard, C. Crépeau, and M. Santha, “Oblivious Transfers and Intersecting Codes,” *IEEE Trans. Inf. Theory*, vol. 42, no. 6, pp. 1769–1780, 1996.
[40] G. Brassard, C. Crépeau, and S. Wolf, “Oblivious Transfers and Privacy Amplification,” *J. Cryptol.*, vol. 16, no. 4, pp. 219–237, 2003.
[41] W. Hoeffding, “Probability inequalities for sums of bounded random variables,” *Journal of the American Statistical Association*, vol. 58, no. 301, pp. 13–30, 1963, doi: 10.1080/01621459.1963.10500830.
[42] U. Maurer, "Secret Key Agreement by Public Discussion from Common Information,” *IEEE Transactions On Information Theory*. 39, 733–742 (1993).
[43] A. Gohari, and V. Anantharam, "Information-Theoretic Key Agreement of Multiple Terminals—Part I,” *IEEE Transactions On Information Theory*. 56, 3973–3996 (2010).
[44] A. Gohari, and V. Anantharam, "Information-Theoretic Key Agreement of Multiple Terminals—Part II: Channel Model," *IEEE Transactions On Information Theory*. 56, 3997–4010 (2010).
[45] S. Salimi, M. Salmasizadeh, M. Aref, and J. Golic, "Key Agreement Over Multiple Access Channel,” *IEEE Transactions On Information Forensics And Security*. 6, 775–790 (2011).
[46] Y. Liang and H. Poor, "Multiple-Access Channels With Confidential Messages," *IEEE Transactions On Information Theory*. 54, 976–1002 (2008).