1668 IEEE COMMUNICATIONS LETTERS, VOL. 16, NO. 10, OCTOBER 2012

# CRC-Aided Decoding of Polar Codes

Kai Niu and Kai Chen

**Abstract**—CRC (cyclic redundancy check)-aided decoding schemes are proposed to improve the performance of polar codes. A unified description of successive cancellation decoding and its improved version with list or stack is provided and the CRC-aided successive cancellation list/stack (CA-SCL/SCS) decoding schemes are proposed. Simulation results in binary-input additive white Gaussian noise channel (BI-AWGNC) show that CA-SCL/SCS can provide significant gain of 0.5 dB over the turbo codes used in 3GPP standard with code rate 1/2 and code length 1024 at the block error probability (BLER) of $10^{-4}$. Moreover, the time complexity of CA-SCS decoder is much lower than that of turbo decoder and can be close to that of successive cancellation (SC) decoder in the high SNR regime.

**Index Terms**—Polar codes, CRC, successive cancellation decoding, stack decoding, list decoding.

## I. INTRODUCTION

POLAR codes, proposed by Arıkan [1], [2], are proved to achieve the symmetric capacity of the binary-input discrete memoryless channels (B-DMCs) under a successive cancellation (SC) decoder. However, the finite-length performance is unsatisfying. So it has stirred great passions to find more powerful decoding methods to improve the performance of polar codes. Belief propagation (BP) [3], [4] and linear programming (LP) [5] decoders are reported to have a significant improvement over SC. Later, successive cancellation list (SCL) [6], [9] and successive cancellation stack (SCS) [10] decoders are introduced to approach the performance of maximum likelihood (ML) decoder with an acceptable complexity. The enhancement of SC using a list (or a stack) has essentially the same idea with the recursive decoding applied to RM codes [11] ([12]).

Cyclic redundancy check (CRC) codes are the most widely used for error-detecting in practical communications systems, e.g. 3rd generation partnership project (3GPP) standard [15]. A $K$-bit input block of the error-correcting encoder consists of $k$ information bits and an $m$-bit CRC sequence, i.e. $K = k + m$. By viewing the CRC bits as part of source bits for the error-correcting code, the code rate is still defined as $R = K/N$. In this paper, we propose a combination of SCL (SCS) decoder and CRC detector to further improve the performance of polar codes. As shown in Fig. 1, at the receiver, the SCL (SCS) decoder outputs the candidate sequences into CRC detector and the latter feeds the check results back to help the codeword determination. We refer to such a decoding scheme as CRC-aided SCL/SCS (CA-SCL/SCS). The performance of CA-SCL/SCS is substantially improved and even outperforms that of turbo codes. In a recent paper [7], Tal and Vardy independently formulate the similar statement on CRC-concatenated polar coding scheme based on SCL decoding. We note at this point that this idea was also mentioned in the presentation by Tal and Vardy and the plenary talk by Arıkan (credited to Tal and Vardy) at ISIT 2011.

The remainder of the paper is organized as follows. Section II describes the preliminaries of polar coding and gives a unified description of SC, SCL and SCS decoding by *a posteriori* probability (APP). Then the CRC-aided decoding algorithms are addressed in Section III. Section IV provides the simulation results for CA-SCL/SCS and turbo codes and presents the decoding complexity comparison. Finally, Section V concludes the paper.

[IMAGE: polar coding and CRC-aided decoding schemes. The diagram shows k bits entering a CRC block (K=k+m bits), then a Polar Encoder (N bits), followed by a channel W. The receiver uses a CA-SCL/SCS decoder consisting of a SCL/SCS unit outputting candidate sequences to a DeCRC unit, which feeds back check results.]
Fig. 1. polar coding and CRC-aided decoding schemes.

## II. PRELIMINARIES

### A. Notations and the A Posteriori Probability

We use the same notations defined in [1]. Assuming the communication over a B-DMC $W : \mathcal{X} \to \mathcal{Y}$, where $\mathcal{X}$ and $\mathcal{Y}$ denote input and output alphabet respectively, the channel transition probabilities are defined as $W(y|x), x \in \mathcal{X}, y \in \mathcal{Y}$ and $x$ is uniformly distributed in $\mathcal{X}$, thus the channel *a posteriori* probabilities can be written as $P(x|y) = \frac{W(y|x)}{\sum_v W(y|v)}$.

After channel combining and splitting operation on $N = 2^n$ independent uses of $W$, we get $N$ successive uses of synthesized binary input channels $W_N^{(i)}$ with $i = 1, 2, \dots, N$. The information bits can be assigned to the channels with indices in the information set $\mathcal{A}$, which are the more reliable subchannels. The complementary set $\mathcal{A}^c$ denotes the frozen bit set and the frozen bits $u_{\mathcal{A}^c}$ can be set to fixed bit values, such as all zeros, for the symmetric channels.

To put it another way, polar coding is performed on the constraint $x_1^N = u_1^N G_N$, where $G_N$ is the generator matrix and $u_1^N, x_1^N \in \{0, 1\}^N$ are the source and code block respectively. The source block $u_1^N$ consists of information bits $u_{\mathcal{A}}$ and frozen bits $u_{\mathcal{A}^c}$. The generator matrix can be recursively defined as $G_N = R_N (F \otimes G_{N/2})$, $G_2 = F = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}$, where $\otimes$ denotes the Kronecker product and $R_N$ is the reverse shuffle matrix.

NIU and CHEN: CRC-AIDED DECODING OF POLAR CODES 1669

[IMAGE: An example of code tree for code length N = 4. The diagram shows a 4-level binary tree. The bold branches show a decoding path of SC with $\hat{u}_1^4 = 0110$.]
Fig. 2. An example of code tree for code length $N = 4$. The bold branches show a decoding path of SC with $\hat{u}_1^4 = 0110$.

From the practical view, the path metric (defined in the next subsection) in SCL and SCS is closely associated with the *a posteriori* probability $P_N^{(i)}(u_1^i | y_1^N)$ of $u_1^i$ given $y_1^N$ and should be taken values in a stable range. The APP representation can be seen as a normalized version of the channel transition probability defined in [1, eq. (5)]. The two probabilities are related by a multiplicative factor. Similar to the original probability expressions, the APPs can also be calculated recursively. Let $u_{1,o}^j$ and $u_{1,e}^j$ denote the subvectors of $u_1^j$ with odd and even indices respectively. For any $n \ge 0$, $N = 2^n, 1 \le i \le N$,

$P_{2N}^{(2i-1)}(u_1^{2i-1} | y_1^{2N}) = \sum_{u_{2i}} P_N^{(i)}(u_{1,o}^{2i} \oplus u_{1,e}^{2i} | y_1^N) \cdot P_N^{(i)}(u_{1,e}^{2i} | y_{N+1}^{2N}) \quad (1)$

$P_{2N}^{(2i)}(u_1^{2i} | y_1^{2N}) = P_N^{(i)}(u_{1,o}^{2i} \oplus u_{1,e}^{2i} | y_1^N) \cdot P_N^{(i)}(u_{1,e}^{2i} | y_{N+1}^{2N}) \quad (2)$

### B. Unified Description of SC/SCL/SCS

A code tree is used to describe the decoding process of polar codes [9], [10]. Fig. 2 gives a simple example. The code tree $\mathcal{T}$ with $N$ levels is composed of all the decoding paths started from the root node. A decoding path $d_1^i = (d_1, d_2, \dots, d_i) \in \mathcal{T}, d_i \in \{0, 1\}, i \in \mathcal{I}$ consists of $i$ branches from level 1 to level $i$ and each branch between two neighbor nodes in the $i$-th level takes the binary value of $d_i$. So the reliability of each path can be evaluated with the APP $P_N^{(i)}(d_i | d_1^{i-1}, y_1^N)$ which can be recursively calculated by the formulas (1) and (2). By using APPs, the sum probability of all the possible paths with the same length is equal to one.

Use the logarithmic APPs as the metrics of the paths:

$M_N^{(i)}(d_1^i | y_1^N) = \begin{cases} M_N^{(i)}(d_1^{i-1} | y_1^N) & i \in \mathcal{A}^c \\ \log P_N^{(i)}(d_1^i | y_1^N) & i \in \mathcal{A} \end{cases} \quad (3)$

For $i \in \mathcal{A}$, the path metric can be recursively calculated by

$M_N^{(2i-1)}(d_1^{2i-1} | y_1^N) = \text{max}^* \left\{ M_{N/2}^{(i)}(d_{1,o}^{2i} \oplus d_{1,e}^{2i} | y_1^{N/2}) + M_{N/2}^{(i)}(d_{1,e}^{2i} | y_{N/2+1}^N), \right. \\ \left. M_{N/2}^{(i)}(d_{1,o}^{2i} \oplus \bar{d}_{1,e}^{2i} | y_1^{N/2}) + M_{N/2}^{(i)}(\bar{d}_{1,e}^{2i} | y_{N/2+1}^N) \right\} \quad (4)$

and

$M_N^{(2i)}(d_1^{2i} | y_1^N) = M_{N/2}^{(i)}(d_{1,o}^{2i} \oplus d_{1,e}^{2i} | y_1^{N/2}) + M_{N/2}^{(i)}(d_{1,e}^{2i} | y_{N/2+1}^N) \quad (5)$

where function $\text{max}^*(a, b) = \max(a, b) + \log(1 + e^{-|a-b|})$ is the Jacobian logarithm and $d_{1,e}^{2i} = \{d_2, d_4, \dots, d_{2i} = 0\}$, $\bar{d}_{1,e}^{2i} = \{d_2, d_4, \dots, d_{2i} = 1\}$.

Theoretically, the performance of maximum *a posteriori* probability (MAP) decoding can be achieved by traversing all the paths in the code tree. But this brute-force search takes exponential complexity and is difficult to be realized.

SC decoding can be seen as a greedy search algorithm in the code tree and only finds one decoding path by step-by-step decision with low complexity $O(N \log N)$. Let the decoding path corresponding to estimated bits be $d_1^N = (\hat{u}_1, \hat{u}_2, \dots, \hat{u}_N)$. If a bit $\hat{u}_i$ is frozen, then $\hat{u}_i = 0$. Otherwise, when $i \in \mathcal{A}$, the decoding rule of SC is as follows:

$\hat{u}_i = \begin{cases} 0 & \frac{M_N^{(i)}(u_i=0, \hat{u}_1^{i-1} | y_1^N)}{M_N^{(i)}(u_i=1, \hat{u}_1^{i-1} | y_1^N)} \ge 1 \\ 1 & \text{otherwise} \end{cases} \quad (6)$

The performance of SC is limited by the bit-by-bit decoding strategy, because once a bit is wrongly decided, there is no chance to correct it in the future decoding procedure.

As an enhanced method of SC, the SCL decoder [6], [9] allows at most $L$ locally best candidates during the decoding process to reduce the chance of missing the correct codeword. In each decoding step, SCL doubles the number of candidate paths and selects the $L$ best ones from the list by a pruning procedure. Finally, the decoder chooses the path with the largest metric from the list as the estimation. A direct implementation of SCL decoder will take $O(L \cdot N^2)$ time and $O(LN \log N)$ space. By using the space-efficient structure and the memory sharing strategy which is called "lazy copy" [6], the time and space complexity of SCL can be reduced to $O(LN \log N)$ and $O(LN)$ respectively.

The SCS decoder [10] uses an ordered stack to store the candidate paths and tries to find the optimal estimation by searching along the best candidate in the stack. Whenever the top path in the stack which has the largest path metric reaches length $N$, the decoding process stops and outputs this path. Unlike the candidate paths in the list of SCL which always have the same length, the candidates in the stack of SCS have different lengths. The latter can also utilize the same technique in [6] to reduce the complexity.

## III. CRC-AIDED DECODING OF POLAR CODES

Traditionally, CRC results provide a stopping criterion for the iterative decoding process or start retransmission requests. In this paper, we propose to utilize the checking information provided by CRC detector in a codeword selection mechanism. The performance of polar codes can be further improved by this CRC-aided SCL/SCS decoding.

### A. CRC-Aided Successive Cancellation List Decoding

Let $\mathcal{L}^{(i)}$ denote the set of candidate paths corresponding to the level-$i$ of code tree in the SCL decoder. The CRC-aided SCL decoding algorithm with the size of list $L$, denoted by CA-SCL $(L)$, can be described as follows:

(A.1) Initialization: One null path is included in the initial list and its metric is set to zero, i.e. $\mathcal{L}^{(0)} = \{\phi\}, M(\phi) = 0$.

1670 IEEE COMMUNICATIONS LETTERS, VOL. 16, NO. 10, OCTOBER 2012

(A.2) Expansion: At the $i$-th level of the code tree, double the number of candidate paths in the list by concatenating new bits $d_i$ taking values of 0 and 1 respectively, that is,

$\mathcal{L}^{(i)} = \{ (d_1^{i-1}, d_i) | d_1^{i-1} \in \mathcal{L}^{(i-1)}, d_i \in \{0, 1\} \} \quad (7)$

for each $d_1^i \in \mathcal{L}^{(i)}$ update path metric(s) by (3), (4) and (5).

(A.3) Competition: If the number of paths in the list after (A.2) is no more than $L$, just skip this step; otherwise, reserve $L$ best paths with the largest metrics and delete the others.

(A.4) CRC-aided path selection: Repeat (A.2) and (A.3) until level-$N$ is reached. Then, the paths in the list are examined one-by-one with decreasing metrics. The decoder outputs the first path passing the CRC detection as the estimation sequence. If none of such a path is found after traversing $\mathcal{L}^{(N)}$, the algorithm declares a decoding failure.

### B. CRC-Aided Successive Cancellation Stack Decoding

Let $D$ and $T$ denote the maximal and instantaneous depth of the stack in SCS decoder respectively. An additional parameter $Q$ is introduced to limit the number of extending paths with a certain length in decoding process. A counting vector $q_1^N = (q_1, q_2, \dots, q_N)$ is used to record the number of the popping paths with specific length, i.e. $q_i$ means the number of popping paths with length-$i$ during the decoding process. Under this configuration, SCS decoder can successively output at most $Q$ length-$N$ candidate paths. Note that, in CRC-aided decoding scheme, SCS will not stop until CRC is passed or the limit $Q$ is reached. The CA-SCS algorithm with the path counting limit $Q$ and maximal stack depth $D$, denoted by CA-SCS $(Q, D)$, is summarized as follows:

(B.1) Initialization: Push the null path into stack and set the corresponding metric $M(\phi) = 0$. Initialize the counting vector $q_1^N$ with all-zeros, and set $T = 1$.

(B.2) Popping: Pop the path $d_1^{i-1}$ from the top of stack, $T = T - 1$ and if the path is not null, set $q_{i-1} = q_{i-1} + 1$.

(B.3) Competition: If $q_{i-1} = Q$, delete all the paths with length less than or equal to $i - 1$ from the stack and update the value of instantaneous stack depth $T$.

(B.4) Expansion: If $d_i$ is a frozen bit, simply extend the path to $d_1^i = (d_1^{i-1}, 0)$; otherwise, if $d_i$ is an information bit, extend current path to $(d_1^{i-1}, 0)$ and $(d_1^{i-1}, 1)$. Then calculate path metric(s) by (3), (4) and (5).

(B.5) Pruning: For information bit $d_i$, if $T > D - 2$, delete the path from the bottom of the stack and set $T = T - 1$. Then push the two extended paths into the stack and set $T = T + 2$. Otherwise, for frozen bit $d_i$, push the path $d_1^i = (d_1^{i-1}, 0)$ into the stack and set $T = T + 1$.

(B.6) Sorting: Resort paths in the stack from top to bottom in descending metrics.

(B.7) CRC-aided decision: If the top path in the stack reaches the leaf node of the code tree, pop it from the stack and set $q_N = q_N + 1$ and $T = T - 1$. Then, a CRC checking is performed on the path $d_1^N$. If this checking is passed, the algorithm stops and outputs $d_1^N$ as the decision sequence. Else, if the CRC checking is not passed and $q_N = Q$ or the stack is empty, the algorithm also stops and declares a decoding failure. Otherwise go back and execute step (B.2).

[IMAGE: BLER performance comparisons for polar codes and turbo codes with block length N = 1024 and code rate R = 1/2. The plot shows Block Error Probability (BLER) vs Eb/N0(dB) for schemes: SC, SCL(32), SCS(32, 1000), CA-SCL(32), CA-SCS(4, 1000), CA-SCS(8, 1000), CA-SCS(16, 1000), CA-SCS(32, 1000), and Turbo+CRC (Imax=8). CA-SCL and CA-SCS show significant gains over SC and traditional turbo codes.]
Fig. 3. BLER performance comparisons for polar codes and turbo codes with block length $N = 1024$ and code rate $R = 1/2$.

Depending on the different properties of underlying SCL and SCS decoders, the main differences between CA-SCL and CA-SCS can be summarized as: (1) the path extension in the list is parallel while the same operation in the stack is serial; (2) the time complexity of the former $O(LN \log N)$ is fixed and that of the latter is variable and no more than that of the former when $Q = L$; (3) the space complexity of the CA-SCS $O(DN)$ is usually larger than that of $O(LN)$ in CA-SCL.

## IV. PERFORMANCE AND COMPLEXITY COMPARISONS

In this section, we compare the performance and complexity of CA-SCL/SCS and turbo decoding via simulations over binary-input additive white Gaussian noise channels (BI-AWGNCs). The underlying SCL and SCS decoders of CA-SCL/SCS are implemented based on the space-efficient structure and "lazy copy" strategy introduced in [6].

A rate $R = 1/2$ eight-state turbo code with generator polynomials $[1, 1 + D + D^3 / 1 + D^2 + D^3]$ and rate matching mechanism in 3GPP standard [15] is used as a reference. For all coding schemes (both for turbo and polar codes), a CRC-24 code with generator polynomial $g(D) = D^{24} + D^{23} + D^6 + D^5 + D + 1$ is used. The $m = 24$ CRC bits are attached to the $k$-length information block and all the $K = k + m$ bits are fed into the error-correcting encoders. Here, the corresponding code rate of both schemes is defined as $R = K/N$.

The Log-MAP algorithm is applied in the decoding of turbo codes. After each decoding iteration, CRC checking is performed for early stopping the iterative decoding until the maximum number of iterations $I_{max}$ is reached. It is generally found that the average number of decoding iterations can be reduced dramatically from $I_{max}$ while maintaining the same BLER performance. So this scheme can be used as a good counterpart for comparing with the polar codes.

Fig. 3 gives the BLER performance comparisons for various coding and decoding schemes. All the schemes have the code length $N = 1024$ and code rate $R = 1/2$ and are running over the BI-AWGN channel. For turbo codes, the maximum number of iterations $I_{max} = 8$. For polar codes, the information set $\mathcal{A}$ is selected using the method in [8]. CA-SCL and CA-SCS decoders with different configurations are simulated. And curves of SC, SCL (32) and SCS (32, 1000) are also given.

NIU and CHEN: CRC-AIDED DECODING OF POLAR CODES 1671

[IMAGE: Average complexity comparisons for polar codes and turbo codes with block length N = 1024 and code rate R = 1/2. The plot shows Average Complexity vs Eb/N0(dB). Schemes include SC (lowest complexity), various CA-SCL (fixed complexity), and CA-SCS (variable complexity decreasing with SNR). Turbo+CRC is also shown for comparison.]
Fig. 4. Average complexity comparisons for polar codes and turbo codes with block length $N = 1024$ and code rate $R = 1/2$

We can see that the performance of SCS and SCL are almost the same and both are far better than that of SC. But there still is 0.7 dB performance gap between turbo codes and polar codes under SC/SCL at the BLER of $10^{-4}$. Similarly, the CA-SCS and CA-SCL decoders with the same configuration, that is, $Q = L$, have equivalent performance. In Fig. 3, only the curve of CA-SCL with $L = 32$ is depicted while the parameter $Q$ of SCS taking values in $\{4, 8, 16, 32\}$. Extraordinarily, the performance of CA-SCL/SCS decoder can outperform that of turbo codes with sufficient big $L$ or $Q$. Specifically, the CA-SCL (32) or CA-SCS (32, 1000) can obtain a prominent performance gain of 0.5 dB at the BLER of $10^{-4}$ compared with turbo codes. Moreover, unlike the turbo codes, polar codes under CA-SCL/SCS decoding show no error floors down to the BLER of $10^{-6}$.

To simplify the complexity evaluation of polar decoding, we define the average complexity in terms of the number of metric recursive operations, Eq. (4) or (5). Thus the time complexity of SC decoder can be directly calculated as $N \log N = 1024 \times 10 \approx 10^4$. Similarly, the complexity of turbo decoding can also be evaluated by metric operations per trellis node. In every single iteration of Log-MAP algorithm, turbo decoder takes $2 \times (8 \times 4) \times N/2 \approx 3 \times 10^4$ operations which includes three factors: 2 constituent decoders with 8 states and 4 metric updates per trellis node.

The complexity comparisons of Log-MAP decoder for turbo codes, SC/SCL/SCS and CA-SCL/SCS decoders for polar codes are shown in Fig. 4. The SC decoder has the lowest complexity, but the corresponding performance is poor. And the time complexity of CA-SCL decoder is fixed, while that of CA-SCS is variable and can be far below that of CA-SCL in the high bit SNR regime. Further, we can see that polar coding under CA-SCS with $Q = 16(32)$ have complexities lower than (nearly equal to) that of turbo coding and obtains an additional 0.4(0.5) dB performance gain.

The performance of turbo codes can also be improved by list decoding [13], [14]. But these enhanced decoders are composed of either combinational structure of Log-MAP/list Viterbi algorithms or modified list Max-log-MAP algorithm. All these list decoding methods have higher complexities than that of Log-MAP algorithm. Hence, in this paper, we only take the performance and complexity of turbo code under Log-MAP decoding for comparison.

## V. CONCLUSIONS

CRC-aided decoding schemes are proposed to improve the performance of polar codes. Simulation results in BI-AWGNC show that CA-SCL/SCS can provide significant gain of 0.5 dB over the turbo codes. Moreover, the time complexity of CA-SCS decoder is much lower than that of turbo decoder and can be close to that of SC decoder in the high SNR regime. For these reasons, polar code under CA-SCL/SCS decoding can be a competitive candidate in future communication systems.

## REFERENCES

[1] E. Arıkan, "Channel polarization: a method for constructing capacity achieving codes for symmetric binary-input memoryless channels," *IEEE Trans. Inf. Theory*, vol. 55, no. 7, pp. 3051–3073, July 2009.
[2] E. Arıkan and E. Telatar, "On the rate of channel polarization," in *Proc. 2009 IEEE Intern. Symp. on Information Theory*.
[3] E. Arıkan, "A performance comparison of polar codes and Reed-Muller codes," *IEEE Commun. Lett.*, vol. 12, pp. 447–449, June 2008.
[4] N. Hussami, S. B. Korada, and R. Urbanke, "Performance of polar codes for channel and source coding," in *Proc. 2009 IEEE Intern. Symp. on Information Theory*.
[5] N. Goela, S. B. Korada, and M. Gastpar, "On LP decoding of polar codes," *2010 Information Theory Workshop*.
[6] I. Tal and Vardy, "List decoding of polar codes," *Inf. Theory Proc.*, pp. 1-5, 2011.
[7] I. Tal and A. Vardy, "List decoding of polar codes," *arXiv:1206.0050v1*, May 2012.
[8] I. Tal and A. Vardy, "How to construct polar codes," *arXiv:1105.6164v1*, May 2011.
[9] K. Chen, K. Niu, and J. R. Lin, "List successive cancellation decoding of polar codes," *Electron. Lett.*, vol. 48, no. 9, pp. 500–501, 2012.
[10] K. Niu and K. Chen, "Stack decoding of polar codes," *Electron. Lett.*, vol. 48, no. 12, pp. 695–696, 2012.
[11] I. Dumer and K. Shabunov, "Soft-decision decoding of Reed-Muller codes: recursive lists," *IEEE Trans. Inf. Theory*, vol. 52, no. 3, pp. 1260–1266, Mar. 2006.
[12] N. Stolte, U. Sorger, and G. Sessler, "Sequential stack decoding of binary Reed-Muller codes," in *2000 ITG Conference Source and Channel Coding*.
[13] K. R. Narayanan and G. L. Stuber, "List decoding of turbo codes," *IEEE Trans. Commun.*, vol. 46, no. 6, pp. 754–762, June 1998.
[14] C. F. Leanderson and C. W. Sundberg, "On list sequence turbo decoding," *IEEE Trans. Commun.*, vol. 53, no. 5, pp. 760–763, May 2005.
[15] 3GPP TS 25.212: "Multiplexing and channel coding (FDD)," Release 9, 2009.