Laser Physics Letters

# LETTER
# IC-LDPC Polar codes-based reconciliation for continuous-variable quantum key distribution at low signal-to-noise ratio

To cite this article: Zhengwen Cao et al 2023 Laser Phys. Lett. 20 045201

View the article online for updates and enhancements.

**You may also like**
- A reduced complexity rate-matching and channel interleaver/de-interleaver for 5G NR
Lakshmi J L and J Jayakumari
- Error Performance of Polar Coded Spatial Pulse Position Modulation Technique
Yinlong Li, Zhongyang Mao and Min Liu
- Optimum Polar Codes Encoder over Binary Discrete Memory-less Channels
Karim H. Moussa, Ahmed H. El-Sakka and Shawki Shaaban

This content was downloaded from IP address 131.155.215.214 on 03/02/2026 at 14:16

---

IOP Publishing | Astro Ltd
Laser Physics Letters
Laser Phys. Lett. 20 (2023) 045201 (10pp)
https://doi.org/10.1088/1612-202X/acb920

# Letter
# IC-LDPC Polar codes-based reconciliation for continuous-variable quantum key distribution at low signal-to-noise ratio

Zhengwen Cao$^{1,2}$, Xinlei Chen$^{1}$, Geng Chai$^{1,*}$ and Jinye Peng$^{1}$

$^{1}$ Institute for Quantum Information and Technology, School of Information Science and Technology, Northwest University, Xi’an 710127, People’s Republic of China
$^{2}$ State Key Laboratory of Integrated Services Networks, Xidian University, Xi’an 710071, People’s Republic of China

E-mail: chai.geng@nwu.edu.cn

Received 30 December 2022
Accepted for publication 31 January 2023
Published 14 February 2023

### Abstract
The error correction of information reconciliation affects the performance of the continuous-variable quantum key distribution (CV-QKD). Polar codes can be strictly proven to reach the Shannon-limit. However, due to the insufficient polarization of finite code-length, partial subchannels are neither completely noise-free nor completely noisy. In this paper, an intermediate channel low-density parity check code concatenated polar code (IC-LDPC Polar codes)-based reconciliation for CV-QKD is proposed for the above shortcomings. The experimental results show that the reconciliation efficiency of IC-LDPC Polar code can be over 98% when the signal-to-noise ratio is from −13.2 dB to −20.8 dB, the secret keys can be extracted, and the minimum frame error rate (FER) is 0.19. Therefore, the proposed scheme can improve the reconciliation efficiency and reduce the FER at a very low signal-to-noise ratio range, and it is more useful for a practical long-distance CV-QKD system.

**Keywords:** continuous variable, quantum key distribution, information reconciliation, polar code

(Some figures may appear in colour only in the online journal)

### 1. Introduction
With the development of mathematical theory and quantum computing, classical cryptosystems of mathematical complexity face severe challenges [1]. In this context, the quantum key distribution (QKD) technology, which uses physical properties to complete encryption, has attracted wide attention. QKD is one of the most promising technologies in quantum cryptography [2, 3]. It uses the fundamental theory of quantum mechanics to enable both sides of the communication (the sender Alice and the receiver Bob) to have unconditionally secure shared keys in an untrusted communication environment far away. Presently, the two leading technologies of QKD are discrete-variable QKD (DV-QKD) [4] and continuous-variable QKD (CV-QKD) [5, 6]. DV-QKD protocol uses single-photon polarization to encode the secret key information and single-photon detection to measure the received quantum state. The CV-QKD protocol modulates the secret key information in the quadrature component of the optical field, and the detection mainly adopts the homodyne or heterodyne detection [7]. The security proofs of the actual DV-QKD system are based on the basic assumptions, but the actual implementation scheme violates the basic assumptions [8]. This imperfection is likely to bring security risks to the actual application system. In contrast, although the CV-QKD system started late, it can only be realized with basic classical optical communication devices in the experiment and has a great application prospect. For the Gaussian-modulated coherent state (GMCS) CV-QKD protocol [9], Leverrier’s team strictly proved that it is safe against any attack under the finite size effects [10, 11].

The CV-QKD system consists of physical links and post-processing. The first part mainly includes the preparation, transmission, and detection of quantum states. After the above processing, Alice and Bob obtain the initial keys [12]. However, due to the noise and the interference of the eavesdropper (Eve), the initial keys obtained by both parties are not consistent. Therefore, Alice and Bob need to perform post-processing through the classical authentication channel to extract the consistent secret keys [13, 14]. In post-processing, the information reconciliation step is one of the main factors limiting the system’s communication distance and secret key rate. Information reconciliation is divided into two parts: reconciliation and error correction [15, 16]. Through reconciliation, the initial keys in the form of continuous-variables are quantized or rotated, then the initial keys after reconciliation are corrected with the classical error-correcting codes. According to different reconciliation methods, it can be divided into direct reconciliation and reverse reconciliation. Among them, the reverse reconciliation overcomes the limit of 3 dB loss of the channel and can realize long-distance communication [17]. At present, the mainstream reconciliation algorithms include slice reconciliation [18] and multidimensional reconciliation [19]. Slice reconciliation has high discrimination among variables at high signal-to-noise ratios (SNRs) and is suitable for short-distance communication systems. But its frame error rate (FER) is high at low SNRs, and the algorithm complexity is high. The multidimensional reconciliation algorithm makes use of the rotation mapping of spherical codes in multidimensional space to rotate Gaussian continuous-variables into uniformly distributed discrete-variables [20]. The algorithm has almost no information loss at very low SNRs, can correct Gaussian continuous-variables at low SNRs with error-correcting code close to the Shannon-limit, and has high error correction performance, so it is widely used in long-distance communication systems [21, 22].

A high performance error-correcting code is the key to obtaining high reconciliation efficiency in the long-distance CV-QKD communication system. In the long-distance CV-QKD system, the transmission loss causes low SNRs, which may be lower than −13 dB or even as low as −20 dB, so improving the error correction performance is necessary. Currently, Chen et al achieve reconciliation efficiency of 96% when the SNR is as low as −16.45 dB to −13 dB and extracts the secret keys with a FER of about 0.3 [22]. Before this, the polar codes proposed by Arikan in 2009 have been proven to be a channel coding method that achieves channel capacity through channel polarization [23]. Thus, polar codes have good error correction performance and low encoding and decoding complexity [24, 25]. Considering these advantages, Zhao et al combine polar codes with multidimensional reconciliation in the CV-QKD system to successfully extract secret keys in 0 dB to 5 dB environments, with the highest reconciliation efficiency of 97.9% [26]. However, the coding subchannels of the finite code-length polar codes are not entirely polarized, and some of them are neither completely noise-free nor completely noisy, called the intermediate channels. The intermediate channels are protected by outer low-density parity check (LDPC) codes [27], called IC-LDPC Polar codes, which can improve the decoding performance of polar codes. Compared with traditional polar codes, the decoding performance of IC-LDPC Polar codes is enhanced by 0.3 dB or more. Hence, this paper applies the IC-LDPC Polar codes into the CV-QKD system. The experimental results show that the IC-LDPC Polar codes can maintain the reconciliation efficiency of over 98% when the SNR is from −13.2 dB to −20.8 dB, the secret keys are extracted, and the minimum FER is 0.19. The reconciliation efficiency of 99.2% can even be achieved in an environment with an SNR of −19.3 dB, while the FER is as low as 0.5. It not only expands the low signal-to-noise ratio range of the secret keys that can be successfully extracted but also improves the reconciliation efficiency and reduces the FER, which can meet the practical application of the long-distance CV-QKD system.

The structure of this paper is as follows: in section 2, the reconciliation scheme based on IC-LDPC Polar codes is proposed. Section 3 completes the construction of IC-LDPC Polar codes in the CV-QKD system. In section 4, the reconciliation performance of IC-LDPC Polar codes in the CV-QKD system is verified by experiment. Finally, section 5 summarizes this paper.

### 2. IC-LDPC Polar codes-based reconciliation for CV-QKD

#### 2.1. Information reconciliation in CV-QKD system

In the GMCS CV-QKD protocol, Alice uses the pulsed light source to generate coherent states, which are sent to Bob through the quantum channel after Gaussian modulation. At this time, the initial keys obtained by Alice and Bob are only relevant and inconsistent. In order to extract the secret keys, the initial keys need post-processing. The post-processing operation is completed on the authenticated common channel called the classical authentication channel (CAC). That is, Eve can obtain the information transmitted by both parties on the channel but cannot change the information. Post-processing is divided into three progresses: parameter estimation, information reconciliation, and privacy amplification. Among them, the information reconciliation stage includes reconciliation and error correction [16]. Usually, continuous-variables are quantized or rotated through reconciliation, then select an appropriate error-correcting method to correct the data after reconciliation so that communication parties can obtain the consistent secret keys.

In the case of considering collective attacks and finite-size effects, when Alice and Bob adopt reverse reconciliation, the secret key rate of the CV-QKD system is as follows:
$$K = (1 - \alpha)(1 - \text{FER}) \frac{n}{\iota} [\beta I_{AB} - \chi_{BE} - \Delta n], \quad (1)$$
where $\alpha$ is the system overhead, FER is the reconciliation frame error rate, $\iota$ is the total number of variables exchanged by Alice and Bob, and $n$ is the number of variables used for key extraction. $\beta$ is the reconciliation efficiency. $I_{AB}$ is the maximum mutual information of Alice and Bob. $\chi_{BE}$ is the maximum of the Holevo information that Eve can obtain from the information of communication parties. $\Delta n$ is the finite-size offset factor. It can be seen from equation (1) that the performance of the reconciliation scheme (reconciliation efficiency and FER) is the key to restricting whether the system can generate the secret keys. Reconciliation efficiency $\beta$ represents the proportion of information extracted from mutual information of communication parties. In the existing system, its calculation method is as follows:
$$\beta = \frac{R}{C} = \frac{R}{0.5 \times \log_2(1 + \text{SNR})}, \quad (2)$$
where $R$ represents the code-rate of the error-correcting code used in the reconciliation process, $C$ represents the security capacity of the quantum channel, and SNR represents the signal-to-noise ratio. The code-rate of the error-correcting code can be calculated according to the channel SNR after parameter estimation and target reconciliation efficiency. Due to the extremely low data SNR of the receiver Bob in the long-distance CV-QKD system, the performance of the reconciliation scheme is required to be high. Under such circumstances, if the reconciliation scheme is not perfect, the reconciliation performance will be reduced, thus limiting the scope of the protocol. Therefore, higher reconciliation efficiency and lower FER are required to obtain higher secret key rates in long-distance communication, which requires selecting appropriate and efficient error-correcting code in the reconciliation scheme.

#### 2.2. IC-LDPC Polar codes-based reconciliation scheme in CV-QKD system

Error correction is part of information reconciliation. The performance of the error-correcting code will affect reconciliation efficiency $\beta$ and FER, thus affecting the generation of the secret keys. This paper applies IC-LDPC Polar codes into the CV-QKD system. First, the polar codes are based on the channel polarization principle to select a specific coding scheme. It is the only known channel coding method that can be strictly proved to reach the channel capacity. Channel polarization refers to the phenomenon that $N$ independent identically distributed channels undergo the channel combining phase and the channel splitting phase, resulting in the polarization of the reliability of the subchannels $\{W_N^{(i)} : 1 \le i \le N\}$ after splitting. When $N \to \infty$, the subchannels either become completely noiseless reliable channels (good channels) or completely noisy unreliable channels (bad channels). However, the polarization phenomenon of finite code-length polar codes is insufficient in practice. Figure 1 shows the channel polarization phenomenon calculated by the Bhattacharyya parameter [23] when the code-length $N = 512$. It can be seen from figure 1 that there exists a few intermediate channels besides good or bad channels. This molecular channel is neither completely noiseless nor completely noisy. For these intermediate channels, shorter LDPC codes can be used as outer codes to provide additional protection for these specific channels, and polar codes can be used as internal codes to improve the overall performance of polar codes [28].

[IMAGE: Figure 1. Channel polarization with code-length N = 512.]

In this paper, the multidimensional reconciliation algorithm and IC-LDPC Polar codes are used as the reconciliation scheme of the CV-QKD system. Figure 2 is the schematic diagram of the scheme. The detailed steps (taking reverse reconciliation as an example) are as follows:

- First, Alice uses the pulsed light source to generate a group of coherent states by Gaussian modulation and transmit through the quantum channel. After Bob uses the homodyne detection to measure the quantum signal, both communication parties hold the initial keys $x$ and $y$, respectively, which obey the Gaussian distribution. Alice’s data satisfies the Gaussian distribution $x \sim \mathcal{N}(0, V_A)$, then, $y = tx + z$, where $V_A$ is the modulation variance, $t$ is related to the channel loss, and $z$ is the noise introduced by the quantum channel, $z \sim \mathcal{N}(0, \sigma_z^2)$, where $\sigma_z^2$ is the noise variance of the quantum channel. Alice and Bob divide the initial keys $x$ and $y$ into $d$-dimensional vectors and normalize them into $x'$ and $y'$.
- Bob uses the quantum random number generator (QRNG) to generate a group of binary bit sequence $b$ satisfying uniform distribution, then calculates the channel SNR according to the parameter estimation results, selects the target reconciliation efficiency and coding scheme with reference to equation (2). Sequence $b$ is the input as the information bits of the IC-LDPC Polar encoder, and sequence $c$ is the output. According to binary phase shift keying, the sequence $c$ is converted into a $d$-dimensional vector $c' \in \{\frac{-1}{\sqrt{d}}, \frac{1}{\sqrt{d}}\}^d$.
- Bob calculates the rotation mapping function $M(y', c')$ that satisfies $M(y', c')y' = c'$, and sends the function $M(y', c')$ along with related side information to Alice through the CAC. Alice maps the local sequence $x$ to $v = M(y', c')x'$ in the same way, and $v$ is the sequence $c$ with noise.
- Alice performs IC-LDPC Polar decoding on sequence $v$ and outputs sequence $b'$. Both communication parties obtain consistent secret keys if the decoding is successful. On the other hand, Bob calculates and sends more mapping functions $M(y', c')$, and Alice prepares for the subsequent decoding.

[IMAGE: Figure 2. IC-LDPC Polar codes-based multidimensional reconciliation for CV-QKD. Alice transmits coherent states to Bob through the quantum channel, and both parties complete information reconciliation on the CAC. QRNG: quantum random number generator. QC: quantum channel. CAC: classical authentication channel. x and y: initial keys. b: random binary sequences generated by QRNG. c: IC-LDPC Polar encoder output sequence. M(y', c'): rotation mapping function. v: IC-LDPC Polar decoder input sequence. b': IC-LDPC Polar decoder output sequence.]

According to equation (1), in theory, higher reconciliation efficiency and lower FER are required to obtain higher secret key rates. The reconciliation efficiency and FER largely depend on the performance of the constructed IC-LDPC Polar codes. Therefore, constructing an efficient IC-LDPC Polar code is the key to this reconciliation scheme. Next, this paper applies the construction method of IC-LDPC Polar codes in detail in section 3.

### 3. Construction of IC-LDPC Polar codes for CV-QKD

#### 3.1. The best design-SNR in reconciliation scheme

The IC-LDPC Polar codes use the polar codes as the internal code. Once code-length $N$ is determined, the polar codes’ generator matrix is also determined. At this time, the selection of each subchannel for transmitting information bits or frozen bits ultimately defines a polar code. This selection process of subchannels is called the construction of polar codes. Non-universality is an important characteristic of polar codes. That is, polar codes constructed with different given SNRs are different, thus correspond to different encoding and decoding results. For a stable CV-QKD system, the quantum channel changes slowly, and the SNR changes little in a short time. Its SNR will fluctuate with the system’s operation, but the fluctuation is usually not too large. If the codec of IC-LDPC Polar codes always changes with the channel SNR, this is undesirable in actual experiments. Hence, it is necessary to design an SNR that can be applied to the varying SNR within a certain range. This SNR is called design-SNR [29]. Combined with the characteristics of the CV-QKD system, the channel state detection sequence can be used to calculate the best design-SNR. The specific method is as follows:

**Step 1:** Alice selects a part of the random string as the synchronization sequence and channel state detection sequence. The length of the channel state detection sequence is $L$. At the same time, Bob generates the sequence $F(a)$, which is identical to the channel state detection sequence. Alice prepares all random strings into coherent states and transmits them to Bob through the quantum channel.

**Step 2:** Bob employs the homodyne detection to measure the coherent state and uses the incremental label algorithm [30] to perform correlation operations to determine whether synchronization is successful based on the correlation peak value. If the synchronization is successful, then calculate the best design-SNR. Bob divides the received channel state detection sequences $M(a)$ and local sequence $F(a)$ into $m$ groups of data $\{M_1, M_2, \dots, M_m\}$ and $\{F_1, F_2, \dots, F_m\}$ at equal intervals, and the length of each group of data is $h = L/m$. Bob respectively estimate the channel state of the $m$ groups of data and calculate the SNR of each group of data. The calculation method is as follows:
$$\text{SNR}_j = \frac{V_A}{\sigma_{zj}^2} = \frac{V_A}{\frac{1}{h} \sum_{i=1}^h (F_{ji} - t M_{ji})^2}, \quad (3)$$
where $\sigma_{zj}^2$ is the noise variance of $j$th group, $j \in [1, m]$, $F_{ji}$ represents the $i$th variable in the $j$th group of data, $M_{ji}$ can be deduced in this way. $t = \sqrt{\eta T}$, $\eta$ is the efficiency of the homodyne detector, $T$ is the transmittance of the quantum channel.

**Step 3:** Bob calculates a set of $\{\text{SNR}_1, \text{SNR}_2, \dots, \text{SNR}_m\}$, that is, the SNR range of the initial keys. Then Bob constructs $m$ IC-LDPC Polar encoders and decoders. Gaussian approximation (GA) [31, 32] is selected as the construction method, and the design-SNR is $\text{SNR}_j$. Corresponding information bits and frozen bits positions are found under each $\text{SNR}_j$.

**Step 4:** Bob compares the FER performance of IC-LDPC Polar codes under the above different design-SNRs, selects the best design-SNR, and declares the corresponding design-SNR for constructing polar codes, which is used for error correction after reconciliation.

[IMAGE: Figure 3. Extended factor graph for IC-LDPC Polar codes. The stopping tree for node (5, 3) is shown in red, with leafset nodes {(0, 0), (1, 0), (4, 0), (5, 0)}. Leafset size for (5, 3) = 4, equal to row-weight of 6th row of G_8.]

#### 3.2. Judgment criteria for intermediate channels in reconciliation scheme

Since the initial keys of the communication parties in the CV-QKD system are two related Gaussian sequences, the GA is used to measure the decoding reliability of each subchannel after polarization. It is assumed that the log-likelihood ratio (LLR) of each subchannel follows Gaussian distribution with variance twice the mean value, i.e. $\text{LLR} \sim \mathcal{N}(E_N^{(i)}, 2E_N^{(i)})$. At this time, the quality of each subchannel $\{W_N^{(i)} : 1 \le i \le N\}$ is measured by the expected $E_N^{(i)}$ of the corresponding LLR. The calculation method of $E_N^{(i)}$ is:
$$E_{2N}^{(2i-1)} = \phi^{-1}(1 - [1 - \phi(E_N^{(i)})]^2), \quad (4)$$
$$E_{2N}^{(2i)} = 2E_N^{(i)}, \quad (5)$$
$$E_1^{(1)} = 2/\sigma^2, \quad (6)$$
where the function $\phi(x)$ can be calculated using the following approximate version of GA [31]:
$$\phi(x) = \begin{cases} \exp(-0.4527x^{0.86} + 0.0218) & 0 < x < 10, \\ \sqrt{\frac{\pi}{x}}(1 - \frac{10}{7x})\exp(-\frac{x}{4}) & x \ge 10. \end{cases} \quad (7)$$
The higher the $E_N^{(i)}$, the lower the corresponding subchannels error probability and the more reliable the channel. This part of the channel is used to transmit information bits $u_i (i \in \mathcal{A})$, where $\mathcal{A}$ is the channel set of transmission information bits. On the other hand, the lower the $E_N^{(i)}$, the higher the corresponding subchannels error probability and the less reliable the channel. This part of the channel is used to transmit the frozen bits $u_i (i \in \mathcal{A}^c)$ for auxiliary decoding, where $\mathcal{A}^c$ is the channel set for transmitting the frozen bits.

When using the belief propagation (BP) algorithm to decode polar codes, the stop set is the key to the success of decoding. In polar-coded factor graphs, a vital concept of stopping sets is stopping trees [33]. For polar codes $(N, K, \mathcal{A}, \mathcal{A}^c)$ with given parameters of the set $\mathcal{A}$, each information bit in $\mathcal{A}$ has a unique stop tree, which takes the information bits (the right side of figure 3) as its root. Its leaf is located in the codeword bits (the left side of figure 3). Figure 3 shows a stop tree with an information bit (node (5, 3)) and its corresponding leafset (nodes {(0, 0), (1, 0), (4, 0), (5, 0)}) as the root. That is to say, each information bit has a corresponding leafset, and the number of codeword bits in the leafset is called the leafset size of the information bits. The leafset size of information bits (node (5, 3)) shown in the figures 3 and 4. When BP decoding is used for two information bits with different leafset sizes, information bits with smaller leafset sizes are more likely to be erased than information bits with larger leafsets. Therefore, after the information set $\mathcal{A}$ is determined, the information bits with smaller leafset sizes can be protected by the outer LDPC codes. Since the size of the leafset of the $i$th information bit is equal to the weight of the $i$th row of the generator matrix $G_N$, the weight of the generator matrix $G_N$ can be directly calculated to determine the leaf-set value of each information bit. Algorithm 1 describes how to select the intermediate channel to be protected by outer LDPC codes when using GA to construct polar codes. At this time, the code-rate of the IC-LDPC Polar codes is calculated as follows:
$$R = \frac{K_p + (R_{ldpc} \times N_{ex})}{N}, \quad (8)$$
where $K_p$ represents the number of good channels (encoded with polar codes only). $N_{ex}$ is the number of channels using outer LDPC coding, whose set is represented as $\mathcal{A}^{IC}$, and its code-rate is represented as $R_{ldpc}$. Figure 3 is the extended factor graph of IC-LDPC Polar codes when $N = 8, K_p = 2, N_{ex} = 5, R_{ldpc} = 0.6$.

---

**Algorithm 1.** Choosing intermediate channels for outer LDPC code
**Input:** $N, K, R, \text{design-SNR (dB)}, G_N, N_{ex}, \{E_N^{(i)} | i \in \mathcal{A}\}$
**Output:** Channel encoded with outer LDPC: $\mathcal{A}^{IC}$
$\sigma^2 = \frac{1}{2R} 10^{-\frac{\text{design-SNR}}{10}}$ and $m = \log_2 N$
$z \in \mathbb{R}^N$, initialize $z[0] = 2/\sigma^2$
**for** $i = 1 : m$ **do**
    $j = 2^{(i-1)}$
    **for** $k = 1 : j$ **do**
        $temp = z[k]$
        $z[k] = \phi^{-1}(1 - (1 - \phi(temp))^2)$   $\triangleright$ Upper channel
        $z[k + j] = 2 \times temp$                  $\triangleright$ Lower channel
$z = bitrevorder(z)$
$\mathcal{A} = \text{indices\_of\_least\_elements}(z, K)$   $\triangleright$ Find indices of the least $K$ elements
**Input:** Vector $z$ of dimension $|z| \times K$ and integer $K$
**Output:** An $K \times 1$ integer vector containing $K$ indices in $\{0, 1, \dots, |z| - 1\}$
**return** (indices\_of\_least\_elements($-z, K$))
Divide $\mathcal{A}$ into different subsets $\mathcal{A}_i, \mathcal{A}_j \dots$ such that each subset contains bits with same row-weight $\omega$, and $\omega_i < \omega_j$ if $i < j$;
Sort bits in each subset $\mathcal{A}_s$ in ascending order based on Expectation of LLR $\{E_N^{(i)} | i \in \mathcal{A}_s, s = 1, 2, \dots\}$ such that first bit in each subset is the least reliable bit among all other bits in that subset;
$\mathcal{A}^{IC} = \text{Choose first } N_{ex} \text{ bits from the set } \{\mathcal{A}_i, \mathcal{A}_j \dots\}$;
**return** $\mathcal{A}^{IC}$

---

#### 3.3. Encoding and decoding for IC-LDPC Polar codes in reconciliation scheme

The coding of IC-LDPC Polar code in the CV-QKD system is divided into two steps: LDPC coding and polar coding. First, Bob uses QRNG to generate a binary bit sequence $b = [u_{\mathcal{A}^g}, b_{IC}]$ with uniform distribution and then uses LDPC code to encode the sequence $b_{IC}$. The encoding method is as follows:
$$u_{\mathcal{A}^{IC}} = b_{IC} G_{N_{ex}}, \quad (9)$$
where $G_{N_{ex}}$ is the generator matrix of the LDPC codes, and the codeword $u_{\mathcal{A}^{IC}}$ with the length of $N_{ex}$ is finally generated. Then, Bob codes the codeword $u_{IC}$ and $u_{\mathcal{A}^g}$ with polar codes. Note that $u_{IC}$ is transmitted through intermediate channel $\mathcal{A}^{IC}$, and $u_{\mathcal{A}^g}$ is transmitted through channel $\mathcal{A}^g$. The coding process can be expressed as follows:
$$c = u_{\mathcal{A}^g} G_{\mathcal{A}^g} + u_{\mathcal{A}^{IC}} G_{\mathcal{A}^{IC}} + u_{\mathcal{A}^c} G_{\mathcal{A}^c}, \quad (10)$$
where $G_{\mathcal{A}^g}$ is the sub-matrix formed by the row number in $\mathcal{A}^g$ in the generator matrix $G_N$. The meaning of $G_{\mathcal{A}^c}$ and $G_{\mathcal{A}^{IC}}$ can be deduced in this way. The final codeword $c$ with length $N$ is obtained by two-step encoding. Then map the sequence $c$ to the spherical code $c'$ on the $d$-dimensional unit sphere. The specific conversion relationship is as follows:
$$(c_1, c_2, \dots, c_d) \to \left( \frac{(-1)^{c_1}}{\sqrt{d}}, \frac{(-1)^{c_2}}{\sqrt{d}}, \dots, \frac{(-1)^{c_d}}{\sqrt{d}} \right) = c'. \quad (11)$$
Then Bob calculates the rotation mapping function $M(y', c')$ from $y'$ to $c'$ on the $d$-dimensional unit sphere, and sends $M(y', c')$ and $\|y\|$ to Alice through the CAC. Alice makes the same rotation on the $d$-dimensional unit sphere, calculates the sequence $v$ (the sequence $c$ with noise). Through the above steps, the physical Gaussian channel is converted into the virtual binary input additive white Gaussian noise (BI-AWGN) channel with input $c$ (Bob) and output $v$ (Alice) through the above steps. and then decodes the sequence $v$ with the BP algorithm.

[IMAGE: Figure 4. (a) Message updating path of a BCB. (b) Tanner graph of LDPC codes with code-length 5. (i, j): the jth node in stage i. N: the code-length of IC-LDPC Polar codes. t: maximum iterations of BP decoding. L^t_{i,j}: LLR propagated by node (i, j) from right to left. R^t_{i,j}: LLR propagated by node (i, j) from left to right. f: check node. h: variable node.]

The decoding process at Alice can be represented by the extended factor graph, as shown in the figure 3, an example of $N = 8$. Generally, round-trip scheduling is employed for decoding [34, 35]. First, Alice sets the information on the left side of the factor graph to the LLR value of the BI-AWGN channel. The value calculation method is as follows:
$$\text{LLR}(v_i) = \ln \frac{P_r(v_i | c'_i(s=0))}{P_r(v_i | c'_i(s=1))}$$
$$= \ln \frac{\frac{1}{\sqrt{2\pi\sigma_z^2}} \exp \left( - \frac{[\|x\|v_i - \|y\|c(s)]^2}{2\sigma_z^2} \right) |_{s=0}}{\frac{1}{\sqrt{2\pi\sigma_z^2}} \exp \left( - \frac{[\|x\|v_i - \|y\|c(s)]^2}{2\sigma_z^2} \right) |_{s=1}}$$
$$= \frac{2 \|x\| \|y\|}{\sqrt{d} \sigma_z^2} v_i, \quad (12)$$
where $P_r(\cdot)$ is the channel posterior probabilities. $c_i$ and $v_i$ are the $i$th components of $c$ and $v$ respectively, where $i = [1, 2, \dots, d]$. $c(s) = \frac{(-1)^s}{\sqrt{d}}$ and $s \in \{0, 1\}$. $\|\cdot\|$ is called a norm on the $d$-dimensional variable. When the information from the left side of the factor graph (i.e. the LLR information of the channel) propagates all the way to the right side of the factor graph, the first iteration of a round trip scheduling is completed. In the factor graph of the IC-LDPC Polar codes with length $N = 2^n$, the polar code part has $n + 1$ stages, and each stage has $N$ nodes, and $(i, j)$ represents the $j$th node in stage $i$. The basic computational blocks (BCBs) [36] of the polar code are shown in figure 4(a). When $N$, there are $n \times N/2$ BCBs, and each BCB is connected to four nodes. The two LLRs (from right to left iterative propagation and from left to right iterative propagation) of node $(i, j)$ iteration $t$ are represented by $L^t_{i,j}$ and $R^t_{i,j}$, respectively, and the information bit sent is judged to be 0 or 1 according to the final LLR value. The value of $L^t_{0,j}$ ($0 \le j \le N$) on the left side of the factor graph is the LLR value of the BI-AWGN channel, and the values of $L^t_{i,j}$ and $R^t_{i,j}$ of a BCB are estimated iteratively using the following calculation method:
$$L^t_{i,j} = g(L^{t-1}_{i+1,2j-1}, L^{t-1}_{i+1,2j} + R^{t-1}_{i,j+N/2}), \quad (13)$$
$$L^t_{i,j+N/2} = g(R^{t-1}_{i,j}, L^{t-1}_{i+1,2j-1}) + L^{t-1}_{i+1,2j}, \quad (14)$$
$$R^t_{i+1,2j-1} = g(R^{t-1}_{i,j}, L^{t-1}_{i+1,2j} + R^{t-1}_{i,j+N/2}), \quad (15)$$
$$R^t_{i+1,2j} = g(R^{t-1}_{i,j}, L^{t-1}_{i+1,2j-1}) + R^{t-1}_{i,j+N/2}, \quad (16)$$
where $g(x, y) = \ln((1 + xy)/(x + y))$ can be approximated to $g(x, y) \approx \text{sign}(x)\text{sign}(y)\min(|x|, |y|)$ to reduce system complexity.

The internal information of the intermediate channel on the right side of the factor graph is transferred to the Tanner graph of LDPC as the priori information (as shown in figure 4(b)) [27]. In the Tanner graph, calculate the propagation information from the variable node to the check node and propagate it to the check node. The initial information is the LLR information of the intermediate channel. The information transmitted from variable node $h$ to check node $f$ in iteration $t$ is:
$$r^t_{hf}(1) = \frac{1}{2} - \frac{1}{2} \prod_{h'}(1 - 2q^{t-1}_{h'f}(1)), \quad (17)$$
$$r^t_{hf}(0) = \frac{1}{2} + \frac{1}{2} \prod_{h'}(1 - 2q^{t-1}_{h'f}(1)), \quad (18)$$
where $h' \in \mathcal{V}_f \setminus h$, $\mathcal{V}_f$ represents the set of all variable nodes connected to check node $f$, and $\mathcal{V}_f \setminus h$ represents the set $\mathcal{V}_f$ except variable node $h$. Then the information is propagated from right to left, and the information from check node $f$ to variable node $h$ is calculated:
$$q^t_{fh}(1) = K_{hf} P_h(1) \prod_{f'} r^t_{f'h}(1), \quad (19)$$
$$q^t_{fh}(0) = K_{hf} P_h(0) \prod_{f'} r^t_{f'h}(0), \quad (20)$$
where $f' \in \mathcal{C}_h \setminus f$, the coefficient $K_{hf}$ ensures that $q^t_{hf}(1) + q^t_{hf}(0) = 1$, $\mathcal{C}_h$ represents the set of all check nodes connected to variable node $h$, and $\mathcal{C}_h \setminus f$ represents the set $\mathcal{C}_h$ except check node $f$. The extrinsic information from the Tanner graph, together with the frozen bits information of the polar codes, propagates to the left of the factor graph to complete the second iteration in a round trip scheduling. Alice continues to iterate BP decoding until the maximum number of iterations is achieved or the early stop condition is satisfied. When Alice succeeds in decoding, the decoding output sequence equals the sequence $b$, and Alice and Bob successfully extract the consistent secret keys.

### 4. Experimental results

This paper analyzes the performance of IC-LDPC Polar codes in CV-QKD systems under low SNR from two aspects of FER and reconciliation efficiency $\beta$. For work on the central processing unit (CPU) platform, eight-dimensional reconciliation is adopted because eight-dimensional reconciliation has the best reconciliation performance compared with other dimensions ($d = 1, 2, 4$) [19]. The modulation variance $V_A$ is 1, the code-length $N$ is 8192, the maximum number of frames is set to $10^4$, the BP algorithm is used for decoding, and the maximum number of iterations $t = 200$.

Figure 5 shows the FER comparison between the IC-LDPC Polar codes with different code-rates and polar codes [26] with different code-rates under different SNRs in the CV-QKD system when the target reconciliation efficiency is set to 98%. It can be observed from the figure 5 that although the SNR is as low as −17 dB to −20.8 dB, the IC-LDPC Polar codes still shows better FER performance than polar codes in the CV-QKD system, and the FER of the IC-LDPC Polar codes can be as low as 0.19. Table 1 further shows the reconciliation efficiency and FER performance of the constructed IC-LDPC Polar codes reconciliation scheme and the previous reconciliation scheme [22] under different SNRs. It can be concluded from the table that when the target reconciliation efficiency $\beta$ of the previous reconciliation scheme is 98%, the secret keys can be extracted with the FER of 0.75 when the SNR is as low as −16.45 dB. When the target reconciliation efficiency $\beta$ is 98%, the scheme in this paper can extract the secret keys when the SNR is from −13.2 dB to −20.8 dB and the minimum FER is 0.19. When the SNR is as low as −19.33 dB, the reconciliation efficiency $\beta$ reaches 99.2%, and the secret keys are extracted with the FER of 0.5. Therefore, the reconciliation scheme proposed in this paper has better reconciliation performance than the previous scheme [22]. It not only extends the range of low SNR that can successfully extract the secret keys but also improves the reconciliation efficiency and reduces the FER.

[IMAGE: Figure 5. FER at target reconciliation efficiency beta = 98% under the channel with different SNRs. The solid line with the crosses results of the polar codes used in the previous scheme [26]. The solid line with the triangles results from the scheme proposed in this paper.]

**Table 1.** The comparison of reconciliation efficiencies ($\beta$) (percentages) and FER for the proposed scheme and previous scheme [22]. The IC-LDPC Polar code-length is 8192.

| Proposed | | | Chen et al [22] | | |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SNR (dB) | $\beta$ | FER | SNR (dB) | $\beta$ | FER |
| −13.20 | 98.45% | 0.46 | −13.43 | 95% | 0.375 |
| −15.56 | 98.75% | 0.36 | −15.42 | 98% | 0.76 |
| −16.44 | 98.56% | 0.32 | −16.45 | 96% | 0.20 |
| −17.12 | 98.60% | 0.28 | −16.45 | 98% | 0.75 |
| −18.57 | 98.07% | 0.30 | — | — | — |
| −19.03 | 98.06% | 0.27 | — | — | — |
| −19.33 | 99.20% | 0.50 | — | — | — |
| −20.13 | 98.06% | 0.19 | — | — | — |
| −20.80 | 98.06% | 0.21 | — | — | — |

Figure 6 shows the change of the secret key rate with transmission distance after applying the constructed IC-LDPC Polar codes into the CV-QKD system. The relationship between the secret key rate and transmission distance of the CV-QKD system between the previous scheme [22] and the scheme proposed in this paper is compared. The secret key rate is compared with Pirandola-Laurenza-Ottaviani-Banchi (PLOB) bound [37] and the theoretical key rate curve of reconciliation efficiency $\beta = 98\%$. The dotted line in the figure represents the key rate of the previous scheme [22] when $\beta = 96\%$ and $\beta = 98\%$, respectively. The solid line with triangles and the solid line with circles in the figure represent the key rate of our scheme when $\beta = 98\%$ and $\beta = 99\%$, respectively. The solid line with crosses represents the theoretical key rate curve of $\beta = 98\%$ (FER = 0), and the solid black line represents the PLOB bound. In contrast, the IC-LDPC Polar codes have practical advantages in the CV-QKD system.

[IMAGE: Figure 6. Finite-size secret key rate vs distance. The solid line with crosses represents the theoretical key rate curve with 98% reconciliation efficiency, and the solid black line represents the PLOB bound. The dotted line in the figure results from the previous scheme [22]. The solid line with triangles and the solid line with circles in the figure result from the scheme proposed in this paper. The security parameters: excess noise is 0.05, the efficiency of the homodyne detector is 0.6, the standard loss of a single-mode optical fiber cable is 0.2 dB km^−1, and the electric noise is 0.015.]

### 5. Summary

In the practical long-distance application of the CV-QKD system, transmission loss leads to severe signal attenuation and a significant reduction of SNR. To a large extent, the secret key extraction depends on efficient post-processing under low SNR. The core part of information reconciliation in the CV-QKD system post-processing is compiling the negotiated data using error correction codes so that both parties can obtain consistent secret keys. Therefore, the performance of the error-correcting codes is the key to limiting the reconciliation efficiency and FER, as well as the secret key rate and transmission distance. This paper applies IC-LDPC Polar codes to the CV-QKD system. The multidimensional reconciliation scheme for continuous-variable QKD with IC-LDPC Polar codes is proposed, which shows better reconciliation performance than polar codes. In this scheme, the best design-SNR is calculated by channel state detection sequence and used to construct polar codes. Then the decoding reliability of each polarization subchannel is calculated by GA. The outer LDPC codes are used to protect the subchannels with smaller leafset sizes and lower decoding reliability to construct IC-LDPC Polar codes. The constructed IC-LDPC Polar code is combined with the multi-dimensional reconciliation algorithm so that both sides of the communication can obtain a consistent key. Compared with the previous scheme, this scheme can maintain above 98% reconciliation efficiency in a lower SNRs range (−13.2 dB to −20.8 dB) to extract the code key, and the minimum FER is 0.19. The reconciliation efficiency of 99.2% can even be achieved in an environment with an SNR of −19.3 dB, while the FER is as low as 0.5. Therefore, this scheme can promote the application of the CV-QKD system in long-distance communication. The data processing in this paper is completed on the CPU platform, and its speed cannot meet the real-time requirements of the system. In future work, the graphics processing unit (GPU) platform can be used to improve the system’s performance. In addition, a rate-adaptive scheme for IC-LDPC Polar codes can be designed to adapt to systems where SNR varies significantly in a short time, such as free-space CV-QKD systems.

### Acknowledgments

This work is supported by the National Natural Science Foundation of China (Grant No. 62071381), Shaanxi Provincial Key R&D Program General Project (Grant No. 2022GY-023), and ISN 23rd Open Project (Grant No. ISN23-06) of the State Key Laboratory of Integrated Services Networks (Xidian University).

### References

[1] Morris J D, Grimaila M R, Hodson D D, Jacques D and Baumgartner G 2014 A survey of quantum key distribution (QKD) technologies *Emerging Trends in ICT Security* (Amsterdam: Elsevier) pp 141–52
[2] Scarani V, Bechmann-Pasquinucci H, Cerf N J, Duˇsek M, Lütkenhaus N and Peev M 2009 *Rev. Mod. Phys.* **81** 1301
[3] Giampouris D 2017 *Genedis 2016* (Berlin: Springer) pp 149–57
[4] Bennett C H and Brassard G 2020 arXiv:2003.06557
[5] Braunstein S L and Loock P V 2005 *Rev. Mod. Phys.* **77** 513
[6] Zhang Y et al 2019 *Quantum Sci. Technol.* **4** 035006
[7] Weedbrook C, Lance A M, Bowen W P, Symul T, Ralph T C and Lam P K 2004 *Phys. Rev. Lett.* **93** 170504
[8] Brassard G, Lütkenhaus N, Mor T and Sanders B C 2000 *Phys. Rev. Lett.* **85** 1330
[9] Grosshans F and Grangier P 2002 *Phys. Rev. Lett.* **88** 057902
[10] Leverrier A, García-Patrón R, Renner R and Cerf N J 2013 *Phys. Rev. Lett.* **110** 030502
[11] Leverrier A 2015 *Phys. Rev. Lett.* **114** 070501
[12] Wu J and Zhuang Q 2021 *Phys. Rev. Appl.* **15** 034073
[13] Zou M, Mao Y and Chen T 2022 *J. Phys. B: At. Mol. Opt. Phys.* **55** 155502
[14] Grosshans F, Cerf N J, Wenger J, Tualle-Brouri R and Grangier P 2003 arXiv:quant-ph/0306141
[15] Grosshans F and Grangier P 2002 arXiv:quant-ph/0204127
[16] Zhou C, Wang X, Zhang Y, Zhang Z, Yu S and Guo H 2019 *Phys. Rev. Appl.* **12** 054013
[17] Silberhorn C, Ralph T C, Lütkenhaus N and Leuchs G 2002 *Phys. Rev. Lett.* **89** 167901
[18] Wang X, Wang H, Zhou C, Chen Z, Yu S and Guo H 2022 *Opt. Express* **30** 30455–65
[19] Leverrier A, Alléaume R, Boutros J, Zémor G and Grangier P 2008 *Phys. Rev. A* **77** 042325
[20] Jouguet P, Kunz-Jacques S and Leverrier A 2011 *Phys. Rev. A* **84** 062317
[21] Wang X, Zhang Y, Li Z, Xu B, Yu S and Guo H 2017 arXiv:1703.04916
[22] Zhou C, Wang X, Zhang Z, Yu S, Chen Z and Guo H 2021 *Sci. China Phys. Mech. Astron.* **64** 260311
[23] Arikan E 2009 *IEEE Trans. Inf. Theory* **55** 3051–73
[24] Feng B, Lv C, Liu J and Zhang T 2021 A continuous variable quantum key distribution protocol based on multi-dimensiondata reconciliation with polar code *J. Phys.: Conf. Ser.* **1757** 012111
[25] Nakassis A and Mink A 2014 Polar codes in a QKD environment *Proc. SPIE* **9123** 32–42
[26] Zhao S, Shen Z, Xiao H and Wang L 2018 *Sci. China Phys. Mech. Astron.* **61** 090323
[27] Gallager R 1962 *IRE Trans. Inf. Theory* **8** 21–28
[28] Abbas S M, Fan Y, Chen J and Tsui C 2017 Concatenated LDPC-polar codes decoding through belief propagation *2017 IEEE Int. Symp. on Circuits and Systems (ISCAS) (IEEE)* pp 1–4
[29] Vangala H, Viterbo E and Hong Y 2015 arXiv:1501.02473
[30] Chen R, Huang P, Li D, Zhu Y and Zeng G 2019 *Entropy* **21** 1146
[31] Chung S Y, Richardson T J and Urbanke R L 2001 *IEEE Trans. Inf. Theory* **47** 657–70
[32] Trifonov P 2012 *IEEE Trans. Commun.* **60** 3221–7
[33] Eslami A and Pishro-Nik H 2013 *IEEE Trans. Commun.* **61** 919–29
[34] Xu J, Che T and Choi G 2015 XJ-BP: express journey belief propagation decoding for polar codes *2015 IEEE Global Communications Conf. (GLOBECOM) (IEEE)* pp 1–6
[35] Guo J, Qin M, i Fabregas A G and Siegel P H 2014 Enhanced belief propagation decoding of polar codes through concatenation *2014 IEEE Int. Symp. on Information Theory (IEEE)* pp 2987–91
[36] Kim J H, Kim I, Kim G and Song H Y 2017 *IEICE Trans. Fundam. Electron. Commun. Comput. Sci.* **100** 2052–5
[37] Pirandola S, Laurenza R, Ottaviani C and Banchi L 2017 *Nat. Commun.* **8** 1–15