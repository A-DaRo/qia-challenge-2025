# nature photonics ARTICLES

PUBLISHED ONLINE: 25 MAY 2015 | DOI: 10.1038/NPHOTON.2015.83

**High-rate measurement-device-independent quantum cryptography**

Stefano Pirandola¹*, Carlo Ottaviani¹, Gaetana Spedalieri¹, Christian Weedbrook²,³, Samuel L. Braunstein¹, Seth Lloyd⁴, Tobias Gehring⁵, Christian S. Jacobsen⁵ and Ulrik L. Andersen⁵

¹Computer Science and York Centre for Quantum Technologies, University of York, York YO10 5GH, UK. ²Department of Physics, University of Toronto, Toronto M5S 3G4, Canada. ³QKD Corporation, 112 College Street, Toronto M5G 1L6, Canada. ⁴Department of Mechanical Engineering and Research Laboratory of Electronics, Massachusetts Institute of Technology, Cambridge, Massachusetts 02139, USA. ⁵Department of Physics, Technical University of Denmark, Fysikvej, Kongens Lyngby 2800, Denmark. *e-mail: stefano.pirandola@york.ac.uk

Quantum cryptography achieves a formidable task—the remote distribution of secret keys by exploiting the fundamental laws of physics. Quantum cryptography is now headed towards solving the practical problem of constructing scalable and secure quantum networks. A significant step in this direction has been the introduction of **measurement-device independence**, where the secret key between two parties is established by the measurement of an untrusted relay. Unfortunately, although qubit-implemented protocols can reach long distances, their key rates are typically very low, unsuitable for the demands of a metropolitan network. Here we show, theoretically and experimentally, that a solution can come from the use of **continuous-variable systems**. We design a **coherent-state network protocol** able to achieve remarkably high key rates at **metropolitan distances**, in fact **three orders of magnitude higher** than those currently achieved. Our protocol could be employed to build high-rate quantum networks where devices securely connect to nearby access points or proxy servers.

Quantum key distribution (QKD)¹,² is one of the most active areas in quantum information³,⁴, with a number of in-field implementations, including the development of networks based on point-to-point QKD protocols⁵⁻⁸. A typical QKD protocol involves two parties, conventionally called Alice and Bob, who aim to generate a secret key by exchanging quantum systems over an insecure communication channel. Security is assessed against the most powerful attack on the channel, where an eavesdropper, conventionally called Eve, perturbs the quantum systems using the most general strategies allowed by quantum mechanics.

Although this theoretical analysis is fundamental for testing the basic security of a protocol, it may be insufficient to guarantee its viability in realistic implementations, where flaws in the devices may provide alternative 'side-channels' to be attacked⁹,¹⁰. These weaknesses naturally arise in realistic models of networks (for example, the Internet) where two end-users are not connected by direct lines but must exploit one or more intermediate nodes, whose operation may be tampered with by Eve. For this scenario, ref. 11 has introduced a general method to guarantee security, designing a swapping-like protocol where secret correlations are established by the measurement of a third untrusted party. This technique is known as 'measurement-device independence' (MDI)¹²⁻²¹.

In this Article we extend the notion of MDI to **continuous-variable** (CV) systems, providing an unconditional security proof in the presence of Gaussian resources. In this way we generalize the field of CV quantum cryptography⁴,²²⁻²⁹ from point-to-point to a more robust end-to-end formulation³⁰,³¹. In fact, we consider the basic network topology where Alice and Bob communicate by connecting to an untrusted relay via insecure links. To create secret correlations, they transmit random coherent states to the relay where a CV Bell detection is performed and the outcome broadcast to the parties. Despite the possibility that the relay could be fully tampered with, and the links subject to optimal coherent attacks, Alice and Bob are still able to extract a secret key.

Our analysis shows that remarkably high rates can be achieved at metropolitan distances ($\le 25$ km). This theoretical prediction is confirmed by a proof-of-principle experiment, where the equivalent of $\approx 10^{-1}$ secret key bits per relay use are distributed at 4 dB loss, corresponding to 20 km in standard optical fibre (at the loss rate of 0.2 dB km⁻¹). This is three orders of magnitude higher than the rate of $\le 10^{-4}$ bits per use achievable at 20 km by qubit-based protocols¹²,¹⁹⁻²¹. In a future field implementation, our rate would correspond to $\approx 10$ Mbits s⁻¹ using state-of-the-art clock rates and fast homodyne detectors at 100 MHz (ref. 32), thus fulfilling the demands of a high-rate metropolitan network.

The optimal configuration of our protocol is **asymmetric**, with the relay being closer to one party, for example, Alice. This resembles the typical topology of a public network where a user connects their device to a proxy server to communicate with remote users. Because our set-up is studied in the near-infrared regime (1,064 nm), it can simulate a wireless infrared device (for example, the infrared port of a laptop or phone) connecting to a nearby untrusted access point, which is in turn the hub of a star network of remote ground users, for example, connected by optical fibres.

For simplicity, we start by describing the protocol with noiseless links, explaining the basic mechanism of the relay. We then consider the most general eavesdropping strategy against the relay and the links, showing how this strategy can be reduced to a coherent Gaussian attack of the links only. Finally, we derive the secret key rate of the protocol and we compare theoretical and experimental performances in the optimal configuration.

### Basic concept

Consider two distant parties, Alice and Bob, aiming to share a secret key. At one side, Alice prepares a mode $A$ in coherent state $| \alpha \rangle$ whose amplitude $\alpha$ is modulated by a Gaussian distribution with zero mean and sufficiently large variance $\sigma^2$ in each quadrature. At

NATURE PHOTONICS | VOL 9 | JUNE 2015 | www.nature.com/naturephotonics
397

ARTICLES
NATURE PHOTONICS DOI: 10.1038/NPHOTON.2015.83

**Figure 1 | Basic protocol and its general eavesdropping.** a, Basic modus operandi of the CV Bell relay (see main text for explanation). b, The most general joint attack against the protocol. In each use of the relay, modes $A$ and $B$ unitarily interact with ancillary vacuum modes. Two outputs are used to simulate the classical outcomes of the relay, while the other output modes $E$ are stored in a quantum memory (QM) measured by Eve at the end of the protocol.

[IMAGE: Diagram showing the basic protocol and its general eavesdropping.
a, Shows the basic protocol where Alice sends mode A ($|\alpha\rangle$) and Bob sends mode B ($|\beta\rangle$) to a Relay, where a Beamsplitter and conjugate homodyning (P+, P-) leads to classical outcome $\gamma$.
b, Shows the general eavesdropping attack, where modes A and B are intercepted by Eve (unitarily $U$), interacting with ancillary vacuum modes, with outputs $A'$ and $B'$ going to a QM Simulator and other outputs $E$ going to a Quantum Memory (QM).]

the other side, Bob prepares his mode $B$ in another coherent state $| \beta \rangle$ whose amplitude $\beta$ is modulated by the same Gaussian distribution as Alice. Modes $A$ and $B$ are then sent to an intermediate station, which is the CV Bell relay shown in Fig. 1a.

The relay performs a CV Bell detection on the incoming modes by mixing them in a balanced beamsplitter whose output ports are conjugately homodyned³³. This detection corresponds to measuring the quadrature operators $q_- = (q_A - q_B) / \sqrt{2}$ and $p_+ = (p_A + p_B) / \sqrt{2}$, whose classical outcomes are combined in a complex variable $\gamma := (q_- + i p_+) / \sqrt{2}$ with probability $p(\gamma)$. The outcome $\gamma$ is then communicated to Alice and Bob via a classical public channel.

In this process the relay acts as a correlator¹¹. One can check that the outcome $\gamma$ creates *a posteriori* correlations between the parties, being equal to $\gamma = \alpha - \beta^* + \delta$, where $\delta$ is the detection noise. As a result, knowledge of $\gamma$ enables each party to infer the variable of the other party by simple postprocessing. For instance, Bob may compute $\beta \approx \alpha - \gamma + \delta$, decoding Alice's variable. Thus, conditioned on $\gamma$, Alice and Bob's mutual information increases from $I(\alpha:\beta) = 0$ to $I(\alpha:\beta|\gamma) > 0$.

Averaging over all possible outputs $\gamma$, the honest parties will share $I_{\text{AB}} = \int d^2\gamma p(\gamma) I(\alpha:\beta|\gamma)$ mean bits per use of the relay, which is logarithmically increasing in the modulation $\sigma^2$. Despite Eve also having access to the classical communication and operating the relay, she cannot steal any information, as she only knows $\gamma$ and $I(\alpha:\gamma) = I(\beta:\gamma) = 0$. As a result, Eve is forced to attack the links and/or corrupt the relay.

### Protocol under general eavesdropping

The most general eavesdropping strategy of our protocol is a joint attack involving both the relay and the two links, as depicted in Fig. 1b. In each use of the relay, Eve may intercept the two modes $A$ and $B$, and make them interact with an ensemble of ancillary vacuum modes via a general unitary $U$. Among the output modes, two are sent to a simulator of the relay, where they are homodyned and the result $\gamma$ broadcast. The remaining modes $E$ are stored in a quantum memory, which is measured at the end of the protocol.

Note that a more general attack may involve a unitary applied to all modes that are transmitted over many uses of the relay. However, this can always be reduced to the previous attack, coherent within a single use, by assuming that Alice and Bob perform random permutations on their data³⁴,³⁵. Also note that, in Eve's simulator, any

398

**Figure 2 | Protocol in the presence of a coherent two-mode Gaussian attack.** The two travelling modes, $A$ and $B$, are mixed with two ancillary modes, $E_1$ and $E_2$, via two beamsplitters, with transmissivities $\tau_A$ and $\tau_B$, which introduce thermal noise with variances, $\omega_A$ and $\omega_B$, respectively. The ancillary modes are generally quantum-correlated and belong to a reservoir of ancillas $(E_1, e, E_2)$, which is globally in a pure Gaussian state. All of Eve's output is stored in a quantum memory (QM) measured at the end of the protocol.

[IMAGE: Diagram showing a coherent two-mode Gaussian attack. Alice sends $|\alpha\rangle$ (mode A) and Bob sends $|\beta\rangle$ (mode B). Mode A is mixed via Beamsplitter $T_A$ with mode $E_1$. Mode B is mixed via Beamsplitter $T_B$ with mode $E_2$. The ancillary modes $E_1, E_2$, along with ancilla $e$, form a pure Gaussian reservoir $\rho_{E_1,e,E_2}$ measured by a Quantum Memory (QM) at the end. The outputs $A'$ and $B'$ from the beamsplitters are sent to the Relay where a QM is performed yielding classical outcome $\gamma$.]

other higher-rank measurement of the quadratures can always be purified into a rank-one measurement by enlarging the set of the ancillas, $E$. If other observables are measured or no detection occurs, the communication of a fake variable $\gamma$ can easily be detected from analysis of the empirical data¹¹.

To deal with the joint attack of Fig. 1b, Alice and Bob must retrieve the joint statistics of the variables $\alpha$, $\beta$ and $\gamma$. As the protocol is performed many times, Alice and Bob can compare a small part of their data via the public channel and reconstruct the probability distribution $p(\alpha, \beta, \gamma)$. As we show in Supplementary Sections IA and IB, for any observed distribution $p(\alpha, \beta, \gamma)$, the security of the protocol does not change if we modify Eve's unitary $U$ in such a way that her simulator works exactly as the original relay (so that the modes are mixed in a balanced beamsplitter and conjugately homodyned). Thus, we can assume that the relay is properly operated (even if by Eve) with the unitary $U$ restricted to be a coherent attack against the two links.

The description of this attack can be simplified further. Because the protocol is based on Gaussian modulation and detection of Gaussian states, its optimal eavesdropping is based on a Gaussian unitary $U$ (ref. 36). Thus, from the first- and second-order statistical moments of the observed distribution $p(\alpha, \beta, \gamma)$, Alice and Bob may construct a Gaussian distribution $p_{\mathcal{G}}(\alpha, \beta, \gamma)$ and design a corresponding optimal Gaussian attack against the links.

From an operational point of view, the first-order moments are used to construct the optimal estimators of each other variable, while the second-order moments are used to derive the secret key rate of the protocol. In particular, Alice and Bob are able to compute their mutual information $I_{\text{AB}}$ and upper-bound Eve's stolen information $I_E$ via the Holevo bound³. As long as the condition $R := I_{\text{AB}} - I_E > 0$ is satisfied, they can postprocess their data via standard procedures of error correction and privacy amplification¹ and distil an average of $R$ secret key bits per use of the relay.

The formula for rate $R$ in the previous paragraph is valid when we consider infinite uses of the relay. However, this can be very well approximated after a large but finite number of rounds ($\ge 10^8$ in our case) by checking that the rate quickly reaches its asymptotic value. Notwithstanding this fast convergence, another finite-size effect⁴,³⁷⁻³⁹ must still be considered: the fact that the reconciliation procedure is based on realistic codes with finite block size³⁷. As a result, one has to introduce a reconciliation efficiency $\xi \le 1$ and consider $R = \xi I_{\text{AB}} - I_E$.

### Coherent Gaussian attack of the links

Following the previous reasoning, security analysis of the protocol can be reduced to studying a coherent Gaussian attack against the

NATURE PHOTONICS | VOL 9 | JUNE 2015 | www.nature.com/naturephotonics

ARTICLES
NATURE PHOTONICS DOI: 10.1038/NPHOTON.2015.83

**Figure 3 | Behaviour of the ideal rate in terms of the transmissivities of Alice's and Bob's links.** Plot of $R(\tau_A, \tau_B, \varepsilon)$ versus $\tau_A$ and $\tau_B$, for pure-loss ($\varepsilon = 0$, solid lines) and non-zero excess noise ($\varepsilon = 0.05$, dashed lines). Contour lines are shown for each case. Note that high secret key rates are achievable, on the order of 1 bit per relay use, and these are robust to the presence of excess noise. When Alice's link has small loss ($\tau_A \ge 0.9$), Bob's link can become very lossy (up to $\tau_B \approx 0$), and still a secret key can be extracted.

[IMAGE: Graph showing contours of the secret key rate R (bits/use) as a function of transmissivities $\tau_A$ (Alice's link) and $\tau_B$ (Bob's link). The axes range from 0.75 to 1.00. Contours show high rates (up to 4.0) achievable even when $\tau_B$ is low, provided $\tau_A$ is high. A region $R < 0$ is shown for low $\tau_A$ and $\tau_B$. Solid lines represent $\varepsilon=0$ (pure loss); dashed lines represent $\varepsilon=0.05$ (excess noise).]

two links, assuming a properly working relay. Attacks of this kind can be constructed by correlating two canonical forms⁴⁰. The most realistic scenario is the Gaussian attack depicted in Fig. 2.

In this attack, the two travelling modes $A$ and $B$ are mixed with two ancillary modes, $E_1$ and $E_2$, by two beamsplitters with transmissivities $\tau_A$ and $\tau_B$, respectively. These ancillary modes are generally quantum-correlated and belong to a reservoir of ancillas ($E_1$, $E_2$ plus an extra set $e$), which is globally described by a pure Gaussian state. The reduced state $\rho_{E_1 E_2}$ of the injected ancillas is a correlated thermal state with zero mean and covariance matrix in the normal form
$$
\mathbf{V}_{E_1 E_2} = \begin{pmatrix} \omega_A \mathbf{I} & \mathbf{G} \\ \mathbf{G}^T & \omega_B \mathbf{I} \end{pmatrix}; \quad \mathbf{G} := \begin{pmatrix} g & 0 \\ 0 & g' \end{pmatrix} \quad \text{(1)}
$$
where $\omega_A, \omega_B \ge 1$ are the variances of the thermal noise affecting each link, and $g$ and $g'$ are correlation parameters, which must satisfy physical constraints⁴¹,⁴² (see Supplementary Section ID). After interaction, all of Eve's ancillas are stored in a quantum memory, which is coherently measured at the end of the protocol.

Note that our description of the attack is very general because any two-mode Gaussian state can be transformed in the previous normal form by local Gaussian unitaries⁴¹. In general, the injected state $\rho_{E_1 E_2}$ of the two ancillas can be separable or entangled. The simplest case is when there are no correlations ($g = g' = 0$), so that $\rho_{E_1 E_2}$ is a tensor product and the attack collapses into a collective one-mode attack with two independent entangling cloners²³. Such one-mode attacks are simple but clearly too restrictive for assessing the unconditional security of our protocol, because the presence of quantum correlations between Eve's ancillas is not only possible but also optimal at fixed thermal noise (Supplementary Section IE5).

### Secret key rate of the protocol

We assume that Alice is the encoder and Bob the decoder, which means that $\alpha$ is inferred by processing $\beta$ into an optimal estimator. This is possible because the relay provides $\gamma \approx \sqrt{\tau_A} \alpha - \sqrt{\tau_B} \beta^*$ in the

**Figure 4 | Free-space experimental set-up.** Alice and Bob apply amplitude and phase modulators to a pair of identical classical phase-locked bright coherent beams (coming from a common local oscillator at 1,064 nm). Alice's and Bob's stations (coloured boxes) are private spaces whose internal loss and noise are fully trusted (this assumption is crucial for MDI and guarantees security in the absence of entanglement⁴³). This feature also allows us to set the signal levels at the output of the stations. From these stations, the modes emerge randomly displaced in phase space according to a Gaussian distribution with high modulation variance $\sigma^2 \approx 60$. Losses in the links are simulated by suitably attenuating the variances of the modulations. At the relay, the modes are mixed in a balanced beamsplitter and the output ports photodetected. Photocurrents are finally processed to realise a CV Bell measurement (see Supplementary Section IIA for more details).

[IMAGE: Diagram of the free-space experimental set-up. Alice and Bob stations, in colored boxes, each contain RNGs (Random Number Generators), PBS (Polarizing Beamsplitters) and EOMs (Electro-Optical Modulators). Classical phase-locked bright coherent beams (p, q quadratures) are shown entering these stations and being modulated. The beams travel through free space and are mixed at an untrusted relay using a 50:50 beamsplitter. The outputs are detected by PDs (Photodetectors), which are connected to a DAQ (Data Acquisition) system.]

NATURE PHOTONICS | VOL 9 | JUNE 2015 | www.nature.com/naturephotonics
399

ARTICLES
NATURE PHOTONICS DOI: 10.1038/NPHOTON.2015.83

**Figure 5 | Experimental results and comparison with theoretical predictions.** a, Secret key rate (bits per relay use) versus Bob's loss (in dB) for (i) $\tau_A \approx 0.98$, that is, Alice connected to the relay by a short free-space link. Experimental points refer to $\xi = 1$ (green circles), $\xi \approx 0.97$ (blue squares) and $\xi \approx 0.95$ (red diamonds). For comparison, we plot the theoretical rate $R_{\sigma^2 \approx 60, \xi}(\tau_A, \tau_B, \varepsilon)$, for $\xi = 1$ and $\varepsilon = 0$ (solid black line), $\xi = 1$ and $\varepsilon \approx 0.01$ (dashed black line), $\xi \approx 0.97$ and $\varepsilon = 0$ (blue line) and $\xi \approx 0.95$ and $\varepsilon = 0$ (red line). b, As in **a**, but considering (ii) $\tau_A \approx 0.975$, equivalent to a 100 m fibre-link for Alice. c, As in **a**, but considering (iii) $\tau_A \approx 0.935$, equivalent to a 1 km fibre-link for Alice. d, Secret key rate versus the total distance between Alice and Bob in simulated fibre. Experimental key rates ($\xi \approx 0.97$, blue squares) for the three configurations (i) to (iii) are compared with the rate achievable by MDI-QKD with qubits¹² (thick solid line) and the secret key capacity of a direct fibre link between Alice and Bob, which lies between the lower bound (LB) of ref. 44 and the upper bound (UB) of ref. 45. At metropolitan distances, we outperform qubit-based protocols by (at least) three orders of magnitude, missing the secret key capacity by approximately one order of magnitude.

[IMAGE: Four graphs comparing secret key rate (bits/use) against loss (dB) or total distance (km).
a: Rate vs Bob's loss (dB) for $\tau_A \approx 0.98$. Experimental data points (circles, squares, diamonds) follow theoretical curves (solid, dashed, blue, red lines).
b: Rate vs Bob's loss (dB) for $\tau_A \approx 0.975$.
c: Rate vs Bob's loss (dB) for $\tau_A \approx 0.935$.
d: Rate vs Total distance (km) for the three configurations (i), (ii), (iii), comparing experimental CV MDI-QKD rates (blue squares) against qubit MDI-QKD (thick solid line), and the theoretical Lower Bound (LB) and Upper Bound (UB) of secret key capacity.]

attack of Fig. 2 and the empirical values of the transmissivities $\tau_A$ and $\tau_B$ are accessible to the parties from the first-order moments of $p(\alpha, \beta, \gamma)$. Then, from the second-order moments of $p(\alpha, \beta, \gamma)$, Alice and Bob can derive their mutual information $I_{\text{AB}} = \log_2(\chi^{-1})$, where $\chi$ is the equivalent noise, decomposable as $\chi = \chi_{\text{loss}} + \varepsilon_{\text{exc}}$, with $\chi_{\text{loss}}(\tau_A \tau_B)$ being the pure-loss noise and $\varepsilon_{\text{exc}}(\tau_A \tau_B \omega_A, \omega_B g, g')$ the 'excess noise'. From the second-order moments, the remote parties can also compute the secret key rate $R_{\sigma^2, \xi}(\tau_A, \tau_B, \varepsilon)$, which depends on the modulation $\sigma^2$ and the reconciliation efficiency $\xi$, besides the main parameters of the attack, that is, transmissivities, $\tau_A$ and $\tau_B$ and excess noise $\varepsilon$.

To investigate the best performances of the protocol, let us consider large modulation $\sigma^2 \gg 1$ and ideal reconciliation $\xi = 1$. In this limit, the rate becomes
$$
R(\tau_A, \tau_B, \varepsilon) = \log_2 \left[ \frac{2(\tau_A + \tau_B)}{ \varepsilon | \tau_A - \tau_B | \chi / (\tau_A + \tau_B) } \right] + h \left[ \frac{\tau_A \chi}{ \tau_A + \tau_B } - h \left[ \frac{1 - \tau_B}{ \tau_A + \tau_B } \chi - \frac{1 - \tau_A}{ \tau_A + \tau_B } \chi + \frac{(\tau_A - \tau_B)^2}{ (\tau_A + \tau_B)^2 } \right] \right]
$$
$$
R(\tau_A, \tau_B, \varepsilon) = \log_2 \left[ \frac{2(\tau_A + \tau_B)}{ \varepsilon |\tau_A - \tau_B| \chi / (\tau_A + \tau_B) } \right] + h \left[ \frac{\tau_A \chi}{ \tau_A + \tau_B } - h \left[ \frac{1 - \tau_B}{ \tau_A + \tau_B } \chi - \frac{1 - \tau_A}{ \tau_A + \tau_B } \chi + \frac{(\tau_A - \tau_B)^2}{ (\tau_A + \tau_B)^2 } \right] \right] \quad \text{(2)}
$$
where $\chi = \chi(\tau_A, \tau_B, \varepsilon) := 2(\tau_A + \tau_B) / \tau_A \tau_B + \varepsilon$ and
$$
h(x) := -\frac{x+1}{2} \log_2 \frac{x+1}{2} - \frac{x-1}{2} \log_2 \frac{x-1}{2}
$$
The behaviour of this ideal rate is plotted in Fig. 3.

As we can see from Fig. 3, extremely high secret key rates ($\approx 1$ bit/use) can theoretically be achieved by our protocol. Symmetric configurations $\tau_A \approx \tau_B$ are not the best solution, as they are secure

400

only for transmissivities $\ge 0.84$, corresponding to links $\le 3.8$ km in standard fibres ($0.2$ dB km⁻¹). The optimal configuration is **asymmetric** and corresponds to a small loss in Alice's link, in which case the transmissivity of Bob's link can be close to zero. These features are robust to the presence of excess noise, for instance at $\varepsilon = 0.05$, which is higher than the typical values appearing in experiments ($\varepsilon \le 0.008$ in ref. 29).

The asymmetry in the transmissivities comes from the term $h[\tau_A \chi/(\tau_A + \tau_B) - 1]$ in equation (2). Physically, it comes from the difference between direct and reverse reconciliation in CV-QKD. In fact, if one link is lossless, the Bell detection is done locally and our scheme approaches a point-to-point protocol in direct reconciliation (for $\tau_B = 1$) or reverse reconciliation (for $\tau_A = 1$). See Supplementary Section IE7 for more details.

### Experimental proof of principle

Our theory has been confirmed experimentally by the free-space set-up depicted in Fig. 4. We have reproduced the asymmetric configuration where Alice's transmissivity $\tau_A$ is sufficiently high (Alice close to the relay), while Bob's transmissivity $\tau_B$ has been progressively decreased (Bob far from the relay). In particular, we have simulated three different scenarios for Alice: (i) Alice connected to the relay by a short free-space link, so that her loss is only due to the global detection efficiency at the relay ($\tau_A \approx 0.98$); (ii) Alice connected to the relay at an equivalent distance of 100 m in standard fibre ($\tau_A \approx 0.975$); and (iii) Alice at an equivalent distance of 1 km in standard fibre ($\tau_A \approx 0.935$). For every experimental point, we have evaluated the second-order moments of $p(\alpha, \beta, \gamma)$ and computed the experimental secret key rate $R$, assuming different

NATURE PHOTONICS | VOL 9 | JUNE 2015 | www.nature.com/naturephotonics

NATURE PHOTONICS DOI: 10.1038/NPHOTON.2015.83
ARTICLES

values of the reconciliation efficiency: $\xi = 1$ (ideal), $\xi \approx 0.97$ (achievable³⁷) and $\xi \approx 0.95$.

The experimental results are plotted in Fig. 5 and compared with theoretical predictions. Assuming ideal reconciliation ($\xi = 1$), the extrapolated experimental rate is not far from the theoretical rate of a pure-loss attack $R_{\sigma^2 \approx 60, \xi=1}(\tau_A, \tau_B, 0)$, which provides the maximum performance achievable at the considered links' transmissivities. It is important to note that, due to inevitable experimental imperfections, there is some excess noise $\varepsilon \approx 0.01$ entering our data, which is assumed to come from a two-mode Gaussian attack in our experiment.

Considering realistic reconciliation performances, the experimental rates are not far from the maximum theoretical predictions. In particular, for $\xi \approx 0.97$, the experimental rate can reach remarkably high values over typical connection lengths of a metropolitan network. For instance, if Alice connects to a public hot spot via a free-space link, she can distill $R \approx 10^{-1}$ secret bits per relay use with Bob being $\approx 20$ km far away in standard fibre (distance equivalent of 4 dB loss). A similar performance is achieved if Alice connects to a network router using a 100 m fibre (as may happen within a building). If Alice is 1 km away (for example, in a full fibre-optic network), then she can extract $\approx 10^{-1}$ bits per use with Bob at $\approx 10$ km.

These experimental secret key rates are (at least) three orders of magnitude higher than those achievable with qubit-based protocols over comparable distances, for which one has $\le 10^{-4}$ bits per use at $\le 25$ km¹²,¹⁹⁻²¹. This is also clear from Fig. 5d, which reports the rate of standard MDI-QKD¹², limiting the number of key bits per use achievable by the various qubit-based implementations¹⁹⁻²¹ (independently from their clock rates). Note that our experimental rates are only one order of magnitude below the secret key capacity of the total lossy channel between Alice and Bob. This is the optimal rate reachable by CV-QKD protocols over a direct lossy link between the two parties, lower-bounded by ref. 44 and upper-bounded by ref. 45. We are therefore able to achieve the high-rate performances of CV-QKD despite the fact we are removing a direct point-to-point connection.

### Discussion

In this work we have extended the field of CV quantum cryptography to a network formulation where two end-users do not access a direct quantum communication channel but are forced to connect to an untrusted relay via insecure quantum links. Despite the possibility that the relay could be fully corrupted and its links subject to coherent attacks, the end-users can still extract a secret key. This surprising result comes from a demanding security analysis of our model, which represents the first CV protocol whose rate has been explicitly computed against a two-mode coherent attack.

An important feature is the simplicity of the relay, which does not possess quantum sources (for example, entanglement) but just performs a standard optical measurement, with all the heavy procedures of data post-processing left to the end-users, fulfilling the idea behind the end-to-end principle. In particular, the relay implements a CV Bell detection that involves highly efficient photodetectors plus linear optics, whereas the discrete-variable version of this measurement needs nonlinear elements to operate deterministically. This feature, combined with the use of coherent states, makes the scheme very attractive, guaranteeing both cheap implementation and high rates.

We have found that the optimal network configuration is asymmetric, with the untrusted relay acting as a proxy server near to one of the parties. We have then proven, experimentally, that remarkable rates can be reached, several orders of magnitude higher than those achievable with qubit-based protocols over comparable distances and only one order of magnitude below the secret key capacity. From this point of view, our protocol could be used for setting up high-rate metropolitan networks, for example, based on nearby

NATURE PHOTONICS | VOL 9 | JUNE 2015 | www.nature.com/naturephotonics
401

access points. A discussion on the potential applications in star and other types of network topologies may be found in the Supplementary Section IIIA. Future directions include a complete analysis of the finite-size effects and a real-time field implementation.

### Methods

Theoretical methods, experimental details and data post-processing can be found in Supplementary Sections I and II. The security analysis of the protocol was performed in the entanglement-based representation. The post-relay conditional state of Alice and Bob has been purified into an environment that is fully controlled by Eve. In these conditions we have studied the general secret key rate of the protocol, which has been analytically computed for the coherent two-mode Gaussian attack.

Methods and any associated references are available in the online version of the paper.

Received 6 May 2014; accepted 19 April 2015; published online 25 May 2015

### References

30. Saltzer, J. H., Reed, D. P. & Clark, D. D. End-to-end arguments in system design. *ACM Trans. Comput. Syst.* 2, 277-288 (1984).
31. Baran, P. On distributed communications networks. *IEEE Trans. Commun. Syst.* 12, 1-9 (1964).
32. Chi, Y.-M. et al. A balanced homodyne detector for high-rate Gaussian-modulated coherent-state quantum key distribution. *New. J. Phys.* 13, 013003 (2011).
33. Spedalieri, G., Ottaviani, C. & Pirandola, S. Covariance matrices under Bell-like detections. *Open Syst. Inf. Dyn.* 20, 1350011 (2013).
34. Renner, R. Symmetry of large physical systems implies independence of subsystems. *Nature Phys.* 3, 645-649 (2007).
35. Renner, R. & Cirac, J. I. de Finetti representation theorem for infinite-dimensional quantum systems and applications to quantum cryptography. *Phys. Rev. Lett.* 102, 110504 (2009).
36. García-Patrón, R. & Cerf, N. J. Unconditional optimality of Gaussian attacks against continuous-variable quantum key distribution. *Phys. Rev. Lett.* 97, 190503 (2006).
37. Jouguet, P., Kunz-Jacques, S. & Leverrier, A. Long-distance continuous-variable quantum key distribution with a Gaussian modulation. *Phys. Rev. A* 84, 062317 (2011).
38. Curty, M. et al. Finite-key analysis for measurement-device-independent quantum key distribution. *Nature Commun.* 5, 3732 (2014).
39. Leverrier, A. Composable security proof for continuous-variable quantum key distribution with coherent states. *Phys. Rev. Lett.* 114, 070501 (2015).
40. Pirandola, S., Braunstein, S. L. & Lloyd, S. Characterization of collective Gaussian attacks and security of coherent-state quantum cryptography. *Phys. Rev. Lett.* 101, 200504 (2008).
41. Pirandola, S., Serafini, A. & Lloyd, S. Correlation matrices of two-mode bosonic systems. *Phys. Rev. A* 79, 052327 (2009).
42. Pirandola, S. Entanglement reactivation in separable environments. *New J. Phys.* 15, 113046 (2013).
43. Pirandola, S. Quantum discord as a resource for quantum cryptography. *Sci. Rep.* 4, 6956 (2014).
44. Pirandola, S., García-Patrón, R., Braunstein, S. L. & Lloyd, S. Direct and reverse secret-key capacities of a quantum channel. *Phys. Rev. Lett.* 102, 050503 (2009).
45. Takeoka, M., Guha, S. & Wilde, M. M. Fundamental rate-loss tradeoff for optical quantum key distribution. *Nature Commun.* 5, 5235 (2014).

### Acknowledgements

S.P. acknowledges support from the Engineering and Physical Sciences Research Council via the 'UK Quantum Communications HUB' (EP/M013472/1) and grants 'qDATA' (EP/L011298/1) and 'HIPERCOM' (EP/J00796X/1). S.P. also acknowledges the Leverhulme Trust (research fellowship 'qBIO'). T.G. acknowledges support from the H.C. Ørsted postdoctoral programme. U.L.A. acknowledges the Danish Agency for Science, Technology and Innovation (Sapere Aude project).

### Author contributions

S.P. conceived the theoretical ideas, developed the methodology, derived the main analytical results, and wrote the manuscript and the Supplementary Information. C.O. contributed to the security analysis of the protocol and performed the post-processing of the experimental data. G.S. contributed to theoretical aspects and performed the post-processing of the experimental data. C.S.J., T.G. and U.L.A. designed and performed the experiment, analysed the experimental data, and contributed to writing the description of the experimental set-up. C.W., S.L. and S.L.B. contributed to theoretical aspects and editing of the manuscript.

### Additional information

Supplementary information is available in the online version of the paper. Reprints and permissions information is available online at www.nature.com/reprints. Correspondence and requests for materials should be addressed to S.P.

### Competing financial interests

The authors declare no competing financial interests.

NATURE PHOTONICS | VOL 9 | JUNE 2015 | www.nature.com/naturephotonics