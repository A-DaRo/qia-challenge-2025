[← Return to Main Index](../index.md)

# 7.3 Key Derivation

## Introduction

Key derivation is the final step of privacy amplification, where the theoretical security guarantees of Toeplitz hashing are translated into operational key material. This section examines the end-to-end pipeline from reconciled bit strings to cryptographically secure keys, including seed distribution, format conversion, and integration with the $\binom{2}{1}$-Oblivious Transfer primitive.

Unlike intermediate protocol stages (which operate on raw bits), key derivation must produce keys in formats compatible with downstream applications: boolean arrays for string oblivious transfer, byte arrays for symmetric encryption, or structured objects for authentication.

## Pipeline Architecture

### Phase III → Phase IV Transition

The input to key derivation is the output of information reconciliation:

**Alice's State**:
- Reconciled key $X_A \in \{0, 1\}^n$
- Syndrome leakage $|\Sigma| = n(1 - R)$ bits (from LDPC decoding)
- QBER estimate $\hat{Q}$

**Bob's State**:
- Reconciled key $X_B \in \{0, 1\}^n$
- Agreement check: $X_B \stackrel{?}{=} X_A$ (verified via hash comparison)

**Critical Assumption**: If reconciliation succeeded, $X_A = X_B$ with probability $1 - \varepsilon_{\text{EC}}$, where $\varepsilon_{\text{EC}} \ll 10^{-6}$ for well-designed codes.

### Key Derivation Stages

```
┌────────────────────────────────────────────────────────────────┐
│                     KEY DERIVATION PIPELINE                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  [1] LENGTH CALCULATION                                        │
│      ℓ = ⌊n·h_min - |Σ| - Δ_sec + 2⌋                          │
│      ↓                                                         │
│  [2] SEED GENERATION (Alice only)                              │
│      seed ← CSPRNG(256 bits)                                   │
│      ↓                                                         │
│  [3] SEED DISTRIBUTION                                         │
│      Alice → Bob: seed (authenticated channel)                 │
│      ↓                                                         │
│  [4] TOEPLITZ CONSTRUCTION                                     │
│      T ← ToeplitzHasher(n, ℓ, seed)                            │
│      ↓                                                         │
│  [5] HASHING                                                   │
│      S_A = T(X_A),  S_B = T(X_B)                               │
│      ↓                                                         │
│  [6] FORMAT CONVERSION                                         │
│      Key ← format(S, application_type)                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Secure Key Length Determination

### Runtime Calculation

Unlike fixed-rate systems (e.g., QKD with predetermined efficiency), Caligo computes $\ell$ **dynamically** based on:

1. **NSM Parameters**: Storage noise $r$, detection efficiency $\eta$
2. **Measured QBER**: From Phase II sifting
3. **Syndrome Leakage**: From Phase III LDPC reconciliation

**Algorithm**:

```python
def derive_key_length(nsm_params: NSMParameters,
                      reconciled_length: int,
                      syndrome_leakage: int,
                      epsilon_sec: float = 1e-10) -> int:
    # Compute min-entropy rate
    entropy_calc = NSMEntropyCalculator(nsm_params.storage_noise_r)
    h_min, _ = entropy_calc.max_bound_entropy_rate()
    
    # Lupo formula
    entropy_available = h_min * reconciled_length
    security_penalty = 2 * math.log2(1 / epsilon_sec) - 2
    
    ell_raw = entropy_available - syndrome_leakage - security_penalty
    ell_final = max(0, int(math.floor(ell_raw)))
    
    return ell_final
```

**Abort Condition**: If $\ell_{\text{final}} = 0$, the protocol terminates without outputting a key (Death Valley regime).

### Efficiency Metrics

The **extraction efficiency** quantifies key material utilization:

$$
\eta_{\text{extract}} = \frac{\ell}{n}
$$

**Example**: For $n = 1000$, $r = 0.75$ ($h_{\min} = 0.25$), syndrome leakage 200 bits, $\varepsilon_{\text{sec}} = 10^{-10}$:

$$
\begin{aligned}
\text{Available entropy} &= 1000 \times 0.25 = 250 \text{ bits} \\
\text{Security penalty} &= 64 \text{ bits} \\
\ell &= \lfloor 250 - 200 - 64 \rfloor = \lfloor -14 \rfloor = 0 \quad (\text{fails})
\end{aligned}
$$

To achieve $\ell > 0$, reduce leakage or increase $n$.

## Seed Distribution Protocol

### Cryptographic Seed Generation

Alice generates the Toeplitz seed using a cryptographically secure pseudo-random number generator (CSPRNG):

```python
import secrets

def generate_toeplitz_seed(num_bits: int) -> bytes:
    """Generate seed for (n + m - 1)-bit Toeplitz matrix."""
    num_bytes = (num_bits + 7) // 8
    return secrets.token_bytes(num_bytes)
```

**Security**: `secrets` module draws from OS entropy pool (`/dev/urandom`, `CryptGenRandom`, etc.), providing cryptographic-grade randomness resistant to prediction attacks.

**Seed Length**: For $n = 1000$, $\ell = 400$, the Toeplitz matrix requires $n + \ell - 1 = 1399$ random bits $\approx$ 175 bytes.

### Authenticated Transmission

The seed is transmitted over the **authenticated classical channel**. This channel provides:

1. **Integrity**: Bob verifies the seed originates from Alice (via MAC or digital signature)
2. **No Confidentiality**: The adversary reads the seed (consistent with Kerckhoffs' principle)

**Caligo Implementation**:
```python
# Alice
seed = ToeplitzHasher.generate_seed(input_length + output_length - 1)
socket.send_classical_authenticated(to="Bob", data=seed)

# Bob
seed = socket.recv_classical_authenticated(from_="Alice")
```

**SquidASM Integration**: In simulation, this maps to `NetSquidConnection.put_bytes()` with integrity checks enforced by the application layer.

### Deterministic Reconstruction

Both parties independently construct the Toeplitz matrix from the shared seed:

```python
# Identical on both sides
hasher = ToeplitzHasher(
    input_length=reconciled_length,
    output_length=final_key_length,
    seed=seed,  # Shared
    use_fft=True
)
```

**Bit-Perfect Reproducibility**: Binary operations (XOR, bit shifts) are deterministic across platforms, avoiding floating-point portability issues.

## Hash Application

### Input Formatting

The reconciled key $X$ is represented as a NumPy boolean array:

```python
reconciled_key: np.ndarray  # dtype=bool, shape=(n,)
# Example: [True, False, True, True, False, ...]
```

**Conversion to Binary**:
```python
binary_key = reconciled_key.astype(np.uint8)  # 0/1 encoding
```

### Hashing Operation

Apply the Toeplitz matrix:

```python
final_key = hasher.hash(binary_key)
# Output: np.ndarray, dtype=uint8, shape=(ell,)
```

**Computational Complexity**:
- **FFT mode** ($n > 64$): $O(n \log n)$
- **Direct mode** ($n \leq 64$): $O(n \ell)$

**Example**: For $n = 10{,}000$, FFT hashing completes in $\sim 10$ ms on a modern CPU (vs. $\sim 200$ ms for direct multiplication).

### Output Validation

Both parties compute a hash digest of the final key:

```python
import hashlib

digest_alice = hashlib.sha256(final_key_alice.tobytes()).hexdigest()
digest_bob = hashlib.sha256(final_key_bob.tobytes()).hexdigest()

# Alice broadcasts digest
socket.send_classical(to="Bob", data=digest_alice)

# Bob verifies
if digest_bob != digest_alice:
    abort("Key mismatch detected")
```

**Purpose**: Detects reconciliation failures (rare but possible due to residual errors or adversarial tampering). Reveals no information about the key (SHA-256 is one-way).

## Format Conversion

### Application-Specific Representations

The raw binary key is converted to match application requirements:

#### 1. Boolean Array (OT Strings)

For $\binom{2}{1}$-OT, the sender's inputs $S_0, S_1$ are boolean vectors:

```python
def to_boolean_string(key: np.ndarray) -> List[bool]:
    """Convert binary key to boolean list."""
    return key.astype(bool).tolist()
```

**Usage**: Directly applied in Caligo's `ObliviousTransferSender` class.

#### 2. Byte Array (Symmetric Keys)

For AES-256, keys must be 32-byte arrays:

```python
def to_byte_array(key: np.ndarray, target_bytes: int = 32) -> bytes:
    """Pack bits into bytes."""
    if len(key) < target_bytes * 8:
        raise ValueError(f"Key too short: {len(key)} < {target_bytes * 8}")
    # Take first target_bytes * 8 bits
    truncated = key[:target_bytes * 8]
    # Pack into bytes (MSB first)
    return np.packbits(truncated).tobytes()
```

#### 3. Integer (Session ID)

For authentication tokens:

```python
def to_integer(key: np.ndarray, max_bits: int = 128) -> int:
    """Convert key to integer."""
    truncated = key[:max_bits]
    return int(''.join(map(str, truncated)), 2)
```

### Entropy Preservation

All conversions are **lossless** (bijective mappings), ensuring:

$$
H_{\min}(S) = H_{\min}(\text{format}(S)) = \ell
$$

No additional privacy amplification is required post-conversion.

## Integration with Oblivious Transfer

### Sender Key Splitting

Alice splits the derived key into two OT strings:

```python
def split_key_for_ot(key: np.ndarray) -> Tuple[List[bool], List[bool]]:
    """Split key into S_0 and S_1."""
    ell = len(key)
    mid = ell // 2
    S_0 = key[:mid].astype(bool).tolist()
    S_1 = key[mid:].astype(bool).tolist()
    return S_0, S_1
```

**Example**: A 512-bit key yields two 256-bit OT strings.

### Receiver Key Selection

Bob uses his choice bit $C \in \{0, 1\}$ to select:

```python
C = random.choice([0, 1])
final_key = hasher.hash(reconciled_key)

if C == 0:
    S_C = final_key[:len(final_key)//2]
else:
    S_C = final_key[len(final_key)//2:]
```

**Security**: The choice bit $C$ is known only to Bob, satisfying receiver-security of OT.

## Error Handling

### Reconciliation Failure

If error correction fails (Bob's key differs from Alice's):

```python
class ReconciliationFailureError(Exception):
    """Raised when EC does not converge."""

try:
    reconcile(alice_key, bob_key, syndrome)
except ReconciliationFailureError:
    log.error("Reconciliation failed, aborting key derivation")
    return None  # No key output
```

**Protocol Behavior**: Abort rather than proceeding with mismatched keys (which would compromise both correctness and security).

### Insufficient Entropy

If Death Valley occurs:

```python
final_length = compute_final_length(n, syndrome_leakage)
if final_length == 0:
    raise EntropyDepletedError(
        f"Entropy insufficient: {entropy_available:.2f} bits available, "
        f"{syndrome_leakage + security_penalty:.2f} bits consumed"
    )
```

**User Action**: Increase $n$ (transmit more qubits) or improve NSM parameters (higher $r$).

## Performance Optimization

### Lazy Key Derivation

Only compute the key if protocol phases succeed:

```python
# Phase II (Sifting) → Phase III (Reconciliation) → Phase IV (Amplification)
if not sifting_result.is_secure:
    return  # Abort before reconciliation

if not reconciliation_result.converged:
    return  # Abort before amplification

# Only now perform key derivation
final_key = derive_key(reconciliation_result.key, ...)
```

**Rationale**: Avoid wasting computation on doomed protocol runs (e.g., excessive QBER detected early).

### Batch Extraction

For multi-party protocols, extract multiple keys from a single reconciled string:

```python
def extract_multiple_keys(reconciled_key: np.ndarray,
                          key_lengths: List[int],
                          seed: bytes) -> List[np.ndarray]:
    """Extract multiple non-overlapping keys."""
    keys = []
    offset = 0
    for ell in key_lengths:
        # Use different Toeplitz matrices (seed + offset)
        hasher = ToeplitzHasher(len(reconciled_key), ell, 
                                seed=seed + offset.to_bytes(4, 'big'))
        keys.append(hasher.hash(reconciled_key))
        offset += 1
    return keys
```

**Application**: Distribute keys to $N$ parties in a single protocol execution.

## Security Guarantees

### Composable Security

The derived key satisfies [1]:

$$
\frac{1}{2} \|\rho_{SE} - \omega_S \otimes \rho_E\|_1 \leq \varepsilon_{\text{sec}}
$$

**Implication**: $S$ can be used as a one-time pad without additional processing:

$$
M_{\text{encrypted}} = M \oplus S
$$

The adversary's advantage in decrypting $M$ is bounded by $\varepsilon_{\text{sec}}$.

### Adversary Model

**Assumptions**:
1. **Channel Control**: Adversary fully controls quantum channel (intercept-resend, entanglement attacks)
2. **Classical Eavesdropping**: Reads all classical messages (seed, syndrome)
3. **Noisy Storage**: Stores qubits with depolarization rate $1 - r$

**Not Assumed**:
- Adversary cannot tamper with authenticated channel (MAC prevents forgery)
- Adversary respects the wait time $\Delta t$ (physical enforcement)

## Practical Considerations

### Minimum Key Length

For cryptographic applications:
- **AES-128**: $\ell \geq 128$ bits
- **AES-256**: $\ell \geq 256$ bits
- **OT strings**: $\ell \geq 2k$ bits (where $k$ is desired security level)

**Caligo Default**: Target $\ell = 512$ bits (256 bits per OT string).

### Seed Reuse

**Prohibition**: The same seed must **never** be reused with different reconciled keys:

$$
S = T_{\text{seed}}(X), \quad S' = T_{\text{seed}}(X') \implies \text{Leak}(X \oplus X')
$$

**Enforcement**: Generate fresh seed for each protocol execution using CSPRNG.

### Performance Benchmarks

| Key Length ($n$) | Toeplitz Seed (bytes) | Hash Time (FFT) | Hash Time (Direct) |
|------------------|----------------------|----------------|-------------------|
| 1,000            | 180                  | 0.8 ms         | 2.1 ms            |
| 10,000           | 1,560                | 7.2 ms         | 68 ms             |
| 100,000          | 15,625               | 95 ms          | 6,200 ms          |

Measurements on Intel Core i7-10700K, single-threaded.

## References

[1] Renner, R. (2008). Security of Quantum Key Distribution. *International Journal of Quantum Information*, 6(1), 1-127.

[2] Lupo, C., Peat, J. T., Andersson, E., & Kok, P. (2023). Error-tolerant oblivious transfer in the noisy-storage model. *Physical Review A*, 107(6), 062403.

[3] Tomamichel, M., Lim, C. C. W., Gisin, N., & Renner, R. (2012). Tight finite-key analysis for quantum cryptography. *Nature Communications*, 3, 634.

[4] Carter, L., & Wegman, M. N. (1979). Universal classes of hash functions. *Journal of Computer and System Sciences*, 18(2), 143-154.

---

[← Return to Main Index](../index.md) | [Previous: Extractable Length Calculation](./extractable_length.md)
